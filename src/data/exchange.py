"""CCXT exchange client for perpetual futures market data.

Supports Binance USDM, Bybit, OKX, Bitget with graceful fallbacks
when derivatives endpoints are unavailable.
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import pandas as pd
from loguru import logger

from src.utils.config import AppConfig, ExchangeConfig
from src.utils.helpers import normalize_symbol, safe_float


# Public market data changes quickly, but duplicate requests inside a scan add
# latency and rate-limit pressure. This small process cache is shared by client
# instances and returns defensive copies so callers cannot mutate cached data.
_CACHE_LOCK = RLock()
_MARKET_DATA_CACHE: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}


def _cache_get(key: Tuple[Any, ...]) -> Any:
    now = time.monotonic()
    with _CACHE_LOCK:
        hit = _MARKET_DATA_CACHE.get(key)
        if hit is None:
            return None
        expires_at, value = hit
        if expires_at <= now:
            _MARKET_DATA_CACHE.pop(key, None)
            return None
        return value.copy(deep=True) if isinstance(value, pd.DataFrame) else copy.deepcopy(value)


def _cache_put(key: Tuple[Any, ...], value: Any, ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        return
    stored = value.copy(deep=True) if isinstance(value, pd.DataFrame) else copy.deepcopy(value)
    with _CACHE_LOCK:
        _MARKET_DATA_CACHE[key] = (time.monotonic() + ttl_seconds, stored)
        # Bound long-running scheduler memory without a maintenance thread.
        if len(_MARKET_DATA_CACHE) > 2000:
            now = time.monotonic()
            stale = [k for k, (expiry, _) in _MARKET_DATA_CACHE.items() if expiry <= now]
            for stale_key in stale:
                _MARKET_DATA_CACHE.pop(stale_key, None)


# Map friendly names → ccxt class names
EXCHANGE_MAP: Dict[str, str] = {
    "binance": "binanceusdm",
    "binanceusdm": "binanceusdm",
    "binance_usdm": "binanceusdm",
    "bybit": "bybit",
    "okx": "okx",
    "bitget": "bitget",
    "mexc": "mexc",
    "bingx": "bingx",
    "bitfinex": "bitfinex",
    "bitmart": "bitmart",
    "gate": "gate",
    "gateio": "gate",
    "htx": "htx",
    "huobi": "huobi",
    "weex": "weex",
}

# Default order when auto-fallback tries other venues
DEFAULT_FALLBACK_EXCHANGES: List[str] = [
    "bybit",
    "binanceusdm",
    "okx",
    "bitget",
    "mexc",
    "bingx",
    "bitmart",
    "gate",
    "htx",
    "weex",
]


def normalize_exchange_id(exchange_id: str) -> str:
    """Normalize friendly / alias exchange names to ccxt ids."""
    if not exchange_id:
        return "bybit"
    raw = exchange_id.strip().lower().replace("-", "").replace("_", "")
    if raw in {"binanceusdm", "binanceus"}:
        return "binanceusdm"
    if raw in {"gateio", "gate"}:
        return "gate"
    if raw in {"huobi", "htx"}:
        return "htx"
    return EXCHANGE_MAP.get(raw, raw)


def build_exchange_attempt_order(
    preferred: str,
    config: Optional[AppConfig] = None,
    *,
    auto_fallback: Optional[bool] = None,
) -> List[str]:
    """Build ordered exchange list: preferred first, then configured fallbacks."""
    preferred_norm = normalize_exchange_id(preferred or "bybit")
    use_fallback = (
        config.exchange.auto_fallback
        if auto_fallback is None and config
        else (auto_fallback if auto_fallback is not None else True)
    )
    if not use_fallback:
        return [preferred_norm]

    fallbacks = (
        list(config.exchange.fallback_exchanges)
        if config and config.exchange.fallback_exchanges
        else list(DEFAULT_FALLBACK_EXCHANGES)
    )
    ordered: List[str] = []
    for ex in [preferred_norm, *fallbacks]:
        norm = normalize_exchange_id(ex)
        if norm not in ordered and hasattr(ccxt, norm):
            ordered.append(norm)
    for ex in list_supported_exchanges():
        if ex not in ordered:
            ordered.append(ex)
    return ordered


@dataclass
class MarketSnapshot:
    """Aggregated derivatives + ticker snapshot for a symbol."""

    symbol: str
    exchange_id: str
    last: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    mark: Optional[float] = None
    index: Optional[float] = None
    percentage_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    funding_rate: Optional[float] = None
    funding_timestamp: Optional[int] = None
    next_funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    open_interest_value: Optional[float] = None
    long_short_ratio: Optional[float] = None
    long_account: Optional[float] = None
    short_account: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def funding_rate_pct(self) -> Optional[float]:
        if self.funding_rate is None:
            return None
        return self.funding_rate * 100.0


class ExchangeClient:
    """Thin, rate-limit-aware wrapper around ccxt perpetual venues."""

    def __init__(
        self,
        exchange_id: Optional[str] = None,
        config: Optional[AppConfig] = None,
        exchange_cfg: Optional[ExchangeConfig] = None,
    ) -> None:
        self.config = config
        self.exchange_cfg = exchange_cfg or (config.exchange if config else ExchangeConfig())
        raw_id = (exchange_id or self.exchange_cfg.default or "bybit").lower()
        self.exchange_id = normalize_exchange_id(raw_id)
        self.cache_ttl_seconds = max(
            0,
            int(getattr(getattr(config, "timeframes", None), "cache_ttl_seconds", 300) or 0),
        )
        self.fetch_workers = max(
            1,
            min(8, int(getattr(getattr(config, "timeframes", None), "fetch_workers", 4) or 4)),
        )
        self._exchange = self._build_exchange()
        self._markets_loaded = False
        self._markets_lock = RLock()

    def _normalize_exchange_id(self, exchange_id: str) -> str:
        return normalize_exchange_id(exchange_id)

    def _build_exchange(self) -> ccxt.Exchange:
        if not hasattr(ccxt, self.exchange_id):
            raise ValueError(
                f"Unsupported exchange '{self.exchange_id}'. "
                f"Supported: {', '.join(sorted(set(EXCHANGE_MAP.values())))}"
            )
        klass = getattr(ccxt, self.exchange_id)
        params: Dict[str, Any] = {
            "enableRateLimit": self.exchange_cfg.enable_rate_limit,
            "timeout": self.exchange_cfg.timeout_ms,
            "options": {"defaultType": "swap"},
        }
        if self.exchange_cfg.api_key:
            params["apiKey"] = self.exchange_cfg.api_key
        if self.exchange_cfg.api_secret:
            params["secret"] = self.exchange_cfg.api_secret
        if self.exchange_cfg.password:
            params["password"] = self.exchange_cfg.password

        exchange: ccxt.Exchange = klass(params)
        if self.exchange_cfg.sandbox and hasattr(exchange, "set_sandbox_mode"):
            try:
                exchange.set_sandbox_mode(True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sandbox mode not available on {}: {}", self.exchange_id, exc)
        logger.info("Initialized exchange client: {}", self.exchange_id)
        return exchange

    def load_markets(self, reload: bool = False) -> Dict[str, Any]:
        if not hasattr(self, "_markets_lock"):
            self._markets_lock = RLock()
        with self._markets_lock:
            if self._markets_loaded and not reload:
                return self._exchange.markets or {}
            try:
                markets = self._exchange.load_markets(reload=reload)
                self._markets_loaded = True
                return markets
            except ccxt.NetworkError as exc:
                logger.warning("Network error loading markets (will try direct fetch): {}", exc)
                return self._exchange.markets or {}
            except ccxt.ExchangeError as exc:
                logger.warning("Exchange error loading markets (will try direct fetch): {}", exc)
                return self._exchange.markets or {}

    def resolve_symbol(self, symbol: str) -> str:
        """Normalize and resolve symbol against exchange markets."""
        unified = normalize_symbol(symbol)
        cache_key = ("symbol", self.exchange_id, unified)
        cached = _cache_get(cache_key)
        if cached:
            return str(cached)
        try:
            self.load_markets()
        except Exception as exc:  # noqa: BLE001
            logger.debug("load_markets skipped: {}", exc)

        logger.info("Symbol input '{}' -> cleaned symbol '{}'", symbol, unified)
        markets = self._exchange.markets or {}
        base = unified.split("/")[0]
        base_quote = unified.split(":")[0] if ":" in unified else unified

        candidates: List[str] = [
            unified,
            base_quote,
            f"{base_quote}:USDT",
            f"{base_quote}:USDC",
            f"{base}/USDT",
            f"{base}/USDT:USDT",
            f"{base}/USD",
            f"{base}/USD:USDT",
            f"{base}USDT",
            f"{base}USDC",
            f"{base}USD",
            symbol.upper(),
        ]

        if unified in markets:
            _cache_put(cache_key, unified, max(3600, getattr(self, "cache_ttl_seconds", 300)))
            return unified

        # Rank perpetual candidates instead of returning whichever venue lists
        # first. Prefer active linear USDT swaps, then USDC/USD derivatives.
        matching: List[Tuple[int, str]] = []
        for m_id, m in markets.items():
            if not m.get("swap") and not m.get("future"):
                continue
            if m.get("base") == base and m.get("quote") in ("USDT", "USDC", "USD"):
                rank = 0
                rank += 8 if m.get("quote") == "USDT" else (4 if m.get("quote") == "USDC" else 1)
                rank += 4 if m.get("linear") else 0
                rank += 2 if m.get("swap") else 0
                rank += 1 if m.get("active") is not False else -10
                matching.append((rank, m_id))
        if matching:
            resolved = max(matching, key=lambda item: item[0])[1]
            logger.debug("Resolved {} → {}", symbol, resolved)
            _cache_put(cache_key, resolved, max(3600, getattr(self, "cache_ttl_seconds", 300)))
            return resolved

        for c in candidates:
            if c in markets:
                logger.debug("Resolved {} → {}", symbol, c)
                _cache_put(cache_key, c, max(3600, getattr(self, "cache_ttl_seconds", 300)))
                return c

        # Last resort: return normalized; fetch may still work
        if markets:
            logger.warning("Symbol {} not in markets map; using {}", symbol, unified)
        _cache_put(cache_key, unified, getattr(self, "cache_ttl_seconds", 300))
        return unified

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 500,
        since: Optional[int] = None,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Fetch OHLCV and return a clean DataFrame indexed by datetime UTC."""
        resolved = self.resolve_symbol(symbol)
        cache_key = ("ohlcv", self.exchange_id, resolved, timeframe, int(limit), since)
        cached = _cache_get(cache_key)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            logger.debug("OHLCV cache hit {} {} {}", self.exchange_id, resolved, timeframe)
            return cached
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = self._exchange.fetch_ohlcv(
                    resolved, timeframe=timeframe, limit=limit, since=since
                )
                if not raw:
                    raise ValueError(f"Empty OHLCV for {resolved} {timeframe}")
                df = pd.DataFrame(
                    raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                df = df.set_index("timestamp")
                for col in ("open", "high", "low", "close", "volume"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["open", "high", "low", "close"])
                logger.debug(
                    "Fetched {} candles for {} {} on {}",
                    len(df),
                    resolved,
                    timeframe,
                    self.exchange_id,
                )
                _cache_put(cache_key, df, getattr(self, "cache_ttl_seconds", 300))
                return df
            except ccxt.RateLimitExceeded as exc:
                last_err = exc
                sleep_s = min(2 ** attempt, 30)
                logger.warning("Rate limited (attempt {}); sleeping {}s", attempt, sleep_s)
                time.sleep(sleep_s)
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                last_err = exc
                sleep_s = min(1.5 ** attempt, 10)
                logger.warning(
                    "OHLCV fetch error (attempt {}): {}; retry in {}s", attempt, exc, sleep_s
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: {last_err}")

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        normalized_symbol = normalize_symbol(symbol)
        logger.info("Ticker input '{}' -> cleaned symbol '{}'", symbol, normalized_symbol)
        resolved = self.resolve_symbol(normalized_symbol)
        cache_key = ("ticker", self.exchange_id, resolved)
        cached = _cache_get(cache_key)
        if isinstance(cached, dict) and cached:
            return cached
        candidates = [resolved, *self._build_symbol_candidates(normalized_symbol, symbol)]
        candidates = list(dict.fromkeys(candidates))
        last_error: Optional[Exception] = None

        for candidate in candidates:
            try:
                result = self._exchange.fetch_ticker(candidate)
                if result:
                    price = None
                    for key in ("last", "close", "mark", "index", "ask", "bid"):
                        price = result.get(key)
                        if price is not None:
                            break
                    if price is not None:
                        logger.debug(
                            "Ticker fetch success for '{}' using '{}' -> price={}",
                            symbol,
                            candidate,
                            price,
                        )
                        _cache_put(cache_key, result, getattr(self, "cache_ttl_seconds", 300))
                        return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.debug("Ticker fetch failed for {} ({}): {}", candidate, symbol, exc)

        logger.warning(
            "Ticker fetch failed for '{}' using candidates {}; last_error={}",
            symbol,
            candidates,
            last_error,
        )
        return {}

    def _build_symbol_candidates(self, normalized_symbol: str, original_symbol: str) -> List[str]:
        base = normalized_symbol.split("/")[0]
        base_quote = normalized_symbol.split(":")[0] if ":" in normalized_symbol else normalized_symbol
        variants = [
            normalized_symbol,
            base_quote,
            f"{base_quote}:USDT",
            f"{base_quote}:USDC",
            f"{base}/USDT",
            f"{base}/USDT:USDT",
            f"{base}/USD",
            f"{base}/USD:USDT",
            f"{base}USDT",
            f"{base}USDC",
            f"{base}USD",
            original_symbol.upper(),
        ]
        return [v for v in dict.fromkeys(variants) if v]

    def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        resolved = self.resolve_symbol(symbol)
        try:
            if self._exchange.has.get("fetchFundingRate"):
                return self._exchange.fetch_funding_rate(resolved) or {}
            if self._exchange.has.get("fetchFundingRates"):
                rates = self._exchange.fetch_funding_rates([resolved]) or {}
                if isinstance(rates, dict):
                    return rates.get(resolved) or next(iter(rates.values()), {}) or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Funding rate unavailable for {}: {}", resolved, exc)
        return {}

    def fetch_open_interest(self, symbol: str) -> Dict[str, Any]:
        resolved = self.resolve_symbol(symbol)
        try:
            if self._exchange.has.get("fetchOpenInterest"):
                return self._exchange.fetch_open_interest(resolved) or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Open interest unavailable for {}: {}", resolved, exc)
        # Fallback: some venues expose OI via tickers / premium index
        try:
            ticker = self.fetch_ticker(resolved)
            info = ticker.get("info") or {}
            oi = info.get("openInterest") or info.get("open_interest")
            if oi is not None:
                return {"openInterestAmount": safe_float(oi), "symbol": resolved, "info": info}
        except Exception:  # noqa: BLE001
            pass
        return {}

    def fetch_long_short_ratio(self, symbol: str) -> Dict[str, Any]:
        """Best-effort global long/short account ratio (venue-dependent)."""
        resolved = self.resolve_symbol(symbol)
        result: Dict[str, Any] = {}

        # Binance USDM public endpoint via implicit API if present
        try:
            if self.exchange_id == "binanceusdm":
                base = resolved.split("/")[0]
                # period 5m, last datapoint
                if hasattr(self._exchange, "fapiDataGetGlobalLongShortAccountRatio"):
                    data = self._exchange.fapiDataGetGlobalLongShortAccountRatio(
                        {"symbol": f"{base}USDT", "period": "1h", "limit": 1}
                    )
                    if data:
                        row = data[-1] if isinstance(data, list) else data
                        result = {
                            "longShortRatio": safe_float(row.get("longShortRatio")),
                            "longAccount": safe_float(row.get("longAccount")),
                            "shortAccount": safe_float(row.get("shortAccount")),
                            "timestamp": row.get("timestamp"),
                        }
                        return result
        except Exception as exc:  # noqa: BLE001
            logger.debug("L/S ratio (binance) failed: {}", exc)

        try:
            if self.exchange_id == "bybit":
                base = resolved.split("/")[0]
                if hasattr(self._exchange, "publicGetV5MarketAccountRatio"):
                    data = self._exchange.publicGetV5MarketAccountRatio(
                        {"category": "linear", "symbol": f"{base}USDT", "period": "1h", "limit": "1"}
                    )
                    rows = (data or {}).get("result", {}).get("list") or []
                    if rows:
                        row = rows[0]
                        buy = safe_float(row.get("buyRatio"))
                        sell = safe_float(row.get("sellRatio"))
                        ratio = buy / sell if sell else None
                        result = {
                            "longShortRatio": ratio,
                            "longAccount": buy,
                            "shortAccount": sell,
                        }
                        return result
        except Exception as exc:  # noqa: BLE001
            logger.debug("L/S ratio (bybit) failed: {}", exc)

        return result

    def fetch_market_snapshot(self, symbol: str) -> MarketSnapshot:
        """Compose ticker + funding + OI + L/S into one snapshot."""
        resolved = self.resolve_symbol(symbol)
        cache_key = ("snapshot", self.exchange_id, resolved)
        cached = _cache_get(cache_key)
        if isinstance(cached, MarketSnapshot):
            logger.debug("Snapshot cache hit {} {}", self.exchange_id, resolved)
            return cached
        snap = MarketSnapshot(symbol=resolved, exchange_id=self.exchange_id)
        errors: List[str] = []

        # These are independent public endpoints. Bound concurrency so a scan is
        # fast without creating an unbounded burst against the venue.
        with ThreadPoolExecutor(max_workers=min(4, getattr(self, "fetch_workers", 4))) as pool:
            futures = {
                "ticker": pool.submit(self.fetch_ticker, resolved),
                "funding": pool.submit(self.fetch_funding_rate, resolved),
                "oi": pool.submit(self.fetch_open_interest, resolved),
                "ls": pool.submit(self.fetch_long_short_ratio, resolved),
            }
            fetched: Dict[str, Dict[str, Any]] = {}
            for name, future in futures.items():
                try:
                    fetched[name] = future.result() or {}
                except Exception as exc:  # noqa: BLE001
                    fetched[name] = {}
                    errors.append(f"{name}_error:{type(exc).__name__}")

        ticker = fetched["ticker"]
        if ticker:
            snap.last = safe_float(ticker.get("last") or ticker.get("close"))
            snap.bid = safe_float(ticker.get("bid"))
            snap.ask = safe_float(ticker.get("ask"))
            snap.percentage_24h = (
                safe_float(ticker["percentage"]) if ticker.get("percentage") is not None else None
            )
            snap.volume_24h = safe_float(ticker.get("quoteVolume") or ticker.get("baseVolume")) or None
            info = ticker.get("info") or {}
            mark = info.get("markPrice") or info.get("mark_price")
            index = info.get("indexPrice") or info.get("index_price")
            if mark is not None:
                snap.mark = safe_float(mark)
            if index is not None:
                snap.index = safe_float(index)
            snap.raw["ticker"] = ticker
        else:
            errors.append("ticker_unavailable")

        fr = fetched["funding"]
        if fr:
            snap.funding_rate = (
                safe_float(fr["fundingRate"]) if fr.get("fundingRate") is not None else None
            )
            snap.funding_timestamp = fr.get("timestamp") or fr.get("fundingTimestamp")
            if fr.get("nextFundingRate") is not None:
                snap.next_funding_rate = safe_float(fr["nextFundingRate"])
            # Some exchanges put rate in info
            if snap.funding_rate is None:
                info = fr.get("info") or {}
                for k in ("lastFundingRate", "fundingRate", "r"):
                    if info.get(k) is not None:
                        snap.funding_rate = safe_float(info[k])
                        break
            snap.raw["funding"] = fr
        else:
            errors.append("funding_unavailable")

        oi = fetched["oi"]
        if oi:
            amount = oi.get("openInterestAmount")
            if amount is None:
                amount = oi.get("openInterest")
            if amount is None and isinstance(oi.get("info"), dict):
                amount = oi["info"].get("openInterest") or oi["info"].get("oi")
            snap.open_interest = safe_float(amount) if amount is not None else None
            value = oi.get("openInterestValue")
            if value is not None:
                snap.open_interest_value = safe_float(value)
            elif snap.open_interest and snap.last:
                snap.open_interest_value = snap.open_interest * snap.last
            snap.raw["open_interest"] = oi
        else:
            errors.append("oi_unavailable")

        ls = fetched["ls"]
        if ls:
            if ls.get("longShortRatio") is not None:
                snap.long_short_ratio = safe_float(ls["longShortRatio"])
            if ls.get("longAccount") is not None:
                snap.long_account = safe_float(ls["longAccount"])
            if ls.get("shortAccount") is not None:
                snap.short_account = safe_float(ls["shortAccount"])
            snap.raw["long_short"] = ls
        else:
            errors.append("ls_ratio_unavailable")

        # A mark/index quote is a safer last-price fallback than returning zero.
        if not snap.last:
            snap.last = next(
                (p for p in (snap.mark, snap.index, snap.bid, snap.ask) if p is not None and p > 0),
                0.0,
            )
        snap.errors = list(dict.fromkeys(errors))
        _cache_put(cache_key, snap, getattr(self, "cache_ttl_seconds", 300))
        return snap

    def close(self) -> None:
        try:
            if hasattr(self._exchange, "close"):
                self._exchange.close()
        except Exception:  # noqa: BLE001
            pass


def list_supported_exchanges() -> List[str]:
    return sorted(set(EXCHANGE_MAP.values()))


def ohlcv_summary(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Return (last_close, period_return_pct, atr_proxy)."""
    if df is None or df.empty:
        return 0.0, 0.0, 0.0
    last = float(df["close"].iloc[-1])
    first = float(df["close"].iloc[0])
    ret = (last - first) / first * 100.0 if first else 0.0
    tr = (df["high"] - df["low"]).abs()
    atr_proxy = float(tr.tail(14).mean()) if len(tr) else 0.0
    return last, ret, atr_proxy
