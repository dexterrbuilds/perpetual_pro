"""CCXT exchange client for perpetual futures market data.

Supports Binance USDM, Bybit, OKX, Bitget with graceful fallbacks
when derivatives endpoints are unavailable.
"""

from __future__ import annotations

import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import pandas as pd
from loguru import logger

from src.utils.config import AppConfig, ExchangeConfig
from src.utils.helpers import normalize_symbol, safe_float, timeframe_to_minutes


# Public market data changes quickly, but duplicate requests inside a scan add
# latency and rate-limit pressure. This small process cache is shared by client
# instances and returns defensive copies so callers cannot mutate cached data.
_CACHE_LOCK = RLock()
_MARKET_DATA_CACHE: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}
_MARKETS_LOCK = RLock()
_MARKETS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


class PermanentExchangeAccessError(RuntimeError):
    """The deployment cannot use this venue until its network/location changes."""


class UnsupportedMarketError(RuntimeError):
    """A loaded venue market map proves the requested perpetual is unavailable."""


_PERMANENT_ACCESS_MARKERS = (
    "403 forbidden",
    "451",
    "restricted location",
    "restricted jurisdiction",
    "not available in your country",
    "not available in your region",
    "country is not supported",
    "service unavailable from a restricted location",
)


def is_permanent_exchange_access_error(error: Any) -> bool:
    """Identify geo/permission failures that retries cannot repair."""
    message = str(error or "").lower()
    return any(marker in message for marker in _PERMANENT_ACCESS_MARKERS)


def is_unsupported_market_error(error: Any) -> bool:
    """Identify a conclusive unsupported-perpetual result without retrying venues."""
    return "unsupported_market:" in str(error or "").lower()


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


def _cache_put(key: Tuple[Any, ...], value: Any, ttl_seconds: float) -> None:
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


def _normalized_timestamp_ms(value: Any, fallback_ms: int) -> int:
    """Return a plausible epoch-millisecond timestamp for source-age checks."""
    numeric = safe_float(value)
    if numeric <= 0:
        return fallback_ms
    # A few APIs expose epoch seconds rather than CCXT's usual milliseconds.
    if numeric < 10_000_000_000:
        numeric *= 1000.0
    return int(numeric)


def _source_age_seconds(timestamp_ms: Any, now_ms: Optional[int] = None) -> float:
    current = int(now_ms if now_ms is not None else time.time() * 1000)
    source = _normalized_timestamp_ms(timestamp_ms, current)
    return max(0.0, (current - source) / 1000.0)


def _ohlcv_cache_ttl_seconds(
    frame: pd.DataFrame,
    timeframe: str,
    configured_ttl: float,
    *,
    now_ms: Optional[int] = None,
) -> float:
    """Never keep an OHLCV response across the close of its newest candle."""
    ttl = max(0.0, float(configured_ttl or 0.0))
    if ttl <= 0 or frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return ttl
    last_open = frame.index[-1]
    if last_open.tzinfo is None:
        last_open = last_open.tz_localize("UTC")
    else:
        last_open = last_open.tz_convert("UTC")
    closes_at_ms = int(
        (last_open + pd.Timedelta(minutes=timeframe_to_minutes(timeframe))).timestamp()
        * 1000
    )
    current_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    until_close = max(0.25, (closes_at_ms - current_ms) / 1000.0)
    return min(ttl, until_close)


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
        return "okx"
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
    preferred_norm = normalize_exchange_id(preferred or "okx")
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
    funding_average_24h: Optional[float] = None
    open_interest: Optional[float] = None
    open_interest_value: Optional[float] = None
    open_interest_change_pct_24h: Optional[float] = None
    long_short_ratio: Optional[float] = None
    long_account: Optional[float] = None
    short_account: Optional[float] = None
    spread_bps: Optional[float] = None
    orderbook_imbalance: Optional[float] = None
    orderbook_bid_depth: Optional[float] = None
    orderbook_ask_depth: Optional[float] = None
    orderbook_bid_depth_usd: Optional[float] = None
    orderbook_ask_depth_usd: Optional[float] = None
    orderbook_depth_bands: Dict[str, Any] = field(default_factory=dict)
    estimated_impact_bps: Optional[float] = None
    impact_reference_notional_usd: Optional[float] = None
    contract_size: Optional[float] = None
    tick_size: Optional[float] = None
    min_notional: Optional[float] = None
    amount_precision: Optional[float] = None
    orderbook_timestamp: Optional[int] = None
    ticker_timestamp: Optional[int] = None
    ticker_age_seconds: Optional[float] = None
    orderbook_age_seconds: Optional[float] = None
    execution_data_fresh: Optional[bool] = None
    mark_index_basis_bps: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def funding_rate_pct(self) -> Optional[float]:
        if self.funding_rate is None:
            return None
        return self.funding_rate * 100.0

    def refresh_source_ages(
        self,
        *,
        max_ticker_age_seconds: float,
        max_orderbook_age_seconds: float,
    ) -> None:
        """Recompute ages even when the aggregate snapshot came from cache."""
        self.ticker_age_seconds = (
            _source_age_seconds(self.ticker_timestamp)
            if self.ticker_timestamp is not None
            else None
        )
        self.orderbook_age_seconds = (
            _source_age_seconds(self.orderbook_timestamp)
            if self.orderbook_timestamp is not None
            else None
        )
        freshness: List[bool] = []
        if self.ticker_age_seconds is not None:
            freshness.append(self.ticker_age_seconds <= max_ticker_age_seconds)
        if self.orderbook_age_seconds is not None:
            freshness.append(self.orderbook_age_seconds <= max_orderbook_age_seconds)
        self.execution_data_fresh = all(freshness) if freshness else None


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
        raw_id = (exchange_id or self.exchange_cfg.default or "okx").lower()
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
            with _MARKETS_LOCK:
                cached = _MARKETS_CACHE.get(self.exchange_id)
                if cached and not reload and cached[0] > time.monotonic():
                    try:
                        self._exchange.set_markets(cached[1])
                        self._markets_loaded = True
                        return self._exchange.markets or {}
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Shared markets cache rejected by {}: {}", self.exchange_id, exc)
                try:
                    markets = self._exchange.load_markets(reload=reload)
                    self._markets_loaded = True
                    _MARKETS_CACHE[self.exchange_id] = (
                        time.monotonic() + 3600.0,
                        markets,
                    )
                    return markets
                except ccxt.NetworkError as exc:
                    if is_permanent_exchange_access_error(exc):
                        raise PermanentExchangeAccessError(
                            f"{self.exchange_id} public market data is blocked: {exc}"
                        ) from exc
                    logger.warning("Network error loading markets (will try direct fetch): {}", exc)
                    return self._exchange.markets or {}
                except ccxt.ExchangeError as exc:
                    if is_permanent_exchange_access_error(exc):
                        raise PermanentExchangeAccessError(
                            f"{self.exchange_id} public market data is blocked: {exc}"
                        ) from exc
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
        except PermanentExchangeAccessError:
            raise
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

        exact_market = markets.get(unified)
        if exact_market and (
            exact_market.get("swap") or exact_market.get("future")
        ):
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
            market = markets.get(c)
            if market and (market.get("swap") or market.get("future")):
                logger.debug("Resolved {} → {}", symbol, c)
                _cache_put(cache_key, c, max(3600, getattr(self, "cache_ttl_seconds", 300)))
                return c

        # A non-empty market map is authoritative: do not turn a known missing
        # perpetual into many slow exchange retries. An empty map may indicate a
        # transient load failure, so the direct-fetch fallback remains intact.
        if markets:
            logger.info(
                "Unsupported market: {} perpetual is not listed on {}",
                base,
                self.exchange_id,
            )
            raise UnsupportedMarketError(
                f"unsupported_market:{self.exchange_id}:{unified}"
            )
        _cache_put(cache_key, unified, getattr(self, "cache_ttl_seconds", 300))
        return unified

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 500,
        since: Optional[int] = None,
        max_retries: int = 3,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV and return a clean DataFrame indexed by datetime UTC."""
        resolved = self.resolve_symbol(symbol)
        cache_key = ("ohlcv", self.exchange_id, resolved, timeframe, int(limit), since)
        cached = None if force_refresh else _cache_get(cache_key)
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
                required = ["open", "high", "low", "close", "volume"]
                finite = df[required].apply(
                    lambda series: series.map(
                        lambda value: math.isfinite(float(value))
                        if value is not None
                        else False
                    )
                ).all(axis=1)
                coherent = (
                    (df["open"] > 0)
                    & (df["high"] > 0)
                    & (df["low"] > 0)
                    & (df["close"] > 0)
                    & (df["volume"] >= 0)
                    & (df["high"] >= df[["open", "close"]].max(axis=1))
                    & (df["low"] <= df[["open", "close"]].min(axis=1))
                    & (df["high"] >= df["low"])
                )
                rejected = int((~(finite & coherent)).sum())
                df = df.loc[finite & coherent].copy()
                if rejected:
                    logger.warning(
                        "Rejected {} malformed/nonfinite {} candles for {}",
                        rejected,
                        timeframe,
                        resolved,
                    )
                if df.empty:
                    raise ValueError(f"No finite coherent OHLCV for {resolved} {timeframe}")
                logger.debug(
                    "Fetched {} candles for {} {} on {}",
                    len(df),
                    resolved,
                    timeframe,
                    self.exchange_id,
                )
                _cache_put(
                    cache_key,
                    df,
                    _ohlcv_cache_ttl_seconds(
                        df,
                        timeframe,
                        getattr(self, "cache_ttl_seconds", 300),
                    ),
                )
                return df
            except ccxt.RateLimitExceeded as exc:
                last_err = exc
                sleep_s = min(2 ** attempt, 30)
                logger.warning("Rate limited (attempt {}); sleeping {}s", attempt, sleep_s)
                time.sleep(sleep_s)
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                last_err = exc
                if is_permanent_exchange_access_error(exc):
                    logger.warning(
                        "Permanent access failure on {}; skipping retries: {}",
                        self.exchange_id,
                        exc,
                    )
                    break
                sleep_s = min(1.5 ** attempt, 10)
                logger.warning(
                    "OHLCV fetch error (attempt {}): {}; retry in {}s", attempt, exc, sleep_s
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"Failed to fetch OHLCV for {symbol}: {last_err}")

    def fetch_ticker(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        normalized_symbol = normalize_symbol(symbol)
        logger.info("Ticker input '{}' -> cleaned symbol '{}'", symbol, normalized_symbol)
        resolved = self.resolve_symbol(normalized_symbol)
        cache_key = ("ticker", self.exchange_id, resolved)
        cached = None if force_refresh else _cache_get(cache_key)
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
                        candidate_price = safe_float(result.get(key))
                        if candidate_price > 0:
                            price = candidate_price
                            break
                    numeric_price = safe_float(price)
                    if numeric_price > 0:
                        observed_ms = int(time.time() * 1000)
                        result = dict(result)
                        result["_perpetual_pro_observed_at_ms"] = observed_ms
                        result["_perpetual_pro_source_timestamp_ms"] = (
                            _normalized_timestamp_ms(result.get("timestamp"), observed_ms)
                        )
                        logger.debug(
                            "Ticker fetch success for '{}' using '{}' -> price={}",
                            symbol,
                            candidate,
                            numeric_price,
                        )
                        # Ticker is used as the live execution reference. Keep
                        # the five-minute cache for closed OHLCV, not live price.
                        _cache_put(
                            cache_key,
                            result,
                            min(30, getattr(self, "cache_ttl_seconds", 300)),
                        )
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

    def fetch_open_interest_history(
        self, symbol: str, timeframe: str = "1h", limit: int = 24
    ) -> List[Dict[str, Any]]:
        """Best-effort OI history for participation/confirmation analysis."""
        resolved = self.resolve_symbol(symbol)
        try:
            if self._exchange.has.get("fetchOpenInterestHistory"):
                rows = self._exchange.fetch_open_interest_history(
                    resolved, timeframe=timeframe, limit=limit
                )
                return list(rows or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug("OI history unavailable for {}: {}", resolved, exc)
        return []

    def fetch_funding_rate_history(
        self, symbol: str, limit: int = 24
    ) -> List[Dict[str, Any]]:
        """Best-effort recent funding history for crowding context."""
        resolved = self.resolve_symbol(symbol)
        try:
            if self._exchange.has.get("fetchFundingRateHistory"):
                rows = self._exchange.fetch_funding_rate_history(resolved, limit=limit)
                return list(rows or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Funding history unavailable for {}: {}", resolved, exc)
        return []

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

    def fetch_order_book_summary(
        self,
        symbol: str,
        limit: int = 25,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """L2 spread and quote-notional depth normalized across contracts."""
        resolved = self.resolve_symbol(symbol)
        cache_key = ("orderbook_summary", self.exchange_id, resolved, int(limit))
        cached = None if force_refresh else _cache_get(cache_key)
        if isinstance(cached, dict) and cached:
            return cached
        try:
            if not self._exchange.has.get("fetchOrderBook"):
                return {}
            book = self._exchange.fetch_order_book(resolved, limit=limit) or {}
            def valid_level(row: Any) -> bool:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    return False
                try:
                    level_price = float(row[0])
                    amount = float(row[1])
                except (TypeError, ValueError):
                    return False
                return bool(
                    math.isfinite(level_price)
                    and math.isfinite(amount)
                    and level_price > 0
                    and amount >= 0
                )

            bids = [row for row in list(book.get("bids") or []) if valid_level(row)][:limit]
            asks = [row for row in list(book.get("asks") or []) if valid_level(row)][:limit]
            if not bids or not asks:
                return {}
            best_bid = safe_float(bids[0][0])
            best_ask = safe_float(asks[0][0])
            mid = (best_bid + best_ask) / 2.0
            spread_bps = (
                (best_ask - best_bid) / mid * 10_000.0
                if mid > 0 and best_ask >= best_bid
                else None
            )
            try:
                market = self._exchange.market(resolved) or {}
            except Exception:  # noqa: BLE001
                market = {}
            contract_size = max(0.0, safe_float(market.get("contractSize"), 1.0)) or 1.0
            inverse = bool(market.get("inverse", False))

            def quote_notional(level: Any) -> float:
                level_price = safe_float(level[0])
                amount = max(0.0, safe_float(level[1]))
                # Linear/spot quantity is base units; inverse contract size is
                # already quote value per contract on major CCXT venues.
                return amount * contract_size * (1.0 if inverse else level_price)

            bid_depth = sum(quote_notional(row) for row in bids)
            ask_depth = sum(quote_notional(row) for row in asks)
            total_depth = bid_depth + ask_depth
            imbalance = (
                (bid_depth - ask_depth) / total_depth
                if total_depth > 0
                else None
            )
            observed_ms = int(time.time() * 1000)
            source_timestamp = _normalized_timestamp_ms(
                book.get("timestamp"), observed_ms
            )
            depth_bands: Dict[str, Any] = {}
            for band in (5, 10, 25):
                bid_band = sum(
                    quote_notional(row)
                    for row in bids
                    if (mid - safe_float(row[0])) / mid * 10_000.0 <= band
                )
                ask_band = sum(
                    quote_notional(row)
                    for row in asks
                    if (safe_float(row[0]) - mid) / mid * 10_000.0 <= band
                )
                band_total = bid_band + ask_band
                depth_bands[str(band)] = {
                    "bid_usd": bid_band,
                    "ask_usd": ask_band,
                    "imbalance": (
                        (bid_band - ask_band) / band_total if band_total > 0 else None
                    ),
                }

            reference_notional = 10_000.0
            analysis_cfg = getattr(self.config, "analysis", None)
            reference_notional = float(
                getattr(analysis_cfg, "execution_impact_notional_usd", reference_notional)
                or reference_notional
            )

            def impact_for(levels: List[Any], side: str) -> Optional[float]:
                cumulative = 0.0
                worst = mid
                for row in levels:
                    cumulative += quote_notional(row)
                    worst = safe_float(row[0], mid)
                    if cumulative >= reference_notional:
                        move = (worst - mid) / mid * 10_000.0
                        return abs(move)
                return None

            bid_impact = impact_for(bids, "sell")
            ask_impact = impact_for(asks, "buy")
            impact_values = [x for x in (bid_impact, ask_impact) if x is not None]
            result = {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread_bps": spread_bps,
                # Compatibility names now contain quote-notional USD, not raw
                # contracts. Explicit names remove ambiguity for new callers.
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "bid_depth_usd": bid_depth,
                "ask_depth_usd": ask_depth,
                "bid_depth_usd_10bps": depth_bands["10"]["bid_usd"],
                "ask_depth_usd_10bps": depth_bands["10"]["ask_usd"],
                "depth_bands_bps": depth_bands,
                "imbalance": imbalance,
                "estimated_impact_bps": max(impact_values) if impact_values else None,
                "impact_reference_notional_usd": reference_notional,
                "contract_size": contract_size,
                "inverse_contract": inverse,
                "tick_size": (market.get("precision") or {}).get("price"),
                "amount_precision": (market.get("precision") or {}).get("amount"),
                "min_notional": ((market.get("limits") or {}).get("cost") or {}).get("min"),
                "timestamp": source_timestamp,
                "observed_at_ms": observed_ms,
                "age_seconds": _source_age_seconds(source_timestamp, observed_ms),
            }
            # Order books age faster than OHLCV; never hold this for five minutes.
            _cache_put(cache_key, result, min(15, self.cache_ttl_seconds))
            return result
        except Exception as exc:  # noqa: BLE001
            logger.debug("Order book unavailable for {}: {}", resolved, exc)
            return {}

    def fetch_market_snapshot(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> MarketSnapshot:
        """Compose ticker + funding + OI + L/S into one snapshot."""
        resolved = self.resolve_symbol(symbol)
        cache_key = ("snapshot", self.exchange_id, resolved)
        cached = None if force_refresh else _cache_get(cache_key)
        if isinstance(cached, MarketSnapshot):
            analysis_cfg = getattr(self.config, "analysis", None)
            cached.refresh_source_ages(
                max_ticker_age_seconds=float(
                    getattr(analysis_cfg, "max_ticker_age_seconds", 45.0)
                ),
                max_orderbook_age_seconds=float(
                    getattr(analysis_cfg, "max_orderbook_age_seconds", 30.0)
                ),
            )
            logger.debug("Snapshot cache hit {} {}", self.exchange_id, resolved)
            return cached
        snap = MarketSnapshot(symbol=resolved, exchange_id=self.exchange_id)
        errors: List[str] = []

        # These are independent public endpoints. Bound concurrency so a scan is
        # fast without creating an unbounded burst against the venue.
        with ThreadPoolExecutor(max_workers=min(4, getattr(self, "fetch_workers", 4))) as pool:
            futures = {
                "ticker": pool.submit(self.fetch_ticker, resolved, force_refresh),
                "funding": pool.submit(self.fetch_funding_rate, resolved),
                "oi": pool.submit(self.fetch_open_interest, resolved),
                "ls": pool.submit(self.fetch_long_short_ratio, resolved),
                "orderbook": pool.submit(
                    self.fetch_order_book_summary,
                    resolved,
                    25,
                    force_refresh,
                ),
                "oi_history": pool.submit(
                    self.fetch_open_interest_history, resolved, "1h", 24
                ),
                "funding_history": pool.submit(
                    self.fetch_funding_rate_history, resolved, 24
                ),
            }
            fetched: Dict[str, Any] = {}
            for name, future in futures.items():
                try:
                    fetched[name] = future.result() or {}
                except Exception as exc:  # noqa: BLE001
                    fetched[name] = {}
                    errors.append(f"{name}_error:{type(exc).__name__}")

        ticker = fetched["ticker"]
        if ticker:
            snap.last = next(
                (
                    candidate
                    for candidate in (
                        safe_float(ticker.get("last")),
                        safe_float(ticker.get("close")),
                    )
                    if candidate > 0
                ),
                0.0,
            )
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
            snap.ticker_timestamp = int(
                ticker.get("_perpetual_pro_source_timestamp_ms")
                or ticker.get("_perpetual_pro_observed_at_ms")
                or int(time.time() * 1000)
            )
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
        funding_history = fetched["funding_history"]
        if funding_history:
            rates = [
                safe_float(row.get("fundingRate"))
                for row in funding_history
                if row.get("fundingRate") is not None
            ]
            if rates:
                snap.funding_average_24h = sum(rates) / len(rates)
            snap.raw["funding_history"] = funding_history

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
        oi_history = fetched["oi_history"]
        if oi_history:
            amounts: List[float] = []
            for row in oi_history:
                amount = row.get("openInterestAmount")
                if amount is None:
                    amount = row.get("openInterestValue")
                if amount is not None:
                    amounts.append(safe_float(amount))
            if len(amounts) >= 2 and amounts[0] > 0:
                snap.open_interest_change_pct_24h = (
                    (amounts[-1] - amounts[0]) / amounts[0] * 100.0
                )
            snap.raw["open_interest_history"] = oi_history

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

        orderbook = fetched["orderbook"]
        if orderbook:
            snap.spread_bps = (
                safe_float(orderbook.get("spread_bps"))
                if orderbook.get("spread_bps") is not None
                else None
            )
            snap.orderbook_imbalance = (
                safe_float(orderbook.get("imbalance"))
                if orderbook.get("imbalance") is not None
                else None
            )
            snap.orderbook_bid_depth = safe_float(orderbook.get("bid_depth")) or None
            snap.orderbook_ask_depth = safe_float(orderbook.get("ask_depth")) or None
            snap.orderbook_bid_depth_usd = safe_float(orderbook.get("bid_depth_usd")) or None
            snap.orderbook_ask_depth_usd = safe_float(orderbook.get("ask_depth_usd")) or None
            snap.orderbook_depth_bands = dict(orderbook.get("depth_bands_bps") or {})
            snap.estimated_impact_bps = (
                safe_float(orderbook.get("estimated_impact_bps"))
                if orderbook.get("estimated_impact_bps") is not None
                else None
            )
            snap.impact_reference_notional_usd = safe_float(
                orderbook.get("impact_reference_notional_usd")
            ) or None
            snap.contract_size = safe_float(orderbook.get("contract_size")) or None
            snap.tick_size = (
                safe_float(orderbook.get("tick_size"))
                if orderbook.get("tick_size") is not None
                else None
            )
            snap.amount_precision = (
                safe_float(orderbook.get("amount_precision"))
                if orderbook.get("amount_precision") is not None
                else None
            )
            snap.min_notional = (
                safe_float(orderbook.get("min_notional"))
                if orderbook.get("min_notional") is not None
                else None
            )
            snap.orderbook_timestamp = orderbook.get("timestamp")
            snap.raw["orderbook_summary"] = orderbook
        else:
            errors.append("orderbook_unavailable")

        if snap.mark is not None and snap.index is not None and snap.index > 0:
            snap.mark_index_basis_bps = (
                (snap.mark - snap.index) / snap.index * 10_000.0
            )
        if snap.spread_bps is None and snap.bid > 0 and snap.ask >= snap.bid:
            midpoint = (snap.bid + snap.ask) / 2.0
            if midpoint > 0:
                snap.spread_bps = (snap.ask - snap.bid) / midpoint * 10_000.0

        analysis_cfg = getattr(self.config, "analysis", None)
        snap.refresh_source_ages(
            max_ticker_age_seconds=float(
                getattr(analysis_cfg, "max_ticker_age_seconds", 45.0)
            ),
            max_orderbook_age_seconds=float(
                getattr(analysis_cfg, "max_orderbook_age_seconds", 30.0)
            ),
        )
        if snap.execution_data_fresh is False:
            errors.append("execution_source_stale")

        # A mark/index quote is a safer last-price fallback than returning zero.
        if not snap.last:
            snap.last = next(
                (p for p in (snap.mark, snap.index, snap.bid, snap.ask) if p is not None and p > 0),
                0.0,
            )
        snap.errors = list(dict.fromkeys(errors))
        _cache_put(
            cache_key,
            snap,
            min(30, getattr(self, "cache_ttl_seconds", 300)),
        )
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
