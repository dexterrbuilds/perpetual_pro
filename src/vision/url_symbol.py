"""Extract trading pair / symbol / timeframe from chart page URLs (esp. TradingView)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from src.utils.helpers import normalize_symbol


@dataclass
class UrlHints:
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    exchange_hint: Optional[str] = None
    source: str = ""
    raw_pair: str = ""
    confidence: float = 0.0


# TradingView interval map (common)
_TV_INTERVAL = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "45": "45m",
    "60": "1h",
    "120": "2h",
    "180": "3h",
    "240": "4h",
    "D": "1d",
    "1D": "1d",
    "W": "1w",
    "1W": "1w",
    "M": "1M",
    "1M": "1M",
}


def parse_chart_url(url: Optional[str]) -> UrlHints:
    """Best-effort parse of TradingView / exchange chart URLs."""
    if not url or not str(url).strip():
        return UrlHints()

    url = str(url).strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return UrlHints()

    host = (parsed.netloc or "").lower()
    path = unquote(parsed.path or "")
    query = parse_qs(parsed.query or "")

    # --- TradingView ---
    if "tradingview.com" in host or "tradingview." in host:
        return _parse_tradingview(path, query, url)

    # --- Binance futures ---
    if "binance.com" in host:
        m = re.search(r"/(?:futures|trade)/(?:[a-z_]+/)?([A-Z0-9]+)([A-Z]{3,5})", path, re.I)
        if m:
            pair = f"{m.group(1).upper()}{m.group(2).upper()}"
            return UrlHints(
                symbol=_safe_norm(pair),
                exchange_hint="binanceusdm" if "futures" in path.lower() else "binance",
                source="binance_url",
                raw_pair=pair,
                confidence=0.85,
            )

    # --- Bybit ---
    if "bybit.com" in host:
        m = re.search(r"/trade/(?:usdt/)?([A-Z0-9]+)", path, re.I)
        if m:
            base = m.group(1).upper()
            pair = base if base.endswith("USDT") else f"{base}USDT"
            return UrlHints(
                symbol=_safe_norm(pair),
                exchange_hint="bybit",
                source="bybit_url",
                raw_pair=pair,
                confidence=0.8,
            )

    # --- OKX ---
    if "okx.com" in host:
        m = re.search(r"([A-Z0-9]+)-USDT", path, re.I)
        if m:
            pair = f"{m.group(1).upper()}USDT"
            return UrlHints(
                symbol=_safe_norm(pair),
                exchange_hint="okx",
                source="okx_url",
                raw_pair=pair,
                confidence=0.8,
            )

    # --- Bitget ---
    if "bitget.com" in host:
        m = re.search(r"([A-Z0-9]+)USDT", path, re.I)
        if m:
            pair = m.group(0).upper()
            return UrlHints(
                symbol=_safe_norm(pair),
                exchange_hint="bitget",
                source="bitget_url",
                raw_pair=pair,
                confidence=0.75,
            )

    # Generic path token: BTCUSDT / BTC-USDT / BTC_USDT
    m = re.search(r"\b([A-Z]{2,12})[-_]?USDT\b", path.upper())
    if m:
        pair = f"{m.group(1)}USDT"
        return UrlHints(
            symbol=_safe_norm(pair),
            source="generic_url",
            raw_pair=pair,
            confidence=0.55,
        )

    return UrlHints(source="url_unparsed")


def _parse_tradingview(path: str, query: dict, url: str) -> UrlHints:
    """
    Examples:
      https://www.tradingview.com/chart/?symbol=BINANCE:BTCUSDT.P
      https://www.tradingview.com/chart/BTCUSDT/
      https://www.tradingview.com/symbols/BTCUSDT/
      symbol=BYBIT:ETHUSDT.P
    """
    hints = UrlHints(source="tradingview", confidence=0.0)
    raw = ""

    # Query symbol=
    sym_q = (query.get("symbol") or [None])[0]
    if sym_q:
        raw = unquote(str(sym_q))
        hints.confidence = 0.95

    # /symbols/EXCHANGE-BTCUSDT/ or /symbols/BTCUSDT/
    if not raw:
        m = re.search(r"/symbols/([^/]+)/?", path, re.I)
        if m:
            raw = m.group(1)
            hints.confidence = 0.9

    # /chart/XXXX/ sometimes short id — weaker
    if not raw:
        m = re.search(r"/chart/([A-Z0-9._:-]{3,40})/?", path, re.I)
        if m and not re.match(r"^[a-f0-9]{8,}$", m.group(1), re.I):
            raw = m.group(1)
            hints.confidence = 0.55

    if not raw:
        # Last resort in full URL
        m = re.search(r"([A-Z]{2,12})USDT(?:\.P)?", url.upper())
        if m:
            raw = m.group(0)
            hints.confidence = 0.5

    if raw:
        exchange, pair = _split_tv_symbol(raw)
        hints.raw_pair = pair
        hints.exchange_hint = exchange
        hints.symbol = _safe_norm(pair)
        if hints.symbol:
            hints.confidence = max(hints.confidence, 0.85)

    # Interval
    for key in ("interval", "i"):
        if key in query and query[key]:
            iv = str(query[key][0]).upper()
            hints.timeframe = _TV_INTERVAL.get(iv, _normalize_tf(iv))
            break

    return hints


def _split_tv_symbol(raw: str) -> tuple:
    """BINANCE:BTCUSDT.P → (binanceusdm, BTCUSDT)."""
    s = raw.strip().upper().replace(" ", "")
    exchange = None
    if ":" in s:
        ex, s = s.split(":", 1)
        ex = ex.lower()
        mapping = {
            "binance": "binanceusdm",
            "binanceusdm": "binanceusdm",
            "binanceus": "binanceusdm",
            "bybit": "bybit",
            "okx": "okx",
            "bitget": "bitget",
            "coinbase": "coinbase",
        }
        exchange = mapping.get(ex, ex)

    # Strip .P perpetual suffix, PERP, etc.
    s = re.sub(r"\.(P|PERP)$", "", s)
    s = s.replace(".P", "").replace("PERP", "")
    s = s.replace("-", "").replace("_", "")
    if s.endswith("USDT.P"):
        s = s.replace("USDT.P", "USDT")
    return exchange, s


def _normalize_tf(iv: str) -> Optional[str]:
    iv = iv.strip().lower()
    if re.match(r"^\d+[mhdw]$", iv):
        return iv
    if iv.isdigit():
        return _TV_INTERVAL.get(iv)
    return _TV_INTERVAL.get(iv.upper())


def _safe_norm(pair: str) -> Optional[str]:
    try:
        return normalize_symbol(pair)
    except Exception:
        return None
