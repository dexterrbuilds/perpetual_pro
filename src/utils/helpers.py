"""Shared helpers for symbol normalization, IO, and numeric safety."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if missing; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce value to float; return default on failure."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Coerce value to int; return default on failure."""
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))


def normalize_symbol(raw: str, quote: str = "USDT") -> str:
    """
    Normalize user/OCR symbol strings to CCXT unified perpetual format.

    Examples:
        btc -> BTC/USDT:USDT
        BTCUSDT -> BTC/USDT:USDT
        BTC/USDT -> BTC/USDT:USDT
        ETH-PERP -> ETH/USDT:USDT
        1000PEPE -> 1000PEPE/USDT:USDT
        BONK -> BONK/USDT:USDT
    """
    if not raw:
        raise ValueError("Symbol is empty")

    s = raw.strip().upper()
    s = s.replace(" ", "")
    s = s.replace("-", "").replace("_", "")

    # Strip common suffixes
    for suffix in ("PERP", "SWAP", "USDTM", "USD-M"):
        if s.endswith(suffix) and len(s) > len(suffix) + 1:
            s = s[: -len(suffix)]

    # Already unified with settle
    if ":" in s and "/" in s:
        return s

    # BTC/USDT or BTC/USDT:USDT
    if "/" in s:
        base, rest = s.split("/", 1)
        rest = rest.split(":")[0] or quote
        if rest in {"USD", "BUSD"}:
            rest = "USDT"
        return f"{base}/{rest}:{rest}"

    # BTCUSDT / ETHUSDC / BONKUSD
    for q in ("USDT", "USDC", "USD", "BUSD"):
        if s.endswith(q) and len(s) > len(q):
            base = s[: -len(q)]
            settle = "USDT" if q in ("USD", "BUSD") else q
            return f"{base}/{settle}:{settle}"

    # Bare base asset (e.g. BONK -> BONK/USDT:USDT)
    base = re.sub(r"[^A-Z0-9]", "", s)
    if not base:
        raise ValueError(f"Cannot parse symbol from: {raw!r}")
    return f"{base}/{quote}:{quote}"


def symbol_base(symbol: str) -> str:
    """Extract base asset from unified or raw symbol (e.g. BTC)."""
    s = symbol.upper().replace(" ", "")
    if "/" in s:
        return s.split("/")[0]
    for q in ("USDT", "USDC", "USD", "BUSD"):
        if s.endswith(q):
            return s[: -len(q)]
    return re.sub(r"[^A-Z0-9]", "", s) or s


def timeframe_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes for sorting/comparison."""
    tf = tf.strip().lower()
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
        "3d": 4320,
        "1w": 10080,
        "1M": 43200,
    }
    if tf in mapping:
        return mapping[tf]
    # Parse like 45m, 2d
    m = re.match(r"^(\d+)([mhdw])$", tf)
    if not m:
        return 15
    n, unit = int(m.group(1)), m.group(2)
    mult = {"m": 1, "h": 60, "d": 1440, "w": 10080}[unit]
    return n * mult


def format_price(price: float, ref: Optional[float] = None) -> str:
    """Format price with adaptive decimals based on magnitude."""
    p = abs(price if ref is None else ref)
    if p >= 1000:
        return f"{price:,.2f}"
    if p >= 1:
        return f"{price:,.4f}"
    if p >= 0.01:
        return f"{price:.6f}"
    return f"{price:.8f}"


def pct_change(a: float, b: float) -> float:
    """Percent change from a to b: (b-a)/a * 100."""
    if a == 0:
        return 0.0
    return (b - a) / a * 100.0


def slugify(text: str) -> str:
    """Filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:80] or "report"
