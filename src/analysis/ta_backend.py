"""Resolve technical-analysis backend: pandas-ta-classic → pandas_ta → None.

``pandas-ta-classic`` (PyPI) installs as ``pandas_ta_classic``.
Original ``pandas-ta`` installs as ``pandas_ta``.

Both expose the same functional API used by this project:
``ta.ema``, ``ta.rsi``, ``ta.macd``, ``ta.bbands``, ``ta.atr``, ``ta.adx``,
``ta.supertrend``, ``ta.ichimoku``, etc.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from loguru import logger

_ta: Any = None
_backend: Optional[str] = None  # "pandas_ta_classic" | "pandas_ta" | "none"
_load_attempted: bool = False


def get_ta() -> Tuple[Any, str]:
    """
    Lazy-load and return ``(ta_module, backend_name)``.

    backend_name is one of: ``pandas_ta_classic``, ``pandas_ta``, ``none``.
    """
    global _ta, _backend, _load_attempted
    if _load_attempted:
        return _ta, _backend or "none"
    _load_attempted = True

    # 1) Preferred: maintained classic fork (Render / modern Python)
    try:
        import pandas_ta_classic as classic  # type: ignore

        _ta = classic
        ver = getattr(classic, "version", None) or getattr(classic, "__version__", "?")
        _backend = "pandas_ta_classic"
        logger.info("Indicator backend: pandas_ta_classic ({})", ver)
        return _ta, _backend
    except Exception as exc:  # noqa: BLE001 — ImportError or broken install
        logger.warning(
            "pandas_ta_classic unavailable ({}); trying pandas_ta…",
            exc,
        )

    # 2) Original package (if present)
    try:
        import pandas_ta as original  # type: ignore

        _ta = original
        ver = getattr(original, "__version__", "?")
        _backend = "pandas_ta"
        logger.info("Indicator backend: pandas_ta ({})", ver)
        return _ta, _backend
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pandas_ta unavailable ({}); using pure-pandas indicator fallback",
            exc,
        )

    _ta = None
    _backend = "none"
    return _ta, _backend


def ta_available() -> bool:
    mod, name = get_ta()
    return mod is not None and name != "none"
