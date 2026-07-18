"""Full technical indicator suite via pandas-ta-classic / pandas-ta, else pure pandas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.ta_backend import get_ta
from src.utils.config import AnalysisConfig, AppConfig
from src.utils.helpers import safe_float

# Resolved lazily on first compute_indicators() call
ta: Any = None
_TA_BACKEND = "none"


@dataclass
class DivergenceHit:
    kind: str  # bullish | bearish
    oscillator: str
    confidence: float
    note: str


@dataclass
class IndicatorSuite:
    """OHLCV with indicators attached + summary metrics for scoring."""

    df: pd.DataFrame
    summary: Dict[str, Any] = field(default_factory=dict)
    divergences: List[DivergenceHit] = field(default_factory=list)
    indicator_count: int = 0
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure pandas implementations (fallback + always available helpers)
# ---------------------------------------------------------------------------

def _ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def _sma(s: pd.Series, length: int) -> pd.Series:
    return s.rolling(length).mean()


def _rma(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(alpha=1 / length, adjust=False).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _rma(tr, length)


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ef = _ema(close, fast)
    es = _ema(close, slow)
    line = ef - es
    sig = _ema(line, signal)
    hist = line - sig
    return line, sig, hist


def _stoch(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    stoch_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _bbands(
    close: pd.Series, length: int = 20, std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = _sma(close, length)
    dev = close.rolling(length).std()
    upper = mid + std * dev
    lower = mid - std * dev
    return lower, mid, upper


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(high, low, close, length)
    plus_di = 100 * _rma(pd.Series(plus_dm, index=high.index), length) / atr.replace(0, np.nan)
    minus_di = 100 * _rma(pd.Series(minus_dm, index=high.index), length) / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return _rma(dx, length)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _cmf(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 20
) -> pd.Series:
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mfv = mfm * volume
    return mfv.rolling(length).sum() / volume.rolling(length).sum().replace(0, np.nan)


def _mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14
) -> pd.Series:
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    delta = tp.diff()
    pos = rmf.where(delta > 0, 0.0)
    neg = rmf.where(delta < 0, 0.0)
    pos_sum = pos.rolling(length).sum()
    neg_sum = neg.rolling(length).sum().replace(0, np.nan)
    ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + ratio))


def _willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return (typical * df["volume"]).cumsum() / cum_vol


def compute_indicators(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
    analysis_cfg: Optional[AnalysisConfig] = None,
) -> IndicatorSuite:
    """
    Compute a broad professional indicator stack.

    Backend order:
      1. ``pandas_ta_classic`` (PyPI: pandas-ta-classic)
      2. ``pandas_ta`` (legacy package)
      3. Pure-pandas fallback (always works)
    """
    global ta, _TA_BACKEND
    ta, _TA_BACKEND = get_ta()

    ac = analysis_cfg or (config.analysis if config else AnalysisConfig())
    if df is None or df.empty or len(df) < 30:
        return IndicatorSuite(
            df=df.copy() if df is not None else pd.DataFrame(),
            errors=["insufficient_bars"],
        )

    out = df.copy()
    rename = {c: c.lower() for c in out.columns}
    out = out.rename(columns=rename)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            return IndicatorSuite(df=out, errors=[f"missing_column_{col}"])

    errors: List[str] = []
    count = 0

    if ta is not None:
        try:
            out, count = _compute_with_pandas_ta(out, ac, ta)
            logger.debug(
                "Indicators via {} — {} columns",
                _TA_BACKEND,
                count,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "{} indicator path failed ({}); using pure-pandas fallback",
                _TA_BACKEND,
                exc,
            )
            errors.append(f"{_TA_BACKEND}_error:{exc}")
            out, count = _compute_fallback(out, ac)
            errors.append("used_fallback_indicators")
    else:
        logger.info("No pandas-ta backend — using pure-pandas indicator suite")
        out, count = _compute_fallback(out, ac)
        errors.append("pandas_ta_not_installed_fallback")

    summary = _build_summary(out, ac)
    summary["ta_backend"] = _TA_BACKEND
    divergences = detect_divergences(out)
    return IndicatorSuite(
        df=out,
        summary=summary,
        divergences=divergences,
        indicator_count=count,
        errors=errors,
    )


def _safe_assign(out: pd.DataFrame, name: str, series: Any) -> None:
    """Assign a Series into out without clobbering OHLCV."""
    if series is None:
        return
    if isinstance(series, pd.Series):
        out[name] = series
    elif isinstance(series, pd.DataFrame) and not series.empty:
        for c in series.columns:
            if str(c).lower() not in ("open", "high", "low", "close", "volume"):
                if c not in out.columns:
                    out[c] = series[c]


def _safe_join_df(out: pd.DataFrame, frame: Any) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return out
    for c in frame.columns:
        if c not in out.columns:
            out[c] = frame[c]
    return out


def _compute_with_pandas_ta(
    out: pd.DataFrame, ac: AnalysisConfig, ta_mod: Any
) -> Tuple[pd.DataFrame, int]:
    """
    Core suite via pandas_ta_classic / pandas_ta functional API.

    Uses ``ta_mod`` (already resolved) — do not import again inside.
    """
    ta = ta_mod
    base_cols = {"open", "high", "low", "close", "volume"}

    # --- Trend / MAs ---
    out["ema_fast"] = ta.ema(out["close"], length=ac.ema_fast)
    out["ema_mid"] = ta.ema(out["close"], length=ac.ema_mid)
    out["ema_slow"] = ta.ema(out["close"], length=ac.ema_slow)
    out["ema_trend"] = ta.ema(
        out["close"], length=min(ac.ema_trend, max(50, len(out) // 2))
    )
    out["sma_20"] = ta.sma(out["close"], length=20)
    out["sma_50"] = ta.sma(out["close"], length=min(50, len(out) - 1))
    out["sma_200"] = ta.sma(out["close"], length=min(200, len(out) - 1))
    try:
        out["hma_21"] = ta.hma(out["close"], length=21)
    except Exception as exc:
        logger.debug("hma failed ({}): {}", type(exc).__name__, exc)
        out["hma_21"] = _ema(out["close"], 21)
    out["vwap"] = _session_vwap(out)

    # --- Momentum ---
    out["rsi"] = ta.rsi(out["close"], length=ac.rsi_period)
    try:
        stoch = ta.stoch(out["high"], out["low"], out["close"], k=14, d=3, smooth_k=3)
        out = _safe_join_df(out, stoch)
    except Exception as exc:
        logger.debug("stoch failed: {}", exc)
    try:
        macd = ta.macd(
            out["close"],
            fast=ac.macd_fast,
            slow=ac.macd_slow,
            signal=ac.macd_signal,
        )
        out = _safe_join_df(out, macd)
    except Exception as exc:
        logger.debug("macd failed: {}", exc)
    try:
        out["willr"] = ta.willr(out["high"], out["low"], out["close"], length=14)
    except Exception as exc:
        logger.debug("willr failed: {}", exc)
    try:
        out["cci"] = ta.cci(out["high"], out["low"], out["close"], length=20)
    except Exception as exc:
        logger.debug("cci failed: {}", exc)
    try:
        out["roc"] = ta.roc(out["close"], length=10)
        out["mom"] = ta.mom(out["close"], length=10)
    except Exception as exc:
        logger.debug("roc/mom failed: {}", exc)
    try:
        out["ao"] = ta.ao(out["high"], out["low"])
    except Exception:
        out["ao"] = (
            _sma(out["high"] + out["low"], 5) / 2
            - _sma(out["high"] + out["low"], 34) / 2
        )

    # --- Volatility ---
    out["atr"] = ta.atr(out["high"], out["low"], out["close"], length=ac.atr_period)
    try:
        out["natr"] = ta.natr(out["high"], out["low"], out["close"], length=ac.atr_period)
    except Exception as exc:
        logger.debug("natr failed: {}", exc)
        out["natr"] = out["atr"] / out["close"] * 100
    try:
        bb = ta.bbands(out["close"], length=20, std=2.0)
        out = _safe_join_df(out, bb)
    except Exception as exc:
        logger.debug("bbands failed: {}", exc)
    out["donchian_h"] = out["high"].rolling(20).max()
    out["donchian_l"] = out["low"].rolling(20).min()
    out["std_20"] = out["close"].rolling(20).std()

    # --- Volume ---
    try:
        out["obv"] = ta.obv(out["close"], out["volume"])
    except Exception as exc:
        logger.debug("obv failed: {}", exc)
    try:
        out["cmf"] = ta.cmf(out["high"], out["low"], out["close"], out["volume"], length=20)
    except Exception:
        out["cmf"] = _cmf(out["high"], out["low"], out["close"], out["volume"], 20)
    try:
        out["mfi"] = ta.mfi(out["high"], out["low"], out["close"], out["volume"], length=14)
    except Exception as exc:
        logger.debug("mfi failed: {}", exc)
    out["vol_sma"] = ta.sma(out["volume"], length=20)
    out["vol_ratio"] = out["volume"] / out["vol_sma"].replace(0, np.nan)

    # --- Trend strength ---
    try:
        adx = ta.adx(out["high"], out["low"], out["close"], length=14)
        out = _safe_join_df(out, adx)
        # Convenience alias for summary
        adx_col = _col_like(out, "adx_14", "adx")
        if adx_col and "adx" not in out.columns:
            out["adx"] = out[adx_col]
    except Exception as exc:
        logger.debug("adx failed: {}", exc)

    try:
        st = ta.supertrend(
            out["high"], out["low"], out["close"], length=10, multiplier=3.0
        )
        out = _safe_join_df(out, st)
        # Aliases for confluence summary
        st_dir = _col_like(out, "supertd", "supertrend_dir")
        st_line = _col_like(out, "supert_")
        if st_line and "supertrend" not in out.columns:
            out["supertrend"] = out[st_line]
        if st_dir and "supertrend_dir" not in out.columns:
            out["supertrend_dir"] = out[st_dir]
    except Exception as exc:
        logger.warning("supertrend failed ({}): {}", type(exc).__name__, exc)

    try:
        ichi = ta.ichimoku(out["high"], out["low"], out["close"])
        if ichi is not None:
            if isinstance(ichi, tuple):
                for part in ichi:
                    if part is not None and hasattr(part, "empty") and not part.empty:
                        out = _safe_join_df(out, part)
            elif hasattr(ichi, "empty") and not ichi.empty:
                out = _safe_join_df(out, ichi)
    except Exception as exc:
        logger.warning("ichimoku failed ({}): {}", type(exc).__name__, exc)

    try:
        squeeze = ta.squeeze(out["high"], out["low"], out["close"])
        out = _safe_join_df(out, squeeze)
    except Exception as exc:
        logger.debug("squeeze failed: {}", exc)

    try:
        if hasattr(ta, "linreg"):
            out["linreg"] = ta.linreg(out["close"], length=20)
        else:
            out["linreg"] = out["close"].rolling(20).mean()
    except Exception:
        out["linreg"] = out["close"].rolling(20).mean()
    out["zscore"] = (out["close"] - out["close"].rolling(20).mean()) / out[
        "close"
    ].rolling(20).std()

    # --- Expanded suite (best-effort; never abort) ---
    extra_calls = [
        ("ema_12", lambda: ta.ema(out["close"], length=12)),
        ("ema_26", lambda: ta.ema(out["close"], length=26)),
        ("sma_100", lambda: ta.sma(out["close"], length=min(100, len(out) - 1))),
        ("wma_20", lambda: ta.wma(out["close"], length=20) if hasattr(ta, "wma") else None),
        ("tema_20", lambda: ta.tema(out["close"], length=20) if hasattr(ta, "tema") else None),
        ("dema_20", lambda: ta.dema(out["close"], length=20) if hasattr(ta, "dema") else None),
        ("kama_10", lambda: ta.kama(out["close"], length=10) if hasattr(ta, "kama") else None),
        ("rsi_7", lambda: ta.rsi(out["close"], length=7)),
        ("rsi_21", lambda: ta.rsi(out["close"], length=21)),
        ("cmo", lambda: ta.cmo(out["close"], length=14) if hasattr(ta, "cmo") else None),
        ("ppo", lambda: ta.ppo(out["close"]) if hasattr(ta, "ppo") else None),
        ("trix", lambda: ta.trix(out["close"]) if hasattr(ta, "trix") else None),
        ("fisher", lambda: ta.fisher(out["high"], out["low"]) if hasattr(ta, "fisher") else None),
        ("stochrsi", lambda: ta.stochrsi(out["close"]) if hasattr(ta, "stochrsi") else None),
        ("uo", lambda: ta.uo(out["high"], out["low"], out["close"]) if hasattr(ta, "uo") else None),
        ("bop", lambda: ta.bop(out["open"], out["high"], out["low"], out["close"]) if hasattr(ta, "bop") else None),
        ("donchian", lambda: ta.donchian(out["high"], out["low"]) if hasattr(ta, "donchian") else None),
        ("kc", lambda: ta.kc(out["high"], out["low"], out["close"]) if hasattr(ta, "kc") else None),
        ("efi", lambda: ta.efi(out["close"], out["volume"]) if hasattr(ta, "efi") else None),
        ("pvt", lambda: ta.pvt(out["close"], out["volume"]) if hasattr(ta, "pvt") else None),
        # classic eom signature: high, low, close, volume
        (
            "eom",
            lambda: ta.eom(out["high"], out["low"], out["close"], out["volume"])
            if hasattr(ta, "eom")
            else None,
        ),
        (
            "adosc",
            lambda: ta.adosc(out["high"], out["low"], out["close"], out["volume"])
            if hasattr(ta, "adosc")
            else None,
        ),
        ("vortex", lambda: ta.vortex(out["high"], out["low"], out["close"]) if hasattr(ta, "vortex") else None),
        ("aroon", lambda: ta.aroon(out["high"], out["low"]) if hasattr(ta, "aroon") else None),
        ("psar", lambda: ta.psar(out["high"], out["low"], out["close"]) if hasattr(ta, "psar") else None),
        ("er", lambda: ta.er(out["close"]) if hasattr(ta, "er") else None),
        ("slope", lambda: ta.slope(out["close"]) if hasattr(ta, "slope") else None),
        ("chop", lambda: ta.chop(out["high"], out["low"], out["close"]) if hasattr(ta, "chop") else None),
        (
            "ttm_trend",
            lambda: ta.ttm_trend(out["high"], out["low"], out["close"])
            if hasattr(ta, "ttm_trend")
            else None,
        ),
    ]
    for name, fn in extra_calls:
        try:
            res = fn()
            if res is None:
                continue
            if isinstance(res, pd.Series):
                if name not in out.columns:
                    out[name] = res
            elif isinstance(res, pd.DataFrame) and not res.empty:
                out = _safe_join_df(out, res)
        except Exception as exc:
            logger.debug("extra indicator {} failed: {}", name, exc)

    # NOTE: Do NOT run CommonStrategy / AllStrategy here.
    # They may spawn multiprocessing pools which break on Render / uvicorn.

    out = out.copy()  # defragment
    count = len([c for c in out.columns if str(c).lower() not in base_cols])
    return out, count


def _compute_fallback(out: pd.DataFrame, ac: AnalysisConfig) -> Tuple[pd.DataFrame, int]:
    """Comprehensive pure-pandas indicator suite (production fallback)."""
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]

    out["ema_fast"] = _ema(c, ac.ema_fast)
    out["ema_mid"] = _ema(c, ac.ema_mid)
    out["ema_slow"] = _ema(c, ac.ema_slow)
    out["ema_trend"] = _ema(c, min(ac.ema_trend, max(50, len(out) // 2)))
    out["sma_10"] = _sma(c, 10)
    out["sma_20"] = _sma(c, 20)
    out["sma_50"] = _sma(c, min(50, len(out) - 1))
    out["sma_100"] = _sma(c, min(100, len(out) - 1))
    out["sma_200"] = _sma(c, min(200, len(out) - 1))
    out["wma_20"] = c.rolling(20).apply(
        lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True
    )
    out["tema_20"] = 3 * _ema(c, 20) - 3 * _ema(_ema(c, 20), 20) + _ema(_ema(_ema(c, 20), 20), 20)
    out["dema_20"] = 2 * _ema(c, 20) - _ema(_ema(c, 20), 20)
    out["hma_21"] = _ema(c, 21)  # approx
    out["kama_10"] = _ema(c, 10)
    out["vwap"] = _session_vwap(out)
    out["vwma_20"] = (c * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    out["rsi"] = _rsi(c, ac.rsi_period)
    out["rsi_7"] = _rsi(c, 7)
    out["rsi_21"] = _rsi(c, 21)
    sk, sd = _stoch(h, l, c, 14, 3)
    out["stochk_14_3_3"] = sk
    out["stochd_14_3_3"] = sd
    # Stochastic RSI
    rsi = out["rsi"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    out["stochrsi_k"] = 100 * (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)

    macd, macds, macdh = _macd(c, ac.macd_fast, ac.macd_slow, ac.macd_signal)
    out[f"MACD_{ac.macd_fast}_{ac.macd_slow}_{ac.macd_signal}"] = macd
    out[f"MACDs_{ac.macd_fast}_{ac.macd_slow}_{ac.macd_signal}"] = macds
    out[f"MACDh_{ac.macd_fast}_{ac.macd_slow}_{ac.macd_signal}"] = macdh
    out["willr"] = _willr(h, l, c, 14)
    out["cci"] = _cci(h, l, c, 20)
    out["roc"] = c.pct_change(10) * 100
    out["mom"] = c.diff(10)
    out["tsi"] = _ema(_ema(c.diff(), 25), 13) / _ema(_ema(c.diff().abs(), 25), 13) * 100
    hl2 = (h + l) / 2
    out["ao"] = _sma(hl2, 5) - _sma(hl2, 34)
    # Ultimate Oscillator (simplified)
    bp = c - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    tr = pd.concat([h, c.shift(1)], axis=1).max(axis=1) - pd.concat([l, c.shift(1)], axis=1).min(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum().replace(0, np.nan)
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum().replace(0, np.nan)
    out["ultimate"] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    out["cmo"] = 100 * (c.diff().clip(lower=0).rolling(14).sum() - (-c.diff().clip(upper=0)).rolling(14).sum()) / (
        c.diff().abs().rolling(14).sum().replace(0, np.nan)
    )
    out["ppo"] = (_ema(c, 12) - _ema(c, 26)) / _ema(c, 26).replace(0, np.nan) * 100
    out["trix"] = _ema(_ema(_ema(c, 15), 15), 15).pct_change() * 100
    out["awesome"] = out["ao"]

    out["atr"] = _atr(h, l, c, ac.atr_period)
    out["atr_21"] = _atr(h, l, c, 21)
    out["natr"] = out["atr"] / c * 100
    bbl, bbm, bbu = _bbands(c, 20, 2.0)
    out["BBL_20_2.0"] = bbl
    out["BBM_20_2.0"] = bbm
    out["BBU_20_2.0"] = bbu
    out["BBB_20_2.0"] = (bbu - bbl) / bbm.replace(0, np.nan)
    out["BBP_20_2.0"] = (c - bbl) / (bbu - bbl).replace(0, np.nan)
    # Keltner
    out["KCLe_20_2"] = _ema(c, 20) - 2 * out["atr"]
    out["KCBe_20_2"] = _ema(c, 20)
    out["KCUe_20_2"] = _ema(c, 20) + 2 * out["atr"]
    out["donchian_h"] = h.rolling(20).max()
    out["donchian_l"] = l.rolling(20).min()
    out["donchian_m"] = (out["donchian_h"] + out["donchian_l"]) / 2
    out["std_10"] = c.rolling(10).std()
    out["std_20"] = c.rolling(20).std()
    out["std_50"] = c.rolling(50).std()
    out["historical_vol"] = c.pct_change().rolling(20).std() * np.sqrt(365 * 24 * 4) * 100
    out["chandelier_long"] = h.rolling(22).max() - out["atr"] * 3
    out["chandelier_short"] = l.rolling(22).min() + out["atr"] * 3

    out["obv"] = _obv(c, v)
    out["obv_ema"] = _ema(out["obv"], 20)
    out["cmf"] = _cmf(h, l, c, v, 20)
    out["mfi"] = _mfi(h, l, c, v, 14)
    out["ad"] = (((c - l) - (h - c)) / (h - l).replace(0, np.nan) * v).cumsum()
    out["adosc"] = _ema(out["ad"], 3) - _ema(out["ad"], 10)
    out["efi"] = _ema(c.diff() * v, 13)
    out["pvt"] = (c.pct_change() * v).cumsum()
    out["vol_sma"] = _sma(v, 20)
    out["vol_sma_50"] = _sma(v, 50)
    out["vol_ratio"] = v / out["vol_sma"].replace(0, np.nan)
    out["vol_zscore"] = (v - out["vol_sma"]) / v.rolling(20).std().replace(0, np.nan)
    out["vwap_upper"] = out["vwap"] + 2 * ((h + l + c) / 3 - out["vwap"]).rolling(20).std()
    out["vwap_lower"] = out["vwap"] - 2 * ((h + l + c) / 3 - out["vwap"]).rolling(20).std()
    # Force Index / Ease of Movement
    out["eom"] = (h + l).diff() / 2 * (h - l) / v.replace(0, np.nan)
    ret = c.pct_change().fillna(0).clip(-0.5, 0.5)
    out["nvi"] = (1 + ret.where(v < v.shift(1), 0.0)).cumprod() * 1000
    out["pvi"] = (1 + ret.where(v > v.shift(1), 0.0)).cumprod() * 1000

    out["adx"] = _adx(h, l, c, 14)
    out["ADX_14"] = out["adx"]
    # Supertrend approx
    hl2 = (h + l) / 2
    basic_ub = hl2 + 3 * out["atr"]
    basic_lb = hl2 - 3 * out["atr"]
    out["supertrend"] = np.where(c > basic_ub.shift(1), basic_lb, basic_ub)
    out["supertrend_dir"] = np.where(c > out["supertrend"], 1, -1)
    # Aroon
    def aroon_up(x):
        return 100 * (len(x) - 1 - np.argmax(x)) / (len(x) - 1)

    def aroon_down(x):
        return 100 * (len(x) - 1 - np.argmin(x)) / (len(x) - 1)

    out["aroon_up"] = h.rolling(25).apply(aroon_up, raw=True)
    out["aroon_down"] = l.rolling(25).apply(aroon_down, raw=True)
    out["aroon_osc"] = out["aroon_up"] - out["aroon_down"]
    # PSAR simplified
    out["psar"] = c.shift(1)
    # Vortex
    tr_s = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    vm_plus = (h - l.shift(1)).abs()
    vm_minus = (l - h.shift(1)).abs()
    out["vortex_pos"] = vm_plus.rolling(14).sum() / tr_s.rolling(14).sum().replace(0, np.nan)
    out["vortex_neg"] = vm_minus.rolling(14).sum() / tr_s.rolling(14).sum().replace(0, np.nan)
    # Mass index
    hl_ema = _ema(h - l, 9)
    out["mass"] = (hl_ema / _ema(hl_ema, 9)).rolling(25).sum()

    # Ichimoku
    out["ichimoku_tenkan"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    out["ichimoku_kijun"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    out["ichimoku_span_a"] = ((out["ichimoku_tenkan"] + out["ichimoku_kijun"]) / 2).shift(26)
    out["ichimoku_span_b"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    out["ichimoku_chikou"] = c.shift(-26)

    # Misc stats
    out["linreg"] = c.rolling(20).apply(
        lambda x: np.polyval(np.polyfit(np.arange(len(x)), x, 1), len(x) - 1), raw=True
    )
    out["slope"] = c.rolling(10).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    out["zscore"] = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
    out["zscore_50"] = (c - c.rolling(50).mean()) / c.rolling(50).std().replace(0, np.nan)
    out["skew"] = c.rolling(20).skew()
    out["kurtosis"] = c.rolling(20).kurt()
    out["quantile"] = c.rolling(20).quantile(0.5)
    out["entropy"] = c.pct_change().rolling(10).apply(
        lambda x: -np.sum(np.histogram(x.dropna(), bins=5, density=True)[0] *
                          np.log(np.histogram(x.dropna(), bins=5, density=True)[0] + 1e-12))
        if len(x.dropna()) > 3 else np.nan,
        raw=False,
    )
    # Squeeze: BB inside KC
    out["squeeze_on"] = ((out["BBL_20_2.0"] > out["KCLe_20_2"]) & (out["BBU_20_2.0"] < out["KCUe_20_2"])).astype(float)
    # Donchian breakout flags
    out["breakout_up"] = (c >= out["donchian_h"].shift(1)).astype(float)
    out["breakout_dn"] = (c <= out["donchian_l"].shift(1)).astype(float)
    # Heikin-ashi close for smoothness
    out["ha_close"] = (out["open"] + h + l + c) / 4
    out["ha_ema"] = _ema(out["ha_close"], 20)
    # Pivot points (classic daily-style on rolling window)
    out["pivot"] = (h.rolling(1).max() + l.rolling(1).min() + c) / 3  # placeholder bar pivot
    out["pivot_r1"] = 2 * out["pivot"] - l
    out["pivot_s1"] = 2 * out["pivot"] - h
    # Fisher transform
    med = (h + l) / 2
    mn, mx = med.rolling(10).min(), med.rolling(10).max()
    x = 0.33 * 2 * ((med - mn) / (mx - mn).replace(0, np.nan) - 0.5)
    x = x.clip(-0.999, 0.999)
    out["fisher"] = 0.5 * np.log((1 + x) / (1 - x))
    # Coppock-like
    out["coppock"] = _ema(c.pct_change(14) + c.pct_change(11), 10) * 100
    # DPO
    out["dpo"] = c.shift(11) - _sma(c, 20)
    # KST simplified
    out["kst"] = (
        c.pct_change(10).rolling(10).mean()
        + 2 * c.pct_change(15).rolling(10).mean()
        + 3 * c.pct_change(20).rolling(10).mean()
        + 4 * c.pct_change(30).rolling(15).mean()
    ) * 100

    # Defragment after many column inserts
    out = out.copy()

    # Count non-ohlcv columns
    base = {"open", "high", "low", "close", "volume"}
    count = len([col for col in out.columns if col not in base])
    return out, count


def _col_like(df: pd.DataFrame, *needles: str) -> Optional[str]:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for n in needles:
        n = n.lower()
        for lc, orig in lower_map.items():
            if n in lc:
                return orig
    return None


def _build_summary(df: pd.DataFrame, ac: AnalysisConfig) -> Dict[str, Any]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    close = safe_float(last.get("close"))

    def g(*names: str, default: float = float("nan")) -> float:
        for n in names:
            if n in df.columns and pd.notna(last.get(n)):
                return safe_float(last.get(n))
            col = _col_like(df, n)
            if col and pd.notna(last.get(col)):
                return safe_float(last.get(col))
        return default

    rsi = g("rsi")
    atr = g("atr")
    adx = g("adx_14", "adx")
    # Classic MACD cols: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
    macd_cols = [
        c
        for c in df.columns
        if str(c).upper().startswith("MACD")
        and "S_" not in str(c).upper()
        and "H_" not in str(c).upper()
        and str(c).upper() != "MACD"
        or str(c).lower() == "macd"
    ]
    # Prefer exact MACD_ prefix without s/h
    macd_line_cols = [
        c
        for c in df.columns
        if str(c).upper().startswith("MACD_")
        and not str(c).upper().startswith("MACDS")
        and not str(c).upper().startswith("MACDH")
    ]
    macds_col = _col_like(df, "macds_")
    macdh_col = _col_like(df, "macdh_")
    if macd_line_cols:
        macd = safe_float(last.get(macd_line_cols[0]))
    elif macd_cols:
        macd = safe_float(last.get(macd_cols[0]))
    else:
        macd = float("nan")
    macds = safe_float(last.get(macds_col)) if macds_col else float("nan")
    macdh = safe_float(last.get(macdh_col)) if macdh_col else float("nan")

    ema_fast = g("ema_fast")
    ema_mid = g("ema_mid")
    ema_slow = g("ema_slow")
    ema_trend = g("ema_trend")
    vol_ratio = g("vol_ratio", default=1.0)
    cmf = g("cmf")
    mfi = g("mfi")
    # Classic bbands: BBL_20_2.0 / BBU_20_2.0
    bb_lower = g("bbl_20_2.0", "bbl_20_2", "bbl", default=float("nan"))
    bb_upper = g("bbu_20_2.0", "bbu_20_2", "bbu", default=float("nan"))

    trend_votes = 0.0
    if not np.isnan(ema_fast) and not np.isnan(ema_slow):
        trend_votes += 0.35 if ema_fast > ema_slow else -0.35
    if not np.isnan(ema_mid) and not np.isnan(ema_trend):
        trend_votes += 0.25 if ema_mid > ema_trend else -0.25
    if not np.isnan(ema_trend):
        trend_votes += 0.25 if close > ema_trend else -0.25
    if len(df) >= 20:
        slope = (close - safe_float(df["close"].iloc[-20])) / max(atr if not np.isnan(atr) else 1e-12, 1e-12)
        trend_votes += float(np.clip(slope / 10.0, -0.15, 0.15))

    mom = 0.0
    if not np.isnan(rsi):
        if rsi >= 55:
            mom += 0.25
        elif rsi <= 45:
            mom -= 0.25
        if rsi >= 70:
            mom -= 0.1
        if rsi <= 30:
            mom += 0.1
    if not np.isnan(macdh):
        mom += 0.2 if macdh > 0 else -0.2
    if not np.isnan(macd) and not np.isnan(macds):
        mom += 0.15 if macd > macds else -0.15

    bb_pos = 0.5
    if not np.isnan(bb_lower) and not np.isnan(bb_upper) and bb_upper != bb_lower:
        bb_pos = float(np.clip((close - bb_lower) / (bb_upper - bb_lower), 0, 1))

    vol_bias = 0.0
    if not np.isnan(vol_ratio):
        if vol_ratio > 1.5:
            vol_bias = 0.2 if close >= safe_float(prev.get("close")) else -0.2
        elif vol_ratio < 0.7:
            vol_bias = -0.05

    return {
        "close": close,
        "rsi": None if np.isnan(rsi) else rsi,
        "atr": None if np.isnan(atr) else atr,
        "adx": None if np.isnan(adx) else adx,
        "macd": None if np.isnan(macd) else macd,
        "macd_signal": None if np.isnan(macds) else macds,
        "macd_hist": None if np.isnan(macdh) else macdh,
        "ema_fast": None if np.isnan(ema_fast) else ema_fast,
        "ema_mid": None if np.isnan(ema_mid) else ema_mid,
        "ema_slow": None if np.isnan(ema_slow) else ema_slow,
        "ema_trend": None if np.isnan(ema_trend) else ema_trend,
        "vol_ratio": None if np.isnan(vol_ratio) else vol_ratio,
        "cmf": None if np.isnan(cmf) else cmf,
        "mfi": None if np.isnan(mfi) else mfi,
        "bb_position": bb_pos,
        "trend_score": float(np.clip(trend_votes, -1, 1)),
        "momentum_score": float(np.clip(mom, -1, 1)),
        "volume_bias": float(np.clip(vol_bias, -1, 1)),
        "natr": g("natr"),
        "willr": g("willr"),
        "cci": g("cci"),
        "stoch_k": g("stochk_14_3_3", "stk_14_3_3"),
        "stoch_d": g("stochd_14_3_3", "std_14_3_3"),
        "supertrend_dir": g("supertrend_dir"),
        "above_vwap": bool(close > g("vwap")) if not np.isnan(g("vwap")) else None,
    }


def detect_divergences(df: pd.DataFrame, lookback: int = 40) -> List[DivergenceHit]:
    """Pivot-based RSI / MACD / Stochastic divergences (approximate)."""
    hits: List[DivergenceHit] = []
    if df is None or len(df) < lookback + 5:
        return hits

    window = df.tail(lookback).copy()
    close = window["close"].values.astype(float)

    oscillators: List[Tuple[str, np.ndarray]] = []
    if "rsi" in window.columns:
        oscillators.append(("RSI", window["rsi"].values.astype(float)))
    macdh = _col_like(window, "macdh")
    if macdh:
        oscillators.append(("MACD_hist", window[macdh].values.astype(float)))
    stoch_k = _col_like(window, "stochk", "stk_")
    if stoch_k:
        oscillators.append(("Stochastic", window[stoch_k].values.astype(float)))

    price_lows = _pivot_indices(close, mode="low", order=3)
    price_highs = _pivot_indices(close, mode="high", order=3)

    for name, osc in oscillators:
        if np.all(np.isnan(osc)):
            continue
        osc_lows = _pivot_indices(osc, mode="low", order=3)
        osc_highs = _pivot_indices(osc, mode="high", order=3)

        if len(price_lows) >= 2 and len(osc_lows) >= 2:
            p1, p2 = price_lows[-2], price_lows[-1]
            o1 = _nearest(osc_lows, p1)
            o2 = _nearest(osc_lows, p2)
            if o1 is not None and o2 is not None and p2 > p1:
                if close[p2] < close[p1] and osc[o2] > osc[o1]:
                    conf = float(
                        np.clip(abs(osc[o2] - osc[o1]) / (abs(osc[o1]) + 1e-9) * 40 + 45, 40, 88)
                    )
                    hits.append(
                        DivergenceHit(
                            kind="bullish",
                            oscillator=name,
                            confidence=conf,
                            note=f"Price LL vs {name} HL (bars {p1}→{p2})",
                        )
                    )

        if len(price_highs) >= 2 and len(osc_highs) >= 2:
            p1, p2 = price_highs[-2], price_highs[-1]
            o1 = _nearest(osc_highs, p1)
            o2 = _nearest(osc_highs, p2)
            if o1 is not None and o2 is not None and p2 > p1:
                if close[p2] > close[p1] and osc[o2] < osc[o1]:
                    conf = float(
                        np.clip(abs(osc[o1] - osc[o2]) / (abs(osc[o1]) + 1e-9) * 40 + 45, 40, 88)
                    )
                    hits.append(
                        DivergenceHit(
                            kind="bearish",
                            oscillator=name,
                            confidence=conf,
                            note=f"Price HH vs {name} LH (bars {p1}→{p2})",
                        )
                    )

    return hits


def _pivot_indices(arr: np.ndarray, mode: str = "low", order: int = 3) -> List[int]:
    idxs: List[int] = []
    n = len(arr)
    for i in range(order, n - order):
        if np.isnan(arr[i]):
            continue
        left = arr[i - order : i]
        right = arr[i + 1 : i + 1 + order]
        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            continue
        if mode == "low" and arr[i] <= left.min() and arr[i] <= right.min():
            idxs.append(i)
        if mode == "high" and arr[i] >= left.max() and arr[i] >= right.max():
            idxs.append(i)
    return idxs


def _nearest(idxs: List[int], target: int, max_dist: int = 5) -> Optional[int]:
    if not idxs:
        return None
    best = min(idxs, key=lambda i: abs(i - target))
    return best if abs(best - target) <= max_dist else None
