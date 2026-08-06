"""Strict closed-candle setup recognition for production day-trade playbooks.

This module classifies *how* an already-directional thesis may be traded.  It
does not create direction, alter Technical/Overall/Execution Quality, or bypass
any publication gate.  Every detected setup still enters the normal execution,
risk, qualification, and pre-delivery revalidation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.analysis.indicators import IndicatorSuite
from src.analysis.market_structure import StructureLevel, StructureReport
from src.utils.helpers import clamp, safe_float


SETUP_POLICY_VERSION = "strict_intraday_setups_v1"


@dataclass(frozen=True)
class StrictSetupPolicy:
    """Thresholds used only to recognize the new setup families."""

    trend_min_score: float = 0.35
    trend_min_adx: float = 18.0
    trend_max_pullback_distance_atr: float = 0.65
    trend_min_candle_score: float = 0.08
    trend_min_directional_momentum: float = -0.05

    reversal_min_candle_score: float = 0.20
    reversal_min_rejection_wick: float = 0.35
    reversal_min_volume_ratio: float = 1.10

    sweep_min_breach_atr: float = 0.03
    sweep_min_reclaim_atr: float = 0.02
    sweep_min_rejection_wick: float = 0.32
    sweep_min_candle_score: float = 0.18
    sweep_min_volume_ratio: float = 1.05

    breakout_min_displacement_atr: float = 0.08
    breakout_min_volume_ratio: float = 1.15
    breakout_max_retest_distance_atr: float = 0.18
    breakout_min_reclaim_atr: float = 0.03
    breakout_min_candle_score: float = 0.10

    zone_max_age_bars: int = 48
    zone_max_touches: int = 1
    zone_max_mitigation: float = 0.75
    zone_touch_buffer_atr: float = 0.08
    zone_min_candle_score: float = 0.10


DEFAULT_STRICT_SETUP_POLICY = StrictSetupPolicy()


@dataclass
class SetupDetection:
    setup_type: str = ""
    label: str = ""
    passed: bool = False
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    policy_version: str = SETUP_POLICY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setup_type": self.setup_type,
            "label": self.label,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
            "policy_version": self.policy_version,
        }


@dataclass
class SetupCandleEvidence:
    score: float
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    volume_ratio: float
    adverse_rejection: bool
    absorption: bool


def _setup_candle_evidence(
    df: pd.DataFrame,
    atr: float,
    direction: str,
) -> SetupCandleEvidence:
    work = df.tail(80)
    o = work["open"].astype(float)
    h = work["high"].astype(float)
    l = work["low"].astype(float)
    c = work["close"].astype(float)
    v = work["volume"].astype(float).clip(lower=0)
    candle_range = max(float(h.iloc[-1] - l.iloc[-1]), atr * 0.05)
    body = abs(float(c.iloc[-1] - o.iloc[-1])) / candle_range
    upper = (float(h.iloc[-1]) - max(float(o.iloc[-1]), float(c.iloc[-1]))) / candle_range
    lower = (min(float(o.iloc[-1]), float(c.iloc[-1])) - float(l.iloc[-1])) / candle_range
    close_location = float(clamp((float(c.iloc[-1]) - float(l.iloc[-1])) / candle_range, 0, 1))
    volume_base = max(float(v.tail(30).median()), 1e-12)
    volume_ratio = float(v.iloc[-1]) / volume_base
    signed = v * (
        2.0 * ((c - l) / (h - l).replace(0, np.nan)).clip(0, 1).fillna(0.5)
        - 1.0
    )
    cvd = float(clamp(float(signed.tail(8).sum()) / max(float(v.tail(8).sum()), 1e-12), -1, 1))
    recent_return = float(c.iloc[-1] / c.iloc[-4] - 1.0) if len(c) >= 4 else 0.0
    impulse = float(clamp(recent_return / max(atr / max(float(c.iloc[-1]), 1e-12), 1e-6), -1, 1))
    order_flow = float(clamp(0.72 * cvd + 0.28 * impulse, -1, 1))
    sign = 1.0 if direction == "long" else -1.0
    score = float(clamp(sign * (
        0.45 * (2.0 * close_location - 1.0)
        + 0.35 * np.sign(float(c.iloc[-1] - o.iloc[-1])) * body
        + 0.20 * order_flow
    ), -1, 1))
    adverse = (
        direction == "long" and upper >= 0.45 and close_location < 0.55
    ) or (
        direction == "short" and lower >= 0.45 and close_location > 0.45
    )
    return SetupCandleEvidence(
        score=score,
        body_ratio=body,
        upper_wick_ratio=upper,
        lower_wick_ratio=lower,
        volume_ratio=volume_ratio,
        adverse_rejection=bool(adverse),
        absorption=bool(volume_ratio >= 1.5 and body <= 0.28),
    )


def detect_strict_setup(
    df: pd.DataFrame,
    indicators: IndicatorSuite,
    structure: StructureReport,
    *,
    direction: str,
    price: float,
    atr: float,
    candle: Optional[Any] = None,
    policy: StrictSetupPolicy = DEFAULT_STRICT_SETUP_POLICY,
) -> Optional[SetupDetection]:
    """Return the highest-priority strict setup supported by closed candles."""
    if direction not in {"long", "short"} or df is None or len(df) < 30 or atr <= 0:
        return None
    candle = candle or _setup_candle_evidence(df, atr, direction)
    side = "Long" if direction == "long" else "Short"

    # Highest priority: join an established trend on a controlled pullback.
    trend = _trend_pullback(indicators, structure, direction, price, atr, candle, policy)
    if trend is not None:
        trend.label = f"{side} Trend Continuation (Pullback)"
        return trend

    breakout = _breakout_retest(df, direction, atr, candle, policy)
    if breakout is not None:
        breakout.label = f"{side} Breakout + Retest"
        return breakout

    zone = _ob_fvg_retest(df, structure, direction, atr, candle, policy)
    if zone is not None:
        zone.label = f"{side} Order Block / FVG Retest"
        return zone

    sweep = _liquidity_sweep(df, direction, atr, candle, policy)
    if sweep is not None:
        sweep.label = f"{side} Liquidity Sweep"
        return sweep

    reversal = _confirmed_counter_trend_reversal(
        indicators, structure, direction, candle, policy
    )
    if reversal is not None:
        reversal.label = f"{side} Confirmed Counter-Trend Reversal"
        return reversal
    return None


def _trend_pullback(
    indicators: IndicatorSuite,
    structure: StructureReport,
    direction: str,
    price: float,
    atr: float,
    candle: Any,
    policy: StrictSetupPolicy,
) -> Optional[SetupDetection]:
    sign = 1.0 if direction == "long" else -1.0
    wanted_trend = "up" if direction == "long" else "down"
    summary = indicators.summary or {}
    trend_score = sign * safe_float(summary.get("trend_score"))
    momentum = sign * safe_float(summary.get("momentum_score"))
    adx = safe_float(summary.get("adx"))
    reference_values = [
        safe_float(summary.get("ema_mid")),
        safe_float(summary.get("vwap")),
    ]
    reference_values = [value for value in reference_values if value > 0]
    pullback_distance = min(
        (abs(price - value) / atr for value in reference_values),
        default=float("inf"),
    )
    structure_aligned = structure.trend == wanted_trend
    passed = bool(
        structure_aligned
        and trend_score >= policy.trend_min_score
        and adx >= policy.trend_min_adx
        and pullback_distance <= policy.trend_max_pullback_distance_atr
        and candle.score >= policy.trend_min_candle_score
        and momentum >= policy.trend_min_directional_momentum
        and not candle.adverse_rejection
    )
    if not passed:
        return None
    return SetupDetection(
        setup_type="trend_pullback",
        passed=True,
        reasons=["Established structure trend retested EMA21/VWAP with a confirming close"],
        evidence={
            "trend_score": round(trend_score, 3),
            "adx": round(adx, 2),
            "pullback_distance_atr": round(pullback_distance, 3),
            "candle_score": round(candle.score, 3),
            "directional_momentum": round(momentum, 3),
        },
    )


def _liquidity_sweep(
    df: pd.DataFrame,
    direction: str,
    atr: float,
    candle: Any,
    policy: StrictSetupPolicy,
) -> Optional[SetupDetection]:
    work = df.tail(32)
    if len(work) < 22:
        return None
    prior = work.iloc[-21:-1]
    last = work.iloc[-1]
    if direction == "long":
        reference = float(prior["low"].min())
        breach = (reference - float(last["low"])) / atr
        reclaim = (float(last["close"]) - reference) / atr
        wick = candle.lower_wick_ratio
    else:
        reference = float(prior["high"].max())
        breach = (float(last["high"]) - reference) / atr
        reclaim = (reference - float(last["close"])) / atr
        wick = candle.upper_wick_ratio
    if not (
        breach >= policy.sweep_min_breach_atr
        and reclaim >= policy.sweep_min_reclaim_atr
        and wick >= policy.sweep_min_rejection_wick
        and candle.score >= policy.sweep_min_candle_score
        and candle.volume_ratio >= policy.sweep_min_volume_ratio
        and not candle.absorption
    ):
        return None
    return SetupDetection(
        setup_type="liquidity_sweep",
        passed=True,
        reasons=["Previous 20-bar extreme was swept and reclaimed on directional volume"],
        evidence={
            "swept_level": reference,
            "breach_atr": round(breach, 3),
            "reclaim_atr": round(reclaim, 3),
            "rejection_wick": round(wick, 3),
            "volume_ratio": round(candle.volume_ratio, 3),
            "candle_score": round(candle.score, 3),
        },
    )


def _breakout_retest(
    df: pd.DataFrame,
    direction: str,
    atr: float,
    candle: Any,
    policy: StrictSetupPolicy,
) -> Optional[SetupDetection]:
    work = df.tail(60).copy()
    if len(work) < 35:
        return None
    ohlcv = work[["open", "high", "low", "close", "volume"]].astype(float)
    last = ohlcv.iloc[-1]
    found: Optional[Dict[str, float]] = None
    # Break must precede the current retest; the last candle cannot satisfy
    # both events and manufacture a breakout-retest label.
    for idx in range(len(ohlcv) - 8, len(ohlcv) - 1):
        prior = ohlcv.iloc[max(0, idx - 20):idx]
        if len(prior) < 15:
            continue
        volume_base = max(float(prior["volume"].median()), 1e-12)
        volume_ratio = float(ohlcv.iloc[idx]["volume"]) / volume_base
        if direction == "long":
            level = float(prior["high"].max())
            displacement = (float(ohlcv.iloc[idx]["close"]) - level) / atr
            retest_distance = abs(float(last["low"]) - level) / atr
            reclaim = (float(last["close"]) - level) / atr
        else:
            level = float(prior["low"].min())
            displacement = (level - float(ohlcv.iloc[idx]["close"])) / atr
            retest_distance = abs(float(last["high"]) - level) / atr
            reclaim = (level - float(last["close"])) / atr
        if (
            displacement >= policy.breakout_min_displacement_atr
            and volume_ratio >= policy.breakout_min_volume_ratio
            and retest_distance <= policy.breakout_max_retest_distance_atr
            and reclaim >= policy.breakout_min_reclaim_atr
        ):
            found = {
                "breakout_level": level,
                "displacement_atr": displacement,
                "breakout_volume_ratio": volume_ratio,
                "retest_distance_atr": retest_distance,
                "reclaim_atr": reclaim,
            }
    if found is None or candle.score < policy.breakout_min_candle_score or candle.adverse_rejection:
        return None
    found["candle_score"] = candle.score
    return SetupDetection(
        setup_type="breakout_retest",
        passed=True,
        reasons=["Closed-candle displacement preceded a separate confirmed retest"],
        evidence={key: round(value, 4) for key, value in found.items()},
    )


def _ob_fvg_retest(
    df: pd.DataFrame,
    structure: StructureReport,
    direction: str,
    atr: float,
    candle: Any,
    policy: StrictSetupPolicy,
) -> Optional[SetupDetection]:
    wanted = "bullish" if direction == "long" else "bearish"
    last = df.iloc[-1]
    candidates: List[StructureLevel] = []
    for level in structure.levels:
        if level.kind not in {"order_block", "fvg"} or level.side != wanted:
            continue
        if level.invalidated or level.fully_mitigated or not level.relevant:
            continue
        if level.age_bars is not None and level.age_bars > policy.zone_max_age_bars:
            continue
        if level.touch_count > policy.zone_max_touches:
            continue
        if level.mitigation_fraction > policy.zone_max_mitigation:
            continue
        touched = (
            float(last["high"]) >= level.price_low - atr * policy.zone_touch_buffer_atr
            and float(last["low"]) <= level.price_high + atr * policy.zone_touch_buffer_atr
        )
        reclaimed = (
            float(last["close"]) >= level.mid
            if direction == "long"
            else float(last["close"]) <= level.mid
        )
        if touched and reclaimed:
            candidates.append(level)
    if not candidates or candle.score < policy.zone_min_candle_score or candle.adverse_rejection:
        return None
    level = max(candidates, key=lambda item: (item.confidence, -(item.age_bars or 0)))
    return SetupDetection(
        setup_type="ob_fvg_retest",
        passed=True,
        reasons=[f"Fresh {level.kind.replace('_', ' ')} was retested and directionally reclaimed"],
        evidence={
            "zone_kind": level.kind,
            "zone_low": level.price_low,
            "zone_high": level.price_high,
            "zone_age_bars": level.age_bars,
            "zone_touches": level.touch_count,
            "zone_mitigation": round(level.mitigation_fraction, 3),
            "candle_score": round(candle.score, 3),
        },
    )


def _confirmed_counter_trend_reversal(
    indicators: IndicatorSuite,
    structure: StructureReport,
    direction: str,
    candle: Any,
    policy: StrictSetupPolicy,
) -> Optional[SetupDetection]:
    wanted = "bullish" if direction == "long" else "bearish"
    opposing_trend = structure.trend == ("down" if direction == "long" else "up")
    wick = candle.lower_wick_ratio if direction == "long" else candle.upper_wick_ratio
    rsi = safe_float((indicators.summary or {}).get("rsi"), 50.0)
    exhausted = rsi <= 42.0 if direction == "long" else rsi >= 58.0
    divergence = any(
        str(getattr(item, "kind", "")).lower() == wanted
        for item in (indicators.divergences or [])
    )
    if not (
        opposing_trend
        and structure.last_choch == wanted
        and wick >= policy.reversal_min_rejection_wick
        and candle.score >= policy.reversal_min_candle_score
        and candle.volume_ratio >= policy.reversal_min_volume_ratio
        and (exhausted or divergence)
        and not candle.absorption
    ):
        return None
    return SetupDetection(
        setup_type="reversal",
        passed=True,
        reasons=["Counter-trend CHoCH, exhaustion, rejection wick, and volume confirmed"],
        evidence={
            "choch": structure.last_choch,
            "rsi": round(rsi, 2),
            "directional_divergence": divergence,
            "rejection_wick": round(wick, 3),
            "volume_ratio": round(candle.volume_ratio, 3),
            "candle_score": round(candle.score, 3),
        },
    )
