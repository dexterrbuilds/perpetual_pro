"""Strict closed-candle setup recognition for production day-trade playbooks.

This module classifies *how* an already-directional thesis may be traded.  It
does not create direction, alter Technical/Overall/Execution Quality, or bypass
any publication gate.  Every detected setup still enters the normal execution,
risk, qualification, and pre-delivery revalidation pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.analysis.indicators import IndicatorSuite
from src.analysis.market_structure import StructureLevel, StructureReport
from src.utils.helpers import clamp, safe_float


SETUP_POLICY_VERSION = "strict_intraday_setups_v2"
UTC = timezone.utc


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

    opening_range_minutes: int = 30
    opening_range_window_minutes: int = 120
    opening_range_min_width_atr: float = 0.35
    opening_range_max_width_atr: float = 2.50
    opening_range_min_breakout_atr: float = 0.06
    opening_range_min_volume_ratio: float = 1.20
    opening_range_max_retest_distance_atr: float = 0.15
    opening_range_min_reclaim_atr: float = 0.02
    opening_range_min_bias_score: float = 0.15
    opening_range_min_momentum: float = 0.10

    session_rejection_min_wick: float = 0.35
    session_rejection_min_volume_ratio: float = 1.10
    session_rejection_min_breach_atr: float = 0.02
    session_rejection_min_reclaim_atr: float = 0.02
    session_rejection_min_candle_score: float = 0.15
    session_rejection_min_bias_score: float = 0.12


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

    opening_range = _opening_range_breakout(
        df, indicators, structure, direction, atr, candle, policy
    )
    if opening_range is not None:
        opening_range.label = f"{side} Opening Range Breakout"
        return opening_range

    breakout = _breakout_retest(df, direction, atr, candle, policy)
    if breakout is not None:
        breakout.label = f"{side} Breakout + Retest"
        return breakout

    zone = _ob_fvg_retest(df, structure, direction, atr, candle, policy)
    if zone is not None:
        zone.label = f"{side} Order Block / FVG Retest"
        return zone

    session_rejection = _session_high_low_rejection(
        df, indicators, structure, direction, atr, candle, policy
    )
    if session_rejection is not None:
        session_rejection.label = f"{side} Session High/Low Rejection"
        return session_rejection

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


@dataclass(frozen=True)
class _SessionInterval:
    name: str
    start: datetime
    end: datetime


def _closed_candle_timing(df: pd.DataFrame) -> Optional[tuple[datetime, timedelta]]:
    if df is None or len(df) < 3 or not isinstance(df.index, pd.DatetimeIndex):
        return None
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize(UTC)
    else:
        index = index.tz_convert(UTC)
    differences = index.to_series().diff().dropna()
    if differences.empty:
        return None
    candle_delta = differences.median().to_pytimedelta()
    if candle_delta <= timedelta(0) or candle_delta > timedelta(minutes=30):
        return None
    last_open = index[-1].to_pydatetime().astimezone(UTC)
    return last_open + candle_delta, candle_delta


def _local_datetime(day: date, zone: str, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(
        day,
        time(hour=hour, minute=minute),
        tzinfo=ZoneInfo(zone),
    ).astimezone(UTC)


def _opening_session(
    closed_at: datetime,
    policy: StrictSetupPolicy,
) -> Optional[_SessionInterval]:
    """Return only an active London or New York ORB window."""
    definitions = (
        ("London", "Europe/London", 8, 0),
        ("New York", "America/New_York", 9, 30),
    )
    for name, zone, hour, minute in definitions:
        local_day = closed_at.astimezone(ZoneInfo(zone)).date()
        opened = _local_datetime(local_day, zone, hour, minute)
        range_end = opened + timedelta(minutes=policy.opening_range_minutes)
        activation_end = opened + timedelta(
            minutes=policy.opening_range_window_minutes
        )
        if range_end <= closed_at <= activation_end:
            return _SessionInterval(name=name, start=opened, end=range_end)
    return None


def _opening_range_breakout(
    df: pd.DataFrame,
    indicators: IndicatorSuite,
    structure: StructureReport,
    direction: str,
    atr: float,
    candle: Any,
    policy: StrictSetupPolicy,
) -> Optional[SetupDetection]:
    timing = _closed_candle_timing(df)
    if timing is None:
        return None
    closed_at, candle_delta = timing
    session = _opening_session(closed_at, policy)
    if session is None:
        return None
    index = pd.DatetimeIndex(df.index)
    index = index.tz_localize(UTC) if index.tz is None else index.tz_convert(UTC)
    work = df.copy()
    work.index = index
    range_candles = work[
        (work.index >= session.start)
        & ((work.index + candle_delta) <= session.end)
    ]
    minimum_range_candles = max(
        1,
        int(policy.opening_range_minutes / max(candle_delta.total_seconds() / 60, 1)),
    )
    if len(range_candles) < minimum_range_candles:
        return None
    range_high = float(range_candles["high"].max())
    range_low = float(range_candles["low"].min())
    range_width_atr = (range_high - range_low) / atr
    if not (
        policy.opening_range_min_width_atr
        <= range_width_atr
        <= policy.opening_range_max_width_atr
    ):
        return None

    post_range = work[(work.index + candle_delta) > session.end]
    if len(post_range) < 2:
        return None
    last = post_range.iloc[-1]
    breakout_rows = post_range.iloc[:-1]
    breakout: Optional[pd.Series] = None
    breakout_time: Optional[pd.Timestamp] = None
    breakout_volume_ratio = 0.0
    level = range_high if direction == "long" else range_low
    for timestamp, candidate in breakout_rows.tail(6).iterrows():
        history = work[work.index < timestamp].tail(30)
        volume_base = max(float(history["volume"].median()), 1e-12)
        volume_ratio = float(candidate["volume"]) / volume_base
        displacement = (
            (float(candidate["close"]) - level) / atr
            if direction == "long"
            else (level - float(candidate["close"])) / atr
        )
        if (
            displacement >= policy.opening_range_min_breakout_atr
            and volume_ratio >= policy.opening_range_min_volume_ratio
        ):
            breakout = candidate
            breakout_time = timestamp
            breakout_volume_ratio = volume_ratio
    if breakout is None or breakout_time is None:
        return None

    if direction == "long":
        retest_distance = abs(float(last["low"]) - level) / atr
        reclaim = (float(last["close"]) - level) / atr
    else:
        retest_distance = abs(float(last["high"]) - level) / atr
        reclaim = (level - float(last["close"])) / atr
    sign = 1.0 if direction == "long" else -1.0
    summary = indicators.summary or {}
    directional_bias = sign * (
        0.55 * safe_float(summary.get("trend_score"))
        + 0.45 * safe_float(summary.get("momentum_score"))
    )
    momentum = sign * safe_float(summary.get("momentum_score"))
    wanted = "bullish" if direction == "long" else "bearish"
    wanted_trend = "up" if direction == "long" else "down"
    structure_supports = bool(
        structure.last_bos == wanted
        or structure.last_choch == wanted
        or structure.trend == wanted_trend
    )
    if not (
        retest_distance <= policy.opening_range_max_retest_distance_atr
        and reclaim >= policy.opening_range_min_reclaim_atr
        and candle.score >= policy.breakout_min_candle_score
        and not candle.adverse_rejection
        and directional_bias >= policy.opening_range_min_bias_score
        and momentum >= policy.opening_range_min_momentum
        and structure_supports
    ):
        return None
    return SetupDetection(
        setup_type="opening_range_breakout",
        passed=True,
        reasons=[
            f"{session.name} 30-minute range broke on volume and passed a separate retest"
        ],
        evidence={
            "session": session.name,
            "session_open": session.start.isoformat(),
            "opening_range_end": session.end.isoformat(),
            "opening_range_high": range_high,
            "opening_range_low": range_low,
            "opening_range_width_atr": round(range_width_atr, 3),
            "breakout_level": level,
            "breakout_time": breakout_time.isoformat(),
            "breakout_volume_ratio": round(breakout_volume_ratio, 3),
            "retest_distance_atr": round(retest_distance, 3),
            "reclaim_atr": round(reclaim, 3),
            "directional_bias": round(directional_bias, 3),
            "directional_momentum": round(momentum, 3),
            "invalidation_level": (
                float(last["low"]) if direction == "long" else float(last["high"])
            ),
        },
    )


def _major_session_intervals(reference: datetime) -> List[_SessionInterval]:
    """DST-aware completed sessions used only for prior-session levels."""
    intervals: List[_SessionInterval] = []
    for offset in range(-3, 2):
        day = (reference + timedelta(days=offset)).date()
        london_open = _local_datetime(day, "Europe/London", 8)
        new_york_open = _local_datetime(day, "America/New_York", 9, 30)
        intervals.extend(
            [
                _SessionInterval(
                    "Asia",
                    datetime.combine(day, time(0), tzinfo=UTC),
                    london_open,
                ),
                _SessionInterval("London", london_open, new_york_open),
                _SessionInterval(
                    "New York",
                    new_york_open,
                    _local_datetime(day, "America/New_York", 16),
                ),
            ]
        )
    return sorted(intervals, key=lambda item: item.end)


def _previous_session_range(
    df: pd.DataFrame,
    closed_at: datetime,
    candle_delta: timedelta,
) -> Optional[tuple[_SessionInterval, float, float]]:
    index = pd.DatetimeIndex(df.index)
    index = index.tz_localize(UTC) if index.tz is None else index.tz_convert(UTC)
    work = df.copy()
    work.index = index
    for session in reversed(_major_session_intervals(closed_at)):
        if session.end > closed_at:
            continue
        candles = work[
            (work.index >= session.start)
            & ((work.index + candle_delta) <= session.end)
        ]
        if len(candles) >= 4:
            return (
                session,
                float(candles["high"].max()),
                float(candles["low"].min()),
            )
    return None


def _session_high_low_rejection(
    df: pd.DataFrame,
    indicators: IndicatorSuite,
    structure: StructureReport,
    direction: str,
    atr: float,
    candle: Any,
    policy: StrictSetupPolicy,
) -> Optional[SetupDetection]:
    timing = _closed_candle_timing(df)
    if timing is None:
        return None
    closed_at, candle_delta = timing
    previous = _previous_session_range(df, closed_at, candle_delta)
    if previous is None:
        return None
    session, previous_high, previous_low = previous
    last = df.iloc[-1]
    if direction == "long":
        rejected_level = previous_low
        opposite_level = previous_high
        breach = (rejected_level - float(last["low"])) / atr
        reclaim = (float(last["close"]) - rejected_level) / atr
        wick = candle.lower_wick_ratio
        invalidation = float(last["low"])
    else:
        rejected_level = previous_high
        opposite_level = previous_low
        breach = (float(last["high"]) - rejected_level) / atr
        reclaim = (rejected_level - float(last["close"])) / atr
        wick = candle.upper_wick_ratio
        invalidation = float(last["high"])
    sign = 1.0 if direction == "long" else -1.0
    summary = indicators.summary or {}
    directional_bias = sign * (
        0.55 * safe_float(summary.get("trend_score"))
        + 0.45 * safe_float(summary.get("momentum_score"))
    )
    wanted = "bullish" if direction == "long" else "bearish"
    wanted_trend = "up" if direction == "long" else "down"
    structure_supports = bool(
        structure.last_bos == wanted
        or structure.last_choch == wanted
        or structure.trend == wanted_trend
    )
    if not (
        breach >= policy.session_rejection_min_breach_atr
        and reclaim >= policy.session_rejection_min_reclaim_atr
        and wick >= policy.session_rejection_min_wick
        and candle.volume_ratio >= policy.session_rejection_min_volume_ratio
        and candle.score >= policy.session_rejection_min_candle_score
        and directional_bias >= policy.session_rejection_min_bias_score
        and structure_supports
        and not candle.absorption
    ):
        return None
    return SetupDetection(
        setup_type="session_high_low_rejection",
        passed=True,
        reasons=[
            f"Previous {session.name} {'low' if direction == 'long' else 'high'} swept and rejected on above-average volume"
        ],
        evidence={
            "previous_session": session.name,
            "previous_session_start": session.start.isoformat(),
            "previous_session_end": session.end.isoformat(),
            "previous_session_high": previous_high,
            "previous_session_low": previous_low,
            "rejected_level": rejected_level,
            "opposite_session_level": opposite_level,
            "breach_atr": round(breach, 3),
            "reclaim_atr": round(reclaim, 3),
            "rejection_wick": round(wick, 3),
            "volume_ratio": round(candle.volume_ratio, 3),
            "directional_bias": round(directional_bias, 3),
            "invalidation_level": invalidation,
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
