"""Entry-quality analysis from closed candles, structure, and volume flow.

The live bias answers *which side*.  This module answers the more important
execution question: *where can that side be entered without chasing price?*
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.indicators import IndicatorSuite
from src.analysis.market_structure import StructureLevel, StructureReport
from src.utils.helpers import clamp, safe_float


@dataclass
class CandleContext:
    score: float = 0.0
    order_flow_score: float = 0.0
    body_ratio: float = 0.0
    upper_wick_ratio: float = 0.0
    lower_wick_ratio: float = 0.0
    close_location: float = 0.5
    volume_ratio: float = 1.0
    cvd_proxy: float = 0.0
    range_atr: float = 0.0
    noise_atr: float = 1.0
    adverse_rejection: bool = False
    absorption: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionProfile:
    direction: str
    status: str = "blocked"  # ready | wait_retest | avoid_chase | blocked
    score: float = 0.0
    entry_low: float = 0.0
    entry_high: float = 0.0
    stop_loss: float = 0.0
    targets: List[float] = field(default_factory=list)
    entry_reason: str = ""
    invalidation_reason: str = ""
    chase_distance_atr: float = 0.0
    stop_distance_atr: float = 0.0
    immediate_sl_risk: float = 100.0
    order_flow_score: float = 0.0
    candle_score: float = 0.0
    anchor_sources: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    candle: CandleContext = field(default_factory=CandleContext)

    @property
    def entry_mid(self) -> float:
        return (self.entry_low + self.entry_high) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["entry_mid"] = self.entry_mid
        return data


def analyze_candles(df: pd.DataFrame, atr: float, direction: str) -> CandleContext:
    """Summarize bodies, wicks, rejection, and signed-volume order flow."""
    if df is None or len(df) < 20:
        return CandleContext(notes=["Insufficient closed candles for execution analysis"])

    work = df.tail(80)
    o = work["open"].astype(float)
    h = work["high"].astype(float)
    l = work["low"].astype(float)
    c = work["close"].astype(float)
    v = work["volume"].astype(float).clip(lower=0)
    ranges = (h - l).replace(0, np.nan)
    bodies = (c - o).abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    close_location = ((c - l) / ranges).clip(0, 1).fillna(0.5)

    last_range = max(safe_float(ranges.iloc[-1]), atr * 0.05)
    body_ratio = safe_float(bodies.iloc[-1] / last_range)
    upper_ratio = safe_float(upper.iloc[-1] / last_range)
    lower_ratio = safe_float(lower.iloc[-1] / last_range)
    last_location = safe_float(close_location.iloc[-1], 0.5)
    vol_base = safe_float(v.tail(30).median(), 1.0)
    volume_ratio = safe_float(v.iloc[-1] / max(vol_base, 1e-12), 1.0)

    # Close-location value assigns volume to buyers/sellers without requiring
    # unavailable trade-by-trade delta data.
    signed_volume = v * (2.0 * close_location - 1.0)
    recent_cvd = safe_float(signed_volume.tail(8).sum())
    recent_volume = max(safe_float(v.tail(8).sum()), 1e-12)
    cvd_proxy = float(clamp(recent_cvd / recent_volume, -1, 1))
    direction_sign = 1.0 if direction == "long" else -1.0
    recent_return = safe_float(c.iloc[-1] / c.iloc[-4] - 1.0) if len(c) >= 4 else 0.0
    impulse = float(clamp(recent_return / max(atr / max(c.iloc[-1], 1e-12), 1e-6), -1, 1))
    order_flow = float(clamp(0.72 * cvd_proxy + 0.28 * impulse, -1, 1))

    candle_score = direction_sign * (
        0.45 * (2.0 * last_location - 1.0)
        + 0.35 * np.sign(c.iloc[-1] - o.iloc[-1]) * body_ratio
        + 0.20 * order_flow
    )
    candle_score = float(clamp(candle_score, -1, 1))
    adverse_rejection = (
        direction == "long" and upper_ratio >= 0.45 and last_location < 0.55
    ) or (
        direction == "short" and lower_ratio >= 0.45 and last_location > 0.45
    )
    absorption = bool(volume_ratio >= 1.5 and body_ratio <= 0.28)
    noise_atr = safe_float(ranges.tail(30).quantile(0.70) / max(atr, 1e-12), 1.0)

    notes = [
        f"Last closed candle body {body_ratio:.0%}; upper/lower wick {upper_ratio:.0%}/{lower_ratio:.0%}",
        f"Close location {last_location:.2f}; volume {volume_ratio:.2f}x median",
        f"Signed-volume flow {order_flow:+.2f}",
    ]
    if adverse_rejection:
        notes.append("Adverse rejection wick: do not enter at market")
    if absorption:
        notes.append("High-volume small body suggests absorption/indecision")

    return CandleContext(
        score=candle_score,
        order_flow_score=order_flow,
        body_ratio=body_ratio,
        upper_wick_ratio=upper_ratio,
        lower_wick_ratio=lower_ratio,
        close_location=last_location,
        volume_ratio=volume_ratio,
        cvd_proxy=cvd_proxy,
        range_atr=last_range / max(atr, 1e-12),
        noise_atr=noise_atr,
        adverse_rejection=adverse_rejection,
        absorption=absorption,
        notes=notes,
    )


def build_execution_profile(
    df: pd.DataFrame,
    indicators: IndicatorSuite,
    structure: StructureReport,
    *,
    direction: str,
    price: float,
    atr: float,
) -> ExecutionProfile:
    """Create a structure-clustered limit/retest entry and realistic targets."""
    if direction not in ("long", "short") or not price or not atr:
        return ExecutionProfile(direction=direction, status="blocked")

    candle = analyze_candles(df, atr, direction)
    anchors = _entry_anchors(indicators, structure, direction, price, atr)
    anchor, anchor_sources, anchor_bounds, cluster_strength = _select_anchor(
        anchors, direction, price, atr
    )
    if anchor is None:
        anchor = price - atr * 0.35 if direction == "long" else price + atr * 0.35
        anchor_bounds = (anchor - atr * 0.08, anchor + atr * 0.08)
        anchor_sources = ["ATR pullback fallback"]
        cluster_strength = 0.0

    zone_half = float(clamp(atr * 0.12, price * 0.0005, atr * 0.22))
    entry_low = min(anchor_bounds[0], anchor - zone_half)
    entry_high = max(anchor_bounds[1], anchor + zone_half)
    # Keep the entry compact even when an old OB candle is unusually wide.
    if entry_high - entry_low > atr * 0.45:
        entry_low, entry_high = anchor - atr * 0.20, anchor + atr * 0.20
    if direction == "long" and entry_high > price + atr * 0.05:
        entry_high = price + atr * 0.05
    if direction == "short" and entry_low < price - atr * 0.05:
        entry_low = price - atr * 0.05
    entry_mid = (entry_low + entry_high) / 2.0

    chase_distance = (
        max(0.0, price - entry_high) / atr
        if direction == "long"
        else max(0.0, entry_low - price) / atr
    )
    relevant_edge = anchor_bounds[0] if direction == "long" else anchor_bounds[1]
    noise_buffer = atr * float(clamp(candle.noise_atr * 0.16, 0.12, 0.30))
    min_stop_distance = atr * float(clamp(0.85 + candle.noise_atr * 0.12, 0.90, 1.25))
    if direction == "long":
        stop = min(relevant_edge - noise_buffer, entry_mid - min_stop_distance)
    else:
        stop = max(relevant_edge + noise_buffer, entry_mid + min_stop_distance)
    stop_distance = abs(entry_mid - stop)

    targets = _structure_targets(
        structure,
        df,
        direction=direction,
        entry=entry_mid,
        stop=stop,
        atr=atr,
        current_price=price,
    )
    target_rr2 = (
        abs(targets[1] - entry_mid) / max(stop_distance, 1e-12)
        if len(targets) > 1
        else 0.0
    )

    direction_sign = 1.0 if direction == "long" else -1.0
    flow_alignment = candle.order_flow_score * direction_sign
    score = 52.0
    score += min(18.0, cluster_strength * 6.0)
    score += candle.score * 11.0
    score += flow_alignment * 10.0
    score += 4.0 if candle.volume_ratio >= 1.05 else -4.0
    score -= 18.0 if candle.adverse_rejection else 0.0
    score -= 10.0 if candle.absorption else 0.0
    score -= max(0.0, chase_distance - 0.5) * 15.0
    score += 6.0 if target_rr2 >= 1.25 else -18.0
    score = float(clamp(score, 0, 100))

    inside = entry_low <= price <= entry_high
    if chase_distance > 1.35 or score < 55:
        status = "avoid_chase"
    elif inside and not candle.adverse_rejection and not candle.absorption:
        status = "ready"
    else:
        status = "wait_retest"

    reasons = [
        f"Entry clustered at {', '.join(anchor_sources[:4])}",
        f"Order-flow approximation {candle.order_flow_score:+.2f}",
        f"TP2 planned at {target_rr2:.2f}R",
    ]
    risks: List[str] = []
    if candle.adverse_rejection:
        risks.append("Latest closed candle rejects the trade direction")
    if candle.absorption:
        risks.append("Absorption candle: wait for a decisive close")
    if chase_distance > 0.45:
        risks.append(f"Price is {chase_distance:.2f} ATR from the entry; use a limit/retest")
    if target_rr2 < 1.25:
        risks.append("Structure does not offer at least 1.25R to TP2")
    immediate_risk = float(
        clamp(
            100.0
            - score
            + (18.0 if candle.adverse_rejection else 0.0)
            + max(0.0, 0.95 - stop_distance / atr) * 22.0,
            0,
            100,
        )
    )
    entry_reason = (
        "Price is inside the validated zone; enter only after candle confirmation."
        if status == "ready"
        else (
            "Place no market order; wait for price to retest this demand/supply cluster."
            if status == "wait_retest"
            else "Setup is extended or poorly confirmed; skip rather than chase."
        )
    )
    invalidation = (
        f"Demand fails below {stop:.8g}" if direction == "long"
        else f"Supply fails above {stop:.8g}"
    )
    return ExecutionProfile(
        direction=direction,
        status=status,
        score=score,
        entry_low=float(min(entry_low, entry_high)),
        entry_high=float(max(entry_low, entry_high)),
        stop_loss=float(stop),
        targets=[float(x) for x in targets],
        entry_reason=entry_reason,
        invalidation_reason=invalidation,
        chase_distance_atr=float(chase_distance),
        stop_distance_atr=float(stop_distance / atr),
        immediate_sl_risk=immediate_risk,
        order_flow_score=candle.order_flow_score,
        candle_score=candle.score,
        anchor_sources=anchor_sources,
        reasons=reasons,
        risks=risks,
        candle=candle,
    )


def _entry_anchors(
    indicators: IndicatorSuite,
    structure: StructureReport,
    direction: str,
    price: float,
    atr: float,
) -> List[Tuple[float, float, float, str, float]]:
    """Return (mid, low, high, source, confidence) anchors on the entry side."""
    anchors: List[Tuple[float, float, float, str, float]] = []
    wanted = "bullish" if direction == "long" else "bearish"
    for level in structure.levels:
        if level.side != wanted or level.kind not in (
            "order_block",
            "fvg",
            "liquidity",
            "support",
            "resistance",
        ):
            continue
        mid = level.mid
        on_side = mid <= price + atr * 0.10 if direction == "long" else mid >= price - atr * 0.10
        if on_side and abs(mid - price) <= atr * 2.2:
            anchors.append(
                (mid, level.price_low, level.price_high, level.kind, level.confidence / 100.0)
            )

    summary = indicators.summary or {}
    for key, label, confidence in (
        ("ema_fast", "EMA fast", 0.75),
        ("ema_mid", "EMA 21", 0.85),
        ("vwap", "session VWAP", 0.90),
    ):
        value = summary.get(key)
        if value is None and indicators.df is not None and key in indicators.df.columns:
            value = indicators.df[key].iloc[-1]
        val = safe_float(value)
        if not val:
            continue
        on_side = val <= price + atr * 0.08 if direction == "long" else val >= price - atr * 0.08
        if on_side and abs(val - price) <= atr * 1.8:
            anchors.append((val, val, val, label, confidence))

    vp = (
        (structure.volume_profile_val, "volume VAL", 0.88),
        (structure.volume_profile_poc, "volume POC", 0.95),
        (structure.volume_profile_vah, "volume VAH", 0.88),
    )
    for value, label, confidence in vp:
        if value is None:
            continue
        on_side = value <= price if direction == "long" else value >= price
        if on_side and abs(value - price) <= atr * 2.0:
            anchors.append((value, value, value, label, confidence))
    return anchors


def _select_anchor(
    anchors: List[Tuple[float, float, float, str, float]],
    direction: str,
    price: float,
    atr: float,
) -> Tuple[Optional[float], List[str], Tuple[float, float], float]:
    if not anchors:
        return None, [], (0.0, 0.0), 0.0
    best: Optional[Tuple[float, float, float, str, float]] = None
    best_score = -1e9
    best_cluster: List[Tuple[float, float, float, str, float]] = []
    for candidate in anchors:
        mid = candidate[0]
        cluster = [a for a in anchors if abs(a[0] - mid) <= atr * 0.28]
        confidence = sum(a[4] for a in cluster)
        distance = abs(price - mid) / atr
        # Slight preference for a meaningful pullback instead of an entry at the
        # current wick, while rejecting deep/outdated zones.
        ideal_distance_penalty = abs(distance - 0.45) * 0.55
        score = confidence + len(cluster) * 0.45 - ideal_distance_penalty
        if score > best_score:
            best, best_score, best_cluster = candidate, score, cluster
    assert best is not None
    weights = np.array([max(a[4], 0.1) for a in best_cluster], dtype=float)
    mids = np.array([a[0] for a in best_cluster], dtype=float)
    anchor = float(np.average(mids, weights=weights))
    lows = [a[1] for a in best_cluster]
    highs = [a[2] for a in best_cluster]
    sources = list(dict.fromkeys(a[3] for a in best_cluster))
    return anchor, sources, (float(min(lows)), float(max(highs))), float(best_score)


def _structure_targets(
    structure: StructureReport,
    df: pd.DataFrame,
    *,
    direction: str,
    entry: float,
    stop: float,
    atr: float,
    current_price: float,
) -> List[float]:
    sign = 1.0 if direction == "long" else -1.0
    risk = max(abs(entry - stop), atr * 0.5)
    candidates: List[float] = []
    for level in structure.levels:
        target_side = level.side == ("bearish" if direction == "long" else "bullish")
        beyond = level.mid > entry if direction == "long" else level.mid < entry
        if target_side and beyond:
            candidates.append(level.mid)
    swing_targets = structure.swing_highs if direction == "long" else structure.swing_lows
    target_values = [
        structure.volume_profile_poc,
        structure.volume_profile_vah if direction == "long" else structure.volume_profile_val,
        *swing_targets,
    ]
    for value in target_values:
        if value is not None and ((value > entry) if direction == "long" else (value < entry)):
            candidates.append(float(value))
    if ((current_price > entry) if direction == "long" else (current_price < entry)):
        candidates.append(float(current_price))
    if df is not None and not df.empty:
        recent = (
            float(df["high"].tail(80).max())
            if direction == "long"
            else float(df["low"].tail(80).min())
        )
        if ((recent > entry) if direction == "long" else (recent < entry)):
            candidates.append(recent)

    candidates = sorted(set(candidates), reverse=direction == "short")
    targets: List[float] = []
    rr_floors = (0.80, 1.30, 2.00, 2.80)
    for rr in rr_floors:
        floor_target = entry + sign * risk * rr
        if direction == "long":
            eligible = [x for x in candidates if x >= floor_target]
            selected = min(eligible) if eligible else floor_target
        else:
            eligible = [x for x in candidates if x <= floor_target]
            selected = max(eligible) if eligible else floor_target
        if targets:
            if direction == "long":
                selected = max(selected, targets[-1] + atr * 0.20)
            else:
                selected = min(selected, targets[-1] - atr * 0.20)
        targets.append(float(selected))
    return targets
