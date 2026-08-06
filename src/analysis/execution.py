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
from src.analysis.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    ExecutionQualityPolicy,
    weighted_execution_quality,
)
from src.analysis.market_structure import StructureLevel, StructureReport
from src.analysis.setups import SETUP_POLICY_VERSION, detect_strict_setup
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
    status: str = "blocked"  # confirmation_pending | wait_retest | avoid_chase | blocked
    # ``score`` remains as a compatibility alias for execution_quality.
    score: float = 0.0
    execution_quality: float = 0.0
    legacy_execution_score: float = 0.0
    legacy_status: str = "blocked"
    legacy_targets: List[float] = field(default_factory=list)
    policy_version: str = DEFAULT_EXECUTION_POLICY.version
    setup_type: str = "retest_continuation"
    setup_policy_version: str = SETUP_POLICY_VERSION
    setup_evidence: Dict[str, Any] = field(default_factory=dict)
    entry_mode: str = "retest"
    components: Dict[str, float] = field(default_factory=dict)
    component_contributions: Dict[str, float] = field(default_factory=dict)
    entry_low: float = 0.0
    entry_high: float = 0.0
    stop_loss: float = 0.0
    targets: List[float] = field(default_factory=list)
    target_feasibility: List[float] = field(default_factory=list)
    gross_risk_reward: List[float] = field(default_factory=list)
    net_risk_reward: List[float] = field(default_factory=list)
    entry_reason: str = ""
    invalidation_reason: str = ""
    chase_distance_atr: float = 0.0
    stop_distance_atr: float = 0.0
    immediate_sl_risk: float = 100.0
    entry_zone_relation: str = "unknown"
    tp1_progress_pct: float = 0.0
    order_flow_score: float = 0.0
    spread_bps: Optional[float] = None
    orderbook_imbalance: Optional[float] = None
    orderbook_alignment: float = 0.0
    mark_index_basis_bps: Optional[float] = None
    market_quality_ok: bool = True
    ticker_age_seconds: Optional[float] = None
    orderbook_age_seconds: Optional[float] = None
    data_freshness_state: str = "unknown"
    entry_distance_pct: float = 0.0
    entry_distance_atr: float = 0.0
    entry_zone_width_atr: float = 0.0
    remaining_expiry_minutes: float = 0.0
    estimated_fee_bps: float = 0.0
    estimated_slippage_bps: float = 0.0
    estimated_funding_bps: float = 0.0
    estimated_total_cost_bps: float = 0.0
    estimated_impact_bps: Optional[float] = None
    depth_bands_bps: Dict[str, Any] = field(default_factory=dict)
    impact_reference_notional_usd: Optional[float] = None
    contract_size: Optional[float] = None
    tick_size: Optional[float] = None
    min_notional: Optional[float] = None
    amount_precision: Optional[float] = None
    structural_obstacle_distances_atr: List[float] = field(default_factory=list)
    hard_failures: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
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
        data["execution_score"] = self.execution_quality
        return data


@dataclass(frozen=True)
class EntryAnchor:
    mid: float
    low: float
    high: float
    source: str
    confidence: float
    age_bars: Optional[int] = None
    touch_count: int = 0
    mitigation_fraction: float = 0.0
    relevant: bool = True


STRUCTURAL_ANCHOR_SOURCES = {
    "order_block",
    "fvg",
    "liquidity",
    "support",
    "resistance",
    "confirmed breakout retest",
    "swept liquidity reclaim",
    "opening range breakout retest",
    "session high/low rejection",
}


def _is_structural_anchor(anchor: EntryAnchor) -> bool:
    return anchor.source in STRUCTURAL_ANCHOR_SOURCES


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
    snapshot: Optional[Any] = None,
    max_pre_entry_tp1_progress_pct: float = 70.0,
    setup_name: str = "",
    strategy_tags: Optional[List[str]] = None,
    remaining_expiry_minutes: float = 90.0,
    expected_hold_hours: float = 8.0,
    policy: ExecutionQualityPolicy = DEFAULT_EXECUTION_POLICY,
) -> ExecutionProfile:
    """Build an explainable, setup-specific deterministic execution plan."""
    if direction not in ("long", "short") or not price or not atr:
        return ExecutionProfile(direction=direction, status="blocked")

    candle = analyze_candles(df, atr, direction)
    setup_type = _canonical_setup_type(setup_name, strategy_tags or [], structure)
    strict_setup_tagged = "strict_setup_confirmed" in (strategy_tags or [])
    strict_detection = (
        detect_strict_setup(
            df,
            indicators,
            structure,
            direction=direction,
            price=safe_float(
                (indicators.summary or {}).get("close"),
                safe_float(df["close"].iloc[-1], price),
            ),
            atr=atr,
            candle=candle,
        )
        if strict_setup_tagged
        else None
    )
    strict_setup_confirmed = bool(
        not strict_setup_tagged
        or (
            strict_detection is not None
            and strict_detection.setup_type == setup_type
        )
    )
    wanted_structure = "bullish" if direction == "long" else "bearish"
    if (
        setup_type == "retest_continuation"
        and "breakout" in setup_name.lower()
        and structure.last_bos == wanted_structure
        and candle.volume_ratio >= 1.10
        and candle.score >= 0.15
    ):
        setup_type = "breakout_continuation"
    anchors = _entry_anchors(indicators, structure, direction, price, atr, policy)
    if setup_type == "trend_pullback":
        pullback_anchors = [
            anchor for anchor in anchors
            if anchor.source in {"EMA 21", "session VWAP", "EMA fast"}
        ]
        if pullback_anchors:
            anchors = pullback_anchors
    elif setup_type == "ob_fvg_retest":
        structural_anchors = [
            anchor for anchor in anchors
            if anchor.source in {"order_block", "fvg"}
        ]
        if structural_anchors:
            anchors = structural_anchors
    elif setup_type in {"opening_range_breakout", "session_high_low_rejection"}:
        # The strict detector contributes the relevant session level below.
        # Retain nearby chart structure, but do not let an indicator-only
        # anchor replace the session level.
        anchors = [
            anchor for anchor in anchors
            if _is_structural_anchor(anchor)
        ]
    if strict_detection is not None:
        evidence = strict_detection.evidence
        derived_level = safe_float(
            evidence.get("breakout_level")
            or evidence.get("rejected_level")
            or evidence.get("swept_level")
        )
        if derived_level > 0 and abs(derived_level - price) <= atr * 2.2:
            derived_anchor = EntryAnchor(
                derived_level,
                derived_level,
                derived_level,
                (
                    "opening range breakout retest"
                    if setup_type == "opening_range_breakout"
                    else "session high/low rejection"
                    if setup_type == "session_high_low_rejection"
                    else "confirmed breakout retest"
                    if setup_type == "breakout_retest"
                    else "swept liquidity reclaim"
                ),
                0.96 if setup_type in {
                    "opening_range_breakout", "session_high_low_rejection"
                } else 0.92,
            )
            nearby = [
                item for item in anchors
                if abs(item.mid - derived_level) <= atr * 0.28
            ]
            anchors = [derived_anchor, *nearby]
    anchor, anchor_sources, anchor_bounds, cluster_strength, anchor_health = _select_anchor(
        anchors, direction, price, atr, policy
    )
    if anchor is None:
        anchor = price - atr * 0.35 if direction == "long" else price + atr * 0.35
        anchor_bounds = (anchor - atr * 0.08, anchor + atr * 0.08)
        anchor_sources = ["ATR pullback fallback"]
        cluster_strength = 0.0
        anchor_health = 35.0

    # Breakout continuation may use a confirmed close near CMP; other setups
    # retain a structural zone. This does not record an immediate fill.
    bos_agrees = structure.last_bos == ("bullish" if direction == "long" else "bearish")
    breakout_cmp = bool(
        setup_type == "breakout_continuation"
        and bos_agrees
        and candle.volume_ratio >= 1.10
        and candle.score >= 0.15
    )
    if breakout_cmp:
        anchor = price
        anchor_bounds = (price - atr * 0.08, price + atr * 0.08)
        anchor_sources = ["confirmed breakout close"]
        anchor_health = max(anchor_health, 72.0)

    zone_half = float(clamp(atr * 0.12, price * 0.0005, atr * 0.22))
    entry_low = min(anchor_bounds[0], anchor - zone_half)
    entry_high = max(anchor_bounds[1], anchor + zone_half)
    if entry_high - entry_low > atr * policy.max_entry_zone_width_atr:
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
    structural_invalidation = _structural_invalidation_level(
        structure,
        direction=direction,
        entry=entry_mid,
        atr=atr,
    )
    detected_invalidation = safe_float(
        (strict_detection.evidence if strict_detection is not None else {}).get(
            "invalidation_level"
        )
    )
    if detected_invalidation > 0:
        structurally_valid = (
            detected_invalidation < entry_mid
            if direction == "long"
            else detected_invalidation > entry_mid
        )
        if structurally_valid:
            structural_invalidation = detected_invalidation
    if structural_invalidation is not None:
        relevant_edge = structural_invalidation
    noise_buffer = atr * float(clamp(candle.noise_atr * 0.16, 0.12, 0.30))
    min_stop_distance = atr * float(clamp(0.85 + candle.noise_atr * 0.12, 0.90, 1.25))
    if direction == "long":
        stop = min(relevant_edge - noise_buffer, entry_mid - min_stop_distance)
    else:
        stop = max(relevant_edge + noise_buffer, entry_mid + min_stop_distance)
    stop_distance = abs(entry_mid - stop)

    legacy_targets = _legacy_structure_targets(
        structure, df, direction=direction, entry=entry_mid, stop=stop,
        atr=atr, current_price=price,
    )

    session_target = safe_float(
        (strict_detection.evidence if strict_detection is not None else {}).get(
            "opposite_session_level"
        )
    )
    extra_structure_targets = [session_target] if session_target > 0 else []
    targets, target_quality, obstacles = _feasible_structure_targets(
        structure,
        df,
        direction=direction,
        entry=entry_mid,
        stop=stop,
        atr=atr,
        current_price=price,
        expected_hold_hours=expected_hold_hours,
        extra_structure_targets=extra_structure_targets,
        policy=policy,
    )

    direction_sign = 1.0 if direction == "long" else -1.0
    flow_alignment = candle.order_flow_score * direction_sign
    spread_bps = (
        safe_float(getattr(snapshot, "spread_bps", None))
        if getattr(snapshot, "spread_bps", None) is not None
        else None
    )
    book_imbalance = (
        safe_float(getattr(snapshot, "orderbook_imbalance", None))
        if getattr(snapshot, "orderbook_imbalance", None) is not None
        else None
    )
    book_alignment = (
        float(clamp(book_imbalance * direction_sign, -1, 1))
        if book_imbalance is not None
        else 0.0
    )
    basis_bps = (
        safe_float(getattr(snapshot, "mark_index_basis_bps", None))
        if getattr(snapshot, "mark_index_basis_bps", None) is not None
        else None
    )
    adverse_basis = bool(
        basis_bps is not None
        and (
            (direction == "long" and basis_bps > 18.0)
            or (direction == "short" and basis_bps < -18.0)
        )
    )
    ticker_age_seconds = getattr(snapshot, "ticker_age_seconds", None)
    orderbook_age_seconds = getattr(snapshot, "orderbook_age_seconds", None)
    sources_fresh = getattr(snapshot, "execution_data_fresh", None)
    raw_book = dict(getattr(snapshot, "raw", {}) or {}).get("orderbook_summary") or {}
    impact_bps = raw_book.get("estimated_impact_bps", getattr(snapshot, "estimated_impact_bps", None))
    bands = dict(getattr(snapshot, "orderbook_depth_bands", {}) or {})
    bid_depth_usd = safe_float(
        raw_book.get("bid_depth_usd_10bps", (bands.get("10") or {}).get("bid_usd")), 0.0
    )
    ask_depth_usd = safe_float(
        raw_book.get("ask_depth_usd_10bps", (bands.get("10") or {}).get("ask_usd")), 0.0
    )
    market_quality_ok = bool(
        (spread_bps is None or spread_bps <= policy.max_spread_bps)
        and sources_fresh is not False
    )

    entry_distance_pct = abs(price - entry_mid) / max(price, 1e-12) * 100.0
    zone_width_atr = (entry_high - entry_low) / max(atr, 1e-12)
    spread_price = price * (spread_bps or 0.0) / 10_000.0
    zone_spread_multiple = (entry_high - entry_low) / max(spread_price, price * 1e-9)
    recent_velocity = _directional_velocity(df, direction, atr)
    movement_toward_entry = -recent_velocity if (
        (direction == "long" and price > entry_high)
        or (direction == "short" and price < entry_low)
    ) else recent_velocity

    fee_bps = policy.default_taker_fee_bps_per_side * 2.0
    slippage_bps = policy.default_slippage_bps_per_side * 2.0
    if impact_bps is not None:
        slippage_bps += max(0.0, safe_float(impact_bps))
    funding_bps = abs(safe_float(getattr(snapshot, "funding_rate", None), 0.0)) * 10_000.0 * max(0.0, expected_hold_hours / 8.0)
    if funding_bps <= 0:
        funding_bps = policy.default_funding_bps_per_8h * max(0.0, expected_hold_hours / 8.0)
    total_cost_bps = fee_bps + slippage_bps + funding_bps + max(0.0, spread_bps or 0.0)
    cost_price = entry_mid * total_cost_bps / 10_000.0
    gross_rr = [abs(tp - entry_mid) / max(stop_distance, 1e-12) for tp in targets]
    net_rr = [
        max(0.0, abs(tp - entry_mid) - cost_price) / max(stop_distance + cost_price, 1e-12)
        for tp in targets
    ]

    legacy_rr = [
        abs(target - entry_mid) / max(stop_distance, 1e-12)
        for target in legacy_targets
    ]
    legacy_target_rr2 = legacy_rr[1] if len(legacy_rr) > 1 else (legacy_rr[0] if legacy_rr else 0.0)
    legacy_score = _legacy_execution_score(
        cluster_strength=cluster_strength,
        candle=candle,
        flow_alignment=flow_alignment,
        book_alignment=book_alignment,
        chase_distance=chase_distance,
        target_rr2=legacy_target_rr2,
        spread_bps=spread_bps,
        adverse_basis=adverse_basis,
    )

    inside = entry_low <= price <= entry_high
    favorable_beyond = (
        price > entry_high if direction == "long" else price < entry_low
    )
    entry_zone_relation = (
        "inside"
        if inside
        else ("favorable_beyond" if favorable_beyond else "adverse_side")
    )
    tp1_distance = abs(targets[0] - entry_mid) if targets else 0.0
    favorable_distance = (
        price - entry_mid if direction == "long" else entry_mid - price
    )
    tp1_progress_pct = float(
        clamp(
            favorable_distance / max(tp1_distance, 1e-12) * 100.0,
            0.0,
            200.0,
        )
    )
    target_already_traded = bool(
        targets
        and (
            price >= targets[0]
            if direction == "long"
            else price <= targets[0]
        )
    )
    late_before_entry = bool(
        favorable_beyond
        and tp1_progress_pct
        >= float(clamp(max_pre_entry_tp1_progress_pct, 0.0, 100.0))
    )
    legacy_tp1_distance = abs(legacy_targets[0] - entry_mid) if legacy_targets else 0.0
    legacy_progress = float(clamp(
        favorable_distance / max(legacy_tp1_distance, 1e-12) * 100.0, 0.0, 200.0
    ))
    legacy_target_traded = bool(
        legacy_targets and (
            price >= legacy_targets[0] if direction == "long" else price <= legacy_targets[0]
        )
    )
    legacy_late = bool(
        favorable_beyond
        and legacy_progress >= float(clamp(max_pre_entry_tp1_progress_pct, 0.0, 100.0))
    )
    if not market_quality_ok or (inside and (candle.adverse_rejection or candle.absorption)):
        legacy_status = "blocked"
    elif chase_distance > 1.35 or legacy_score < 55:
        legacy_status = "avoid_chase"
    elif legacy_target_traded or legacy_late:
        legacy_status = "avoid_chase"
    elif inside:
        legacy_status = "confirmation_pending"
    else:
        legacy_status = "wait_retest"

    # Component scores are deliberately independent: target mathematical R:R
    # is not an input to target feasibility, and technical direction is not
    # counted again in liquidity or freshness.
    entry_accessibility = _entry_accessibility_quality(
        chase_distance, entry_distance_pct, zone_width_atr,
        zone_spread_multiple, remaining_expiry_minutes, movement_toward_entry,
        tp1_progress_pct, policy,
    )
    entry_zone_quality = float(clamp(
        0.65 * anchor_health + 0.35 * min(100.0, cluster_strength * 24.0), 0, 100
    ))
    pre_entry_survival = _pre_entry_survival_quality(
        direction=direction, price=price, entry=entry_mid, stop=stop, atr=atr,
        candle=candle, structure=structure, remaining_minutes=remaining_expiry_minutes,
        anchor_health=anchor_health,
    )
    confirmation_quality = _confirmation_quality(
        candle, flow_alignment, book_alignment, setup_type, bos_agrees, structure
    )
    stop_quality = _stop_quality(
        stop_distance / atr, candle, spread_bps, slippage_bps, anchor_health, expected_hold_hours
    )
    target_feasibility_score = (
        float(clamp(0.60 * target_quality[0] + 0.40 * np.mean(target_quality), 0, 100))
        if target_quality else 0.0
    )
    liquidity_cost_quality, liquidity_uncertainties = _liquidity_cost_quality(
        spread_bps=spread_bps, book_alignment=book_alignment,
        bid_depth_usd=bid_depth_usd, ask_depth_usd=ask_depth_usd,
        impact_bps=impact_bps, total_cost_bps=total_cost_bps, policy=policy,
    )
    data_market_quality, data_uncertainties = _data_market_quality(
        ticker_age_seconds, orderbook_age_seconds, sources_fresh, policy
    )
    metadata_uncertainties = []
    if snapshot is not None and getattr(snapshot, "contract_size", None) is None:
        metadata_uncertainties.append("contract_size_unknown")
    if snapshot is not None and getattr(snapshot, "tick_size", None) is None:
        metadata_uncertainties.append("tick_size_unknown")
    components = {
        "entry_accessibility": entry_accessibility,
        "entry_zone_quality": entry_zone_quality,
        "pre_entry_survival": pre_entry_survival,
        "confirmation_quality": confirmation_quality,
        "stop_quality": stop_quality,
        "target_feasibility": target_feasibility_score,
        "liquidity_cost_quality": liquidity_cost_quality,
        "data_market_quality": data_market_quality,
    }
    score, contributions = weighted_execution_quality(components, setup_type, policy)
    hard_failures: List[str] = []
    if sources_fresh is False:
        hard_failures.append("stale_execution_data")
    if spread_bps is not None and spread_bps > policy.max_spread_bps:
        hard_failures.append("spread_above_hard_limit")
    if not targets:
        hard_failures.append("no_feasible_target")
    if obstacles and min(obstacles) < 0.35:
        hard_failures.append("tp1_blocked_by_nearby_structure")
    if targets and abs(targets[0] - entry_mid) < cost_price * policy.min_net_tp1_cost_multiple:
        hard_failures.append("tp1_reward_does_not_cover_cost_buffer")
    if stop_distance / atr > policy.max_stop_atr:
        hard_failures.append("stop_beyond_maximum_structure_distance")
    if setup_type == "range_mean_reversion" and structure.trend != "range":
        hard_failures.append("range_setup_during_trend_expansion")
    if setup_type == "reversal" and not _reversal_confirmed(direction, structure, candle):
        hard_failures.append("reversal_confirmation_insufficient")
    if not strict_setup_confirmed:
        hard_failures.append("setup_confirmation_insufficient")
    zone_position = (price - entry_low) / max(entry_high - entry_low, 1e-12)
    if inside and setup_type in ("cmp_confirmation", "breakout_continuation"):
        acceptable = zone_position <= 0.80 if direction == "long" else zone_position >= 0.20
        if not acceptable:
            hard_failures.append("cmp_position_excessively_extended_inside_zone")
        conflicting = structure.last_bos == ("bearish" if direction == "long" else "bullish")
        if conflicting:
            hard_failures.append("adverse_structure_change_before_confirmation")

    if hard_failures:
        status = "blocked"
    elif inside and (candle.adverse_rejection or candle.absorption):
        status = "blocked"
    elif chase_distance > policy.max_retest_distance_atr or score < 55:
        status = "avoid_chase"
    elif target_already_traded or late_before_entry:
        status = "avoid_chase"
    elif inside or breakout_cmp:
        status = "confirmation_pending"
    else:
        status = "wait_retest"

    reasons = [
        f"Entry clustered at {', '.join(anchor_sources[:4])}",
        f"Order-flow approximation {candle.order_flow_score:+.2f}",
        f"Execution Quality {score:.0f}/100 ({policy.version})",
    ]
    if strict_detection is not None:
        reasons.extend(strict_detection.reasons[:1])
    if spread_bps is not None:
        reasons.append(
            f"L2 spread {spread_bps:.2f} bps; book alignment {book_alignment:+.2f}"
        )
    risks: List[str] = []
    if candle.adverse_rejection:
        risks.append("Latest closed candle rejects the trade direction")
    if candle.absorption:
        risks.append("Absorption candle: wait for a decisive close")
    if chase_distance > 0.45:
        risks.append(f"Price is {chase_distance:.2f} ATR from the entry; use a limit/retest")
    if target_already_traded:
        risks.append("TP1 had already traded before the setup could be published")
    elif late_before_entry:
        risks.append(
            f"Price already completed {tp1_progress_pct:.0f}% of the Entry-to-TP1 move"
        )
    if gross_rr and max(gross_rr[:2]) < 1.25:
        risks.append("Feasible targets do not offer at least 1.25 gross R")
    if net_rr and net_rr[0] < gross_rr[0] * 0.85:
        risks.append("Trading costs materially reduce TP1 net R:R")
    if not market_quality_ok:
        risks.append(
            f"Execution market quality failed"
            + (f" ({spread_bps:.2f} bps spread)" if spread_bps is not None else " (stale source)")
        )
    if book_alignment < -0.45:
        risks.append("Order-book depth is strongly adverse to the setup")
    if adverse_basis:
        risks.append(f"Mark/index premium is crowded against entry ({basis_bps:+.1f} bps)")
    risks.extend(reason.replace("_", " ") for reason in hard_failures)
    immediate_risk = float(clamp(
        0.38 * (100.0 - stop_quality)
        + 0.27 * (100.0 - confirmation_quality)
        + 0.20 * (100.0 - pre_entry_survival)
        + 0.15 * (100.0 - liquidity_cost_quality), 0, 100
    ))
    entry_reason = (
        "Price is inside the validated zone; enter only after candle confirmation."
        if status == "confirmation_pending"
        else (
            "Place no market order; wait for price to retest this demand/supply cluster."
            if status == "wait_retest"
            else (
                "Price is inside the zone but confirmation is weak; wait for a fresh scan."
                if inside and status == "blocked"
                else "Setup is extended or poorly confirmed; skip rather than chase."
            )
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
        execution_quality=score,
        legacy_execution_score=legacy_score,
        legacy_status=legacy_status,
        legacy_targets=[float(x) for x in legacy_targets],
        policy_version=policy.version,
        setup_type=setup_type,
        setup_policy_version=SETUP_POLICY_VERSION,
        setup_evidence=(
            dict(strict_detection.evidence) if strict_detection is not None else {}
        ),
        entry_mode=("cmp_confirmation" if (inside or breakout_cmp) else "retest"),
        components=components,
        component_contributions=contributions,
        entry_low=float(min(entry_low, entry_high)),
        entry_high=float(max(entry_low, entry_high)),
        stop_loss=float(stop),
        targets=[float(x) for x in targets],
        target_feasibility=[float(x) for x in target_quality],
        gross_risk_reward=[float(x) for x in gross_rr],
        net_risk_reward=[float(x) for x in net_rr],
        entry_reason=entry_reason,
        invalidation_reason=invalidation,
        chase_distance_atr=float(chase_distance),
        stop_distance_atr=float(stop_distance / atr),
        immediate_sl_risk=immediate_risk,
        entry_zone_relation=entry_zone_relation,
        tp1_progress_pct=tp1_progress_pct,
        order_flow_score=candle.order_flow_score,
        spread_bps=spread_bps,
        orderbook_imbalance=book_imbalance,
        orderbook_alignment=book_alignment,
        mark_index_basis_bps=basis_bps,
        market_quality_ok=market_quality_ok,
        ticker_age_seconds=ticker_age_seconds,
        orderbook_age_seconds=orderbook_age_seconds,
        data_freshness_state=("stale" if sources_fresh is False else ("fresh" if sources_fresh is True else "unknown")),
        entry_distance_pct=float(entry_distance_pct),
        entry_distance_atr=float(abs(price - entry_mid) / max(atr, 1e-12)),
        entry_zone_width_atr=float(zone_width_atr),
        remaining_expiry_minutes=float(max(0.0, remaining_expiry_minutes)),
        estimated_fee_bps=float(fee_bps),
        estimated_slippage_bps=float(slippage_bps),
        estimated_funding_bps=float(funding_bps),
        estimated_total_cost_bps=float(total_cost_bps),
        estimated_impact_bps=(safe_float(impact_bps) if impact_bps is not None else None),
        depth_bands_bps=dict(raw_book.get("depth_bands_bps") or bands),
        impact_reference_notional_usd=getattr(snapshot, "impact_reference_notional_usd", None),
        contract_size=getattr(snapshot, "contract_size", None),
        tick_size=getattr(snapshot, "tick_size", None),
        min_notional=getattr(snapshot, "min_notional", None),
        amount_precision=getattr(snapshot, "amount_precision", None),
        structural_obstacle_distances_atr=[float(x) for x in obstacles],
        hard_failures=hard_failures,
        uncertainties=list(dict.fromkeys([
            *liquidity_uncertainties, *data_uncertainties, *metadata_uncertainties
        ])),
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
    policy: ExecutionQualityPolicy = DEFAULT_EXECUTION_POLICY,
) -> List[EntryAnchor]:
    """Return only directionally relevant, not-invalidated entry anchors."""
    anchors: List[EntryAnchor] = []
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
        if getattr(level, "invalidated", False) or getattr(level, "fully_mitigated", False):
            continue
        if (
            getattr(level, "age_bars", None) is not None
            and int(level.age_bars) > policy.expired_level_bars
        ):
            continue
        mid = level.mid
        on_side = mid <= price + atr * 0.10 if direction == "long" else mid >= price - atr * 0.10
        if on_side and abs(mid - price) <= atr * 2.2:
            anchors.append(EntryAnchor(
                mid, level.price_low, level.price_high, level.kind,
                level.confidence / 100.0,
                age_bars=getattr(level, "age_bars", None),
                touch_count=int(getattr(level, "touch_count", 0) or 0),
                mitigation_fraction=float(getattr(level, "mitigation_fraction", 0.0) or 0.0),
                relevant=bool(getattr(level, "relevant", True)),
            ))

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
            anchors.append(EntryAnchor(val, val, val, label, confidence))

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
            anchors.append(EntryAnchor(value, value, value, label, confidence))
    return anchors


def _select_anchor(
    anchors: List[EntryAnchor],
    direction: str,
    price: float,
    atr: float,
    policy: ExecutionQualityPolicy = DEFAULT_EXECUTION_POLICY,
) -> Tuple[Optional[float], List[str], Tuple[float, float], float, float]:
    if not anchors:
        return None, [], (0.0, 0.0), 0.0, 0.0
    structural = [anchor for anchor in anchors if _is_structural_anchor(anchor)]
    # Structure owns price geometry whenever a usable chart level exists.
    # Indicators may strengthen a nearby cluster but cannot replace its center.
    candidates = structural or anchors
    best: Optional[EntryAnchor] = None
    best_score = -1e9
    best_cluster: List[EntryAnchor] = []
    for candidate in candidates:
        mid = candidate.mid
        cluster = [a for a in anchors if abs(a.mid - mid) <= atr * 0.28]
        confidence = sum(
            a.confidence
            * (1.20 if _is_structural_anchor(a) else 0.80)
            * max(0.20, 1.0 - 0.80 * a.touch_count / max(1, policy.max_level_touches))
            * max(0.15, 1.0 - 0.70 * a.mitigation_fraction)
            for a in cluster
        )
        distance = abs(price - mid) / atr
        # Slight preference for a meaningful pullback instead of an entry at the
        # current wick, while rejecting deep/outdated zones.
        ideal_distance_penalty = abs(distance - 0.45) * 0.55
        score = confidence + len(cluster) * 0.45 - ideal_distance_penalty
        if score > best_score:
            best, best_score, best_cluster = candidate, score, cluster
    assert best is not None
    weights = np.array([max(a.confidence, 0.1) for a in best_cluster], dtype=float)
    mids = np.array([a.mid for a in best_cluster], dtype=float)
    anchor = float(np.average(mids, weights=weights))
    lows = [a.low for a in best_cluster]
    highs = [a.high for a in best_cluster]
    sources = list(dict.fromkeys(a.source for a in best_cluster))
    health_values = [
        100.0
        * max(0.20, 1.0 - 0.80 * a.touch_count / max(1, policy.max_level_touches))
        * max(0.15, 1.0 - 0.70 * a.mitigation_fraction)
        * (
            1.0
            if a.age_bars is None
            else max(
                0.25,
                1.0
                - max(0, a.age_bars - policy.stale_level_bars)
                / max(1, policy.expired_level_bars - policy.stale_level_bars),
            )
        )
        for a in best_cluster
    ]
    health = float(np.average(health_values, weights=weights)) if health_values else 0.0
    return anchor, sources, (float(min(lows)), float(max(highs))), float(best_score), health


def _structural_invalidation_level(
    structure: StructureReport,
    *,
    direction: str,
    entry: float,
    atr: float,
) -> Optional[float]:
    """Nearest valid thesis-invalidation edge; indicators never define it."""
    candidates: List[float] = []
    wanted = "bullish" if direction == "long" else "bearish"
    for level in structure.levels:
        if level.side != wanted or level.kind not in {
            "order_block", "fvg", "liquidity", "support", "resistance"
        }:
            continue
        if getattr(level, "invalidated", False) or getattr(
            level, "fully_mitigated", False
        ):
            continue
        edge = float(level.price_low if direction == "long" else level.price_high)
        valid_side = edge < entry if direction == "long" else edge > entry
        if valid_side and abs(edge - entry) <= atr * 3.0:
            candidates.append(edge)
    swings = structure.swing_lows if direction == "long" else structure.swing_highs
    for value in swings:
        edge = float(value)
        valid_side = edge < entry if direction == "long" else edge > entry
        if valid_side and abs(edge - entry) <= atr * 3.0:
            candidates.append(edge)
    if not candidates:
        return None
    return max(candidates) if direction == "long" else min(candidates)


def _legacy_structure_targets(
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


def _canonical_setup_type(
    setup_name: str,
    tags: List[str],
    structure: StructureReport,
) -> str:
    text = " ".join([setup_name.lower(), *(str(tag).lower() for tag in tags)])
    if "trend_pullback" in text or "trend continuation" in text:
        return "trend_pullback"
    if "opening_range_breakout" in text or "opening range breakout" in text:
        return "opening_range_breakout"
    if "session_high_low_rejection" in text or "session high/low rejection" in text:
        return "session_high_low_rejection"
    if "liquidity_sweep" in text or "liquidity sweep" in text:
        return "liquidity_sweep"
    if "ob_fvg_retest" in text or "order block / fvg" in text:
        return "ob_fvg_retest"
    if "breakout_retest" in text or "breakout + retest" in text:
        return "breakout_retest"
    if "reversal" in text:
        return "reversal"
    if "mean_reversion" in text or "mean reversion" in text or "range" in text:
        return "range_mean_reversion"
    if "breakout" in text or "breakdown" in text:
        if "retest" in text:
            return "retest_continuation"
        return "breakout_continuation"
    if "momentum" in text and structure.last_bos:
        return "breakout_continuation"
    if "momentum" in text:
        return "cmp_confirmation"
    return "retest_continuation"


def _directional_velocity(df: pd.DataFrame, direction: str, atr: float) -> float:
    if df is None or len(df) < 4:
        return 0.0
    close = df["close"].astype(float)
    raw = (safe_float(close.iloc[-1]) - safe_float(close.iloc[-4])) / max(atr, 1e-12)
    return float(clamp(raw * (1.0 if direction == "long" else -1.0), -2.0, 2.0))


def _entry_accessibility_quality(
    distance_atr: float,
    distance_pct: float,
    width_atr: float,
    zone_spread_multiple: float,
    remaining_minutes: float,
    movement_toward_entry: float,
    tp1_progress_pct: float,
    policy: ExecutionQualityPolicy,
) -> float:
    distance = 100.0 - abs(distance_atr - policy.ideal_retest_distance_atr) * 48.0
    if distance_atr > policy.max_retest_distance_atr:
        distance -= (distance_atr - policy.max_retest_distance_atr) * 45.0
    width = 100.0 - abs(width_atr - 0.24) * 120.0
    precision = min(100.0, zone_spread_multiple / policy.min_zone_spread_multiple * 100.0)
    time_quality = float(clamp(remaining_minutes / 90.0 * 100.0, 20.0, 100.0))
    velocity = float(clamp(55.0 + movement_toward_entry * 28.0, 0.0, 100.0))
    progress = float(clamp(100.0 - tp1_progress_pct * 1.25, 0.0, 100.0))
    percent_penalty = max(0.0, distance_pct - 1.5) * 12.0
    return float(clamp(
        0.27 * distance + 0.17 * width + 0.13 * precision
        + 0.13 * time_quality + 0.15 * velocity + 0.15 * progress
        - percent_penalty
        - max(0.0, distance_atr - policy.max_retest_distance_atr) * 32.0,
        0, 100
    ))


def _pre_entry_survival_quality(
    *, direction: str, price: float, entry: float, stop: float, atr: float,
    candle: CandleContext, structure: StructureReport, remaining_minutes: float,
    anchor_health: float,
) -> float:
    price_to_invalidation = abs(price - stop) / max(atr, 1e-12)
    entry_to_invalidation = abs(entry - stop) / max(atr, 1e-12)
    stop_room = float(clamp((min(price_to_invalidation, entry_to_invalidation) - 0.45) / 1.2 * 100.0, 0, 100))
    wanted = "bullish" if direction == "long" else "bearish"
    bos = 82.0 if structure.last_bos == wanted else (38.0 if structure.last_bos else 58.0)
    choch = 80.0 if structure.last_choch == wanted else (35.0 if structure.last_choch else 58.0)
    adverse = 25.0 if candle.adverse_rejection else (42.0 if candle.absorption else 78.0)
    expiry = float(clamp(remaining_minutes / 90.0 * 100.0, 20.0, 100.0))
    return float(clamp(
        0.27 * stop_room + 0.18 * bos + 0.10 * choch + 0.15 * adverse
        + 0.20 * anchor_health + 0.10 * expiry, 0, 100
    ))


def _confirmation_quality(
    candle: CandleContext,
    flow_alignment: float,
    book_alignment: float,
    setup_type: str,
    bos_agrees: bool,
    structure: StructureReport,
) -> float:
    candle_value = (candle.score + 1.0) * 50.0
    flow_value = (float(clamp(flow_alignment, -1, 1)) + 1.0) * 50.0
    book_value = (float(clamp(book_alignment, -1, 1)) + 1.0) * 50.0
    structure_value = 78.0 if bos_agrees else 52.0
    score = 0.36 * candle_value + 0.27 * flow_value + 0.12 * book_value + 0.25 * structure_value
    if candle.adverse_rejection:
        score -= 24.0
    if candle.absorption:
        score -= 14.0
    if setup_type == "reversal" and structure.last_choch is None:
        score -= 20.0
    return float(clamp(score, 0, 100))


def _stop_quality(
    stop_atr: float,
    candle: CandleContext,
    spread_bps: Optional[float],
    slippage_bps: float,
    anchor_health: float,
    expected_hold_hours: float,
) -> float:
    noise_floor = max(0.70, candle.noise_atr * 0.82)
    tight = max(0.0, noise_floor - stop_atr) * 85.0
    wide = max(0.0, stop_atr - 2.20) * (18.0 + min(12.0, expected_hold_hours))
    friction = max(0.0, (spread_bps or 0.0) + slippage_bps - 12.0) * 1.5
    wick_risk = 14.0 if candle.adverse_rejection else 0.0
    structural = 0.20 * anchor_health
    return float(clamp(82.0 + structural - tight - wide - friction - wick_risk, 0, 100))


def _data_market_quality(
    ticker_age: Optional[float],
    orderbook_age: Optional[float],
    sources_fresh: Optional[bool],
    policy: ExecutionQualityPolicy,
) -> Tuple[float, List[str]]:
    uncertainties: List[str] = []
    score = 100.0
    if ticker_age is None:
        score -= policy.important_missing_penalty
        uncertainties.append("ticker_age_missing")
    else:
        score -= max(0.0, ticker_age - policy.max_ticker_age_seconds * 0.5) * 1.2
    if orderbook_age is None:
        score -= policy.important_missing_penalty
        uncertainties.append("orderbook_age_missing")
    else:
        score -= max(0.0, orderbook_age - policy.max_orderbook_age_seconds * 0.5) * 1.5
    if sources_fresh is False:
        score = 0.0
    return float(clamp(score, 0, 100)), uncertainties


def _liquidity_cost_quality(
    *, spread_bps: Optional[float], book_alignment: float,
    bid_depth_usd: float, ask_depth_usd: float, impact_bps: Optional[float],
    total_cost_bps: float, policy: ExecutionQualityPolicy,
) -> Tuple[float, List[str]]:
    uncertainties: List[str] = []
    score = 88.0
    if spread_bps is None:
        score -= policy.important_missing_penalty
        uncertainties.append("spread_missing")
    else:
        score -= max(0.0, spread_bps - 2.0) * 4.0
    score += book_alignment * 8.0
    if bid_depth_usd <= 0 or ask_depth_usd <= 0:
        score -= policy.important_missing_penalty
        uncertainties.append("notional_depth_missing")
    else:
        shallow = min(bid_depth_usd, ask_depth_usd)
        score += float(clamp((shallow / policy.reference_impact_notional_usd - 1.0) * 5.0, -15.0, 8.0))
    if impact_bps is None:
        score -= policy.optional_missing_penalty
        uncertainties.append("market_impact_unknown")
    else:
        score -= max(0.0, safe_float(impact_bps) - 1.0) * 4.0
    score -= max(0.0, total_cost_bps - 15.0) * 1.2
    return float(clamp(score, 0, 100)), uncertainties


def _reversal_confirmed(
    direction: str,
    structure: StructureReport,
    candle: CandleContext,
) -> bool:
    wanted = "bullish" if direction == "long" else "bearish"
    structure_shift = structure.last_choch == wanted or structure.last_bos == wanted
    rejection = (
        candle.lower_wick_ratio >= 0.35 if direction == "long"
        else candle.upper_wick_ratio >= 0.35
    )
    return bool(structure_shift and (rejection or candle.score >= 0.18))


def _feasible_structure_targets(
    structure: StructureReport,
    df: pd.DataFrame,
    *, direction: str, entry: float, stop: float, atr: float,
    current_price: float, expected_hold_hours: float,
    extra_structure_targets: Optional[List[float]] = None,
    policy: ExecutionQualityPolicy,
) -> Tuple[List[float], List[float], List[float]]:
    """Choose reachable structure targets; never manufacture a minimum R:R."""
    sign = 1.0 if direction == "long" else -1.0
    target_side = "bearish" if direction == "long" else "bullish"
    candidates: List[Tuple[float, float, str]] = []
    for level in structure.levels:
        beyond = level.mid > entry if direction == "long" else level.mid < entry
        if not beyond or level.side != target_side:
            continue
        if getattr(level, "invalidated", False):
            continue
        quality = float(level.confidence)
        if getattr(level, "fully_mitigated", False):
            quality -= 25.0
        candidates.append((float(level.mid), quality, level.kind))
    for value, quality, name in (
        (structure.volume_profile_poc, 62.0, "volume_poc"),
        (structure.volume_profile_vah if direction == "long" else structure.volume_profile_val, 70.0, "value_area"),
        *((value, 68.0, "swing") for value in (
            structure.swing_highs if direction == "long" else structure.swing_lows
        )),
    ):
        if value is not None and ((value > entry) if direction == "long" else (value < entry)):
            candidates.append((float(value), quality, name))
    for value in extra_structure_targets or []:
        if value and ((value > entry) if direction == "long" else (value < entry)):
            candidates.append((float(value), 82.0, "session_structure"))
    candidates.sort(key=lambda item: abs(item[0] - entry))
    targets: List[float] = []
    qualities: List[float] = []
    obstacles: List[float] = []
    seen: List[float] = []
    hold_scale = float(clamp(expected_hold_hours / 8.0, 0.55, 1.35))
    for value, base_quality, _source in candidates:
        distance_atr = abs(value - entry) / max(atr, 1e-12)
        if distance_atr < 0.35:
            if _source != "volume_poc":
                obstacles.append(distance_atr)
            continue
        index = len(targets)
        if index >= len(policy.max_target_atr_by_index):
            break
        max_distance = policy.max_target_atr_by_index[index] * hold_scale
        if distance_atr > max_distance:
            continue
        if any(abs(value - prior) < atr * 0.18 for prior in seen):
            continue
        temporal = float(clamp(100.0 - distance_atr / max(max_distance, 1e-9) * 45.0, 25.0, 100.0))
        quality = float(clamp(0.60 * base_quality + 0.40 * temporal, 0, 100))
        targets.append(value)
        qualities.append(quality)
        seen.append(value)
    # A volatility projection is allowed for TP1 only when no real structure
    # level exists; it is not moved to satisfy the minimum-R gate.
    if not targets:
        projection_atr = min(1.0 * hold_scale, policy.max_target_atr_by_index[0])
        projected = entry + sign * atr * projection_atr
        targets = [float(projected)]
        qualities = [48.0]
    ordered = sorted(zip(targets, qualities), key=lambda item: item[0], reverse=direction == "short")
    return [x for x, _ in ordered], [q for _, q in ordered], obstacles


def _legacy_execution_score(
    *, cluster_strength: float, candle: CandleContext, flow_alignment: float,
    book_alignment: float, chase_distance: float, target_rr2: float,
    spread_bps: Optional[float], adverse_basis: bool,
) -> float:
    """Frozen pre-Phase-2A formula used only for shadow comparison."""
    score = 52.0 + min(18.0, cluster_strength * 6.0)
    score += candle.score * 11.0 + flow_alignment * 10.0 + book_alignment * 7.0
    score += 4.0 if candle.volume_ratio >= 1.05 else -4.0
    score -= 18.0 if candle.adverse_rejection else 0.0
    score -= 10.0 if candle.absorption else 0.0
    score -= max(0.0, chase_distance - 0.5) * 15.0
    score += 6.0 if target_rr2 >= 1.25 else -18.0
    if spread_bps is not None:
        score += 3.0 if spread_bps <= 2.0 else (-25.0 if spread_bps > 12.0 else (-10.0 if spread_bps > 7.0 else 0.0))
    if adverse_basis:
        score -= 6.0
    return float(clamp(score, 0, 100))
