"""Versioned deterministic policy for execution and interim signal ranking.

All values in this module are quality heuristics, not probabilities.  Hard
safety failures stay outside the weighted formula and reject the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple

from src.utils.helpers import clamp


EXECUTION_POLICY_VERSION = "execution_quality_v2a.1"
RANK_POLICY_VERSION = "deterministic_rank_v2a.1"


@dataclass(frozen=True)
class ExecutionQualityPolicy:
    """Central Phase 2A policy; component scores and weights are 0..100."""

    version: str = EXECUTION_POLICY_VERSION
    component_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "entry_accessibility": 0.18,
            "entry_zone_quality": 0.10,
            "pre_entry_survival": 0.16,
            "confirmation_quality": 0.10,
            "stop_quality": 0.14,
            "target_feasibility": 0.14,
            "liquidity_cost_quality": 0.10,
            "data_market_quality": 0.08,
        }
    )
    setup_weights: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            "cmp_confirmation": {
                "entry_accessibility": 0.16, "entry_zone_quality": 0.10,
                "pre_entry_survival": 0.12, "confirmation_quality": 0.18,
                "stop_quality": 0.15, "target_feasibility": 0.14,
                "liquidity_cost_quality": 0.09, "data_market_quality": 0.06,
            },
            "retest_continuation": {
                "entry_accessibility": 0.22, "entry_zone_quality": 0.13,
                "pre_entry_survival": 0.20, "confirmation_quality": 0.07,
                "stop_quality": 0.12, "target_feasibility": 0.12,
                "liquidity_cost_quality": 0.08, "data_market_quality": 0.06,
            },
            # New setup families reuse the same eight independent components;
            # only their emphasis changes. Global quality floors and hard
            # gates remain unchanged.
            "trend_pullback": {
                "entry_accessibility": 0.20, "entry_zone_quality": 0.12,
                "pre_entry_survival": 0.18, "confirmation_quality": 0.10,
                "stop_quality": 0.13, "target_feasibility": 0.13,
                "liquidity_cost_quality": 0.08, "data_market_quality": 0.06,
            },
            "breakout_retest": {
                "entry_accessibility": 0.19, "entry_zone_quality": 0.12,
                "pre_entry_survival": 0.17, "confirmation_quality": 0.13,
                "stop_quality": 0.12, "target_feasibility": 0.13,
                "liquidity_cost_quality": 0.08, "data_market_quality": 0.06,
            },
            "ob_fvg_retest": {
                "entry_accessibility": 0.20, "entry_zone_quality": 0.16,
                "pre_entry_survival": 0.18, "confirmation_quality": 0.09,
                "stop_quality": 0.12, "target_feasibility": 0.11,
                "liquidity_cost_quality": 0.08, "data_market_quality": 0.06,
            },
            "liquidity_sweep": {
                "entry_accessibility": 0.12, "entry_zone_quality": 0.12,
                "pre_entry_survival": 0.17, "confirmation_quality": 0.22,
                "stop_quality": 0.15, "target_feasibility": 0.11,
                "liquidity_cost_quality": 0.06, "data_market_quality": 0.05,
            },
            "opening_range_breakout": {
                "entry_accessibility": 0.15, "entry_zone_quality": 0.14,
                "pre_entry_survival": 0.15, "confirmation_quality": 0.18,
                "stop_quality": 0.13, "target_feasibility": 0.13,
                "liquidity_cost_quality": 0.07, "data_market_quality": 0.05,
            },
            "session_high_low_rejection": {
                "entry_accessibility": 0.12, "entry_zone_quality": 0.14,
                "pre_entry_survival": 0.16, "confirmation_quality": 0.20,
                "stop_quality": 0.16, "target_feasibility": 0.11,
                "liquidity_cost_quality": 0.06, "data_market_quality": 0.05,
            },
            "breakout_continuation": {
                "entry_accessibility": 0.14, "entry_zone_quality": 0.08,
                "pre_entry_survival": 0.12, "confirmation_quality": 0.18,
                "stop_quality": 0.13, "target_feasibility": 0.16,
                "liquidity_cost_quality": 0.11, "data_market_quality": 0.08,
            },
            "range_mean_reversion": {
                "entry_accessibility": 0.16, "entry_zone_quality": 0.16,
                "pre_entry_survival": 0.14, "confirmation_quality": 0.14,
                "stop_quality": 0.15, "target_feasibility": 0.13,
                "liquidity_cost_quality": 0.07, "data_market_quality": 0.05,
            },
            "reversal": {
                "entry_accessibility": 0.12, "entry_zone_quality": 0.11,
                "pre_entry_survival": 0.18, "confirmation_quality": 0.20,
                "stop_quality": 0.15, "target_feasibility": 0.12,
                "liquidity_cost_quality": 0.07, "data_market_quality": 0.05,
            },
        }
    )
    # Entry geometry and lifecycle.
    ideal_retest_distance_atr: float = 0.45
    max_retest_distance_atr: float = 1.35
    max_entry_zone_width_atr: float = 0.45
    min_zone_spread_multiple: float = 4.0
    stale_level_bars: int = 48
    expired_level_bars: int = 96
    max_level_touches: int = 2
    max_pre_entry_tp1_progress_pct: float = 70.0
    # Stop/target/cost policy.
    tight_stop_atr: float = 0.85
    wide_stop_atr: float = 2.20
    max_stop_atr: float = 3.00
    max_target_atr_by_index: Tuple[float, ...] = (2.2, 4.0, 6.0, 8.0)
    min_net_tp1_cost_multiple: float = 2.0
    # Nearby opposing structure is resolved conservatively: a small clearance
    # may become TP1 only when a later real structure level leaves open space
    # and the adjusted target still returns at least 0.75R net of costs.
    nearby_target_obstacle_atr: float = 0.35
    structure_target_clearance_atr: float = 0.08
    min_structure_adjusted_target_net_rr: float = 0.75
    default_taker_fee_bps_per_side: float = 5.0
    default_slippage_bps_per_side: float = 1.5
    default_funding_bps_per_8h: float = 1.0
    reference_impact_notional_usd: float = 10_000.0
    important_missing_penalty: float = 18.0
    optional_missing_penalty: float = 5.0
    # Existing hard safety thresholds are intentionally preserved.
    max_spread_bps: float = 12.0
    max_ticker_age_seconds: float = 45.0
    max_orderbook_age_seconds: float = 30.0
    minimum_execution_quality: float = 72.0


DEFAULT_EXECUTION_POLICY = ExecutionQualityPolicy()


def weighted_execution_quality(
    components: Mapping[str, float],
    setup_type: str,
    policy: ExecutionQualityPolicy = DEFAULT_EXECUTION_POLICY,
) -> Tuple[float, Dict[str, float]]:
    """Return bounded quality and inspectable weighted contributions."""
    weights = dict(policy.setup_weights.get(setup_type, policy.component_weights))
    total_weight = sum(max(0.0, float(v)) for v in weights.values()) or 1.0
    contributions: Dict[str, float] = {}
    for name, weight in weights.items():
        value = float(clamp(float(components.get(name, 0.0)), 0.0, 100.0))
        contributions[name] = value * max(0.0, float(weight)) / total_weight
    return float(clamp(sum(contributions.values()), 0.0, 100.0)), contributions


def deterministic_rank_score(
    *,
    overall_quality: float,
    execution_quality: float,
    target_feasibility: float,
    stop_quality: float,
    net_rr: float,
    market_data_quality: float,
    setup_validity: float,
    uncertainty_penalty: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Interim quality rank. It is deliberately not named expected value."""
    rr_quality = float(clamp((float(net_rr) - 0.5) / 2.0 * 100.0, 0.0, 100.0))
    values = {
        "legacy_v2_overall_quality": float(clamp(overall_quality, 0, 100)),
        "execution_quality": float(clamp(execution_quality, 0, 100)),
        "target_feasibility": float(clamp(target_feasibility, 0, 100)),
        "stop_quality": float(clamp(stop_quality, 0, 100)),
        "net_rr_quality": rr_quality,
        "market_data_quality": float(clamp(market_data_quality, 0, 100)),
        "setup_validity": float(clamp(setup_validity, 0, 100)),
    }
    weights = {
        "legacy_v2_overall_quality": 0.25,
        "execution_quality": 0.25,
        "target_feasibility": 0.13,
        "stop_quality": 0.12,
        "net_rr_quality": 0.10,
        "market_data_quality": 0.08,
        "setup_validity": 0.07,
    }
    contributions = {name: values[name] * weight for name, weight in weights.items()}
    penalty = float(clamp(uncertainty_penalty, 0.0, 30.0))
    contributions["uncertainty_penalty"] = -penalty
    return float(clamp(sum(contributions.values()), 0.0, 100.0)), contributions
