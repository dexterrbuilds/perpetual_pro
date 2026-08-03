"""Stable feature extraction for live scans and historical replay.

Only information available at the decision timestamp belongs here.  This module
is deliberately deterministic so model artifacts remain reproducible.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from src.utils.helpers import safe_float


FEATURE_SCHEMA_VERSION = "3.0"
TECHNICAL_FEATURES: Tuple[str, ...] = (
    "trend",
    "momentum",
    "structure",
    "multi_tf",
    "volume",
    "derivatives",
    "volatility",
    "patterns",
    "news",
    "confluence_abs",
    "adx_scaled",
    "atr_pct_scaled",
    "funding_abs_scaled",
    "oi_change_scaled",
    "funding_missing",
    "oi_missing",
    "news_missing",
)
EXECUTION_FEATURES: Tuple[str, ...] = (
    "chase_distance_atr",
    "zone_width_atr",
    "stop_distance_atr",
    "entry_distance_atr",
    "order_flow_alignment",
    "orderbook_alignment",
    "spread_scaled",
    "basis_support_scaled",
    "candle_alignment",
    "body_ratio",
    "adverse_wick_ratio",
    "volume_ratio_scaled",
    "noise_atr",
    "rr_tp1_scaled",
    "rr_tp2_scaled",
    "expiry_hours_scaled",
    "status_ready",
    "status_wait_retest",
    "adverse_rejection",
    "absorption",
    "spread_missing",
    "orderbook_missing",
)
SETUP_TYPES: Tuple[str, ...] = (
    "cmp_momentum",
    "breakout",
    "breakout_retest",
    "trend_pullback",
    "range_rejection",
    "mean_reversion",
    "structure_reversal",
    "mixed_unknown",
)
INTERACTION_BASES: Tuple[str, ...] = (
    "trend",
    "momentum",
    "structure",
    "multi_tf",
    "chase_distance_atr",
    "entry_distance_atr",
    "order_flow_alignment",
    "adverse_wick_ratio",
    "rr_tp2_scaled",
)


def canonical_setup_type(
    setup_name: str,
    strategy_tags: Optional[Iterable[Any]] = None,
    entry_status: str = "",
) -> str:
    """Map legacy free-form tags into one stable setup taxonomy."""
    blob = " ".join(
        [
            str(setup_name or ""),
            " ".join(str(item) for item in (strategy_tags or [])),
        ]
    ).lower()
    status = str(entry_status or "").lower()
    if "mean reversion" in blob or "mean_reversion" in blob:
        return "mean_reversion"
    if "range" in blob or "rejection" in blob:
        return "range_rejection"
    if "choch" in blob or "reversal" in blob or "structure shift" in blob:
        return "structure_reversal"
    if "breakout" in blob and ("retest" in blob or status == "wait_retest"):
        return "breakout_retest"
    if "breakout" in blob:
        return "breakout"
    if "pullback" in blob or ("trend" in blob and status == "wait_retest"):
        return "trend_pullback"
    if status in ("ready", "confirmation_pending") and any(
        token in blob for token in ("momentum", "continuation", "market")
    ):
        return "cmp_momentum"
    return "mixed_unknown"


def all_model_feature_names() -> Tuple[str, ...]:
    setup_names = tuple(f"setup__{name}" for name in SETUP_TYPES)
    interactions = tuple(
        f"{base}_x_setup__{setup}"
        for setup in SETUP_TYPES
        for base in INTERACTION_BASES
    )
    return (*TECHNICAL_FEATURES, *EXECUTION_FEATURES, *setup_names, *interactions)


def technical_model_feature_names() -> Tuple[str, ...]:
    setup_names = tuple(f"setup__{name}" for name in SETUP_TYPES)
    interactions = tuple(
        f"{base}_x_setup__{setup}"
        for setup in SETUP_TYPES
        for base in INTERACTION_BASES
        if base in TECHNICAL_FEATURES
    )
    return (*TECHNICAL_FEATURES, *setup_names, *interactions)


def execution_model_feature_names() -> Tuple[str, ...]:
    setup_names = tuple(f"setup__{name}" for name in SETUP_TYPES)
    interactions = tuple(
        f"{base}_x_setup__{setup}"
        for setup in SETUP_TYPES
        for base in INTERACTION_BASES
        if base in EXECUTION_FEATURES
    )
    return (*EXECUTION_FEATURES, *setup_names, *interactions)


def extract_candidate_features(
    analysis: Any,
    row: Mapping[str, Any],
) -> Tuple[Dict[str, float], str]:
    """Extract bounded, asset-normalized features from one analysis result."""
    factors = {
        str(item.get("name") or ""): safe_float(item.get("score"))
        for item in (
            analysis.factor_breakdown()
            if analysis is not None and hasattr(analysis, "factor_breakdown")
            else (row.get("factors") or [])
        )
        if isinstance(item, Mapping)
    }
    execution_obj = getattr(analysis, "execution", None) if analysis is not None else None
    execution = (
        execution_obj.to_dict()
        if execution_obj is not None and hasattr(execution_obj, "to_dict")
        else dict(row.get("execution") or {})
    )
    candle = dict(execution.get("candle") or {})
    plan = getattr(analysis, "trade_plan", None) if analysis is not None else None
    meta = dict(getattr(analysis, "meta", None) or row.get("meta") or {})
    indicators = getattr(analysis, "indicators", None) if analysis is not None else None
    indicator_summary = dict(getattr(indicators, "summary", None) or {})

    price = _positive(meta.get("price") or row.get("price"))
    atr = _positive(meta.get("atr") or getattr(plan, "atr", None))
    entry_low = _number(row.get("entry_low") or getattr(plan, "entry_low", None))
    entry_high = _number(row.get("entry_high") or getattr(plan, "entry_high", None))
    stop = _number(row.get("stop_loss") or getattr(plan, "stop_loss", None))
    entry_mid = (
        (entry_low + entry_high) / 2.0
        if entry_low > 0 and entry_high > 0
        else price
    )
    rr_values = list(
        row.get("risk_reward")
        or getattr(plan, "risk_reward", None)
        or []
    )
    if not rr_values and entry_mid > 0 and stop > 0:
        risk = abs(entry_mid - stop)
        rr_values = [
            abs(_number(target) - entry_mid) / max(risk, entry_mid * 1e-9)
            for target in list(
                row.get("take_profits")
                or getattr(plan, "take_profits", None)
                or []
            )
        ]

    direction = str(
        row.get("direction") or getattr(analysis, "direction", "") or ""
    ).lower()
    # A flat analysis is research context, not a synthetic directional example.
    # Never derive a training direction from bias for a row that did not qualify
    # as an actual long/short candidate.
    if direction not in ("long", "short"):
        direction = "flat"
    direction_sign = 1.0 if direction == "long" else -1.0
    status = str(
        row.get("entry_status") or execution.get("status") or "blocked"
    ).lower()
    setup_type = canonical_setup_type(
        str(row.get("setup_name") or getattr(analysis, "setup_name", "") or ""),
        row.get("strategy_tags")
        or getattr(analysis, "strategy_tags", None)
        or [],
        status,
    )
    snapshot = getattr(analysis, "snapshot", None) if analysis is not None else None
    funding_raw = _first_not_none(
        row.get("funding_rate"),
        meta.get("funding_rate"),
        getattr(snapshot, "funding_rate", None),
    )
    oi_raw = _first_not_none(
        row.get("open_interest_change_pct_24h"),
        getattr(snapshot, "open_interest_change_pct_24h", None),
    )
    funding = _number(funding_raw)
    oi_change = _number(oi_raw)
    news_obj = getattr(analysis, "news", None) if analysis is not None else None
    spread_raw = execution.get("spread_bps")
    book_raw = execution.get("orderbook_imbalance")
    basis = _number(execution.get("mark_index_basis_bps"))
    adverse_wick = (
        _number(candle.get("upper_wick_ratio"))
        if direction == "long"
        else _number(candle.get("lower_wick_ratio"))
    )

    features: Dict[str, float] = {
        "trend": _bounded(_number(factors.get("trend")) * direction_sign, -1, 1),
        "momentum": _bounded(_number(factors.get("momentum")) * direction_sign, -1, 1),
        "structure": _bounded(_number(factors.get("structure")) * direction_sign, -1, 1),
        "multi_tf": _bounded(_number(factors.get("multi_tf")) * direction_sign, -1, 1),
        "volume": _bounded(_number(factors.get("volume")) * direction_sign, -1, 1),
        "derivatives": _bounded(_number(factors.get("derivatives")) * direction_sign, -1, 1),
        "volatility": _bounded(_number(factors.get("volatility")) * direction_sign, -1, 1),
        "patterns": _bounded(_number(factors.get("patterns")) * direction_sign, -1, 1),
        "news": _bounded(_number(factors.get("news")) * direction_sign, -1, 1),
        "confluence_abs": _bounded(
            abs(_number(row.get("confluence_score") or getattr(analysis, "confluence_total", 0))),
            0,
            1,
        ),
        "adx_scaled": _bounded(_number(indicator_summary.get("adx")) / 50.0, 0, 2),
        "atr_pct_scaled": _bounded(
            _number(meta.get("atr_pct") or getattr(plan, "atr_pct", None)) / 3.0,
            0,
            3,
        ),
        "funding_abs_scaled": _bounded(abs(funding * 100.0) / 0.05, 0, 4),
        "oi_change_scaled": _bounded(oi_change / 10.0, -3, 3),
        "funding_missing": 1.0 if funding_raw is None else 0.0,
        "oi_missing": 1.0 if oi_raw is None else 0.0,
        "news_missing": 1.0 if news_obj is None else 0.0,
        "chase_distance_atr": _bounded(
            _number(execution.get("chase_distance_atr")), 0, 4
        ),
        "zone_width_atr": _bounded(
            abs(entry_high - entry_low) / max(atr, price * 1e-9, 1e-12),
            0,
            4,
        ),
        "stop_distance_atr": _bounded(
            _number(execution.get("stop_distance_atr"))
            or abs(entry_mid - stop) / max(atr, price * 1e-9, 1e-12),
            0,
            5,
        ),
        "entry_distance_atr": _bounded(
            abs(price - entry_mid) / max(atr, price * 1e-9, 1e-12),
            0,
            5,
        ),
        "order_flow_alignment": _bounded(
            _number(execution.get("order_flow_score")) * direction_sign,
            -1,
            1,
        ),
        "orderbook_alignment": _bounded(
            _number(execution.get("orderbook_alignment")), -1, 1
        ),
        "spread_scaled": _bounded(_number(spread_raw) / 12.0, 0, 5),
        "basis_support_scaled": _bounded(
            -basis * direction_sign / 18.0, -4, 4
        ),
        "candle_alignment": _bounded(
            _number(execution.get("candle_score")), -1, 1
        ),
        "body_ratio": _bounded(_number(candle.get("body_ratio")), 0, 1.5),
        "adverse_wick_ratio": _bounded(adverse_wick, 0, 1.5),
        "volume_ratio_scaled": _bounded(
            _number(candle.get("volume_ratio"), 1.0) / 2.0, 0, 4
        ),
        "noise_atr": _bounded(_number(candle.get("noise_atr"), 1.0), 0, 3),
        "rr_tp1_scaled": _bounded(_at(rr_values, 0) / 2.0, 0, 3),
        "rr_tp2_scaled": _bounded(_at(rr_values, 1) / 2.0, 0, 4),
        "expiry_hours_scaled": _bounded(
            _number(
                row.get("entry_valid_for_minutes")
                or getattr(plan, "entry_valid_for_minutes", None)
            )
            / 180.0,
            0,
            2,
        ),
        "status_ready": 1.0 if status in ("ready", "confirmation_pending") else 0.0,
        "status_wait_retest": 1.0 if status == "wait_retest" else 0.0,
        "adverse_rejection": 1.0 if candle.get("adverse_rejection") else 0.0,
        "absorption": 1.0 if candle.get("absorption") else 0.0,
        "spread_missing": 1.0 if spread_raw is None else 0.0,
        "orderbook_missing": 1.0 if book_raw is None else 0.0,
    }
    for setup in SETUP_TYPES:
        features[f"setup__{setup}"] = 1.0 if setup == setup_type else 0.0
    for setup in SETUP_TYPES:
        setup_value = features[f"setup__{setup}"]
        for base in INTERACTION_BASES:
            features[f"{base}_x_setup__{setup}"] = (
                features.get(base, 0.0) * setup_value
            )
    return {name: float(features.get(name, 0.0)) for name in all_model_feature_names()}, setup_type


def build_candidate_record(
    analysis: Any,
    row: Mapping[str, Any],
    *,
    source: str,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> Dict[str, Any]:
    if str(feature_schema_version) != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "Feature schema mismatch: extractor produces "
            f"{FEATURE_SCHEMA_VERSION}, requested {feature_schema_version}"
        )
    features, setup_type = extract_candidate_features(analysis, row)
    generated_at = str(
        row.get("signal_generated_at")
        or getattr(analysis, "generated_at", "")
        or datetime.now(timezone.utc).isoformat()
    )
    symbol = str(row.get("symbol") or getattr(analysis, "symbol", "") or "")
    direction = str(
        row.get("direction") or getattr(analysis, "direction", "flat") or "flat"
    ).lower()
    if direction not in ("long", "short"):
        direction = "flat"
    is_directional_candidate = direction in ("long", "short")
    identity = "|".join(
        [
            generated_at,
            symbol,
            direction,
            str(row.get("entry_low") or ""),
            str(row.get("entry_high") or ""),
            str(row.get("stop_loss") or ""),
            str(source),
            str(feature_schema_version),
        ]
    )
    candidate_id = "cand_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:28]
    decision = {
        key: row.get(key)
        for key in (
            "price",
            "entry_low",
            "entry_high",
            "stop_loss",
            "take_profits",
            "entry_status",
            "entry_valid_until",
            "entry_valid_for_minutes",
            "hold_hours_max",
            "risk_pct",
            "prop_safe",
            "prop_guidance",
            "universal_eligible",
            "production_qualified",
            "signal_eligible",
            "historical_edge_ok",
            "data_quality_ok",
            "market_quality_ok",
            "legacy_signal_eligible",
            "legacy_v2_signal_eligible",
            "rejection_reasons",
            "execution_policy_version",
            "legacy_execution_status",
            "legacy_execution_targets",
            "execution_setup_type",
            "entry_mode",
            "execution_components",
            "entry_accessibility",
            "pre_entry_survival",
            "confirmation_quality",
            "stop_quality",
            "target_feasibility",
            "gross_risk_reward",
            "net_risk_reward",
            "estimated_total_cost_bps",
            "estimated_fee_bps",
            "estimated_slippage_bps",
            "estimated_funding_bps",
            "estimated_impact_bps",
            "depth_bands_bps",
            "structural_obstacle_distances_atr",
            "data_freshness_state",
            "spread_bps",
            "ticker_age_seconds",
            "orderbook_age_seconds",
            "chase_distance_atr",
            "entry_distance_atr",
            "entry_distance_pct",
            "entry_zone_width_atr",
            "remaining_expiry_minutes",
            "stop_distance_atr",
            "hard_failures",
            "execution_uncertainties",
            "rank_policy_version",
            "rank_breakdown",
        )
    }
    decision["atr"] = _number(
        (getattr(analysis, "meta", None) or {}).get("atr")
        if analysis is not None
        else None
    )
    decision["atr_pct"] = _number(
        (getattr(analysis, "meta", None) or {}).get("atr_pct")
        if analysis is not None
        else None
    )
    decision["risk_reward"] = list(
        getattr(getattr(analysis, "trade_plan", None), "risk_reward", None) or []
    )
    production_scores = {
        "technical": row.get("technical_confidence"),
        "execution": row.get("execution_score"),
        "execution_quality": row.get("execution_quality", row.get("execution_score")),
        "legacy_execution_score": row.get("legacy_execution_score"),
        "execution_policy_version": row.get("execution_policy_version"),
        "execution_components": dict(row.get("execution_components") or {}),
        "gross_risk_reward": list(row.get("gross_risk_reward") or []),
        "net_risk_reward": list(row.get("net_risk_reward") or []),
        "confidence": row.get("confidence"),
        "legacy_confidence": row.get("legacy_confidence"),
        "legacy_v2_confidence": row.get("legacy_v2_confidence"),
        "rank": row.get("rank_score"),
        "confluence": row.get("confluence_score"),
        "llm": row.get("llm_confidence"),
        "immediate_sl_risk": row.get("immediate_sl_risk"),
        "prop_safe": bool(row.get("prop_safe", False)),
        "prop_guidance": dict(row.get("prop_guidance") or {}),
        "universal_eligible": bool(row.get("universal_eligible", False)),
        "production_qualified": bool(
            row.get("production_qualified", row.get("signal_eligible", False))
        ),
        "legacy_signal_eligible": bool(
            row.get("legacy_signal_eligible", False)
        ),
        "legacy_v2_signal_eligible": bool(
            row.get("legacy_v2_signal_eligible", False)
        ),
        "rejection_reasons": list(row.get("rejection_reasons") or []),
    }
    return {
        "id": candidate_id,
        "generated_at": generated_at,
        "symbol": symbol,
        "exchange_id": str(
            row.get("exchange") or getattr(analysis, "exchange_id", "") or ""
        ),
        "timeframe": str(
            row.get("primary_tf") or getattr(analysis, "primary_tf", "15m")
        ),
        "direction": direction,
        "is_directional_candidate": is_directional_candidate,
        "setup_type": setup_type,
        "setup_name": str(
            row.get("setup_name") or getattr(analysis, "setup_name", "") or ""
        ),
        "feature_schema_version": str(feature_schema_version),
        "features": features,
        "decision": decision,
        "production_scores": production_scores,
        "production_eligible": bool(
            row.get("production_qualified", row.get("signal_eligible", False))
        ),
        "production_rank": _number(row.get("rank_score")),
        "source": source,
    }


def vector_from_features(
    features: Mapping[str, Any],
    names: Iterable[str],
) -> List[float]:
    return [_number(features.get(name)) for name in names]


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _positive(value: Any) -> float:
    return max(0.0, _number(value))


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _bounded(value: Any, low: float, high: float) -> float:
    return min(high, max(low, _number(value)))


def _at(values: List[Any], index: int) -> float:
    return _number(values[index]) if len(values) > index else 0.0
