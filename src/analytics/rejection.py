"""Canonical, non-authoritative rejection diagnostics.

The evaluator mirrors existing production gates.  It does not set thresholds,
alter scores, fetch market data, or make a rejected candidate eligible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from math import isfinite
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


GATE_POLICY_VERSION = "rejection_analytics_v1.0"
ALERT_MIN_OVERALL_QUALITY = 80.0

# Stable public diagnostic vocabulary. Existing internal detail strings may be
# retained separately, but persistence and aggregation use these codes.
STABLE_REJECTION_CODES = frozenset(
    {
        "FLAT_DIRECTION",
        "CONFLUENCE_BELOW_MINIMUM",
        "TECHNICAL_QUALITY_BELOW_MINIMUM",
        "EXECUTION_QUALITY_BELOW_MINIMUM",
        "OVERALL_QUALITY_BELOW_MINIMUM",
        "RANK_BELOW_MINIMUM",
        "IMMEDIATE_SL_RISK_TOO_HIGH",
        "DATA_QUALITY_FAILED",
        "MARKET_QUALITY_FAILED",
        "STALE_TICKER",
        "STALE_ORDER_BOOK",
        "SPREAD_TOO_WIDE",
        "ENTRY_BLOCKED",
        "ENTRY_EXPIRED",
        "CONFIRMATION_PENDING",
        "AVOID_CHASE",
        "PRICE_TOO_EXTENDED",
        "INVALIDATED_BEFORE_ENTRY",
        "TP1_ALREADY_PROGRESSING",
        "NO_FEASIBLE_TARGET",
        "TP1_BLOCKED",
        "NET_RR_BELOW_MINIMUM",
        "GROSS_RR_BELOW_MINIMUM",
        "STOP_QUALITY_FAILED",
        "STOP_TOO_TIGHT",
        "STOP_TOO_WIDE",
        "PROP_COMPATIBILITY_FAILED",
        "RANGE_DURING_TREND_EXPANSION",
        "REVERSAL_CONFIRMATION_INSUFFICIENT",
        "STRUCTURE_CONFLICT",
        "REVALIDATION_FAILED",
        "LLM_UNAVAILABLE_NON_BLOCKING",
        "ANALYSIS_ERROR",
        "MARKET_UNAVAILABLE",
        "HISTORICAL_EDGE_FAILED",
    }
)


@dataclass(frozen=True)
class GateResult:
    code: str
    gate_name: str
    actual_value: Any
    required_value: Any
    passed: bool
    distance: Optional[float]
    normalized_distance: float
    severity: str
    stage: str
    explanation: str
    authoritative: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateEvaluation:
    stage: str
    eligible: bool
    gates: List[GateResult] = field(default_factory=list)
    policy_version: str = GATE_POLICY_VERSION
    primary_rejection_reason: Optional[str] = None
    all_rejection_reasons: List[str] = field(default_factory=list)
    closest_to_passing_gate: Optional[str] = None
    distance_to_eligibility: float = 0.0
    failed_hard_gates: int = 0
    failed_soft_gates: int = 0
    would_still_fail_without_primary: bool = False
    proximity_label: str = "ELIGIBLE"

    def finalize(self, *, prior_eligible: bool = True) -> "GateEvaluation":
        failed = [gate for gate in self.gates if not gate.passed]
        blocking = [
            gate
            for gate in failed
            if gate.authoritative and gate.severity == "hard"
        ]
        soft = [gate for gate in failed if gate not in blocking]
        self.primary_rejection_reason = blocking[0].code if blocking else None
        self.all_rejection_reasons = list(dict.fromkeys(gate.code for gate in failed))
        numeric = [gate for gate in blocking if gate.distance is not None]
        self.closest_to_passing_gate = (
            min(numeric, key=lambda gate: gate.normalized_distance).code
            if numeric
            else (blocking[0].code if blocking else None)
        )
        self.failed_hard_gates = len(blocking)
        self.failed_soft_gates = len(soft)
        self.distance_to_eligibility = round(
            sum(max(0.0, gate.normalized_distance) for gate in blocking), 6
        )
        self.eligible = bool(prior_eligible and not blocking)
        self.would_still_fail_without_primary = bool(
            len({gate.code for gate in blocking}) > 1
            or (not prior_eligible and not blocking)
        )
        codes = set(self.all_rejection_reasons)
        if not self.eligible:
            if "FLAT_DIRECTION" in codes:
                self.proximity_label = "NON_DIRECTIONAL"
            elif codes & {
                "DATA_QUALITY_FAILED",
                "MARKET_QUALITY_FAILED",
                "STALE_TICKER",
                "STALE_ORDER_BOOK",
                "MARKET_UNAVAILABLE",
                "ANALYSIS_ERROR",
            }:
                self.proximity_label = "DATA_OR_MARKET_FAILURE"
            elif self.failed_hard_gates <= 1 and self.distance_to_eligibility <= 0.10:
                self.proximity_label = "NEAR_PASS"
            elif self.failed_hard_gates <= 2 and self.distance_to_eligibility <= 0.40:
                self.proximity_label = "MODERATE_GAP"
            else:
                self.proximity_label = "FAR_FROM_ELIGIBLE"
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "stage": self.stage,
            "eligible": self.eligible,
            "primary_rejection_reason": self.primary_rejection_reason,
            "all_rejection_reasons": self.all_rejection_reasons,
            "closest_to_passing_gate": self.closest_to_passing_gate,
            "distance_to_eligibility": self.distance_to_eligibility,
            "failed_hard_gates": self.failed_hard_gates,
            "failed_soft_gates": self.failed_soft_gates,
            "would_still_fail_without_primary": self.would_still_fail_without_primary,
            "proximity_label": self.proximity_label,
            "gates": [gate.to_dict() for gate in self.gates],
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "GateEvaluation":
        data = dict(payload or {})
        gates = [GateResult(**dict(item)) for item in data.get("gates") or []]
        return cls(
            stage=str(data.get("stage") or "unknown"),
            eligible=bool(data.get("eligible")),
            gates=gates,
            policy_version=str(data.get("policy_version") or GATE_POLICY_VERSION),
        ).finalize(prior_eligible=bool(data.get("eligible", True)) or not gates)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _minimum_gate(
    code: str,
    name: str,
    actual: Any,
    required: float,
    *,
    stage: str,
    severity: str = "hard",
    authoritative: bool = True,
    explanation: Optional[str] = None,
    missing_passes: bool = False,
) -> GateResult:
    value = _number(actual)
    passed = missing_passes if value is None else value >= required
    shortfall = max(0.0, required - value) if value is not None else None
    normalized = (
        shortfall / max(abs(required), 1.0) if shortfall is not None else 1.0
    )
    return GateResult(
        code=code,
        gate_name=name,
        actual_value=value,
        required_value={"operator": ">=", "value": required},
        passed=passed,
        distance=round(shortfall, 6) if shortfall is not None else None,
        normalized_distance=round(normalized, 6),
        severity=severity,
        stage=stage,
        explanation=explanation or (
            f"{name} was not present on this compatible legacy row"
            if value is None and passed
            else (
                f"{name} {value:.2f} meets minimum {required:.2f}"
                if passed and value is not None
                else f"{name} is below minimum {required:.2f}"
            )
        ),
        authoritative=authoritative,
    )


def _maximum_gate(
    code: str,
    name: str,
    actual: Any,
    required: float,
    *,
    stage: str,
    severity: str = "hard",
    authoritative: bool = True,
    missing_passes: bool = False,
) -> GateResult:
    value = _number(actual)
    passed = missing_passes if value is None else value <= required
    excess = max(0.0, value - required) if value is not None else None
    normalized = excess / max(abs(required), 1.0) if excess is not None else (0.0 if passed else 1.0)
    return GateResult(
        code=code,
        gate_name=name,
        actual_value=value,
        required_value={"operator": "<=", "value": required},
        passed=passed,
        distance=round(excess, 6) if excess is not None else None,
        normalized_distance=round(normalized, 6),
        severity=severity,
        stage=stage,
        explanation=(
            f"{name} was not present on this compatible legacy row"
            if value is None and passed
            else (
                f"{name} {value:.2f} is within maximum {required:.2f}"
                if passed and value is not None
                else f"{name} exceeds maximum {required:.2f}"
            )
        ),
        authoritative=authoritative,
    )


def _condition_gate(
    code: str,
    name: str,
    passed: bool,
    actual: Any,
    required: Any,
    *,
    stage: str,
    explanation: str,
    severity: str = "hard",
    authoritative: bool = True,
) -> GateResult:
    return GateResult(
        code=code,
        gate_name=name,
        actual_value=actual,
        required_value=required,
        passed=bool(passed),
        distance=0.0 if passed else None,
        normalized_distance=0.0 if passed else 1.0,
        severity=severity,
        stage=stage,
        explanation=explanation,
        authoritative=authoritative,
    )


_HARD_FAILURE_CODES = {
    "stale_execution_data": "DATA_QUALITY_FAILED",
    "spread_above_hard_limit": "SPREAD_TOO_WIDE",
    "no_feasible_target": "NO_FEASIBLE_TARGET",
    "tp1_blocked_by_nearby_structure": "TP1_BLOCKED",
    "tp1_reward_does_not_cover_cost_buffer": "NET_RR_BELOW_MINIMUM",
    "stop_beyond_maximum_structure_distance": "STOP_TOO_WIDE",
    "range_setup_during_trend_expansion": "RANGE_DURING_TREND_EXPANSION",
    "reversal_confirmation_insufficient": "REVERSAL_CONFIRMATION_INSUFFICIENT",
    "cmp_position_excessively_extended_inside_zone": "PRICE_TOO_EXTENDED",
    "adverse_structure_change_before_confirmation": "STRUCTURE_CONFLICT",
}


def _entry_gate(
    status: Any,
    *,
    stage: str,
    missing_passes: bool = False,
) -> GateResult:
    if status is None and missing_passes:
        return _condition_gate(
            "ENTRY_BLOCKED",
            "Entry state",
            True,
            None,
            ["confirmation_pending", "wait_retest"],
            stage=stage,
            explanation="Entry state was not present on this compatible legacy row",
        )
    value = str(status or "blocked").lower()
    allowed = value in {"ready", "confirmation_pending", "wait_retest"}
    code = {
        "avoid_chase": "AVOID_CHASE",
        "expired": "ENTRY_EXPIRED",
        "invalidated": "INVALIDATED_BEFORE_ENTRY",
        "confirmation_pending": "CONFIRMATION_PENDING",
    }.get(value, "ENTRY_BLOCKED")
    return _condition_gate(
        code,
        "Entry state",
        allowed,
        value,
        ["confirmation_pending", "wait_retest"],
        stage=stage,
        explanation=(
            f"Entry state {value} is eligible for publication"
            if allowed
            else f"Entry state {value} blocks publication"
        ),
    )


def evaluate_analysis_gates(
    *,
    direction: str,
    technical_quality: Any,
    execution_quality: Any,
    overall_quality: Any,
    confluence_magnitude: Any,
    entry_status: Any,
    immediate_sl_risk: Any,
    market_quality_ok: bool,
    data_quality_ok: bool,
    prop_safe: bool,
    confidence_floor: float,
    score_floor: float,
    execution_floor: float,
    max_immediate_sl_risk: float,
    hard_failures: Sequence[str] = (),
) -> GateEvaluation:
    """Mirror the authoritative analysis eligibility expression exactly."""
    stage = "analysis"
    directional = str(direction or "").lower() in {"long", "short"}
    gates = [
        _condition_gate(
            "FLAT_DIRECTION", "Directional candidate", directional,
            str(direction or "flat").lower(), ["long", "short"], stage=stage,
            explanation="Candidate is directional" if directional else "Candidate is flat/non-directional",
        ),
        # Technical Quality is visible diagnostics. Legacy V2's Overall cap
        # already makes this constraint implicit; it is not a new authority.
        _minimum_gate(
            "TECHNICAL_QUALITY_BELOW_MINIMUM", "Technical Quality",
            technical_quality, confidence_floor, stage=stage,
            severity="soft", authoritative=False,
        ),
        _minimum_gate(
            "OVERALL_QUALITY_BELOW_MINIMUM", "Overall Quality",
            overall_quality, confidence_floor, stage=stage,
        ),
        _minimum_gate(
            "CONFLUENCE_BELOW_MINIMUM", "Confluence magnitude",
            confluence_magnitude, score_floor, stage=stage,
        ),
        _minimum_gate(
            "EXECUTION_QUALITY_BELOW_MINIMUM", "Execution Quality",
            execution_quality, execution_floor, stage=stage,
        ),
        _entry_gate(entry_status, stage=stage),
        _maximum_gate(
            "IMMEDIATE_SL_RISK_TOO_HIGH", "Immediate-SL risk",
            immediate_sl_risk, max_immediate_sl_risk, stage=stage,
        ),
        _condition_gate(
            "MARKET_QUALITY_FAILED", "Market quality", market_quality_ok,
            bool(market_quality_ok), True, stage=stage,
            explanation="Market quality passed" if market_quality_ok else "Market quality failed",
        ),
        _condition_gate(
            "DATA_QUALITY_FAILED", "Market-data quality", data_quality_ok,
            bool(data_quality_ok), True, stage=stage,
            explanation="Market data passed" if data_quality_ok else "Market data failed",
        ),
        _condition_gate(
            "PROP_COMPATIBILITY_FAILED", "Prop compatibility", prop_safe,
            bool(prop_safe), True, stage=stage,
            explanation="Plan is prop-compatible" if prop_safe else "Plan failed prop compatibility",
        ),
    ]
    for failure in hard_failures:
        code = _HARD_FAILURE_CODES.get(str(failure), "ANALYSIS_ERROR")
        gates.append(
            _condition_gate(
                code,
                "Execution hard-failure detail",
                False,
                str(failure),
                "absent",
                stage=stage,
                explanation=str(failure).replace("_", " "),
                severity="diagnostic",
                authoritative=False,
            )
        )
    return GateEvaluation(stage=stage, eligible=False, gates=gates).finalize()


def _tp2_rr(row: Mapping[str, Any]) -> Optional[float]:
    values = list(row.get("gross_risk_reward") or row.get("risk_reward") or [])
    if not values:
        payload = dict(row.get("payload") or {})
        setup = dict(payload.get("primary_setup") or payload.get("trade_plan") or {})
        values = list(setup.get("risk_reward") or [])
    if not values:
        return None
    index = 1 if len(values) > 1 else 0
    return _number(values[index])


def evaluate_alert_gates(
    row: Mapping[str, Any],
    *,
    min_rank: float,
    only_prop_safe: bool,
    min_confidence: float,
    min_execution_quality: float,
    max_immediate_sl_risk: float,
    max_chase_distance_atr: float,
    max_pre_entry_tp1_progress_pct: float,
    min_tp2_rr: float,
    max_spread_bps: float,
    max_ticker_age_seconds: float = 45.0,
    max_orderbook_age_seconds: float = 30.0,
    prior: Optional[Mapping[str, Any]] = None,
) -> GateEvaluation:
    """Mirror the scheduled/manual alert filter without new authority."""
    stage = "alert_filter"
    direction = str(
        row.get("evaluated_direction") or row.get("direction") or ""
    ).lower()
    directional = direction in {"long", "short"}
    market_ok = row.get("market_quality_ok") is not False
    data_ok = row.get("data_quality_ok") is not False
    historical_ok = row.get("historical_edge_ok") is not False
    prop_ok = not only_prop_safe or row.get("prop_safe") is not False
    progress_blocked = bool(
        str(row.get("entry_zone_relation") or "") == "favorable_beyond"
        and _number(row.get("tp1_progress_pct")) is not None
        and float(row.get("tp1_progress_pct")) >= max_pre_entry_tp1_progress_pct
    )
    rr = _tp2_rr(row)
    gates = [
        _condition_gate(
            "FLAT_DIRECTION", "Directional candidate", directional, direction,
            ["long", "short"], stage=stage,
            explanation="Candidate is directional" if directional else "Candidate is flat/non-directional",
        ),
        _minimum_gate(
            "OVERALL_QUALITY_BELOW_MINIMUM", "Overall Quality",
            row.get("confidence"), min_confidence, stage=stage,
        ),
        _minimum_gate(
            "RANK_BELOW_MINIMUM", "Rank Score", row.get("rank_score"),
            min_rank, stage=stage,
        ),
        _minimum_gate(
            "EXECUTION_QUALITY_BELOW_MINIMUM", "Execution Quality",
            row.get("execution_quality", row.get("execution_score")),
            min_execution_quality, stage=stage, missing_passes=True,
        ),
        _maximum_gate(
            "IMMEDIATE_SL_RISK_TOO_HIGH", "Immediate-SL risk",
            row.get("immediate_sl_risk"), max_immediate_sl_risk,
            stage=stage, missing_passes=True,
        ),
        _maximum_gate(
            "PRICE_TOO_EXTENDED", "Chase distance (ATR)",
            row.get("chase_distance_atr"), max_chase_distance_atr,
            stage=stage, missing_passes=True,
        ),
        _condition_gate(
            "TP1_ALREADY_PROGRESSING", "Pre-entry TP1 progress",
            not progress_blocked, row.get("tp1_progress_pct"),
            {"operator": "<", "value": max_pre_entry_tp1_progress_pct},
            stage=stage,
            explanation="TP1 progress is acceptable" if not progress_blocked else "Too much of the TP1 move occurred before entry",
        ),
        _maximum_gate(
            "SPREAD_TOO_WIDE", "Spread (bps)", row.get("spread_bps"),
            max_spread_bps, stage=stage, missing_passes=True,
        ),
        _maximum_gate(
            "STALE_TICKER", "Ticker age (seconds)",
            row.get("ticker_age_seconds"), max_ticker_age_seconds,
            stage=stage, severity="diagnostic", authoritative=False,
            missing_passes=True,
        ),
        _maximum_gate(
            "STALE_ORDER_BOOK", "Order-book age (seconds)",
            row.get("orderbook_age_seconds"), max_orderbook_age_seconds,
            stage=stage, severity="diagnostic", authoritative=False,
            missing_passes=True,
        ),
        _condition_gate(
            "MARKET_QUALITY_FAILED", "Market quality", market_ok,
            row.get("market_quality_ok"), True, stage=stage,
            explanation="Market quality passed" if market_ok else "Market quality failed",
        ),
        _condition_gate(
            "DATA_QUALITY_FAILED", "Data quality", data_ok,
            row.get("data_quality_ok"), True, stage=stage,
            explanation="Data quality passed" if data_ok else "Data quality failed",
        ),
        _condition_gate(
            "HISTORICAL_EDGE_FAILED", "Diagnostic historical edge",
            historical_ok, row.get("historical_edge_ok"), True, stage=stage,
            explanation="Historical diagnostic passed" if historical_ok else "Historical diagnostic failed",
        ),
        _minimum_gate(
            "GROSS_RR_BELOW_MINIMUM", "TP2 gross R:R", rr,
            min_tp2_rr, stage=stage,
            # Existing code permits missing R:R rather than inventing a failure.
            authoritative=rr is not None,
            severity="hard" if rr is not None else "diagnostic",
            missing_passes=True,
        ),
        _entry_gate(
            row.get("entry_status"), stage=stage, missing_passes=True
        ),
        _condition_gate(
            "PROP_COMPATIBILITY_FAILED", "Prop compatibility", prop_ok,
            row.get("prop_safe"), True, stage=stage,
            explanation="Prop compatibility passed" if prop_ok else "Prop compatibility failed",
        ),
    ]
    prior_eval = GateEvaluation.from_dict(prior) if prior else None
    prior_eligible = (
        prior_eval.eligible
        if prior_eval is not None
        else row.get("signal_eligible") is not False
    )
    if prior_eval is not None:
        # Preserve the actual stage order while avoiding a second copy of the
        # same unchanged gate. A stricter downstream threshold (for example
        # Overall Quality 80 after the analysis floor) remains a distinct gate.
        current_keys = {(gate.code, repr(gate.required_value)) for gate in gates}
        gates = [
            *[
                gate
                for gate in prior_eval.gates
                if (gate.code, repr(gate.required_value)) not in current_keys
            ],
            *gates,
        ]
    elif row.get("signal_eligible") is False:
        gates.insert(
            0,
            _condition_gate(
                "ANALYSIS_ERROR", "Upstream analysis eligibility", False,
                False, True, stage="analysis",
                explanation="Upstream analysis rejected the candidate without canonical detail",
            ),
        )
    return GateEvaluation(stage=stage, eligible=False, gates=gates).finalize(
        prior_eligible=prior_eligible
    )


_PRE_SEND_CODE_MAP = {
    "PRE_SEND_LEVELS_INVALID": "REVALIDATION_FAILED",
    "PRE_SEND_ENTRY_EXPIRED": "ENTRY_EXPIRED",
    "PRE_SEND_EXPIRY_INVALID": "REVALIDATION_FAILED",
    "PRE_SEND_TICKER_STALE": "STALE_TICKER",
    "PRE_SEND_ORDERBOOK_STALE": "STALE_ORDER_BOOK",
    "PRE_SEND_SPREAD_UNAVAILABLE": "MARKET_QUALITY_FAILED",
    "PRE_SEND_SPREAD_TOO_WIDE": "SPREAD_TOO_WIDE",
    "PRE_SEND_CANDLES_STALE": "DATA_QUALITY_FAILED",
    "PRE_SEND_STOP_ALREADY_TRADED": "INVALIDATED_BEFORE_ENTRY",
    "PRE_SEND_STRUCTURE_INVALIDATED": "STRUCTURE_CONFLICT",
    "PRE_SEND_TP1_ALREADY_TRADED": "TP1_ALREADY_PROGRESSING",
    "PRE_SEND_ENTRY_MOVE_MOSTLY_MISSED": "TP1_ALREADY_PROGRESSING",
}


def evaluate_revalidation_gates(
    legacy_reasons: Sequence[str],
    *,
    prior: Optional[Mapping[str, Any]] = None,
) -> GateEvaluation:
    stage = "pre_delivery_revalidation"
    prior_eval = GateEvaluation.from_dict(prior) if prior else None
    gates = list(prior_eval.gates) if prior_eval else []
    for reason in legacy_reasons:
        raw = str(reason)
        base = raw.split(":", 1)[0]
        code = _PRE_SEND_CODE_MAP.get(
            base,
            "MARKET_UNAVAILABLE" if "DATA_FAILURE" in base else "REVALIDATION_FAILED",
        )
        gates.append(
            _condition_gate(
                code, "Pre-delivery revalidation", False, raw, "absent",
                stage=stage,
                explanation=raw.replace("PRE_SEND_", "").replace("_", " ").lower(),
            )
        )
    return GateEvaluation(stage=stage, eligible=False, gates=gates).finalize(
        prior_eligible=prior_eval.eligible if prior_eval else True
    )


def candidate_analytics_snapshot(
    row: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    scan_id: str,
    candidate_id: str,
    analyzed_at: str,
) -> Dict[str, Any]:
    execution_components = dict(row.get("execution_components") or {})
    targets = list(row.get("take_profits") or [])
    target_feasibility = list(row.get("target_feasibility") or [])
    direction = str(
        row.get("evaluated_direction") or row.get("direction") or "flat"
    ).lower()
    if direction not in {"long", "short"}:
        direction = "flat"
    return {
        "scan_id": scan_id,
        "candidate_id": candidate_id,
        "symbol": row.get("symbol"),
        "exchange_id": row.get("exchange"),
        "timeframe": row.get("primary_tf"),
        "direction": direction,
        "setup_type": row.get("execution_setup_type") or row.get("setup_name"),
        "analyzed_at": analyzed_at,
        "feature_schema_version": row.get("feature_schema_version") or "3.0",
        "execution_policy_version": row.get("execution_policy_version"),
        "rank_policy_version": row.get("rank_policy_version"),
        "gate_policy_version": GATE_POLICY_VERSION,
        "technical_quality": _number(row.get("technical_confidence")),
        "execution_quality": _number(row.get("execution_quality", row.get("execution_score"))),
        "overall_quality": _number(row.get("confidence")),
        "rank_score": _number(row.get("rank_score")),
        "immediate_sl_risk": _number(row.get("immediate_sl_risk")),
        "gross_rr": list(row.get("gross_risk_reward") or []),
        "net_rr": list(row.get("net_risk_reward") or []),
        "spread_bps": _number(row.get("spread_bps")),
        "ticker_age_seconds": _number(row.get("ticker_age_seconds")),
        "orderbook_age_seconds": _number(row.get("orderbook_age_seconds")),
        "prop_safe": bool(row.get("prop_safe")),
        "entry_state": row.get("entry_status"),
        "target_count": len(targets),
        "target_feasibility": target_feasibility,
        "target_feasibility_summary": (
            min(target_feasibility) if target_feasibility else None
        ),
        "stop_quality": _number(
            row.get("stop_quality", execution_components.get("stop_quality"))
        ),
        "data_quality_score": _number(row.get("data_quality_score")),
        "market_quality_ok": row.get("market_quality_ok") is not False,
        "chase_distance_atr": _number(row.get("chase_distance_atr")),
        "confluence_magnitude": abs(_number(row.get("confluence_score")) or 0.0),
        "lifecycle_state": row.get("lifecycle_state") or "not_published",
        "gate_evaluation": dict(evaluation),
        "primary_rejection_reason": evaluation.get("primary_rejection_reason"),
        "all_rejection_reasons": list(evaluation.get("all_rejection_reasons") or []),
        "closest_to_passing_gate": evaluation.get("closest_to_passing_gate"),
        "distance_to_eligibility": _number(evaluation.get("distance_to_eligibility")) or 0.0,
        "proximity_label": evaluation.get("proximity_label"),
        "eligible": bool(evaluation.get("eligible")),
    }


def detect_suspicious_gate_behavior(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_directional_sample: int = 10,
) -> List[Dict[str, Any]]:
    directional = [
        row for row in rows if str(row.get("direction") or "").lower() in {"long", "short"}
    ]
    if len(directional) < max(1, minimum_directional_sample):
        return []
    alerts: List[Dict[str, Any]] = []
    primary = Counter(str(row.get("primary_rejection_reason") or "") for row in directional)
    for code, count in primary.items():
        if code and count / len(directional) > 0.80:
            alerts.append(
                {
                    "code": "DOMINANT_GATE",
                    "gate": code,
                    "count": count,
                    "sample": len(directional),
                    "percentage": round(count / len(directional) * 100.0, 1),
                }
            )
    all_reason_sets = [set(row.get("all_rejection_reasons") or []) for row in directional]
    if all_reason_sets:
        common = set.intersection(*all_reason_sets)
        for code in sorted(common):
            alerts.append(
                {"code": "EVERY_DIRECTIONAL_FAILED_GATE", "gate": code, "sample": len(directional)}
            )
    if all(str(row.get("entry_state") or "") in {"blocked", "avoid_chase"} for row in directional):
        alerts.append({"code": "ALL_ENTRIES_BLOCKED", "sample": len(directional)})
    if all(int(row.get("target_count") or 0) == 0 for row in directional):
        alerts.append({"code": "ALL_TARGETS_INFEASIBLE", "sample": len(directional)})
    if all(row.get("prop_safe") is False for row in directional):
        alerts.append({"code": "ALL_PROP_INCOMPATIBLE", "sample": len(directional)})
    for metric in (
        "execution_quality",
        "overall_quality",
        "immediate_sl_risk",
        "spread_bps",
    ):
        values = [_number(row.get(metric)) for row in directional]
        present = [value for value in values if value is not None]
        if not present:
            alerts.append(
                {"code": "METRIC_ALWAYS_MISSING", "metric": metric, "sample": len(directional)}
            )
        elif len({round(value, 8) for value in present}) == 1 and len(
            {str(row.get("symbol") or "") for row in directional}
        ) >= 3:
            alerts.append(
                {
                    "code": "METRIC_CONSTANT_ACROSS_ASSETS",
                    "metric": metric,
                    "value": present[0],
                    "sample": len(present),
                }
            )
    sides = {
        side: [
            _number(row.get("overall_quality"))
            for row in directional
            if str(row.get("direction") or "").lower() == side
        ]
        for side in ("long", "short")
    }
    long_values = [value for value in sides["long"] if value is not None]
    short_values = [value for value in sides["short"] if value is not None]
    if len(long_values) >= 5 and len(short_values) >= 5:
        gap = abs(median(long_values) - median(short_values))
        if gap >= 30.0:
            alerts.append(
                {"code": "LONG_SHORT_SCORE_ASYMMETRY", "gap": round(gap, 2), "sample": len(directional)}
            )
    return alerts


def aggregate_rejection_rows(
    scans: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    suspicious_minimum_sample: int = 10,
) -> Dict[str, Any]:
    primary = Counter(str(row.get("primary_rejection_reason") or "ELIGIBLE") for row in candidates)
    all_reasons: Counter[str] = Counter()
    setup = Counter(str(row.get("setup_type") or "unknown") for row in candidates)
    symbol = Counter(str(row.get("symbol") or "unknown") for row in candidates)
    direction = Counter(str(row.get("direction") or "flat") for row in candidates)
    timeframe = Counter(str(row.get("timeframe") or "unknown") for row in candidates)
    primary_by_setup: Dict[str, Counter[str]] = defaultdict(Counter)
    primary_by_symbol: Dict[str, Counter[str]] = defaultdict(Counter)
    primary_by_direction: Dict[str, Counter[str]] = defaultdict(Counter)
    primary_by_timeframe: Dict[str, Counter[str]] = defaultdict(Counter)
    failures_by_stage: Counter[str] = Counter()
    for row in candidates:
        all_reasons.update(str(code) for code in row.get("all_rejection_reasons") or [])
        code = str(row.get("primary_rejection_reason") or "ELIGIBLE")
        primary_by_setup[str(row.get("setup_type") or "unknown")][code] += 1
        primary_by_symbol[str(row.get("symbol") or "unknown")][code] += 1
        primary_by_direction[str(row.get("direction") or "flat")][code] += 1
        primary_by_timeframe[str(row.get("timeframe") or "unknown")][code] += 1
        for gate in (row.get("gate_evaluation") or {}).get("gates") or []:
            if not gate.get("passed"):
                failures_by_stage[str(gate.get("stage") or "unknown")] += 1
    rejected = [row for row in candidates if not row.get("eligible")]
    nearest = sorted(
        rejected,
        key=lambda row: (
            0 if row.get("proximity_label") == "NEAR_PASS" else 1,
            float(row.get("distance_to_eligibility") or 999),
        ),
    )[:10]
    score_fields = (
        "technical_quality", "execution_quality", "overall_quality", "rank_score"
    )
    distributions: Dict[str, Any] = {}
    for field_name in score_fields:
        values = [
            value
            for value in (_number(row.get(field_name)) for row in candidates)
            if value is not None
        ]
        distributions[field_name] = {
            "count": len(values),
            "minimum": min(values) if values else None,
            "median": median(values) if values else None,
            "maximum": max(values) if values else None,
        }
    suspicious = detect_suspicious_gate_behavior(
        candidates, minimum_directional_sample=suspicious_minimum_sample
    )
    if len(scans) >= 3 and all(int(row.get("directional_candidates") or 0) == 0 for row in scans):
        suspicious.append(
            {"code": "REPEATED_ZERO_DIRECTIONAL_CANDIDATES", "sample": len(scans)}
        )
    return {
        "scans_completed": len(scans),
        "candidates_analyzed": len(candidates),
        "directional_candidates": sum(
            1 for row in candidates if str(row.get("direction") or "").lower() in {"long", "short"}
        ),
        "eligible_candidates": sum(1 for row in candidates if row.get("eligible")),
        "failed_one_gate": sum(1 for row in rejected if int((row.get("gate_evaluation") or {}).get("failed_hard_gates") or 0) == 1),
        "failed_multiple_gates": sum(1 for row in rejected if int((row.get("gate_evaluation") or {}).get("failed_hard_gates") or 0) > 1),
        "primary_rejection_counts": dict(primary),
        "all_rejection_counts": dict(all_reasons),
        "primary_rejection_percentages": {
            code: round(count / max(len(candidates), 1) * 100.0, 1)
            for code, count in primary.items()
        },
        "setup_distribution": dict(setup),
        "symbol_distribution": dict(symbol),
        "direction_distribution": dict(direction),
        "timeframe_distribution": dict(timeframe),
        "primary_rejections_by_setup": {key: dict(value) for key, value in primary_by_setup.items()},
        "primary_rejections_by_symbol": {key: dict(value) for key, value in primary_by_symbol.items()},
        "primary_rejections_by_direction": {key: dict(value) for key, value in primary_by_direction.items()},
        "primary_rejections_by_timeframe": {key: dict(value) for key, value in primary_by_timeframe.items()},
        "failed_gates_by_stage": dict(failures_by_stage),
        "score_distributions": distributions,
        "closest_rejected_candidates": nearest,
        "suspicious_diagnostics": suspicious,
    }
