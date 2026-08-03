"""Deduplicated private-beta qualification built from existing gate results.

This module does not calculate indicators, scores, levels, or thresholds. It
classifies and deduplicates the canonical checks already emitted by the live
analysis, alert-filter, and pre-delivery revalidation paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


QUALIFICATION_POLICY_VERSION = "private_beta_important_soft_v1"
PRIVATE_BETA_MIN_IMPORTANT_SOFT_PASSES = 2
PRIVATE_BETA_NET_RR_FLOOR = 0.75
PRIVATE_BETA_PREFERRED_NET_RR = 1.25
IMPORTANT_SOFT_CHECK_IDS = frozenset(
    {"directional_confluence", "immediate_sl_risk", "net_rr"}
)
SUPPORTING_SOFT_CHECK_IDS = frozenset({"deterministic_rank", "gross_rr"})


@dataclass(frozen=True)
class CheckPolicy:
    canonical_id: str
    display_name: str
    classification: str
    duplicate_group: str
    sources: Tuple[str, ...]
    applicable_setup_types: Tuple[str, ...] = ("all",)
    participates_private_beta: bool = True
    qualification_role: str = "standard"
    may_improve_before_entry: bool = False
    soft_tier: Optional[str] = None


def _policy(
    canonical_id: str,
    display_name: str,
    classification: str,
    sources: Sequence[str],
    *,
    duplicate_group: Optional[str] = None,
    setup_types: Sequence[str] = ("all",),
    participates: bool = True,
    role: str = "standard",
    may_improve: bool = False,
    soft_tier: Optional[str] = None,
) -> CheckPolicy:
    return CheckPolicy(
        canonical_id=canonical_id,
        display_name=display_name,
        classification=classification,
        duplicate_group=duplicate_group or canonical_id,
        sources=tuple(sources),
        applicable_setup_types=tuple(setup_types),
        participates_private_beta=participates,
        qualification_role=role,
        may_improve_before_entry=may_improve,
        soft_tier=soft_tier,
    )


_ANALYSIS = "src/analytics/rejection.py:evaluate_analysis_gates"
_ALERT = "src/analytics/rejection.py:evaluate_alert_gates"
_EXECUTION = "src/analysis/execution.py:build_execution_profile"
_REVALIDATION = "src/analysis/revalidation.py:evaluate_pre_delivery_candidate"


# Registry entries correspond to canonical codes already produced by the live
# gate pipeline. Aliases sharing a duplicate_group represent one underlying
# risk and count once.
CHECK_REGISTRY: Dict[str, CheckPolicy] = {
    "FLAT_DIRECTION": _policy("direction_valid", "Valid LONG/SHORT direction", "hard", [_ANALYSIS, _ALERT]),
    "TECHNICAL_QUALITY_BELOW_MINIMUM": _policy(
        "technical_quality", "Technical Quality", "advisory", [_ANALYSIS],
        participates=False,
    ),
    "OVERALL_QUALITY_BELOW_MINIMUM": _policy(
        "overall_quality_floor", "Overall Quality minimum", "soft", [_ANALYSIS, _ALERT],
        role="mandatory_floor",
    ),
    "EXECUTION_QUALITY_BELOW_MINIMUM": _policy(
        "execution_quality_floor", "Execution Quality minimum", "soft", [_ANALYSIS, _ALERT],
        role="mandatory_floor",
    ),
    "CONFLUENCE_BELOW_MINIMUM": _policy(
        "directional_confluence", "Directional confluence", "soft", [_ANALYSIS],
        may_improve=True, soft_tier="important",
    ),
    "RANK_BELOW_MINIMUM": _policy(
        "deterministic_rank", "Deterministic Rank", "soft", [_ALERT],
        may_improve=True, soft_tier="supporting",
    ),
    "IMMEDIATE_SL_RISK_TOO_HIGH": _policy(
        "immediate_sl_risk", "Immediate-SL risk preference", "soft", [_ANALYSIS, _ALERT, _EXECUTION],
        may_improve=True, soft_tier="important",
    ),
    "GROSS_RR_BELOW_MINIMUM": _policy(
        "gross_rr", "Gross R:R", "soft", [_ALERT], may_improve=True,
        soft_tier="supporting",
    ),
    "NET_RR_BELOW_MINIMUM": _policy(
        "net_rr", "Net R:R after costs", "soft", [_ANALYSIS, _ALERT, _EXECUTION],
        may_improve=True, soft_tier="important",
    ),
    "DATA_QUALITY_FAILED": _policy("data_quality", "Finite, complete, fresh candle data", "hard", [_ANALYSIS, _ALERT, _REVALIDATION]),
    "MARKET_QUALITY_FAILED": _policy("market_quality", "Usable market and liquidity quality", "hard", [_ANALYSIS, _ALERT, _EXECUTION]),
    "STALE_TICKER": _policy("ticker_freshness", "Fresh ticker", "hard", [_ALERT, _REVALIDATION]),
    "STALE_ORDER_BOOK": _policy("orderbook_freshness", "Fresh order book", "hard", [_ALERT, _REVALIDATION]),
    "SPREAD_TOO_WIDE": _policy("spread_liquidity", "Usable spread", "hard", [_ALERT, _EXECUTION, _REVALIDATION]),
    "ENTRY_BLOCKED": _policy("entry_state", "Valid live entry state", "hard", [_ANALYSIS, _ALERT, _EXECUTION, _REVALIDATION]),
    "CONFIRMATION_PENDING": _policy(
        "entry_state", "Valid live entry state", "hard", [_ANALYSIS, _ALERT, _EXECUTION],
        duplicate_group="entry_state", setup_types=("cmp_confirmation", "breakout_continuation"),
    ),
    "ENTRY_EXPIRED": _policy("entry_expiry", "Entry not expired", "hard", [_ALERT, _REVALIDATION]),
    "INVALIDATED_BEFORE_ENTRY": _policy("entry_invalidation", "Entry thesis not invalidated", "hard", [_ALERT, _EXECUTION, _REVALIDATION]),
    "AVOID_CHASE": _policy("entry_extension", "Entry not chased or excessively extended", "hard", [_ALERT, _EXECUTION, _REVALIDATION]),
    "PRICE_TOO_EXTENDED": _policy(
        "entry_extension", "Entry not chased or excessively extended", "hard", [_ALERT, _EXECUTION, _REVALIDATION],
        duplicate_group="entry_extension",
    ),
    "TP1_ALREADY_PROGRESSING": _policy("pre_entry_tp1_progress", "TP1 move not already missed", "hard", [_ALERT, _EXECUTION, _REVALIDATION]),
    "NO_FEASIBLE_TARGET": _policy("target_validity", "At least one feasible target", "hard", [_EXECUTION, _ANALYSIS]),
    "TP1_BLOCKED": _policy(
        "target_validity", "At least one feasible target", "hard", [_EXECUTION, _ANALYSIS],
        duplicate_group="target_validity",
    ),
    "STOP_QUALITY_FAILED": _policy("stop_validity", "Valid structural stop", "hard", [_EXECUTION, _ANALYSIS]),
    "STOP_TOO_TIGHT": _policy(
        "stop_validity", "Valid structural stop", "hard", [_EXECUTION, _ANALYSIS],
        duplicate_group="stop_validity",
    ),
    "STOP_TOO_WIDE": _policy(
        "stop_validity", "Valid structural stop", "hard", [_EXECUTION, _ANALYSIS],
        duplicate_group="stop_validity",
    ),
    "RANGE_DURING_TREND_EXPANSION": _policy(
        "range_regime_alignment", "Range regime agreement", "hard", [_EXECUTION, _ANALYSIS],
        setup_types=("range_mean_reversion",),
    ),
    "REVERSAL_CONFIRMATION_INSUFFICIENT": _policy(
        "reversal_confirmation", "Reversal confirmation", "hard", [_EXECUTION, _ANALYSIS],
        setup_types=("reversal",),
    ),
    "STRUCTURE_CONFLICT": _policy("structure_direction_alignment", "Direction-consistent structure", "hard", [_EXECUTION, _ANALYSIS, _REVALIDATION]),
    "REVALIDATION_FAILED": _policy("final_revalidation", "Final pre-delivery revalidation", "hard", [_REVALIDATION]),
    "ANALYSIS_ERROR": _policy("analysis_integrity", "Complete deterministic analysis", "hard", [_ANALYSIS, _ALERT, _REVALIDATION]),
    "MARKET_UNAVAILABLE": _policy("supported_market", "Supported perpetual market", "hard", [_ANALYSIS, _REVALIDATION]),
    "PROP_COMPATIBILITY_FAILED": _policy(
        "prop_compatibility", "Prop compatibility", "prop_only", [_ANALYSIS, _ALERT],
        participates=False,
    ),
    "LLM_UNAVAILABLE_NON_BLOCKING": _policy(
        "llm_narrative", "LLM narrative availability", "advisory", [_ALERT],
        participates=False,
    ),
    "HISTORICAL_EDGE_FAILED": _policy(
        "quick_backtest_diagnostic", "Quick-backtest diagnostic", "operational", [_ALERT],
        participates=False,
    ),
}


EXCLUDED_QUALIFICATION_CHECKS = (
    "telegram_configuration",
    "delivery_success",
    "message_idempotency",
    "scheduler_state",
    "scan_budget",
    "candidate_truncation",
    "duplicate_message_prevention",
    "llm_confidence",
    "shadow_outcome_model",
    "quick_backtest_diagnostic",
    "prop_compatibility",
    "portfolio_routing_preference",
    "execution_component_ingredients_without_independent_thresholds",
)


def _setup_type(row: Mapping[str, Any]) -> str:
    return str(
        row.get("execution_setup_type")
        or row.get("setup_type")
        or (row.get("payload") or {}).get("execution", {}).get("setup_type")
        or "unknown"
    ).lower()


def _is_applicable(policy: CheckPolicy, row: Mapping[str, Any], passed: bool) -> bool:
    if "all" in policy.applicable_setup_types:
        return True
    # A recorded failure is always applicable; this avoids hiding a genuine
    # setup-classification conflict behind a stale or unexpected setup label.
    if not passed:
        return True
    return _setup_type(row) in policy.applicable_setup_types


def _result_from_gate(
    gate: Mapping[str, Any],
    policy: CheckPolicy,
) -> Dict[str, Any]:
    return {
        "canonical_check_id": policy.canonical_id,
        "display_name": policy.display_name,
        "classification": policy.classification,
        "applicable_setup_types": list(policy.applicable_setup_types),
        "actual_value": gate.get("actual_value"),
        "required_value": gate.get("required_value"),
        "passed": bool(gate.get("passed")),
        "failure_reason": "" if gate.get("passed") else str(gate.get("explanation") or "Check failed"),
        "source_files_functions": list(policy.sources),
        "source_gate_codes": [str(gate.get("code") or "")],
        "evaluation_stages": [str(gate.get("stage") or "unknown")],
        "duplicate_group": policy.duplicate_group,
        "participates_private_beta": policy.participates_private_beta,
        "qualification_role": policy.qualification_role,
        "may_improve_before_entry": policy.may_improve_before_entry,
        "soft_tier": policy.soft_tier,
    }


def build_deduplicated_check_registry(
    row: Mapping[str, Any],
    gate_evaluation: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return applicable canonical check results, merged by underlying risk."""
    evaluation = dict(gate_evaluation or row.get("gate_evaluation") or {})
    observations: Dict[str, List[Dict[str, Any]]] = {}
    for raw_gate in evaluation.get("gates") or []:
        gate = dict(raw_gate)
        code = str(gate.get("code") or "")
        policy = CHECK_REGISTRY.get(code)
        # The existing execution planner records this specific TP1 cost-buffer
        # condition in ``hard_failures`` and the rejection vocabulary aliases
        # it to NET_RR_BELOW_MINIMUM. Preserve that real hard authority without
        # turning an ordinary numeric net-R:R shortfall into a hard check.
        if (
            code == "NET_RR_BELOW_MINIMUM"
            and not gate.get("passed")
            and str(gate.get("actual_value") or "")
            == "tp1_reward_does_not_cover_cost_buffer"
        ):
            policy = _policy(
                "target_validity",
                "At least one feasible and cost-justified target",
                "hard",
                [_EXECUTION, _ANALYSIS],
                duplicate_group="target_validity",
            )
        if policy is None:
            if code.startswith("HARD_FAILURE_"):
                policy = _policy(
                    code.lower(),
                    code.replace("HARD_FAILURE_", "").replace("_", " ").title(),
                    "hard",
                    [_EXECUTION, _ANALYSIS],
                )
            else:
                # Unknown diagnostics remain visible but never gain new
                # qualification authority merely by being unregistered.
                policy = _policy(
                    code.lower() or "unknown_check",
                    code.replace("_", " ").title() or "Unknown check",
                    "advisory",
                    [str(gate.get("stage") or "unknown")],
                    participates=False,
                )
        passed = bool(gate.get("passed"))
        if not _is_applicable(policy, row, passed):
            continue
        observations.setdefault(policy.duplicate_group, []).append(
            _result_from_gate(gate, policy)
        )

    # A successful or failed force refresh is an existing final safety check.
    # It joins the same duplicate group as REVALIDATION_FAILED and therefore
    # cannot inflate the check count.
    if "pre_delivery_revalidation_ok" in row:
        policy = CHECK_REGISTRY["REVALIDATION_FAILED"]
        passed = bool(row.get("pre_delivery_revalidation_ok"))
        observations.setdefault(policy.duplicate_group, []).append(
            _result_from_gate(
                {
                    "code": "REVALIDATION_FAILED",
                    "actual_value": passed,
                    "required_value": True,
                    "passed": passed,
                    "stage": "pre_delivery_revalidation",
                    "explanation": "Final pre-delivery revalidation failed",
                },
                policy,
            )
        )

    deduplicated: List[Dict[str, Any]] = []
    for group, items in observations.items():
        failed = [item for item in items if not item["passed"]]
        selected = dict((failed or items)[-1])
        selected["passed"] = not failed
        selected["source_gate_codes"] = list(
            dict.fromkeys(code for item in items for code in item["source_gate_codes"])
        )
        selected["evaluation_stages"] = list(
            dict.fromkeys(stage for item in items for stage in item["evaluation_stages"])
        )
        selected["source_files_functions"] = list(
            dict.fromkeys(source for item in items for source in item["source_files_functions"])
        )
        if failed:
            selected["failure_reason"] = "; ".join(
                dict.fromkeys(
                    item["failure_reason"] for item in failed if item["failure_reason"]
                )
            )
        selected["duplicate_group"] = group
        deduplicated.append(selected)
    return sorted(deduplicated, key=lambda item: item["canonical_check_id"])


def _authoritative_net_rr(
    row: Mapping[str, Any],
    important_checks: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    values = list(row.get("net_risk_reward") or [])
    if not values:
        payload = dict(row.get("payload") or {})
        execution = dict(payload.get("execution") or {})
        setup = dict(payload.get("primary_setup") or payload.get("trade_plan") or {})
        values = list(
            execution.get("net_risk_reward")
            or setup.get("net_risk_reward")
            or []
        )
    if values:
        value = _numeric(values[1 if len(values) > 1 else 0])
        if value is not None:
            return value
    check = next(
        (
            item for item in important_checks
            if item.get("canonical_check_id") == "net_rr"
        ),
        None,
    )
    return _numeric(check.get("actual_value")) if check else None


def _authoritative_rank(
    row: Mapping[str, Any],
    supporting_checks: Sequence[Mapping[str, Any]],
) -> Tuple[bool, Optional[float]]:
    explicit_available = row.get("rank_available")
    raw = row.get("authoritative_rank")
    if raw is None and explicit_available is not False:
        raw = row.get("rank_score")
    value = _numeric(raw)
    if value is None:
        check = next(
            (
                item for item in supporting_checks
                if item.get("canonical_check_id") == "deterministic_rank"
            ),
            None,
        )
        value = _numeric(check.get("actual_value")) if check else None
    available = bool(value is not None and explicit_available is not False)
    return available, value if available else None


def evaluate_private_beta_qualification(
    row: Mapping[str, Any],
    gate_evaluation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    checks = build_deduplicated_check_registry(row, gate_evaluation)
    participating = [item for item in checks if item["participates_private_beta"]]
    mandatory = [item for item in participating if item["qualification_role"] == "mandatory_floor"]
    hard = [
        item for item in participating
        if item["classification"] == "hard" and item["qualification_role"] == "standard"
    ]
    soft = [
        item for item in participating
        if item["classification"] == "soft" and item["qualification_role"] == "standard"
    ]
    important = [
        item for item in soft
        if item["canonical_check_id"] in IMPORTANT_SOFT_CHECK_IDS
    ]
    supporting = [
        item for item in soft
        if item["canonical_check_id"] in SUPPORTING_SOFT_CHECK_IDS
    ]
    rank_available, authoritative_rank = _authoritative_rank(row, supporting)
    rank_check = next(
        (
            item for item in supporting
            if item["canonical_check_id"] == "deterministic_rank"
        ),
        None,
    )
    if rank_check is not None:
        required_raw = rank_check.get("required_value")
        required_rank = _numeric(
            required_raw.get("value")
            if isinstance(required_raw, Mapping)
            else required_raw
        )
        rank_check["actual_value"] = authoritative_rank
        rank_check["passed"] = bool(
            rank_available
            and required_rank is not None
            and authoritative_rank is not None
            and authoritative_rank >= required_rank
        )
        rank_check["failure_reason"] = (
            ""
            if rank_check["passed"]
            else (
                "Deterministic Rank is unavailable"
                if not rank_available
                else "Deterministic Rank is below the existing minimum"
            )
        )
    passed_hard = [item for item in hard if item["passed"]]
    failed_hard = [item for item in hard if not item["passed"]]
    passed_soft = [item for item in soft if item["passed"]]
    failed_soft = [item for item in soft if not item["passed"]]
    passed_important = [item for item in important if item["passed"]]
    failed_important = [item for item in important if not item["passed"]]
    passed_supporting = [item for item in supporting if item["passed"]]
    failed_supporting = [item for item in supporting if not item["passed"]]
    soft_pct = (
        len(passed_soft) / len(soft) * 100.0
        if soft
        else 100.0
    )
    hard_pct = (
        len(passed_hard) / len(hard) * 100.0
        if hard
        else 100.0
    )
    overall = next(
        (item for item in mandatory if item["canonical_check_id"] == "overall_quality_floor"),
        None,
    )
    execution = next(
        (item for item in mandatory if item["canonical_check_id"] == "execution_quality_floor"),
        None,
    )
    overall_passed = bool(overall and overall["passed"])
    execution_passed = bool(execution and execution["passed"])
    hard_passed = not failed_hard
    net_rr = _authoritative_net_rr(row, important)
    net_rr_floor_passed = bool(
        net_rr is not None and net_rr + 1e-9 >= PRIVATE_BETA_NET_RR_FLOOR
    )
    qualifies = private_beta_threshold_passes(
        hard_passed=hard_passed,
        overall_quality_passed=overall_passed,
        execution_quality_passed=execution_passed,
        important_soft_passed=len(passed_important),
        net_rr_floor_passed=net_rr_floor_passed,
    )
    if (
        qualifies
        and len(important) >= 3
        and len(passed_important) == len(important)
    ):
        qualification_type = "fully_qualified"
    elif qualifies:
        qualification_type = "qualified_beta"
    else:
        qualification_type = "rejected"
    return {
        "qualification_policy_version": QUALIFICATION_POLICY_VERSION,
        "qualification_type": qualification_type,
        "private_beta_qualified": qualifies,
        "applicable_hard_checks": hard,
        "passed_hard_checks": passed_hard,
        "failed_hard_checks": failed_hard,
        "applicable_soft_checks": soft,
        "passed_soft_checks": passed_soft,
        "failed_soft_checks": failed_soft,
        "applicable_important_soft_checks": important,
        "passed_important_soft_checks": passed_important,
        "failed_important_soft_checks": failed_important,
        "important_soft_applicable_count": len(important),
        "important_soft_pass_count": len(passed_important),
        "important_soft_required_count": PRIVATE_BETA_MIN_IMPORTANT_SOFT_PASSES,
        "passed_supporting_soft_checks": passed_supporting,
        "failed_supporting_soft_checks": failed_supporting,
        "supporting_soft_applicable_count": len(supporting),
        "supporting_soft_pass_count": len(passed_supporting),
        "hard_pass_count": len(passed_hard),
        "hard_applicable_count": len(hard),
        "hard_pass_percentage": round(hard_pct, 1),
        "soft_pass_count": len(passed_soft),
        "soft_applicable_count": len(soft),
        "soft_pass_percentage": round(soft_pct, 1),
        "overall_quality_result": overall,
        "execution_quality_result": execution,
        "overall_quality_passed": overall_passed,
        "execution_quality_passed": execution_passed,
        "authoritative_rank": authoritative_rank,
        "rank_available": rank_available,
        "net_rr": net_rr,
        "preferred_net_rr": PRIVATE_BETA_PREFERRED_NET_RR,
        "private_beta_net_rr_floor": PRIVATE_BETA_NET_RR_FLOOR,
        "net_rr_floor_passed": net_rr_floor_passed,
        "reduced_reward_efficiency": bool(
            net_rr_floor_passed
            and net_rr is not None
            and net_rr < PRIVATE_BETA_PREFERRED_NET_RR
        ),
        "deduplicated_checks": checks,
        "excluded_checks": list(EXCLUDED_QUALIFICATION_CHECKS),
    }


def private_beta_threshold_passes(
    *,
    hard_passed: bool,
    overall_quality_passed: bool,
    execution_quality_passed: bool,
    important_soft_passed: int,
    net_rr_floor_passed: bool,
) -> bool:
    """Private-beta policy authority; public eligibility does not call this."""
    return bool(
        hard_passed
        and overall_quality_passed
        and execution_quality_passed
        and int(important_soft_passed) >= PRIVATE_BETA_MIN_IMPORTANT_SOFT_PASSES
        and net_rr_floor_passed
    )


def private_beta_sort_key(row: Mapping[str, Any]) -> tuple:
    qualification = dict(row.get("qualification") or {})
    rank = _numeric(qualification.get("authoritative_rank"))
    return (
        int(qualification.get("important_soft_pass_count") or 0),
        int(qualification.get("soft_pass_count") or 0),
        float(row.get("confidence") or row.get("overall_quality") or 0.0),
        float(row.get("execution_quality") or row.get("execution_score") or 0.0),
        rank if rank is not None else float("-inf"),
    )


def qualify_private_beta_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int = 2,
) -> List[Dict[str, Any]]:
    qualified: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        qualification = evaluate_private_beta_qualification(
            row, row.get("gate_evaluation")
        )
        row["qualification"] = qualification
        row["qualification_policy_version"] = qualification[
            "qualification_policy_version"
        ]
        evaluation = dict(row.get("gate_evaluation") or {})
        evaluation["private_beta_qualification"] = qualification
        row["gate_evaluation"] = evaluation
        payload = dict(row.get("payload") or {})
        payload["qualification"] = qualification
        row["payload"] = payload
        if qualification["private_beta_qualified"]:
            qualified.append(row)
    qualified.sort(key=private_beta_sort_key, reverse=True)
    return qualified[: max(0, int(limit))]


def analyze_private_beta_net_rr_floors(
    rows: Iterable[Mapping[str, Any]],
    *,
    floors: Sequence[float] = (0.50, 0.75, 1.00),
) -> Dict[str, Any]:
    """Diagnostic-only comparison using stored gates and no score mutation."""
    normalized_floors = tuple(sorted({round(float(value), 2) for value in floors}))
    counts = {f"{value:.2f}": 0 for value in normalized_floors}
    rejections = {f"{value:.2f}": 0 for value in normalized_floors}
    directional = 0
    base_candidates = 0
    net_values: List[float] = []
    for source in rows:
        row = dict(source)
        direction = str(row.get("direction") or "").lower()
        if direction not in {"long", "short"}:
            continue
        directional += 1
        row.setdefault("confidence", row.get("overall_quality"))
        row.setdefault("execution_quality", row.get("execution_score"))
        qualification = evaluate_private_beta_qualification(
            row, row.get("gate_evaluation")
        )
        base = bool(
            not qualification.get("failed_hard_checks")
            and qualification.get("overall_quality_passed")
            and qualification.get("execution_quality_passed")
            and int(qualification.get("important_soft_pass_count") or 0)
            >= PRIVATE_BETA_MIN_IMPORTANT_SOFT_PASSES
        )
        if not base:
            continue
        base_candidates += 1
        net_rr = _numeric(qualification.get("net_rr"))
        if net_rr is not None:
            net_values.append(net_rr)
        for floor in normalized_floors:
            key = f"{floor:.2f}"
            if net_rr is not None and net_rr + 1e-9 >= floor:
                counts[key] += 1
            else:
                rejections[key] += 1
    return {
        "directional_candidates": directional,
        "base_candidates_before_absolute_net_rr_floor": base_candidates,
        "qualifying_by_floor": counts,
        "rejected_by_floor": rejections,
        "observed_base_net_rr": [round(value, 4) for value in sorted(net_values)],
        "diagnostic_only": True,
    }


def build_private_beta_rejection_summary(
    base_summary: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Attach beta-specific highest and closest rejected candidates."""
    summary = dict(base_summary or {})
    evaluated: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        qualification = dict(row.get("qualification") or {})
        if not qualification:
            qualification = evaluate_private_beta_qualification(
                row, row.get("gate_evaluation")
            )
        row["qualification"] = qualification
        row["qualification_policy_version"] = qualification.get(
            "qualification_policy_version"
        )
        row["overall_quality"] = row.get("confidence", row.get("overall_quality"))
        row["execution_quality"] = row.get(
            "execution_quality", row.get("execution_score")
        )
        row["eligible"] = bool(qualification.get("private_beta_qualified"))
        evaluated.append(row)
    rejected = [
        row for row in evaluated
        if not row["eligible"]
        and str(row.get("direction") or "").lower() in {"long", "short"}
    ]
    highest = max(
        rejected,
        key=lambda row: (
            float(row.get("overall_quality") or 0.0),
            float(row.get("execution_quality") or 0.0),
        ),
        default=None,
    )
    comparable: List[Tuple[float, Dict[str, Any]]] = []
    for row in rejected:
        qualification = dict(row.get("qualification") or {})
        if qualification.get("failed_hard_checks"):
            continue
        gap = 0.0
        valid = True
        for key in ("overall_quality_result", "execution_quality_result"):
            result = qualification.get(key)
            if not isinstance(result, Mapping):
                valid = False
                break
            if not result.get("passed"):
                actual = _numeric(result.get("actual_value"))
                required_raw = result.get("required_value")
                required = _numeric(
                    required_raw.get("value")
                    if isinstance(required_raw, Mapping)
                    else required_raw
                )
                if actual is None or required is None:
                    valid = False
                    break
                gap += max(0.0, required - actual) / max(abs(required), 1.0)
        if not valid:
            continue
        important_passed = int(
            qualification.get("important_soft_pass_count") or 0
        )
        gap += max(
            0,
            PRIVATE_BETA_MIN_IMPORTANT_SOFT_PASSES - important_passed,
        ) / max(1, len(IMPORTANT_SOFT_CHECK_IDS))
        if not qualification.get("net_rr_floor_passed"):
            net_rr = _numeric(qualification.get("net_rr"))
            if net_rr is None:
                continue
            gap += max(0.0, PRIVATE_BETA_NET_RR_FLOOR - net_rr) / max(
                PRIVATE_BETA_NET_RR_FLOOR,
                1.0,
            )
        comparable.append((gap, row))
    closest = None
    if comparable:
        gap, row = min(
            comparable,
            key=lambda item: (
                item[0],
                -float(item[1].get("overall_quality") or 0.0),
            ),
        )
        closest = {**row, "normalized_qualification_gap": round(gap, 6)}
    summary.update(
        {
            "directional_candidates": len(evaluated),
            "eligible_candidates": sum(1 for row in evaluated if row["eligible"]),
            "highest_quality_rejected_candidate": highest,
            "closest_to_full_qualification": closest,
            "closest_heading": "Closest to Qualification",
            "highest_is_closest": bool(
                highest
                and closest
                and _row_identity(highest) == _row_identity(closest)
            ),
            "qualification_policy_version": QUALIFICATION_POLICY_VERSION,
        }
    )
    return summary


def _numeric(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _row_identity(row: Mapping[str, Any]) -> tuple:
    candidate_id = str(row.get("candidate_id") or "").strip()
    if candidate_id:
        return (candidate_id,)
    return (
        str(row.get("symbol") or ""),
        str(row.get("direction") or "").lower(),
        row.get("overall_quality"),
    )
