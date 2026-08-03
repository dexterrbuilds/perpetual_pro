"""Focused tests for deduplicated private-beta qualification."""

from __future__ import annotations

from copy import deepcopy

from src.analysis.qualification import (
    EXCLUDED_QUALIFICATION_CHECKS,
    QUALIFICATION_POLICY_VERSION,
    build_deduplicated_check_registry,
    build_private_beta_rejection_summary,
    evaluate_private_beta_qualification,
    private_beta_threshold_passes,
    qualify_private_beta_candidates,
)
from src.analytics.rejection import evaluate_revalidation_gates
from src.notify.telegram import format_prop_scan_report, format_signal_photo_caption
from src.scheduler.scan_job import filter_high_confidence
from src.scoring.features import build_candidate_record


def _gate(
    code: str,
    passed: bool,
    actual,
    required,
    *,
    stage: str = "alert_filter",
    normalized: float = 0.0,
    authoritative: bool = True,
    severity: str = "hard",
):
    return {
        "code": code,
        "gate_name": code.replace("_", " ").title(),
        "actual_value": actual,
        "required_value": required,
        "passed": passed,
        "distance": 0.0 if passed else normalized,
        "normalized_distance": normalized,
        "severity": severity,
        "stage": stage,
        "explanation": "passed" if passed else f"{code} failed",
        "authoritative": authoritative,
    }


def _row() -> dict:
    gates = [
        _gate("FLAT_DIRECTION", True, "long", ["long", "short"], stage="analysis"),
        _gate("OVERALL_QUALITY_BELOW_MINIMUM", True, 84, {"operator": ">=", "value": 68}, stage="analysis"),
        _gate("OVERALL_QUALITY_BELOW_MINIMUM", True, 84, {"operator": ">=", "value": 78}),
        _gate("EXECUTION_QUALITY_BELOW_MINIMUM", True, 79, {"operator": ">=", "value": 72}, stage="analysis"),
        _gate("EXECUTION_QUALITY_BELOW_MINIMUM", True, 79, {"operator": ">=", "value": 72}),
        _gate("CONFLUENCE_BELOW_MINIMUM", True, 0.4, {"operator": ">=", "value": 0.2}, stage="analysis"),
        _gate("RANK_BELOW_MINIMUM", True, 74, {"operator": ">=", "value": 50}),
        _gate("IMMEDIATE_SL_RISK_TOO_HIGH", True, 20, {"operator": "<=", "value": 32}),
        _gate("GROSS_RR_BELOW_MINIMUM", True, 1.5, {"operator": ">=", "value": 1.25}),
        _gate("NET_RR_BELOW_MINIMUM", True, 1.35, {"operator": ">=", "value": 1.25}),
        _gate("DATA_QUALITY_FAILED", True, True, True),
        _gate("MARKET_QUALITY_FAILED", True, True, True),
        _gate("STALE_TICKER", True, 2, {"operator": "<=", "value": 45}, authoritative=False, severity="diagnostic"),
        _gate("STALE_ORDER_BOOK", True, 3, {"operator": "<=", "value": 30}, authoritative=False, severity="diagnostic"),
        _gate("SPREAD_TOO_WIDE", True, 2, {"operator": "<=", "value": 12}),
        _gate("ENTRY_BLOCKED", True, "wait_retest", ["confirmation_pending", "wait_retest"]),
        _gate("TP1_ALREADY_PROGRESSING", True, 10, {"operator": "<", "value": 70}),
    ]
    return {
        "candidate_id": "candidate-1",
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "evaluated_direction": "long",
        "execution_setup_type": "retest_continuation",
        "confidence": 84,
        "technical_confidence": 84,
        "execution_quality": 79,
        "execution_score": 79,
        "rank_score": 74,
        "entry_status": "wait_retest",
        "entry_low": 100,
        "entry_high": 101,
        "stop_loss": 98,
        "take_profits": [103, 105],
        "gate_evaluation": {"gates": gates},
        "payload": {"execution": {}, "primary_setup": {}},
    }


def _set_gate(row: dict, code: str, passed: bool, actual=None) -> None:
    for gate in row["gate_evaluation"]["gates"]:
        if gate["code"] == code:
            gate["passed"] = passed
            if actual is not None:
                gate["actual_value"] = actual
            gate["normalized_distance"] = 0.1 if not passed else 0.0
            gate["explanation"] = "passed" if passed else f"{code} failed"


def test_duplicate_checks_count_once_and_revalidation_does_not_inflate_count():
    row = _row()
    checks = build_deduplicated_check_registry(row)
    ids = [check["canonical_check_id"] for check in checks]
    assert ids.count("overall_quality_floor") == 1
    assert ids.count("execution_quality_floor") == 1

    prior = {"eligible": True, "gates": row["gate_evaluation"]["gates"]}
    revalidated = evaluate_revalidation_gates(
        ["PRE_SEND_SPREAD_TOO_WIDE"], prior=prior
    ).to_dict()
    row["gate_evaluation"] = revalidated
    row["pre_delivery_revalidation_ok"] = False
    checks = build_deduplicated_check_registry(row)
    assert sum(check["duplicate_group"] == "spread_liquidity" for check in checks) == 1
    assert sum(check["duplicate_group"] == "final_revalidation" for check in checks) == 1


def test_nonapplicable_and_excluded_checks_do_not_enter_denominators():
    row = _row()
    row["gate_evaluation"]["gates"].extend(
        [
            _gate("REVERSAL_CONFIRMATION_INSUFFICIENT", True, True, True),
            _gate("PROP_COMPATIBILITY_FAILED", False, False, True, authoritative=False, severity="advisory"),
            _gate("LLM_UNAVAILABLE_NON_BLOCKING", False, "unavailable", "available", authoritative=False, severity="advisory"),
            _gate("HISTORICAL_EDGE_FAILED", False, False, True, authoritative=False, severity="diagnostic"),
        ]
    )
    qualification = evaluate_private_beta_qualification(row)
    participating_ids = {
        item["canonical_check_id"]
        for item in [
            *qualification["applicable_hard_checks"],
            *qualification["applicable_soft_checks"],
        ]
    }
    assert "reversal_confirmation" not in participating_ids
    assert "prop_compatibility" not in participating_ids
    assert "llm_narrative" not in participating_ids
    assert "quick_backtest_diagnostic" not in participating_ids
    assert "portfolio_routing_preference" in EXCLUDED_QUALIFICATION_CHECKS


def test_hard_and_mandatory_failures_always_reject():
    hard = _row()
    _set_gate(hard, "SPREAD_TOO_WIDE", False, 13)
    assert evaluate_private_beta_qualification(hard)["private_beta_qualified"] is False

    overall = _row()
    _set_gate(overall, "OVERALL_QUALITY_BELOW_MINIMUM", False, 77.9)
    assert evaluate_private_beta_qualification(overall)["overall_quality_passed"] is False
    assert evaluate_private_beta_qualification(overall)["private_beta_qualified"] is False

    execution = _row()
    _set_gate(execution, "EXECUTION_QUALITY_BELOW_MINIMUM", False, 71.9)
    assert evaluate_private_beta_qualification(execution)["execution_quality_passed"] is False
    assert evaluate_private_beta_qualification(execution)["private_beta_qualified"] is False

    cost_blocked = _row()
    cost_blocked["gate_evaluation"]["gates"].append(
        _gate(
            "NET_RR_BELOW_MINIMUM",
            False,
            "tp1_reward_does_not_cover_cost_buffer",
            "absent",
            stage="analysis",
            authoritative=False,
            severity="diagnostic",
        )
    )
    cost_result = evaluate_private_beta_qualification(cost_blocked)
    assert any(
        check["canonical_check_id"] == "target_validity"
        for check in cost_result["failed_hard_checks"]
    )
    assert cost_result["private_beta_qualified"] is False


def test_soft_threshold_is_inclusive_and_zero_denominator_is_safe():
    assert private_beta_threshold_passes(
        hard_passed=True,
        overall_quality_passed=True,
        execution_quality_passed=True,
        soft_pass_percentage=70.0,
    ) is True
    assert private_beta_threshold_passes(
        hard_passed=True,
        overall_quality_passed=True,
        execution_quality_passed=True,
        soft_pass_percentage=70.1,
    ) is True
    assert private_beta_threshold_passes(
        hard_passed=True,
        overall_quality_passed=True,
        execution_quality_passed=True,
        soft_pass_percentage=69.9,
    ) is False

    row = _row()
    soft_codes = {
        "CONFLUENCE_BELOW_MINIMUM",
        "RANK_BELOW_MINIMUM",
        "IMMEDIATE_SL_RISK_TOO_HIGH",
        "GROSS_RR_BELOW_MINIMUM",
        "NET_RR_BELOW_MINIMUM",
    }
    row["gate_evaluation"]["gates"] = [
        gate for gate in row["gate_evaluation"]["gates"]
        if gate["code"] not in soft_codes
    ]
    qualification = evaluate_private_beta_qualification(row)
    assert qualification["soft_applicable_count"] == 0
    assert qualification["soft_pass_percentage"] == 100.0
    assert qualification["private_beta_qualified"] is True


def test_beta_can_accept_one_soft_failure_while_public_policy_is_unchanged():
    row = _row()
    _set_gate(row, "RANK_BELOW_MINIMUM", False, 49)
    qualification = evaluate_private_beta_qualification(row)
    assert qualification["soft_pass_percentage"] == 80.0
    assert qualification["qualification_type"] == "qualified_beta"
    assert qualify_private_beta_candidates([row])[0]["symbol"] == "BTC/USDT:USDT"

    public_row = {
        **row,
        "signal_eligible": True,
        "prop_safe": True,
        "immediate_sl_risk": 20,
        "chase_distance_atr": 0.2,
        "spread_bps": 2,
        "gross_risk_reward": [1.0, 1.5],
        "net_risk_reward": [0.9, 1.35],
    }
    assert filter_high_confidence(
        [public_row], min_llm=65, min_rank=50, only_prop_safe=False,
        min_execution_score=72,
    ) == []


def test_private_beta_delivery_keeps_only_best_two_by_policy_order():
    lower = _row()
    lower["candidate_id"] = "lower"
    lower["symbol"] = "ETH"
    lower["confidence"] = 82.0

    middle = _row()
    middle["candidate_id"] = "middle"
    middle["symbol"] = "SOL"
    middle["confidence"] = 86.0

    best = _row()
    best["candidate_id"] = "best"
    best["symbol"] = "BTC"
    best["confidence"] = 90.0

    selected = qualify_private_beta_candidates([lower, best, middle], limit=2)
    assert [row["candidate_id"] for row in selected] == ["best", "middle"]


def test_private_summary_persistence_and_telegram_use_same_result():
    highest = _row()
    highest["candidate_id"] = "hard-high"
    highest["symbol"] = "TIA"
    highest["confidence"] = 84.2
    _set_gate(highest, "ENTRY_BLOCKED", False, "blocked")

    closest = _row()
    closest["candidate_id"] = "soft-close"
    closest["symbol"] = "SOL"
    closest["confidence"] = 82.0
    _set_gate(closest, "RANK_BELOW_MINIMUM", False, 49)
    _set_gate(closest, "GROSS_RR_BELOW_MINIMUM", False, 1.2)
    # Two of five soft failures => 60%, below beta qualification.
    summary = build_private_beta_rejection_summary({}, [highest, closest])
    assert summary["highest_quality_rejected_candidate"]["candidate_id"] == "hard-high"
    assert summary["closest_to_full_qualification"]["candidate_id"] == "soft-close"

    beta = _row()
    _set_gate(beta, "RANK_BELOW_MINIMUM", False, 49)
    qualification = evaluate_private_beta_qualification(beta)
    beta["qualification"] = qualification
    beta["qualification_policy_version"] = QUALIFICATION_POLICY_VERSION
    beta["gate_evaluation"]["private_beta_qualification"] = qualification
    record = build_candidate_record(None, beta, source="beta-test")
    assert record["decision"]["qualification"] == qualification
    assert record["decision"]["qualification_policy_version"] == QUALIFICATION_POLICY_VERSION

    caption = format_signal_photo_caption(beta)
    assert "QUALIFIED BETA SIGNAL" in caption
    assert "Hard checks:" in caption
    assert "Soft checks: 4/5 passed · 80%" in caption
    assert "Deterministic Rank" in caption
    assert len(caption) <= 1024

    report = format_prop_scan_report(
        [], scanned_count=2, ranked_count=2, rejection_summary=summary
    )
    assert "Highest-Quality Rejected Setup" in report
    assert "Closest to Qualification" in report
    assert "Hard checks:" in report
    assert "Soft checks:" in report
