"""Focused tests for deduplicated private-beta qualification."""

from __future__ import annotations

from copy import deepcopy

from src.analysis.qualification import (
    EXCLUDED_QUALIFICATION_CHECKS,
    QUALIFICATION_POLICY_VERSION,
    analyze_private_beta_net_rr_floors,
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
        "authoritative_rank": 74,
        "rank_available": True,
        "gross_risk_reward": [1.0, 1.5],
        "net_risk_reward": [0.9, 1.35],
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


def _high_quality_override_row(chase_distance_atr: float = 1.20) -> dict:
    row = _row()
    row["confidence"] = 82.0
    row["overall_quality"] = 82.0
    row["execution_quality"] = 82.0
    row["execution_score"] = 82.0
    row["chase_distance_atr"] = chase_distance_atr
    _set_gate(row, "OVERALL_QUALITY_BELOW_MINIMUM", True, 82.0)
    _set_gate(row, "EXECUTION_QUALITY_BELOW_MINIMUM", True, 82.0)
    row["gate_evaluation"]["gates"].append(
        _gate(
            "PRICE_TOO_EXTENDED",
            False,
            chase_distance_atr,
            {"operator": "<=", "value": 1.0},
        )
    )
    return row


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


def test_derived_blocked_entry_is_not_counted_beside_original_hard_failure():
    row = _row()
    _set_gate(row, "ENTRY_BLOCKED", False, "blocked")
    original = _gate(
        "TP1_BLOCKED",
        False,
        "tp1_blocked_by_nearby_structure",
        "absent",
        stage="analysis",
        authoritative=False,
        severity="diagnostic",
    )
    original["gate_name"] = "Execution hard-failure detail"
    row["gate_evaluation"]["gates"].append(original)

    qualification = evaluate_private_beta_qualification(row)
    failed_ids = {
        check["canonical_check_id"]
        for check in qualification["failed_hard_checks"]
    }
    assert failed_ids == {"target_validity"}
    assert qualification["private_beta_qualified"] is False

    standalone = _row()
    _set_gate(standalone, "ENTRY_BLOCKED", False, "blocked")
    standalone_result = evaluate_private_beta_qualification(standalone)
    assert {
        check["canonical_check_id"]
        for check in standalone_result["failed_hard_checks"]
    } == {"entry_state"}


def test_private_beta_overall_floor_can_be_77_without_changing_public(monkeypatch):
    row = _row()
    row["confidence"] = 77.5
    row["overall_quality"] = 77.5
    _set_gate(row, "OVERALL_QUALITY_BELOW_MINIMUM", False, 77.5)

    monkeypatch.delenv("PRIVATE_BETA_MIN_OVERALL_QUALITY", raising=False)
    default_result = evaluate_private_beta_qualification(row)
    assert default_result["private_beta_min_overall_quality"] == 78.0
    assert default_result["overall_quality_passed"] is False

    monkeypatch.setenv("PRIVATE_BETA_MIN_OVERALL_QUALITY", "77")
    trial_result = evaluate_private_beta_qualification(row)
    assert trial_result["private_beta_min_overall_quality"] == 77.0
    assert ":overall_77:" in trial_result["qualification_policy_variant"]
    assert trial_result["overall_quality_passed"] is True
    assert trial_result["private_beta_qualified"] is True

    public_row = {
        **row,
        "signal_eligible": True,
        "prop_safe": True,
        "immediate_sl_risk": 20,
        "chase_distance_atr": 0.2,
        "spread_bps": 2,
    }
    assert filter_high_confidence(
        [public_row],
        min_llm=65,
        min_rank=50,
        only_prop_safe=False,
        min_execution_score=72,
    ) == []


def test_invalid_private_beta_overall_override_fails_safe_to_78(monkeypatch):
    row = _row()
    row["confidence"] = 77.5
    _set_gate(row, "OVERALL_QUALITY_BELOW_MINIMUM", False, 77.5)
    monkeypatch.setenv("PRIVATE_BETA_MIN_OVERALL_QUALITY", "60")
    result = evaluate_private_beta_qualification(row)
    assert result["private_beta_min_overall_quality"] == 78.0
    assert result["private_beta_qualified"] is False


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
    overall["confidence"] = 77.9
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


def test_high_quality_override_allows_one_moderate_entry_extension_only():
    row = _high_quality_override_row()
    result = evaluate_private_beta_qualification(row)
    assert result["private_beta_qualified"] is True
    assert result["base_private_beta_qualified"] is False
    assert result["high_quality_override_applied"] is True
    assert result["qualification_type"] == "high_quality_override"
    assert result["immediate_sl_override_guard_passed"] is True
    assert [
        check["canonical_check_id"] for check in result["failed_hard_checks"]
    ] == ["entry_extension"]
    assert result["high_quality_override_check"] == {
        "eligible": True,
        "canonical_check_id": "entry_extension",
        "actual_chase_distance_atr": 1.2,
        "normal_max_chase_distance_atr": 1.0,
        "absolute_max_chase_distance_atr": 1.35,
        "reason": (
            "Moderate entry extension passed the existing absolute execution boundary"
        ),
    }


def test_high_quality_override_never_allows_extreme_or_critical_failures():
    extreme = _high_quality_override_row(1.36)
    extreme_result = evaluate_private_beta_qualification(extreme)
    assert extreme_result["high_quality_override_applied"] is False
    assert extreme_result["private_beta_qualified"] is False

    stale = _high_quality_override_row()
    _set_gate(stale, "DATA_QUALITY_FAILED", False, False)
    stale_result = evaluate_private_beta_qualification(stale)
    assert stale_result["high_quality_override_applied"] is False
    assert stale_result["private_beta_qualified"] is False

    wide_stop = _high_quality_override_row()
    wide_stop["gate_evaluation"]["gates"].append(
        _gate("STOP_TOO_WIDE", False, 3.2, {"operator": "<=", "value": 3.0})
    )
    stop_result = evaluate_private_beta_qualification(wide_stop)
    assert stop_result["high_quality_override_applied"] is False
    assert stop_result["private_beta_qualified"] is False


def test_high_quality_override_critical_failure_allowlist_is_fail_closed():
    critical_failures = (
        ("INVALIDATED_BEFORE_ENTRY", "invalidated", "valid"),
        ("TP1_ALREADY_PROGRESSING", 100.0, {"operator": "<", "value": 70}),
        ("DATA_QUALITY_FAILED", False, True),
        ("STALE_TICKER", 46.0, {"operator": "<=", "value": 45}),
        ("STALE_ORDER_BOOK", 31.0, {"operator": "<=", "value": 30}),
        ("MARKET_UNAVAILABLE", "unsupported", "supported"),
        ("REVALIDATION_FAILED", False, True),
    )
    for code, actual, required in critical_failures:
        row = _row()
        row["confidence"] = 82.0
        row["execution_quality"] = 82.0
        row["execution_score"] = 82.0
        _set_gate(row, "OVERALL_QUALITY_BELOW_MINIMUM", True, 82.0)
        _set_gate(row, "EXECUTION_QUALITY_BELOW_MINIMUM", True, 82.0)
        row["gate_evaluation"]["gates"].append(
            _gate(code, False, actual, required)
        )
        result = evaluate_private_beta_qualification(row)
        assert result["high_quality_override_applied"] is False, code
        assert result["private_beta_qualified"] is False, code


def test_high_quality_override_preserves_scores_immediate_sl_and_net_rr_guards():
    low_overall = _high_quality_override_row()
    low_overall["confidence"] = 79.9
    _set_gate(low_overall, "OVERALL_QUALITY_BELOW_MINIMUM", True, 79.9)
    assert evaluate_private_beta_qualification(low_overall)[
        "private_beta_qualified"
    ] is False

    low_execution = _high_quality_override_row()
    low_execution["execution_quality"] = 79.9
    low_execution["execution_score"] = 79.9
    _set_gate(low_execution, "EXECUTION_QUALITY_BELOW_MINIMUM", True, 79.9)
    assert evaluate_private_beta_qualification(low_execution)[
        "private_beta_qualified"
    ] is False

    immediate_sl = _high_quality_override_row()
    _set_gate(immediate_sl, "IMMEDIATE_SL_RISK_TOO_HIGH", False, 33.0)
    immediate_result = evaluate_private_beta_qualification(immediate_sl)
    assert immediate_result["important_soft_pass_count"] == 2
    assert immediate_result["immediate_sl_override_guard_passed"] is False
    assert immediate_result["private_beta_qualified"] is False

    low_net_rr = _high_quality_override_row()
    low_net_rr["net_risk_reward"] = [0.5, 0.74]
    _set_gate(low_net_rr, "NET_RR_BELOW_MINIMUM", False, 0.74)
    low_rr_result = evaluate_private_beta_qualification(low_net_rr)
    assert low_rr_result["net_rr_floor_passed"] is False
    assert low_rr_result["private_beta_qualified"] is False


def test_high_quality_override_is_persisted_displayed_and_private_only():
    row = _high_quality_override_row()
    qualification = evaluate_private_beta_qualification(row)
    row["qualification"] = qualification
    row["qualification_policy_version"] = qualification[
        "qualification_policy_version"
    ]
    record = build_candidate_record(None, row, source="override-test")
    stored = record["decision"]["qualification"]
    assert stored["high_quality_override_applied"] is True
    assert stored["high_quality_override_check"][
        "canonical_check_id"
    ] == "entry_extension"

    caption = format_signal_photo_caption(row)
    assert "HIGH QUALITY OVERRIDE" in caption
    assert "Entry extension: 1.20 ATR" in caption
    assert "1.35" in caption
    assert len(caption) <= 1024

    public_row = {
        **row,
        "signal_eligible": True,
        "prop_safe": True,
        "immediate_sl_risk": 20,
        "spread_bps": 2,
    }
    assert filter_high_confidence(
        [public_row], min_llm=65, min_rank=50, only_prop_safe=False,
        min_execution_score=72,
    ) == []


def test_two_important_checks_are_required_and_missing_evidence_rejects():
    assert private_beta_threshold_passes(
        hard_passed=True,
        overall_quality_passed=True,
        execution_quality_passed=True,
        important_soft_passed=2,
        net_rr_floor_passed=True,
    ) is True
    assert private_beta_threshold_passes(
        hard_passed=True,
        overall_quality_passed=True,
        execution_quality_passed=True,
        important_soft_passed=3,
        net_rr_floor_passed=True,
    ) is True
    assert private_beta_threshold_passes(
        hard_passed=True,
        overall_quality_passed=True,
        execution_quality_passed=True,
        important_soft_passed=1,
        net_rr_floor_passed=True,
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
    assert qualification["important_soft_applicable_count"] == 0
    assert qualification["private_beta_qualified"] is False


def test_two_important_checks_qualify_while_public_policy_is_unchanged():
    row = _row()
    _set_gate(row, "NET_RR_BELOW_MINIMUM", False, 1.10)
    row["net_risk_reward"] = [0.8, 1.10]
    qualification = evaluate_private_beta_qualification(row)
    assert qualification["soft_pass_percentage"] == 80.0
    assert qualification["important_soft_pass_count"] == 2
    assert qualification["qualification_type"] == "qualified_beta"
    assert qualification["reduced_reward_efficiency"] is True
    assert qualify_private_beta_candidates([row])[0]["symbol"] == "BTC/USDT:USDT"

    public_row = {
        **row,
        "signal_eligible": True,
        "prop_safe": True,
        "immediate_sl_risk": 20,
        "chase_distance_atr": 0.2,
        "spread_bps": 2,
        "gross_risk_reward": [1.0, 1.5],
        "net_risk_reward": [0.8, 1.10],
    }
    assert filter_high_confidence(
        [public_row], min_llm=65, min_rank=50, only_prop_safe=False,
        min_execution_score=72,
    ) == []


def test_supporting_checks_cannot_compensate_for_one_important_pass():
    row = _row()
    _set_gate(row, "CONFLUENCE_BELOW_MINIMUM", False, 0.1)
    _set_gate(row, "IMMEDIATE_SL_RISK_TOO_HIGH", False, 40)
    qualification = evaluate_private_beta_qualification(row)
    assert qualification["important_soft_pass_count"] == 1
    assert qualification["supporting_soft_pass_count"] == 2
    assert qualification["private_beta_qualified"] is False


def test_all_three_important_checks_are_fully_qualified():
    qualification = evaluate_private_beta_qualification(_row())
    assert qualification["important_soft_pass_count"] == 3
    assert qualification["qualification_type"] == "fully_qualified"
    assert qualification["private_beta_qualified"] is True


def test_absolute_net_rr_floor_and_missing_rank_handling():
    poor_rr = _row()
    _set_gate(poor_rr, "NET_RR_BELOW_MINIMUM", False, 0.50)
    poor_rr["net_risk_reward"] = [0.4, 0.50]
    result = evaluate_private_beta_qualification(poor_rr)
    assert result["important_soft_pass_count"] == 2
    assert result["net_rr_floor_passed"] is False
    assert result["private_beta_qualified"] is False

    missing_rank = _row()
    missing_rank["rank_score"] = None
    missing_rank["authoritative_rank"] = None
    missing_rank["rank_available"] = False
    result = evaluate_private_beta_qualification(missing_rank)
    assert result["rank_available"] is False
    assert result["authoritative_rank"] is None
    rank_check = next(
        check for check in result["applicable_soft_checks"]
        if check["canonical_check_id"] == "deterministic_rank"
    )
    assert rank_check["actual_value"] is None
    assert rank_check["passed"] is False


def test_net_rr_floor_impact_compares_requested_values_without_changing_rows():
    low = _row()
    low["net_risk_reward"] = [0.6, 0.80]
    _set_gate(low, "NET_RR_BELOW_MINIMUM", False, 0.80)
    high = _row()
    high["candidate_id"] = "high-rr"
    high["net_risk_reward"] = [0.9, 1.10]
    _set_gate(high, "NET_RR_BELOW_MINIMUM", False, 1.10)
    impact = analyze_private_beta_net_rr_floors([low, high])
    assert impact["base_candidates_before_absolute_net_rr_floor"] == 2
    assert impact["qualifying_by_floor"] == {
        "0.50": 2,
        "0.75": 2,
        "1.00": 1,
    }
    assert impact["floor_comparison"]["0.75"]["newly_qualifying_setups"] == 2
    assert impact["floor_comparison"]["0.75"]["setups_still_rejected"] == 0
    assert impact["floor_comparison"]["0.75"]["all_qualifiers_pass_hard_checks"] is True
    assert impact["floor_comparison"]["1.00"]["average_overall_quality"] == 84.0


def test_private_beta_delivery_keeps_only_best_two_by_policy_order():
    lower = _row()
    lower["candidate_id"] = "lower"
    lower["symbol"] = "ETH"
    lower["confidence"] = 95.0
    lower["net_risk_reward"] = [0.8, 1.10]
    _set_gate(lower, "NET_RR_BELOW_MINIMUM", False, 1.10)

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


def test_private_beta_sort_prioritizes_net_rr_after_important_passes():
    higher_quality = _row()
    higher_quality["candidate_id"] = "higher-quality"
    higher_quality["confidence"] = 92.0
    higher_quality["net_risk_reward"] = [0.8, 1.30]

    balanced = _row()
    balanced["candidate_id"] = "balanced"
    balanced["confidence"] = 84.0
    balanced["net_risk_reward"] = [1.0, 1.70]

    selected = qualify_private_beta_candidates(
        [higher_quality, balanced],
        limit=2,
    )
    assert [row["candidate_id"] for row in selected] == [
        "balanced",
        "higher-quality",
    ]


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
    _set_gate(closest, "CONFLUENCE_BELOW_MINIMUM", False, 0.1)
    _set_gate(closest, "IMMEDIATE_SL_RISK_TOO_HIGH", False, 40)
    # Supporting checks pass, but only one important check passes.
    summary = build_private_beta_rejection_summary({}, [highest, closest])
    assert summary["highest_quality_rejected_candidate"]["candidate_id"] == "hard-high"
    assert summary["closest_to_full_qualification"]["candidate_id"] == "soft-close"

    beta = _row()
    beta["net_risk_reward"] = [0.7, 0.86]
    _set_gate(beta, "NET_RR_BELOW_MINIMUM", False, 0.86)
    qualification = evaluate_private_beta_qualification(beta)
    beta["qualification"] = qualification
    beta["qualification_policy_version"] = QUALIFICATION_POLICY_VERSION
    beta["gate_evaluation"]["private_beta_qualification"] = qualification
    record = build_candidate_record(None, beta, source="beta-test")
    assert record["decision"]["qualification"] == qualification
    assert record["decision"]["qualification_policy_version"] == QUALIFICATION_POLICY_VERSION
    assert record["decision"]["authoritative_rank"] == 74
    assert record["decision"]["rank_available"] is True
    assert record["production_scores"]["rank"] == 74
    assert record["production_rank"] == 74

    caption = format_signal_photo_caption(beta)
    assert "QUALIFIED BETA SIGNAL" in caption
    assert "Hard checks:" in caption
    assert "Important soft checks: 2/3" in caption
    assert "Reduced reward efficiency" in caption
    assert "Private-beta floor: 0.75R" in caption
    assert "Deterministic Rank" in caption
    assert len(caption) <= 1024

    report = format_prop_scan_report(
        [], scanned_count=2, ranked_count=2, rejection_summary=summary
    )
    assert "Highest-Quality Rejected Setup" in report
    assert "Closest to Qualification" in report
    assert "Hard checks:" in report
    assert "Important soft checks:" in report
