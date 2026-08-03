from __future__ import annotations

from pathlib import Path
import sys

import pytest
import requests
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.rejection import (
    GATE_POLICY_VERSION,
    aggregate_rejection_rows,
    candidate_analytics_snapshot,
    detect_suspicious_gate_behavior,
    evaluate_alert_gates,
    evaluate_analysis_gates,
    evaluate_revalidation_gates,
)
from src.analytics.repository import RejectionAnalyticsRepository
from src.analysis.llm import NarrativeLLM
from src.notify.telegram import format_prop_scan_report
from src.notify import telegram_bot
from src.scheduler.scan_job import filter_high_confidence
from src.utils.config import load_config


def _analysis(**overrides):
    values = {
        "direction": "long",
        "technical_quality": 84,
        "execution_quality": 79,
        "overall_quality": 84,
        "confluence_magnitude": 0.4,
        "entry_status": "wait_retest",
        "immediate_sl_risk": 20,
        "market_quality_ok": True,
        "data_quality_ok": True,
        "prop_safe": True,
        "confidence_floor": 68,
        "score_floor": 0.2,
        "execution_floor": 72,
        "max_immediate_sl_risk": 32,
    }
    values.update(overrides)
    return evaluate_analysis_gates(**values)


def _alert_row(**overrides):
    row = {
        "symbol": "BTC/USDT:USDT",
        "primary_tf": "15m",
        "evaluated_direction": "long",
        "direction": "long",
        "confidence": 84,
        "technical_confidence": 84,
        "execution_quality": 79,
        "execution_score": 79,
        "rank_score": 78,
        "immediate_sl_risk": 20,
        "chase_distance_atr": 0.5,
        "tp1_progress_pct": 10,
        "entry_zone_relation": "adverse_side",
        "spread_bps": 2,
        "market_quality_ok": True,
        "data_quality_ok": True,
        "historical_edge_ok": True,
        "gross_risk_reward": [0.8, 1.5],
        "entry_status": "wait_retest",
        "prop_safe": True,
        "signal_eligible": True,
    }
    row.update(overrides)
    return row


def _alert(row, prior=None):
    return evaluate_alert_gates(
        row,
        min_rank=50,
        only_prop_safe=True,
        min_confidence=80,
        min_execution_quality=72,
        max_immediate_sl_risk=32,
        max_chase_distance_atr=1,
        max_pre_entry_tp1_progress_pct=70,
        min_tp2_rr=1.25,
        max_spread_bps=12,
        prior=prior,
    )


def test_canonical_gate_result_records_first_and_all_blockers_with_shortfalls():
    decision = _analysis(
        overall_quality=67,
        execution_quality=70,
        immediate_sl_risk=35,
    )
    assert decision.eligible is False
    assert decision.primary_rejection_reason == "OVERALL_QUALITY_BELOW_MINIMUM"
    assert {
        "OVERALL_QUALITY_BELOW_MINIMUM",
        "EXECUTION_QUALITY_BELOW_MINIMUM",
        "IMMEDIATE_SL_RISK_TOO_HIGH",
    }.issubset(decision.all_rejection_reasons)
    overall = next(gate for gate in decision.gates if gate.code == "OVERALL_QUALITY_BELOW_MINIMUM")
    assert overall.distance == 1
    assert decision.would_still_fail_without_primary is True
    assert decision.failed_hard_gates == 3


def test_prop_assessment_is_advisory_not_signal_authority():
    decision = _analysis(prop_safe=False, net_rr=1.5)
    assert decision.eligible is True
    assert decision.universal_eligible is True
    assert decision.production_qualified is True
    prop_gate = next(
        gate for gate in decision.gates
        if gate.code == "PROP_COMPATIBILITY_FAILED"
    )
    assert prop_gate.passed is False
    assert prop_gate.authoritative is False
    assert prop_gate.severity == "advisory"

    alert = _alert(
        _alert_row(prop_safe=False, net_risk_reward=[0.8, 1.4]),
        prior=decision.to_dict(),
    )
    assert alert.eligible is True


def test_quality_thresholds_remain_production_gates_after_prop_separation():
    overall = _analysis(overall_quality=67, prop_safe=False, net_rr=1.5)
    execution = _analysis(execution_quality=71.9, prop_safe=False, net_rr=1.5)
    assert overall.eligible is False
    assert overall.universal_eligible is True
    assert overall.production_qualified is False
    assert "OVERALL_QUALITY_BELOW_MINIMUM" in overall.all_rejection_reasons
    assert execution.eligible is False
    assert execution.universal_eligible is True
    assert execution.production_qualified is False
    assert "EXECUTION_QUALITY_BELOW_MINIMUM" in execution.all_rejection_reasons

    net_rr = _analysis(prop_safe=False, net_rr=1.24)
    assert net_rr.eligible is False
    assert net_rr.universal_eligible is False
    assert "NET_RR_BELOW_MINIMUM" in net_rr.all_rejection_reasons


def test_flat_candidate_is_non_directional_and_not_a_near_trade():
    decision = _analysis(direction="flat")
    assert decision.primary_rejection_reason == "FLAT_DIRECTION"
    assert decision.proximity_label == "NON_DIRECTIONAL"


def test_flat_candidate_stays_first_blocker_after_alert_stage_merge():
    row = _alert_row(direction="flat", evaluated_direction="flat")
    prior = _analysis(direction="flat", overall_quality=40).to_dict()
    decision = _alert(row, prior=prior)
    snapshot = candidate_analytics_snapshot(
        row,
        decision.to_dict(),
        scan_id="scan-flat",
        candidate_id="candidate-flat",
        analyzed_at="2026-08-02T00:00:00Z",
    )
    assert decision.primary_rejection_reason == "FLAT_DIRECTION"
    assert decision.proximity_label == "NON_DIRECTIONAL"
    assert snapshot["direction"] == "flat"


def test_rejected_directional_candidate_keeps_pre_gate_direction_for_analytics():
    row = _alert_row(direction="flat", evaluated_direction="short", confidence=60)
    decision = _alert(row)
    snapshot = candidate_analytics_snapshot(
        row,
        decision.to_dict(),
        scan_id="scan-short",
        candidate_id="candidate-short",
        analyzed_at="2026-08-02T00:00:00Z",
    )
    assert snapshot["direction"] == "short"
    assert decision.primary_rejection_reason == "OVERALL_QUALITY_BELOW_MINIMUM"


def test_flat_rows_are_counted_but_excluded_from_nearest_candidates():
    directional = {
        "candidate_id": "directional",
        "symbol": "BTC",
        "direction": "long",
        "eligible": False,
        "distance_to_eligibility": 0.1,
        "proximity_label": "NEAR_PASS",
        "primary_rejection_reason": "OVERALL_QUALITY_BELOW_MINIMUM",
        "all_rejection_reasons": ["OVERALL_QUALITY_BELOW_MINIMUM"],
        "gate_evaluation": {"failed_hard_gates": 1, "gates": []},
    }
    flat = {
        **directional,
        "candidate_id": "flat",
        "symbol": "ETH",
        "direction": "flat",
        "distance_to_eligibility": 0.0,
        "proximity_label": "NON_DIRECTIONAL",
        "primary_rejection_reason": "FLAT_DIRECTION",
        "all_rejection_reasons": ["FLAT_DIRECTION"],
    }
    summary = aggregate_rejection_rows([], [flat, directional])
    assert summary["directional_candidates"] == 1
    assert summary["direction_distribution"]["flat"] == 1
    assert [row["candidate_id"] for row in summary["closest_rejected_candidates"]] == [
        "directional"
    ]


@pytest.mark.parametrize("direction", ["long", "short"])
def test_gate_evaluation_is_long_short_symmetric(direction):
    decision = _analysis(direction=direction)
    assert decision.eligible is True
    assert decision.all_rejection_reasons == []


def test_alert_filter_preserves_existing_pass_and_reject_behavior():
    passing = _alert_row()
    rejected = _alert_row(symbol="SOL/USDT:USDT", execution_quality=71, execution_score=71)
    result = filter_high_confidence(
        [passing, rejected], min_llm=99, min_rank=50, only_prop_safe=True,
        min_execution_score=72,
    )
    assert [row["symbol"] for row in result] == ["BTC/USDT:USDT"]
    assert rejected["gate_evaluation"]["primary_rejection_reason"] == "EXECUTION_QUALITY_BELOW_MINIMUM"


def test_same_gate_is_not_double_counted_across_analysis_and_alert_stages():
    prior = _analysis(execution_quality=70).to_dict()
    decision = _alert(_alert_row(execution_quality=70), prior=prior)
    failed_execution = [
        gate for gate in decision.gates
        if not gate.passed and gate.code == "EXECUTION_QUALITY_BELOW_MINIMUM"
    ]
    assert len(failed_execution) == 1
    assert decision.failed_hard_gates == 1
    assert decision.would_still_fail_without_primary is False


def test_alert_filter_legacy_missing_optional_fields_remains_compatible():
    row = {"direction": "long", "confidence": 80, "rank_score": 50, "prop_safe": True}
    assert filter_high_confidence([row], min_llm=65, min_rank=50, only_prop_safe=True) == [row]


def test_revalidation_appends_stable_codes_without_promoting_prior_reject():
    prior = _analysis().to_dict()
    decision = evaluate_revalidation_gates(
        ["PRE_SEND_TICKER_STALE", "PRE_SEND_SPREAD_TOO_WIDE"], prior=prior
    )
    assert decision.primary_rejection_reason == "STALE_TICKER"
    assert "SPREAD_TOO_WIDE" in decision.all_rejection_reasons
    rejected_prior = _analysis(overall_quality=60).to_dict()
    assert evaluate_revalidation_gates([], prior=rejected_prior).eligible is False


def test_proximity_and_aggregation_support_time_range_diagnostics():
    near = _alert(_alert_row(confidence=79.5)).to_dict()
    far = _alert(_alert_row(symbol="SOL", confidence=60, execution_quality=50)).to_dict()
    rows = []
    for index, (symbol, evaluation) in enumerate((("BTC", near), ("SOL", far))):
        rows.append(
            {
                "candidate_id": str(index), "scan_id": "scan-1", "symbol": symbol,
                "direction": "long", "timeframe": "15m", "setup_type": "retest",
                "overall_quality": 79.5 if symbol == "BTC" else 60,
                "execution_quality": 79 if symbol == "BTC" else 50,
                "technical_quality": 84, "rank_score": 70,
                "eligible": False, "primary_rejection_reason": evaluation["primary_rejection_reason"],
                "all_rejection_reasons": evaluation["all_rejection_reasons"],
                "distance_to_eligibility": evaluation["distance_to_eligibility"],
                "proximity_label": evaluation["proximity_label"],
                "gate_evaluation": evaluation,
            }
        )
    summary = aggregate_rejection_rows([{"scan_id": "scan-1"}], rows)
    assert summary["scans_completed"] == 1
    assert summary["failed_one_gate"] == 1
    assert summary["failed_multiple_gates"] == 1
    assert summary["closest_rejected_candidates"][0]["symbol"] == "BTC"
    assert summary["primary_rejections_by_symbol"]["BTC"]["OVERALL_QUALITY_BELOW_MINIMUM"] == 1


def test_suspicious_detection_obeys_minimum_sample():
    rows = [
        {
            "symbol": f"S{i}", "direction": "long", "primary_rejection_reason": "ENTRY_BLOCKED",
            "all_rejection_reasons": ["ENTRY_BLOCKED"], "entry_state": "blocked",
            "target_count": 1, "prop_safe": True, "execution_quality": 70 + i,
            "overall_quality": 75 + i, "immediate_sl_risk": 20 + i, "spread_bps": 2 + i,
        }
        for i in range(9)
    ]
    assert detect_suspicious_gate_behavior(rows, minimum_directional_sample=10) == []
    rows.append({**rows[-1], "symbol": "S9"})
    alerts = detect_suspicious_gate_behavior(rows, minimum_directional_sample=10)
    assert any(item["code"] == "DOMINANT_GATE" for item in alerts)
    assert any(item["code"] == "ALL_ENTRIES_BLOCKED" for item in alerts)


def test_candidate_snapshot_persists_policy_and_build_inputs():
    row = _alert_row(
        exchange="okx", execution_policy_version="execution_quality_v2a.1",
        rank_policy_version="deterministic_rank_v2a.1", feature_schema_version="3.0",
        take_profits=[110, 115], target_feasibility=[80, 70],
    )
    evaluation = _alert(row).to_dict()
    snapshot = candidate_analytics_snapshot(
        row, evaluation, scan_id="scan-1", candidate_id="candidate-1",
        analyzed_at="2026-08-02T00:00:00Z",
    )
    assert snapshot["gate_policy_version"] == GATE_POLICY_VERSION
    assert snapshot["feature_schema_version"] == "3.0"
    assert snapshot["execution_policy_version"] == "execution_quality_v2a.1"
    assert snapshot["rank_policy_version"] == "deterministic_rank_v2a.1"


def test_manual_no_quality_report_is_explicitly_non_actionable():
    evaluation = _alert(_alert_row(confidence=79.4)).to_dict()
    summary = aggregate_rejection_rows([], [{
        "symbol": "BTC", "direction": "long", "eligible": False,
        "overall_quality": 79.4, "execution_quality": 78,
        "primary_rejection_reason": evaluation["primary_rejection_reason"],
        "all_rejection_reasons": evaluation["all_rejection_reasons"],
        "closest_to_passing_gate": evaluation["closest_to_passing_gate"],
        "distance_to_eligibility": evaluation["distance_to_eligibility"],
        "proximity_label": evaluation["proximity_label"],
        "gate_evaluation": evaluation,
    }])
    report = format_prop_scan_report(
        [], scanned_count=21, ranked_count=1, rejection_summary=summary
    )
    assert "Highest-Quality Rejected Setup" in report
    assert "No rules were relaxed" in report


def test_highest_quality_and_closest_qualification_are_distinct_and_authoritative():
    highest = {
        "candidate_id": "high-hard",
        "symbol": "TIA",
        "direction": "long",
        "eligible": False,
        "overall_quality": 84.2,
        "execution_quality": 82.0,
        "primary_rejection_reason": "ENTRY_BLOCKED",
        "all_rejection_reasons": ["ENTRY_BLOCKED"],
        "distance_to_eligibility": 0.01,
        "proximity_label": "NEAR_PASS",
        "gate_evaluation": {
            "failed_hard_gates": 1,
            "gates": [{
                "code": "ENTRY_BLOCKED",
                "passed": False,
                "actual_value": "blocked",
                "required_value": ["confirmation_pending", "wait_retest"],
                "normalized_distance": 0.01,
                "severity": "hard",
                "authoritative": True,
            }],
        },
    }
    closest = {
        "candidate_id": "lower-numeric",
        "symbol": "SOL",
        "direction": "short",
        "eligible": False,
        "overall_quality": 65.3,
        "payload": {"confidence": 99.0},
        "execution_quality": 74.0,
        "primary_rejection_reason": "OVERALL_QUALITY_BELOW_MINIMUM",
        "all_rejection_reasons": ["OVERALL_QUALITY_BELOW_MINIMUM"],
        "distance_to_eligibility": 0.18375,
        "proximity_label": "MODERATE_GAP",
        "gate_evaluation": {
            "failed_hard_gates": 1,
            "gates": [{
                "code": "OVERALL_QUALITY_BELOW_MINIMUM",
                "passed": False,
                "actual_value": 65.3,
                "required_value": {"operator": ">=", "value": 80.0},
                "normalized_distance": 0.18375,
                "severity": "hard",
                "authoritative": True,
            }],
        },
    }
    flat = {
        **closest,
        "candidate_id": "flat-high",
        "symbol": "BTC",
        "direction": "flat",
        "overall_quality": 99.0,
    }
    summary = aggregate_rejection_rows([], [flat, highest, closest])

    assert summary["highest_quality_rejected_candidate"]["candidate_id"] == "high-hard"
    assert summary["highest_quality_rejected_candidate"]["overall_quality"] == 84.2
    assert summary["closest_to_full_qualification"]["candidate_id"] == "lower-numeric"
    assert summary["closest_to_full_qualification"]["overall_quality"] == 65.3
    assert summary["highest_is_closest"] is False

    report = format_prop_scan_report(
        [], scanned_count=3, ranked_count=2, rejection_summary=summary
    )
    assert "<b>TIA LONG</b>" in report
    assert "Overall Quality 84.2/100" in report
    assert "<b>SOL SHORT</b>" in report
    assert "Closest to Full Qualification" in report
    assert "<b>BTC FLAT</b>" not in report
    assert report.index("<b>TIA LONG</b>") < report.index("<b>SOL SHORT</b>")


def test_same_candidate_is_not_duplicated_in_no_quality_report():
    evaluation = _alert(_alert_row(confidence=79.5)).to_dict()
    candidate = {
        "candidate_id": "same",
        "symbol": "BTC",
        "direction": "long",
        "eligible": False,
        "overall_quality": 79.5,
        "execution_quality": 79.0,
        "primary_rejection_reason": evaluation["primary_rejection_reason"],
        "all_rejection_reasons": evaluation["all_rejection_reasons"],
        "distance_to_eligibility": evaluation["distance_to_eligibility"],
        "proximity_label": evaluation["proximity_label"],
        "gate_evaluation": evaluation,
    }
    summary = aggregate_rejection_rows([], [candidate])
    report = format_prop_scan_report(
        [], scanned_count=1, ranked_count=1, rejection_summary=summary
    )
    assert summary["highest_is_closest"] is True
    assert report.count("<b>BTC LONG</b>") == 1
    assert report.count("REJECTED / NON-ACTIONABLE") == 1
    assert "both the highest-quality rejected setup and the closest" in report


def test_llm_rate_limit_is_observability_only(monkeypatch):
    cfg = load_config(ROOT / "config.yaml")
    llm = NarrativeLLM(cfg)
    llm.groq_key = "configured"
    llm.gemini_key = ""
    response = requests.Response()
    response.status_code = 429
    error = requests.HTTPError(response=response)
    monkeypatch.setattr(llm, "_call_groq", lambda _prompt: (_ for _ in ()).throw(error))
    narrative = llm.generate({"direction": "long"})
    assert narrative.provider == "local_fallback"
    assert narrative.rate_limit_events == 1
    assert narrative.provider_errors == ["groq_rate_limited"]


def test_analytics_persistence_failure_cannot_change_gate_result(monkeypatch):
    repository = RejectionAnalyticsRepository("postgresql://configured")
    monkeypatch.setattr(repository, "_connect", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    before = _alert(_alert_row()).eligible
    assert repository.record_scan({"scan_id": "x"}, []) is False
    assert _alert(_alert_row()).eligible is before is True


def test_private_rejections_command_is_authorized_and_concise(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("TELEGRAM_COMMAND_CHAT_IDS", "123")
    cfg = load_config(ROOT / "config.yaml")

    class Repository:
        def latest_scan(self):
            return {"analytics": {"candidates_analyzed": 21, "directional_candidates": 3, "eligible_candidates": 0, "primary_rejection_counts": {"OVERALL_QUALITY_BELOW_MINIMUM": 3}, "closest_rejected_candidates": []}}

        def summary_for_hours(self, _hours):
            return self.latest_scan()["analytics"]

    delivered = []
    monkeypatch.setattr(telegram_bot, "get_rejection_repository", lambda _cfg: Repository())
    monkeypatch.setattr(telegram_bot, "send_telegram_message_detailed", lambda text, **kwargs: delivered.append((text, kwargs)) or {"ok": True})
    result = telegram_bot.process_telegram_update(
        {"message": {"text": "/rejections", "chat": {"id": 123, "type": "private"}}}, cfg
    )
    assert result["ok"] is True
    assert "Rejection Summary" in delivered[0][0]
    assert delivered[0][1]["chat_id"] == "123"


def test_rejection_admin_endpoints_are_protected(monkeypatch):
    import main_server

    monkeypatch.setenv("SCAN_API_KEY", "analytics-secret")

    class Repository:
        def latest_scan(self):
            return {"scan": {"scan_id": "x"}, "analytics": {}, "candidates": []}

        def candidate(self, _candidate_id):
            return {"candidate_id": "c"}

    monkeypatch.setattr(main_server, "get_rejection_repository", lambda _cfg: Repository())
    client = TestClient(main_server.app)
    assert client.get("/admin/rejections/latest").status_code == 401
    response = client.get(
        "/admin/rejections/latest", headers={"X-Scan-API-Key": "analytics-secret"}
    )
    assert response.status_code == 200
    assert response.json()["scan"]["scan_id"] == "x"


def test_rejection_analytics_migration_is_additive_and_indexed():
    migration = (ROOT / "migrations" / "004_rejection_analytics.sql").read_text()
    assert "create table if not exists public.scan_rejection_analytics" in migration
    assert "create table if not exists public.candidate_rejection_analytics" in migration
    assert "candidate_rejection_codes_gin_idx" in migration
    assert "drop table" not in migration.lower()
    assert "delete from" not in migration.lower()
