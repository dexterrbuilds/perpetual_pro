from __future__ import annotations

import pytest

from src.analysis.legacy_v2 import (
    compare_legacy_v2_outcomes,
    execution_aware_legacy_confidence,
)
from src.analysis.execution_policy import deterministic_rank_score
from src.notify.telegram import format_prop_scan_report, format_signal_photo_caption
from src.scheduler.scan_job import filter_high_confidence
from src.scoring.features import build_candidate_record
from src.utils.config import load_config


def test_execution_caps_overall_confidence_without_changing_technical():
    comparison = execution_aware_legacy_confidence(
        technical_confidence=92,
        execution_score=69,
        immediate_sl_risk=20,
        data_quality_score=100,
        confidence_min=15,
        confidence_max=92,
        execution_confidence_buffer=5,
    )
    assert comparison.technical_confidence == 92
    assert comparison.legacy_confidence > 90
    assert comparison.execution_confidence_cap == 74
    assert comparison.legacy_v2_confidence == 74
    assert comparison.execution_confidence_penalty > 17


def test_clean_execution_never_boosts_confidence_above_technical():
    comparison = execution_aware_legacy_confidence(
        technical_confidence=82,
        execution_score=95,
        immediate_sl_risk=20,
        data_quality_score=100,
        confidence_min=15,
        confidence_max=92,
        execution_confidence_buffer=5,
    )
    assert comparison.legacy_v2_confidence == 82


def test_sl_and_data_quality_penalties_remain_enforced():
    clean = execution_aware_legacy_confidence(
        technical_confidence=88,
        execution_score=85,
        immediate_sl_risk=20,
        data_quality_score=100,
        confidence_min=15,
        confidence_max=92,
        execution_confidence_buffer=5,
    )
    risky = execution_aware_legacy_confidence(
        technical_confidence=88,
        execution_score=85,
        immediate_sl_risk=38,
        data_quality_score=65,
        confidence_min=15,
        confidence_max=92,
        execution_confidence_buffer=5,
    )
    assert risky.legacy_v2_confidence == pytest.approx(
        clean.legacy_v2_confidence - 3.2
    )


def test_immediate_sl_penalty_starts_only_above_hard_maximum():
    at_limit = execution_aware_legacy_confidence(
        technical_confidence=88,
        execution_score=85,
        immediate_sl_risk=32,
        data_quality_score=100,
        confidence_min=15,
        confidence_max=92,
        execution_confidence_buffer=5,
    )
    above_limit = execution_aware_legacy_confidence(
        technical_confidence=88,
        execution_score=85,
        immediate_sl_risk=33,
        data_quality_score=100,
        confidence_min=15,
        confidence_max=92,
        execution_confidence_buffer=5,
    )
    assert at_limit.immediate_sl_penalty == 0.0
    assert above_limit.immediate_sl_penalty == pytest.approx(0.2)
    assert above_limit.legacy_v2_confidence == pytest.approx(
        at_limit.legacy_v2_confidence - 0.2
    )


def test_historical_policy_comparison_counts_avoided_and_rejected_outcomes():
    rows = [
        {
            "id": "avoided_failure",
            "generated_at": "2026-01-01T00:00:00Z",
            "production_scores": {
                "legacy_confidence": 91,
                "legacy_v2_confidence": 74,
                "legacy_signal_eligible": True,
                "legacy_v2_signal_eligible": True,
            },
            "valid_fill": True,
            "alert_success": False,
            "tp2_hit": False,
            "realized_r": -1,
        },
        {
            "id": "shared_success",
            "generated_at": "2026-01-02T00:00:00Z",
            "production_scores": {
                "legacy_confidence": 86,
                "legacy_v2_confidence": 84,
                "legacy_signal_eligible": True,
                "legacy_v2_signal_eligible": True,
            },
            "valid_fill": True,
            "alert_success": True,
            "tp2_hit": True,
            "realized_r": 1.5,
        },
        {
            "id": "rejected_success",
            "generated_at": "2026-01-03T00:00:00Z",
            "production_scores": {
                "legacy_confidence": 82,
                "legacy_v2_confidence": 78,
                "legacy_signal_eligible": True,
                "legacy_v2_signal_eligible": True,
            },
            "valid_fill": True,
            "alert_success": True,
            "tp2_hit": False,
            "realized_r": 0.8,
        },
    ]
    result = compare_legacy_v2_outcomes(rows, alert_confidence_floor=80)
    assert result["legacy"]["selected_count"] == 3
    assert result["legacy_v2"]["selected_count"] == 1
    assert result["v2_newly_rejected"]["avoided_failures"] == 1
    assert result["v2_newly_rejected"]["rejected_successes"] == 1
    assert result["legacy_v2"]["expectancy_r"] > result["legacy"]["expectancy_r"]


def test_default_policy_uses_72_execution_floor():
    config = load_config()
    assert config.analysis.legacy_v2_enabled is True
    assert config.analysis.legacy_execution_min_score == 65
    assert config.analysis.execution_min_score == 72
    assert config.analysis.execution_confidence_buffer == 5


def test_legacy_v2_cap_penalties_and_telegram_use_one_final_overall_quality():
    """The displayed/ranked/eligible/stored value is the capped final value."""
    config = load_config()
    technical = 84.0
    execution = 79.0
    buffer = config.analysis.execution_confidence_buffer
    pre_penalty_cap = min(technical, execution + buffer)
    assert pre_penalty_cap == 84.0

    clean = execution_aware_legacy_confidence(
        technical_confidence=technical,
        execution_score=execution,
        immediate_sl_risk=20,
        data_quality_score=100,
        confidence_min=config.analysis.min_confidence,
        confidence_max=config.analysis.max_confidence,
        execution_confidence_buffer=buffer,
    )
    penalized = execution_aware_legacy_confidence(
        technical_confidence=technical,
        execution_score=execution,
        immediate_sl_risk=38,  # 1.2-point penalty above the 32 threshold
        data_quality_score=65,  # 2-point penalty below the 75 threshold
        confidence_min=config.analysis.min_confidence,
        confidence_max=config.analysis.max_confidence,
        execution_confidence_buffer=buffer,
    )
    assert clean.legacy_v2_confidence <= pre_penalty_cap
    assert clean.legacy_v2_confidence == 84.0
    assert penalized.immediate_sl_penalty == pytest.approx(1.2)
    assert penalized.data_quality_penalty == 2.0
    assert penalized.legacy_v2_confidence == pytest.approx(80.8)
    assert penalized.legacy_v2_confidence <= clean.legacy_v2_confidence
    assert penalized.legacy_v2_confidence <= pre_penalty_cap

    # Use the clean capped value for the eligibility/rank/Telegram path; the
    # assertions above separately prove that penalties only reduce it.
    final_quality = clean.legacy_v2_confidence
    rank, breakdown = deterministic_rank_score(
        overall_quality=final_quality,
        execution_quality=execution,
        target_feasibility=75,
        stop_quality=78,
        net_rr=1.5,
        market_data_quality=80,
        setup_validity=100,
    )
    assert breakdown["legacy_v2_overall_quality"] == final_quality * 0.25

    row = {
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "confidence": final_quality,
        "legacy_v2_confidence": final_quality,
        "technical_confidence": technical,
        "execution_score": execution,
        "execution_quality": execution,
        "rank_score": rank,
        "signal_eligible": True,
        "prop_safe": True,
        "entry_status": "wait_retest",
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 98.0,
        "take_profits": [103.0, 105.0],
        "risk_pct": 1.0,
        "historical_edge_ok": True,
        "data_quality_ok": True,
        "market_quality_ok": True,
        "immediate_sl_risk": 20.0,
        "chase_distance_atr": 0.4,
        "spread_bps": 2.0,
        "payload": {
            # A stale nested alias must not override the authoritative row.
            "confidence": 99.0,
            "execution": {"status": "wait_retest", "score": execution},
            "primary_setup": {"risk_reward": [1.2, 2.0]},
        },
    }
    delivered = filter_high_confidence(
        [dict(row)],
        min_llm=0,
        min_rank=0,
        only_prop_safe=True,
        min_confidence=80,
        min_execution_score=72,
        min_tp2_rr=1.25,
    )
    assert len(delivered) == 1
    assert delivered[0]["confidence"] == final_quality

    candidate = build_candidate_record(None, row, source="legacy_v2_cap_test")
    assert candidate["production_scores"]["confidence"] == final_quality
    assert candidate["production_scores"]["legacy_v2_confidence"] == final_quality
    assert candidate["production_scores"]["rank"] == rank

    photo = format_signal_photo_caption(row)
    report = format_prop_scan_report([row])
    expected_label = "Overall Quality 84/100"
    assert expected_label in photo
    assert expected_label in report
    assert "Overall Quality 99/100" not in photo
    assert "Calibrated TP1 probability" not in photo
