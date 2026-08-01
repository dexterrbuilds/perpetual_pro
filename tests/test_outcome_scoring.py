from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.scheduler.scan_job import filter_high_confidence
from src.scoring.features import (
    FEATURE_SCHEMA_VERSION,
    all_model_feature_names,
    build_candidate_record,
    canonical_setup_type,
)
from src.scoring.labels import label_candidate_from_ohlcv
from src.scoring.model import (
    LinearProbabilityHead,
    LinearValueHead,
    OutcomeModelArtifact,
)
from src.scoring.training import train_outcome_model


def _candidate() -> dict:
    generated = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        "id": "cand_test",
        "generated_at": generated.isoformat(),
        "direction": "long",
        "decision": {
            "price": 102.0,
            "entry_low": 100.0,
            "entry_high": 101.0,
            "stop_loss": 98.0,
            "take_profits": [103.0, 105.0],
            "entry_valid_until": (generated + timedelta(hours=1)).isoformat(),
            "hold_hours_max": 4,
            "atr": 2.0,
        },
    }


def _candles(*rows: tuple[str, float, float, float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"open": open_, "high": high, "low": low, "close": close}
            for _, open_, high, low, close in rows
        ],
        index=pd.to_datetime([row[0] for row in rows], utc=True),
    )


def test_setup_taxonomy_and_candidate_id_are_stable():
    assert canonical_setup_type("Breakout", ["retest"], "wait_retest") == (
        "breakout_retest"
    )
    assert canonical_setup_type("Momentum continuation", [], "ready") == (
        "cmp_momentum"
    )
    generated = "2026-01-01T00:00:00+00:00"
    row = {
        "symbol": "BTC/USDT:USDT",
        "exchange": "okx",
        "primary_tf": "15m",
        "direction": "long",
        "setup_name": "Momentum continuation",
        "signal_generated_at": generated,
        "price": 100,
        "entry_low": 99.8,
        "entry_high": 100.1,
        "stop_loss": 98.8,
        "take_profits": [101.5, 102.5],
        "entry_status": "ready",
        "entry_valid_for_minutes": 45,
        "hold_hours_max": 8,
        "execution_score": 75,
        "technical_confidence": 82,
        "confidence": 80,
        "rank_score": 70,
        "factors": [{"name": "trend", "score": 0.7}],
        "execution": {"status": "ready", "score": 75},
    }
    first = build_candidate_record(None, row, source="test")
    second = build_candidate_record(None, row, source="test")
    assert first["id"] == second["id"]
    assert first["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert first["setup_type"] == "cmp_momentum"
    assert set(all_model_feature_names()).issubset(first["features"])


def test_ohlcv_label_is_stop_first_when_fill_tp_and_stop_share_a_candle():
    outcome = label_candidate_from_ohlcv(
        _candidate(),
        _candles(("2026-01-01T00:15:00Z", 102, 104, 97, 100)),
    )
    assert outcome.valid_fill is True
    assert outcome.alert_success is False
    assert outcome.tp1_hit is False
    assert outcome.terminal_status == "stopped"
    assert outcome.realized_r == -1.0


def test_ohlcv_label_counts_target_before_entry_as_missed_failure():
    outcome = label_candidate_from_ohlcv(
        _candidate(),
        _candles(("2026-01-01T00:15:00Z", 102, 104, 102, 103)),
    )
    assert outcome.valid_fill is False
    assert outcome.alert_success is False
    assert outcome.missed_before_fill is True
    assert outcome.terminal_status == "missed"


def test_ohlcv_label_counts_tp1_before_cmp_confirmation_as_missed():
    candidate = _candidate()
    candidate["decision"].update(
        {
            "entry_status": "ready",
            "price": 100.5,
        }
    )
    outcome = label_candidate_from_ohlcv(
        candidate,
        _candles(("2026-01-01T00:15:00Z", 100.5, 103.2, 100.2, 103.0)),
    )

    assert outcome.valid_fill is False
    assert outcome.alert_success is False
    assert outcome.tp1_hit is False
    assert outcome.missed_before_fill is True
    assert outcome.terminal_status == "missed"


def test_ohlcv_label_protects_tp1_profit_from_later_original_stop():
    outcome = label_candidate_from_ohlcv(
        _candidate(),
        _candles(
            ("2026-01-01T00:15:00Z", 102, 103.2, 100.5, 103.0),
            ("2026-01-01T00:30:00Z", 103, 103.1, 97.5, 98.0),
        ),
    )

    assert outcome.valid_fill is True
    assert outcome.alert_success is True
    assert outcome.tp1_hit is True
    assert outcome.terminal_status == "completed"
    assert outcome.realized_r > 0


def test_model_artifact_scores_are_bounded_and_explainable():
    probability = LinearProbabilityHead(
        name="test",
        feature_names=["trend"],
        mean=[0.0],
        scale=[1.0],
        coefficients=[1.0],
        intercept=0.0,
    )
    value = LinearValueHead(
        feature_names=["trend"],
        mean=[0.0],
        scale=[1.0],
        coefficients=[0.5],
        intercept=0.1,
        uncertainty_penalty_r=0.05,
        reference_ev=[-0.5, 0.0, 0.3, 0.8],
    )
    artifact = OutcomeModelArtifact(
        version="test-v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        technical=probability,
        fill=probability,
        conditional_win=probability,
        alert_success=probability,
        expected_value=value,
        training_samples=500,
        calibration_samples=200,
        calibration_ready=True,
    )
    score = artifact.score({"trend": 0.8})
    assert 0 <= score["technical_score"] <= 100
    assert 0 <= score["execution_score"] <= 100
    assert 0 <= score["confidence"] <= 100
    assert 0 <= score["rank_score"] <= 100
    assert score["technical_breakdown"]
    assert score["top_positive_factors"]


def test_promoted_model_does_not_use_llm_as_a_numeric_gate():
    base = {
        "direction": "long",
        "confidence": 84,
        "technical_confidence": 80,
        "execution_score": 78,
        "rank_score": 75,
        "llm_confidence": 0,
        "signal_eligible": True,
        "entry_status": "ready",
        "prop_safe": True,
        "market_quality_ok": True,
        "data_quality_ok": True,
        "historical_edge_ok": True,
        "immediate_sl_risk": 20,
        "chase_distance_atr": 0.2,
        "spread_bps": 2,
        "risk_reward": [1.0, 1.5],
    }
    champion = {**base, "scoring_source": "outcome_champion_veto"}
    legacy = {**base, "scoring_source": "legacy_production"}
    assert filter_high_confidence(
        [champion],
        min_llm=65,
        min_rank=50,
        only_prop_safe=True,
    )
    assert filter_high_confidence(
        [legacy],
        min_llm=65,
        min_rank=50,
        only_prop_safe=True,
    )


def test_walk_forward_training_produces_calibrated_portable_artifact():
    rng = np.random.default_rng(42)
    names = all_model_feature_names()
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(360):
        trend = float(rng.normal())
        execution = float(rng.normal())
        noise = float(rng.normal(scale=0.5))
        valid_fill = execution + noise > -0.35
        tp1 = bool(valid_fill and trend + execution * 0.45 + noise > 0.25)
        features = {name: float(rng.normal(scale=0.2)) for name in names}
        features["trend"] = trend
        features["order_flow_alignment"] = execution
        features["setup__cmp_momentum"] = 1.0
        rows.append(
            {
                "generated_at": (start + timedelta(hours=index * 30)).isoformat(),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "features": features,
                "setup_type": "cmp_momentum",
                "valid_fill": valid_fill,
                "technical_success": trend + noise > 0,
                "alert_success": tp1,
                "tp1_hit": tp1,
                "tp2_hit": bool(tp1 and trend > 0.8),
                "invalidated_before_fill": not valid_fill and execution < -0.8,
                "missed_before_fill": not valid_fill and execution >= -0.8,
                "expired_before_fill": False,
                "entry_delay_minutes": 20 if valid_fill else None,
                "trade_duration_minutes": 120 if valid_fill else None,
                "realized_r": 1.4 if tp1 else (-1.0 if valid_fill else 0.0),
                "production_scores": {"confidence": 75, "rank": 50},
            }
        )
    result = train_outcome_model(
        rows,
        minimum_training_samples=200,
        minimum_calibration_samples=50,
    )
    assert result.artifact.calibration_ready is True
    assert result.artifact.training_samples == 310
    assert result.artifact.calibration_samples == 50
    assert result.validation["method"] == "expanding_walk_forward"
    assert result.validation["unseen_samples"] > 0
    assert "top_quintile_fill_rate" in result.metrics["new"]
    restored = OutcomeModelArtifact.from_dict(result.artifact.to_dict())
    score = restored.score(rows[-1]["features"])
    assert 0 <= score["confidence"] <= 100
