from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from src.scoring.readiness import assess_shadow_readiness
from src.tracking.signal_tracker import SignalStore
from src.utils.config import load_config


UTC = timezone.utc


def _row(now: datetime, *, symbol: str = "BTC/USDT:USDT") -> dict:
    return {
        "candidate_id": f"candidate-{symbol}",
        "symbol": symbol,
        "exchange": "okx",
        "direction": "long",
        "primary_tf": "15m",
        "confidence": 86,
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 98.0,
        "take_profits": [103.0, 105.0, 108.0, 112.0],
        "entry_status": "wait_retest",
        "price": 102.0,
        "signal_generated_at": now.isoformat(),
        "entry_valid_until": (now + timedelta(minutes=90)).isoformat(),
        "entry_valid_for_minutes": 90,
        "hold_hours_max": 8,
        "feature_schema_version": "3.0",
        "execution_policy_version": "execution_quality_v2a.1",
        "rank_policy_version": "deterministic_rank_v2a.1",
    }


def _store(tmp_path) -> SignalStore:
    return SignalStore(
        tmp_path / "lifecycle.db",
        target_allocations=[0.25, 0.25, 0.25, 0.25],
    )


def test_manual_and_scheduled_signals_receive_full_level_tracking(tmp_path):
    now = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    store = _store(tmp_path)
    manual, _ = store.register_signal(
        _row(now), ["operator"], source="manual_beta_scan", now=now
    )
    scheduled, _ = store.register_signal(
        _row(now, symbol="ETH/USDT:USDT"),
        ["tester"],
        source="scheduled_beta_scan",
        now=now,
    )

    for signal in (manual, scheduled):
        assert store.process_price(
            signal["symbol"],
            100.5,
            now + timedelta(minutes=2),
            source="websocket",
        ) == 1

    # One sparse but ordered observation crosses TP1-TP4. Each published level
    # is retained, even though only one compact lifecycle update is emitted.
    assert store.process_price(
        manual["symbol"],
        112.2,
        now + timedelta(minutes=40),
        source="websocket",
    ) == 1
    assert store.process_price(
        scheduled["symbol"],
        97.8,
        now + timedelta(minutes=30),
        source="websocket",
    ) == 1

    rows = {item["symbol"]: item for item in store.signals_for_sync()}
    winner = rows[manual["symbol"]]
    loser = rows[scheduled["symbol"]]
    assert winner["source"] == "manual_beta_scan"
    assert winner["outcome_classification"] == "profitable"
    assert winner["profitable"] is True
    assert winner["profitable_at"] is not None
    assert winner["highest_tp"] == 4
    assert set(winner["level_hits"]) >= {"ENTRY", "TP1", "TP2", "TP3", "TP4"}
    assert winner["level_hits"]["TP1"]["level"] == 103.0
    assert winner["level_hits"]["TP4"]["observed_price"] == 112.2

    assert loser["source"] == "scheduled_beta_scan"
    assert loser["status"] == "stopped"
    assert loser["outcome_classification"] == "not_profitable"
    assert loser["profitable"] is False
    assert loser["level_hits"]["SL"]["level"] == 98.0
    assert loser["level_hits"]["SL"]["observed_price"] == 97.8

    winner_events = store.events_for_sync(winner["id"])
    completion = next(item for item in winner_events if item["event_type"] == "completed")
    assert [item["tp_number"] for item in completion["payload"]["target_hits"]] == [
        1,
        2,
        3,
        4,
    ]
    store.close()


def test_tp1_sets_profitable_before_trade_reaches_terminal_state(tmp_path):
    now = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
    store = _store(tmp_path)
    signal, _ = store.register_signal(
        _row(now), ["operator"], source="manual_beta_scan", now=now
    )
    store.process_price(
        signal["symbol"], 100.5, now + timedelta(minutes=2), source="websocket"
    )
    store.process_price(
        signal["symbol"], 103.0, now + timedelta(minutes=20), source="websocket"
    )
    active = store.active_signals()[0]
    assert active["status"] == "entered"
    assert active["highest_tp"] == 1
    assert active["outcome_classification"] == "profitable"
    assert active["profitable"] is True
    assert active["profitable_at"] is not None
    assert active["level_hits"]["TP1"]["hit_at"] == active["profitable_at"]
    store.close()


def test_outcome_detail_migration_is_additive():
    sql = Path("migrations/005_lifecycle_outcome_detail.sql").read_text(
        encoding="utf-8"
    ).lower()
    assert "add column if not exists outcome_classification" in sql
    assert "add column if not exists profitable" in sql
    assert "add column if not exists tp3_hit" in sql
    assert "add column if not exists tp4_hit" in sql
    assert "drop " not in sql
    assert "delete " not in sql


def test_shadow_readiness_reports_sample_shortfall_without_training():
    cfg = load_config()
    report = assess_shadow_readiness([], cfg)
    assert report["ready_to_replace_current"] is False
    assert report["status"] == "collecting_outcomes"
    assert report["sample_shortfall"] == 700
    assert report["recommendation"] == "Remain in shadow mode"


def test_shadow_readiness_requires_every_existing_promotion_gate(monkeypatch):
    cfg = load_config()
    cfg.outcome_scoring.minimum_training_samples = 1
    cfg.outcome_scoring.minimum_calibration_samples = 1
    rows = [
        {
            "feature_schema_version": "3.0",
            "is_directional_candidate": True,
            "direction": "long",
            "terminal_status": "completed",
            "ambiguity_policy": "observed_sequence",
            "features": {"trend": 1.0},
            "generated_at": "2026-08-01T00:00:00Z",
        },
        {
            "feature_schema_version": "3.0",
            "is_directional_candidate": True,
            "direction": "short",
            "terminal_status": "stopped",
            "ambiguity_policy": "observed_sequence",
            "features": {"trend": -1.0},
            "generated_at": "2026-08-02T00:00:00Z",
        },
    ]
    checks = {
        "brier_improved": True,
        "log_loss_not_worse": True,
        "calibration_not_worse": True,
        "absolute_calibration_quality": True,
        "minimum_unseen_sample": True,
        "top_ev_improved": True,
        "profit_factor_not_worse": True,
        "all_folds_positive_top_ev": True,
    }

    def result(passed: bool):
        local_checks = dict(checks)
        if not passed:
            local_checks["absolute_calibration_quality"] = False
        return SimpleNamespace(
            artifact=SimpleNamespace(
                version="shadow-test",
                training_samples=1,
                calibration_samples=1,
                calibration_ready=True,
            ),
            validation={
                "fold_count": 2,
                "unseen_samples": 200,
                "promotion_gate": {"passed": passed, "checks": local_checks},
            },
            metrics={"new": {"ece": 0.04}, "old": {"ece": 0.08}},
        )

    monkeypatch.setattr(
        "src.scoring.readiness.train_outcome_model", lambda *args, **kwargs: result(False)
    )
    rejected = assess_shadow_readiness(rows, cfg)
    assert rejected["ready_to_replace_current"] is False
    assert rejected["failed_checks"] == ["absolute_calibration_quality"]

    monkeypatch.setattr(
        "src.scoring.readiness.train_outcome_model", lambda *args, **kwargs: result(True)
    )
    approved = assess_shadow_readiness(rows, cfg)
    assert approved["ready_to_replace_current"] is True
    assert approved["automatic_promotion"] is False
    assert "explicit champion review" in approved["recommendation"]
