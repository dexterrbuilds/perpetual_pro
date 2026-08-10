from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.experiments.comparison import build_comparison_report
from src.experiments.identity import bot_namespace, is_legacy_comparison
from src.experiments.legacy_policy import (
    LEGACY_POLICY_VERSION,
    evaluate_legacy_qualification,
    qualify_legacy_candidates,
)
from src.api.service import _build_directional_comparison_row
from src.notify.telegram import format_signal_photo_caption, get_delivery_status
from src.scheduler.run_repository import SchedulerRunRepository
from src.scheduler.scan_job import _durable_run_id
from src.scoring.features import build_candidate_record
from src.scoring.repository import OutcomeRepository
from src.tracking.durable_repository import LifecycleRepository
from src.tracking.signal_tracker import SignalStore


UTC = timezone.utc


def _row(**overrides):
    now = datetime.now(UTC)
    row = {
        "candidate_id": "legacy-candidate",
        "symbol": "BTC/USDT:USDT",
        "exchange": "okx",
        "direction": "long",
        "evaluated_direction": "long",
        "primary_tf": "15m",
        "confidence": 84.0,
        "technical_confidence": 86.0,
        "execution_quality": 72.0,
        "execution_score": 72.0,
        "authoritative_rank": 62.0,
        "rank_available": True,
        "confluence_score": 0.35,
        "immediate_sl_risk": 25.0,
        "data_quality_ok": True,
        "market_quality_ok": True,
        "market_supported": True,
        "ticker_age_seconds": 3.0,
        "orderbook_age_seconds": 4.0,
        "spread_bps": 2.0,
        "entry_status": "wait_retest",
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 98.0,
        "take_profits": [103.0, 105.0],
        "gross_risk_reward": [1.1, 1.05],
        "net_risk_reward": [0.90, 0.86],
        "entry_valid_until": (now + timedelta(hours=2)).isoformat(),
        "chase_distance_atr": 0.4,
        "tp1_progress_pct": 10.0,
        "entry_zone_relation": "before_entry",
        "hard_failures": [],
        "stop_quality": 70.0,
        "target_feasibility": [70.0, 65.0],
        "pre_delivery_revalidation_ok": True,
        "payload": {"execution": {}, "primary_setup": {}},
    }
    row.update(overrides)
    return row


def test_strict_defaults_remain_unchanged(monkeypatch):
    monkeypatch.delenv("BOT_VARIANT", raising=False)
    monkeypatch.delenv("BOT_NAMESPACE", raising=False)
    assert is_legacy_comparison() is False
    assert bot_namespace() == "perpetual_pro_strict"
    candidate = build_candidate_record(None, _row(), source="test")
    assert candidate["id"].startswith("cand_")
    assert not candidate["id"].startswith("legacy_")
    assert candidate["is_directional_candidate"] is True


def test_legacy_policy_relaxes_selectivity_but_preserves_absolute_floors(monkeypatch):
    monkeypatch.setenv("BOT_VARIANT", "legacy")
    relaxed = evaluate_legacy_qualification(
        _row(hard_failures=["setup_confirmation_insufficient"])
    )
    assert relaxed["qualified"] is True
    assert relaxed["qualification_policy_version"] == LEGACY_POLICY_VERSION
    assert any(
        item["code"] == "CAVEAT_SETUP_CONFIRMATION_INSUFFICIENT"
        for item in relaxed["caveats"]
    )

    assert evaluate_legacy_qualification(_row(confidence=79.99))["qualified"] is False
    assert evaluate_legacy_qualification(_row(execution_quality=64.99, execution_score=64.99))["qualified"] is False
    assert evaluate_legacy_qualification(_row(net_risk_reward=[0.5, 0.74]))["qualified"] is False


def test_legacy_recovers_entry_state_only_for_relaxable_selectivity(monkeypatch):
    monkeypatch.setenv("BOT_VARIANT", "legacy")
    relaxed = _row(
        candidate_id="relaxed-block",
        entry_status="blocked",
        legacy_execution_status="wait_retest",
        execution_setup_type="trend_pullback",
        setup_name="No Trade / Wait for Confirmation",
        hard_failures=["setup_confirmation_insufficient"],
    )
    decision = evaluate_legacy_qualification(relaxed)
    assert decision["qualified"] is True
    assert decision["effective_entry_status"] == "wait_retest"
    selected = qualify_legacy_candidates([relaxed], limit=2)
    assert selected[0]["entry_status"] == "wait_retest"
    assert selected[0]["strict_entry_status"] == "blocked"
    assert selected[0]["setup_name"] == "Trend Continuation (Pullback)"

    universal = _row(
        entry_status="blocked",
        legacy_execution_status="wait_retest",
        hard_failures=["stale_execution_data"],
    )
    assert evaluate_legacy_qualification(universal)["qualified"] is False


def test_legacy_journal_uses_preserved_directional_thesis(monkeypatch):
    monkeypatch.setenv("BOT_VARIANT", "legacy")
    flattened = _row(
        direction="flat",
        evaluated_direction="short",
        entry_low=99.0,
        entry_high=100.0,
        stop_loss=102.0,
        take_profits=[97.0, 95.0],
        payload={
            "direction": "flat",
            "primary_setup": {"direction": "flat"},
            "chart": {"trade": {"direction": "flat"}},
        },
    )
    comparison = _build_directional_comparison_row(flattened)
    assert comparison is not None
    assert comparison["direction"] == "short"
    assert comparison["payload"]["direction"] == "short"
    assert comparison["payload"]["primary_setup"]["direction"] == "short"
    assert comparison["payload"]["chart"]["trade"]["direction"] == "short"
    assert evaluate_legacy_qualification(comparison)["qualified"] is True
    assert flattened["direction"] == "flat"


def test_legacy_correctness_failures_remain_hard(monkeypatch):
    monkeypatch.setenv("BOT_VARIANT", "legacy")
    cases = (
        {"market_supported": False},
        {"data_quality_ok": False},
        {"ticker_age_seconds": 46.0},
        {"orderbook_age_seconds": 31.0},
        {"spread_bps": 12.1},
        {"entry_status": "blocked"},
        {"stop_loss": 102.0},
        {"tp1_progress_pct": 70.0, "entry_zone_relation": "favorable_beyond"},
        {"pre_delivery_revalidation_ok": False},
    )
    for overrides in cases:
        assert evaluate_legacy_qualification(_row(**overrides))["qualified"] is False


def test_legacy_candidate_and_operational_namespaces_are_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("BOT_VARIANT", "legacy")
    monkeypatch.setenv("BOT_NAMESPACE", "perpetual_pro_legacy_v1")
    candidate = build_candidate_record(None, _row(), source="comparison")
    assert candidate["id"].startswith("legacy_cand_")
    assert candidate["is_directional_candidate"] is False
    assert candidate["decision"]["comparison_directional_candidate"] is True

    lifecycle = LifecycleRepository("postgresql://configured")
    scheduler = SchedulerRunRepository("postgresql://configured")
    outcomes = OutcomeRepository("postgresql://configured")
    assert lifecycle._signals == "legacy_comparison.tracked_signals"
    assert lifecycle._events == "legacy_comparison.signal_lifecycle_events"
    assert lifecycle._ledger == "legacy_comparison.telegram_notification_ledger"
    assert scheduler._runs == "legacy_comparison.scheduler_runs"
    assert scheduler._deliveries == "legacy_comparison.scheduler_run_deliveries"
    assert outcomes._tracked_signals == "legacy_comparison.tracked_signals"
    assert _durable_run_id("London open", "2026-08-09T07:20:00+00:00").startswith("legacy_sched_")

    store = SignalStore(tmp_path / "legacy.db", target_allocations=[0.5, 0.5])
    signal, created = store.register_signal(_row(), ["private"], source="legacy_test")
    assert created is True
    assert signal["id"].startswith("legacy_sig_")
    store.close()


def test_legacy_candidate_journal_preserves_comparison_direction(monkeypatch):
    monkeypatch.setenv("BOT_VARIANT", "legacy")
    candidate = build_candidate_record(
        None,
        _row(direction="flat", evaluated_direction="short"),
        source="comparison",
    )
    assert candidate["direction"] == "short"
    assert candidate["decision"]["comparison_directional_candidate"] is True
    assert candidate["is_directional_candidate"] is False


def test_legacy_telegram_is_private_and_unmistakable(monkeypatch):
    monkeypatch.setenv("BOT_VARIANT", "legacy")
    monkeypatch.setenv("DELIVERY_MODE", "private_beta")
    monkeypatch.setenv("PRIVATE_BETA_CHAT_IDS", "111, 222,111")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100999")
    row = _row()
    row["legacy_decision"] = evaluate_legacy_qualification(row)
    row["quality_tier"] = row["legacy_decision"]["quality_tier"]
    row["caveats"] = row["legacy_decision"]["caveats"]
    caption = format_signal_photo_caption(row, slot_label="London open")
    status = get_delivery_status()
    assert caption.startswith("🧪 <b>PERPETUAL PRO LEGACY</b>")
    assert "SETUP" in caption
    assert len(caption) <= 1024
    assert status["mode"] == "private_beta"
    assert status["beta_recipient_count"] == 2
    assert status["public_delivery_enabled"] is False


def test_legacy_ranking_and_strict_shadow_are_recorded(monkeypatch):
    monkeypatch.setenv("BOT_VARIANT", "legacy")
    first = _row(candidate_id="first", confidence=82.0)
    second = _row(candidate_id="second", confidence=88.0)
    selected = qualify_legacy_candidates([first, second], limit=2)
    assert [item["candidate_id"] for item in selected] == ["second", "first"]
    assert all("strict_shadow_decision" in item for item in selected)
    assert all("legacy_decision" in item for item in selected)


def test_comparison_report_separates_legacy_only_and_agreed_signals():
    rows = [
        {
            "generated_at": "2026-08-08T10:00:00+00:00",
            "legacy_decision": {"qualified": True},
            "strict_shadow_decision": {"decision": "REJECTED"},
            "outcome": {"terminal_status": "completed", "valid_fill": True, "tp1_hit": True, "realized_r": 1.2},
        },
        {
            "generated_at": "2026-08-09T10:00:00+00:00",
            "legacy_decision": {"qualified": True},
            "strict_shadow_decision": {"decision": "SIGNAL"},
            "outcome": {"terminal_status": "stopped", "valid_fill": True, "tp1_hit": False, "realized_r": -1.0},
        },
    ]
    report = build_comparison_report(rows)
    assert report["legacy_signal_count"] == 2
    assert report["strict_would_reject_count"] == 1
    assert report["both_policies"] == 1
    assert report["maximum_losing_streak"] == 1
    assert report["maximum_winning_streak"] == 1


def test_legacy_entrypoint_requires_separate_credentials_when_enabled():
    source = Path("legacy_server.py").read_text(encoding="utf-8")
    assert '_require("LEGACY_TELEGRAM_BOT_TOKEN")' in source
    assert '_require("LEGACY_BETA_CHAT_IDS")' in source
    assert 'os.environ["DELIVERY_MODE"] = "private_beta"' in source
    assert 'os.getenv("TELEGRAM_ENABLED", "1")' in source
