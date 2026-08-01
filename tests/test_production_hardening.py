from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.api.security import (
    MAX_SCAN_SYMBOLS,
    ScanAccessController,
    normalize_requested_symbols,
    validate_timeframe,
)
from src.scoring.features import build_candidate_record
from src.scoring.training import _prepare_rows
from src.tracking.durable_repository import (
    LifecycleRepository,
    destination_hash,
    event_id_for,
    remaining_size_for,
)
from src.tracking.signal_tracker import SignalStore, SignalTracker
from src.utils.build_info import get_build_identity
from src.utils.config import load_config


UTC = timezone.utc


def _request(headers=None, client=("127.0.0.1", 1234)):
    encoded = [
        (str(key).lower().encode(), str(value).encode())
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/scan",
            "headers": encoded,
            "client": client,
        }
    )


def _row(now: datetime, **overrides):
    value = {
        "symbol": "BTC/USDT:USDT",
        "exchange": "okx",
        "direction": "long",
        "primary_tf": "15m",
        "confidence": 86,
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 98.0,
        "take_profits": [103.0, 105.0],
        "entry_status": "wait_retest",
        "price": 102.0,
        "signal_generated_at": now.isoformat(),
        "entry_valid_until": (now + timedelta(minutes=90)).isoformat(),
        "hold_hours_max": 8,
        "feature_schema_version": "3.0",
        "execution_policy_version": "execution_quality_v2a.1",
        "rank_policy_version": "deterministic_rank_v2a.1",
    }
    value.update(overrides)
    return value


def _store(path):
    return SignalStore(path, target_allocations=[0.5, 0.5])


@pytest.mark.parametrize("state", ["pending", "confirmation_pending", "entered", "tp1"])
def test_active_lifecycle_state_survives_sqlite_cache_rebuild(tmp_path, state):
    now = datetime.now(UTC)
    first = _store(tmp_path / "first.db")
    row = _row(
        now,
        entry_status=("confirmation_pending" if state == "confirmation_pending" else "wait_retest"),
    )
    signal, _ = first.register_signal(row, ["private"], source="test", now=now)
    if state in {"entered", "tp1"}:
        first.process_price(signal["symbol"], 100.5, now + timedelta(minutes=2), source="test")
    if state == "tp1":
        first.process_price(signal["symbol"], 103.0, now + timedelta(minutes=5), source="test")
    durable = first.signals_for_sync(signal["symbol"])[0]
    first.close()

    rebuilt = _store(tmp_path / "rebuilt.db")
    assert rebuilt.restore_signal(durable) is True
    restored = rebuilt.active_signals()[0]
    assert restored["id"] == signal["id"]
    assert restored["status"] == ("pending" if state in {"pending", "confirmation_pending"} else "entered")
    assert restored["highest_tp"] == (1 if state == "tp1" else 0)
    rebuilt.close()


def test_expired_during_downtime_is_reconciled_after_restore(tmp_path):
    now = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    first = _store(tmp_path / "a.db")
    signal, _ = first.register_signal(
        _row(now, entry_valid_until=(now + timedelta(minutes=15)).isoformat()),
        ["private"], source="test", now=now,
    )
    durable = first.signals_for_sync(signal["symbol"])[0]
    first.close()
    rebuilt = _store(tmp_path / "b.db")
    rebuilt.restore_signal(durable)
    assert rebuilt.apply_time_rules(now + timedelta(minutes=16)) == 1
    assert rebuilt.active_signals() == []
    assert rebuilt.events_for_sync(signal["id"])[0]["event_type"] == "expired"
    rebuilt.close()


def test_restart_gap_without_previous_observation_is_ambiguous(tmp_path):
    now = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    first = _store(tmp_path / "a.db")
    signal, _ = first.register_signal(_row(now, price=None), ["private"], source="test", now=now)
    durable = first.signals_for_sync(signal["symbol"])[0]
    first.close()
    rebuilt = _store(tmp_path / "b.db")
    rebuilt.restore_signal(durable)
    assert rebuilt.process_price(signal["symbol"], 103.0, now + timedelta(minutes=5), source="recovery") == 1
    event = rebuilt.events_for_sync(signal["id"])[0]
    assert event["event_type"] == "ambiguous_gap"
    rebuilt.close()


def test_duplicate_tick_and_candle_do_not_duplicate_events(tmp_path):
    now = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    store = _store(tmp_path / "signals.db")
    signal, _ = store.register_signal(
        _row(now, entry_status="confirmation_pending", price=100.5),
        ["private"], source="test", now=now,
    )
    closed = now + timedelta(minutes=15)
    assert store.process_closed_candle(signal["symbol"], 100.5, closed) == 1
    assert store.process_closed_candle(signal["symbol"], 100.5, closed) == 0
    tick = closed + timedelta(minutes=1)
    assert store.process_price(signal["symbol"], 103.0, tick, source="test") == 1
    assert store.process_price(signal["symbol"], 103.0, tick, source="test") == 0
    events = store.events_for_sync(signal["id"])
    assert [event["event_type"] for event in events] == ["published", "entered", "target_hit"]
    assert len({event["event_uid"] for event in events}) == 3
    store.close()


def test_recovery_deduplicates_durable_active_rows(tmp_path):
    cfg = load_config()
    cfg.signal_tracker.database_path = str(tmp_path / "tracker.db")
    cfg.signal_tracker.websocket_enabled = False
    cfg.signal_tracker.durable_lifecycle_required = True
    tracker = SignalTracker(cfg)
    now = datetime.now(UTC)
    seed = _store(tmp_path / "seed.db")
    signal, _ = seed.register_signal(_row(now), ["private"], source="test", now=now)
    durable = seed.signals_for_sync()[0]
    seed.close()

    class Repo:
        def check_ready(self): return True
        def load_active(self): return [durable, durable]
        def status(self): return {"ready": True}
        def persist_state_and_events(self, signal, events=()): return True
        def claim_notifications(self, limit=25): return []

    tracker.lifecycle_repository = Repo()
    tracker._reconcile = lambda: {"attempted": 1, "errors": []}
    assert tracker.start() is True
    status = tracker.status()
    assert status["recovery"]["active_found"] == 2
    assert status["recovery"]["restored"] == 1
    assert status["recovery"]["duplicates_skipped"] == 1
    assert status["active_count"] == 1
    tracker.stop()


def test_durable_hashes_and_event_keys_are_stable_and_redacted():
    assert destination_hash("-100123") == destination_hash("-100123")
    assert "-100123" not in destination_hash("-100123")
    event = {"signal_id": "sig", "lifecycle_version": 2, "event_type": "entered", "occurred_at": "now"}
    assert event_id_for(event) == event_id_for(event)


def test_durable_remaining_size_is_zero_for_every_terminal_state():
    allocations = [0.5, 0.5]
    assert remaining_size_for("entered", allocations, 1) == pytest.approx(0.5)
    for status in (
        "completed",
        "stopped",
        "missed",
        "expired",
        "invalidated",
        "time_exit",
        "ambiguous_gap",
    ):
        assert remaining_size_for(status, allocations, 1) == 0.0


@pytest.mark.parametrize(
    "delivered,attempts,expected_status,message_id",
    [(True, 1, "delivered", 42), (False, 6, "dead_letter", None)],
)
def test_terminal_notification_ack_preserves_nonnull_retry_timestamp(
    monkeypatch, delivered, attempts, expected_status, message_id
):
    """Terminal ledger states must be durable under the SQL constraint."""
    executed = []

    class Cursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params):
            executed.append((sql, params))

        def fetchone(self):
            return {"attempt_count": attempts}

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return Cursor()

        def commit(self):
            return None

    repository = LifecycleRepository("postgresql://configured")
    monkeypatch.setattr(repository, "_connect", lambda: Connection())

    assert repository.finish_notification(
        7,
        delivered=delivered,
        message_id=message_id,
        error_category="test_failure",
    ) is True
    update_params = executed[-1][1]
    assert update_params[0] == expected_status
    assert update_params[1] == message_id
    assert update_params[3] is not None


def test_scan_auth_rejects_unauthorized_and_rate_limits(monkeypatch):
    monkeypatch.setenv("SCAN_API_KEY", "secret-key")
    controller = ScanAccessController()
    with pytest.raises(HTTPException) as unauthorized:
        controller.authorize(_request(), None)
    assert unauthorized.value.status_code == 401
    for _ in range(6):
        controller.authorize(_request(), "secret-key")
    with pytest.raises(HTTPException) as limited:
        controller.authorize(_request(), "secret-key")
    assert limited.value.status_code == 429


def test_scan_limits_and_unsupported_inputs_fail_before_market_fetch():
    with pytest.raises(HTTPException) as unsupported:
        normalize_requested_symbols(["FAKE"], approved_bases=["BTC", "ETH"])
    assert unsupported.value.status_code == 422
    with pytest.raises(HTTPException):
        normalize_requested_symbols(["BTC"] * (MAX_SCAN_SYMBOLS + 1), approved_bases=["BTC"])
    with pytest.raises(HTTPException):
        validate_timeframe("5m", "15m")


def test_scan_concurrency_is_bounded():
    controller = ScanAccessController()
    first = controller.slot()
    second = controller.slot()
    first.__enter__()
    second.__enter__()
    try:
        with pytest.raises(HTTPException) as limited:
            with controller.slot():
                pass
        assert limited.value.status_code == 429
    finally:
        second.__exit__(None, None, None)
        first.__exit__(None, None, None)


def test_flat_and_ambiguous_rows_cannot_enter_directional_training():
    generated = "2026-08-01T00:00:00Z"
    flat = build_candidate_record(
        None,
        {
            "symbol": "BTC/USDT:USDT", "direction": "flat", "bias": "bullish",
            "signal_generated_at": generated, "price": 100, "entry_low": 99,
            "entry_high": 100, "stop_loss": 98, "take_profits": [102],
        },
        source="test",
    )
    assert flat["direction"] == "flat"
    assert flat["is_directional_candidate"] is False
    rows = [
        {**flat, "terminal_status": "completed"},
        {
            **flat,
            "direction": "long",
            "is_directional_candidate": True,
            "terminal_status": "ambiguous_gap",
        },
    ]
    assert _prepare_rows(rows, "3.0") == []


def test_build_identity_reports_policy_versions(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT_SHA", "abc123")
    monkeypatch.setenv("BUILD_TIMESTAMP", "2026-08-01T00:00:00Z")
    identity = get_build_identity()
    assert identity["git_commit_sha"] == "abc123"
    assert identity["feature_schema"] == "3.0"
    assert identity["execution_policy"] == "execution_quality_v2a.1"
    assert identity["rank_policy"] == "deterministic_rank_v2a.1"
    assert identity["identity_complete"] is True


def test_supabase_outage_blocks_tracker_recovery_but_not_process_object(tmp_path):
    cfg = load_config()
    cfg.signal_tracker.database_path = str(tmp_path / "tracker.db")
    cfg.signal_tracker.websocket_enabled = False
    cfg.signal_tracker.durable_lifecycle_required = True
    tracker = SignalTracker(cfg)

    class Outage:
        def check_ready(self): return False
        def status(self): return {"ready": False, "last_error": "OperationalError"}

    tracker.lifecycle_repository = Outage()
    assert tracker.start() is False
    status = tracker.status()
    assert status["running"] is False
    assert status["recovery_failed"] is True
    # Construction/liveness remains possible despite dependency degradation.
    assert status["enabled"] is True
    tracker.stop()


def test_incompatible_active_lifecycle_schema_fails_recovery_safely(tmp_path):
    cfg = load_config()
    cfg.signal_tracker.database_path = str(tmp_path / "tracker.db")
    cfg.signal_tracker.websocket_enabled = False
    cfg.signal_tracker.durable_lifecycle_required = True
    tracker = SignalTracker(cfg)

    class LegacyRepo:
        def check_ready(self): return True
        def status(self): return {"ready": True}
        def load_active(self):
            return [{"id": "legacy", "_lifecycle_schema_version": "legacy"}]

    tracker.lifecycle_repository = LegacyRepo()
    assert tracker.start() is False
    status = tracker.status()
    assert status["recovery_failed"] is True
    assert status["recovery"]["invalid"] == 1
    tracker.stop()


def test_readiness_healthy_guarded_mode_and_liveness_survives_dependency_failure(monkeypatch):
    import main_server

    cfg = load_config()
    cfg.scheduler.enabled = False
    cfg.signal_tracker.durable_lifecycle_required = True

    class OutcomeRepo:
        def check_ready(self): return True

    class Runtime:
        repository = OutcomeRepo()

    monkeypatch.setattr(main_server, "get_config", lambda: cfg)
    monkeypatch.setattr(main_server, "get_outcome_scoring_runtime", lambda _cfg: Runtime())
    monkeypatch.setattr(
        main_server,
        "get_signal_tracker_status",
        lambda: {
            "enabled": True, "running": True, "thread_alive": True,
            "durable_lifecycle_ready": True, "recovery_completed": True,
            "recovery_failed": False,
        },
    )
    monkeypatch.setattr(
        main_server,
        "get_scheduler_status",
        lambda: {"enabled": False, "running": False, "active_windows": cfg.scheduler.sessions},
    )
    response = main_server.readiness()
    payload = json.loads(response.body)
    assert response.status_code == 200
    assert payload["status"] == "ready"
    assert payload["guarded_mode"] is True
    assert payload["scheduler"]["state"] == "disabled_intentionally"

    OutcomeRepo.check_ready = lambda self: False
    response = main_server.readiness()
    assert response.status_code == 503
    # Liveness remains a process check and never inherits readiness status.
    monkeypatch.setattr(main_server, "get_outcome_scoring_status", lambda _cfg: {"database": {"ready": False}})
    assert main_server.health()["status"] == "ok"


def test_migration_contains_transition_and_delivery_idempotency_constraints():
    migration = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "migrations" / "002_durable_lifecycle.sql"
    ).read_text(encoding="utf-8")
    assert "unique (signal_id, lifecycle_version)" in migration
    assert "unique (event_id, destination_hash)" in migration
    assert "enforce_signal_lifecycle_transition" in migration
    assert "is_directional_candidate boolean not null default false" in migration
