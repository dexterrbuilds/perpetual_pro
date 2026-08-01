from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.tracking.signal_tracker import (
    SignalTracker,
    SignalStore,
    format_tracker_event,
)
from src.utils.config import load_config


def _row(now: datetime, **overrides):
    row = {
        "symbol": "BTC/USDT:USDT",
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
    }
    row.update(overrides)
    return row


def _store(tmp_path):
    return SignalStore(
        tmp_path / "signals.db",
        target_allocations=[0.25, 0.25, 0.25, 0.25],
    )


def test_tp1_before_entry_marks_setup_missed(tmp_path):
    now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    store = _store(tmp_path)
    signal, created = store.register_signal(
        _row(now),
        ["123"],
        source="test",
        now=now,
    )

    assert created is True
    assert signal["status"] == "pending"
    assert store.process_price(
        signal["symbol"],
        103.0,
        now + timedelta(minutes=3),
        source="websocket",
    ) == 1
    assert store.active_signals() == []
    events = store.pending_events(retry_after_seconds=0)
    assert events[0]["event_type"] == "missed"
    assert "SETUP MISSED" in format_tracker_event(events[0])
    store.close()


def test_entry_then_tp1_protects_profit_and_never_sends_stop_alert(tmp_path):
    now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    store = _store(tmp_path)
    signal, _ = store.register_signal(
        _row(now),
        ["123"],
        source="test",
        now=now,
    )

    assert store.process_price(
        signal["symbol"],
        100.5,
        now + timedelta(minutes=2),
        source="websocket",
    ) == 1
    assert store.process_price(
        signal["symbol"],
        103.0,
        now + timedelta(minutes=20),
        source="websocket",
    ) == 1
    active = store.active_signals()[0]
    assert active["status"] == "entered"
    assert active["highest_tp"] == 1

    assert store.process_price(
        signal["symbol"],
        98.0,
        now + timedelta(minutes=30),
        source="websocket",
    ) == 1
    assert store.active_signals() == []
    events = store.pending_events(retry_after_seconds=0)
    assert [event["event_type"] for event in events] == [
        "entered",
        "target_hit",
        "protected_exit",
    ]
    protected = events[-1]
    assert protected["payload"]["highest_tp"] == 1
    assert protected["payload"]["result_r"] > 0
    message = format_tracker_event(protected)
    assert "PROFIT SECURED" in message
    assert "STOP HIT" not in message
    assert "not a stop-loss" in message
    store.close()


def test_legacy_queued_stop_after_tp1_is_formatted_as_profit():
    message = format_tracker_event(
        {
            "event_type": "stopped",
            "price": 98.0,
            "payload": {"highest_tp": 1, "result_r": -0.5},
            "signal": {
                "symbol": "BTC/USDT:USDT",
                "direction": "long",
            },
        }
    )

    assert "PROFIT SECURED" in message
    assert "STOP HIT" not in message
    assert "No stop-loss is recorded after TP1" in message


def test_cmp_starts_pending_and_closed_candle_confirms_entry(tmp_path):
    now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    store = _store(tmp_path)
    signal, created = store.register_signal(
        _row(now, entry_status="confirmation_pending", price=100.4),
        ["123"],
        source="test",
        now=now,
    )

    assert created is True
    assert signal["status"] == "pending"
    assert store.pending_events(retry_after_seconds=0) == []
    assert store.process_closed_candle(
        signal["symbol"],
        100.4,
        now + timedelta(minutes=15),
        high_price=101.0,
        low_price=100.0,
    ) == 1
    active = store.active_signals()[0]
    assert active["status"] == "entered"
    assert active["entry_price"] == 100.4
    assert [
        event["event_type"]
        for event in store.pending_events(retry_after_seconds=0)
    ] == ["entered"]
    store.close()


def test_short_cmp_confirmation_then_tp1_counts_profit(tmp_path):
    now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    store = _store(tmp_path)
    signal, created = store.register_signal(
        _row(
            now,
            direction="short",
            entry_status="confirmation_pending",
            entry_low=100.0,
            entry_high=101.0,
            price=100.4,
            stop_loss=103.0,
            take_profits=[98.0, 96.0, 94.0, 92.0],
        ),
        ["123"],
        source="test",
        now=now,
    )

    assert created is True
    assert signal["status"] == "pending"
    assert store.process_closed_candle(
        signal["symbol"],
        100.4,
        now + timedelta(minutes=15),
        high_price=101.0,
        low_price=100.0,
    ) == 1
    assert store.process_price(
        signal["symbol"],
        98.0,
        now + timedelta(minutes=20),
        source="websocket",
    ) == 1
    active = store.active_signals()[0]
    assert active["status"] == "entered"
    assert active["highest_tp"] == 1
    assert active["realized_r"] > 0
    events = store.pending_events(retry_after_seconds=0)
    assert [event["event_type"] for event in events] == ["entered", "target_hit"]
    store.close()


def test_closed_candle_invalidates_pending_but_wick_price_does_not(tmp_path):
    now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    store = _store(tmp_path)
    signal, _ = store.register_signal(
        _row(now, entry_status="confirmation_pending", price=100.4),
        ["123"],
        source="test",
        now=now,
    )

    # A pre-entry trade through Stop is not itself a confirmed invalidation.
    assert store.process_price(
        signal["symbol"],
        97.9,
        now + timedelta(minutes=5),
        source="websocket",
    ) == 0
    assert store.active_signals()[0]["status"] == "pending"

    assert store.process_closed_candle(
        signal["symbol"],
        97.8,
        now + timedelta(minutes=15),
    ) == 1
    events = store.pending_events(retry_after_seconds=0)
    assert events[0]["event_type"] == "invalidated"
    assert "STRUCTURE INVALIDATED" in format_tracker_event(events[0])
    store.close()


def test_matching_signal_merges_destinations_instead_of_double_counting(tmp_path):
    now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    store = _store(tmp_path)
    first, first_created = store.register_signal(
        _row(now),
        ["group"],
        source="scheduled",
        now=now,
    )
    second, second_created = store.register_signal(
        _row(now + timedelta(minutes=1)),
        ["dm"],
        source="manual",
        now=now + timedelta(minutes=1),
    )

    assert first_created is True
    assert second_created is False
    assert first["id"] == second["id"]
    assert second["destinations"] == ["group", "dm"]
    assert len(store.active_signals()) == 1
    store.close()


def test_expiry_is_applied_without_market_polling(tmp_path):
    now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    store = _store(tmp_path)
    store.register_signal(
        _row(now, entry_valid_until=(now + timedelta(minutes=30)).isoformat()),
        ["123"],
        source="test",
        now=now,
    )

    assert store.apply_time_rules(now + timedelta(minutes=31)) == 1
    assert store.active_signals() == []
    assert store.pending_events(retry_after_seconds=0)[0]["event_type"] == "expired"
    store.close()


def test_reliability_summary_uses_only_entered_terminal_outcomes(tmp_path):
    now = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
    store = _store(tmp_path)

    winner, _ = store.register_signal(
        _row(now, confidence=86),
        ["123"],
        source="test",
        now=now,
    )
    store.process_price(
        winner["symbol"],
        100.5,
        now + timedelta(minutes=2),
        source="websocket",
    )
    store.process_price(
        winner["symbol"],
        112.0,
        now + timedelta(hours=2),
        source="websocket",
    )

    loser, _ = store.register_signal(
        _row(
            now + timedelta(hours=3),
            symbol="ETH/USDT:USDT",
            confidence=86,
        ),
        ["123"],
        source="test",
        now=now + timedelta(hours=3),
    )
    store.process_price(
        loser["symbol"],
        100.5,
        now + timedelta(hours=3, minutes=2),
        source="websocket",
    )
    store.process_price(
        loser["symbol"],
        98.0,
        now + timedelta(hours=3, minutes=20),
        source="websocket",
    )

    summary = store.reliability_summary()
    band = next(
        item for item in summary["bands"] if item["confidence_band"] == "85–89%"
    )
    assert summary["completed_outcomes"] == 2
    assert summary["status"] == "collecting_forward_outcomes"
    assert band["sample"] == 2
    assert band["profitable_rate_pct"] == 50.0
    assert band["calibration_ready"] is False
    store.close()


def test_tracker_worker_starts_and_stops_without_market_subscriptions(tmp_path):
    cfg = load_config()
    cfg.signal_tracker.database_path = str(tmp_path / "worker.db")
    cfg.signal_tracker.websocket_enabled = False
    cfg.signal_tracker.reconcile_interval_seconds = 1800
    tracker = SignalTracker(cfg)

    assert tracker.start() is True
    assert tracker.status()["running"] is True
    assert tracker.status()["subscribed_symbols"] == []
    tracker.stop()
    status = tracker.status()
    assert status["running"] is False
    assert status["websocket_connected"] is False
