"""Focused scheduler misfire and private-beta manual-routing regressions."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

import main_server
from src.scheduler import scan_job
from src.utils.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class _RunRepository:
    enabled = True
    claims = set()
    deliveries = set()
    finishes = []

    def __init__(self, database_url):
        self.database_url = database_url

    @classmethod
    def reset(cls):
        cls.claims = set()
        cls.deliveries = set()
        cls.finishes = []

    def claim(self, **kwargs):
        run_id = kwargs["run_id"]
        if run_id in self.claims:
            return False
        self.claims.add(run_id)
        return True

    def finish(self, **kwargs):
        self.finishes.append(kwargs)
        for delivery in kwargs.get("deliveries") or []:
            if delivery.get("ok"):
                self.deliveries.add(delivery.get("idempotency_key"))
        return True

    def delivery_succeeded(self, key):
        return key in self.deliveries

    def record_delivery(self, *, run_id, delivery):
        if delivery.get("ok"):
            self.deliveries.add(delivery.get("idempotency_key"))
        return True

    def get(self, run_id):
        return None

    def latest(self):
        return None

    def latest_scheduled(self):
        return None


def _signal(symbol="BTC/USDT:USDT"):
    return {
        "candidate_id": "candidate-" + symbol.split("/")[0].lower(),
        "symbol": symbol,
        "direction": "long",
        "confidence": 84.0,
        "execution_quality": 79.0,
        "execution_score": 79.0,
        "authoritative_rank": 72.0,
        "rank_available": True,
        "rank_score": 72.0,
        "entry_status": "wait_retest",
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 98.0,
        "take_profits": [103.0, 105.0],
        "gross_risk_reward": [1.0, 1.5],
        "net_risk_reward": [0.9, 1.35],
        "signal_generated_at": "2026-08-04T12:30:00Z",
        "gate_evaluation": {"gates": []},
        "qualification": {
            "qualification_policy_version": "private_beta_important_soft_v1",
            "qualification_type": "fully_qualified",
            "private_beta_qualified": True,
            "important_soft_pass_count": 3,
            "soft_pass_count": 5,
            "authoritative_rank": 72.0,
            "rank_available": True,
            "net_rr": 1.35,
        },
        "payload": {"chart": {"candles": [{}] * 60}},
    }


def _configure_private_beta(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100-public")
    monkeypatch.setenv("DELIVERY_MODE", "private_beta")
    monkeypatch.setenv("PRIVATE_BETA_CHAT_IDS", "111111,222222")
    monkeypatch.setenv("TELEGRAM_COMMAND_CHAT_IDS", "111111")
    cfg = load_config(ROOT / "config.yaml")
    _RunRepository.reset()
    monkeypatch.setattr(scan_job, "SchedulerRunRepository", _RunRepository)
    return cfg


def _mock_signal_scan(monkeypatch, row=None):
    candidate = row or _signal()
    monkeypatch.setattr(
        scan_job,
        "scan_symbols",
        lambda *a, **k: {
            "ok": True,
            "ranked_results": [candidate],
            "qualification_candidates": [candidate],
            "analyzed_count": 1,
        },
    )
    monkeypatch.setattr(
        scan_job,
        "qualify_private_beta_candidates",
        lambda rows, limit=2: [dict(candidate)],
    )
    monkeypatch.setattr(
        scan_job,
        "evaluate_private_beta_qualification",
        lambda row, gate=None: dict(candidate["qualification"]),
    )
    monkeypatch.setattr(
        scan_job,
        "revalidate_candidate_for_delivery",
        lambda row, cfg: {"ok": True, "row": row, "reasons": []},
    )
    monkeypatch.setattr(scan_job, "render_signal_chart_png", lambda row: b"png")
    monkeypatch.setattr(scan_job, "format_signal_photo_caption", lambda *a, **k: "signal")


def test_all_four_dst_windows_register_and_london_night_conversion():
    cfg = load_config(ROOT / "config.yaml")
    start = datetime(2026, 8, 4, 0, 0, tzinfo=ZoneInfo("UTC"))
    end = start + timedelta(days=1)
    windows = scan_job.scheduled_session_windows_between(
        cfg.scheduler.sessions,
        start=start,
        end=end,
    )
    assert len(windows) == 4
    london, name = windows[0]
    assert name == "London confirmation"
    assert london.strftime("%H:%M") == "07:20"
    assert london.astimezone(ZoneInfo("Africa/Lagos")).strftime("%H:%M") == "08:20"


def test_scheduled_overlap_inside_grace_runs_once(monkeypatch):
    cfg = load_config(ROOT / "config.yaml")
    _RunRepository.reset()
    monkeypatch.setattr(scan_job, "SchedulerRunRepository", _RunRepository)
    monkeypatch.setattr(scan_job, "_SCAN_RUN_LOCK", Lock())
    monkeypatch.setattr(
        scan_job,
        "_run_scheduled_scan_once_unlocked",
        lambda *a, **k: {
            "ok": True,
            "started_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "completed_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "result_code": "no_quality_setup",
            "telegram_delivery_status": "sent_empty_report",
            "alert_count": 0,
            "delivery_audit": [],
        },
    )
    scheduled_for = (datetime.now(ZoneInfo("UTC")) - timedelta(seconds=30)).isoformat()
    result = scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="London confirmation",
        send=False,
        scheduled_for=scheduled_for,
    )
    assert result["ok"] is True
    assert len(_RunRepository.claims) == 1
    assert len(_RunRepository.finishes) == 1


def test_restart_recovery_runs_due_window_once_then_keeps_next_future(monkeypatch):
    cfg = load_config(ROOT / "config.yaml")
    now = datetime(2026, 8, 4, 7, 21, tzinfo=ZoneInfo("UTC"))

    class RecoveryRepository:
        enabled = True

        def __init__(self):
            self.records = set()

        def latest_scheduled(self):
            return {
                "scheduled_for": "2026-08-03T19:20:00+00:00",
            }

        def get(self, run_id):
            return {"run_id": run_id} if run_id in self.records else None

    repository = RecoveryRepository()
    calls = []

    def run_once(*args, **kwargs):
        repository.records.add(kwargs["run_id"])
        calls.append(kwargs["scheduled_for"])
        return {"ok": True, "run_id": kwargs["run_id"]}

    monkeypatch.setattr(scan_job, "run_scheduled_scan_once", run_once)
    first = scan_job._recover_expected_scheduler_windows(
        cfg,
        repository,
        now=now,
    )
    second = scan_job._recover_expected_scheduler_windows(
        cfg,
        repository,
        now=now,
    )
    assert len(first) == 1
    assert second == []
    assert calls == ["2026-08-04T07:20:00+00:00"]

    next_run, name = scan_job.next_session_datetime(
        cfg.scheduler.sessions,
        now=now,
    )
    assert name == "New York macro follow-through"
    assert next_run.astimezone(ZoneInfo("UTC")).strftime("%H:%M") == "12:50"


def test_scheduled_window_outside_grace_records_miss_and_warns_operator(monkeypatch):
    cfg = _configure_private_beta(monkeypatch)
    monkeypatch.setattr(scan_job, "_SCAN_RUN_LOCK", Lock())
    warnings = []
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda text, **kwargs: warnings.append((text, kwargs["chat_id"]))
        or {"ok": True, "message_id": 91},
    )
    scheduled_for = (
        datetime.now(ZoneInfo("UTC"))
        - timedelta(seconds=scan_job.SCHEDULER_MISFIRE_GRACE_SECONDS + 1)
    ).isoformat()
    result = scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="New York open confirmation",
        scheduled_for=scheduled_for,
    )
    assert result["error"] == "scheduled_window_missed"
    assert result["misfire_status"] == "missed_beyond_grace"
    assert _RunRepository.finishes[-1]["status"] == "skipped"
    assert warnings and warnings[0][1] == "111111"


def test_manual_beta_signal_fans_out_and_deduplicates_per_destination(monkeypatch):
    cfg = _configure_private_beta(monkeypatch)
    _mock_signal_scan(monkeypatch)
    photo_chats = []
    monkeypatch.setattr(
        scan_job,
        "send_telegram_photo_detailed",
        lambda *a, **kwargs: photo_chats.append(kwargs["chat_id"])
        or {"ok": True, "message_id": len(photo_chats)},
    )
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda *a, **k: {"ok": False, "error": "fallback_not_expected"},
    )
    registrations = []
    monkeypatch.setattr(
        scan_job,
        "register_delivered_signals",
        lambda rows, destinations, **kwargs: registrations.append(
            (destinations, kwargs.get("source"))
        ) or {"ok": True, "registered": 1},
    )

    first = scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="Telegram on-demand scan",
        telegram_chat_ids=["111111"],
    )
    second = scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="Telegram on-demand scan",
        telegram_chat_ids=["222222"],
    )

    assert photo_chats == ["111111", "222222"]
    assert first["scan_origin"] == "manual_beta_scan"
    assert first["delivery_recipient_count"] == 2
    assert second["telegram_delivery_status"] == "skipped_duplicate_signals"
    assert len(registrations) == 1
    assert registrations[0][1] == "manual_beta_scan"


def test_manual_beta_no_setup_only_returns_to_requester(monkeypatch):
    cfg = _configure_private_beta(monkeypatch)
    monkeypatch.setattr(
        scan_job,
        "scan_symbols",
        lambda *a, **k: {
            "ok": True,
            "ranked_results": [],
            "qualification_candidates": [],
            "analyzed_count": 1,
        },
    )
    chats = []
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda text, **kwargs: chats.append(kwargs["chat_id"])
        or {"ok": True, "message_id": 1},
    )
    result_b = scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="Telegram on-demand scan",
        telegram_chat_ids=["222222"],
        notify_on_empty=True,
    )
    result_a = scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="Telegram on-demand scan",
        telegram_chat_ids=["111111"],
        notify_on_empty=True,
    )
    assert chats == ["222222", "111111"]
    assert result_b["telegram_delivery_status"] == "sent_empty_report"
    assert result_a["telegram_delivery_status"] == "sent_empty_report"


def test_manual_beta_failure_is_requester_and_operator_only(monkeypatch):
    cfg = _configure_private_beta(monkeypatch)
    monkeypatch.setenv("TELEGRAM_COMMAND_CHAT_IDS", "333333")
    monkeypatch.setattr(
        scan_job,
        "scan_symbols",
        lambda *a, **k: {
            "ok": False,
            "ranked_results": [],
            "qualification_candidates": [],
            "analyzed_count": 0,
        },
    )
    chats = []
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda text, **kwargs: chats.append(kwargs["chat_id"])
        or {"ok": True, "message_id": len(chats)},
    )
    scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="Telegram on-demand scan",
        telegram_chat_ids=["222222"],
    )
    assert chats == ["222222", "333333"]
    assert "111111" not in chats


def test_scheduler_status_endpoint_is_protected(monkeypatch):
    class Request:
        client = None

    monkeypatch.setattr(main_server.SCAN_ACCESS, "authorize", lambda *a, **k: None)
    payload = main_server.admin_scheduler_status(Request(), "test-key")
    assert payload["ok"] is True
    assert len(payload["scheduler"].get("active_windows") or []) in {0, 4}
