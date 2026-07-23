"""Telegram report formatter + env-only credentials (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notify.telegram import (
    diagnose_telegram,
    format_prop_scan_report,
    format_signal_photo_caption,
    get_telegram_credentials,
    is_telegram_ready,
    send_telegram_message_detailed,
    send_telegram_photo_detailed,
)
from src.scheduler.scan_job import (
    filter_high_confidence,
    get_scheduler_status,
    next_slot_datetime,
    run_scheduler_loop,
)
from src.utils.config import load_config


def test_format_prop_scan_report_with_rows():
    text = format_prop_scan_report(
        [
            {
                "symbol": "BTC/USDT:USDT",
                "price": 112450,
                "direction": "long",
                "llm_confidence": 78,
                "leverage": 5,
                "risk_pct": 1.0,
                "rank_score": 72,
                "llm_confidence_reason": "MTF aligned with volume",
                "prop_flags": ["LEV_CAPPED_5X"],
                "entry_low": 112300,
                "entry_high": 112500,
                "stop_loss": 111900,
                "take_profits": [113200, 113900, 114800, 116000],
                "entry_status": "wait_retest",
                "execution_score": 76,
                "immediate_sl_risk": 24,
                "hold_label": "2–8h day trade",
                "backtest": {
                    "sample_ok": True,
                    "n_signals": 14,
                    "win_rate": 60,
                    "profit_factor": 1.4,
                    "n_trades": 10,
                    "stop_out_rate": 30,
                },
            }
        ],
        slot_label="16:00 WAT",
    )
    assert "Prop Scan" in text
    assert "BTC" in text
    assert "112,450" in text or "112450" in text
    assert "LLM 78%" in text
    assert "MTF aligned" in text
    assert "Entry" in text
    assert "Wait Retest" in text
    assert "execution 76/100" in text
    assert "immediate-SL risk 24%" in text
    assert "TP1" in text
    assert "10/14 fills" in text
    assert "PF 1.40" in text


def test_format_empty():
    text = format_prop_scan_report([], slot_label="05:00 WAT")
    assert "No high-confidence" in text


def test_filter_high_confidence():
    rows = [
        {"direction": "long", "llm_confidence": 80, "rank_score": 70, "prop_safe": True},
        {"direction": "long", "llm_confidence": 40, "rank_score": 30, "prop_safe": True},
        {"direction": "flat", "llm_confidence": 90, "rank_score": 90, "prop_safe": True},
        {"direction": "short", "llm_confidence": 70, "rank_score": 60, "prop_safe": False},
    ]
    out = filter_high_confidence(rows, min_llm=65, min_rank=50, only_prop_safe=True)
    assert len(out) == 1
    assert out[0]["llm_confidence"] == 80


def test_filter_rejects_weak_execution_and_signal_gate():
    rows = [
        {
            "direction": "long",
            "confidence": 75,
            "llm_confidence": 80,
            "rank_score": 70,
            "prop_safe": True,
            "signal_eligible": True,
            "entry_status": "avoid_chase",
            "execution_score": 80,
        },
        {
            "direction": "short",
            "confidence": 75,
            "llm_confidence": 80,
            "rank_score": 70,
            "prop_safe": True,
            "signal_eligible": True,
            "entry_status": "wait_retest",
            "execution_score": 50,
        },
        {
            "direction": "short",
            "confidence": 75,
            "llm_confidence": 80,
            "rank_score": 70,
            "prop_safe": True,
            "signal_eligible": False,
            "entry_status": "ready",
            "execution_score": 80,
        },
    ]
    assert (
        filter_high_confidence(rows, min_llm=65, min_rank=50, only_prop_safe=True)
        == []
    )


def test_next_slot_datetime_future():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime(2026, 7, 21, 10, 0, tzinfo=ZoneInfo("Africa/Lagos"))
    nxt = next_slot_datetime(["05:00", "16:00", "20:00"], "Africa/Lagos", now=now)
    assert nxt.hour == 16


def test_credentials_from_env_only(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    # Even if YAML had secrets historically, load_config must not invent them
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.telegram.bot_token == ""
    assert cfg.telegram.chat_id == ""
    assert is_telegram_ready(cfg) is False

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-not-real")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    # Re-apply env path via helper
    token, chat = get_telegram_credentials()
    assert token == "test-token-not-real"
    assert chat == "123456"

    # YAML must still not contain secret keys as the source of truth
    raw = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "bot_token:" not in raw
    assert "chat_id:" not in raw
    assert "TELEGRAM_BOT_TOKEN" in raw  # documentation comment only


class _FakeTelegramResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


def test_detailed_sender_returns_telegram_error_without_token(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")

    def fake_post(url, json, timeout):
        assert "123456:secret-token" in url
        assert json["chat_id"] == "-1001234567890"
        return _FakeTelegramResponse(
            403,
            {
                "ok": False,
                "error_code": 403,
                "description": "Forbidden: bot is not a member of the channel chat",
            },
        )

    monkeypatch.setattr("src.notify.telegram.requests.post", fake_post)
    result = send_telegram_message_detailed("test")
    assert result["ok"] is False
    assert result["telegram_error_code"] == 403
    assert "not a member" in result["description"]
    assert "secret-token" not in str(result)
    assert result["chat_id_masked"].endswith("7890")


def test_detailed_sender_reports_message_id(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    monkeypatch.setattr(
        "src.notify.telegram.requests.post",
        lambda *a, **k: _FakeTelegramResponse(
            200,
            {"ok": True, "result": {"message_id": 77}},
        ),
    )
    result = send_telegram_message_detailed("test")
    assert result["ok"] is True
    assert result["message_id"] == 77


def test_photo_sender_uploads_png_and_caption(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    captured = {}

    def fake_post(url, data, files, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["files"] = files
        return _FakeTelegramResponse(
            200,
            {"ok": True, "result": {"message_id": 91}},
        )

    monkeypatch.setattr("src.notify.telegram.requests.post", fake_post)
    result = send_telegram_photo_detailed(
        b"\x89PNG\r\n\x1a\nfake",
        "<b>BTC LONG</b>",
    )
    assert result["ok"] is True
    assert result["message_id"] == 91
    assert captured["data"]["caption"] == "<b>BTC LONG</b>"
    assert captured["files"]["photo"][2] == "image/png"


def test_signal_photo_caption_is_clean_and_actionable():
    row = {
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "primary_tf": "15m",
        "confidence": 78,
        "technical_confidence": 75,
        "llm_confidence": 80,
        "entry_status": "wait_retest",
        "execution_score": 76,
        "entry_low": 112300,
        "entry_high": 112500,
        "stop_loss": 111900,
        "take_profits": [113200, 113900, 114800, 116000],
        "leverage": 5,
        "risk_pct": 1,
        "immediate_sl_risk": 24,
        "order_flow_score": 0.42,
        "funding_rate": 0.0001,
        "open_interest_change_pct_24h": 4.2,
        "llm_confidence_reason": "1h and 4h momentum align with expanding volume.",
        "payload": {
            "execution": {
                "status": "wait_retest",
                "entry_reason": "Wait for the demand-zone retest; do not enter at market.",
            },
            "chart": {"timeframe": "15m"},
        },
    }
    caption = format_signal_photo_caption(row, slot_label="16:00 WAT")
    assert "BTC LONG RETEST" in caption
    assert "DO NOT CHASE" in caption
    assert "CONFIDENCE 78%" in caption
    assert "Entry" in caption
    assert "Stop" in caption
    assert "TP1" in caption
    assert "Why:" in caption
    assert "funding" in caption
    assert "OI 24h" in caption
    assert len(caption) <= 1024


def test_telegram_diagnostics_checks_bot_chat_and_membership(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001234567890")

    def fake_get(url, params=None, timeout=None):
        method = url.rsplit("/", 1)[-1]
        if method == "getMe":
            return _FakeTelegramResponse(
                200,
                {"ok": True, "result": {"id": 99, "username": "perp_test_bot"}},
            )
        if method == "getChat":
            return _FakeTelegramResponse(
                200,
                {"ok": True, "result": {"id": -1001234567890, "type": "channel"}},
            )
        assert method == "getChatMember"
        assert params["user_id"] == 99
        return _FakeTelegramResponse(
            200,
            {
                "ok": True,
                "result": {
                    "status": "administrator",
                    "can_post_messages": True,
                },
            },
        )

    monkeypatch.setattr("src.notify.telegram.requests.get", fake_get)
    result = diagnose_telegram()
    assert result["ok"] is True
    assert result["bot_username"] == "perp_test_bot"
    assert result["chat_type"] == "channel"
    assert result["membership_status"] == "administrator"
    assert result["can_send_inferred"] is True


def test_scheduler_loop_triggers_due_slot(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from src.scheduler import scan_job

    cfg = load_config(ROOT / "config.yaml")
    calls = []
    due = datetime.now(ZoneInfo(cfg.scheduler.timezone))
    monkeypatch.setattr(scan_job, "next_slot_datetime", lambda *a, **k: due)
    monkeypatch.setattr(
        scan_job,
        "run_scheduled_scan_once",
        lambda *a, **k: calls.append(k.get("slot_label")) or {
            "completed_at": datetime.now(ZoneInfo("UTC")).isoformat(),
            "telegram_delivery_status": "sent",
            "alert_count": 1,
        },
    )

    run_scheduler_loop(cfg, max_iterations=1)
    status = get_scheduler_status()
    assert calls
    assert status["last_delivery_status"] == "sent"
    assert status["last_alert_count"] == 1
    assert status["running"] is False


def test_scheduled_scan_calls_detailed_sender(monkeypatch):
    from src.scheduler import scan_job

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    cfg = load_config(ROOT / "config.yaml")
    row = {
        "direction": "long",
        "confidence": 76,
        "llm_confidence": 78,
        "rank_score": 72,
        "prop_safe": True,
        "signal_eligible": True,
        "entry_status": "wait_retest",
        "execution_score": 75,
    }
    monkeypatch.setattr(
        scan_job,
        "scan_symbols",
        lambda *a, **k: {"ok": True, "ranked_results": [row]},
    )
    sent = []
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda text, **kwargs: sent.append((text, kwargs))
        or {"ok": True, "message_id": 42, "description": "Message delivered"},
    )

    result = scan_job.run_scheduled_scan_once(cfg, slot_label="test", send=True)
    assert sent
    assert result["telegram_sent"] is True
    assert result["telegram_delivery_status"] == "sent_with_text_fallback"
    assert result["telegram_delivery"]["text_fallback"]["message_id"] == 42


def test_scheduled_scan_sends_chart_alert_without_text_fallback(monkeypatch):
    from src.scheduler import scan_job

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    cfg = load_config(ROOT / "config.yaml")
    row = {
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "confidence": 78,
        "llm_confidence": 80,
        "rank_score": 72,
        "prop_safe": True,
        "signal_eligible": True,
        "entry_status": "wait_retest",
        "execution_score": 76,
        "payload": {"chart": {"candles": [{}] * 10}},
    }
    monkeypatch.setattr(
        scan_job,
        "scan_symbols",
        lambda *a, **k: {"ok": True, "ranked_results": [row]},
    )
    monkeypatch.setattr(scan_job, "render_signal_chart_png", lambda row: b"png")
    monkeypatch.setattr(
        scan_job,
        "format_signal_photo_caption",
        lambda row, slot_label="": "<b>BTC LONG</b>",
    )
    photo_calls = []
    monkeypatch.setattr(
        scan_job,
        "send_telegram_photo_detailed",
        lambda photo, caption, **kwargs: photo_calls.append((photo, caption, kwargs))
        or {"ok": True, "message_id": 99},
    )
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("text fallback called")),
    )

    result = scan_job.run_scheduled_scan_once(cfg, slot_label="test", send=True)
    assert photo_calls
    assert result["telegram_sent"] is True
    assert result["telegram_delivery_status"] == "sent_chart_alerts"
    assert result["telegram_delivery"]["items"][0]["message_id"] == 99
