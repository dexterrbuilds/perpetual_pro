from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notify import telegram_bot
from src.utils.config import load_config


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._payload


def test_parse_scan_command_supports_timeframe_and_custom_symbols():
    timeframe, symbols, error = telegram_bot.parse_scan_command(
        "/scan 1h BTC,ETH SOL"
    )
    assert timeframe == "1h"
    assert symbols == ["BTC", "ETH", "SOL"]
    assert error is None


def test_process_scan_command_runs_production_workflow(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    monkeypatch.delenv("TELEGRAM_COMMAND_CHAT_IDS", raising=False)
    cfg = load_config(ROOT / "config.yaml")
    messages = []
    scan_calls = []
    monkeypatch.setattr(telegram_bot, "scan_in_progress", lambda: False)
    monkeypatch.setattr(
        telegram_bot,
        "send_telegram_message_detailed",
        lambda text, **kwargs: messages.append((text, kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        telegram_bot,
        "run_scheduled_scan_once",
        lambda *args, **kwargs: scan_calls.append(kwargs)
        or {
            "ok": True,
            "scanned": 2,
            "alert_count": 1,
            "telegram_delivery_status": "sent_chart_alerts",
        },
    )

    result = telegram_bot.process_telegram_update(
        {
            "message": {
                "text": "/scan 4h BTC ETH",
                "chat": {"id": 123456, "type": "private"},
            }
        },
        cfg,
    )

    assert result["ok"] is True
    assert result["alert_count"] == 1
    assert "Scan started" in messages[0][0]
    assert messages[0][1]["chat_id"] == "123456"
    assert scan_calls[0]["symbols"] == ["BTC", "ETH"]
    assert scan_calls[0]["timeframe"] == "4h"
    assert scan_calls[0]["notify_on_empty"] is True


def test_process_update_ignores_unauthorized_chat(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    monkeypatch.delenv("TELEGRAM_COMMAND_CHAT_IDS", raising=False)
    cfg = load_config(ROOT / "config.yaml")
    monkeypatch.setattr(
        telegram_bot,
        "send_telegram_message_detailed",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unauthorized chat received a reply")
        ),
    )
    result = telegram_bot.process_telegram_update(
        {"message": {"text": "/scan", "chat": {"id": 999999}}},
        cfg,
    )
    assert result["handled"] is False


def test_private_command_chat_is_separate_from_group_alert_chat(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1009876543210")
    monkeypatch.setenv("TELEGRAM_COMMAND_CHAT_IDS", "123456, 777888")
    cfg = load_config(ROOT / "config.yaml")
    deliveries = []
    monkeypatch.setattr(telegram_bot, "scan_in_progress", lambda: False)
    monkeypatch.setattr(
        telegram_bot,
        "send_telegram_message_detailed",
        lambda text, **kwargs: deliveries.append((text, kwargs)) or {"ok": True},
    )

    result = telegram_bot.process_telegram_update(
        {"message": {"text": "/status", "chat": {"id": 123456, "type": "private"}}},
        cfg,
    )

    assert result["ok"] is True
    assert result["handled"] is True
    assert deliveries[0][1]["chat_id"] == "123456"
    assert telegram_bot.get_telegram_command_chat_ids() == ["123456", "777888"]


def test_chatid_bootstrap_replies_to_unauthorized_private_chat(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1009876543210")
    monkeypatch.setenv("TELEGRAM_COMMAND_CHAT_IDS", "123456")
    cfg = load_config(ROOT / "config.yaml")
    deliveries = []
    monkeypatch.setattr(
        telegram_bot,
        "send_telegram_message_detailed",
        lambda text, **kwargs: deliveries.append((text, kwargs)) or {"ok": True},
    )

    result = telegram_bot.process_telegram_update(
        {"message": {"text": "/chatid", "chat": {"id": 999999, "type": "private"}}},
        cfg,
    )

    assert result["ok"] is True
    assert result["handled"] is True
    assert "999999" in deliveries[0][0]
    assert deliveries[0][1]["chat_id"] == "999999"


def test_configure_webhook_uses_render_url_and_registers_commands(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://perpetual-pro.onrender.com")
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    cfg = load_config(ROOT / "config.yaml")
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return _FakeResponse({"ok": True, "result": True})

    monkeypatch.setattr(telegram_bot.requests, "post", fake_post)
    assert telegram_bot.configure_telegram_webhook(cfg) is True
    assert calls[0][1]["url"].endswith("/telegram/webhook")
    assert calls[0][1]["secret_token"] == telegram_bot.telegram_webhook_secret()
    assert calls[1][1]["commands"][0]["command"] == "scan"
    assert calls[1][1]["commands"][3]["command"] == "chatid"
    assert telegram_bot.get_telegram_webhook_status()["configured"] is True


def test_webhook_endpoint_rejects_bad_secret_and_accepts_update(monkeypatch):
    import main_server

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    handled = []
    monkeypatch.setattr(
        main_server,
        "process_telegram_update",
        lambda update, config: handled.append(update),
    )
    client = TestClient(main_server.app)
    update = {"update_id": 1, "message": {"text": "/status", "chat": {"id": 123456}}}

    rejected = client.post(
        "/telegram/webhook",
        json=update,
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
    )
    assert rejected.status_code == 403

    accepted = client.post(
        "/telegram/webhook",
        json=update,
        headers={
            "X-Telegram-Bot-Api-Secret-Token": telegram_bot.telegram_webhook_secret()
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["accepted"] is True
    assert handled == [update]
