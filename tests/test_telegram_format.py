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
    get_telegram_alert_chat_ids,
    get_telegram_credentials,
    is_telegram_ready,
    send_telegram_message_detailed,
    send_telegram_photo_detailed,
)
from src.scheduler.scan_job import (
    cap_signals_by_portfolio_risk,
    filter_high_confidence,
    get_scheduler_status,
    next_session_datetime,
    next_slot_datetime,
    remember_sent_scheduled_signals,
    run_scheduler_loop,
    suppress_recent_scheduled_signals,
)
from src.utils.config import load_config


def test_format_prop_scan_report_with_rows():
    text = format_prop_scan_report(
        [
            {
                "symbol": "BTC/USDT:USDT",
                "price": 112450,
                "direction": "long",
                "confidence": 84,
                "technical_confidence": 82,
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
    assert "112,300" in text or "112300" in text
    assert "Overall Quality 84/100" in text
    assert "Technical Quality 82/100" in text
    assert "MTF aligned" in text
    assert "Entry" in text
    assert "Retest Only" in text
    assert "Execution Quality 76/100" in text
    assert "TP1" in text
    assert "NFA · DYOR · Trade at your own risk" in text
    assert "Educational" not in text


def test_format_empty():
    text = format_prop_scan_report(
        [],
        slot_label="London open",
        scanned_count=15,
        ranked_count=2,
    )
    assert "NO QUALITY SETUP" in text
    assert "STAND ASIDE" in text
    assert "Scanned 15 symbols" in text
    assert "2 directional candidate" in text
    assert "≥80% confidence" in text
    assert text.endswith("NFA · DYOR · Trade at your own risk")


def test_filter_high_confidence():
    rows = [
        {
            "direction": "long",
            "confidence": 80,
            "llm_confidence": 80,
            "rank_score": 70,
            "prop_safe": True,
        },
        {
            "direction": "long",
            "confidence": 79.9,
            "llm_confidence": 90,
            "rank_score": 90,
            "prop_safe": True,
        },
        {
            "direction": "long",
            "confidence": 92,
            "llm_confidence": 40,
            "rank_score": 30,
            "prop_safe": True,
        },
        {
            "direction": "flat",
            "confidence": 90,
            "llm_confidence": 90,
            "rank_score": 90,
            "prop_safe": True,
        },
        {
            "direction": "short",
            "confidence": 90,
            "llm_confidence": 70,
            "rank_score": 60,
            "prop_safe": False,
        },
    ]
    out = filter_high_confidence(rows, min_llm=65, min_rank=50, only_prop_safe=True)
    assert len(out) == 1
    assert out[0]["confidence"] == 80
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


def test_filter_rejects_immediate_stop_market_data_and_backtest_risks():
    base = {
        "direction": "long",
        "confidence": 86,
        "llm_confidence": 75,
        "rank_score": 78,
        "prop_safe": True,
        "signal_eligible": True,
        "entry_status": "wait_retest",
        "execution_score": 78,
        "immediate_sl_risk": 20,
        "chase_distance_atr": 0.5,
        "spread_bps": 2.0,
        "market_quality_ok": True,
        "data_quality_ok": True,
        "historical_edge_ok": True,
        "payload": {"primary_setup": {"risk_reward": [0.8, 1.4, 2.0]}},
    }
    assert len(
        filter_high_confidence(
            [base],
            min_llm=65,
            min_rank=50,
            only_prop_safe=True,
        )
    ) == 1

    for field, bad_value in (
        ("immediate_sl_risk", 40),
        ("chase_distance_atr", 1.2),
        ("spread_bps", 15),
        ("market_quality_ok", False),
        ("data_quality_ok", False),
        ("historical_edge_ok", False),
    ):
        row = dict(base)
        row[field] = bad_value
        assert (
            filter_high_confidence(
                [row],
                min_llm=65,
                min_rank=50,
                only_prop_safe=True,
            )
            == []
        )

    low_rr = dict(base)
    low_rr["payload"] = {"primary_setup": {"risk_reward": [0.7, 1.1]}}
    assert (
        filter_high_confidence(
            [low_rr],
            min_llm=65,
            min_rank=50,
            only_prop_safe=True,
        )
        == []
    )

    late_entry = {
        **base,
        "entry_zone_relation": "favorable_beyond",
        "tp1_progress_pct": 82,
    }
    assert (
        filter_high_confidence(
            [late_entry],
            min_llm=65,
            min_rank=50,
            only_prop_safe=True,
        )
        == []
    )
    assert "ENTRY_MOVE_MOSTLY_MISSED" in late_entry[
        "delivery_rejection_reasons"
    ]


def test_scheduled_signal_dedup_lasts_only_for_entry_validity(monkeypatch):
    import src.scheduler.scan_job as scan_job

    monkeypatch.setattr(scan_job, "_RECENT_SIGNAL_ALERTS", {})
    row = {
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "entry_low": 100.0,
        "entry_high": 100.2,
        "stop_loss": 99.0,
        "entry_valid_for_minutes": 90,
    }
    remember_sent_scheduled_signals([row], now_monotonic=1000.0)

    kept, suppressed = suppress_recent_scheduled_signals(
        [dict(row)],
        now_monotonic=1001.0,
    )
    assert kept == []
    assert suppressed == [row]

    changed = dict(row, entry_low=101.0, entry_high=101.2)
    kept, suppressed = suppress_recent_scheduled_signals(
        [changed],
        now_monotonic=1001.0,
    )
    assert kept == [changed]
    assert suppressed == []

    kept, suppressed = suppress_recent_scheduled_signals(
        [dict(row)],
        now_monotonic=1000.0 + 90 * 60 + 1,
    )
    assert kept == [row]
    assert suppressed == []


def test_portfolio_risk_cap_keeps_highest_ranked_setups():
    rows = [
        {"symbol": "BTC", "rank_score": 90, "risk_pct": 1.0},
        {"symbol": "ETH", "rank_score": 85, "risk_pct": 0.5},
        {"symbol": "SOL", "rank_score": 80, "risk_pct": 1.0},
        {"symbol": "SUI", "rank_score": 75, "risk_pct": 0.5},
    ]
    kept, excluded = cap_signals_by_portfolio_risk(
        rows,
        max_open_risk_pct=2.0,
    )
    assert [row["symbol"] for row in kept] == ["BTC", "ETH", "SUI"]
    assert [row["symbol"] for row in excluded] == ["SOL"]


def test_next_slot_datetime_future():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime(2026, 7, 21, 10, 0, tzinfo=ZoneInfo("Africa/Lagos"))
    nxt = next_slot_datetime(["05:00", "16:00", "20:00"], "Africa/Lagos", now=now)
    assert nxt.hour == 16


def test_session_schedule_follows_london_and_new_york_dst():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = load_config(ROOT / "config.yaml")
    summer_now = datetime(2026, 7, 23, 6, 0, tzinfo=ZoneInfo("UTC"))
    london, london_name = next_session_datetime(cfg.scheduler.sessions, now=summer_now)
    assert london_name == "London confirmation"
    assert london.strftime("%H:%M %Z") == "08:20 BST"
    assert london.astimezone(ZoneInfo("Africa/Lagos")).strftime("%H:%M") == "08:20"

    after_london = datetime(2026, 7, 23, 12, 0, tzinfo=ZoneInfo("UTC"))
    ny_macro, ny_name = next_session_datetime(cfg.scheduler.sessions, now=after_london)
    assert ny_name == "New York macro follow-through"
    assert ny_macro.strftime("%H:%M %Z") == "08:50 EDT"
    assert ny_macro.astimezone(ZoneInfo("Africa/Lagos")).strftime("%H:%M") == "13:50"

    after_macro = datetime(2026, 7, 23, 13, 0, tzinfo=ZoneInfo("UTC"))
    ny_open, ny_open_name = next_session_datetime(
        cfg.scheduler.sessions,
        now=after_macro,
    )
    assert ny_open_name == "New York open confirmation"
    assert ny_open.strftime("%H:%M %Z") == "09:50 EDT"

    winter_now = datetime(2026, 1, 15, 13, 30, tzinfo=ZoneInfo("UTC"))
    winter_macro, winter_name = next_session_datetime(
        cfg.scheduler.sessions,
        now=winter_now,
    )
    assert winter_name == "New York macro follow-through"
    assert winter_macro.strftime("%H:%M %Z") == "08:50 EST"
    assert winter_macro.astimezone(ZoneInfo("Africa/Lagos")).strftime("%H:%M") == "14:50"


def test_alert_destinations_include_primary_and_additional(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001111")
    monkeypatch.setenv(
        "TELEGRAM_ADDITIONAL_ALERT_CHAT_IDS",
        "-1002222, @private_channel -1001111",
    )
    assert get_telegram_alert_chat_ids() == [
        "-1001111",
        "-1002222",
        "@private_channel",
    ]
    assert get_telegram_alert_chat_ids(["999999"]) == ["999999"]


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
        "confidence": 84,
        "technical_confidence": 84,
        "llm_confidence": 90,
        "entry_status": "wait_retest",
        "execution_score": 79,
        "price": 112900,
        "entry_zone_relation": "favorable_beyond",
        "entry_low": 112300,
        "entry_high": 112500,
        "stop_loss": 111900,
        "take_profits": [113200, 113900, 114800, 116000],
        "leverage": 5,
        "risk_pct": 1,
        "setup_name": "Long Momentum",
        "entry_valid_for_minutes": 90,
        "entry_valid_until": "2026-07-28T14:30:00Z",
        "hold_hours_min": 1,
        "hold_hours_typical_max": 8,
        "hold_hours_max": 12,
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
            "primary_setup": {
                "hold_label": "Intraday",
                "hold_hours_max": 12,
                "risk_reward": [1.2, 2.4, 3.1, 4.0],
            },
        },
    }
    caption = format_signal_photo_caption(row, slot_label="New York open")
    assert "BTC LONG — RETEST" in caption
    assert "Overall Quality 84/100" in caption
    assert "Technical Quality 84/100" in caption
    assert "Execution Quality 79/100" in caption
    assert "Intraday" in caption
    assert "NY open" in caption
    assert "<b>Entry valid:</b> 90m · until 14:30 UTC" in caption
    assert "<b>Hold after fill:</b> 1–8h · hard max 12h" in caption
    assert "<b>Price at scan:</b> $112,900.00" in caption
    assert "already beyond Entry toward TP1" in caption
    assert "cancel at expiry" in caption
    assert "<b>Entry:</b>" in caption
    assert "<b>Entry mode:</b> RETEST ONLY" in caption
    assert "<b>Stop:</b>" in caption
    assert "<b>TP1:</b>" in caption
    assert "<b>Setup:</b> Long Momentum" in caption
    assert "<b>R:R (TP2):</b> gross 2.40R" in caption
    assert "<b>Why:</b>" in caption
    assert "<b>Beginner rule:</b>" in caption
    assert "after TP1, move it to breakeven" in caption
    assert "Educational" not in caption
    assert caption.endswith("NFA · DYOR · Trade at your own risk")
    assert len(caption) <= 1024


def test_signal_photo_caption_explains_cmp_ready_entry():
    row = {
        "symbol": "ETH/USDT:USDT",
        "direction": "short",
        "confidence": 88,
        "technical_confidence": 86,
        "entry_status": "ready",
        "execution_score": 82,
        "price": 3804,
        "entry_zone_relation": "inside",
        "entry_low": 3800,
        "entry_high": 3810,
        "stop_loss": 3830,
        "take_profits": [3770, 3740],
        "leverage": 5,
        "risk_pct": 0.5,
        "primary_tf": "15m",
        "setup_name": "Short Momentum",
        "reason": "Bearish structure and execution align.",
        "payload": {
            "primary_setup": {
                "hold_label": "Intraday",
                "hold_hours_max": 12,
                "risk_reward": [1.0, 2.0],
            },
            "execution": {"status": "ready"},
            "chart": {"timeframe": "15m"},
        },
    }
    caption = format_signal_photo_caption(row)
    assert "ETH SHORT — CMP CONFIRMATION" in caption
    assert "<b>Price at scan:</b> $3,804.00 · inside Entry zone" in caption
    assert "<b>Entry mode:</b> CMP PENDING" in caption
    assert "closed candle before the tracker records a fill" in caption
    assert "candle closes above Stop" in caption


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
    monkeypatch.setattr(
        scan_job,
        "next_session_datetime",
        lambda *a, **k: (due, "Test session"),
    )
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
        "confidence": 82,
        "llm_confidence": 82,
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
    monkeypatch.setattr(
        scan_job,
        "revalidate_candidate_for_delivery",
        lambda row, cfg: {"ok": True, "row": row, "reasons": []},
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


def test_scheduled_scan_sends_no_quality_setup_message(monkeypatch):
    from src.scheduler import scan_job

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.telegram.notify_on_empty is True
    monkeypatch.setattr(
        scan_job,
        "scan_symbols",
        lambda *a, **k: {"ok": True, "ranked_results": []},
    )
    messages = []
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda text, **kwargs: messages.append(text)
        or {"ok": True, "message_id": 55},
    )

    result = scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="London open",
        send=True,
    )
    assert messages
    assert "NO QUALITY SETUP" in messages[0]
    assert result["telegram_sent"] is True
    assert result["telegram_delivery_status"] == "sent_empty_report"


def test_scheduled_scan_sends_chart_alert_without_text_fallback(monkeypatch):
    from src.scheduler import scan_job

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456")
    cfg = load_config(ROOT / "config.yaml")
    row = {
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "confidence": 84,
        "llm_confidence": 84,
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
    monkeypatch.setattr(
        scan_job,
        "revalidate_candidate_for_delivery",
        lambda row, cfg: {"ok": True, "row": row, "reasons": []},
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


def test_scheduled_scan_fans_out_but_manual_override_stays_private(monkeypatch):
    from src.scheduler import scan_job

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:secret-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1001111")
    monkeypatch.setenv("TELEGRAM_ADDITIONAL_ALERT_CHAT_IDS", "-1002222")
    cfg = load_config(ROOT / "config.yaml")
    row = {
        "symbol": "ETH/USDT:USDT",
        "direction": "short",
        "confidence": 86,
        "technical_confidence": 84,
        "llm_confidence": 88,
        "rank_score": 81,
        "prop_safe": True,
        "signal_eligible": True,
        "entry_status": "wait_retest",
        "execution_score": 79,
        "payload": {"chart": {"candles": [{}] * 10}},
    }
    monkeypatch.setattr(
        scan_job,
        "scan_symbols",
        lambda *a, **k: {"ok": True, "ranked_results": [row]},
    )
    monkeypatch.setattr(
        scan_job,
        "revalidate_candidate_for_delivery",
        lambda row, cfg: {"ok": True, "row": row, "reasons": []},
    )
    monkeypatch.setattr(scan_job, "render_signal_chart_png", lambda row: b"png")
    monkeypatch.setattr(
        scan_job,
        "format_signal_photo_caption",
        lambda row, slot_label="": "<b>ETH SHORT</b>",
    )
    photo_chats = []
    monkeypatch.setattr(
        scan_job,
        "send_telegram_photo_detailed",
        lambda photo, caption, **kwargs: photo_chats.append(kwargs["chat_id"])
        or {"ok": True, "message_id": len(photo_chats)},
    )
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback called")),
    )

    scheduled = scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="scheduled",
        send=True,
    )
    assert photo_chats == ["-1001111", "-1002222"]
    assert scheduled["telegram_delivery"]["destination_count"] == 2
    assert scheduled["telegram_delivery_status"] == "sent_chart_alerts"

    photo_chats.clear()
    manual = scan_job.run_scheduled_scan_once(
        cfg,
        slot_label="manual",
        send=True,
        telegram_chat_ids=["999999"],
    )
    assert photo_chats == ["999999"]
    assert manual["telegram_delivery"]["destination_count"] == 1
    assert manual["telegram_delivery_status"] == "sent_chart_alerts"
