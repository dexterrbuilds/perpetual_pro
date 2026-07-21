"""Telegram report formatter + env-only credentials (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notify.telegram import (
    format_prop_scan_report,
    get_telegram_credentials,
    is_telegram_ready,
)
from src.scheduler.scan_job import filter_high_confidence, next_slot_datetime
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
                "hold_label": "2–8h day trade",
                "backtest": {
                    "sample_ok": True,
                    "win_rate": 60,
                    "profit_factor": 1.4,
                    "n_trades": 10,
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
