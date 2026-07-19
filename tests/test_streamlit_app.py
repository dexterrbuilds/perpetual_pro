from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app import build_report_markdown, normalize_symbol_list


def test_normalize_symbol_list_filters_empty_entries():
    assert normalize_symbol_list("BTC, ETH , , SOL") == ["BTC", "ETH", "SOL"]


def test_build_report_markdown_contains_key_sections():
    payload = {
        "symbol": "BTC/USDT:USDT",
        "direction": "LONG",
        "confidence": 81.2,
        "setup_name": "Breakout",
        "trade_plan": {"entry": 50000, "stop_loss": 48000, "take_profit": 52000},
        "signal": {"confluence_score": 0.89},
    }
    report = build_report_markdown(payload)
    assert "BTC/USDT:USDT" in report
    assert "Breakout" in report
    assert "Entry" in report
