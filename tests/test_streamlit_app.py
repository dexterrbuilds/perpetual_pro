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
        "direction": "long",
        "confidence": 81.2,
        "setup_name": "Breakout",
        "exchange": "bybit",
        "display_leverage": 5,
        "model_leverage": 45,
        "trade_plan": {
            "direction": "long",
            "entry_low": 50000,
            "entry_high": 50200,
            "stop_loss": 48000,
            "take_profits": [51000, 52000, 53000, 54000],
            "risk_reward": [0.5, 1.0, 1.5, 2.0],
            "leverage_suggested": 45,
            "hold_label": "Intraday",
            "hold_detail": "Suggested hold: 1–8 hours",
            "invalidation": "Close below 48000",
            "quality": "good",
            "simulated_capital": 1000.0,
            "risk_pct": 1.0,
            "risk_amount": 10.0,
            "position_size_notional": 4500.0,
            "margin_required": 100.0,
            "potential_profits": [50.0, 100.0, 150.0, 200.0],
            "potential_profit_pcts": [5.0, 10.0, 15.0, 20.0],
        },
        "signal": {"confluence_score": 0.89},
        "key_reasons": ["Momentum aligned"],
        "key_risks": ["High ATR"],
    }
    report = build_report_markdown(payload)
    assert "BTC/USDT:USDT" in report
    assert "Breakout" in report
    assert "Entry zone" in report
    assert "Stop loss" in report
    assert "TP1" in report
    assert "TP4" in report
    assert "R:R" in report
    assert "leverage" in report.lower()
    assert "Simulation example" in report
    assert "$100" in report or "100" in report
    assert "Hold" in report
