from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app import (
    build_report_markdown,
    format_ticker_price,
    normalize_symbol_list,
    send_manual_telegram_test,
)


def test_normalize_symbol_list_filters_empty_entries():
    assert normalize_symbol_list("BTC, ETH , , SOL") == ["BTC", "ETH", "SOL"]


def test_format_ticker_price_shows_base_and_price():
    assert format_ticker_price("BTC/USDT:USDT", 112450.25) == "BTC $112,450"
    assert format_ticker_price("ETH", 3456.78).startswith("ETH $")
    assert format_ticker_price("SOL", None) == "SOL —"


def test_manual_telegram_button_uses_local_fallback(monkeypatch):
    monkeypatch.delenv("TELEGRAM_TEST_KEY", raising=False)
    monkeypatch.setattr(
        "src.notify.telegram.send_test_telegram_alert",
        lambda source: {"ok": True, "source": source, "delivery": {"message_id": 9}},
    )
    result = send_manual_telegram_test()
    assert result["ok"] is True
    assert result["test_path"] == "streamlit_process"


def test_build_report_markdown_contains_key_sections():
    payload = {
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "confidence": 81.2,
        "technical_confidence": 70.0,
        "llm_confidence": 78.0,
        "llm_confidence_reason": "Strong MTF alignment with expanding volume.",
        "rank_score": 74.5,
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
            "entry_status": "wait_retest",
            "entry_reason": "Wait for the demand-zone retest.",
            "execution_score": 78,
            "immediate_sl_risk": 22,
            "chase_distance_atr": 0.6,
            "order_flow_score": 0.42,
        },
        "execution": {
            "status": "wait_retest",
            "score": 78,
            "entry_reason": "Wait for the demand-zone retest.",
            "immediate_sl_risk": 22,
            "chase_distance_atr": 0.6,
            "order_flow_score": 0.42,
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
    assert "LLM Confidence" in report
    assert "78" in report
    assert "Strong MTF alignment" in report
    assert "Entry quality" in report
    assert "Wait Retest" in report
    assert "Immediate-SL risk" in report
