"""Regression coverage for the approved Phase 1 correctness/safety changes."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis.confluence import ConfluenceEngine
from src.analysis.indicators import _session_vwap
from src.analysis.revalidation import evaluate_pre_delivery_candidate
from src.analysis.risk import RiskManager
from src.api import service
from src.scheduler import scan_job
from src.api.service import AnalyzeRequest
from src.data.exchange import (
    ExchangeClient,
    MarketSnapshot,
    _ohlcv_cache_ttl_seconds,
)
from src.scoring.features import (
    FEATURE_SCHEMA_VERSION,
    build_candidate_record,
    extract_candidate_features,
)
from src.scoring.labels import (
    label_candidate_from_ohlcv,
    technical_success_from_range,
)
from src.scoring.training import _prepare_rows
from src.tracking.signal_tracker import SignalStore, format_tracker_event
from src.utils.config import load_config
from src.utils.helpers import safe_float


def _config():
    return load_config()


def _signal_row(now: datetime, **overrides):
    row = {
        "symbol": "BTC/USDT:USDT",
        "exchange": "okx",
        "direction": "short",
        "primary_tf": "15m",
        "confidence": 86,
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 103.0,
        "take_profits": [98.0, 96.0],
        "entry_status": "wait_retest",
        "price": 102.0,
        "atr": 2.0,
        "signal_generated_at": now.isoformat(),
        "entry_valid_until": (now + timedelta(minutes=90)).isoformat(),
        "entry_valid_for_minutes": 90,
        "hold_hours_max": 8,
    }
    row.update(overrides)
    return row


def test_request_risk_overrides_do_not_mutate_shared_config():
    cfg = _config()
    original_capital = cfg.risk.simulated_capital
    original_risk = cfg.risk.risk_per_trade_pct
    first = RiskManager(config=cfg, simulated_capital=25_000, risk_pct=0.5)
    second = RiskManager(config=cfg, simulated_capital=5_000, risk_pct=0.8)

    assert cfg.risk.simulated_capital == original_capital
    assert cfg.risk.risk_per_trade_pct == original_risk
    assert first.risk.simulated_capital == 25_000
    assert second.risk.simulated_capital == 5_000
    first.risk.risk_per_trade_pct = 0.25
    assert second.risk.risk_per_trade_pct == 0.8


def test_short_sparse_crossing_records_entry_before_tp1(tmp_path):
    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    store = SignalStore(tmp_path / "tracker.db", target_allocations=[0.5, 0.5])
    signal, _ = store.register_signal(_signal_row(now), ["1"], source="test", now=now)

    assert store.process_price(
        signal["symbol"], 98.0, now + timedelta(minutes=1), source="websocket"
    ) == 2
    active = store.active_signals()[0]
    assert active["status"] == "entered"
    assert active["highest_tp"] == 1
    assert [event["event_type"] for event in store.pending_events(retry_after_seconds=0)] == [
        "entered",
        "target_hit",
    ]
    store.close()


def test_missing_prior_observation_uses_ambiguous_gap_not_missed(tmp_path):
    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    store = SignalStore(tmp_path / "tracker.db", target_allocations=[0.5, 0.5])
    signal, _ = store.register_signal(
        _signal_row(now, price=0.0), ["1"], source="test", now=now
    )
    assert store.process_price(
        signal["symbol"], 98.0, now + timedelta(minutes=1), source="rest"
    ) == 1
    terminal = store.signals_for_sync()[0]
    assert terminal["status"] == "ambiguous_gap"
    event = store.pending_events(retry_after_seconds=0)[0]
    assert "OUTCOME AMBIGUOUS" in format_tracker_event(event)
    store.close()


def test_ohlcv_cache_ttl_stops_at_newest_candle_close():
    frame = pd.DataFrame(
        {"close": [100.0]},
        index=pd.to_datetime(["2026-08-01T12:00:00Z"]),
    )
    now_ms = int(pd.Timestamp("2026-08-01T12:14:30Z").timestamp() * 1000)
    ttl = _ohlcv_cache_ttl_seconds(frame, "15m", 300, now_ms=now_ms)
    assert 29 <= ttl <= 31


def test_nonfinite_values_are_rejected_from_helpers_ticker_and_ohlcv():
    assert safe_float(float("nan"), 7.0) == 7.0
    assert safe_float(float("inf"), 7.0) == 7.0
    assert safe_float(float("-inf"), 7.0) == 7.0

    class DummyExchange:
        has = {}

        def fetch_ticker(self, symbol):
            return {"last": float("nan"), "close": 101.0}

        def fetch_ohlcv(self, symbol, **kwargs):
            return [
                [1_700_000_000_000, 100, 101, 99, 100.5, 10],
                [1_700_000_900_000, 100, math.inf, 99, 100.5, 10],
            ]

    client = ExchangeClient.__new__(ExchangeClient)
    client.exchange_id = "phase1-nonfinite"
    client._exchange = DummyExchange()
    client._markets_loaded = True
    client.cache_ttl_seconds = 0
    client.resolve_symbol = lambda symbol: symbol
    client._build_symbol_candidates = lambda normalized, original: []

    assert client.fetch_ticker("BTC")["close"] == 101.0
    candles = client.fetch_ohlcv("BTC", force_refresh=True)
    assert len(candles) == 1
    assert math.isfinite(float(candles["high"].iloc[0]))


def test_market_snapshot_source_age_is_recomputed_and_validated():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    snap = MarketSnapshot(
        symbol="BTC/USDT:USDT",
        exchange_id="okx",
        ticker_timestamp=now_ms - 60_000,
        orderbook_timestamp=now_ms - 5_000,
    )
    snap.refresh_source_ages(
        max_ticker_age_seconds=45,
        max_orderbook_age_seconds=30,
    )
    assert snap.ticker_age_seconds >= 59
    assert snap.orderbook_age_seconds < 10
    assert snap.execution_data_fresh is False


def test_session_vwap_resets_at_midnight_utc():
    frame = pd.DataFrame(
        {
            "high": [101.0, 103.0, 201.0],
            "low": [99.0, 101.0, 199.0],
            "close": [100.0, 102.0, 200.0],
            "volume": [1.0, 1.0, 2.0],
        },
        index=pd.to_datetime(
            [
                "2026-07-31T23:30:00Z",
                "2026-07-31T23:45:00Z",
                "2026-08-01T00:00:00Z",
            ]
        ),
    )
    vwap = _session_vwap(frame)
    assert vwap.iloc[1] == pytest.approx(101.0)
    assert vwap.iloc[2] == pytest.approx(200.0)


def test_opposing_bos_cannot_create_breakout_setup_label():
    engine = ConfluenceEngine(_config())
    indicators = SimpleNamespace(
        summary={
            "adx": 25,
            "rsi": 50,
            "trend_score": -0.5,
            "momentum_score": -0.4,
            "bb_position": 0.5,
            "vol_ratio": 1.0,
        }
    )
    structure = SimpleNamespace(
        last_bos="bullish",
        last_choch=None,
        volume_profile_poc=None,
    )
    tags = engine._strategy_tags(
        indicators,
        structure,
        SimpleNamespace(hits=[]),
        SimpleNamespace(primary_tf="15m"),
        "short",
        80,
    )
    assert "breakout" not in tags
    assert "breakdown" not in tags
    assert "breakout_retest" not in tags
    assert "Breakout" not in engine._setup_name(tags, "short", structure)


def _delivery_row(now: datetime) -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
        "exchange": "okx",
        "direction": "long",
        "primary_tf": "15m",
        "entry_low": 100.0,
        "entry_high": 101.0,
        "stop_loss": 98.0,
        "take_profits": [103.0, 105.0],
        "entry_valid_until": (now + timedelta(hours=1)).isoformat(),
    }


def test_pre_delivery_revalidation_updates_cmp_state_and_rejects_stale_sources():
    cfg = _config()
    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    accepted = evaluate_pre_delivery_candidate(
        _delivery_row(now),
        current_price=100.5,
        spread_bps=2.0,
        ticker_age_seconds=2.0,
        orderbook_age_seconds=3.0,
        latest_closed_price=100.4,
        candle_data_ok=True,
        config=cfg,
        now=now,
    )
    assert accepted["ok"] is True
    assert accepted["row"]["entry_status"] == "confirmation_pending"

    rejected = evaluate_pre_delivery_candidate(
        _delivery_row(now),
        current_price=103.0,
        spread_bps=20.0,
        ticker_age_seconds=60.0,
        orderbook_age_seconds=40.0,
        latest_closed_price=97.5,
        candle_data_ok=False,
        config=cfg,
        now=now,
    )
    assert rejected["ok"] is False
    assert {
        "PRE_SEND_TICKER_STALE",
        "PRE_SEND_ORDERBOOK_STALE",
        "PRE_SEND_SPREAD_TOO_WIDE",
        "PRE_SEND_CANDLES_STALE",
        "PRE_SEND_STRUCTURE_INVALIDATED",
        "PRE_SEND_TP1_ALREADY_TRADED",
    }.issubset(set(rejected["reasons"]))


def test_directional_features_use_signal_support_sign_and_schema_isolated():
    common = {
        "price": 100.0,
        "atr": 2.0,
        "entry_low": 99.0,
        "entry_high": 100.0,
        "stop_loss": 97.0,
        "take_profits": [103.0, 105.0],
        "entry_status": "wait_retest",
        "execution": {"status": "wait_retest"},
    }
    long_features, _ = extract_candidate_features(
        None,
        {**common, "direction": "long", "factors": [{"name": "trend", "score": 0.8}]},
    )
    short_features, _ = extract_candidate_features(
        None,
        {**common, "direction": "short", "factors": [{"name": "trend", "score": -0.8}]},
    )
    assert long_features["trend"] == pytest.approx(0.8)
    assert short_features["trend"] == pytest.approx(0.8)
    assert FEATURE_SCHEMA_VERSION == "3.0"

    with pytest.raises(ValueError, match="Feature schema mismatch"):
        build_candidate_record(
            None,
            {
                **common,
                "symbol": "BTC/USDT:USDT",
                "direction": "long",
                "signal_generated_at": "2026-08-01T12:00:00+00:00",
            },
            source="test",
            feature_schema_version="2.0",
        )
    rows = [
        {"feature_schema_version": "2.0", "generated_at": "2026-01-01T00:00:00Z", "features": {"trend": 1}},
        {"feature_schema_version": "3.0", "generated_at": "2026-01-02T00:00:00Z", "features": {"trend": 1}},
    ]
    assert len(_prepare_rows(rows, "3.0")) == 1


def test_forward_and_replay_share_adverse_first_technical_success(tmp_path):
    assert technical_success_from_range(
        direction="long", start_price=100, atr=2, high=103, low=97
    ) is False
    candidate = {
        "generated_at": "2026-08-01T12:00:00+00:00",
        "direction": "long",
        "decision": {
            "price": 100,
            "atr": 2,
            "entry_low": 99,
            "entry_high": 100,
            "stop_loss": 97,
            "take_profits": [103],
            "entry_valid_until": "2026-08-01T13:00:00+00:00",
        },
    }
    replay = label_candidate_from_ohlcv(
        candidate,
        pd.DataFrame(
            [{"open": 100, "high": 103, "low": 97, "close": 98}],
            index=pd.to_datetime(["2026-08-01T12:15:00Z"]),
        ),
    )
    assert replay.technical_success is False

    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    store = SignalStore(tmp_path / "tracker.db", target_allocations=[0.5, 0.5])
    signal, _ = store.register_signal(
        _signal_row(
            now,
            direction="long",
            price=100,
            atr=2,
            entry_low=99,
            entry_high=100,
            stop_loss=97,
            take_profits=[103, 105],
            entry_status="confirmation_pending",
        ),
        ["1"],
        source="test",
        now=now,
    )
    store.process_closed_candle(
        signal["symbol"],
        98,
        now + timedelta(minutes=15),
        high_price=103,
        low_price=97,
    )
    assert store.signals_for_sync()[0]["technical_success"] == 0
    store.close()


def test_complete_scan_failure_is_not_reported_as_legitimate_empty(monkeypatch):
    def fail_fetch(*args, **kwargs):
        raise RuntimeError("venue unavailable")

    monkeypatch.setattr(service, "fetch_multi_timeframe_with_fallback", fail_fetch)
    result = service.scan_symbols(
        ["BTC", "ETH"],
        request=AnalyzeRequest(no_news=True, use_llm=False),
        config=_config(),
    )
    assert result["ok"] is False
    assert result["error"] == "scan_analysis_unavailable"
    assert result["analyzed_count"] == 0
    assert len(result["analysis_failures"]) == 2


def test_scheduler_reports_complete_failure_differently_from_no_setup(monkeypatch):
    cfg = _config()
    cfg.telegram.notify_on_empty = False
    monkeypatch.setenv("DELIVERY_MODE", "public")
    monkeypatch.setenv("TELEGRAM_COMMAND_CHAT_IDS", "1")
    monkeypatch.setattr(scan_job, "is_telegram_ready", lambda config: True)
    monkeypatch.setattr(scan_job, "get_telegram_alert_chat_ids", lambda override=None: ["1"])
    monkeypatch.setattr(
        scan_job,
        "scan_symbols",
        lambda *args, **kwargs: {
            "ok": False,
            "error": "scan_analysis_unavailable",
            "ranked_results": [],
        },
    )
    messages = []
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda message, **kwargs: messages.append(message) or {"ok": True},
    )

    result = scan_job.run_scheduled_scan_once(cfg, slot_label="manual phase1", send=True)
    assert result["telegram_delivery_status"] == "sent_scan_failure"
    assert "SCAN UNAVAILABLE" in messages[0]
    assert "not a no-setup result" in messages[0]


def test_scheduler_revalidation_rejects_before_chart_or_signal_delivery(monkeypatch):
    cfg = _config()
    monkeypatch.setattr(scan_job, "is_telegram_ready", lambda config: True)
    monkeypatch.setattr(scan_job, "get_telegram_alert_chat_ids", lambda override=None: ["1"])
    row = {
        "symbol": "BTC/USDT:USDT",
        "direction": "long",
        "confidence": 85,
        "rank_score": 75,
        "technical_confidence": 84,
        "execution_score": 78,
        "signal_eligible": True,
        "entry_status": "wait_retest",
        "prop_safe": True,
        "market_quality_ok": True,
        "data_quality_ok": True,
        "historical_edge_ok": True,
        "immediate_sl_risk": 20,
        "chase_distance_atr": 0.2,
        "spread_bps": 2,
        "risk_reward": [1.0, 1.5],
    }
    monkeypatch.setattr(
        scan_job,
        "scan_symbols",
        lambda *args, **kwargs: {"ok": True, "ranked_results": [row]},
    )
    monkeypatch.setattr(
        scan_job,
        "revalidate_candidate_for_delivery",
        lambda candidate, config: {
            "ok": False,
            "row": candidate,
            "reasons": ["PRE_SEND_TP1_ALREADY_TRADED"],
        },
    )
    monkeypatch.setattr(
        scan_job,
        "render_signal_chart_png",
        lambda candidate: (_ for _ in ()).throw(AssertionError("chart rendered")),
    )
    monkeypatch.setattr(
        scan_job,
        "send_telegram_message_detailed",
        lambda message, **kwargs: {"ok": True},
    )

    result = scan_job.run_scheduled_scan_once(cfg, slot_label="manual phase1", send=True)
    assert result["alert_count"] == 0
    assert result["pre_delivery_rejected_count"] == 1
    assert result["pre_delivery_rejected"][0][
        "pre_delivery_rejection_reasons"
    ] == ["PRE_SEND_TP1_ALREADY_TRADED"]
