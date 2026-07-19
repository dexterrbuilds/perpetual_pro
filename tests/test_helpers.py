"""Unit tests for helpers and pure analysis logic (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.helpers import (
    clamp,
    normalize_symbol,
    symbol_base,
    timeframe_to_minutes,
)
from src.analysis.patterns import PatternDetector
from src.analysis.market_structure import MarketStructureAnalyzer
from src.analysis.risk import RiskManager
from src.utils.config import load_config


def test_normalize_symbol_variants():
    assert normalize_symbol("btc") == "BTC/USDT:USDT"
    assert normalize_symbol("BTCUSDT") == "BTC/USDT:USDT"
    assert normalize_symbol("ETH/USDT") == "ETH/USDT:USDT"
    assert normalize_symbol("SOL/USDT:USDT") == "SOL/USDT:USDT"
    assert normalize_symbol("1000PEPEUSDT") == "1000PEPE/USDT:USDT"
    assert normalize_symbol("BONK") == "BONK/USDT:USDT"
    assert normalize_symbol("BONK/USD") == "BONK/USDT:USDT"


def test_symbol_base():
    assert symbol_base("BTC/USDT:USDT") == "BTC"
    assert symbol_base("ethusdt") == "ETH"


def test_timeframe_minutes():
    assert timeframe_to_minutes("15m") == 15
    assert timeframe_to_minutes("1h") == 60
    assert timeframe_to_minutes("4h") == 240
    assert timeframe_to_minutes("1d") == 1440


def test_clamp():
    assert clamp(5, 0, 1) == 1
    assert clamp(-1, 0, 1) == 0
    assert clamp(0.5, 0, 1) == 0.5


def _synthetic_ohlcv(n: int = 200, trend: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(trend * 0.05, 0.8, size=n))
    high = close + rng.uniform(0.1, 1.0, size=n)
    low = close - rng.uniform(0.1, 1.0, size=n)
    open_ = close + rng.normal(0, 0.3, size=n)
    vol = rng.uniform(100, 1000, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_pattern_detector_runs():
    df = _synthetic_ohlcv()
    report = PatternDetector().detect(df)
    assert report is not None
    assert isinstance(report.hits, list)


def test_structure_analyzer_runs():
    df = _synthetic_ohlcv(250, trend=1.0)
    report = MarketStructureAnalyzer().analyze(df)
    assert report.trend in ("up", "down", "range")
    assert -1 <= report.structure_score <= 1


def test_risk_manager_long_plan():
    rm = RiskManager(simulated_capital=1000, risk_pct=1.0)
    plan = rm.build_plan(
        "long",
        price=100.0,
        atr=2.0,
        confidence=70,
        funding_rate=0.0001,
        primary_tf="15m",
    )
    assert plan.direction == "long"
    assert plan.stop_loss < 100
    assert len(plan.take_profits) >= 4  # TP1–TP4
    assert plan.position_size_units > 0
    assert plan.primary_rr > 0
    assert plan.simulated_capital == 1000
    # Aggressive perp band: never sub-20x
    assert plan.leverage_suggested >= 20
    assert plan.leverage_suggested <= 100
    assert len(plan.potential_profits) >= 4
    assert plan.is_simulation is True
    assert plan.hold_detail
    assert plan.hold_hours_max <= 24 or "swing" in plan.hold_label.lower()
    headline = plan.setup_headline("BTC/USDT:USDT")
    assert "LONG SETUP" in headline
    assert plan.alternative_entry_low is not None
    setup = plan.to_primary_setup()
    assert setup["tp1"] is not None
    assert setup["tp4"] is not None
    sim = plan.to_position_simulation()
    assert sim["risk_amount"] > 0


def test_load_config_defaults():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.exchange.default
    assert cfg.risk.risk_per_trade_pct > 0
    assert cfg.timeframes.primary
