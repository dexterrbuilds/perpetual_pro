"""Closed-candle execution zones, chart payloads, and anti-chase behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.execution import analyze_candles, build_execution_profile
from src.analysis.indicators import IndicatorSuite
from src.analysis.market_structure import StructureLevel, StructureReport
from src.analysis.risk import RiskManager
from src.report.charts import build_market_chart_payload, render_signal_chart_png
from src.utils.config import RiskConfig


def _execution_df(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(98.0, 100.7, n)
    open_ = close - 0.18
    high = close + 0.20
    low = close - 0.25
    volume = np.linspace(800.0, 1300.0, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "ema_fast": np.linspace(97.9, 100.15, n),
            "ema_mid": np.linspace(97.7, 100.0, n),
            "vwap": np.linspace(97.8, 99.95, n),
        },
        index=idx,
    )


def _structure() -> StructureReport:
    return StructureReport(
        trend="up",
        structure_score=0.7,
        volume_profile_poc=100.0,
        volume_profile_val=99.8,
        volume_profile_vah=101.7,
        swing_highs=[101.8, 102.6, 103.4],
        swing_lows=[98.8, 99.4],
        levels=[
            StructureLevel("order_block", "bullish", 99.82, 100.05, 88),
            StructureLevel("fvg", "bullish", 99.90, 100.12, 80),
            StructureLevel("liquidity", "bearish", 101.75, 101.85, 75),
            StructureLevel("resistance", "bearish", 102.55, 102.65, 70),
        ],
    )


def test_execution_profile_waits_for_retest_and_uses_structure_targets():
    df = _execution_df()
    suite = IndicatorSuite(
        df=df,
        summary={"ema_fast": 100.15, "ema_mid": 100.0, "vwap": 99.95, "atr": 1.0},
    )
    profile = build_execution_profile(
        df,
        suite,
        _structure(),
        direction="long",
        price=100.7,
        atr=1.0,
    )

    assert profile.status in ("ready", "wait_retest")
    assert profile.score >= 65
    assert profile.entry_high < 100.7
    assert profile.stop_loss < profile.entry_low
    assert len(profile.targets) == 4
    assert profile.targets == sorted(profile.targets)
    assert profile.targets[0] > profile.entry_high
    assert profile.immediate_sl_risk < 50
    assert profile.anchor_sources

    risk = RiskManager(
        risk_cfg=RiskConfig(prop_mode=True, max_leverage=5, leverage_ceiling=5),
        simulated_capital=1000,
        risk_pct=1.0,
    )
    plan = risk.build_plan(
        "long",
        price=100.7,
        atr=1.0,
        confidence=75,
        execution=profile.to_dict(),
    )
    assert plan.entry_low == profile.entry_low
    assert plan.entry_high == profile.entry_high
    assert plan.stop_loss == profile.stop_loss
    assert plan.take_profits == profile.targets
    assert plan.entry_status == profile.status
    entry_mid = (plan.entry_low + plan.entry_high) / 2.0
    expected_rr = (plan.take_profits[0] - entry_mid) / (entry_mid - plan.stop_loss)
    assert plan.risk_reward[0] == expected_rr

    chart = build_market_chart_payload(
        df, suite, _structure(), None, plan, timeframe="15m", limit=60
    )
    assert len(chart["candles"]) == 60
    assert chart["trade"]["entry_status"] == profile.status
    assert chart["trade"]["take_profits"] == profile.targets
    assert any(level["kind"] == "order_block" for level in chart["levels"])

    png = render_signal_chart_png(
        {
            "symbol": "BTC/USDT:USDT",
            "direction": "long",
            "primary_tf": "15m",
            "confidence": 78,
            "entry_status": profile.status,
            "payload": {"chart": chart},
        }
    )
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 10_000


def test_adverse_rejection_wick_is_detected():
    df = _execution_df()
    df.loc[df.index[-1], ["open", "high", "low", "close", "volume"]] = [
        100.4,
        102.2,
        99.9,
        100.0,
        2400.0,
    ]
    context = analyze_candles(df, atr=1.0, direction="long")
    assert context.adverse_rejection is True
    assert context.upper_wick_ratio > 0.45
    assert any("do not enter" in note.lower() for note in context.notes)
