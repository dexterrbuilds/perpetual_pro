"""Strict setup-family recognition and safety-pipeline regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.execution import _canonical_setup_type, build_execution_profile
from src.analysis.execution_policy import DEFAULT_EXECUTION_POLICY
from src.analysis.indicators import IndicatorSuite
from src.analysis.market_structure import StructureLevel, StructureReport
from src.analysis.setups import SETUP_POLICY_VERSION, detect_strict_setup
from src.notify.telegram import format_signal_photo_caption
from src.utils.config import load_config


def _frame(n: int = 60) -> pd.DataFrame:
    index = pd.date_range("2026-08-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(99.2, 100.0, n)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.15,
            "low": close - 0.15,
            "close": close,
            "volume": np.full(n, 100.0),
        },
        index=index,
    )


def _suite(df: pd.DataFrame, **updates) -> IndicatorSuite:
    summary = {
        "close": float(df.close.iloc[-1]),
        "atr": 1.0,
        "ema_fast": 99.95,
        "ema_mid": 99.85,
        "vwap": 99.80,
        "trend_score": 0.60,
        "momentum_score": 0.12,
        "adx": 26.0,
        "rsi": 52.0,
    }
    summary.update(updates)
    return IndicatorSuite(df=df, summary=summary)


def test_strict_trend_continuation_pullback_is_highest_priority():
    df = _frame()
    df.iloc[-1, df.columns.get_loc("open")] = 99.72
    df.iloc[-1, df.columns.get_loc("low")] = 99.62
    df.iloc[-1, df.columns.get_loc("high")] = 100.18
    df.iloc[-1, df.columns.get_loc("close")] = 100.10
    df.iloc[-1, df.columns.get_loc("volume")] = 125.0
    setup = detect_strict_setup(
        df, _suite(df), StructureReport(trend="up", structure_score=.7),
        direction="long", price=100.10, atr=1.0,
    )
    assert setup is not None
    assert setup.setup_type == "trend_pullback"
    assert setup.label == "Long Trend Continuation (Pullback)"
    assert setup.policy_version == SETUP_POLICY_VERSION


def test_strict_trend_pullback_detection_is_long_short_symmetric():
    df = _frame()
    df.iloc[-1] = [100.28, 100.38, 99.82, 99.90, 125.0]
    setup = detect_strict_setup(
        df,
        _suite(
            df, close=99.90, ema_fast=100.02, ema_mid=100.10, vwap=100.15,
            trend_score=-.60, momentum_score=-.12, adx=26,
        ),
        StructureReport(trend="down", structure_score=-.7),
        direction="short", price=99.90, atr=1.0,
    )
    assert setup is not None
    assert setup.setup_type == "trend_pullback"
    assert setup.label == "Short Trend Continuation (Pullback)"


def test_strict_breakout_requires_separate_break_and_retest_candles():
    df = _frame()
    df.loc[df.index[-30:-3], ["high", "low", "close", "open"]] = [100.0, 99.6, 99.8, 99.75]
    df.iloc[-3] = [99.85, 100.70, 99.80, 100.55, 180.0]
    df.iloc[-2] = [100.50, 100.65, 100.25, 100.40, 120.0]
    df.iloc[-1] = [100.08, 100.35, 99.98, 100.28, 130.0]
    setup = detect_strict_setup(
        df,
        _suite(df, trend_score=0.05, momentum_score=0.10, adx=15),
        StructureReport(trend="range"),
        direction="long", price=100.28, atr=1.0,
    )
    assert setup is not None
    assert setup.setup_type == "breakout_retest"
    assert setup.evidence["breakout_volume_ratio"] >= 1.15

    no_prior_break = df.copy()
    no_prior_break.iloc[-3] = [99.85, 100.0, 99.75, 99.90, 100.0]
    no_prior_break.iloc[-2] = [99.85, 100.0, 99.75, 99.90, 100.0]
    assert detect_strict_setup(
        no_prior_break,
        _suite(no_prior_break, trend_score=0.05, momentum_score=0.10, adx=15),
        StructureReport(trend="range"),
        direction="long", price=100.28, atr=1.0,
    ) is None


def test_fresh_order_block_fvg_retest_requires_unconsumed_zone():
    df = _frame()
    df.iloc[-1] = [99.85, 100.20, 99.78, 100.12, 125.0]
    fresh = StructureLevel(
        "order_block", "bullish", 99.75, 100.0, 84,
        age_bars=8, touch_count=1, mitigation_fraction=.55,
    )
    setup = detect_strict_setup(
        df,
        _suite(df, trend_score=0.05, momentum_score=0.05, adx=14),
        StructureReport(trend="range", levels=[fresh]),
        direction="long", price=100.12, atr=1.0,
    )
    assert setup is not None and setup.setup_type == "ob_fvg_retest"

    fresh.fully_mitigated = True
    fresh.relevant = False
    assert detect_strict_setup(
        df,
        _suite(df, trend_score=0.05, momentum_score=0.05, adx=14),
        StructureReport(trend="range", levels=[fresh]),
        direction="long", price=100.12, atr=1.0,
    ) is None


def test_liquidity_sweep_requires_breach_reclaim_wick_and_volume():
    df = _frame()
    df.loc[df.index[-21:-1], "low"] = 100.0
    df.loc[df.index[-21:-1], "high"] = 100.5
    df.loc[df.index[-21:-1], "close"] = 100.2
    df.loc[df.index[-21:-1], "open"] = 100.2
    df.iloc[-1] = [100.15, 100.38, 99.84, 100.31, 130.0]
    setup = detect_strict_setup(
        df,
        _suite(df, trend_score=0.0, momentum_score=0.0, adx=12),
        StructureReport(trend="range"),
        direction="long", price=100.31, atr=1.0,
    )
    assert setup is not None
    assert setup.setup_type == "liquidity_sweep"
    assert setup.evidence["breach_atr"] >= .03


def test_counter_trend_reversal_uses_stronger_confirmation():
    df = _frame()
    df.loc[df.index[-21:-1], "low"] = 98.8  # prevents sweep classification
    df.iloc[-1] = [99.75, 100.12, 99.30, 100.02, 130.0]
    setup = detect_strict_setup(
        df,
        _suite(df, trend_score=-.35, momentum_score=.10, adx=24, rsi=38),
        StructureReport(trend="down", last_choch="bullish", structure_score=-.4),
        direction="long", price=100.02, atr=1.0,
    )
    assert setup is not None
    assert setup.setup_type == "reversal"
    assert "Counter-Trend Reversal" in setup.label

    weak = detect_strict_setup(
        df,
        _suite(df, trend_score=-.35, momentum_score=.10, adx=24, rsi=50),
        StructureReport(trend="down", last_choch=None, structure_score=-.4),
        direction="long", price=100.02, atr=1.0,
    )
    assert weak is None


def test_new_setup_types_reuse_existing_global_safety_thresholds():
    for setup_type in (
        "trend_pullback", "breakout_retest", "ob_fvg_retest", "liquidity_sweep"
    ):
        values = DEFAULT_EXECUTION_POLICY.setup_weights[setup_type]
        assert set(values) == set(DEFAULT_EXECUTION_POLICY.component_weights)
        assert sum(values.values()) == 1.0
    assert DEFAULT_EXECUTION_POLICY.minimum_execution_quality == 72.0
    assert DEFAULT_EXECUTION_POLICY.max_spread_bps == 12.0
    assert _canonical_setup_type(
        "Long Order Block / FVG Retest", ["ob_fvg_retest"], StructureReport()
    ) == "ob_fvg_retest"
    config = load_config("config.yaml")
    assert config.analysis.execution_min_score == 72
    assert config.analysis.max_immediate_sl_risk == 32
    assert config.analysis.directional_score_threshold == .20
    assert config.risk.min_rr == 1.25


def test_unconfirmed_strict_label_fails_closed_in_execution():
    df = _frame()
    profile = build_execution_profile(
        df,
        _suite(df, trend_score=0.0, momentum_score=0.0, adx=10),
        StructureReport(trend="range"),
        direction="long",
        price=float(df.close.iloc[-1]),
        atr=1.0,
        setup_name="Long Liquidity Sweep",
        strategy_tags=["liquidity_sweep", "strict_setup_confirmed"],
    )
    assert "setup_confirmation_insufficient" in profile.hard_failures
    assert profile.status == "blocked"


def test_telegram_preserves_clear_new_setup_label():
    caption = format_signal_photo_caption({
        "symbol": "BTC/USDT:USDT", "direction": "long", "confidence": 82,
        "technical_confidence": 84, "execution_quality": 78,
        "entry_status": "wait_retest", "entry_low": 100, "entry_high": 101,
        "stop_loss": 98, "take_profits": [103], "risk_pct": .5,
        "setup_name": "Long Trend Continuation (Pullback)",
        "payload": {"execution": {}},
    })
    assert "Trend Continuation (Pullback)" in caption
    assert len(caption) <= 1024
