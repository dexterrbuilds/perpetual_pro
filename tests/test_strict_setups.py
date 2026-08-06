"""Strict setup-family recognition and safety-pipeline regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.execution import (
    _canonical_setup_type,
    _entry_anchors,
    _select_anchor,
    build_execution_profile,
)
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
        "trend_pullback", "breakout_retest", "ob_fvg_retest", "liquidity_sweep",
        "opening_range_breakout", "session_high_low_rejection",
    ):
        values = DEFAULT_EXECUTION_POLICY.setup_weights[setup_type]
        assert set(values) == set(DEFAULT_EXECUTION_POLICY.component_weights)
        assert round(sum(values.values()), 10) == 1.0
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


def _session_frame(last_open: str) -> pd.DataFrame:
    index = pd.date_range(
        "2026-08-03T00:00:00Z",
        pd.Timestamp(last_open),
        freq="15min",
        tz="UTC",
    )
    close = np.full(len(index), 99.6)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": np.full(len(index), 100.0),
        },
        index=index,
    )


def test_london_opening_range_breakout_requires_window_volume_and_retest():
    df = _session_frame("2026-08-03T08:30:00Z")
    # London is UTC+1 on this date: 08:00 local = 07:00 UTC.
    df.loc["2026-08-03T07:00:00Z"] = [99.40, 100.00, 99.00, 99.70, 100.0]
    df.loc["2026-08-03T07:15:00Z"] = [99.70, 99.90, 99.10, 99.60, 100.0]
    df.loc["2026-08-03T07:30:00Z"] = [99.70, 100.35, 99.65, 100.25, 160.0]
    for timestamp in df.loc["2026-08-03T07:45:00Z":].index[:-1]:
        df.loc[timestamp] = [100.20, 100.35, 100.12, 100.25, 105.0]
    df.iloc[-1] = [100.06, 100.30, 99.98, 100.22, 120.0]
    setup = detect_strict_setup(
        df,
        _suite(df, trend_score=.20, momentum_score=.22, adx=22),
        StructureReport(trend="up", last_bos="bullish", structure_score=.45),
        direction="long",
        price=100.22,
        atr=1.0,
    )
    assert setup is not None
    assert setup.setup_type == "opening_range_breakout"
    assert setup.label == "Long Opening Range Breakout"
    assert setup.evidence["session"] == "London"
    assert setup.evidence["opening_range_high"] == 100.0

    # Identical geometry outside the two-hour activation window must not be ORB.
    outside = pd.concat(
        [
            df,
            pd.DataFrame(
                [[100.2, 100.3, 100.0, 100.2, 120.0]],
                columns=df.columns,
                index=pd.DatetimeIndex(["2026-08-03T10:30:00Z"]),
            ),
        ]
    )
    outside_setup = detect_strict_setup(
        outside,
        _suite(outside, trend_score=.20, momentum_score=.22, adx=22),
        StructureReport(trend="up", last_bos="bullish", structure_score=.45),
        direction="long",
        price=100.2,
        atr=1.0,
    )
    assert outside_setup is None or outside_setup.setup_type != "opening_range_breakout"


def test_new_york_opening_range_uses_only_new_york_levels():
    df = _session_frame("2026-08-03T15:00:00Z")
    # New York is UTC-4: 09:30 local = 13:30 UTC.
    df.loc["2026-08-03T13:30:00Z"] = [100.4, 101.0, 100.0, 100.5, 100.0]
    df.loc["2026-08-03T13:45:00Z"] = [100.5, 100.9, 100.1, 100.4, 100.0]
    df.loc["2026-08-03T14:00:00Z"] = [100.4, 100.5, 99.65, 99.75, 165.0]
    for timestamp in df.loc["2026-08-03T14:15:00Z":].index[:-1]:
        df.loc[timestamp] = [99.75, 99.88, 99.67, 99.76, 105.0]
    df.iloc[-1] = [99.93, 100.02, 99.62, 99.76, 125.0]
    setup = detect_strict_setup(
        df,
        _suite(
            df, close=99.76, trend_score=-.20, momentum_score=-.22,
            ema_fast=100.2, ema_mid=100.3, vwap=100.4, adx=22,
        ),
        StructureReport(trend="down", last_bos="bearish", structure_score=-.45),
        direction="short",
        price=99.76,
        atr=1.0,
    )
    assert setup is not None
    assert setup.setup_type == "opening_range_breakout"
    assert setup.evidence["session"] == "New York"
    assert setup.evidence["opening_range_low"] == 100.0


def test_session_high_low_rejection_requires_wick_volume_and_bias():
    df = _session_frame("2026-08-03T13:30:00Z")
    # The completed London block runs from 07:00 to 13:30 UTC on this date.
    df.loc["2026-08-03T07:00:00Z":"2026-08-03T13:15:00Z", "high"] = 101.0
    df.loc["2026-08-03T07:00:00Z":"2026-08-03T13:15:00Z", "low"] = 99.0
    df.iloc[-1] = [99.50, 100.00, 98.78, 99.82, 145.0]
    setup = detect_strict_setup(
        df,
        _suite(df, trend_score=.20, momentum_score=.16, adx=20),
        StructureReport(trend="up", last_choch="bullish", structure_score=.35),
        direction="long",
        price=99.82,
        atr=1.0,
    )
    assert setup is not None
    assert setup.setup_type == "session_high_low_rejection"
    assert setup.label == "Long Session High/Low Rejection"
    assert setup.evidence["previous_session"] == "London"
    assert setup.evidence["rejection_wick"] >= .35
    assert setup.evidence["volume_ratio"] >= 1.10

    weak_volume = df.copy()
    weak_volume.iloc[-1, weak_volume.columns.get_loc("volume")] = 90.0
    weak = detect_strict_setup(
        weak_volume,
        _suite(weak_volume, trend_score=.20, momentum_score=.16, adx=20),
        StructureReport(trend="up", last_choch="bullish", structure_score=.35),
        direction="long",
        price=99.82,
        atr=1.0,
    )
    assert weak is None or weak.setup_type != "session_high_low_rejection"


def test_session_high_low_rejection_is_long_short_symmetric():
    df = _session_frame("2026-08-03T13:30:00Z")
    df.loc["2026-08-03T07:00:00Z":"2026-08-03T13:15:00Z", "high"] = 101.0
    df.loc["2026-08-03T07:00:00Z":"2026-08-03T13:15:00Z", "low"] = 99.0
    df.iloc[-1] = [100.50, 101.22, 100.00, 100.18, 145.0]
    setup = detect_strict_setup(
        df,
        _suite(
            df, close=100.18, trend_score=-.20, momentum_score=-.16,
            ema_fast=100.4, ema_mid=100.5, vwap=100.55, adx=20,
        ),
        StructureReport(trend="down", last_choch="bearish", structure_score=-.35),
        direction="short",
        price=100.18,
        atr=1.0,
    )
    assert setup is not None
    assert setup.setup_type == "session_high_low_rejection"
    assert setup.label == "Short Session High/Low Rejection"
    assert setup.evidence["previous_session"] == "London"
    assert setup.evidence["rejection_wick"] >= .35


def test_structure_leads_entry_anchor_and_stop_geometry():
    df = _frame()
    df.iloc[-1] = [100.0, 100.2, 99.8, 100.1, 120.0]
    block = StructureLevel(
        "order_block", "bullish", 99.0, 99.25, 88,
        age_bars=6, touch_count=0, mitigation_fraction=.10,
    )
    resistance = StructureLevel("resistance", "bearish", 100.8, 101.0, 82)
    structure = StructureReport(
        trend="up",
        last_bos="bullish",
        structure_score=.5,
        levels=[block, resistance],
        swing_lows=[98.9],
        swing_highs=[101.0],
    )
    indicators = _suite(
        df, close=100.1, ema_fast=99.95, ema_mid=99.90, vwap=99.85,
    )
    anchors = _entry_anchors(indicators, structure, "long", 100.1, 1.0)
    anchor, sources, _, _, _ = _select_anchor(anchors, "long", 100.1, 1.0)
    assert anchor is not None
    assert "order_block" in sources
    assert anchor < 99.5  # indicators are nearer, but structure owns the level.

    profile = build_execution_profile(
        df,
        indicators,
        structure,
        direction="long",
        price=100.1,
        atr=1.0,
        setup_name="Long Day-Trade Confluence",
        strategy_tags=["day_trade"],
    )
    assert "order_block" in profile.anchor_sources
    assert profile.stop_loss < block.price_low
    assert any(target >= 100.8 for target in profile.targets)


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
    for label in (
        "Long Trend Continuation (Pullback)",
        "Long Opening Range Breakout",
        "Short Session High/Low Rejection",
    ):
        caption = format_signal_photo_caption({
            "symbol": "BTC/USDT:USDT", "direction": "long", "confidence": 82,
            "technical_confidence": 84, "execution_quality": 78,
            "entry_status": "wait_retest", "entry_low": 100, "entry_high": 101,
            "stop_loss": 98, "take_profits": [103], "risk_pct": .5,
            "setup_name": label,
            "payload": {"execution": {}},
        })
        assert label.removeprefix("Long ").removeprefix("Short ") in caption
        assert len(caption) <= 1024
