"""Phase 2A execution integrity, costs, rank, and compatibility regressions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analysis.execution import (
    CandleContext,
    TARGET_STRUCTURE_POLICY_VERSION,
    TargetSelection,
    _data_market_quality,
    _entry_accessibility_quality,
    _liquidity_cost_quality,
    _resolve_nearby_target_obstacle,
    _stop_quality,
    build_execution_profile,
)
from src.analysis.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    deterministic_rank_score,
    weighted_execution_quality,
)
from src.analysis.indicators import IndicatorSuite
from src.analysis.legacy_v2 import execution_aware_legacy_confidence
from src.analysis.market_structure import MarketStructureAnalyzer, StructureLevel, StructureReport
from src.api.service import apply_diagnostic_backtest_to_rank
from src.data.exchange import ExchangeClient, MarketSnapshot
from src.notify.telegram import format_signal_photo_caption
from src.scoring.phase2a_comparison import build_phase2a_shadow_report
from src.scoring.features import build_candidate_record


def frame(direction: str = "long", n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(98.0, 100.7, n)
    if direction == "short":
        close = 200.0 - close
    open_ = close - 0.18 if direction == "long" else close + 0.18
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.20,
            "low": np.minimum(open_, close) - 0.25,
            "close": close,
            "volume": np.linspace(800.0, 1300.0, n),
            "ema_fast": close - (0.2 if direction == "long" else -0.2),
            "ema_mid": close - (0.4 if direction == "long" else -0.4),
            "vwap": close - (0.45 if direction == "long" else -0.45),
        },
        index=idx,
    )


def structure(direction: str = "long", *, consumed: bool = False) -> StructureReport:
    if direction == "long":
        levels = [
            StructureLevel("order_block", "bullish", 99.80, 100.05, 88, touch_count=3 if consumed else 0, mitigation_fraction=1.0 if consumed else 0.0, fully_mitigated=consumed, relevant=not consumed),
            StructureLevel("fvg", "bullish", 99.90, 100.12, 80),
            StructureLevel("liquidity", "bearish", 101.75, 101.85, 75),
            StructureLevel("resistance", "bearish", 102.55, 102.65, 70),
        ]
        return StructureReport(trend="up", last_bos="bullish", structure_score=.7, volume_profile_poc=100.0, volume_profile_val=99.8, volume_profile_vah=101.7, swing_highs=[101.8, 102.6, 103.4], swing_lows=[98.8, 99.4], levels=levels)
    return StructureReport(
        trend="down", last_bos="bearish", structure_score=-.7,
        volume_profile_poc=100.0, volume_profile_val=98.3, volume_profile_vah=100.2,
        swing_highs=[100.6, 101.2], swing_lows=[98.2, 97.4, 96.6],
        levels=[
            StructureLevel("order_block", "bearish", 99.95, 100.20, 88),
            StructureLevel("fvg", "bearish", 99.88, 100.10, 80),
            StructureLevel("liquidity", "bullish", 98.15, 98.25, 75),
            StructureLevel("support", "bullish", 97.35, 97.45, 70),
        ],
    )


def suite(df: pd.DataFrame) -> IndicatorSuite:
    return IndicatorSuite(df=df, summary={"ema_fast": float(df.ema_fast.iloc[-1]), "ema_mid": float(df.ema_mid.iloc[-1]), "vwap": float(df.vwap.iloc[-1]), "atr": 1.0})


def test_component_weights_are_normalized_and_explainable():
    values = {name: 80.0 for name in DEFAULT_EXECUTION_POLICY.component_weights}
    quality, contributions = weighted_execution_quality(values, "retest_continuation")
    assert quality == pytest.approx(80.0)
    assert sum(contributions.values()) == pytest.approx(quality)
    assert set(contributions) == set(values)


@pytest.mark.parametrize("distance,expected_relation", [(0.45, "good"), (2.0, "poor")])
def test_entry_accessibility_penalizes_unreachable_retests(distance, expected_relation):
    score = _entry_accessibility_quality(distance, distance, .24, 8, 90, .4, 0, DEFAULT_EXECUTION_POLICY)
    assert (score >= 65) if expected_relation == "good" else (score < 55)


def test_missing_and_stale_data_have_graded_behavior():
    missing, notes = _data_market_quality(None, None, None, DEFAULT_EXECUTION_POLICY)
    stale, _ = _data_market_quality(90, 60, False, DEFAULT_EXECUTION_POLICY)
    assert 0 < missing < 100
    assert notes == ["ticker_age_missing", "orderbook_age_missing"]
    assert stale == 0


def test_liquidity_missing_is_not_neutral():
    missing, notes = _liquidity_cost_quality(spread_bps=2, book_alignment=0, bid_depth_usd=0, ask_depth_usd=0, impact_bps=None, total_cost_bps=15, policy=DEFAULT_EXECUTION_POLICY)
    deep, _ = _liquidity_cost_quality(spread_bps=2, book_alignment=.2, bid_depth_usd=100_000, ask_depth_usd=100_000, impact_bps=1, total_cost_bps=15, policy=DEFAULT_EXECUTION_POLICY)
    assert missing < deep
    assert "notional_depth_missing" in notes


def test_stop_quality_penalizes_tight_and_wide_stops_symmetrically():
    candle = CandleContext(noise_atr=1.1)
    good = _stop_quality(1.2, candle, 80, 8)
    tight = _stop_quality(.45, candle, 80, 8)
    wide = _stop_quality(3.0, candle, 80, 8)
    assert good > tight
    assert good > wide


def test_spread_and_slippage_affect_liquidity_but_not_stop_quality():
    df = frame()
    common = {
        "symbol": "BTC/USDT:USDT",
        "exchange_id": "okx",
        "last": 100.7,
        "orderbook_imbalance": 0.1,
        "ticker_age_seconds": 1.0,
        "orderbook_age_seconds": 1.0,
        "execution_data_fresh": True,
        "estimated_impact_bps": 1.0,
        "raw": {
            "orderbook_summary": {
                "bid_depth_usd_10bps": 100_000,
                "ask_depth_usd_10bps": 100_000,
                "estimated_impact_bps": 1.0,
            }
        },
    }
    low_friction = build_execution_profile(
        df,
        suite(df),
        structure(),
        direction="long",
        price=100.7,
        atr=1.0,
        snapshot=MarketSnapshot(spread_bps=2.0, **common),
    )
    high_friction = build_execution_profile(
        df,
        suite(df),
        structure(),
        direction="long",
        price=100.7,
        atr=1.0,
        snapshot=MarketSnapshot(spread_bps=11.0, **common),
    )
    assert high_friction.components["stop_quality"] == pytest.approx(
        low_friction.components["stop_quality"]
    )
    assert (
        high_friction.components["liquidity_cost_quality"]
        < low_friction.components["liquidity_cost_quality"]
    )


def test_fresh_zone_outscores_consumed_zone():
    df = frame()
    fresh = build_execution_profile(df, suite(df), structure(), direction="long", price=100.6, atr=1, setup_name="Long Breakout Retest")
    used = build_execution_profile(df, suite(df), structure(consumed=True), direction="long", price=100.6, atr=1, setup_name="Long Breakout Retest")
    assert fresh.components["entry_zone_quality"] >= used.components["entry_zone_quality"]


def test_structure_lifecycle_detects_touches_and_mitigation():
    df = frame()
    lifecycle = [StructureLevel("order_block", "bullish", 99.5, 100.0, 75, index=20)]
    MarketStructureAnalyzer._annotate_level_lifecycle(lifecycle, df)
    assert lifecycle[0].age_bars is not None and lifecycle[0].creation_time
    assert 0 <= lifecycle[0].mitigation_fraction <= 1


def test_targets_are_not_fabricated_to_four_and_net_rr_is_lower():
    df = frame()
    snapshot = MarketSnapshot(symbol="BTC", exchange_id="okx", spread_bps=4, funding_rate=.0001)
    profile = build_execution_profile(df, suite(df), structure(), direction="long", price=100.6, atr=1, snapshot=snapshot)
    assert 1 <= len(profile.targets) <= 4
    assert len(profile.legacy_targets) == 4
    assert len(profile.targets) == len(profile.target_feasibility)
    assert all(net <= gross for net, gross in zip(profile.net_risk_reward, profile.gross_risk_reward))


def _blocked_target_selection(
    *,
    direction: str = "long",
    next_target: float = 101.2,
    next_source: str = "resistance",
) -> TargetSelection:
    obstacle_price = 100.2 if direction == "long" else 99.8
    return TargetSelection(
        targets=[next_target],
        qualities=[76.0],
        sources=[next_source],
        obstacles=[{
            "price": obstacle_price,
            "distance_atr": 0.2,
            "source": "liquidity",
            "quality": 74.0,
        }],
    )


def test_nearby_structure_prefers_small_cost_justified_tp1_adjustment():
    selection = _blocked_target_selection()
    selection.obstacles.append({
        "price": 100.32,
        "distance_atr": 0.32,
        "source": "order_block",
        "quality": 72.0,
    })
    resolved, detail = _resolve_nearby_target_obstacle(
        selection,
        direction="long",
        entry=100.0,
        stop_distance=0.30,
        atr=1.0,
        cost_price=0.01,
        expected_hold_hours=8.0,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    assert detail["resolved"] is True
    assert detail["action"] == "adjusted_beyond_structure"
    assert detail["policy_version"] == TARGET_STRUCTURE_POLICY_VERSION
    assert detail["adjusted_tp1_net_rr"] >= 0.75
    assert resolved.targets[0] == pytest.approx(100.40)
    assert resolved.targets[0] > max(
        obstacle["price"] for obstacle in selection.obstacles
    )
    assert resolved.sources[0] == "structure_clearance_adjustment"


@pytest.mark.parametrize(
    "direction,next_target",
    [("long", 101.2), ("short", 98.8)],
)
def test_next_real_structure_is_promoted_when_clearance_tp_is_too_weak(
    direction,
    next_target,
):
    resolved, detail = _resolve_nearby_target_obstacle(
        _blocked_target_selection(direction=direction, next_target=next_target),
        direction=direction,
        entry=100.0,
        stop_distance=1.0,
        atr=1.0,
        cost_price=0.02,
        expected_hold_hours=8.0,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    assert detail["resolved"] is True
    assert detail["action"] == "promoted_next_structure"
    assert detail["adjusted_tp1_net_rr"] >= 0.75
    assert resolved.targets == [next_target]


def test_execution_profile_clears_tp1_block_only_after_structural_promotion():
    df = frame()
    report = structure()
    report.levels.append(
        StructureLevel("resistance", "bearish", 100.15, 100.25, 82)
    )
    profile = build_execution_profile(
        df,
        suite(df),
        report,
        direction="long",
        price=100.6,
        atr=1.0,
    )
    assert profile.target_adjustment["action"] == "promoted_next_structure"
    assert profile.target_adjustment["adjusted_tp1_net_rr"] >= 0.75
    assert "tp1_blocked_by_nearby_structure" not in profile.hard_failures
    assert profile.targets[0] == pytest.approx(101.7)


@pytest.mark.parametrize(
    "direction,next_target",
    [("long", 100.6), ("short", 99.4)],
)
def test_blocked_tp1_still_rejects_when_no_target_clears_net_rr_floor(
    direction,
    next_target,
):
    resolved, detail = _resolve_nearby_target_obstacle(
        _blocked_target_selection(direction=direction, next_target=next_target),
        direction=direction,
        entry=100.0,
        stop_distance=1.0,
        atr=1.0,
        cost_price=0.03,
        expected_hold_hours=8.0,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    assert detail["resolved"] is False
    assert detail["action"] == "rejected"
    assert detail["next_structure_net_rr"] < 0.75
    assert resolved.targets == [next_target]


def test_projection_cannot_resolve_a_blocked_tp1():
    selection = _blocked_target_selection(
        next_target=101.0,
        next_source="volatility_projection",
    )
    selection.used_projection = True
    _, detail = _resolve_nearby_target_obstacle(
        selection,
        direction="long",
        entry=100.0,
        stop_distance=0.5,
        atr=1.0,
        cost_price=0.01,
        expected_hold_hours=8.0,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    assert detail["resolved"] is False
    assert detail["action"] == "rejected"


def test_long_short_component_symmetry():
    long_df, short_df = frame("long"), frame("short")
    long_profile = build_execution_profile(
        long_df, suite(long_df), structure("long"), direction="long",
        price=100.6, atr=1, setup_name="Long Breakout Retest",
    )
    short_profile = build_execution_profile(
        short_df, suite(short_df), structure("short"), direction="short",
        price=99.4, atr=1, setup_name="Short Breakout Retest",
    )
    assert long_profile.stop_distance_atr == pytest.approx(short_profile.stop_distance_atr)
    assert long_profile.gross_risk_reward == pytest.approx(short_profile.gross_risk_reward)
    assert long_profile.status == short_profile.status


def test_rank_does_not_double_count_raw_confluence():
    first, breakdown = deterministic_rank_score(overall_quality=80, execution_quality=75, target_feasibility=70, stop_quality=80, net_rr=1.4, market_data_quality=90, setup_validity=100)
    second, _ = deterministic_rank_score(overall_quality=80, execution_quality=75, target_feasibility=70, stop_quality=80, net_rr=1.4, market_data_quality=90, setup_validity=100)
    assert first == second
    assert "confluence" not in breakdown


def test_legacy_v2_plus_five_cap_remains():
    result = execution_aware_legacy_confidence(technical_confidence=92, execution_score=69, immediate_sl_risk=0, data_quality_score=100, confidence_min=0, confidence_max=100, execution_confidence_buffer=5)
    assert result.legacy_v2_confidence == 74


def test_quick_backtest_has_no_live_rank_authority():
    assert apply_diagnostic_backtest_to_rank(73.5, {"validation_score": 0, "historical_edge_ok": False}) == 73.5
    assert apply_diagnostic_backtest_to_rank(73.5, {"validation_score": 100, "historical_edge_ok": True}) == 73.5


def test_orderbook_uses_quote_notional_and_contract_size():
    class DummyExchange:
        has = {"fetchOrderBook": True}
        def fetch_order_book(self, symbol, limit=25):
            return {"bids": [[100, 10], [99.95, 10]], "asks": [[100.1, 10], [100.15, 10]], "timestamp": 1_800_000_000_000}
        def market(self, symbol):
            return {"contractSize": .1, "inverse": False, "precision": {"price": .01, "amount": .1}, "limits": {"cost": {"min": 5}}}
    client = ExchangeClient.__new__(ExchangeClient)
    client.exchange_id = "phase2-notional"
    client._exchange = DummyExchange()
    client.config = None
    client.cache_ttl_seconds = 0
    client.resolve_symbol = lambda symbol: symbol
    book = client.fetch_order_book_summary("BTC", limit=25, force_refresh=True)
    assert book["bid_depth_usd"] == pytest.approx(199.95)
    assert book["ask_depth_usd"] == pytest.approx(200.25)
    assert book["contract_size"] == .1
    assert "10" in book["depth_bands_bps"]


def test_shadow_report_covers_counts_groups_and_geometry():
    report = build_phase2a_shadow_report([
        {"symbol": "ARB", "setup_type": "retest_continuation", "direction": "short", "legacy_execution_score": 75, "execution_quality": 60, "hard_failures": ["stale_zone"], "targets": [1, .9]},
        {"symbol": "BNB", "setup_type": "cmp_confirmation", "direction": "long", "legacy_execution_score": 69, "execution_quality": 80, "hard_failures": [], "targets": [1, 2]},
    ])
    assert report["old_signal_count"] == 1
    assert report["new_signal_count"] == 1
    assert report["by_direction"]["long"]["new_pass"] == 1
    assert report["geometry_changes"]


def test_telegram_labels_heuristics_as_quality_not_probability():
    caption = format_signal_photo_caption({
        "symbol": "BTC", "direction": "long", "confidence": 87,
        "technical_confidence": 84, "execution_quality": 79,
        "entry_status": "wait_retest", "entry_low": 100, "entry_high": 101,
        "stop_loss": 98, "take_profits": [103], "risk_pct": 1,
        "execution_components": {"entry_accessibility": 75, "stop_quality": 80, "target_feasibility": 70},
        "data_freshness_state": "fresh", "payload": {"execution": {}},
    })
    assert "Overall Quality 87/100" in caption
    assert "Execution Quality 79/100" in caption
    assert "87% Confidence" not in caption


def test_telegram_discloses_structure_adjusted_tp1_with_net_floor():
    caption = format_signal_photo_caption({
        "symbol": "BTC", "direction": "long", "confidence": 84,
        "technical_confidence": 86, "execution_quality": 82,
        "entry_status": "wait_retest", "entry_low": 100, "entry_high": 100.1,
        "stop_loss": 99, "take_profits": [101.2, 102.0], "risk_pct": 1,
        "target_adjustment": {
            "resolved": True,
            "action": "promoted_next_structure",
            "adjusted_tp1_net_rr": 0.86,
            "minimum_net_rr": 0.75,
        },
        "payload": {"execution": {}},
    })
    assert "Next feasible structure promoted to TP1" in caption
    assert "net 0.86R (floor 0.75R)" in caption
    assert len(caption) <= 1024


def test_component_outputs_are_finite_and_bounded_under_adversarial_values():
    quality, parts = weighted_execution_quality({name: math.inf for name in DEFAULT_EXECUTION_POLICY.component_weights}, "retest_continuation")
    assert 0 <= quality <= 100
    assert all(math.isfinite(value) for value in parts.values())


def test_phase2a_instrumentation_is_persisted_additively():
    row = {
        "symbol": "BTC", "direction": "long", "price": 100,
        "entry_low": 99.8, "entry_high": 100.0, "stop_loss": 99,
        "take_profits": [101.5], "setup_name": "Long Momentum",
        "execution_policy_version": "execution_quality_v2a.1",
        "execution_setup_type": "cmp_confirmation",
        "execution_components": {"entry_accessibility": 80},
        "stop_distance_atr": 1.0, "entry_distance_atr": .1,
        "gross_risk_reward": [1.5], "net_risk_reward": [1.2],
        "target_policy_version": TARGET_STRUCTURE_POLICY_VERSION,
        "target_adjustment": {
            "resolved": True,
            "action": "promoted_next_structure",
        },
        "depth_bands_bps": {"10": {"bid_usd": 10000}},
        "hard_failures": [], "execution_uncertainties": ["tick_size_unknown"],
        "execution": {"status": "confirmation_pending"},
        "universal_eligible": True,
        "production_qualified": True,
        "signal_eligible": True,
        "prop_safe": False,
        "prop_guidance": {
            "status": "not_recommended_for_strict_prop",
            "suggested_risk_pct_min": 0.25,
            "suggested_risk_pct_max": 0.25,
            "suggested_leverage_min": 1,
            "suggested_leverage_max": 1,
            "reasons": ["Wider stop geometry."],
        },
    }
    record = build_candidate_record(None, row, source="phase2a_test")
    assert record["decision"]["execution_policy_version"] == "execution_quality_v2a.1"
    assert record["decision"]["stop_distance_atr"] == 1.0
    assert record["decision"]["depth_bands_bps"]["10"]["bid_usd"] == 10000
    assert record["decision"]["target_policy_version"] == TARGET_STRUCTURE_POLICY_VERSION
    assert record["decision"]["target_adjustment"]["action"] == "promoted_next_structure"
    assert record["decision"]["universal_eligible"] is True
    assert record["decision"]["production_qualified"] is True
    assert record["decision"]["prop_safe"] is False
    assert record["decision"]["prop_guidance"]["status"] == "not_recommended_for_strict_prop"
    assert record["production_eligible"] is True
    assert record["feature_schema_version"] == "3.0"
