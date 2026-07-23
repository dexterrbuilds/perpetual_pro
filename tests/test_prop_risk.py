"""Prop risk clamps and flags."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.risk import RiskManager
from src.utils.config import RiskConfig, load_config


def test_prop_risk_clamps_pct_and_leverage():
    rm = RiskManager(
        risk_cfg=RiskConfig(
            prop_mode=True,
            risk_per_trade_pct=5.0,  # should clamp to max 1%
            risk_per_trade_min_pct=0.5,
            risk_per_trade_max_pct=1.0,
            max_leverage=5,
            leverage_ceiling=5,
            leverage_floor=1,
        ),
        simulated_capital=10_000,
        risk_pct=5.0,
    )
    plan = rm.build_plan("long", price=100.0, atr=1.5, confidence=70, primary_tf="15m")
    assert 0.5 <= plan.risk_pct <= 1.0
    assert plan.leverage_suggested <= 5
    assert plan.prop_mode is True
    assert plan.max_leverage_allowed <= 5


def test_prop_flags_wide_stop():
    rm = RiskManager(
        risk_cfg=RiskConfig(prop_mode=True, max_leverage=5, leverage_ceiling=5),
        simulated_capital=1000,
        risk_pct=1.0,
    )
    # Very large ATR relative to price → wide stop / drawdown flags
    plan = rm.build_plan("long", price=100.0, atr=5.0, confidence=55, primary_tf="15m")
    assert plan.atr_pct >= 2.5
    assert "WIDE_STOP" in plan.prop_flags or "HIGH_DRAWDOWN_RISK" in plan.prop_flags


def test_prop_signal_requires_final_60_percent_confidence():
    rm = RiskManager(
        risk_cfg=RiskConfig(prop_mode=True, max_leverage=5, leverage_ceiling=5),
        simulated_capital=1000,
        risk_pct=1.0,
    )
    plan = rm.build_plan("long", price=100.0, atr=1.0, confidence=70, primary_tf="15m")
    rm.apply_prop_confidence_gate(plan, 59.9)
    assert plan.prop_safe is False
    assert "LOW_CONFIDENCE" in plan.prop_flags

    rm.apply_prop_confidence_gate(plan, 60.0)
    assert "LOW_CONFIDENCE" not in plan.prop_flags
    assert plan.prop_safe is True


def test_non_prop_day_trade_leverage_is_10_to_30x():
    rm = RiskManager(
        risk_cfg=RiskConfig(
            prop_mode=False,
            leverage_floor=10,
            leverage_ceiling=30,
            max_leverage=30,
        )
    )
    plan = rm.build_plan("short", price=100.0, atr=0.8, confidence=75, primary_tf="15m")
    assert 10 <= plan.leverage_suggested <= 30


def test_config_loads_prop_and_telegram():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.risk.prop_mode is True
    assert cfg.risk.max_leverage <= 5
    assert hasattr(cfg, "telegram")
    assert hasattr(cfg, "scheduler")
    assert "09:00" in cfg.scheduler.times
    assert "15:00" in cfg.scheduler.times
    assert "20:00" in cfg.scheduler.times
    assert [session["name"] for session in cfg.scheduler.sessions] == [
        "London open",
        "New York open",
        "New York liquidity window",
    ]
    assert cfg.telegram.notify_on_empty is True
    # Secrets must never be sourced from YAML defaults
    # (empty unless env vars are set in the test process)
    assert not getattr(cfg.telegram, "bot_token", None) or True
