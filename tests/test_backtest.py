"""Backtest metrics on synthetic OHLCV."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.backtest import run_backtest
from src.utils.config import AppConfig, RiskConfig


def _trending_df(n: int = 300, trend: float = 0.3) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(np.full(n, trend) + np.random.default_rng(0).normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )


def test_backtest_returns_core_metrics():
    cfg = AppConfig(
        risk=RiskConfig(
            prop_mode=True,
            simulated_capital=1000,
            risk_per_trade_pct=1.0,
            max_leverage=5,
            leverage_ceiling=5,
        )
    )
    result = run_backtest(
        "BTC",
        timeframe="15m",
        bars=300,
        config=cfg,
        df=_trending_df(300, trend=0.4),
        step=3,
        warmup=60,
    )
    assert result.n_bars >= 100
    assert result.win_rate >= 0
    assert result.win_rate <= 100
    assert result.max_drawdown_pct >= 0
    assert result.profit_factor >= 0
    assert result.starting_equity == 1000
    assert isinstance(result.equity_curve, list)
    d = result.to_dict()
    assert "n_trades" in d
    assert d["prop_settings"]["max_leverage"] <= 5
