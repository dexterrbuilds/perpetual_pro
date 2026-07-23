"""End-to-end offline pipeline test with synthetic OHLCV."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.confluence import ConfluenceEngine
from src.analysis.indicators import compute_indicators
from src.data.exchange import MarketSnapshot
from src.data.multi_tf import MultiTimeframeData
from src.data.multi_tf import closed_candles
from src.data.news import NewsBundle, NewsItem
from src.report.generator import ReportGenerator
from src.utils.config import load_config


def _ohlcv(n: int = 300, drift: float = 0.08, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    close = 50000 + np.cumsum(rng.normal(drift, 40, size=n))
    high = close + rng.uniform(10, 80, size=n)
    low = close - rng.uniform(10, 80, size=n)
    open_ = close + rng.normal(0, 20, size=n)
    vol = rng.uniform(50, 400, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_indicator_fallback_count():
    df = _ohlcv()
    suite = compute_indicators(df)
    assert suite.indicator_count >= 40
    assert "trend_score" in suite.summary
    assert suite.summary.get("atr") is not None or suite.summary.get("close")


def test_incomplete_exchange_candle_is_removed():
    forming = pd.Timestamp.now(tz="UTC").floor("15min")
    idx = pd.date_range(end=forming, periods=4, freq="15min")
    df = pd.DataFrame(
        {
            "open": [1.0] * 4,
            "high": [2.0] * 4,
            "low": [0.5] * 4,
            "close": [1.5] * 4,
            "volume": [100.0] * 4,
        },
        index=idx,
    )
    assert len(closed_candles(df, "15m")) == 3

    historical = df.copy()
    historical.index = historical.index - pd.Timedelta(days=2)
    assert len(closed_candles(historical, "15m")) == 4


def test_full_confluence_pipeline(tmp_path):
    cfg = load_config(ROOT / "config.yaml")
    primary = _ohlcv(320, drift=0.12)
    h1 = primary.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    h4 = primary.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    snap = MarketSnapshot(
        symbol="BTC/USDT:USDT",
        exchange_id="binanceusdm",
        last=float(primary["close"].iloc[-1]),
        funding_rate=0.0001,
        open_interest=1_000_000,
        long_short_ratio=1.2,
        percentage_24h=2.5,
    )
    mtf = MultiTimeframeData(
        symbol="BTC/USDT:USDT",
        exchange_id="binanceusdm",
        primary_tf="15m",
        frames={"15m": primary, "1h": h1, "4h": h4},
        snapshot=snap,
    )
    news = NewsBundle(
        symbol="BTC",
        items=[
            NewsItem(title="Bitcoin ETF sees record inflow", source="test", sentiment_score=0.4),
            NewsItem(title="Market cautious amid volatility", source="test", sentiment_score=-0.1),
        ],
        aggregate_sentiment=0.2,
        bias="bullish",
        summary="Synthetic bullish lean",
    )

    engine = ConfluenceEngine(cfg)
    analysis = engine.analyze(
        mtf, news=news, simulated_capital=1000, risk_pct=1.0, use_llm=False
    )

    assert analysis.bias in ("bullish", "bearish", "neutral")
    assert 0 <= analysis.confidence <= 100
    assert analysis.factors
    assert analysis.trade_plan is not None
    assert analysis.structure is not None
    assert analysis.patterns is not None
    assert analysis.trader_commentary
    assert analysis.scenarios is not None
    assert analysis.execution is not None
    assert analysis.trade_plan.entry_status in (
        "ready",
        "wait_retest",
        "avoid_chase",
        "blocked",
    )
    assert analysis.trade_plan.is_simulation
    assert analysis.key_reasons is not None

    # Report export
    cfg.output.output_dir = str(tmp_path)
    cfg.output.save_json = True
    cfg.output.save_markdown = True
    reporter = ReportGenerator(cfg)
    paths = reporter.save(analysis)
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    payload = paths["json"].read_text(encoding="utf-8")
    assert "NOT FINANCIAL ADVICE" in payload or "not financial advice" in payload.lower()
    assert "confluence_total" in payload
    assert "primary_setup" in payload
    assert "position_simulation" in payload
