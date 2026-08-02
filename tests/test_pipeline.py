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
from src.analysis import confluence as confluence_module
from src.analysis.llm import NarrativeLLM
from src.analysis.indicators import compute_indicators
from src.data.exchange import MarketSnapshot
from src.data.multi_tf import MultiTimeframeData
from src.data.multi_tf import assess_candle_quality, closed_candles
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


def test_live_candle_quality_rejects_stale_or_gapped_data():
    now = pd.Timestamp("2026-07-28T12:00:00Z")
    idx = pd.date_range(end=now - pd.Timedelta(minutes=15), periods=100, freq="15min")
    df = pd.DataFrame(
        {
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.5] * 100,
            "volume": [1000.0] * 100,
        },
        index=idx,
    )
    fresh = assess_candle_quality(df, "15m", now=now)
    assert fresh["ok"] is True
    assert fresh["age_intervals"] == 0

    stale = assess_candle_quality(
        df.iloc[:-8],
        "15m",
        now=now,
    )
    assert stale["ok"] is False
    assert "stale" in stale["reason"]


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
        "confirmation_pending",
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


def test_stale_confirmation_timeframe_blocks_live_signal(monkeypatch):
    cfg = load_config(ROOT / "config.yaml")
    primary = _ohlcv(320, drift=0.12)
    h1 = primary.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    h4 = primary.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    mtf = MultiTimeframeData(
        symbol="BTC/USDT:USDT",
        exchange_id="okx",
        primary_tf="15m",
        frames={"15m": primary, "1h": h1, "4h": h4},
        snapshot=MarketSnapshot(
            symbol="BTC/USDT:USDT",
            exchange_id="okx",
            last=float(primary["close"].iloc[-1]),
        ),
        quality={
            "15m": {"ok": True, "score": 100.0, "reason": "fresh"},
            "1h": {"ok": False, "score": 20.0, "reason": "stale_3.0_intervals"},
            "4h": {"ok": True, "score": 100.0, "reason": "fresh"},
        },
    )

    monkeypatch.setattr(
        NarrativeLLM,
        "generate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("deterministic reject must not call the LLM")
        ),
    )
    result = ConfluenceEngine(cfg).analyze(mtf, use_llm=True)

    assert result.meta["primary_data_quality_ok"] is False
    assert result.confidence <= 45.0
    assert result.meta["signal_eligible"] is False
    assert result.meta["llm_invocation_status"] == "skipped_deterministic_reject"


def test_llm_rate_limit_cannot_change_deterministic_result(monkeypatch):
    cfg = load_config(ROOT / "config.yaml")
    primary = _ohlcv(320, drift=0.12)
    h1 = primary.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    h4 = primary.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    mtf = MultiTimeframeData(
        symbol="BTC/USDT:USDT",
        exchange_id="okx",
        primary_tf="15m",
        frames={"15m": primary, "1h": h1, "4h": h4},
        snapshot=MarketSnapshot(
            symbol="BTC/USDT:USDT",
            exchange_id="okx",
            last=float(primary["close"].iloc[-1]),
        ),
    )
    baseline = ConfluenceEngine(cfg).analyze(mtf, use_llm=False)
    calls = []
    monkeypatch.setattr(
        confluence_module,
        "deterministic_narrative_eligible",
        lambda **kwargs: True,
    )

    def rate_limited(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("Groq 429")

    monkeypatch.setattr(NarrativeLLM, "generate", rate_limited)
    limited = ConfluenceEngine(cfg).analyze(mtf, use_llm=True)

    assert calls == [True]
    assert limited.meta["llm_invocation_status"] == "fallback:RuntimeError"
    assert limited.confidence == baseline.confidence
    assert limited.rank_score == baseline.rank_score
    assert limited.direction == baseline.direction
    assert limited.meta["signal_eligible"] == baseline.meta["signal_eligible"]
    assert limited.trade_plan.entry_low == baseline.trade_plan.entry_low
    assert limited.trade_plan.entry_high == baseline.trade_plan.entry_high
    assert limited.trade_plan.stop_loss == baseline.trade_plan.stop_loss
    assert limited.trade_plan.take_profits == baseline.trade_plan.take_profits
