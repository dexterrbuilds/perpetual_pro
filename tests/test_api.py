"""API service + FastAPI route tests (offline image path)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fake_chart_png(text: str = "BTCUSDT 15m") -> bytes:
    """Create a simple dark chart-like PNG with text for OCR path smoke."""
    img = Image.new("RGB", (640, 360), color=(20, 24, 32))
    draw = ImageDraw.Draw(img)
    # Fake candle bodies
    rng = np.random.default_rng(0)
    x = 40
    y_base = 200
    for _ in range(30):
        h = int(rng.integers(10, 60))
        bull = bool(rng.integers(0, 2))
        color = (40, 180, 90) if bull else (200, 60, 60)
        draw.rectangle([x, y_base - h, x + 8, y_base], fill=color)
        x += 18
        y_base += int(rng.integers(-8, 9))
    draw.text((20, 12), text, fill=(220, 220, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_from_image_requires_symbol_without_ocr_hit(monkeypatch):
    """When OCR finds nothing and no symbol passed, return structured error."""
    from src.api import service as svc
    from src.vision.ocr import OCRResult
    from src.vision.chart_detect import VisionChartResult

    class FakeOCR:
        def __init__(self, *a, **k):
            pass

        def extract(self, image, dark_theme=None):
            return OCRResult(raw_text="", confidence=0.0)

    class FakeVision:
        def __init__(self, *a, **k):
            pass

        def analyze(self, image, dark_theme=None):
            return VisionChartResult(candles_detected=10, trend_guess="up", confidence=0.5)

    monkeypatch.setattr(svc, "OCREngine", FakeOCR)
    monkeypatch.setattr(svc, "ChartVision", FakeVision)

    result = svc.analyze_from_image(_fake_chart_png(), request=svc.AnalyzeRequest())
    assert result["ok"] is False
    assert result["error"] == "symbol_required"


def test_analyze_from_image_full_pipeline_mocked(monkeypatch):
    """Full path with symbol override; mock exchange data."""
    from src.api import service as svc
    from src.vision.ocr import OCRResult
    from src.vision.chart_detect import VisionChartResult
    from src.data.multi_tf import MultiTimeframeData
    from src.data.exchange import MarketSnapshot
    import pandas as pd

    class FakeOCR:
        def __init__(self, *a, **k):
            pass

        def extract(self, image, dark_theme=None):
            return OCRResult(
                raw_text="BTCUSDT 15m",
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                confidence=0.8,
                prices=[50000.0, 50100.0],
            )

    class FakeVision:
        def __init__(self, *a, **k):
            pass

        def analyze(self, image, dark_theme=None):
            return VisionChartResult(
                candles_detected=40,
                trend_guess="up",
                confidence=0.6,
                notes=["test"],
            )

    def fake_fetch(client, symbol, primary_tf, higher_tfs=None, limit=500, include_snapshot=True, config=None):
        n = 200
        idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(1)
        close = 50000 + np.cumsum(rng.normal(0.1, 30, size=n))
        df = pd.DataFrame(
            {
                "open": close + rng.normal(0, 10, n),
                "high": close + 40,
                "low": close - 40,
                "close": close,
                "volume": rng.uniform(100, 500, n),
            },
            index=idx,
        )
        return MultiTimeframeData(
            symbol=symbol,
            exchange_id=getattr(client, "exchange_id", "binanceusdm"),
            primary_tf=primary_tf,
            frames={primary_tf: df, "1h": df.iloc[::4].copy()},
            snapshot=MarketSnapshot(
                symbol=symbol,
                exchange_id="binanceusdm",
                last=float(close[-1]),
                funding_rate=0.0001,
                open_interest=1e6,
            ),
        )

    class FakeClient:
        exchange_id = "binanceusdm"

        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    class FakeNews:
        def __init__(self, *a, **k):
            pass

        def analyze(self, symbol):
            from src.data.news import NewsBundle

            return NewsBundle(symbol="BTC", summary="test", bias="neutral", aggregate_sentiment=0.0)

    monkeypatch.setattr(svc, "OCREngine", FakeOCR)
    monkeypatch.setattr(svc, "ChartVision", FakeVision)
    monkeypatch.setattr(svc, "ExchangeClient", FakeClient)
    monkeypatch.setattr(svc, "fetch_multi_timeframe", fake_fetch)
    monkeypatch.setattr(svc, "NewsAnalyzer", FakeNews)

    result = svc.analyze_from_image(
        _fake_chart_png(),
        request=svc.AnalyzeRequest(symbol="BTC", timeframe="15m", no_news=False),
    )
    assert result["ok"] is True
    assert result["data_mode"] == "full"
    assert "bias" in result
    assert "confidence" in result
    assert "trade_plan" in result
    assert "factors" in result
    assert "vision" in result
    assert result["symbol"]


def test_fastapi_analyze_endpoint(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    # Import after path setup
    import main_server

    # Mock the service so we don't hit exchanges
    def fake_analyze(raw, request=None, config=None):
        return {
            "ok": True,
            "data_mode": "full",
            "symbol": "BTC/USDT:USDT",
            "bias": "bullish",
            "confidence": 72.0,
            "trade_plan": {"direction": "long", "stop_loss": 1.0},
            "factors": [],
            "vision": {"ocr": {"symbol": "BTC/USDT:USDT"}},
            "disclaimer": "NOT FINANCIAL ADVICE",
        }

    monkeypatch.setattr(main_server, "analyze_from_image", fake_analyze)

    client = TestClient(main_server.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    files = {"image": ("chart.png", _fake_chart_png(), "image/png")}
    data = {"symbol": "BTC", "timeframe": "15m"}
    r = client.post("/analyze", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["bias"] == "bullish"
    assert "trade_plan" in body
