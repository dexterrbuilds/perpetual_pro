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

    def fake_fetch_fallback(
        symbol,
        primary_tf,
        preferred_exchange,
        higher_tfs=None,
        limit=500,
        include_snapshot=True,
        config=None,
        auto_fallback=None,
    ):
        from src.data.multi_tf import FallbackFetchResult

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
        mtf = MultiTimeframeData(
            symbol=symbol,
            exchange_id=preferred_exchange or "binanceusdm",
            primary_tf=primary_tf,
            frames={primary_tf: df, "1h": df.iloc[::4].copy()},
            snapshot=MarketSnapshot(
                symbol=symbol,
                exchange_id=preferred_exchange or "binanceusdm",
                last=float(close[-1]),
                funding_rate=0.0001,
                open_interest=1e6,
            ),
        )
        client = FakeClient(exchange_id=preferred_exchange)
        return FallbackFetchResult(
            mtf=mtf,
            client=client,
            requested_exchange=preferred_exchange or "binanceusdm",
            exchange_used=preferred_exchange or "binanceusdm",
            fallback_used=False,
            attempted_exchanges=[preferred_exchange or "binanceusdm"],
        )

    class FakeClient:
        def __init__(self, *a, **k):
            self.exchange_id = k.get("exchange_id") or (a[0] if a else "binanceusdm")

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
    monkeypatch.setattr(svc, "fetch_multi_timeframe_with_fallback", fake_fetch_fallback)
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


def test_analyze_from_image_prefers_client_hint_exchange(monkeypatch):
    from src.api import service as svc
    from src.vision.ocr import OCRResult
    from src.vision.chart_detect import VisionChartResult
    from src.data.multi_tf import FallbackFetchResult, MultiTimeframeData
    import pandas as pd

    captured = {}

    class FakeOCR:
        def __init__(self, *a, **k):
            pass

        def extract(self, image, dark_theme=None):
            return OCRResult(raw_text="BTCUSDT 15m", symbol="BTC/USDT:USDT", timeframe="15m", confidence=0.8, prices=[50000.0])

    class FakeVision:
        def __init__(self, *a, **k):
            pass

        def analyze(self, image, dark_theme=None):
            return VisionChartResult(candles_detected=10, trend_guess="up", confidence=0.5, notes=["test"])

    class FakeClient:
        def __init__(self, exchange_id=None, config=None, exchange_cfg=None):
            self.exchange_id = exchange_id or "mexc"

        def close(self):
            pass

    def fake_fallback(symbol, primary_tf, preferred_exchange, **kwargs):
        captured["preferred_exchange"] = preferred_exchange
        mtf = MultiTimeframeData(
            symbol=symbol,
            exchange_id=preferred_exchange,
            primary_tf=primary_tf,
            frames={primary_tf: pd.DataFrame()},
        )
        return FallbackFetchResult(
            mtf=mtf,
            client=FakeClient(exchange_id=preferred_exchange),
            requested_exchange=preferred_exchange,
            exchange_used=preferred_exchange,
            fallback_used=False,
            attempted_exchanges=[preferred_exchange],
        )

    monkeypatch.setattr(svc, "OCREngine", FakeOCR)
    monkeypatch.setattr(svc, "ChartVision", FakeVision)
    monkeypatch.setattr(svc, "fetch_multi_timeframe_with_fallback", fake_fallback)
    monkeypatch.setattr(svc, "NewsAnalyzer", lambda *a, **k: None)

    svc.analyze_from_image(
        _fake_chart_png(),
        request=svc.AnalyzeRequest(symbol="BTC", timeframe="15m", no_news=True, client_hints={"exchange": "mexc"}),
        config=svc.load_config(),
    )

    assert captured["preferred_exchange"] == "mexc"


def test_scan_symbols_ranks_results(monkeypatch):
    from src.api import service as svc
    from src.data.exchange import MarketSnapshot
    from src.data.multi_tf import FallbackFetchResult, MultiTimeframeData
    import pandas as pd

    class FakeClient:
        def __init__(self, *a, **k):
            self.exchange_id = k.get("exchange_id") or "binanceusdm"

        def close(self):
            pass

    class FakeNews:
        def __init__(self, *a, **k):
            pass

        def analyze(self, symbol):
            from src.data.news import NewsBundle

            return NewsBundle(symbol=symbol, summary="test", bias="neutral", aggregate_sentiment=0.0)

    def fake_fetch_fallback(
        symbol,
        primary_tf,
        preferred_exchange,
        higher_tfs=None,
        limit=500,
        include_snapshot=True,
        config=None,
        auto_fallback=None,
    ):
        n = 80
        idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
        close = 1000.0 if symbol.startswith("BTC") else 200.0
        df = pd.DataFrame(
            {
                "open": close + 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": 100.0,
            },
            index=idx,
        )
        mtf = MultiTimeframeData(
            symbol=symbol,
            exchange_id=preferred_exchange or "binanceusdm",
            primary_tf=primary_tf,
            frames={primary_tf: df},
            snapshot=MarketSnapshot(
                symbol=symbol,
                exchange_id=preferred_exchange or "binanceusdm",
                last=float(close),
            ),
        )
        return FallbackFetchResult(
            mtf=mtf,
            client=FakeClient(exchange_id=preferred_exchange),
            requested_exchange=preferred_exchange or "binanceusdm",
            exchange_used=preferred_exchange or "binanceusdm",
            fallback_used=False,
            attempted_exchanges=[preferred_exchange or "binanceusdm"],
        )

    monkeypatch.setattr(svc, "fetch_multi_timeframe_with_fallback", fake_fetch_fallback)
    monkeypatch.setattr(svc, "NewsAnalyzer", FakeNews)

    result = svc.scan_symbols(
        ["BTC", "ETH"],
        request=svc.AnalyzeRequest(timeframe="15m", no_news=True),
        config=svc.load_config(),
    )

    assert result["ok"] is True
    assert len(result["ranked_results"]) >= 2
    assert result["ranked_results"][0]["symbol"]
    assert result["ranked_results"][0]["leverage"] <= svc.SCAN_LEVERAGE_CAP



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
