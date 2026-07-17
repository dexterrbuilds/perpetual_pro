"""Shared analysis service for API (and optional reuse by CLI).

Runs OCR + CV on a chart image, then live data + indicators + patterns +
news + confluence, returning a clean JSON-serializable dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from PIL import Image

from src.analysis.confluence import ConfluenceEngine, FullAnalysis
from src.analysis.risk import RiskManager
from src.data.exchange import ExchangeClient
from src.data.multi_tf import fetch_multi_timeframe
from src.data.news import NewsAnalyzer
from src.report.generator import ReportGenerator
from src.utils.config import AppConfig, load_config
from src.utils.helpers import normalize_symbol
from src.vision.chart_detect import ChartVision
from src.vision.ocr import OCREngine


@dataclass
class AnalyzeRequest:
    """Parameters for screenshot-driven analysis."""

    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    exchange: Optional[str] = None
    higher: Optional[List[str]] = None
    account_balance: Optional[float] = None
    risk_pct: Optional[float] = None
    no_news: bool = False
    dark_theme: Optional[bool] = None


def _parse_higher(higher: Optional[Union[str, List[str]]], config: AppConfig) -> List[str]:
    if higher is None:
        return list(config.timeframes.higher)
    if isinstance(higher, str):
        return [x.strip() for x in higher.split(",") if x.strip()]
    return list(higher)


def _load_image(image: Union[Image.Image, bytes, BytesIO]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, BytesIO):
        return Image.open(image).convert("RGB")
    return Image.open(BytesIO(image)).convert("RGB")


def analyze_from_image(
    image: Union[Image.Image, bytes, BytesIO],
    request: Optional[AnalyzeRequest] = None,
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """
    Full pro pipeline from a chart screenshot.

    Steps: OCR → CV → resolve symbol/TF → multi-TF data → news → confluence.
    Returns a clean JSON-serializable dict (ReportGenerator.to_dict + vision extras).
    """
    req = request or AnalyzeRequest()
    cfg = config or load_config()
    img = _load_image(image)
    dark = req.dark_theme if req.dark_theme is not None else cfg.screen.dark_theme

    # --- OCR ---
    ocr_engine = OCREngine(config=cfg)
    ocr_result = ocr_engine.extract(img, dark_theme=dark)
    logger.info(
        "API OCR symbol={} tf={} conf={:.2f}",
        ocr_result.symbol,
        ocr_result.timeframe,
        ocr_result.confidence,
    )

    # --- Computer vision ---
    vision = ChartVision(config=cfg)
    # Disable Ollama by default in API path if slow; still honor config
    vis = vision.analyze(img, dark_theme=dark)

    # --- Resolve symbol / timeframe ---
    resolved_symbol = req.symbol or ocr_result.symbol
    if not resolved_symbol:
        return {
            "ok": False,
            "error": "symbol_required",
            "message": (
                "Could not detect symbol from image. "
                "Pass form field 'symbol' (e.g. BTC or BTC/USDT:USDT)."
            ),
            "ocr": {
                "raw_preview": (ocr_result.raw_text or "")[:500],
                "confidence": ocr_result.confidence,
                "timeframe": ocr_result.timeframe,
                "prices": ocr_result.prices[:20],
                "engine_notes": ocr_result.engine_notes,
            },
            "vision": {
                "candles_detected": vis.candles_detected,
                "trend_guess": vis.trend_guess,
                "confidence": vis.confidence,
                "notes": vis.notes,
            },
        }

    try:
        resolved_symbol = normalize_symbol(resolved_symbol)
    except ValueError as exc:
        return {
            "ok": False,
            "error": "invalid_symbol",
            "message": str(exc),
        }

    primary_tf = req.timeframe or ocr_result.timeframe or cfg.timeframes.primary
    ex_id = (req.exchange or cfg.exchange.default).lower()
    higher_tfs = _parse_higher(req.higher, cfg)

    vision_notes = (
        f"OCR: symbol={ocr_result.symbol} tf={ocr_result.timeframe} "
        f"conf={ocr_result.confidence:.2f}; "
        f"CV: candles={vis.candles_detected} trend≈{vis.trend_guess} "
        f"conf={vis.confidence:.2f}"
    )
    if vis.ollama_summary:
        vision_notes += f" | Ollama: {vis.ollama_summary[:200]}"

    vision_payload = {
        "notes": vision_notes,
        "ocr": {
            "symbol": ocr_result.symbol,
            "timeframe": ocr_result.timeframe,
            "confidence": round(ocr_result.confidence, 4),
            "prices": ocr_result.prices[:20],
            "indicators_mentioned": ocr_result.indicators_mentioned,
            "engine_notes": ocr_result.engine_notes,
        },
        "cv": {
            "candles_detected": vis.candles_detected,
            "trend_guess": vis.trend_guess,
            "confidence": round(vis.confidence, 4),
            "notes": vis.notes,
            "horizontal_levels": len(vis.horizontal_levels_y),
            "volume_bars": vis.volume_bars,
            "ollama_summary": vis.ollama_summary or None,
        },
        "resolved_symbol": resolved_symbol,
        "resolved_timeframe": primary_tf,
        "exchange": ex_id,
    }

    client = ExchangeClient(exchange_id=ex_id, config=cfg)
    try:
        mtf = fetch_multi_timeframe(
            client,
            symbol=resolved_symbol,
            primary_tf=primary_tf,
            higher_tfs=higher_tfs,
            limit=cfg.timeframes.ohlcv_limit,
            include_snapshot=True,
            config=cfg,
        )

        if mtf.primary.empty:
            analysis = _vision_only_analysis(
                resolved_symbol=resolved_symbol,
                exchange_id=ex_id,
                primary_tf=primary_tf,
                vis_trend=vis.trend_guess,
                vis_conf=vis.confidence,
                ocr_prices=ocr_result.prices,
                vision_notes=vision_notes,
                config=cfg,
                account_balance=req.account_balance,
                risk_pct=req.risk_pct,
            )
            data_mode = "vision_only"
        else:
            news_bundle = None
            if not req.no_news and cfg.news.enabled:
                try:
                    news_bundle = NewsAnalyzer(config=cfg).analyze(resolved_symbol)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("News fetch failed in API: {}", exc)

            engine = ConfluenceEngine(cfg)
            analysis = engine.analyze(
                mtf,
                news=news_bundle,
                account_balance=req.account_balance,
                risk_pct=req.risk_pct,
            )
            data_mode = "full"
            _apply_vision_conflict_warnings(analysis, vis.trend_guess, vis.confidence)

        reporter = ReportGenerator(cfg)
        payload = reporter.to_dict(analysis, extra={"vision": vision_payload})
        payload["ok"] = True
        payload["data_mode"] = data_mode
        payload["vision"] = vision_payload
        # Keep response clean: drop huge raw blobs if any slipped in
        payload.pop("extra", None)
        return payload
    finally:
        client.close()


def _apply_vision_conflict_warnings(
    analysis: FullAnalysis, trend_guess: str, vis_conf: float
) -> None:
    if trend_guess not in ("up", "down"):
        return
    visual_bias = "bullish" if trend_guess == "up" else "bearish"
    if visual_bias != analysis.bias and analysis.bias != "neutral":
        analysis.warnings.append(
            f"Screen trend guess ({trend_guess}) conflicts with data bias "
            f"({analysis.bias}) — trust data more; verify chart symbol/TF."
        )
    elif analysis.bias == "neutral" and vis_conf > 0.4:
        analysis.warnings.append(
            f"Data neutral; screen suggests {trend_guess}. Wait for confirmation."
        )


def _vision_only_analysis(
    *,
    resolved_symbol: str,
    exchange_id: str,
    primary_tf: str,
    vis_trend: str,
    vis_conf: float,
    ocr_prices: List[float],
    vision_notes: str,
    config: AppConfig,
    account_balance: Optional[float],
    risk_pct: Optional[float],
) -> FullAnalysis:
    analysis = FullAnalysis(
        symbol=resolved_symbol,
        exchange_id=exchange_id,
        primary_tf=primary_tf,
    )
    analysis.bias = (
        "bullish" if vis_trend == "up" else ("bearish" if vis_trend == "down" else "neutral")
    )
    analysis.direction = (
        "long"
        if analysis.bias == "bullish"
        else ("short" if analysis.bias == "bearish" else "flat")
    )
    analysis.confidence = max(25.0, vis_conf * 100 * 0.6)
    analysis.setup_name = "Vision-only (data unavailable)"
    analysis.trader_commentary = (
        vision_notes
        + " Live market data unavailable; treat this as low-confidence visual read only."
    )
    analysis.warnings.append("Data fallback failed — vision-only mode")
    price = ocr_prices[len(ocr_prices) // 2] if ocr_prices else 0.0
    analysis.meta = {"price": price, "atr": price * 0.01 if price else 0}
    if price:
        rm = RiskManager(
            config=config,
            account_balance=account_balance,
            risk_pct=risk_pct,
        )
        analysis.trade_plan = rm.build_plan(
            analysis.direction, price, price * 0.01, confidence=analysis.confidence
        )
    return analysis
