"""Shared analysis service for API (and optional reuse by CLI).

Runs OCR + CV on a chart image, then live data + indicators + patterns +
news + confluence, returning a clean JSON-serializable dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from PIL import Image

from src.analysis.confluence import ConfluenceEngine, FullAnalysis
from src.analysis.risk import RiskManager
from src.data.exchange import EXCHANGE_MAP, normalize_exchange_id
from src.data.multi_tf import fetch_multi_timeframe_with_fallback
from src.data.news import NewsAnalyzer
from src.report.generator import ReportGenerator
from src.utils.config import AppConfig, load_config
from src.utils.helpers import normalize_symbol
from src.vision.chart_detect import ChartVision
from src.vision.ocr import OCREngine
from src.vision.url_symbol import parse_chart_url

# Web / scan UI prioritizes conservative display leverage (model may suggest higher)
SCAN_LEVERAGE_CAP = 5


@dataclass
class AnalyzeRequest:
    """Parameters for screenshot-driven analysis."""

    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    exchange: Optional[str] = None
    higher: Optional[List[str]] = None
    simulated_capital: Optional[float] = None  # default $1000 sim capital
    risk_pct: Optional[float] = None  # default 1%
    account_balance: Optional[float] = None  # legacy alias → simulated_capital
    no_news: bool = False
    dark_theme: Optional[bool] = None
    use_llm: bool = True
    # Client / browser fusion inputs
    page_url: Optional[str] = None
    client_ocr: Optional[Dict[str, Any]] = field(default_factory=dict)
    client_vision: Optional[Dict[str, Any]] = field(default_factory=dict)
    client_hints: Optional[Dict[str, Any]] = field(default_factory=dict)


def _parse_higher(higher: Optional[Union[str, List[str]]], config: AppConfig) -> List[str]:
    if higher is None:
        return list(config.timeframes.higher)
    if isinstance(higher, str):
        return [x.strip() for x in higher.split(",") if x.strip()]
    return list(higher)


def _resolve_exchange_id(
    raw: Optional[str],
    config: AppConfig,
    extra_hints: Optional[Dict[str, Any]] = None,
) -> str:
    """Normalize exchange id from request / hints / config default."""
    ex_id: Optional[str] = raw
    hints = extra_hints or {}
    if not ex_id:
        for key in ("exchange", "exchange_hint"):
            val = hints.get(key)
            if isinstance(val, str) and val.strip():
                ex_id = val
                break
    if not ex_id:
        ex_id = config.exchange.default
    if isinstance(ex_id, str):
        cleaned = ex_id.lower().replace("-", "").replace("_", "")
        ex_id = EXCHANGE_MAP.get(cleaned, cleaned)
    return normalize_exchange_id(str(ex_id))


def _cap_display_leverage(raw: Any, cap: int = SCAN_LEVERAGE_CAP) -> int:
    try:
        lev = int(round(float(raw or 1)))
    except (TypeError, ValueError):
        lev = 1
    return min(cap, max(1, lev))


def _load_image(image: Union[Image.Image, bytes, BytesIO]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, BytesIO):
        return Image.open(image).convert("RGB")
    return Image.open(BytesIO(image)).convert("RGB")


def _build_analysis_payload(
    analysis: FullAnalysis,
    request: Optional[AnalyzeRequest] = None,
    config: Optional[AppConfig] = None,
    image: Optional[Union[Image.Image, bytes, BytesIO]] = None,
) -> Dict[str, Any]:
    plan = analysis.trade_plan or None
    price = analysis.meta.get("price") if analysis.meta else None
    direction = analysis.direction or analysis.bias or "flat"
    payload = {
        "ok": True,
        "data_mode": "full",
        "symbol": analysis.symbol,
        "exchange": analysis.exchange_id,
        "primary_tf": analysis.primary_tf,
        "generated_at": analysis.generated_at,
        "bias": analysis.bias,
        "direction": analysis.direction,
        "confidence": analysis.confidence,
        "setup_name": analysis.setup_name,
        "strategy_tags": analysis.strategy_tags,
        "confluence_total": analysis.confluence_total,
        "confluence_breakdown": analysis.factor_breakdown(),
        "factors": analysis.factor_breakdown(),
        "trade_plan": plan.to_position_simulation() if plan else None,
        "primary_setup": plan.to_primary_setup() if plan else None,
        "position_simulation": plan.to_position_simulation() if plan else None,
        "patterns": [],
        "key_levels": analysis.key_levels,
        "key_reasons": analysis.key_reasons,
        "key_risks": analysis.key_risks,
        "trader_commentary": analysis.trader_commentary,
        "news": None,
        "snapshot": None,
        "structure": None,
        "warnings": analysis.warnings,
        "vision": None,
        "meta": analysis.meta,
        "disclaimer": "NOT FINANCIAL ADVICE",
        "signal": {
            "bias": analysis.bias,
            "direction": direction,
            "confidence_pct": analysis.confidence,
            "setup_name": analysis.setup_name,
            "strategy_tags": analysis.strategy_tags,
            "confluence_score": analysis.confluence_total,
        },
    }
    if analysis.snapshot:
        payload["snapshot"] = {
            "symbol": analysis.snapshot.symbol,
            "exchange_id": analysis.snapshot.exchange_id,
            "last": analysis.snapshot.last,
            "bid": analysis.snapshot.bid,
            "ask": analysis.snapshot.ask,
            "mark": analysis.snapshot.mark,
            "index": analysis.snapshot.index,
            "percentage_24h": analysis.snapshot.percentage_24h,
            "funding_rate": analysis.snapshot.funding_rate,
            "open_interest": analysis.snapshot.open_interest,
            "open_interest_value": analysis.snapshot.open_interest_value,
        }
    if analysis.news:
        payload["news"] = {
            "summary": analysis.news.summary,
            "aggregate_sentiment": analysis.news.aggregate_sentiment,
            "bias": analysis.news.bias,
            "items": [item.to_dict() if hasattr(item, "to_dict") else item for item in analysis.news.items],
        }
    if analysis.structure:
        payload["structure"] = {
            "summary": analysis.structure.summary,
            "structure_score": analysis.structure.structure_score,
            "wyckoff_phase": analysis.structure.wyckoff_phase,
            "volume_profile_poc": analysis.structure.volume_profile_poc,
            "volume_profile_val": analysis.structure.volume_profile_val,
            "volume_profile_vah": analysis.structure.volume_profile_vah,
        }
    if analysis.indicators:
        payload["indicators"] = {
            "summary": analysis.indicators.summary,
            "divergences": [d.to_dict() if hasattr(d, "to_dict") else d for d in analysis.indicators.divergences],
        }
    return payload


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

    # --- URL hints (TradingView first-class) ---
    url_hints = parse_chart_url(req.page_url)
    client_ocr = req.client_ocr or {}
    client_vision = req.client_vision or {}
    client_hints = req.client_hints or {}

    # --- Server OCR: Tesseract + EasyOCR (full text) ---
    ocr_engine = OCREngine(config=cfg)
    ocr_result = ocr_engine.extract(img, dark_theme=dark)
    # Merge client OCR text if richer
    if client_ocr.get("all_text") or client_ocr.get("raw"):
        extra = str(client_ocr.get("all_text") or client_ocr.get("raw") or "")
        if extra and len(extra) > 20:
            ocr_result.raw_text = (ocr_result.raw_text + "\n" + extra).strip()
            ocr_result.meta["merged_client_ocr"] = True
            # Re-parse combined text for more recall
            ocr_engine._parse(ocr_result)
    logger.info(
        "API OCR symbol={} tf={} conf={:.2f} url_symbol={}",
        ocr_result.symbol,
        ocr_result.timeframe,
        ocr_result.confidence,
        url_hints.symbol,
    )

    # --- Computer vision (OpenCV chart structure) ---
    vision = ChartVision(config=cfg)
    vis = vision.analyze(img, dark_theme=dark)
    # Soft-blend client vision trend if server weak
    if vis.trend_guess == "unknown" and client_vision.get("trend_guess") in ("up", "down", "range"):
        vis.trend_guess = client_vision["trend_guess"]
        vis.notes.append("trend from client vision")
        vis.confidence = max(vis.confidence, float(client_vision.get("confidence") or 0.3))

    # --- Resolve symbol / timeframe (priority: user > URL > server OCR > client OCR) ---
    resolve_notes: List[str] = []
    resolved_symbol = req.symbol
    if resolved_symbol:
        resolve_notes.append("symbol=user")
    elif url_hints.symbol and url_hints.confidence >= 0.5:
        resolved_symbol = url_hints.symbol
        resolve_notes.append(f"symbol=url({url_hints.source})")
    elif ocr_result.symbol:
        resolved_symbol = ocr_result.symbol
        resolve_notes.append("symbol=server_ocr")
    elif client_ocr.get("symbol"):
        resolved_symbol = str(client_ocr["symbol"])
        resolve_notes.append("symbol=client_ocr")

    if not resolved_symbol:
        # Even without symbol, return vision analysis so client can prompt manually
        return {
            "ok": False,
            "error": "symbol_required",
            "message": (
                "Could not detect symbol from image, URL, or OCR. "
                "Pass form field 'symbol' (e.g. BTC) — chart vision still ran."
            ),
            "ocr": {
                "raw_preview": (ocr_result.raw_text or "")[:800],
                "confidence": ocr_result.confidence,
                "timeframe": ocr_result.timeframe,
                "prices": ocr_result.prices[:20],
                "indicators_mentioned": ocr_result.indicators_mentioned,
                "engine_notes": ocr_result.engine_notes,
                "all_text_len": len(ocr_result.raw_text or ""),
            },
            "vision": {
                "candles_detected": vis.candles_detected,
                "trend_guess": vis.trend_guess,
                "confidence": vis.confidence,
                "notes": vis.notes,
                "horizontal_levels": len(vis.horizontal_levels_y),
            },
            "url_hints": {
                "symbol": url_hints.symbol,
                "timeframe": url_hints.timeframe,
                "exchange_hint": url_hints.exchange_hint,
                "source": url_hints.source,
                "confidence": url_hints.confidence,
                "page_url": req.page_url,
            },
            "client_vision": client_vision or None,
            "resolve_notes": resolve_notes,
        }

    try:
        resolved_symbol = normalize_symbol(resolved_symbol)
    except ValueError as exc:
        return {
            "ok": False,
            "error": "invalid_symbol",
            "message": str(exc),
            "url_hints": {"raw": url_hints.raw_pair, "source": url_hints.source},
        }

    primary_tf = (
        req.timeframe
        or url_hints.timeframe
        or ocr_result.timeframe
        or (client_ocr.get("timeframe") if client_ocr else None)
        or cfg.timeframes.primary
    )
    ex_id = _resolve_exchange_id(
        req.exchange or url_hints.exchange_hint,
        cfg,
        extra_hints=client_hints if isinstance(client_hints, dict) else None,
    )
    higher_tfs = _parse_higher(req.higher, cfg)
    sim_capital = (
        req.simulated_capital
        if req.simulated_capital is not None
        else req.account_balance
    )
    risk_pct = req.risk_pct

    vision_notes = (
        f"OCR: symbol={ocr_result.symbol} tf={ocr_result.timeframe} "
        f"conf={ocr_result.confidence:.2f}; "
        f"URL: {url_hints.symbol or '—'} ({url_hints.source or 'n/a'}); "
        f"CV: candles={vis.candles_detected} trend≈{vis.trend_guess} "
        f"conf={vis.confidence:.2f}; resolve={','.join(resolve_notes) or 'n/a'}"
    )
    if vis.ollama_summary:
        vision_notes += f" | Ollama: {vis.ollama_summary[:200]}"
    if client_vision.get("trend_guess"):
        vision_notes += f" | client_cv={client_vision.get('trend_guess')}"

    vision_payload = {
        "notes": vision_notes,
        "ocr": {
            "symbol": ocr_result.symbol,
            "timeframe": ocr_result.timeframe,
            "confidence": round(ocr_result.confidence, 4),
            "prices": ocr_result.prices[:30],
            "indicators_mentioned": ocr_result.indicators_mentioned,
            "engine_notes": ocr_result.engine_notes,
            "all_text_preview": (ocr_result.raw_text or "")[:600],
            "lines_count": len(ocr_result.lines),
        },
        "url_hints": {
            "symbol": url_hints.symbol,
            "timeframe": url_hints.timeframe,
            "exchange_hint": url_hints.exchange_hint,
            "source": url_hints.source,
            "confidence": url_hints.confidence,
            "page_url": req.page_url,
        },
        "cv": {
            "candles_detected": vis.candles_detected,
            "trend_guess": vis.trend_guess,
            "confidence": round(vis.confidence, 4),
            "notes": vis.notes,
            "horizontal_levels": len(vis.horizontal_levels_y),
            "approx_support_ys": getattr(vis, "approx_support_ys", [])[:5],
            "approx_resistance_ys": getattr(vis, "approx_resistance_ys", [])[:5],
            "volume_bars": vis.volume_bars,
            "ollama_summary": vis.ollama_summary or None,
        },
        "client_ocr": {
            "symbol": client_ocr.get("symbol"),
            "timeframe": client_ocr.get("timeframe"),
            "confidence": client_ocr.get("confidence"),
            "indicators": client_ocr.get("indicators") or client_ocr.get("indicators_mentioned"),
        }
        if client_ocr
        else None,
        "client_vision": client_vision or None,
        "resolve_notes": resolve_notes,
        "resolved_symbol": resolved_symbol,
        "resolved_timeframe": primary_tf,
        "exchange": ex_id,
    }

    fetch = fetch_multi_timeframe_with_fallback(
        symbol=resolved_symbol,
        primary_tf=primary_tf,
        preferred_exchange=ex_id,
        higher_tfs=higher_tfs,
        limit=cfg.timeframes.ohlcv_limit,
        include_snapshot=True,
        config=cfg,
    )
    client = fetch.client
    try:
        mtf = fetch.mtf
        exchange_used = fetch.exchange_used
        vision_payload["exchange"] = exchange_used
        vision_payload["exchange_requested"] = fetch.requested_exchange
        vision_payload["fallback_used"] = fetch.fallback_used
        vision_payload["attempted_exchanges"] = list(fetch.attempted_exchanges)

        if mtf.primary.empty:
            analysis = _vision_only_analysis(
                resolved_symbol=resolved_symbol,
                exchange_id=exchange_used,
                primary_tf=primary_tf,
                vis_trend=vis.trend_guess,
                vis_conf=vis.confidence,
                ocr_prices=ocr_result.prices,
                vision_notes=vision_notes,
                config=cfg,
                simulated_capital=sim_capital,
                risk_pct=risk_pct,
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
                simulated_capital=sim_capital,
                risk_pct=risk_pct,
                use_llm=req.use_llm,
            )
            data_mode = "full"
            _apply_vision_conflict_warnings(analysis, vis.trend_guess, vis.confidence)

        reporter = ReportGenerator(cfg)
        payload = reporter.to_dict(analysis, extra={"vision": vision_payload})
        payload["ok"] = True
        payload["data_mode"] = data_mode
        payload["vision"] = vision_payload
        payload["exchange"] = exchange_used
        payload["exchange_requested"] = fetch.requested_exchange
        payload["fallback_used"] = fetch.fallback_used
        payload["attempted_exchanges"] = list(fetch.attempted_exchanges)
        if fetch.fallback_used:
            payload.setdefault("warnings", [])
            if isinstance(payload["warnings"], list):
                payload["warnings"].append(
                    f"Data from {exchange_used} (requested {fetch.requested_exchange}; "
                    f"auto-fallback)"
                )
        # Keep response clean: drop huge raw blobs if any slipped in
        payload.pop("extra", None)
        return payload
    finally:
        client.close()


def analyze_market_data(
    symbol: str,
    request: Optional[AnalyzeRequest] = None,
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """
    Data-only analysis (no chart image) with multi-exchange fallback.

    Used by the Streamlit web app and any symbol-first workflow.
    """
    req = request or AnalyzeRequest()
    cfg = config or load_config()
    try:
        resolved_symbol = normalize_symbol(symbol or req.symbol or "")
    except ValueError as exc:
        return {"ok": False, "error": "invalid_symbol", "message": str(exc)}

    primary_tf = req.timeframe or cfg.timeframes.primary
    ex_id = _resolve_exchange_id(req.exchange, cfg)
    higher_tfs = _parse_higher(req.higher, cfg)
    sim_capital = (
        req.simulated_capital
        if req.simulated_capital is not None
        else (req.account_balance if req.account_balance is not None else 100.0)
    )
    risk_pct = req.risk_pct if req.risk_pct is not None else 1.0

    fetch = fetch_multi_timeframe_with_fallback(
        symbol=resolved_symbol,
        primary_tf=primary_tf,
        preferred_exchange=ex_id,
        higher_tfs=higher_tfs,
        limit=cfg.timeframes.ohlcv_limit,
        include_snapshot=True,
        config=cfg,
    )
    client = fetch.client
    try:
        mtf = fetch.mtf
        if mtf.primary.empty:
            return {
                "ok": False,
                "error": "no_market_data",
                "message": (
                    f"No OHLCV for {resolved_symbol} after trying: "
                    f"{', '.join(fetch.attempted_exchanges) or ex_id}"
                ),
                "exchange_requested": fetch.requested_exchange,
                "attempted_exchanges": list(fetch.attempted_exchanges),
            }

        news_bundle = None
        if not req.no_news and cfg.news.enabled:
            try:
                news_bundle = NewsAnalyzer(config=cfg).analyze(resolved_symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning("News fetch failed in analyze_market_data: {}", exc)

        engine = ConfluenceEngine(cfg)
        analysis = engine.analyze(
            mtf,
            news=news_bundle,
            simulated_capital=sim_capital,
            risk_pct=risk_pct,
            use_llm=req.use_llm,
        )
        reporter = ReportGenerator(cfg)
        payload = reporter.to_dict(analysis)
        payload["ok"] = True
        payload["data_mode"] = "full"
        payload["exchange"] = fetch.exchange_used
        payload["exchange_requested"] = fetch.requested_exchange
        payload["fallback_used"] = fetch.fallback_used
        payload["attempted_exchanges"] = list(fetch.attempted_exchanges)
        if fetch.fallback_used:
            payload.setdefault("warnings", [])
            if isinstance(payload["warnings"], list):
                payload["warnings"].append(
                    f"Data from {fetch.exchange_used} "
                    f"(requested {fetch.requested_exchange}; auto-fallback)"
                )
        # Practical web display: prioritize 5x for sim example alongside model lev
        plan = analysis.trade_plan
        model_lev = float(getattr(plan, "leverage_suggested", 20) or 20) if plan else 20.0
        payload["display_leverage"] = _cap_display_leverage(model_lev, SCAN_LEVERAGE_CAP)
        payload["model_leverage"] = int(round(model_lev))
        payload.pop("extra", None)
        return payload
    finally:
        client.close()


def scan_symbols(
    symbols: Optional[List[str]] = None,
    request: Optional[AnalyzeRequest] = None,
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """Run a fast multi-symbol scan and rank setups by confluence score."""
    req = request or AnalyzeRequest()
    cfg = config or load_config()
    symbol_list = [s.strip() for s in (symbols or []) if s and s.strip()]
    if not symbol_list:
        return {"ok": False, "error": "no_symbols", "ranked_results": []}

    primary_tf = req.timeframe or cfg.timeframes.primary
    ex_id = _resolve_exchange_id(req.exchange, cfg)

    ranked_results: List[Dict[str, Any]] = []
    for symbol in symbol_list[:40]:
        try:
            normalized_symbol = normalize_symbol(symbol)
            fetch = fetch_multi_timeframe_with_fallback(
                symbol=normalized_symbol,
                primary_tf=primary_tf,
                preferred_exchange=ex_id,
                higher_tfs=["1h", "4h"],
                limit=120,
                include_snapshot=True,
                config=cfg,
            )
            client = fetch.client
            try:
                mtf = fetch.mtf
                if mtf.primary.empty:
                    logger.info(
                        "Scan skip {}: empty OHLCV after {}",
                        normalized_symbol,
                        " → ".join(fetch.attempted_exchanges),
                    )
                    continue

                news_bundle = None
                if not req.no_news and cfg.news.enabled:
                    try:
                        news_bundle = NewsAnalyzer(config=cfg).analyze(normalized_symbol)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Scan news failed for {}: {}", normalized_symbol, exc)

                engine = ConfluenceEngine(cfg)
                analysis = engine.analyze(
                    mtf,
                    news=news_bundle,
                    simulated_capital=req.simulated_capital or 100.0,
                    risk_pct=req.risk_pct or 1.0,
                    use_llm=False,
                )
                model_lev = getattr(analysis.trade_plan, "leverage_suggested", 0) or 1
                leverage = _cap_display_leverage(model_lev, SCAN_LEVERAGE_CAP)
                reason = (analysis.key_reasons[:2] or [analysis.setup_name or ""])[0]
                primary = (
                    analysis.trade_plan.to_primary_setup() if analysis.trade_plan else None
                )
                ranked_results.append(
                    {
                        "symbol": normalized_symbol,
                        "exchange": fetch.exchange_used,
                        "exchange_requested": fetch.requested_exchange,
                        "fallback_used": fetch.fallback_used,
                        "primary_tf": primary_tf,
                        "direction": analysis.direction,
                        "bias": analysis.bias,
                        "confidence": round(float(analysis.confidence), 1),
                        "confluence_score": round(float(analysis.confluence_total), 3),
                        "setup_name": analysis.setup_name,
                        "leverage": leverage,
                        "model_leverage": int(round(float(model_lev))),
                        "reason": reason,
                        "price": analysis.meta.get("price") if analysis.meta else None,
                        "support": analysis.key_levels[0].get("mid") if analysis.key_levels else None,
                        "resistance": analysis.key_levels[1].get("mid") if len(analysis.key_levels) > 1 else None,
                        "entry_low": getattr(analysis.trade_plan, "entry_low", None) if analysis.trade_plan else None,
                        "entry_high": getattr(analysis.trade_plan, "entry_high", None) if analysis.trade_plan else None,
                        "stop_loss": getattr(analysis.trade_plan, "stop_loss", None) if analysis.trade_plan else None,
                        "hold_label": getattr(analysis.trade_plan, "hold_label", "") if analysis.trade_plan else "",
                        "payload": {
                            "bias": analysis.bias,
                            "direction": analysis.direction,
                            "confidence": analysis.confidence,
                            "setup_name": analysis.setup_name,
                            "confluence_total": analysis.confluence_total,
                            "key_levels": analysis.key_levels[:4],
                            "key_reasons": analysis.key_reasons[:3],
                            "trade_plan": primary,
                            "primary_setup": primary,
                            "position_simulation": (
                                analysis.trade_plan.to_position_simulation()
                                if analysis.trade_plan
                                else None
                            ),
                        },
                    }
                )
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scan failed for {}: {}", symbol, exc)

    ranked_results.sort(key=lambda item: item["confluence_score"], reverse=True)
    return {
        "ok": True,
        "ranked_results": ranked_results[:10],
        "count": len(ranked_results),
        "timeframe": primary_tf,
        "exchange": ex_id,
        "leverage_display_cap": SCAN_LEVERAGE_CAP,
    }


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
    simulated_capital: Optional[float],
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
    analysis.meta = {"price": price, "atr": price * 0.01 if price else 0, "is_simulation": True}
    if price:
        rm = RiskManager(
            config=config,
            simulated_capital=simulated_capital,
            risk_pct=risk_pct,
        )
        analysis.trade_plan = rm.build_plan(
            analysis.direction, price, price * 0.01, confidence=analysis.confidence
        )
    return analysis
