"""Shared analysis service for API (and optional reuse by CLI).

Runs OCR + CV on a chart image, then live data + indicators + patterns +
news + confluence, returning a clean JSON-serializable dict.
"""

from __future__ import annotations

import copy
import json
import time
import inspect
from math import isfinite
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Union
from uuid import uuid4

from loguru import logger
from PIL import Image

from src.analytics.rejection import (
    ALERT_MIN_OVERALL_QUALITY,
    GATE_POLICY_VERSION,
    aggregate_rejection_rows,
    candidate_analytics_snapshot,
    evaluate_alert_gates,
)
from src.analytics.runtime import get_rejection_repository
from src.analysis.confluence import ConfluenceEngine, FullAnalysis
from src.analysis.qualification import evaluate_private_beta_qualification
from src.experiments.identity import identity_metadata, is_legacy_comparison
from src.experiments.legacy_policy import (
    LEGACY_POLICY_VERSION,
    evaluate_legacy_qualification,
    strict_shadow_decision,
)
from src.analysis.risk import RiskManager
from src.data.exchange import EXCHANGE_MAP, normalize_exchange_id
from src.data.multi_tf import fetch_multi_timeframe_with_fallback
from src.data.news import NewsAnalyzer
from src.report.charts import build_market_chart_payload
from src.report.generator import ReportGenerator
from src.scoring.features import build_candidate_record
from src.scoring.runtime import (
    get_outcome_scoring_status,
    journal_scan_candidates,
    score_candidate_shadow,
)
from src.utils.config import AppConfig, load_config
from src.utils.build_info import get_build_identity
from src.utils.helpers import clamp, normalize_symbol
from src.api.security import SCAN_BUDGET_SECONDS, SCAN_FALLBACK_EXCHANGES
from src.vision.chart_detect import ChartVision
from src.vision.ocr import OCREngine
from src.vision.url_symbol import parse_chart_url

# Web / scan UI prioritizes conservative display leverage (model may suggest higher)
SCAN_LEVERAGE_CAP = 5
MAX_INTERNAL_SCAN_SYMBOLS = 50


def apply_diagnostic_backtest_to_rank(
    live_rank_score: Optional[float],
    backtest_summary: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Quick proxy backtests are diagnostics and never alter live rank."""
    if live_rank_score is None:
        return None
    value = float(live_rank_score)
    return value if isfinite(value) else None


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
            "funding_average_24h": analysis.snapshot.funding_average_24h,
            "open_interest": analysis.snapshot.open_interest,
            "open_interest_value": analysis.snapshot.open_interest_value,
            "open_interest_change_pct_24h": analysis.snapshot.open_interest_change_pct_24h,
            "spread_bps": analysis.snapshot.spread_bps,
            "orderbook_imbalance": analysis.snapshot.orderbook_imbalance,
            "mark_index_basis_bps": analysis.snapshot.mark_index_basis_bps,
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
        payload["chart"] = build_market_chart_payload(
            mtf.primary,
            analysis.indicators,
            analysis.structure,
            analysis.patterns,
            analysis.trade_plan,
            timeframe=primary_tf,
        )
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
        payload["chart"] = build_market_chart_payload(
            mtf.primary,
            analysis.indicators,
            analysis.structure,
            analysis.patterns,
            analysis.trade_plan,
            timeframe=primary_tf,
        )
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


def _build_directional_comparison_row(
    row: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Restore the evaluated thesis for policy comparison and journaling.

    Strict publication gates may intentionally flatten ``direction`` while
    retaining ``evaluated_direction``. The Legacy A/B experiment must compare
    both policies against that same preserved directional thesis.
    """
    evaluated_direction = str(row.get("evaluated_direction") or "").lower()
    if evaluated_direction not in {"long", "short"}:
        return None
    comparison_row = copy.deepcopy(dict(row))
    comparison_row["direction"] = evaluated_direction
    payload = comparison_row.get("payload") or {}
    payload["direction"] = evaluated_direction
    for key in ("primary_setup", "trade_plan"):
        if isinstance(payload.get(key), dict):
            payload[key]["direction"] = evaluated_direction
            payload[key]["entry_status"] = comparison_row.get("entry_status")
    chart_trade = (
        (payload.get("chart") or {}).get("trade")
        if isinstance(payload.get("chart"), dict)
        else None
    )
    if isinstance(chart_trade, dict):
        chart_trade["direction"] = evaluated_direction
        chart_trade["entry_status"] = comparison_row.get("entry_status")
    comparison_row["payload"] = payload
    return comparison_row


def scan_symbols(
    symbols: Optional[List[str]] = None,
    request: Optional[AnalyzeRequest] = None,
    config: Optional[AppConfig] = None,
    *,
    scan_id: Optional[str] = None,
    trigger_type: str = "manual",
) -> Dict[str, Any]:
    """
    Multi-symbol scan ranked by deterministic technical/execution quality.

    Flat/neutral setups are excluded from the leaderboard (low priority).
    """
    req = request or AnalyzeRequest()
    cfg = config or load_config()
    resolved_scan_id = str(scan_id or ("scan_" + uuid4().hex))
    scan_started_at = datetime.now(timezone.utc)
    scan_started_monotonic = time.monotonic()
    symbol_list = [s.strip() for s in (symbols or []) if s and s.strip()]
    if not symbol_list:
        return {"ok": False, "error": "no_symbols", "ranked_results": [], "skipped_flat": []}

    primary_tf = req.timeframe or cfg.timeframes.primary
    ex_id = _resolve_exchange_id(req.exchange, cfg)
    # LLM output is explanation/warnings only; deterministic code owns approval.
    use_llm = bool(getattr(req, "use_llm", True))

    ranked_results: List[Dict[str, Any]] = []
    qualification_candidates: List[Dict[str, Any]] = []
    skipped_flat: List[Dict[str, Any]] = []
    journal_records: List[Dict[str, Any]] = []
    analytics_candidates: List[Dict[str, Any]] = []
    llm_call_count = 0
    llm_rate_limit_events = 0
    analysis_failures: List[Dict[str, str]] = []
    analyzed_count = 0
    scan_deadline = time.monotonic() + SCAN_BUDGET_SECONDS
    for symbol in symbol_list[:MAX_INTERNAL_SCAN_SYMBOLS]:
        if time.monotonic() >= scan_deadline:
            analysis_failures.append(
                {
                    "symbol": symbol,
                    "code": "MARKET_UNAVAILABLE",
                    "reason": "scan_budget_exhausted",
                }
            )
            continue
        try:
            normalized_symbol = normalize_symbol(symbol)
            fetch_kwargs = {
                "symbol": normalized_symbol,
                "primary_tf": primary_tf,
                "preferred_exchange": ex_id,
                "higher_tfs": ["1h", "4h"],
                "limit": 120,
                "include_snapshot": True,
                "config": cfg,
            }
            # Test adapters and third-party wrappers written before the bounded
            # fallback parameters remain compatible without weakening the real
            # production budget.
            signature = inspect.signature(fetch_multi_timeframe_with_fallback)
            accepts_kwargs = any(
                item.kind == inspect.Parameter.VAR_KEYWORD
                for item in signature.parameters.values()
            )
            if accepts_kwargs or "max_exchanges" in signature.parameters:
                fetch_kwargs["max_exchanges"] = SCAN_FALLBACK_EXCHANGES
            if accepts_kwargs or "deadline_monotonic" in signature.parameters:
                fetch_kwargs["deadline_monotonic"] = scan_deadline
            fetch = fetch_multi_timeframe_with_fallback(**fetch_kwargs)
            client = fetch.client
            try:
                mtf = fetch.mtf
                if mtf.primary.empty:
                    unsupported_market = any(
                        "unsupported_market:" in str(error).lower()
                        for error in mtf.errors
                    )
                    logger.info(
                        "Scan skip {}: {} after {}",
                        normalized_symbol,
                        "unsupported market" if unsupported_market else "empty OHLCV",
                        " → ".join(fetch.attempted_exchanges),
                    )
                    analysis_failures.append(
                        {
                            "symbol": normalized_symbol,
                            "code": "MARKET_UNAVAILABLE",
                            "reason": (
                                "unsupported_market"
                                if unsupported_market
                                else "no_primary_market_data"
                            ),
                        }
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
                    simulated_capital=(
                        req.simulated_capital
                        if req.simulated_capital is not None
                        else cfg.risk.simulated_capital
                    ),
                    risk_pct=(
                        req.risk_pct
                        if req.risk_pct is not None
                        else cfg.risk.risk_per_trade_pct
                    ),
                    use_llm=use_llm,
                )
                backtest_summary: Dict[str, Any] = {}
                try:
                    from src.analysis.backtest import run_backtest

                    bt = run_backtest(
                        normalized_symbol,
                        timeframe=primary_tf,
                        bars=len(mtf.primary),
                        config=cfg,
                        df=mtf.primary,
                        step=3,
                        warmup=min(80, max(40, len(mtf.primary) // 3)),
                        indicator_suite=analysis.indicators,
                    )
                    sample_ok = bt.n_trades >= 8
                    sample_reliable = bt.n_trades >= 12
                    validation_score = float(
                        clamp(
                            50.0
                            + bt.expectancy_r * 18.0
                            + (min(bt.profit_factor, 2.5) - 1.0) * 8.0
                            + (bt.win_rate_lower_bound - 35.0) * 0.25
                            - max(0.0, bt.early_stop_rate - 35.0) * 0.35
                            - max(0.0, bt.max_drawdown_pct - 3.0) * 2.0,
                            0.0,
                            100.0,
                        )
                    ) if sample_ok else 50.0
                    historical_edge_ok = bool(
                        not sample_reliable
                        or (
                            bt.expectancy_r > -0.10
                            and bt.profit_factor >= 0.75
                            and bt.early_stop_rate <= 65.0
                        )
                    )
                    backtest_summary = {
                        "strategy_scope": "diagnostic_proxy_not_production_engine",
                        "live_authority": False,
                        "sample_ok": sample_ok,
                        "sample_reliable": sample_reliable,
                        "n_signals": bt.n_signals,
                        "n_trades": bt.n_trades,
                        "unfilled_signals": bt.unfilled_signals,
                        "win_rate": bt.win_rate,
                        "win_rate_lower_bound": bt.win_rate_lower_bound,
                        "stop_out_rate": bt.stop_out_rate,
                        "early_stop_rate": bt.early_stop_rate,
                        "expectancy_r": bt.expectancy_r,
                        "median_mae_r": bt.median_mae_r,
                        "profit_factor": bt.profit_factor,
                        "max_drawdown_pct": bt.max_drawdown_pct,
                        "net_pnl_pct": bt.net_pnl_pct,
                        "validation_score": round(validation_score, 1),
                        "diagnostic_historical_edge_ok": historical_edge_ok,
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Scan backtest skipped for {}: {}", normalized_symbol, exc)
                    validation_score = 50.0
                    sample_ok = False
                    historical_edge_ok = True

                model_lev = getattr(analysis.trade_plan, "leverage_suggested", 0) or 1
                leverage = _cap_display_leverage(model_lev, SCAN_LEVERAGE_CAP)
                reason = (
                    (analysis.key_reasons[:1] or [analysis.setup_name or ""])[0]
                )
                plan = analysis.trade_plan
                primary = plan.to_primary_setup() if plan else None
                prop_safe = bool(getattr(plan, "prop_safe", True)) if plan else True
                prop_flags = list(getattr(plan, "prop_flags", None) or []) if plan else []
                prop_guidance = dict(
                    getattr(plan, "prop_guidance", None) or {}
                ) if plan else {}
                execution = (
                    analysis.execution.to_dict()
                    if getattr(analysis, "execution", None)
                    else {}
                )
                chart = build_market_chart_payload(
                    mtf.primary,
                    analysis.indicators,
                    analysis.structure,
                    analysis.patterns,
                    plan,
                    timeframe=primary_tf,
                    limit=100,
                )
                rank_available = bool(
                    analysis.meta.get("rank_available", False)
                )
                live_rank_score = (
                    float(analysis.meta.get("rank_score"))
                    if rank_available
                    and analysis.meta.get("rank_score") is not None
                    and isfinite(float(analysis.meta.get("rank_score")))
                    else None
                )
                # This quick backtest uses a proxy strategy, not the production
                # signal lifecycle. It is diagnostic only and has no live rank,
                # approval, or rejection authority.
                scan_rank_score = apply_diagnostic_backtest_to_rank(
                    live_rank_score, backtest_summary
                )
                primary_quality = dict(
                    (analysis.meta.get("data_quality") or {}).get(primary_tf) or {}
                )
                row = {
                    "symbol": normalized_symbol,
                    "exchange": fetch.exchange_used,
                    "exchange_requested": fetch.requested_exchange,
                    "fallback_used": fetch.fallback_used,
                    "primary_tf": primary_tf,
                    "direction": analysis.direction,
                    "bias": analysis.bias,
                    "confidence": round(float(analysis.confidence), 1),
                    "legacy_confidence": round(
                        float(analysis.meta.get("legacy_confidence", analysis.confidence)),
                        1,
                    ),
                    "legacy_v2_confidence": round(
                        float(
                            analysis.meta.get(
                                "legacy_v2_confidence",
                                analysis.confidence,
                            )
                        ),
                        1,
                    ),
                    "scoring_policy": (
                        "legacy_v2"
                        if analysis.meta.get("legacy_v2_enabled", False)
                        else "legacy"
                    ),
                    "technical_confidence": round(float(analysis.technical_confidence), 1),
                    "llm_confidence": round(float(analysis.llm_confidence), 1),
                    "llm_confidence_reason": analysis.llm_confidence_reason,
                    "llm_confidence_detail": getattr(analysis, "llm_confidence_detail", {}) or {},
                    "llm_invocation_status": analysis.meta.get(
                        "llm_invocation_status", "unknown"
                    ),
                    "llm_provider": analysis.meta.get("llm_provider", "none"),
                    "llm_rate_limit_events": int(
                        analysis.meta.get("llm_rate_limit_events") or 0
                    ),
                    "rank_score": (
                        round(scan_rank_score, 2)
                        if scan_rank_score is not None
                        else None
                    ),
                    "authoritative_rank": (
                        round(live_rank_score, 2)
                        if live_rank_score is not None
                        else None
                    ),
                    "rank_available": rank_available,
                    "live_rank_score": (
                        round(live_rank_score, 2)
                        if live_rank_score is not None
                        else None
                    ),
                    "rank_policy_version": analysis.meta.get("rank_policy_version"),
                    "rank_breakdown": dict(analysis.meta.get("rank_breakdown") or {}),
                    "confluence_score": round(float(analysis.confluence_total), 3),
                    "setup_name": analysis.setup_name,
                    "leverage": leverage,
                    "model_leverage": int(round(float(model_lev))),
                    "risk_pct": round(float(getattr(plan, "risk_pct", 1.0) or 1.0), 2) if plan else 1.0,
                    "prop_safe": prop_safe,
                    "prop_guidance": prop_guidance,
                    "universal_eligible": bool(
                        analysis.meta.get("universal_eligible", False)
                        if analysis.meta else False
                    ),
                    "production_qualified": bool(
                        analysis.meta.get("production_qualified", False)
                        if analysis.meta else False
                    ),
                    "signal_eligible": bool(
                        analysis.meta.get("signal_eligible", False) if analysis.meta else False
                    ),
                    "evaluated_direction": analysis.meta.get(
                        "evaluated_direction", analysis.direction
                    ),
                    "gate_evaluation": dict(
                        analysis.meta.get("gate_evaluation") or {}
                    ),
                    "gate_policy_version": analysis.meta.get(
                        "gate_policy_version", GATE_POLICY_VERSION
                    ),
                    "feature_schema_version": cfg.outcome_scoring.feature_schema_version,
                    "legacy_signal_eligible": bool(
                        analysis.meta.get("legacy_signal_eligible", False)
                    ),
                    "legacy_v2_signal_eligible": bool(
                        analysis.meta.get("legacy_v2_signal_eligible", False)
                    ),
                    "rejection_reasons": list(
                        analysis.meta.get("rejection_reasons") or []
                    ),
                    "prop_flags": prop_flags,
                    "reason": reason,
                    "price": analysis.meta.get("price") if analysis.meta else None,
                    "atr": analysis.meta.get("atr") if analysis.meta else None,
                    "support": analysis.key_levels[0].get("mid") if analysis.key_levels else None,
                    "resistance": analysis.key_levels[1].get("mid") if len(analysis.key_levels) > 1 else None,
                    "entry_low": getattr(plan, "entry_low", None) if plan else None,
                    "entry_high": getattr(plan, "entry_high", None) if plan else None,
                    "stop_loss": getattr(plan, "stop_loss", None) if plan else None,
                    "take_profits": list(getattr(plan, "take_profits", None) or []),
                    "hold_label": getattr(plan, "hold_label", "") if plan else "",
                    "hold_detail": getattr(plan, "hold_detail", "") if plan else "",
                    "hold_hours_min": getattr(plan, "hold_hours_min", 0.0) if plan else 0.0,
                    "hold_hours_typical_max": (
                        getattr(plan, "hold_hours_typical_max", 0.0)
                        if plan
                        else 0.0
                    ),
                    "hold_hours_max": getattr(plan, "hold_hours_max", 24.0) if plan else 24.0,
                    "signal_generated_at": (
                        getattr(plan, "signal_generated_at", "") if plan else ""
                    ),
                    "entry_valid_until": (
                        getattr(plan, "entry_valid_until", "") if plan else ""
                    ),
                    "entry_valid_for_minutes": (
                        getattr(plan, "entry_valid_for_minutes", 0) if plan else 0
                    ),
                    "entry_expiry_bars": (
                        getattr(plan, "entry_expiry_bars", 0) if plan else 0
                    ),
                    "entry_expiry_reason": (
                        getattr(plan, "entry_expiry_reason", "") if plan else ""
                    ),
                    "time_stop_reason": (
                        getattr(plan, "time_stop_reason", "") if plan else ""
                    ),
                    "backtest": backtest_summary,
                    "historical_edge_ok": True,
                    "diagnostic_historical_edge_ok": historical_edge_ok,
                    "data_quality_ok": bool(
                        analysis.meta.get("primary_data_quality_ok", True)
                    ),
                    "data_quality_score": round(
                        float(primary_quality.get("score", 100.0)),
                        1,
                    ),
                    "data_quality_reason": primary_quality.get("reason"),
                    "entry_status": execution.get("status", "blocked"),
                    "execution_score": round(float(execution.get("execution_quality", execution.get("score")) or 0), 1),
                    "execution_quality": round(float(execution.get("execution_quality", execution.get("score")) or 0), 1),
                    "legacy_execution_score": round(float(execution.get("legacy_execution_score") or 0), 1),
                    "legacy_execution_status": execution.get("legacy_status"),
                    "legacy_execution_targets": list(execution.get("legacy_targets") or []),
                    "execution_policy_version": execution.get("policy_version"),
                    "execution_setup_type": execution.get("setup_type"),
                    "entry_mode": execution.get("entry_mode"),
                    "execution_components": dict(execution.get("components") or {}),
                    "entry_accessibility": (execution.get("components") or {}).get("entry_accessibility"),
                    "pre_entry_survival": (execution.get("components") or {}).get("pre_entry_survival"),
                    "confirmation_quality": (execution.get("components") or {}).get("confirmation_quality"),
                    "stop_quality": (execution.get("components") or {}).get("stop_quality"),
                    "target_feasibility": list(execution.get("target_feasibility") or []),
                    "target_policy_version": execution.get("target_policy_version"),
                    "target_adjustment": dict(execution.get("target_adjustment") or {}),
                    "gross_risk_reward": list(execution.get("gross_risk_reward") or []),
                    "net_risk_reward": list(execution.get("net_risk_reward") or []),
                    "estimated_total_cost_bps": execution.get("estimated_total_cost_bps"),
                    "estimated_fee_bps": execution.get("estimated_fee_bps"),
                    "estimated_slippage_bps": execution.get("estimated_slippage_bps"),
                    "estimated_funding_bps": execution.get("estimated_funding_bps"),
                    "estimated_impact_bps": execution.get("estimated_impact_bps"),
                    "depth_bands_bps": dict(execution.get("depth_bands_bps") or {}),
                    "structural_obstacle_distances_atr": list(execution.get("structural_obstacle_distances_atr") or []),
                    "data_freshness_state": execution.get("data_freshness_state"),
                    "entry_distance_atr": execution.get("entry_distance_atr"),
                    "entry_distance_pct": execution.get("entry_distance_pct"),
                    "entry_zone_width_atr": execution.get("entry_zone_width_atr"),
                    "remaining_expiry_minutes": execution.get("remaining_expiry_minutes"),
                    "stop_distance_atr": execution.get("stop_distance_atr"),
                    "hard_failures": list(execution.get("hard_failures") or []),
                    "execution_uncertainties": list(execution.get("uncertainties") or []),
                    "immediate_sl_risk": round(
                        float(execution.get("immediate_sl_risk") or 100), 1
                    ),
                    "chase_distance_atr": round(
                        float(execution.get("chase_distance_atr") or 0), 2
                    ),
                    "entry_zone_relation": execution.get(
                        "entry_zone_relation",
                        "unknown",
                    ),
                    "tp1_progress_pct": round(
                        float(execution.get("tp1_progress_pct") or 0),
                        1,
                    ),
                    "order_flow_score": round(
                        float(execution.get("order_flow_score") or 0), 3
                    ),
                    "spread_bps": execution.get("spread_bps"),
                    "orderbook_imbalance": execution.get("orderbook_imbalance"),
                    "orderbook_alignment": execution.get("orderbook_alignment"),
                    "ticker_age_seconds": execution.get("ticker_age_seconds"),
                    "orderbook_age_seconds": execution.get("orderbook_age_seconds"),
                    "market_quality_ok": bool(
                        execution.get("market_quality_ok", True)
                    ),
                    "funding_rate": (
                        analysis.snapshot.funding_rate
                        if analysis.snapshot is not None
                        else None
                    ),
                    "funding_average_24h": (
                        analysis.snapshot.funding_average_24h
                        if analysis.snapshot is not None
                        else None
                    ),
                    "open_interest_change_pct_24h": (
                        analysis.snapshot.open_interest_change_pct_24h
                        if analysis.snapshot is not None
                        else None
                    ),
                    "payload": {
                        "bias": analysis.bias,
                        "direction": analysis.direction,
                        "confidence": analysis.confidence,
                        "legacy_confidence": analysis.meta.get(
                            "legacy_confidence",
                            analysis.confidence,
                        ),
                        "legacy_v2_confidence": analysis.meta.get(
                            "legacy_v2_confidence",
                            analysis.confidence,
                        ),
                        "legacy_confidence_comparison": analysis.meta.get(
                            "legacy_confidence_comparison"
                        )
                        or {},
                        "scoring_policy": (
                            "legacy_v2"
                            if analysis.meta.get("legacy_v2_enabled", False)
                            else "legacy"
                        ),
                        "technical_confidence": analysis.technical_confidence,
                        "llm_confidence": analysis.llm_confidence,
                        "llm_confidence_reason": analysis.llm_confidence_reason,
                        "llm_confidence_detail": getattr(analysis, "llm_confidence_detail", {}) or {},
                        "rank_score": scan_rank_score,
                        "authoritative_rank": live_rank_score,
                        "rank_available": rank_available,
                        "live_rank_score": live_rank_score,
                        "backtest": backtest_summary,
                        "data_quality": analysis.meta.get("data_quality") or {},
                        "execution": execution,
                        "chart": chart,
                        "setup_name": analysis.setup_name,
                        "confluence_total": analysis.confluence_total,
                        "key_levels": analysis.key_levels[:4],
                        "key_reasons": analysis.key_reasons[:3],
                        "trade_plan": primary,
                        "primary_setup": primary,
                        "prop_safe": prop_safe,
                        "prop_flags": prop_flags,
                        "prop_guidance": prop_guidance,
                        "universal_eligible": bool(
                            analysis.meta.get("universal_eligible", False)
                        ),
                        "production_qualified": bool(
                            analysis.meta.get("production_qualified", False)
                        ),
                        "legacy_signal_eligible": analysis.meta.get(
                            "legacy_signal_eligible",
                            False,
                        ),
                        "legacy_v2_signal_eligible": analysis.meta.get(
                            "legacy_v2_signal_eligible",
                            False,
                        ),
                        "rejection_reasons": list(
                            analysis.meta.get("rejection_reasons") or []
                        ),
                        "position_simulation": (
                            plan.to_position_simulation() if plan else None
                        ),
                    },
                }
                candidate = build_candidate_record(
                    analysis,
                    row,
                    source="watchlist",
                    feature_schema_version=cfg.outcome_scoring.feature_schema_version,
                )
                shadow_scores = score_candidate_shadow(candidate, cfg)
                if shadow_scores is not None:
                    candidate["shadow_model_version"] = shadow_scores.get(
                        "model_version"
                    )
                    candidate["shadow_scores"] = shadow_scores
                else:
                    candidate["shadow_model_version"] = None
                    candidate["shadow_scores"] = None
                row["candidate_id"] = candidate["id"]
                row["outcome_scoring"] = shadow_scores or {
                    "status": "collecting_data",
                    "mode": cfg.outcome_scoring.mode,
                }
                row["scoring_source"] = "legacy_production"
                if (
                    cfg.outcome_scoring.mode == "production"
                    and shadow_scores
                    and shadow_scores.get("calibration_ready")
                ):
                    legacy_scores = {
                        "technical_confidence": row["technical_confidence"],
                        "execution_score": row["execution_score"],
                        "confidence": row["confidence"],
                        "rank_score": row["rank_score"],
                    }
                    row["legacy_scores"] = legacy_scores
                    row["technical_confidence"] = round(
                        float(shadow_scores.get("technical_score") or 0), 1
                    )
                    row["execution_score"] = round(
                        float(shadow_scores.get("execution_score") or 0), 1
                    )
                    row["confidence"] = round(
                        float(shadow_scores.get("confidence") or 0), 1
                    )
                    row["rank_score"] = round(
                        float(shadow_scores.get("rank_score") or 0), 2
                    )
                    outcome_gate = bool(
                        row["confidence"]
                        >= float(cfg.outcome_scoring.confidence_floor)
                        and float(
                            shadow_scores.get("conservative_ev_r") or 0
                        )
                        > 0
                        and bool(shadow_scores.get("model_applicable", False))
                    )
                    # Initial champion deployment is veto-only: it may reject or
                    # reorder a legacy-eligible plan, never resurrect a setup
                    # whose deterministic safety gates already blocked it.
                    row["signal_eligible"] = bool(
                        row.get("signal_eligible") and outcome_gate
                    )
                    row["scoring_source"] = "outcome_champion_veto"
                    row["payload"]["confidence"] = row["confidence"]
                    row["payload"]["technical_confidence"] = row[
                        "technical_confidence"
                    ]
                    row["payload"]["rank_score"] = row["rank_score"]
                    row["payload"]["execution"]["score"] = row[
                        "execution_score"
                    ]
                row["payload"]["candidate_id"] = candidate["id"]
                row["payload"]["outcome_scoring"] = row["outcome_scoring"]
                if str(row.get("llm_invocation_status") or "").startswith(
                    ("completed:", "fallback:")
                ):
                    llm_call_count += 1
                llm_rate_limit_events += int(
                    row.get("llm_rate_limit_events") or 0
                )
                alert_evaluation = evaluate_alert_gates(
                    row,
                    min_rank=float(cfg.telegram.min_rank_score or 50.0),
                    only_prop_safe=bool(cfg.scheduler.only_prop_safe),
                    min_confidence=max(
                        ALERT_MIN_OVERALL_QUALITY,
                        float(
                            getattr(
                                cfg.analysis,
                                "directional_confidence_threshold",
                                68.0,
                            )
                        ),
                    ),
                    min_execution_quality=float(
                        getattr(cfg.analysis, "execution_min_score", 72.0)
                    ),
                    max_immediate_sl_risk=float(
                        getattr(cfg.analysis, "max_immediate_sl_risk", 32.0)
                    ),
                    max_chase_distance_atr=float(
                        getattr(cfg.analysis, "max_chase_distance_atr", 1.0)
                    ),
                    max_pre_entry_tp1_progress_pct=float(
                        getattr(
                            cfg.analysis,
                            "max_pre_entry_tp1_progress_pct",
                            70.0,
                        )
                    ),
                    min_tp2_rr=float(
                        getattr(cfg.analysis, "min_tp2_rr", 1.25)
                    ),
                    max_spread_bps=float(
                        getattr(cfg.analysis, "max_spread_bps", 12.0)
                    ),
                    max_ticker_age_seconds=float(
                        getattr(cfg.analysis, "max_ticker_age_seconds", 45.0)
                    ),
                    max_orderbook_age_seconds=float(
                        getattr(cfg.analysis, "max_orderbook_age_seconds", 30.0)
                    ),
                    prior=row.get("gate_evaluation"),
                )
                row["gate_evaluation"] = alert_evaluation.to_dict()
                row["gate_evaluation"]["authoritative_rank"] = row.get(
                    "authoritative_rank"
                )
                row["gate_evaluation"]["rank_available"] = bool(
                    row.get("rank_available")
                )
                row["universal_eligible"] = bool(
                    alert_evaluation.universal_eligible
                )
                # A promoted outcome model remains veto-only. It cannot make a
                # deterministically rejected candidate production-qualified.
                row["production_qualified"] = bool(
                    alert_evaluation.production_qualified
                    and row.get("signal_eligible") is not False
                )
                row["signal_eligible"] = row["production_qualified"]
                row["payload"]["gate_evaluation"] = row["gate_evaluation"]
                row["payload"]["universal_eligible"] = row[
                    "universal_eligible"
                ]
                row["payload"]["production_qualified"] = row[
                    "production_qualified"
                ]
                row["payload"]["signal_eligible"] = row["signal_eligible"]
                candidate["decision"].update(
                    {
                        "prop_safe": prop_safe,
                        "prop_guidance": prop_guidance,
                        "universal_eligible": row["universal_eligible"],
                        "production_qualified": row["production_qualified"],
                        "signal_eligible": row["signal_eligible"],
                        "gate_evaluation": row["gate_evaluation"],
                    }
                )
                candidate["production_scores"].update(
                    {
                        "prop_safe": prop_safe,
                        "prop_guidance": prop_guidance,
                        "universal_eligible": row["universal_eligible"],
                        "production_qualified": row["production_qualified"],
                    }
                )
                candidate["production_eligible"] = row[
                    "production_qualified"
                ]
                qualification = evaluate_private_beta_qualification(
                    row, row.get("gate_evaluation")
                )
                row["qualification"] = qualification
                row["qualification_policy_version"] = qualification[
                    "qualification_policy_version"
                ]
                row["gate_evaluation"]["private_beta_qualification"] = qualification
                row["payload"]["qualification"] = qualification
                candidate["decision"]["qualification"] = qualification
                candidate["decision"]["qualification_policy_version"] = qualification[
                    "qualification_policy_version"
                ]
                candidate["production_scores"]["private_beta_qualification"] = {
                    "qualification_type": qualification["qualification_type"],
                    "hard_pass_percentage": qualification["hard_pass_percentage"],
                    "soft_pass_percentage": qualification["soft_pass_percentage"],
                    "important_soft_pass_count": qualification[
                        "important_soft_pass_count"
                    ],
                    "authoritative_rank": qualification["authoritative_rank"],
                    "rank_available": qualification["rank_available"],
                    "net_rr": qualification["net_rr"],
                    "private_beta_net_rr_floor": qualification[
                        "private_beta_net_rr_floor"
                    ],
                }
                evaluated_direction = str(
                    row.get("evaluated_direction") or ""
                ).lower()
                beta_row = _build_directional_comparison_row(row)

                if is_legacy_comparison():
                    # Strict analysis may flatten a directional thesis after
                    # publication-selectivity gates fail. Compare both policies
                    # against the preserved directional row, not that display
                    # flattening, so the A/B journal records the real experiment.
                    comparison_row = beta_row or row
                    legacy_decision = evaluate_legacy_qualification(comparison_row)
                    shadow_decision = strict_shadow_decision(comparison_row)
                    row.update(identity_metadata())
                    row["strict_production_qualified"] = bool(
                        row.get("production_qualified")
                    )
                    row["legacy_decision"] = legacy_decision
                    row["strict_shadow_decision"] = shadow_decision
                    row["legacy_comparison_qualified"] = bool(
                        legacy_decision.get("qualified")
                    )
                    row["quality_tier"] = legacy_decision.get("quality_tier")
                    row["caveats"] = list(legacy_decision.get("caveats") or [])
                    row["legacy_qualification_policy_version"] = (
                        LEGACY_POLICY_VERSION
                    )
                    row["payload"].update(identity_metadata())
                    row["payload"]["legacy_decision"] = legacy_decision
                    row["payload"]["strict_shadow_decision"] = shadow_decision
                    row["payload"]["quality_tier"] = row["quality_tier"]
                    row["payload"]["caveats"] = row["caveats"]
                    if beta_row is not None:
                        beta_row.update(identity_metadata())
                        beta_row["strict_production_qualified"] = row[
                            "strict_production_qualified"
                        ]
                        beta_row["legacy_decision"] = legacy_decision
                        beta_row["strict_shadow_decision"] = shadow_decision
                        beta_row["legacy_comparison_qualified"] = bool(
                            legacy_decision.get("qualified")
                        )
                        beta_row["quality_tier"] = legacy_decision.get(
                            "quality_tier"
                        )
                        beta_row["caveats"] = list(
                            legacy_decision.get("caveats") or []
                        )
                        beta_row["legacy_qualification_policy_version"] = (
                            LEGACY_POLICY_VERSION
                        )
                        beta_row["payload"].update(identity_metadata())
                        beta_row["payload"]["legacy_decision"] = legacy_decision
                        beta_row["payload"]["strict_shadow_decision"] = (
                            shadow_decision
                        )
                        beta_row["payload"]["quality_tier"] = beta_row[
                            "quality_tier"
                        ]
                        beta_row["payload"]["caveats"] = beta_row["caveats"]
                    candidate["decision"].update(identity_metadata())
                    candidate["decision"]["legacy_decision"] = legacy_decision
                    candidate["decision"]["strict_shadow_decision"] = (
                        shadow_decision
                    )
                    candidate["decision"]["strict_production_qualified"] = (
                        row["strict_production_qualified"]
                    )
                    candidate["production_scores"].update(identity_metadata())
                    candidate["production_scores"]["legacy_decision"] = {
                        "qualified": bool(legacy_decision.get("qualified")),
                        "quality_tier": legacy_decision.get("quality_tier"),
                        "overall_quality": legacy_decision.get("overall_quality"),
                        "execution_quality": legacy_decision.get("execution_quality"),
                        "net_rr": legacy_decision.get("net_rr"),
                        "gross_rr": legacy_decision.get("gross_rr"),
                        "caveats": list(legacy_decision.get("caveats") or []),
                        "policy_version": LEGACY_POLICY_VERSION,
                    }
                    candidate["production_scores"]["strict_shadow_decision"] = (
                        shadow_decision
                    )
                    candidate["production_eligible"] = bool(
                        legacy_decision.get("qualified")
                    )
                    # Legacy shares the candidate journal for A/B reporting,
                    # but its rows must never enter Strict model training.
                    candidate["decision"]["comparison_directional_candidate"] = (
                        evaluated_direction in {"long", "short"}
                    )
                    candidate["is_directional_candidate"] = False
                if beta_row is not None:
                    qualification_candidates.append(beta_row)
                analytics_candidates.append(
                    candidate_analytics_snapshot(
                        row,
                        row["gate_evaluation"],
                        scan_id=resolved_scan_id,
                        candidate_id=candidate["id"],
                        analyzed_at=analysis.generated_at,
                    )
                )
                journal_records.append(candidate)
                analyzed_count += 1

                direction = (analysis.direction or "flat").lower()
                if direction not in ("long", "short"):
                    # Flat / neutral: do not rank on the leaderboard
                    skipped_flat.append(
                        {
                            "symbol": normalized_symbol,
                            "direction": analysis.direction,
                            "bias": analysis.bias,
                            "llm_confidence": row["llm_confidence"],
                            "technical_confidence": row["technical_confidence"],
                            "legacy_confidence": row["legacy_confidence"],
                            "legacy_v2_confidence": row[
                                "legacy_v2_confidence"
                            ],
                            "execution_score": row["execution_score"],
                            "confluence_score": row["confluence_score"],
                            "rejection_reasons": row["rejection_reasons"],
                            "candidate_id": candidate["id"],
                            "gate_evaluation": row["gate_evaluation"],
                            "reason": analysis.llm_confidence_reason
                            or "Flat/neutral — not ranked",
                        }
                    )
                    logger.info(
                        "Scan deprioritize {}: direction={} llm={:.0f}%",
                        normalized_symbol,
                        analysis.direction,
                        analysis.llm_confidence,
                    )
                    continue

                ranked_results.append(row)
            finally:
                client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scan failed for {}: {}", symbol, exc)
            analysis_failures.append(
                {
                    "symbol": str(symbol),
                    "code": "ANALYSIS_ERROR",
                    "reason": type(exc).__name__,
                }
            )

    journaled_count = journal_scan_candidates(journal_records, cfg)

    scan_completed_at = datetime.now(timezone.utc)
    rejection_summary = aggregate_rejection_rows([], analytics_candidates)
    for diagnostic in rejection_summary.get("suspicious_diagnostics") or []:
        logger.warning(
            "Protected rejection diagnostic: scan={} code={} gate={} sample={}",
            resolved_scan_id,
            diagnostic.get("code"),
            diagnostic.get("gate") or "n/a",
            diagnostic.get("sample") or 0,
        )
    build_identity = get_build_identity()
    analytics_repository = get_rejection_repository(cfg)
    scan_record = {
        "scan_id": resolved_scan_id,
        "trigger_type": (
            trigger_type
            if trigger_type in {"scheduled", "manual", "telegram", "api"}
            else "manual"
        ),
        "started_at": scan_started_at.isoformat(),
        "completed_at": scan_completed_at.isoformat(),
        "requested_symbols": list(symbol_list),
        "analyzed_symbols": [
            str(row.get("symbol") or "") for row in analytics_candidates
        ],
        "failed_symbols": list(analysis_failures),
        "directional_candidates": int(
            rejection_summary.get("directional_candidates") or 0
        ),
        "eligible_candidates": int(
            rejection_summary.get("eligible_candidates") or 0
        ),
        "revalidated_candidates": 0,
        "scan_duration_seconds": round(
            time.monotonic() - scan_started_monotonic, 3
        ),
        "analytics_latency_ms": 0.0,
        "llm_calls": llm_call_count,
        "llm_rate_limit_events": llm_rate_limit_events,
        "public_messages": 0,
        "private_messages": 0,
        "no_quality_result": not bool(
            rejection_summary.get("eligible_candidates")
        ),
        "status": (
            "completed"
            if analyzed_count and not analysis_failures
            else ("partial" if analyzed_count else "failed")
        ),
        "gate_policy_version": GATE_POLICY_VERSION,
        "feature_schema_version": cfg.outcome_scoring.feature_schema_version,
        "execution_policy_version": cfg.analysis.execution_policy_version,
        "rank_policy_version": str(
            build_identity.get("rank_policy") or "deterministic_rank_v2a.1"
        ),
        "build_commit_sha": build_identity.get("git_commit_sha"),
        "summary": rejection_summary,
    }
    analytics_persisted = analytics_repository.record_scan(
        scan_record, analytics_candidates
    )
    analytics_latency_ms = analytics_repository.last_latency_ms
    if not analytics_persisted and analytics_repository.enabled:
        logger.error(
            "Rejection analytics unavailable for scan={} error_type={}; "
            "signal eligibility remains unchanged",
            resolved_scan_id,
            analytics_repository.last_error or "unknown",
        )

    # Rank directional signals deterministically. LLM numbers are never a tie
    # breaker because that would still grant them delivery authority.
    ranked_results.sort(
        key=lambda item: (
            float(item.get("rank_score") or 0),
            abs(float(item.get("confluence_score") or 0)),
        ),
        reverse=True,
    )
    return {
        "ok": analyzed_count > 0,
        "scan_id": resolved_scan_id,
        "error": None if analyzed_count > 0 else "scan_analysis_unavailable",
        "ranked_results": ranked_results[:10],
        "qualification_candidates": qualification_candidates,
        "skipped_flat": skipped_flat[:20],
        "count": len(ranked_results),
        "flat_count": len(skipped_flat),
        "analyzed_count": analyzed_count,
        "analysis_failures": analysis_failures,
        "timeframe": primary_tf,
        "exchange": ex_id,
        "leverage_display_cap": SCAN_LEVERAGE_CAP,
        "ranking": (
            "Deterministic closed-candle confluence/execution with bounded "
            "robust-sample backtest adjustment; outcome model remains shadow-only "
            "until calibrated and explicitly promoted"
        ),
        "prop_mode": bool(getattr(cfg.risk, "prop_mode", True)),
        "candidate_journaled_count": journaled_count,
        "outcome_scoring": get_outcome_scoring_status(cfg),
        "rejection_analytics": rejection_summary,
        # Internal consumers use these already-computed snapshots to append
        # revalidation/delivery stages. API serialization is protected by the
        # existing scan authorization layer.
        "rejection_candidates": analytics_candidates,
        "rejection_analytics_persisted": bool(analytics_persisted),
        "analytics_latency_ms": analytics_latency_ms,
        "llm_calls": llm_call_count,
        "llm_rate_limit_events": llm_rate_limit_events,
    }


def run_symbol_backtest(
    symbol: str,
    *,
    timeframe: Optional[str] = None,
    bars: int = 500,
    exchange: Optional[str] = None,
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """Prop-oriented historical backtest wrapper for API / Streamlit / CLI."""
    from src.analysis.backtest import run_backtest

    cfg = config or load_config()
    try:
        result = run_backtest(
            symbol,
            timeframe=timeframe or cfg.timeframes.primary,
            bars=bars,
            config=cfg,
            exchange=exchange or cfg.exchange.default,
        )
        out = result.to_dict()
        out["ok"] = True
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Backtest failed for {}: {}", symbol, exc)
        return {"ok": False, "error": str(exc), "symbol": symbol}


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
