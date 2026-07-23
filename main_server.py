#!/usr/bin/env python3
"""
perpetual_pro FastAPI server.

Run:
    uvicorn main_server:app --reload --port 8000

Or:
    python main_server.py
"""

from __future__ import annotations

import hmac
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root on path when launched as script / uvicorn module
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src import __version__
from src.api.service import AnalyzeRequest, analyze_from_image, scan_symbols
from src.notify.telegram import (
    get_telegram_credentials,
    is_telegram_ready,
    send_test_telegram_alert,
)
from src.scheduler.scan_job import (
    get_scheduler_status,
    run_scheduled_scan_once,
    start_scheduler_background,
    stop_scheduler_background,
)
from src.utils.config import load_config, setup_logging

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_config = None


def get_config():
    global _config
    if _config is None:
        _config = load_config()
        setup_logging(_config)
    return _config


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    logger.info("perpetual_pro API v{} starting (exchange={})", __version__, cfg.exchange.default)
    scheduler_started = start_scheduler_background(cfg)
    logger.info(
        "Scheduler startup result: enabled={} started={} times={} timezone={}",
        cfg.scheduler.enabled,
        scheduler_started,
        cfg.scheduler.times,
        cfg.scheduler.timezone,
    )
    try:
        yield
    finally:
        stop_scheduler_background()
        logger.info("perpetual_pro API shutdown complete")


app = FastAPI(
    title="perpetual_pro",
    description=(
        "Professional crypto perpetual futures analysis API. "
        "Upload a chart screenshot for OCR + full pro analysis "
        "(live data, indicators, patterns, news, confluence)."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "app": "perpetual_pro",
        "version": __version__,
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze (multipart: image + optional symbol/timeframe)",
            "telegram_status": "GET /telegram/status",
            "telegram_test": "POST /telegram/test (X-Telegram-Test-Key required)",
            "telegram_test_scan": (
                "POST /telegram/test-scan (run watchlist now; "
                "X-Telegram-Test-Key required)"
            ),
        },
        "disclaimer": "Not financial advice. For research/education only.",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    cfg = get_config()
    return {
        "status": "ok",
        "version": __version__,
        "default_exchange": cfg.exchange.default,
        "default_timeframe": cfg.timeframes.primary,
        "telegram_ready": is_telegram_ready(cfg),
        "scheduler": get_scheduler_status(),
    }


@app.get("/telegram/status")
def telegram_status() -> Dict[str, Any]:
    """Redacted configuration and scheduler state; does not call Telegram."""
    cfg = get_config()
    token, chat = get_telegram_credentials()
    return {
        "ok": True,
        "telegram": {
            "enabled": bool(cfg.telegram.enabled),
            "ready": is_telegram_ready(cfg),
            "token_configured": bool(token),
            "chat_id_configured": bool(chat),
            "chat_id_suffix": chat[-4:] if chat else None,
            "test_endpoint_secured": bool(
                (os.getenv("TELEGRAM_TEST_KEY") or "").strip()
            ),
        },
        "scheduler": get_scheduler_status(),
    }


def _authorize_telegram_test(provided_key: Optional[str]) -> None:
    expected = (os.getenv("TELEGRAM_TEST_KEY") or "").strip()
    if not expected:
        logger.error(
            "Telegram test endpoint blocked: TELEGRAM_TEST_KEY is not configured"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Set TELEGRAM_TEST_KEY in the API environment, then send it as "
                "X-Telegram-Test-Key"
            ),
        )
    if not provided_key or not hmac.compare_digest(provided_key, expected):
        logger.warning("Telegram test endpoint rejected an invalid admin key")
        raise HTTPException(status_code=403, detail="Invalid Telegram test key")


@app.post("/telegram/test")
def telegram_test(
    x_telegram_test_key: Optional[str] = Header(
        None,
        alias="X-Telegram-Test-Key",
        description="Must match the TELEGRAM_TEST_KEY environment variable",
    ),
) -> JSONResponse:
    """Run live bot/chat permission checks and send one fixed test alert."""
    _authorize_telegram_test(x_telegram_test_key)
    logger.info("Manual Telegram test requested via API")
    result = send_test_telegram_alert(source="FastAPI /telegram/test")
    if result.get("ok"):
        logger.info("Manual Telegram test succeeded via API")
        return JSONResponse(status_code=200, content=result)
    delivery = result.get("delivery") or {}
    diagnostics = result.get("diagnostics") or {}
    logger.error(
        "Manual Telegram test failed via API: delivery_error={} diagnostics_error={}",
        delivery.get("error"),
        diagnostics.get("error"),
    )
    return JSONResponse(status_code=502, content=result)


@app.post("/telegram/test-scan")
def telegram_test_scan(
    x_telegram_test_key: Optional[str] = Header(
        None,
        alias="X-Telegram-Test-Key",
        description="Must match the TELEGRAM_TEST_KEY environment variable",
    ),
) -> JSONResponse:
    """Run the real scheduled watchlist workflow now and send its alert."""
    _authorize_telegram_test(x_telegram_test_key)
    cfg = get_config()
    logger.info("Manual Telegram scheduled-scan test requested via API")
    try:
        result = run_scheduled_scan_once(
            cfg,
            slot_label="Manual test scan",
            send=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual Telegram scheduled-scan test failed: {}", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Manual scheduled scan failed: {type(exc).__name__}",
        ) from exc

    compact = {
        "ok": bool(result.get("ok") and result.get("telegram_sent")),
        "scan_ok": bool(result.get("ok")),
        "telegram_sent": bool(result.get("telegram_sent")),
        "telegram_ready": bool(result.get("telegram_ready")),
        "delivery_status": result.get("telegram_delivery_status"),
        "delivery": result.get("telegram_delivery"),
        "scanned": result.get("scanned"),
        "ranked_count": result.get("ranked_count"),
        "alert_count": result.get("alert_count"),
        "started_at": result.get("started_at"),
        "completed_at": result.get("completed_at"),
        "slot_label": result.get("slot_label"),
    }
    if compact["ok"]:
        logger.info(
            "Manual Telegram scheduled-scan test succeeded: actionable={} status={}",
            compact["alert_count"],
            compact["delivery_status"],
        )
        return JSONResponse(status_code=200, content=compact)
    logger.error(
        "Manual Telegram scheduled-scan test did not deliver: scan_ok={} status={}",
        compact["scan_ok"],
        compact["delivery_status"],
    )
    return JSONResponse(status_code=502, content=compact)


@app.post("/scan")
async def scan(
    symbols: Optional[str] = Form(
        None,
        description="Comma-separated symbols to scan, e.g. BTC,ETH,SOL",
    ),
    timeframe: Optional[str] = Form(None, description="Primary timeframe, e.g. 15m"),
    exchange: Optional[str] = Form(None, description="binanceusdm | bybit | okx | bitget"),
    no_news: bool = Form(False, description="Skip news fetch"),
    simulated_capital: Optional[float] = Form(None, description="Simulated capital"),
    risk: Optional[float] = Form(None, description="Risk percent"),
) -> JSONResponse:
    cfg = get_config()
    symbol_list = [x.strip() for x in (symbols or "").split(",") if x and x.strip()]
    if not symbol_list:
        return JSONResponse(status_code=422, content={"ok": False, "error": "no_symbols"})

    req = AnalyzeRequest(
        timeframe=timeframe.strip() if timeframe else None,
        exchange=exchange.strip().lower() if exchange else None,
        simulated_capital=simulated_capital,
        risk_pct=risk,
        no_news=bool(no_news),
    )
    try:
        result = scan_symbols(symbol_list, request=req, config=cfg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scan failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Scan failed: {exc}") from exc
    return JSONResponse(content=result)


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(..., description="Chart screenshot (PNG/JPEG/WebP)"),
    symbol: Optional[str] = Form(
        None,
        description="Override symbol, e.g. BTC or BTC/USDT:USDT. If omitted, OCR tries to detect it.",
    ),
    timeframe: Optional[str] = Form(
        None,
        description="Primary timeframe, e.g. 15m. If omitted, OCR/config default.",
    ),
    exchange: Optional[str] = Form(
        None,
        description="binanceusdm | bybit | okx | bitget",
    ),
    higher: Optional[str] = Form(
        None,
        description="Comma-separated higher TFs, e.g. 5m,1h,4h,1d",
    ),
    simulated_capital: Optional[float] = Form(
        None,
        description="Simulated capital for position sizing (default $1000). Not a live balance.",
    ),
    balance: Optional[float] = Form(
        None,
        description="Legacy alias for simulated_capital",
    ),
    risk: Optional[float] = Form(
        None,
        description="Risk percent of simulated capital per trade (default 1.0)",
    ),
    no_news: bool = Form(False, description="Skip news fetch"),
    no_llm: bool = Form(False, description="Skip Groq/Gemini narrative layer"),
    dark_theme: Optional[bool] = Form(
        None,
        description="Chart is dark theme (default from config)",
    ),
    page_url: Optional[str] = Form(
        None,
        description="Active tab URL (TradingView preferred) for symbol/timeframe extraction",
    ),
    client_ocr: Optional[str] = Form(
        None,
        description="JSON string of client-side OCR results (Tesseract.js)",
    ),
    client_vision: Optional[str] = Form(
        None,
        description="JSON string of client-side light vision results",
    ),
    client_hints: Optional[str] = Form(
        None,
        description="JSON string of fused client hints",
    ),
) -> JSONResponse:
    """
    Maximum-signal chart analysis.

    Pipeline: OCR + CV → multi-TF OHLCV + funding/OI/L-S → full indicator suite →
    patterns + market structure → news → weighted confluence → dynamic leverage
    simulation → optional LLM narrative.
    """
    cfg = get_config()

    # Validate content type lightly (some clients send octet-stream)
    content_type = (image.content_type or "").lower()
    allowed = ("image/", "application/octet-stream", "")
    if content_type and not any(content_type.startswith(a) or content_type == a for a in allowed):
        if content_type not in ("application/octet-stream",):
            # Still try if filename looks like an image
            name = (image.filename or "").lower()
            if not name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")):
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected an image upload, got content-type={content_type!r}",
                )

    try:
        raw = await image.read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read upload: {exc}") from exc

    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")

    # Soft size guard (25 MB)
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 25 MB)")

    higher_list: Optional[List[str]] = None
    if higher:
        higher_list = [x.strip() for x in higher.split(",") if x.strip()]

    sim_cap = simulated_capital if simulated_capital is not None else balance

    def _parse_json_field(raw: Optional[str]) -> dict:
        if not raw:
            return {}
        try:
            import json

            val = json.loads(raw)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}

    req = AnalyzeRequest(
        symbol=symbol.strip() if symbol else None,
        timeframe=timeframe.strip() if timeframe else None,
        exchange=exchange.strip().lower() if exchange else None,
        higher=higher_list,
        simulated_capital=sim_cap,
        risk_pct=risk,
        no_news=bool(no_news),
        dark_theme=dark_theme,
        use_llm=not bool(no_llm),
        page_url=page_url.strip() if page_url else None,
        client_ocr=_parse_json_field(client_ocr),
        client_vision=_parse_json_field(client_vision),
        client_hints=_parse_json_field(client_hints),
    )

    try:
        result = analyze_from_image(raw, request=req, config=cfg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analyze failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    if not result.get("ok", True) and result.get("error") in ("symbol_required", "invalid_symbol"):
        # Client fixable — return 422 with structured body
        return JSONResponse(status_code=422, content=result)

    return JSONResponse(content=result)


def main() -> None:
    """Run with: python main_server.py"""
    import uvicorn

    uvicorn.run(
        "main_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
