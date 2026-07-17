#!/usr/bin/env python3
"""
perpetual_pro FastAPI server.

Run:
    uvicorn main_server:app --reload --port 8000

Or:
    python main_server.py
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root on path when launched as script / uvicorn module
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src import __version__
from src.api.service import AnalyzeRequest, analyze_from_image
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
    yield


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
    }


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
        description="Comma-separated higher TFs, e.g. 1h,4h,1d",
    ),
    balance: Optional[float] = Form(None, description="Account balance for position sizing"),
    risk: Optional[float] = Form(None, description="Risk percent per trade"),
    no_news: bool = Form(False, description="Skip news fetch"),
    dark_theme: Optional[bool] = Form(
        None,
        description="Chart is dark theme (default from config)",
    ),
) -> JSONResponse:
    """
    Analyze an uploaded chart screenshot.

    Pipeline: OCR + computer vision → live OHLCV/derivatives → indicators →
    patterns → market structure → news → weighted confluence → trade plan.
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

    req = AnalyzeRequest(
        symbol=symbol.strip() if symbol else None,
        timeframe=timeframe.strip() if timeframe else None,
        exchange=exchange.strip().lower() if exchange else None,
        higher=higher_list,
        account_balance=balance,
        risk_pct=risk,
        no_news=bool(no_news),
        dark_theme=dark_theme,
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
