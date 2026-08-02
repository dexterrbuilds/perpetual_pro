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
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root on path when launched as script / uvicorn module
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src import __version__
from src.analytics.rejection import GATE_POLICY_VERSION
from src.analytics.runtime import get_rejection_repository
from src.api.service import AnalyzeRequest, analyze_from_image, scan_symbols
from src.api.security import (
    SCAN_ACCESS,
    SCAN_TIMEOUT_SECONDS,
    normalize_requested_symbols,
    validate_exchange,
    validate_timeframe,
)
from src.notify.telegram import (
    get_delivery_status,
    get_telegram_alert_chat_ids,
    get_telegram_credentials,
    is_telegram_ready,
    send_test_telegram_alert,
)
from src.notify.telegram_bot import (
    configure_telegram_webhook,
    get_telegram_command_chat_ids,
    get_telegram_webhook_status,
    process_telegram_update,
    telegram_webhook_secret,
)
from src.scheduler.scan_job import (
    get_scheduler_status,
    run_scheduled_scan_once,
    start_scheduler_background,
    stop_scheduler_background,
)
from src.scoring.runtime import (
    get_outcome_scoring_runtime,
    get_outcome_scoring_status,
)
from src.tracking.signal_tracker import (
    get_signal_reliability_summary,
    get_signal_tracker_status,
    start_signal_tracker_background,
    stop_signal_tracker_background,
)
from src.utils.config import load_config, setup_logging
from src.utils.build_info import get_build_identity

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
    identity = get_build_identity()
    logger.info(
        "perpetual_pro API v{} starting (exchange={} commit={} build={} "
        "feature_schema={} execution_policy={} rank_policy={} environment={})",
        __version__, cfg.exchange.default,
        str(identity["git_commit_sha"])[:12], identity["build_timestamp"],
        identity["feature_schema"], identity["execution_policy"],
        identity["rank_policy"], identity["environment"],
    )
    delivery = get_delivery_status()
    logger.info(
        "Delivery Mode: {} beta_recipients={} public_enabled={} "
        "active_signal_recipients={}",
        delivery["mode_label"],
        delivery["beta_recipient_count"],
        delivery["public_delivery_enabled"],
        delivery["active_signal_recipient_count"],
    )
    scoring_runtime = get_outcome_scoring_runtime(cfg)
    scoring_database_ready = scoring_runtime.repository.check_ready()
    tracker_started = start_signal_tracker_background(cfg)
    scheduler_started = start_scheduler_background(cfg)
    webhook_started = configure_telegram_webhook(cfg)
    logger.info(
        "Background services: tracker_started={} scheduler_started={} webhook_started={} "
        "outcome_database_ready={} scoring_mode={} "
        "scheduler_enabled={} active_windows={} timezone={}",
        tracker_started,
        scheduler_started,
        webhook_started,
        scoring_database_ready,
        cfg.outcome_scoring.mode,
        cfg.scheduler.enabled,
        (cfg.scheduler.sessions or cfg.scheduler.times),
        cfg.scheduler.timezone,
    )
    try:
        yield
    finally:
        stop_scheduler_background()
        stop_signal_tracker_background()
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
            "readiness": "GET /ready",
            "analyze": "POST /analyze (multipart: image + optional symbol/timeframe)",
            "telegram_status": "GET /telegram/status",
            "telegram_test": "POST /telegram/test (X-Telegram-Test-Key required)",
            "telegram_test_scan": (
                "POST /telegram/test-scan (run watchlist now; "
                "X-Telegram-Test-Key required)"
            ),
            "telegram_commands": "/scan · /status · /help",
            "signal_reliability": "GET /signal-tracker/reliability",
            "outcome_scoring": "GET /outcome-scoring/status",
        },
        "disclaimer": "Not financial advice. High leverage perps can liquidate quickly.",
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
        "signal_tracker": get_signal_tracker_status(),
        "outcome_scoring": get_outcome_scoring_status(cfg),
    }


@app.get("/ready")
def readiness() -> JSONResponse:
    """Dependency-aware readiness; never sends messages or places orders."""
    cfg = get_config()
    tracker = get_signal_tracker_status()
    scoring_runtime = get_outcome_scoring_runtime(cfg)
    supabase_ready = scoring_runtime.repository.check_ready()
    scheduler = get_scheduler_status()
    identity = get_build_identity()
    production = bool(os.getenv("RAILWAY_ENVIRONMENT_NAME"))
    checks = {
        "configuration_loaded": True,
        "supabase_reachable": bool(supabase_ready),
        "lifecycle_repository_available": bool(
            tracker.get("durable_lifecycle_ready")
        ),
        "startup_recovery_completed": bool(tracker.get("recovery_completed")),
        "tracker_running": bool(
            tracker.get("running") and tracker.get("thread_alive")
        ),
        "scheduler_state_known": "enabled" in scheduler,
        "telegram_configuration_valid": bool(
            not cfg.telegram.enabled or is_telegram_ready(cfg)
        ),
        "exchange_market_data_capability": cfg.exchange.default
        in {"okx", "bybit", "bitget", "binanceusdm"},
        "policy_versions_compatible": bool(
            identity["feature_schema"] == cfg.outcome_scoring.feature_schema_version
            and identity["execution_policy"] == cfg.analysis.execution_policy_version
        ),
        "scan_auth_configured": bool(
            SCAN_ACCESS.configured_key() or not production
        ),
        "deployment_identity_complete": bool(
            identity["identity_complete"] or not production
        ),
        "background_workers_healthy": not bool(
            tracker.get("recovery_failed")
        ),
    }
    ready = all(checks.values())
    payload = {
        "status": "ready" if ready else "not_ready",
        "guarded_mode": bool(not cfg.scheduler.enabled),
        "scheduler": {
            "enabled": bool(cfg.scheduler.enabled),
            "state": "disabled_intentionally" if not cfg.scheduler.enabled else "enabled",
            "active_windows": cfg.scheduler.sessions or cfg.scheduler.times,
        },
        "checks": checks,
        "build": identity,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.get("/admin/status")
def admin_status(
    request: Request,
    x_scan_api_key: Optional[str] = Header(None, alias="X-Scan-API-Key"),
) -> Dict[str, Any]:
    SCAN_ACCESS.authorize(request, x_scan_api_key)
    rejection_repository = get_rejection_repository(get_config())
    return {
        "ok": True,
        "build": get_build_identity(),
        "delivery": get_delivery_status(),
        "scheduler": get_scheduler_status(),
        "signal_tracker": get_signal_tracker_status(),
        "outcome_scoring": get_outcome_scoring_status(get_config()),
        "rejection_analytics": {
            "ready": rejection_repository.check_ready(),
            "gate_policy_version": GATE_POLICY_VERSION,
            "last_error": rejection_repository.last_error,
            "last_latency_ms": rejection_repository.last_latency_ms,
        },
    }


@app.get("/admin/rejections/latest")
def latest_rejection_analytics(
    request: Request,
    x_scan_api_key: Optional[str] = Header(None, alias="X-Scan-API-Key"),
) -> JSONResponse:
    """Latest completed scan diagnostics; protected like the scan endpoint."""
    SCAN_ACCESS.authorize(request, x_scan_api_key)
    result = get_rejection_repository(get_config()).latest_scan()
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "rejection_analytics_not_available"},
        )
    return JSONResponse(content={"ok": True, **result})


@app.get("/admin/rejections/summary")
def rejection_analytics_summary(
    request: Request,
    hours: int = 24,
    start: Optional[str] = None,
    end: Optional[str] = None,
    x_scan_api_key: Optional[str] = Header(None, alias="X-Scan-API-Key"),
) -> JSONResponse:
    """Protected 1h–31d aggregate or an explicit ISO-8601 range."""
    SCAN_ACCESS.authorize(request, x_scan_api_key)
    repository = get_rejection_repository(get_config())
    if start or end:
        try:
            end_at = datetime.fromisoformat((end or datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00"))
            start_at = datetime.fromisoformat((start or (end_at - timedelta(hours=hours)).isoformat()).replace("Z", "+00:00"))
            if start_at.tzinfo is None:
                start_at = start_at.replace(tzinfo=timezone.utc)
            if end_at.tzinfo is None:
                end_at = end_at.replace(tzinfo=timezone.utc)
            if start_at >= end_at or end_at - start_at > timedelta(days=31):
                raise ValueError("invalid range")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid analytics time range") from exc
        result = repository.range_summary(start=start_at, end=end_at)
    else:
        if hours < 1 or hours > 24 * 31:
            raise HTTPException(status_code=422, detail="hours must be between 1 and 744")
        result = repository.summary_for_hours(hours)
    return JSONResponse(content={"ok": "error" not in result, **result})


@app.get("/admin/rejections/candidates/{candidate_id}")
def rejection_candidate_details(
    candidate_id: str,
    request: Request,
    x_scan_api_key: Optional[str] = Header(None, alias="X-Scan-API-Key"),
) -> JSONResponse:
    """Complete canonical gate evaluation for one candidate."""
    SCAN_ACCESS.authorize(request, x_scan_api_key)
    if not candidate_id or len(candidate_id) > 128:
        raise HTTPException(status_code=422, detail="Invalid candidate ID")
    result = get_rejection_repository(get_config()).candidate(candidate_id)
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "candidate_not_found"},
        )
    return JSONResponse(content={"ok": True, "candidate": result})


@app.get("/signal-tracker/reliability")
def signal_tracker_reliability() -> Dict[str, Any]:
    """Forward outcome bands; scores are not probabilities until calibrated."""
    return get_signal_reliability_summary()


@app.get("/outcome-scoring/status")
def outcome_scoring_status() -> Dict[str, Any]:
    """Redacted durable-journal and calibrated-model readiness."""
    return get_outcome_scoring_status(get_config())


@app.get("/telegram/status")
def telegram_status() -> Dict[str, Any]:
    """Redacted configuration and scheduler state; does not call Telegram."""
    cfg = get_config()
    token, chat = get_telegram_credentials()
    alert_chats = get_telegram_alert_chat_ids()
    delivery = get_delivery_status()
    command_chats = get_telegram_command_chat_ids(chat)
    return {
        "ok": True,
        "telegram": {
            "enabled": bool(cfg.telegram.enabled),
            "ready": is_telegram_ready(cfg),
            "token_configured": bool(token),
            "chat_id_configured": bool(chat),
            "alert_chat_count": len(alert_chats),
            "delivery_mode": delivery["mode"],
            "beta_recipient_count": delivery["beta_recipient_count"],
            "public_delivery_enabled": delivery["public_delivery_enabled"],
            "additional_alert_chats_configured": bool(
                (os.getenv("TELEGRAM_ADDITIONAL_ALERT_CHAT_IDS") or "").strip()
            ),
            "command_chat_ids_explicit": bool(
                (os.getenv("TELEGRAM_COMMAND_CHAT_IDS") or "").strip()
            ),
            "command_chat_count": len(command_chats),
            "test_endpoint_secured": bool(
                (os.getenv("TELEGRAM_TEST_KEY") or "").strip()
            ),
        },
        "webhook": get_telegram_webhook_status(),
        "scheduler": get_scheduler_status(),
        "signal_tracker": get_signal_tracker_status(),
    }


@app.post("/telegram/webhook", include_in_schema=False)
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: Optional[str] = Header(
        None,
        alias="X-Telegram-Bot-Api-Secret-Token",
    ),
) -> Dict[str, Any]:
    """Accept Telegram commands quickly, then process scans after responding."""
    cfg = get_config()
    token, _ = get_telegram_credentials()
    expected = telegram_webhook_secret(token)
    provided = (x_telegram_bot_api_secret_token or "").strip()
    if not expected or not provided or not hmac.compare_digest(provided, expected):
        logger.warning("Rejected Telegram webhook request with invalid secret")
        raise HTTPException(status_code=403, detail="Invalid Telegram webhook secret")
    try:
        update = await request.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ignored malformed Telegram webhook payload: {}", type(exc).__name__)
        return {"ok": True, "accepted": False}
    if not isinstance(update, dict):
        return {"ok": True, "accepted": False}
    background_tasks.add_task(process_telegram_update, update, cfg)
    return {"ok": True, "accepted": True}


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
    request: Request,
    symbols: Optional[str] = Form(
        None,
        description="Comma-separated symbols to scan, e.g. BTC,ETH,SOL",
    ),
    timeframe: Optional[str] = Form(None, description="Primary timeframe, e.g. 15m"),
    exchange: Optional[str] = Form(None, description="binanceusdm | bybit | okx | bitget"),
    no_news: bool = Form(False, description="Skip news fetch"),
    simulated_capital: Optional[float] = Form(None, description="Simulated capital"),
    risk: Optional[float] = Form(None, description="Risk percent"),
    x_scan_api_key: Optional[str] = Header(None, alias="X-Scan-API-Key"),
) -> JSONResponse:
    cfg = get_config()
    SCAN_ACCESS.authorize(request, x_scan_api_key)
    symbol_list = normalize_requested_symbols(
        (symbols or "").split(","), approved_bases=cfg.scheduler.watchlist
    )
    validated_timeframe = validate_timeframe(timeframe, cfg.timeframes.primary)
    validated_exchange = validate_exchange(exchange, cfg.exchange.default)

    req = AnalyzeRequest(
        timeframe=validated_timeframe,
        exchange=validated_exchange,
        simulated_capital=simulated_capital,
        risk_pct=risk,
        no_news=bool(no_news),
    )
    try:
        with SCAN_ACCESS.slot():
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    scan_symbols,
                    symbol_list,
                    req,
                    cfg,
                    trigger_type="api",
                ),
                timeout=SCAN_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={"error": "scan_timeout", "budget_seconds": SCAN_TIMEOUT_SECONDS},
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scan failed: {}", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "scan_failed", "error_type": type(exc).__name__},
        ) from exc
    return JSONResponse(content=result)


@app.post("/analyze")
async def analyze(
    request: Request,
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
    x_scan_api_key: Optional[str] = Header(None, alias="X-Scan-API-Key"),
) -> JSONResponse:
    """
    Maximum-signal chart analysis.

    Pipeline: OCR + CV → multi-TF OHLCV + funding/OI/L-S → full indicator suite →
    patterns + market structure → news → weighted confluence → dynamic leverage
    simulation → optional LLM narrative.
    """
    cfg = get_config()
    SCAN_ACCESS.authorize(request, x_scan_api_key)
    if timeframe:
        timeframe = validate_timeframe(timeframe, cfg.timeframes.primary)
    if exchange:
        exchange = validate_exchange(exchange, cfg.exchange.default)

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
        with SCAN_ACCESS.slot():
            result = await asyncio.wait_for(
                asyncio.to_thread(analyze_from_image, raw, req, cfg),
                timeout=SCAN_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={"error": "analysis_timeout", "budget_seconds": SCAN_TIMEOUT_SECONDS},
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Analyze failed: {}", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "analysis_failed", "error_type": type(exc).__name__},
        ) from exc

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
