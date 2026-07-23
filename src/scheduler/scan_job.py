"""Scheduled prop watchlist scan + Telegram delivery."""

from __future__ import annotations

from datetime import datetime, timedelta
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from loguru import logger

from src.api.service import AnalyzeRequest, scan_symbols
from src.notify.telegram import (
    format_prop_scan_report,
    is_telegram_ready,
    send_telegram_message_detailed,
)
from src.utils.config import AppConfig, load_config

# Fallback watchlist when scheduler.watchlist is empty
DEFAULT_WATCHLIST = [
    "BTC", "ETH", "SOL", "BNB", "AAVE", "ARB", "NEAR", "INJ", "SEI", "TIA",
    "SUI", "APT", "AVAX", "TRX", "UNI",
]

_STATUS_LOCK = Lock()
_SCHEDULER_STATUS: Dict[str, Any] = {
    "enabled": False,
    "running": False,
    "thread_alive": False,
    "timezone": None,
    "times": [],
    "started_at": None,
    "next_run_at": None,
    "last_triggered_at": None,
    "last_completed_at": None,
    "last_error": None,
    "last_delivery_status": None,
    "last_alert_count": None,
}
_BACKGROUND_THREAD: Optional[Thread] = None
_BACKGROUND_STOP: Optional[Event] = None


def _status_update(**values: Any) -> None:
    with _STATUS_LOCK:
        _SCHEDULER_STATUS.update(values)
        _SCHEDULER_STATUS["thread_alive"] = bool(
            _BACKGROUND_THREAD and _BACKGROUND_THREAD.is_alive()
        )


def get_scheduler_status() -> Dict[str, Any]:
    """Return a JSON-safe snapshot for health checks and the webapp."""
    with _STATUS_LOCK:
        status = dict(_SCHEDULER_STATUS)
    status["thread_alive"] = bool(
        _BACKGROUND_THREAD and _BACKGROUND_THREAD.is_alive()
    )
    return status


def _parse_hhmm(s: str) -> Tuple[int, int]:
    parts = (s or "00:00").strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m


def next_slot_datetime(
    times: List[str],
    timezone: str = "Africa/Lagos",
    now: Optional[datetime] = None,
) -> datetime:
    """Next future slot in the given timezone."""
    tz = ZoneInfo(timezone)
    now = now.astimezone(tz) if now else datetime.now(tz)
    candidates: List[datetime] = []
    for t in times or ["05:00", "16:00", "20:00"]:
        try:
            h, m = _parse_hhmm(t)
        except (TypeError, ValueError):
            continue
        candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if candidate <= now:
            candidate = candidate + timedelta(days=1)
        candidates.append(candidate)
    if not candidates:
        return now + timedelta(hours=1)
    return min(candidates)


def filter_high_confidence(
    ranked: List[Dict[str, Any]],
    *,
    min_llm: float,
    min_rank: float,
    only_prop_safe: bool,
    min_confidence: float = 68.0,
    min_execution_score: float = 65.0,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in ranked or []:
        direction = str(row.get("direction") or "").lower()
        if direction not in ("long", "short"):
            continue
        llm = float(row.get("llm_confidence") or 0)
        rank = float(row.get("rank_score") or 0)
        blended = row.get("confidence")
        if blended is not None and float(blended or 0) < min_confidence:
            continue
        if llm < min_llm or rank < min_rank:
            continue
        if row.get("signal_eligible") is False:
            continue
        execution_score = row.get("execution_score")
        if execution_score is not None and float(execution_score or 0) < min_execution_score:
            continue
        entry_status = row.get("entry_status")
        if entry_status is not None and entry_status not in ("ready", "wait_retest"):
            continue
        if only_prop_safe and row.get("prop_safe") is False:
            continue
        out.append(row)
    out.sort(
        key=lambda r: (float(r.get("rank_score") or 0), float(r.get("llm_confidence") or 0)),
        reverse=True,
    )
    return out


def run_scheduled_scan_once(
    config: Optional[AppConfig] = None,
    *,
    slot_label: str = "",
    send: bool = True,
) -> Dict[str, Any]:
    """Run one watchlist scan and optionally Telegram high-conf results."""
    cfg = config or load_config()
    started_at = datetime.now(ZoneInfo("UTC")).isoformat()
    watchlist = list(cfg.scheduler.watchlist or []) or list(DEFAULT_WATCHLIST)
    req = AnalyzeRequest(
        timeframe=cfg.scheduler.timeframe or cfg.timeframes.primary,
        exchange=cfg.scheduler.exchange or cfg.exchange.default,
        no_news=bool(cfg.scheduler.no_news),
        simulated_capital=cfg.risk.simulated_capital,
        risk_pct=cfg.risk.risk_per_trade_pct,
        use_llm=True,
    )
    logger.info(
        "Scheduled scan triggered: slot={} symbols={} timeframe={} exchange={} send={}",
        slot_label or "manual",
        len(watchlist),
        req.timeframe,
        req.exchange,
        send,
    )
    result = scan_symbols(watchlist, request=req, config=cfg)
    ranked = result.get("ranked_results") or []
    filtered = filter_high_confidence(
        ranked,
        min_llm=float(cfg.telegram.min_llm_confidence or 65),
        min_rank=float(cfg.telegram.min_rank_score or 50),
        only_prop_safe=bool(cfg.scheduler.only_prop_safe),
        min_confidence=float(
            getattr(cfg.analysis, "directional_confidence_threshold", 68.0)
        ),
        min_execution_score=float(
            getattr(cfg.analysis, "execution_min_score", 65.0)
        ),
    )
    report = format_prop_scan_report(
        filtered,
        slot_label=slot_label or "scan",
        timezone=cfg.scheduler.timezone or "Africa/Lagos",
        min_signal_confidence=float(
            getattr(cfg.analysis, "directional_confidence_threshold", 68.0)
        ),
    )
    sent = False
    delivery: Optional[Dict[str, Any]] = None
    delivery_status = "not_requested"
    # Credentials from env only (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — never YAML
    tg_ready = is_telegram_ready(cfg)
    if send and tg_ready:
        if filtered or cfg.telegram.notify_on_empty:
            delivery = send_telegram_message_detailed(
                report,
                parse_mode=cfg.telegram.parse_mode or "HTML",
            )
            sent = bool(delivery.get("ok"))
            if sent:
                delivery_status = "sent"
                logger.info(
                    "Scheduled Telegram alert succeeded: slot={} signals={} message_id={}",
                    slot_label or "scan",
                    len(filtered),
                    delivery.get("message_id"),
                )
            else:
                delivery_status = "failed"
                logger.error(
                    "Scheduled Telegram alert failed: slot={} signals={} error={} description={}",
                    slot_label or "scan",
                    len(filtered),
                    delivery.get("error"),
                    delivery.get("description"),
                )
        else:
            delivery_status = "skipped_no_actionable_signals"
            logger.info(
                "Scheduled Telegram alert skipped: slot={} no actionable signals and "
                "notify_on_empty=false",
                slot_label or "scan",
            )
    elif send and not tg_ready:
        delivery_status = "failed_not_configured"
        logger.warning(
            "Scheduled Telegram alert failed before send: credentials disabled or missing. "
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the running process environment."
        )
    elif not send:
        delivery_status = "disabled_for_run"
    completed_at = datetime.now(ZoneInfo("UTC")).isoformat()
    logger.info(
        "Scheduled scan completed: slot={} scanned={} ranked={} actionable={} "
        "delivery_status={}",
        slot_label or "scan",
        len(watchlist),
        len(ranked),
        len(filtered),
        delivery_status,
    )
    return {
        "ok": bool(result.get("ok")),
        "started_at": started_at,
        "completed_at": completed_at,
        "scanned": len(watchlist),
        "ranked_count": len(ranked),
        "alert_count": len(filtered),
        "filtered": filtered,
        "report": report,
        "telegram_sent": sent,
        "telegram_ready": tg_ready,
        "telegram_delivery_status": delivery_status,
        "telegram_delivery": delivery,
        "slot_label": slot_label,
    }


def run_scheduler_loop(
    config: Optional[AppConfig] = None,
    *,
    once: bool = False,
    max_iterations: Optional[int] = None,
    stop_event: Optional[Event] = None,
) -> None:
    """
    Sleep until next WAT slot, run scan, repeat.

    ``once=True`` runs a single scan immediately (for cron / CLI).
    """
    cfg = config or load_config()
    stop = stop_event or Event()
    times = list(cfg.scheduler.times or ["05:00", "16:00", "20:00"])
    tz_name = cfg.scheduler.timezone or "Africa/Lagos"
    _status_update(
        enabled=bool(cfg.scheduler.enabled),
        running=True,
        timezone=tz_name,
        times=times,
        started_at=datetime.now(ZoneInfo("UTC")).isoformat(),
        last_error=None,
    )
    if once:
        try:
            _status_update(last_triggered_at=datetime.now(ZoneInfo("UTC")).isoformat())
            outcome = run_scheduled_scan_once(cfg, slot_label="manual")
            _status_update(
                last_completed_at=outcome.get("completed_at"),
                last_delivery_status=outcome.get("telegram_delivery_status"),
                last_alert_count=outcome.get("alert_count"),
            )
        except Exception as exc:  # noqa: BLE001
            _status_update(last_error=f"{type(exc).__name__}: {exc}")
            logger.exception("Manual scheduled scan failed: {}", exc)
            raise
        finally:
            _status_update(running=False, next_run_at=None)
        return

    iterations = 0
    if not is_telegram_ready(cfg):
        logger.warning(
            "Scheduler running without Telegram credentials. "
            "Export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to receive alerts."
        )
    else:
        logger.info("Telegram credentials loaded from environment (token redacted)")
    logger.info(
        "Scheduler loop started: times={} timezone={} high-confidence-only=true",
        times,
        tz_name,
    )
    try:
        while not stop.is_set():
            nxt = next_slot_datetime(times, tz_name)
            _status_update(next_run_at=nxt.isoformat())
            now = datetime.now(ZoneInfo(tz_name))
            sleep_s = max(0.0, (nxt - now).total_seconds())
            logger.info("Next scheduled scan: {} (in {:.0f}s)", nxt.isoformat(), sleep_s)
            if stop.wait(timeout=sleep_s):
                break
            label = nxt.strftime("%H:%M %Z")
            triggered_at = datetime.now(ZoneInfo("UTC")).isoformat()
            _status_update(last_triggered_at=triggered_at, last_error=None)
            try:
                outcome = run_scheduled_scan_once(cfg, slot_label=label, send=True)
                _status_update(
                    last_completed_at=outcome.get("completed_at"),
                    last_delivery_status=outcome.get("telegram_delivery_status"),
                    last_alert_count=outcome.get("alert_count"),
                )
            except Exception as exc:  # noqa: BLE001
                _status_update(
                    last_completed_at=datetime.now(ZoneInfo("UTC")).isoformat(),
                    last_error=f"{type(exc).__name__}: {exc}",
                    last_delivery_status="scan_failed",
                )
                logger.exception("Scheduled scan failed: {}", exc)
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
    finally:
        _status_update(running=False, next_run_at=None)
        logger.info("Scheduler loop stopped")


def start_scheduler_background(config: Optional[AppConfig] = None) -> bool:
    """Start one daemon scheduler thread for the API process."""
    global _BACKGROUND_THREAD, _BACKGROUND_STOP
    cfg = config or load_config()
    if not cfg.scheduler.enabled:
        _status_update(
            enabled=False,
            running=False,
            timezone=cfg.scheduler.timezone,
            times=list(cfg.scheduler.times),
        )
        logger.warning("Scheduler disabled by configuration (scheduler.enabled=false)")
        return False
    if _BACKGROUND_THREAD and _BACKGROUND_THREAD.is_alive():
        logger.info("Scheduler background thread already running")
        return True
    _BACKGROUND_STOP = Event()
    _BACKGROUND_THREAD = Thread(
        target=run_scheduler_loop,
        kwargs={"config": cfg, "stop_event": _BACKGROUND_STOP},
        name="perpetual-pro-scheduler",
        daemon=True,
    )
    _BACKGROUND_THREAD.start()
    _status_update(
        enabled=True,
        thread_alive=True,
        timezone=cfg.scheduler.timezone,
        times=list(cfg.scheduler.times),
    )
    logger.info("Scheduler background thread started")
    return True


def stop_scheduler_background(timeout: float = 5.0) -> None:
    """Request scheduler shutdown when the API process exits."""
    global _BACKGROUND_THREAD, _BACKGROUND_STOP
    if _BACKGROUND_STOP is not None:
        _BACKGROUND_STOP.set()
    if _BACKGROUND_THREAD is not None and _BACKGROUND_THREAD.is_alive():
        _BACKGROUND_THREAD.join(timeout=max(0.0, timeout))
    alive = bool(_BACKGROUND_THREAD and _BACKGROUND_THREAD.is_alive())
    _status_update(thread_alive=alive, running=alive, next_run_at=None)
    if alive:
        logger.warning("Scheduler thread did not stop within {:.1f}s", timeout)
    else:
        logger.info("Scheduler background thread stopped")
        _BACKGROUND_THREAD = None
        _BACKGROUND_STOP = None
