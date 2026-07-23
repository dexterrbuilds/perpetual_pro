"""Scheduled prop watchlist scan + Telegram delivery."""

from __future__ import annotations

from datetime import datetime, timedelta
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from loguru import logger

from src.api.service import AnalyzeRequest, scan_symbols
from src.notify.telegram import (
    format_signal_photo_caption,
    format_prop_scan_report,
    is_telegram_ready,
    send_telegram_message_detailed,
    send_telegram_photo_detailed,
)
from src.report.charts import render_signal_chart_png
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
    "sessions": [],
    "next_session": None,
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
_SCAN_RUN_LOCK = Lock()


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
    for t in times or ["09:00", "15:00", "20:00"]:
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


def next_session_datetime(
    sessions: List[Dict[str, str]],
    now: Optional[datetime] = None,
) -> Tuple[datetime, str]:
    """Return the next named market session, preserving its local timezone."""
    now_utc = (
        now.astimezone(ZoneInfo("UTC"))
        if now is not None and now.tzinfo is not None
        else (
            now.replace(tzinfo=ZoneInfo("UTC"))
            if now is not None
            else datetime.now(ZoneInfo("UTC"))
        )
    )
    candidates: List[Tuple[datetime, str]] = []
    for session in sessions or []:
        try:
            name = str(session.get("name") or "Trading session")
            tz_name = str(session.get("timezone") or "UTC")
            hour, minute = _parse_hhmm(str(session.get("time") or "00:00"))
            tz = ZoneInfo(tz_name)
        except (AttributeError, TypeError, ValueError, KeyError):
            continue
        local_now = now_utc.astimezone(tz)
        candidate = local_now.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0,
        )
        if candidate <= local_now:
            candidate += timedelta(days=1)
        candidates.append((candidate, name))
    if not candidates:
        fallback = now_utc + timedelta(hours=1)
        return fallback, "Fallback scan"
    return min(candidates, key=lambda item: item[0].astimezone(ZoneInfo("UTC")))


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


def _run_scheduled_scan_once_unlocked(
    config: Optional[AppConfig] = None,
    *,
    slot_label: str = "",
    send: bool = True,
    symbols: Optional[List[str]] = None,
    timeframe: Optional[str] = None,
    notify_on_empty: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run one watchlist scan and optionally Telegram high-conf results."""
    cfg = config or load_config()
    started_at = datetime.now(ZoneInfo("UTC")).isoformat()
    watchlist = list(symbols or cfg.scheduler.watchlist or []) or list(DEFAULT_WATCHLIST)
    req = AnalyzeRequest(
        timeframe=timeframe or cfg.scheduler.timeframe or cfg.timeframes.primary,
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
        scanned_count=len(watchlist),
        ranked_count=len(ranked),
    )
    sent = False
    delivery: Optional[Dict[str, Any]] = None
    delivery_status = "not_requested"
    # Credentials from env only (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — never YAML
    tg_ready = is_telegram_ready(cfg)
    if send and tg_ready:
        if filtered:
            chart_items: List[Dict[str, Any]] = []
            # Each actionable signal is self-contained: plotted chart + concise call.
            for index, row in enumerate(filtered[:6], 1):
                symbol = str(row.get("symbol") or f"signal-{index}")
                try:
                    chart_png = render_signal_chart_png(row)
                    caption = format_signal_photo_caption(
                        row,
                        slot_label=slot_label or "scan",
                    )
                    item = send_telegram_photo_detailed(
                        chart_png,
                        caption,
                        filename=f"{symbol.split('/')[0].lower()}-signal.png",
                        parse_mode=cfg.telegram.parse_mode or "HTML",
                    )
                    item["symbol"] = symbol
                    item["mode"] = "photo"
                    chart_items.append(item)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Telegram chart alert failed before upload: symbol={} error_type={}",
                        symbol,
                        type(exc).__name__,
                    )
                    chart_items.append(
                        {
                            "ok": False,
                            "symbol": symbol,
                            "mode": "photo",
                            "error": "chart_render_failed",
                            "description": type(exc).__name__,
                        }
                    )

            chart_sent = sum(1 for item in chart_items if item.get("ok"))
            chart_failed = len(chart_items) - chart_sent
            all_covered = len(filtered) <= len(chart_items)
            photo_ok = bool(chart_items) and chart_failed == 0 and all_covered
            delivery = {
                "ok": photo_ok,
                "mode": "photo",
                "sent_count": chart_sent,
                "failed_count": chart_failed,
                "total_actionable": len(filtered),
                "items": chart_items,
            }
            if photo_ok:
                sent = True
                delivery_status = "sent_chart_alerts"
                logger.info(
                    "Scheduled Telegram chart alerts succeeded: slot={} sent={}",
                    slot_label or "scan",
                    chart_sent,
                )
            else:
                # A clean text report is a dependable fallback for rendering,
                # Telegram media, or six-photo limit failures.
                fallback = send_telegram_message_detailed(
                    report,
                    parse_mode=cfg.telegram.parse_mode or "HTML",
                )
                delivery["text_fallback"] = fallback
                sent = bool(fallback.get("ok")) or chart_sent > 0
                delivery["ok"] = sent
                delivery_status = (
                    "sent_with_text_fallback" if sent else "failed"
                )
                if sent:
                    logger.warning(
                        "Telegram chart delivery incomplete: slot={} photos={}/{}; "
                        "text fallback sent={}",
                        slot_label or "scan",
                        chart_sent,
                        len(chart_items),
                        bool(fallback.get("ok")),
                    )
                else:
                    logger.error(
                        "Telegram chart and text fallback both failed: slot={}",
                        slot_label or "scan",
                    )
        elif (
            bool(notify_on_empty)
            if notify_on_empty is not None
            else cfg.telegram.notify_on_empty
        ):
            delivery = send_telegram_message_detailed(
                report,
                parse_mode=cfg.telegram.parse_mode or "HTML",
            )
            sent = bool(delivery.get("ok"))
            if sent:
                delivery_status = "sent_empty_report"
                logger.info(
                    "Scheduled empty Telegram report succeeded: slot={} message_id={}",
                    slot_label or "scan",
                    delivery.get("message_id"),
                )
            else:
                delivery_status = "failed"
                logger.error(
                    "Scheduled empty Telegram report failed: slot={} error={} description={}",
                    slot_label or "scan",
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


def scan_in_progress() -> bool:
    """Whether a scheduled or on-demand watchlist scan currently owns the worker."""
    return _SCAN_RUN_LOCK.locked()


def run_scheduled_scan_once(
    config: Optional[AppConfig] = None,
    *,
    slot_label: str = "",
    send: bool = True,
    symbols: Optional[List[str]] = None,
    timeframe: Optional[str] = None,
    notify_on_empty: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run one scan without overlapping another scheduler, API, or bot request."""
    started_at = datetime.now(ZoneInfo("UTC")).isoformat()
    if not _SCAN_RUN_LOCK.acquire(blocking=False):
        logger.warning("Scan request skipped because another scan is already running")
        return {
            "ok": False,
            "error": "scan_in_progress",
            "started_at": started_at,
            "completed_at": started_at,
            "scanned": 0,
            "ranked_count": 0,
            "alert_count": 0,
            "filtered": [],
            "report": "",
            "telegram_sent": False,
            "telegram_ready": is_telegram_ready(config),
            "telegram_delivery_status": "scan_in_progress",
            "telegram_delivery": None,
            "slot_label": slot_label,
        }
    try:
        return _run_scheduled_scan_once_unlocked(
            config,
            slot_label=slot_label,
            send=send,
            symbols=symbols,
            timeframe=timeframe,
            notify_on_empty=notify_on_empty,
        )
    finally:
        _SCAN_RUN_LOCK.release()


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
    times = list(cfg.scheduler.times or ["09:00", "15:00", "20:00"])
    sessions = list(getattr(cfg.scheduler, "sessions", None) or [])
    tz_name = cfg.scheduler.timezone or "Africa/Lagos"
    _status_update(
        enabled=bool(cfg.scheduler.enabled),
        running=True,
        timezone=tz_name,
        times=times,
        sessions=sessions,
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
            _status_update(running=False, next_run_at=None, next_session=None)
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
        (
            [
                f"{session.get('name')} {session.get('time')} {session.get('timezone')}"
                for session in sessions
            ]
            if sessions
            else times
        ),
        tz_name,
    )
    try:
        while not stop.is_set():
            if sessions:
                nxt, session_name = next_session_datetime(sessions)
            else:
                nxt = next_slot_datetime(times, tz_name)
                session_name = "Scheduled scan"
            _status_update(
                next_run_at=nxt.isoformat(),
                next_session=session_name,
            )
            now = datetime.now(nxt.tzinfo or ZoneInfo("UTC"))
            sleep_s = max(0.0, (nxt - now).total_seconds())
            logger.info(
                "Next scheduled scan: {} · {} (in {:.0f}s)",
                session_name,
                nxt.isoformat(),
                sleep_s,
            )
            if stop.wait(timeout=sleep_s):
                break
            wat = nxt.astimezone(ZoneInfo(tz_name))
            label = (
                f"{session_name} · {nxt.strftime('%H:%M %Z')} "
                f"({wat.strftime('%H:%M %Z')})"
            )
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
        _status_update(running=False, next_run_at=None, next_session=None)
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
            sessions=list(getattr(cfg.scheduler, "sessions", None) or []),
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
        sessions=list(getattr(cfg.scheduler, "sessions", None) or []),
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
    _status_update(
        thread_alive=alive,
        running=alive,
        next_run_at=None,
        next_session=None,
    )
    if alive:
        logger.warning("Scheduler thread did not stop within {:.1f}s", timeout)
    else:
        logger.info("Scheduler background thread stopped")
        _BACKGROUND_THREAD = None
        _BACKGROUND_STOP = None
