"""Scheduled prop watchlist scan + Telegram delivery."""

from __future__ import annotations

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from threading import Event, Lock, Thread
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from zoneinfo import ZoneInfo

from loguru import logger

from src.analytics.rejection import (
    ALERT_MIN_OVERALL_QUALITY,
    evaluate_alert_gates,
    evaluate_revalidation_gates,
)
from src.analytics.runtime import get_rejection_repository
from src.api.service import AnalyzeRequest, scan_symbols
from src.analysis.revalidation import revalidate_candidate_for_delivery
from src.analysis.qualification import (
    build_private_beta_rejection_summary,
    evaluate_private_beta_qualification,
    qualify_private_beta_candidates,
)
from src.experiments.identity import (
    bot_namespace,
    identity_metadata,
    is_legacy_comparison,
)
from src.experiments.legacy_policy import (
    LEGACY_POLICY_VERSION,
    evaluate_legacy_qualification,
    qualify_legacy_candidates,
    strict_shadow_decision,
)
from src.notify.telegram import (
    format_signal_photo_caption,
    format_prop_scan_report,
    get_delivery_status,
    get_private_beta_chat_ids,
    get_telegram_alert_chat_ids,
    get_telegram_private_operator_chat_ids,
    get_telegram_report_chat_ids,
    is_telegram_ready,
    send_telegram_message_detailed,
    send_telegram_photo_detailed,
)
from src.report.charts import render_signal_chart_png
from src.scheduler.run_repository import SchedulerRunRepository
from src.tracking.durable_repository import destination_hash
from src.tracking.signal_tracker import register_delivered_signals
from src.utils.config import (
    DEFAULT_CRYPTO_WATCHLIST,
    AppConfig,
    load_config,
)

# Fallback watchlist when scheduler.watchlist is empty
DEFAULT_WATCHLIST = list(DEFAULT_CRYPTO_WATCHLIST)
MIN_TELEGRAM_SIGNAL_CONFIDENCE = ALERT_MIN_OVERALL_QUALITY
SCHEDULER_MISFIRE_GRACE_SECONDS = 300
SCHEDULER_RECOVERY_LOOKBACK_HOURS = 24
PRIVATE_BETA_HOURLY_SESSION_NAME = "Private beta hourly scan"

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
    "last_run": None,
    "previous_expected_run_at": None,
    "previous_actual_run_at": None,
    "previous_run_status": None,
    "previous_misfire_status": None,
    "previous_durable_run_id": None,
    "misfire_grace_seconds": SCHEDULER_MISFIRE_GRACE_SECONDS,
    "private_beta_hourly_enabled": False,
}
_BACKGROUND_THREAD: Optional[Thread] = None
_BACKGROUND_STOP: Optional[Event] = None
_SCAN_RUN_LOCK = Lock()
_RECENT_SIGNAL_LOCK = Lock()
_RECENT_SIGNAL_ALERTS: Dict[Tuple[str, str], Dict[str, float]] = {}


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


def _durable_run_id(slot_label: str, scheduled_for: Optional[str]) -> str:
    if scheduled_for:
        material = f"{bot_namespace()}|{slot_label}|{scheduled_for}"
        prefix = "legacy_sched_" if is_legacy_comparison() else "sched_"
        return prefix + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    prefix = "legacy_run_" if is_legacy_comparison() else "run_"
    return prefix + uuid4().hex


def _parse_utc_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed.astimezone(ZoneInfo("UTC"))


def scheduled_session_windows_between(
    sessions: List[Dict[str, str]],
    *,
    start: datetime,
    end: datetime,
) -> List[Tuple[datetime, str]]:
    """Enumerate DST-aware expected windows in ``(start, end]``."""
    start_utc = start.astimezone(ZoneInfo("UTC"))
    end_utc = end.astimezone(ZoneInfo("UTC"))
    windows: List[Tuple[datetime, str]] = []
    for session in sessions or []:
        try:
            name = str(session.get("name") or "Trading session")
            tz = ZoneInfo(str(session.get("timezone") or "UTC"))
            hour, minute = _parse_hhmm(str(session.get("time") or "00:00"))
        except (AttributeError, TypeError, ValueError, KeyError):
            continue
        local_start = start_utc.astimezone(tz)
        local_end = end_utc.astimezone(tz)
        cursor = local_start.replace(hour=0, minute=0, second=0, microsecond=0)
        last_day = local_end.replace(hour=0, minute=0, second=0, microsecond=0)
        while cursor <= last_day:
            candidate = cursor.replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            ).astimezone(ZoneInfo("UTC"))
            if start_utc < candidate <= end_utc:
                windows.append((candidate, name))
            cursor += timedelta(days=1)
    return sorted(windows, key=lambda item: item[0])


def next_hourly_datetime(
    now: Optional[datetime] = None,
) -> datetime:
    """Return the next top-of-hour UTC boundary, strictly in the future."""
    now_utc = (
        now.astimezone(ZoneInfo("UTC"))
        if now is not None and now.tzinfo is not None
        else (
            now.replace(tzinfo=ZoneInfo("UTC"))
            if now is not None
            else datetime.now(ZoneInfo("UTC"))
        )
    )
    return now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(
        hours=1
    )


def scheduled_hourly_windows_between(
    *,
    start: datetime,
    end: datetime,
) -> List[Tuple[datetime, str]]:
    """Enumerate private-beta top-of-hour windows in ``(start, end]``."""
    start_utc = start.astimezone(ZoneInfo("UTC"))
    end_utc = end.astimezone(ZoneInfo("UTC"))
    cursor = start_utc.replace(minute=0, second=0, microsecond=0)
    if cursor <= start_utc:
        cursor += timedelta(hours=1)
    windows: List[Tuple[datetime, str]] = []
    while cursor <= end_utc:
        windows.append((cursor, PRIVATE_BETA_HOURLY_SESSION_NAME))
        cursor += timedelta(hours=1)
    return windows


def _private_beta_hourly_enabled(cfg: AppConfig) -> bool:
    return bool(
        getattr(cfg.scheduler, "private_beta_hourly_enabled", True)
        and get_delivery_status().get("mode") == "private_beta"
    )


def _active_scheduler_windows(
    cfg: AppConfig,
    sessions: List[Dict[str, str]],
    times: List[str],
) -> List[Any]:
    windows: List[Any] = list(sessions if sessions else times)
    if _private_beta_hourly_enabled(cfg):
        windows.append(
            {
                "name": PRIVATE_BETA_HOURLY_SESSION_NAME,
                "schedule": "hourly",
                "minute": 0,
                "timezone": "UTC",
                "delivery_mode": "private_beta",
                "notify_on_empty": False,
            }
        )
    return windows


def previous_session_datetime(
    sessions: List[Dict[str, str]],
    *,
    now: Optional[datetime] = None,
) -> Tuple[Optional[datetime], Optional[str]]:
    """Return the most recent expected DST-aware production window."""
    now_utc = (now or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo("UTC"))
    windows = scheduled_session_windows_between(
        sessions,
        start=now_utc - timedelta(days=2),
        end=now_utc,
    )
    return windows[-1] if windows else (None, None)


def _scheduled_slot_label(
    session_name: str,
    scheduled_at: datetime,
    status_timezone: str,
    sessions: Optional[List[Dict[str, str]]] = None,
) -> str:
    session_timezone = next(
        (
            str(item.get("timezone") or "UTC")
            for item in sessions or []
            if str(item.get("name") or "") == session_name
        ),
        "UTC",
    )
    local = scheduled_at.astimezone(ZoneInfo(session_timezone))
    wat = scheduled_at.astimezone(ZoneInfo(status_timezone))
    return (
        f"{session_name} · {local.strftime('%H:%M %Z')} "
        f"({wat.strftime('%H:%M %Z')})"
    )


def _safe_requester_hash(chat_id: Optional[str]) -> Optional[str]:
    value = str(chat_id or "").strip()
    return destination_hash(value) if value else None


def _merge_destinations(*groups: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for group in groups:
        for destination in group or []:
            value = str(destination or "").strip()
            if value and value not in merged:
                merged.append(value)
    return merged


def _signal_delivery_idempotency_key(
    row: Dict[str, Any],
    destination: str,
) -> str:
    """Stable same-day signal/destination key shared by manual and scheduled scans."""
    generated = _parse_utc_datetime(row.get("signal_generated_at"))
    date_bucket = (generated or datetime.now(ZoneInfo("UTC"))).date().isoformat()
    levels = [
        row.get("symbol"),
        str(row.get("direction") or "").lower(),
        row.get("execution_setup_type") or row.get("setup_name"),
        row.get("entry_low"),
        row.get("entry_high"),
        row.get("stop_loss"),
        (list(row.get("take_profits") or []) or [None])[0],
        date_bucket,
    ]
    material = "|".join(str(value) for value in levels)
    signal_digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return (
        f"{bot_namespace()}:signal_initial:v1:{signal_digest}:"
        f"{destination_hash(destination)}"
    )


def _recover_expected_scheduler_windows(
    cfg: AppConfig,
    repository: SchedulerRunRepository,
    *,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Persist/recover windows lost to restart without replaying stale data."""
    sessions = list(getattr(cfg.scheduler, "sessions", None) or [])
    hourly_enabled = _private_beta_hourly_enabled(cfg)
    if (not sessions and not hourly_enabled) or not repository.enabled:
        return []
    now_utc = (now or datetime.now(ZoneInfo("UTC"))).astimezone(ZoneInfo("UTC"))
    latest = repository.latest_scheduled()
    latest_expected = _parse_utc_datetime(
        (latest or {}).get("scheduled_for")
    )
    start = max(
        now_utc - timedelta(hours=SCHEDULER_RECOVERY_LOOKBACK_HOURS),
        latest_expected if latest_expected is not None else (
            now_utc - timedelta(seconds=SCHEDULER_MISFIRE_GRACE_SECONDS)
        ),
    )
    latest_hourly_reader = getattr(repository, "latest_hourly", None)
    latest_hourly = (
        latest_hourly_reader() if callable(latest_hourly_reader) else None
    )
    latest_hourly_expected = _parse_utc_datetime(
        (latest_hourly or {}).get("scheduled_for")
    )
    # A newly introduced hourly schedule has no historical obligations before
    # deployment. Once its first durable row exists, normal 24-hour restart
    # recovery resumes from that independent cursor.
    hourly_start = max(
        now_utc - timedelta(hours=SCHEDULER_RECOVERY_LOOKBACK_HOURS),
        latest_hourly_expected if latest_hourly_expected is not None else (
            now_utc - timedelta(seconds=SCHEDULER_MISFIRE_GRACE_SECONDS)
        ),
    )
    outcomes: List[Dict[str, Any]] = []
    expected_windows = scheduled_session_windows_between(
        sessions, start=start, end=now_utc
    )
    if hourly_enabled:
        expected_windows.extend(
            scheduled_hourly_windows_between(
                start=hourly_start,
                end=now_utc,
            )
        )
    for expected_at, session_name in sorted(
        expected_windows, key=lambda item: item[0]
    ):
        hourly_window = session_name == PRIVATE_BETA_HOURLY_SESSION_NAME
        scheduled_for = expected_at.isoformat()
        label = _scheduled_slot_label(
            session_name,
            expected_at,
            cfg.scheduler.timezone or "Africa/Lagos",
            sessions,
        )
        run_id = _durable_run_id(label, scheduled_for)
        if repository.get(run_id) is not None:
            continue
        logger.warning(
            "Recovering missing scheduler window: run={} slot={} age_seconds={:.0f}",
            run_id,
            session_name,
            max(0.0, (now_utc - expected_at).total_seconds()),
        )
        outcomes.append(
            run_scheduled_scan_once(
                cfg,
                slot_label=label,
                send=True,
                scheduled_for=scheduled_for,
                run_id=run_id,
                notify_on_empty=False if hourly_window else None,
                skip_if_busy=hourly_window,
                warn_on_misfire=not hourly_window,
            )
        )
    return outcomes


def _scheduler_run_source(
    slot_label: str,
    telegram_chat_ids: Optional[List[str]],
    scheduled_for: Optional[str],
) -> str:
    if scheduled_for:
        return "scheduled"
    if telegram_chat_ids is not None or "telegram" in str(slot_label or "").lower():
        return "telegram"
    return "manual"


def _compact_run_summary(outcome: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbols_analyzed": int(outcome.get("analyzed_count") or 0),
        "symbol_failures": int(outcome.get("analysis_failure_count") or 0),
        "eligible_candidates": int(outcome.get("alert_count") or 0),
        "revalidated_candidates": int(outcome.get("alert_count") or 0),
        "rejected_candidates": int(outcome.get("rejected_count") or 0),
        "scan_duration_seconds": outcome.get("scan_duration_seconds"),
        "max_ticker_age_seconds": outcome.get("max_ticker_age_seconds"),
        "max_orderbook_age_seconds": outcome.get("max_orderbook_age_seconds"),
        "result_code": outcome.get("result_code"),
        "telegram_delivery_status": outcome.get("telegram_delivery_status"),
        "main_rejection_reasons": dict(outcome.get("main_rejection_reasons") or {}),
        "llm_invocation_counts": dict(outcome.get("llm_invocation_counts") or {}),
        "public_empty_suppressed": bool(outcome.get("public_empty_suppressed")),
    }


def _send_missed_window_warning(
    *,
    run_id: str,
    slot_label: str,
    scheduled_for: str,
    reason: str,
) -> List[Dict[str, Any]]:
    """Warn private operators only; never fall back to a public destination."""
    deliveries: List[Dict[str, Any]] = []
    text = (
        ("🧪 <b>PERPETUAL PRO LEGACY</b>\n" if is_legacy_comparison() else "")
        + "⚠️ <b>SCHEDULED WINDOW MISSED</b>\n\n"
        f"Window: <b>{slot_label}</b>\n"
        f"Expected: <code>{scheduled_for}</code>\n"
        f"Reason: {reason}\n\n"
        "The grace period expired, so no stale market scan was replayed."
    )
    for destination in get_telegram_private_operator_chat_ids():
        result = send_telegram_message_detailed(
            text,
            chat_id=destination,
            parse_mode="HTML",
        )
        digest = destination_hash(destination)
        deliveries.append(
            {
                "idempotency_key": f"{run_id}:{digest}:misfire_warning",
                "destination_hash": digest,
                "delivery_type": "misfire_warning",
                "ok": bool(result.get("ok")),
                "message_id": result.get("message_id"),
                "error": result.get("error"),
            }
        )
    return deliveries


def _missed_window_result(
    *,
    repository: SchedulerRunRepository,
    run_id: str,
    slot_label: str,
    scheduled_for: str,
    reason: str,
    warn_operator: bool = True,
) -> Dict[str, Any]:
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    deliveries = (
        _send_missed_window_warning(
            run_id=run_id,
            slot_label=slot_label,
            scheduled_for=scheduled_for,
            reason=reason,
        )
        if warn_operator
        else []
    )
    summary = {
        "result_code": "missed_beyond_grace",
        "misfire_status": "missed_beyond_grace",
        "misfire_reason": reason,
        "symbols_analyzed": 0,
        "symbol_failures": 0,
        "eligible_candidates": 0,
        "revalidated_candidates": 0,
        "rejected_candidates": 0,
        "telegram_delivery_status": (
            "sent_misfire_warning"
            if any(item.get("ok") for item in deliveries)
            else (
                "misfire_warning_not_delivered"
                if warn_operator
                else "skipped_misfire_silent"
            )
        ),
    }
    persisted = repository.finish(
        run_id=run_id,
        status="skipped",
        summary=summary,
        deliveries=deliveries,
    )
    result = {
        "ok": False,
        "error": "scheduled_window_missed",
        "run_id": run_id,
        "started_at": now,
        "completed_at": now,
        "scanned": 0,
        "analyzed_count": 0,
        "ranked_count": 0,
        "alert_count": 0,
        "filtered": [],
        "report": "",
        "telegram_sent": any(item.get("ok") for item in deliveries),
        "telegram_ready": bool(get_telegram_private_operator_chat_ids()),
        "telegram_delivery_status": summary["telegram_delivery_status"],
        "telegram_delivery": None,
        "delivery_audit": deliveries,
        "slot_label": slot_label,
        "scheduled_for": scheduled_for,
        "result_code": "missed_beyond_grace",
        "misfire_status": "missed_beyond_grace",
        "run_persisted": bool(persisted) if repository.enabled else None,
    }
    _status_update(
        previous_expected_run_at=scheduled_for,
        previous_actual_run_at=None,
        previous_run_status="skipped",
        previous_misfire_status="missed_beyond_grace",
        previous_durable_run_id=run_id,
        last_run={
            "run_id": run_id,
            "status": "skipped",
            "result_code": "missed_beyond_grace",
            "scheduled_for": scheduled_for,
            "delivery_status": summary["telegram_delivery_status"],
            "persisted": result["run_persisted"],
        },
    )
    return result


def _overlap_skipped_result(
    *,
    repository: SchedulerRunRepository,
    run_id: str,
    slot_label: str,
    scheduled_for: str,
) -> Dict[str, Any]:
    """Durably skip a busy hourly window without waiting or sending Telegram."""
    now = datetime.now(ZoneInfo("UTC")).isoformat()
    summary = {
        "result_code": "hourly_overlap_skipped",
        "misfire_status": "overlap_skipped",
        "symbols_analyzed": 0,
        "symbol_failures": 0,
        "eligible_candidates": 0,
        "revalidated_candidates": 0,
        "rejected_candidates": 0,
        "telegram_delivery_status": "skipped_scan_in_progress",
    }
    persisted = repository.finish(
        run_id=run_id,
        status="skipped",
        summary=summary,
        deliveries=[],
    )
    result = {
        "ok": False,
        "error": "hourly_scan_overlap",
        "run_id": run_id,
        "started_at": now,
        "completed_at": now,
        "scanned": 0,
        "analyzed_count": 0,
        "ranked_count": 0,
        "alert_count": 0,
        "filtered": [],
        "report": "",
        "telegram_sent": False,
        "telegram_ready": is_telegram_ready(),
        "telegram_delivery_status": "skipped_scan_in_progress",
        "telegram_delivery": None,
        "delivery_audit": [],
        "slot_label": slot_label,
        "scheduled_for": scheduled_for,
        "result_code": "hourly_overlap_skipped",
        "misfire_status": "overlap_skipped",
        "run_persisted": bool(persisted) if repository.enabled else None,
    }
    _status_update(
        previous_expected_run_at=scheduled_for,
        previous_actual_run_at=None,
        previous_run_status="skipped",
        previous_misfire_status="overlap_skipped",
        previous_durable_run_id=run_id,
        last_run={
            "run_id": run_id,
            "status": "skipped",
            "result_code": "hourly_overlap_skipped",
            "scheduled_for": scheduled_for,
            "delivery_status": "skipped_scan_in_progress",
            "persisted": result["run_persisted"],
        },
    )
    logger.info(
        "Private-beta hourly scan skipped because another scan is running: "
        "run={} scheduled_for={}",
        run_id,
        scheduled_for,
    )
    return result


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
    for t in times:
        try:
            h, m = _parse_hhmm(t)
        except (TypeError, ValueError):
            continue
        candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if candidate <= now:
            candidate = candidate + timedelta(days=1)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("No valid scheduler times are configured")
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


def next_scheduler_datetime(
    cfg: AppConfig,
    sessions: List[Dict[str, str]],
    times: List[str],
    *,
    now: Optional[datetime] = None,
) -> Tuple[datetime, str]:
    """Select the next named session or private-beta hourly window."""
    candidates: List[Tuple[datetime, str, int]] = []
    if sessions:
        session_at, session_name = next_session_datetime(sessions, now=now)
        candidates.append((session_at, session_name, 0))
    elif times:
        slot_at = next_slot_datetime(
            times,
            cfg.scheduler.timezone or "Africa/Lagos",
            now=now,
        )
        candidates.append((slot_at, "Scheduled scan", 0))
    if _private_beta_hourly_enabled(cfg):
        candidates.append(
            (next_hourly_datetime(now), PRIVATE_BETA_HOURLY_SESSION_NAME, 1)
        )
    if not candidates:
        raise ValueError("No scheduler windows are configured")
    selected = min(
        candidates,
        key=lambda item: (
            item[0].astimezone(ZoneInfo("UTC")),
            item[2],
        ),
    )
    return selected[0], selected[1]


def filter_high_confidence(
    ranked: List[Dict[str, Any]],
    *,
    min_llm: float,
    min_rank: float,
    only_prop_safe: bool,
    min_confidence: float = MIN_TELEGRAM_SIGNAL_CONFIDENCE,
    min_execution_score: float = 65.0,
    max_immediate_sl_risk: float = 32.0,
    max_chase_distance_atr: float = 1.0,
    max_pre_entry_tp1_progress_pct: float = 70.0,
    min_tp2_rr: float = 1.25,
    max_spread_bps: float = 12.0,
    max_ticker_age_seconds: float = 45.0,
    max_orderbook_age_seconds: float = 30.0,
) -> List[Dict[str, Any]]:
    """Apply deterministic delivery gates; ``min_llm`` is legacy API-only."""
    confidence_floor = max(
        MIN_TELEGRAM_SIGNAL_CONFIDENCE,
        float(min_confidence or 0),
    )
    out: List[Dict[str, Any]] = []
    for row in ranked or []:
        decision = evaluate_alert_gates(
            row,
            min_rank=min_rank,
            only_prop_safe=only_prop_safe,
            min_confidence=confidence_floor,
            min_execution_quality=min_execution_score,
            max_immediate_sl_risk=max_immediate_sl_risk,
            max_chase_distance_atr=max_chase_distance_atr,
            max_pre_entry_tp1_progress_pct=max_pre_entry_tp1_progress_pct,
            min_tp2_rr=min_tp2_rr,
            max_spread_bps=max_spread_bps,
            max_ticker_age_seconds=max_ticker_age_seconds,
            max_orderbook_age_seconds=max_orderbook_age_seconds,
            prior=row.get("gate_evaluation"),
        )
        row["gate_evaluation"] = decision.to_dict()
        if not decision.eligible:
            compatibility_codes = {
                "FLAT_DIRECTION": "NOT_DIRECTIONAL",
                "OVERALL_QUALITY_BELOW_MINIMUM": "CONFIDENCE_BELOW_ALERT_MINIMUM",
                "RANK_BELOW_MINIMUM": "RANK_BELOW_MINIMUM",
                "EXECUTION_QUALITY_BELOW_MINIMUM": "EXECUTION_BELOW_ALERT_MINIMUM",
                "IMMEDIATE_SL_RISK_TOO_HIGH": "IMMEDIATE_SL_RISK_HIGH",
                "PRICE_TOO_EXTENDED": "CHASE_DISTANCE_HIGH",
                "TP1_ALREADY_PROGRESSING": "ENTRY_MOVE_MOSTLY_MISSED",
                "SPREAD_TOO_WIDE": "SPREAD_TOO_WIDE",
                "MARKET_QUALITY_FAILED": "MARKET_QUALITY_BLOCKED",
                "DATA_QUALITY_FAILED": "DATA_QUALITY_BLOCKED",
                "HISTORICAL_EDGE_FAILED": "HISTORICAL_EDGE_BLOCKED",
                "GROSS_RR_BELOW_MINIMUM": "TP2_RR_BELOW_MINIMUM",
                "ENTRY_BLOCKED": "ENTRY_STATUS_BLOCKED",
                "AVOID_CHASE": "ENTRY_STATUS_BLOCKED",
                "ENTRY_EXPIRED": "ENTRY_STATUS_BLOCKED",
                "INVALIDATED_BEFORE_ENTRY": "ENTRY_STATUS_BLOCKED",
                "PROP_COMPATIBILITY_FAILED": "PROP_RISK_BLOCKED",
                "ANALYSIS_ERROR": "ANALYSIS_SIGNAL_GATE",
            }
            row["delivery_rejection_reasons"] = list(
                dict.fromkeys(
                    compatibility_codes.get(code, code)
                    for code in decision.all_rejection_reasons
                )
            )
            logger.info(
                "Telegram candidate rejected: symbol={} direction={} "
                "confidence={:.1f} execution={} reasons={}",
                row.get("symbol"),
                str(row.get("direction") or "").lower(),
                float(row.get("confidence") or 0),
                row.get("execution_score"),
                "|".join(row["delivery_rejection_reasons"]),
            )
            continue
        out.append(row)
    out.sort(
        key=lambda r: (
            float(r.get("confidence") or 0),
            float(r.get("rank_score") or 0),
            float(r.get("execution_score") or 0),
        ),
        reverse=True,
    )
    return out


def _filter_with_configured_public_policy(
    rows: List[Dict[str, Any]],
    cfg: AppConfig,
) -> List[Dict[str, Any]]:
    """Apply the unchanged public policy and populate canonical gate details."""
    return filter_high_confidence(
        rows,
        min_llm=float(cfg.telegram.min_llm_confidence or 65),
        min_rank=float(cfg.telegram.min_rank_score or 50),
        only_prop_safe=bool(cfg.scheduler.only_prop_safe),
        min_confidence=max(
            MIN_TELEGRAM_SIGNAL_CONFIDENCE,
            float(getattr(cfg.analysis, "directional_confidence_threshold", 68.0)),
        ),
        min_execution_score=float(
            getattr(cfg.analysis, "execution_min_score", 65.0)
        ),
        max_immediate_sl_risk=float(
            getattr(cfg.analysis, "max_immediate_sl_risk", 32.0)
        ),
        max_chase_distance_atr=float(
            getattr(cfg.analysis, "max_chase_distance_atr", 1.0)
        ),
        max_pre_entry_tp1_progress_pct=float(
            getattr(cfg.analysis, "max_pre_entry_tp1_progress_pct", 70.0)
        ),
        min_tp2_rr=float(getattr(cfg.analysis, "min_tp2_rr", 1.25)),
        max_spread_bps=float(getattr(cfg.analysis, "max_spread_bps", 12.0)),
        max_ticker_age_seconds=float(
            getattr(cfg.analysis, "max_ticker_age_seconds", 45.0)
        ),
        max_orderbook_age_seconds=float(
            getattr(cfg.analysis, "max_orderbook_age_seconds", 30.0)
        ),
    )


def suppress_recent_scheduled_signals(
    rows: List[Dict[str, Any]],
    *,
    now_monotonic: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Suppress nearly identical scheduled alerts while the first entry is live."""
    now = time.monotonic() if now_monotonic is None else now_monotonic
    kept: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    with _RECENT_SIGNAL_LOCK:
        stale = [
            key
            for key, value in _RECENT_SIGNAL_ALERTS.items()
            if value.get("expires_at", 0.0) <= now
        ]
        for key in stale:
            _RECENT_SIGNAL_ALERTS.pop(key, None)

        for row in rows:
            key = (
                str(row.get("symbol") or "").upper(),
                str(row.get("direction") or "").lower(),
            )
            previous = _RECENT_SIGNAL_ALERTS.get(key)
            entry = _row_entry_mid(row)
            stop = _safe_float_or_none(row.get("stop_loss"))
            same_levels = bool(
                previous
                and _within_bps(entry, previous.get("entry"), 8.0)
                and _within_bps(stop, previous.get("stop"), 12.0)
            )
            if same_levels:
                suppressed.append(row)
            else:
                kept.append(row)
    return kept, suppressed


def cap_signals_by_portfolio_risk(
    rows: List[Dict[str, Any]],
    *,
    max_open_risk_pct: float,
    default_risk_pct: float = 1.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep highest-ranked signals inside one conservative correlated-risk cap."""
    budget = max(0.1, float(max_open_risk_pct or 0.0))
    used = 0.0
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for row in rows:
        risk = _safe_float_or_none(row.get("risk_pct"))
        risk = risk if risk is not None else max(0.1, float(default_risk_pct or 1.0))
        if used + risk <= budget + 1e-9:
            kept.append(row)
            used += risk
        else:
            excluded.append(row)
    return kept, excluded


def remember_sent_scheduled_signals(
    rows: List[Dict[str, Any]],
    *,
    now_monotonic: Optional[float] = None,
) -> None:
    """Remember successfully delivered setups until their pending-entry expiry."""
    now = time.monotonic() if now_monotonic is None else now_monotonic
    with _RECENT_SIGNAL_LOCK:
        for row in rows:
            validity_minutes = max(
                15.0,
                _safe_float_or_none(row.get("entry_valid_for_minutes")) or 60.0,
            )
            key = (
                str(row.get("symbol") or "").upper(),
                str(row.get("direction") or "").lower(),
            )
            _RECENT_SIGNAL_ALERTS[key] = {
                "entry": _row_entry_mid(row) or 0.0,
                "stop": _safe_float_or_none(row.get("stop_loss")) or 0.0,
                "expires_at": now + validity_minutes * 60.0,
            }


def _row_entry_mid(row: Dict[str, Any]) -> Optional[float]:
    low = _safe_float_or_none(row.get("entry_low"))
    high = _safe_float_or_none(row.get("entry_high"))
    if low is None or high is None:
        return _safe_float_or_none(row.get("price"))
    return (low + high) / 2.0


def _safe_float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _within_bps(current: Optional[float], previous: Optional[float], bps: float) -> bool:
    if current is None or previous is None or previous <= 0:
        return False
    return abs(current - previous) / previous * 10_000.0 <= bps


def _row_tp2_rr(row: Dict[str, Any]) -> Optional[float]:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    primary = (
        payload.get("primary_setup")
        if isinstance(payload.get("primary_setup"), dict)
        else {}
    )
    risk_rewards = list(primary.get("risk_reward") or [])
    if len(risk_rewards) > 1:
        try:
            return float(risk_rewards[1])
        except (TypeError, ValueError):
            return None
    targets = list(row.get("take_profits") or [])
    if len(targets) < 2:
        return None
    try:
        entry = (float(row["entry_low"]) + float(row["entry_high"])) / 2.0
        stop = float(row["stop_loss"])
        target = float(targets[1])
    except (KeyError, TypeError, ValueError):
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    direction = str(row.get("direction") or "").lower()
    reward = target - entry if direction == "long" else entry - target
    return max(0.0, reward / risk)


def _run_scheduled_scan_once_unlocked(
    config: Optional[AppConfig] = None,
    *,
    slot_label: str = "",
    send: bool = True,
    symbols: Optional[List[str]] = None,
    timeframe: Optional[str] = None,
    notify_on_empty: Optional[bool] = None,
    telegram_chat_ids: Optional[List[str]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one watchlist scan and optionally Telegram high-conf results."""
    cfg = config or load_config()
    scan_started_monotonic = time.monotonic()
    run_id = str(run_id or _durable_run_id(slot_label, None))
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
        "Scheduled scan triggered: run={} slot={} symbols={} timeframe={} exchange={} send={}",
        run_id,
        slot_label or "manual",
        len(watchlist),
        req.timeframe,
        req.exchange,
        send,
    )
    slot_lower = str(slot_label or "").lower()
    manual_delivery = bool(
        telegram_chat_ids is not None
        or any(
            marker in slot_lower
            for marker in ("manual", "on-demand", "telegram")
        )
    )
    routing_status = get_delivery_status()
    private_beta_mode = routing_status["mode"] == "private_beta"
    requester_chat_id = (
        str((telegram_chat_ids or [""])[0]).strip()
        if telegram_chat_ids else None
    )
    scan_origin = (
        "manual_beta_scan"
        if private_beta_mode and manual_delivery
        else (
            "scheduled_beta_scan"
            if private_beta_mode and not manual_delivery
            else ("manual_scan" if manual_delivery else "scheduled_scan")
        )
    )
    requester_identity_hash = _safe_requester_hash(requester_chat_id)
    trigger_type = "telegram" if telegram_chat_ids is not None else (
        "manual" if manual_delivery else "scheduled"
    )
    result = scan_symbols(
        watchlist,
        request=req,
        config=cfg,
        scan_id=run_id,
        trigger_type=trigger_type,
    )
    ranked = result.get("ranked_results") or []
    qualification_candidates = list(
        result.get("qualification_candidates") or ranked
    )
    for row in [*qualification_candidates, *ranked]:
        row["scan_origin"] = scan_origin
        row["requester_identity_hash"] = requester_identity_hash
        evaluation = dict(row.get("gate_evaluation") or {})
        evaluation["delivery_context"] = {
            "scan_origin": scan_origin,
            "requester_identity_hash": requester_identity_hash,
        }
        row["gate_evaluation"] = evaluation
    if is_legacy_comparison():
        # Legacy comparison consumes the same directional candidates and gate
        # evidence, but only its isolated publication policy has authority.
        # The strict policy remains journaled as a shadow decision.
        filtered = qualify_legacy_candidates(
            qualification_candidates, limit=2
        )
        for row in qualification_candidates:
            row.setdefault("strict_shadow_decision", strict_shadow_decision(row))
            row.setdefault("legacy_decision", evaluate_legacy_qualification(row))
            row.update(identity_metadata())
    elif private_beta_mode:
        required_alert_codes = {
            "OVERALL_QUALITY_BELOW_MINIMUM",
            "EXECUTION_QUALITY_BELOW_MINIMUM",
            "RANK_BELOW_MINIMUM",
        }
        missing_gate_rows = []
        for row in qualification_candidates:
            existing_codes = {
                str(gate.get("code") or "")
                for gate in (row.get("gate_evaluation") or {}).get("gates") or []
            }
            if not required_alert_codes.issubset(existing_codes):
                missing_gate_rows.append(row)
        if missing_gate_rows:
            # Compatibility for stored/tests/older API producers: use the
            # existing alert evaluator solely to materialize canonical checks.
            _filter_with_configured_public_policy(missing_gate_rows, cfg)
        filtered = qualify_private_beta_candidates(
            qualification_candidates, limit=2
        )
    else:
        # Public-mode qualification remains byte-for-byte governed by the
        # existing deterministic alert filter.
        filtered = _filter_with_configured_public_policy(ranked, cfg)
    suppressed_duplicates: List[Dict[str, Any]] = []
    if send and not manual_delivery and filtered:
        filtered, suppressed_duplicates = suppress_recent_scheduled_signals(filtered)
        if suppressed_duplicates:
            logger.info(
                "Suppressed {} duplicate scheduled signal(s) still inside "
                "their entry-validity window",
                len(suppressed_duplicates),
            )
    if is_legacy_comparison():
        # Portfolio/prop routing is advisory in the Legacy experiment. The
        # same per-trade risk guidance remains visible in Telegram.
        portfolio_risk_excluded = []
    else:
        filtered, portfolio_risk_excluded = cap_signals_by_portfolio_risk(
            filtered,
            max_open_risk_pct=float(
                getattr(cfg.risk, "max_open_risk_pct", 2.0) or 2.0
            ),
            default_risk_pct=float(cfg.risk.risk_per_trade_pct or 1.0),
        )
    if portfolio_risk_excluded:
        logger.info(
            "Withheld {} lower-ranked signal(s) to keep total proposed open "
            "risk within {:.2f}%",
            len(portfolio_risk_excluded),
            float(getattr(cfg.risk, "max_open_risk_pct", 2.0) or 2.0),
        )
    pre_delivery_rejected: List[Dict[str, Any]] = []
    if send and filtered:
        accepted: List[Optional[Dict[str, Any]]] = [None] * len(filtered)
        with ThreadPoolExecutor(max_workers=min(4, len(filtered))) as pool:
            futures = {
                pool.submit(revalidate_candidate_for_delivery, row, cfg): index
                for index, row in enumerate(filtered)
            }
            for future in as_completed(futures):
                index = futures[future]
                original = filtered[index]
                try:
                    validation = future.result()
                except Exception as exc:  # noqa: BLE001
                    decision = evaluate_revalidation_gates(
                        [f"PRE_SEND_INTERNAL_FAILURE:{type(exc).__name__}"],
                        prior=original.get("gate_evaluation"),
                    )
                    validation = {
                        "ok": False,
                        "row": original,
                        "reasons": [f"PRE_SEND_INTERNAL_FAILURE:{type(exc).__name__}"],
                        "gate_evaluation": decision.to_dict(),
                    }
                if validation.get("ok"):
                    refreshed = dict(validation.get("row") or original)
                    if is_legacy_comparison():
                        refreshed["pre_delivery_revalidation_ok"] = True
                        decision = evaluate_legacy_qualification(refreshed)
                        refreshed["legacy_decision"] = decision
                        refreshed["strict_shadow_decision"] = strict_shadow_decision(
                            refreshed
                        )
                        refreshed["quality_tier"] = decision.get("quality_tier")
                        refreshed["caveats"] = list(decision.get("caveats") or [])
                        refreshed.update(identity_metadata())
                        if not decision["qualified"]:
                            refreshed["pre_delivery_rejection_reasons"] = [
                                "LEGACY_QUALIFICATION_FAILED"
                            ]
                            pre_delivery_rejected.append(refreshed)
                            continue
                    elif private_beta_mode:
                        refreshed["pre_delivery_revalidation_ok"] = True
                        qualification = evaluate_private_beta_qualification(
                            refreshed, refreshed.get("gate_evaluation")
                        )
                        refreshed["qualification"] = qualification
                        refreshed["qualification_policy_version"] = qualification[
                            "qualification_policy_version"
                        ]
                        refreshed["gate_evaluation"][
                            "private_beta_qualification"
                        ] = qualification
                        refreshed.setdefault("payload", {})[
                            "qualification"
                        ] = qualification
                        if not qualification["private_beta_qualified"]:
                            refreshed["pre_delivery_rejection_reasons"] = [
                                "PRIVATE_BETA_QUALIFICATION_FAILED"
                            ]
                            pre_delivery_rejected.append(refreshed)
                            continue
                    accepted[index] = refreshed
                else:
                    # Preserve the pre-existing public-mode rejection payload.
                    # Private beta needs the refreshed row so its final
                    # revalidation check can be included in the deduplicated
                    # qualification record.
                    rejected = dict(
                        (
                            validation.get("row")
                            if private_beta_mode
                            else original
                        )
                        or original
                    )
                    rejected["pre_delivery_rejection_reasons"] = list(
                        validation.get("reasons") or ["PRE_SEND_REJECTED"]
                    )
                    rejected["gate_evaluation"] = dict(
                        validation.get("gate_evaluation")
                        or rejected.get("gate_evaluation")
                        or {}
                    )
                    if is_legacy_comparison():
                        rejected["pre_delivery_revalidation_ok"] = False
                        rejected["legacy_decision"] = (
                            evaluate_legacy_qualification(rejected)
                        )
                        rejected["strict_shadow_decision"] = (
                            strict_shadow_decision(rejected)
                        )
                        rejected.update(identity_metadata())
                    elif private_beta_mode:
                        rejected["pre_delivery_revalidation_ok"] = False
                        qualification = evaluate_private_beta_qualification(
                            rejected, rejected.get("gate_evaluation")
                        )
                        rejected["qualification"] = qualification
                        rejected["qualification_policy_version"] = qualification[
                            "qualification_policy_version"
                        ]
                        rejected["gate_evaluation"][
                            "private_beta_qualification"
                        ] = qualification
                    pre_delivery_rejected.append(rejected)
                    logger.warning(
                        "Telegram candidate failed final revalidation: symbol={} reasons={}",
                        original.get("symbol"),
                        "|".join(rejected["pre_delivery_rejection_reasons"]),
                    )
        filtered = [row for row in accepted if row is not None]

    scan_failed = not bool(result.get("ok"))
    report_summary = dict(result.get("rejection_analytics") or {})
    if is_legacy_comparison():
        report_summary.update(
            {
                "bot_variant": "legacy",
                "bot_namespace": bot_namespace(),
                "experiment_id": identity_metadata()["experiment_id"],
                "qualification_policy_version": LEGACY_POLICY_VERSION,
                "strict_shadow_signals": sum(
                    1
                    for row in qualification_candidates
                    if (row.get("strict_shadow_decision") or {}).get("decision")
                    == "SIGNAL"
                ),
                "legacy_signals": len(filtered),
            }
        )
    elif private_beta_mode:
        latest_beta_rows: Dict[str, Dict[str, Any]] = {
            str(row.get("candidate_id") or index): row
            for index, row in enumerate(qualification_candidates)
        }
        for row in [*pre_delivery_rejected, *filtered]:
            latest_beta_rows[str(row.get("candidate_id") or id(row))] = row
        report_summary = build_private_beta_rejection_summary(
            report_summary, latest_beta_rows.values()
        )
    report_summary.update(
        {
            "scan_duration_seconds": round(
                time.monotonic() - scan_started_monotonic, 3
            ),
            "accepted_candidates": len(filtered),
            "revalidated_candidates": len(filtered),
            "pre_delivery_rejected_count": len(pre_delivery_rejected),
        }
    )
    if scan_failed:
        report = (
            ("🧪 <b>PERPETUAL PRO LEGACY</b>\n" if is_legacy_comparison() else "")
            + "⚠️ <b>SCAN UNAVAILABLE</b>\n\n"
            "Market data or analysis failed for the complete watchlist. "
            "No trade decision was produced; this is not a no-setup result.\n\n"
            "NFA · DYOR · Trade at your own risk"
        )
    else:
        report = format_prop_scan_report(
            filtered,
            slot_label=slot_label or "scan",
            timezone=cfg.scheduler.timezone or "Africa/Lagos",
            min_signal_confidence=max(
                MIN_TELEGRAM_SIGNAL_CONFIDENCE,
                float(getattr(cfg.analysis, "directional_confidence_threshold", 68.0)),
            ),
            scanned_count=len(watchlist),
            ranked_count=len(ranked),
            rejection_summary=report_summary,
        )
    sent = False
    delivery: Optional[Dict[str, Any]] = None
    delivery_audit: List[Dict[str, Any]] = []
    tracking: Optional[Dict[str, Any]] = None
    delivery_status = "not_requested"
    # In private beta, a qualified signal found by either beta user is shared
    # with the complete configured beta cohort. Empty reports remain exclusive
    # to the requester; failures add only the private operator diagnostics
    # destinations. Public-mode routing is intentionally unchanged.
    if private_beta_mode and manual_delivery:
        destinations = _merge_destinations(
            get_private_beta_chat_ids(),
            list(telegram_chat_ids or []),
        )
        report_destinations = _merge_destinations(
            list(telegram_chat_ids or [])
        )
        failure_report_destinations = _merge_destinations(
            list(telegram_chat_ids or []),
            get_telegram_private_operator_chat_ids(),
        )
    else:
        destinations = get_telegram_alert_chat_ids(telegram_chat_ids)
        report_destinations = get_telegram_report_chat_ids(
            telegram_chat_ids if manual_delivery else None
        )
        failure_report_destinations = (
            get_telegram_private_operator_chat_ids()
            if not manual_delivery
            else report_destinations
        )
    if is_legacy_comparison() and not manual_delivery and not scan_failed:
        # Legacy scheduled no-setup reports are intentionally silent.
        report_destinations = []
    logger.info(
        "Delivery Mode: {} recipients_attempted={} report_recipients={} "
        "public_enabled={}",
        routing_status["mode_label"],
        len(destinations),
        len(report_destinations),
        routing_status["public_delivery_enabled"],
    )
    # Credentials from env only (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — never YAML
    tg_ready = is_telegram_ready(cfg) and bool(
        destinations or report_destinations
    )
    if send and tg_ready:
        if filtered:
            delivery_repository = SchedulerRunRepository(
                cfg.outcome_scoring.database_url
            )
            rendered: List[Dict[str, Any]] = []
            tracking_destinations: List[List[str]] = [
                [] for _ in range(len(filtered[:6]))
            ]
            # Render each chart once, then fan the immutable payload out.
            for index, row in enumerate(filtered[:6], 1):
                symbol = str(row.get("symbol") or f"signal-{index}")
                try:
                    rendered.append(
                        {
                            "ok": True,
                            "symbol": symbol,
                            "photo": render_signal_chart_png(row),
                            "caption": format_signal_photo_caption(
                                row,
                                slot_label=slot_label or "scan",
                            ),
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Telegram chart alert failed before upload: symbol={} error_type={}",
                        symbol,
                        type(exc).__name__,
                    )
                    rendered.append(
                        {
                            "ok": False,
                            "symbol": symbol,
                            "error": "chart_render_failed",
                            "description": type(exc).__name__,
                        }
                    )

            destination_results: List[Dict[str, Any]] = []
            chart_items: List[Dict[str, Any]] = []
            fallback_items: List[Dict[str, Any]] = []
            for destination in destinations:
                destination_items: List[Dict[str, Any]] = []
                for signal_index, signal in enumerate(rendered):
                    symbol = str(signal.get("symbol") or "signal")
                    signal_row = filtered[signal_index]
                    signal_delivery_key = _signal_delivery_idempotency_key(
                        signal_row,
                        destination,
                    )
                    if not signal.get("ok"):
                        item = {
                            "ok": False,
                            "symbol": symbol,
                            "mode": "photo",
                            "error": signal.get("error"),
                            "description": signal.get("description"),
                        }
                    elif (
                        callable(getattr(delivery_repository, "delivery_succeeded", None))
                        and delivery_repository.delivery_succeeded(
                            signal_delivery_key
                        )
                    ):
                        item = {
                            "ok": True,
                            "symbol": symbol,
                            "mode": "photo",
                            "skipped_duplicate": True,
                            "description": "already delivered to this destination",
                        }
                    else:
                        item = send_telegram_photo_detailed(
                            signal["photo"],
                            signal["caption"],
                            filename=f"{symbol.split('/')[0].lower()}-signal.png",
                            chat_id=destination,
                            parse_mode=cfg.telegram.parse_mode or "HTML",
                        )
                        item["symbol"] = symbol
                        item["mode"] = "photo"
                        digest = destination_hash(destination)
                        delivery_audit.append(
                            {
                                "idempotency_key": signal_delivery_key,
                                "destination_hash": digest,
                                "delivery_type": f"signal_photo:{symbol}",
                                "ok": bool(item.get("ok")),
                                "message_id": item.get("message_id"),
                                "error": item.get("error"),
                            }
                        )
                        if callable(
                            getattr(delivery_repository, "record_delivery", None)
                        ):
                            delivery_repository.record_delivery(
                                run_id=run_id,
                                delivery=delivery_audit[-1],
                            )
                    destination_items.append(item)
                    chart_items.append(item)
                    if item.get("ok") and not item.get("skipped_duplicate"):
                        tracking_destinations[signal_index].append(destination)

                destination_photo_ok = bool(destination_items) and all(
                    item.get("ok") for item in destination_items
                )
                fallback: Optional[Dict[str, Any]] = None
                if not destination_photo_ok:
                    fallback = send_telegram_message_detailed(
                        report,
                        chat_id=destination,
                        parse_mode=cfg.telegram.parse_mode or "HTML",
                    )
                    fallback_items.append(fallback)
                    digest = destination_hash(destination)
                    delivery_audit.append(
                        {
                            "idempotency_key": f"{run_id}:{digest}:signal_text_fallback",
                            "destination_hash": digest,
                            "delivery_type": "signal_text_fallback",
                            "ok": bool(fallback.get("ok")),
                            "message_id": fallback.get("message_id"),
                            "error": fallback.get("error"),
                        }
                    )
                    if fallback.get("ok"):
                        for signal_index, signal_destinations in enumerate(
                            tracking_destinations
                        ):
                            if destination not in signal_destinations:
                                signal_destinations.append(destination)
                            signal_row = filtered[signal_index]
                            stable_key = _signal_delivery_idempotency_key(
                                signal_row,
                                destination,
                            )
                            stable_delivery = {
                                "idempotency_key": stable_key,
                                "destination_hash": digest,
                                "delivery_type": (
                                    "signal_text_fallback:"
                                    + str(signal_row.get("symbol") or signal_index)
                                ),
                                "ok": True,
                                "message_id": fallback.get("message_id"),
                                "error": None,
                            }
                            delivery_audit.append(stable_delivery)
                            if callable(
                                getattr(
                                    delivery_repository,
                                    "record_delivery",
                                    None,
                                )
                            ):
                                delivery_repository.record_delivery(
                                    run_id=run_id,
                                    delivery=stable_delivery,
                                )
                destination_ok = destination_photo_ok or bool(
                    fallback and fallback.get("ok")
                )
                destination_results.append(
                    {
                        "ok": destination_ok,
                        "photo_ok": destination_photo_ok,
                        "items": destination_items,
                        "text_fallback": fallback,
                    }
                )

            chart_sent = sum(
                1 for item in chart_items
                if item.get("ok") and not item.get("skipped_duplicate")
            )
            duplicate_skipped = sum(
                1 for item in chart_items if item.get("skipped_duplicate")
            )
            chart_failed = len(chart_items) - chart_sent - duplicate_skipped
            all_destinations_ok = bool(destination_results) and all(
                item.get("ok") for item in destination_results
            )
            photo_ok = bool(destination_results) and all(
                item.get("photo_ok") for item in destination_results
            )
            delivery = {
                "ok": all_destinations_ok,
                "mode": "photo",
                "sent_count": chart_sent,
                "failed_count": chart_failed,
                "total_actionable": len(filtered),
                "destination_count": len(destinations),
                "destinations": destination_results,
                "items": chart_items,
            }
            if len(fallback_items) == 1:
                delivery["text_fallback"] = fallback_items[0]
            if photo_ok and chart_sent:
                sent = True
                delivery_status = "sent_chart_alerts"
                logger.info(
                    "Scheduled Telegram chart alerts succeeded: slot={} sent={} "
                    "destinations={}",
                    slot_label or "scan",
                    chart_sent,
                    len(destinations),
                )
            elif photo_ok and duplicate_skipped == len(chart_items):
                sent = False
                delivery_status = "skipped_duplicate_signals"
                logger.info(
                    "Telegram signal delivery skipped: all {} per-destination "
                    "deliveries were already confirmed",
                    duplicate_skipped,
                )
            else:
                sent = any(item.get("ok") for item in destination_results)
                delivery_status = (
                    "sent_with_text_fallback"
                    if sent and all_destinations_ok
                    else ("partial_delivery" if sent else "failed")
                )
                if sent:
                    logger.warning(
                        "Telegram chart delivery incomplete: slot={} photos={}/{} "
                        "destinations_ok={}/{}",
                        slot_label or "scan",
                        chart_sent,
                        len(chart_items),
                        sum(1 for item in destination_results if item.get("ok")),
                        len(destinations),
                    )
                else:
                    logger.error(
                        "Telegram chart and text fallback both failed: slot={}",
                        slot_label or "scan",
                    )
            if sent and not manual_delivery:
                remember_sent_scheduled_signals(filtered)
            if sent:
                tracking = register_delivered_signals(
                    filtered[:6],
                    tracking_destinations,
                    source=(
                        f"legacy_{scan_origin}"
                        if is_legacy_comparison()
                        else scan_origin
                        if private_beta_mode
                        else (
                            "telegram_manual"
                            if manual_delivery
                            else "telegram_scheduled"
                        )
                    ),
                    config=cfg,
                )
                if not tracking.get("ok"):
                    logger.error(
                        "Telegram alerts were delivered but tracker registration "
                        "was incomplete: errors={}",
                        tracking.get("errors") or tracking.get("error"),
                    )
            logger.info(
                "Telegram signal delivery summary: mode={} attempted={} "
                "successful={} failed={}",
                routing_status["mode_label"],
                len(destinations),
                sum(1 for item in destination_results if item.get("ok")),
                sum(1 for item in destination_results if not item.get("ok")),
            )
        elif suppressed_duplicates:
            delivery_status = "skipped_duplicate_signals"
            logger.info(
                "Scheduled Telegram delivery skipped: all {} actionable "
                "signal(s) were already sent and remain valid",
                len(suppressed_duplicates),
            )
        elif (
            scan_failed
            or (
                False
                if is_legacy_comparison() and not manual_delivery and not scan_failed
                else bool(notify_on_empty)
                if notify_on_empty is not None
                else (
                    routing_status["mode"] == "private_beta"
                    or cfg.telegram.notify_on_empty
                )
            )
        ):
            destination_results = []
            report_type = "scan_failure" if scan_failed else "no_quality_report"
            effective_report_destinations = (
                failure_report_destinations if scan_failed else report_destinations
            )
            for destination in effective_report_destinations:
                item = send_telegram_message_detailed(
                    report,
                    chat_id=destination,
                    parse_mode=cfg.telegram.parse_mode or "HTML",
                )
                destination_results.append(item)
                digest = destination_hash(destination)
                delivery_audit.append(
                    {
                        "idempotency_key": f"{run_id}:{digest}:{report_type}",
                        "destination_hash": digest,
                        "delivery_type": report_type,
                        "ok": bool(item.get("ok")),
                        "message_id": item.get("message_id"),
                        "error": item.get("error"),
                    }
                )
            sent_count = sum(1 for item in destination_results if item.get("ok"))
            sent = sent_count > 0
            delivery = (
                destination_results[0]
                if len(destination_results) == 1
                else {
                    "ok": sent_count == len(destination_results),
                    "sent_count": sent_count,
                    "destination_count": len(destination_results),
                    "destinations": destination_results,
                }
            )
            if sent:
                delivery_status = (
                    ("sent_scan_failure" if scan_failed else "sent_empty_report")
                    if sent_count == len(destination_results)
                    else "partial_delivery"
                )
                logger.info(
                    "Scheduled empty Telegram report succeeded: slot={} "
                    "destinations={}/{}",
                    slot_label or "scan",
                    sent_count,
                    len(destination_results),
                )
            elif effective_report_destinations:
                delivery_status = "failed"
                logger.error(
                    "Scheduled empty Telegram report failed: slot={} error={} description={}",
                    slot_label or "scan",
                    delivery.get("error"),
                    delivery.get("description"),
                )
            else:
                delivery_status = (
                    "skipped_private_scan_failure_no_operator"
                    if scan_failed
                    else "skipped_public_empty_report"
                )
                logger.info(
                    "Scheduled report persisted without Telegram delivery: "
                    "slot={} result={} public_empty_suppressed=true "
                    "private_operator_configured=false",
                    slot_label or "scan",
                    "scan_unavailable" if scan_failed else "no_quality_setup",
                )
            logger.info(
                "Telegram report delivery summary: mode={} attempted={} "
                "successful={} failed={}",
                routing_status["mode_label"],
                len(effective_report_destinations),
                sent_count,
                max(0, len(effective_report_destinations) - sent_count),
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
            "Set TELEGRAM_BOT_TOKEN and recipients for the active delivery mode."
        )
    elif not send:
        delivery_status = "disabled_for_run"
    completed_at = datetime.now(ZoneInfo("UTC")).isoformat()
    scan_duration_seconds = round(time.monotonic() - scan_started_monotonic, 3)
    ticker_ages = [
        value
        for value in (_safe_float_or_none(row.get("ticker_age_seconds")) for row in ranked)
        if value is not None
    ]
    orderbook_ages = [
        value
        for value in (_safe_float_or_none(row.get("orderbook_age_seconds")) for row in ranked)
        if value is not None
    ]
    rejection_counts: Dict[str, int] = {}
    llm_counts: Dict[str, int] = {}
    for row in ranked:
        status = str(row.get("llm_invocation_status") or "unknown")
        llm_counts[status] = llm_counts.get(status, 0) + 1
        reasons = list(row.get("rejection_reasons") or []) + list(
            row.get("delivery_rejection_reasons") or []
        )
        for reason in reasons:
            code = str(reason.get("code") if isinstance(reason, dict) else reason)
            if code:
                rejection_counts[code] = rejection_counts.get(code, 0) + 1
    result_code = (
        "scan_unavailable"
        if scan_failed
        else ("eligible_signals" if filtered else "no_quality_setup")
    )
    public_empty_suppressed = bool(
        not manual_delivery and not filtered and not scan_failed
    )
    successful_delivery_events = sum(
        1 for item in delivery_audit if item.get("ok")
    )
    public_messages = (
        successful_delivery_events
        if (
            not manual_delivery
            and filtered
            and routing_status["public_delivery_enabled"]
        )
        else 0
    )
    private_messages = successful_delivery_events - public_messages
    candidate_updates: List[Dict[str, Any]] = []
    latest_by_id: Dict[str, Dict[str, Any]] = {
        str(row.get("candidate_id")): row
        for row in [*ranked, *pre_delivery_rejected, *filtered]
        if row.get("candidate_id")
    }
    for candidate_id, row in latest_by_id.items():
        if row.get("gate_evaluation"):
            candidate_updates.append(
                {
                    "candidate_id": candidate_id,
                    "gate_evaluation": row.get("gate_evaluation"),
                }
            )
    analytics_finalized = get_rejection_repository(cfg).finalize_scan(
        scan_id=run_id,
        revalidated_candidates=len(filtered),
        public_messages=public_messages,
        private_messages=private_messages,
        no_quality_result=not bool(filtered),
        summary_patch={
            "result_code": result_code,
            "delivery_status": delivery_status,
            "pre_delivery_rejected_count": len(pre_delivery_rejected),
            "public_empty_suppressed": public_empty_suppressed,
            "scan_origin": scan_origin,
            "requester_identity_hash": requester_identity_hash,
            "delivery_destination_hashes": sorted(
                {
                    str(item.get("destination_hash"))
                    for item in delivery_audit
                    if item.get("destination_hash")
                }
            ),
            "delivery_results": [
                {
                    "destination_hash": item.get("destination_hash"),
                    "delivery_type": item.get("delivery_type"),
                    "ok": bool(item.get("ok")),
                }
                for item in delivery_audit
            ],
            "fully_qualified_signals": sum(
                1
                for row in filtered
                if (row.get("qualification") or {}).get("qualification_type")
                == "fully_qualified"
            ),
            "relaxed_private_beta_signals": sum(
                1
                for row in filtered
                if (row.get("qualification") or {}).get("qualification_type")
                == "qualified_beta"
            ),
        },
        candidate_updates=candidate_updates,
    )
    if not analytics_finalized and get_rejection_repository(cfg).enabled:
        logger.error(
            "Rejection analytics finalization failed for run={}; delivery and "
            "eligibility remain unchanged",
            run_id,
        )
    logger.info(
        "Scheduled scan completed: run={} slot={} scanned={} ranked={} actionable={} "
        "delivery_status={} duration_seconds={:.3f}",
        run_id,
        slot_label or "scan",
        len(watchlist),
        len(ranked),
        len(filtered),
        delivery_status,
        scan_duration_seconds,
    )
    return {
        "ok": bool(result.get("ok")),
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "scanned": len(watchlist),
        "symbols_requested": list(watchlist),
        "analyzed_count": int(result.get("analyzed_count") or len(ranked)),
        "analysis_failure_count": len(result.get("analysis_failures") or []),
        "ranked_count": len(ranked),
        "alert_count": len(filtered),
        "duplicate_signal_count": len(suppressed_duplicates),
        "portfolio_risk_excluded_count": len(portfolio_risk_excluded),
        "pre_delivery_rejected_count": len(pre_delivery_rejected),
        "pre_delivery_rejected": pre_delivery_rejected,
        "rejected_count": max(0, len(ranked) - len(filtered)),
        "filtered": filtered,
        "report": report,
        "telegram_sent": sent,
        "telegram_ready": tg_ready,
        "telegram_delivery_status": delivery_status,
        "telegram_delivery": delivery,
        "signal_tracking": tracking,
        "delivery_audit": delivery_audit,
        "result_code": result_code,
        "scan_duration_seconds": scan_duration_seconds,
        "max_ticker_age_seconds": max(ticker_ages) if ticker_ages else None,
        "max_orderbook_age_seconds": max(orderbook_ages) if orderbook_ages else None,
        "main_rejection_reasons": rejection_counts,
        "llm_invocation_counts": llm_counts,
        "public_empty_suppressed": public_empty_suppressed,
        "delivery_mode": routing_status["mode"],
        "delivery_recipient_count": len(destinations),
        "scan_origin": scan_origin,
        "requester_identity_hash": requester_identity_hash,
        "rejection_analytics": result.get("rejection_analytics") or {},
        "rejection_analytics_finalized": analytics_finalized,
        "slot_label": slot_label,
        **identity_metadata(),
        "qualification_policy_version": (
            LEGACY_POLICY_VERSION
            if is_legacy_comparison()
            else None
        ),
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
    telegram_chat_ids: Optional[List[str]] = None,
    scheduled_for: Optional[str] = None,
    run_id: Optional[str] = None,
    skip_if_busy: bool = False,
    warn_on_misfire: bool = True,
) -> Dict[str, Any]:
    """Run one scan without overlapping another scheduler, API, or bot request."""
    cfg = config or load_config()
    started_at = datetime.now(ZoneInfo("UTC")).isoformat()
    resolved_run_id = str(run_id or _durable_run_id(slot_label, scheduled_for))
    source = _scheduler_run_source(slot_label, telegram_chat_ids, scheduled_for)
    watchlist = list(symbols or cfg.scheduler.watchlist or []) or list(
        DEFAULT_WATCHLIST
    )
    repository = SchedulerRunRepository(cfg.outcome_scoring.database_url)
    claimed: Optional[bool] = None
    lock_acquired = False

    if source == "scheduled":
        # Claim the expected window before process-lock arbitration. Previously
        # an overlapping manual scan returned before this claim, leaving no
        # durable evidence that the production window had been missed.
        claimed = repository.claim(
            run_id=resolved_run_id,
            source=source,
            slot_label=slot_label or "scan",
            scheduled_for=scheduled_for,
            symbols_requested=len(watchlist),
        )
        if claimed is False:
            logger.warning(
                "Duplicate durable scheduler run suppressed: run={} slot={}",
                resolved_run_id,
                slot_label or "scan",
            )
            return {
                "ok": False,
                "error": "duplicate_scheduler_run",
                "run_id": resolved_run_id,
                "started_at": started_at,
                "completed_at": started_at,
                "scanned": 0,
                "ranked_count": 0,
                "alert_count": 0,
                "filtered": [],
                "report": "",
                "telegram_sent": False,
                "telegram_ready": is_telegram_ready(cfg),
                "telegram_delivery_status": "duplicate_run_suppressed",
                "telegram_delivery": None,
                "slot_label": slot_label,
            }
        if repository.enabled and claimed is None:
            logger.error(
                "Scheduled run aborted because durable run claim is unavailable: run={}",
                resolved_run_id,
            )
            return {
                "ok": False,
                "error": "scheduler_run_persistence_unavailable",
                "run_id": resolved_run_id,
                "started_at": started_at,
                "completed_at": started_at,
                "scanned": 0,
                "ranked_count": 0,
                "alert_count": 0,
                "filtered": [],
                "report": "",
                "telegram_sent": False,
                "telegram_ready": is_telegram_ready(cfg),
                "telegram_delivery_status": "persistence_unavailable",
                "telegram_delivery": None,
                "slot_label": slot_label,
            }
        expected_at = _parse_utc_datetime(scheduled_for)
        grace_deadline = (
            expected_at + timedelta(seconds=SCHEDULER_MISFIRE_GRACE_SECONDS)
            if expected_at is not None else None
        )
        remaining_grace = (
            (grace_deadline - datetime.now(ZoneInfo("UTC"))).total_seconds()
            if grace_deadline is not None else float(SCHEDULER_MISFIRE_GRACE_SECONDS)
        )
        if remaining_grace <= 0:
            return _missed_window_result(
                repository=repository,
                run_id=resolved_run_id,
                slot_label=slot_label,
                scheduled_for=str(scheduled_for or ""),
                reason=(
                    f"misfire grace of {SCHEDULER_MISFIRE_GRACE_SECONDS}s expired"
                ),
                warn_operator=warn_on_misfire,
            )
        lock_acquired = _SCAN_RUN_LOCK.acquire(
            blocking=False
        ) if skip_if_busy else _SCAN_RUN_LOCK.acquire(
            timeout=max(0.0, remaining_grace)
        )
        if not lock_acquired:
            if skip_if_busy:
                return _overlap_skipped_result(
                    repository=repository,
                    run_id=resolved_run_id,
                    slot_label=slot_label,
                    scheduled_for=str(scheduled_for or ""),
                )
            return _missed_window_result(
                repository=repository,
                run_id=resolved_run_id,
                slot_label=slot_label,
                scheduled_for=str(scheduled_for or ""),
                reason=(
                    "another scan occupied the worker through the documented "
                    f"{SCHEDULER_MISFIRE_GRACE_SECONDS}s grace period"
                ),
                warn_operator=warn_on_misfire,
            )
    else:
        lock_acquired = _SCAN_RUN_LOCK.acquire(blocking=False)
        if not lock_acquired:
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
        if source != "scheduled":
            claimed = repository.claim(
                run_id=resolved_run_id,
                source=source,
                slot_label=slot_label or "scan",
                scheduled_for=scheduled_for,
                symbols_requested=len(watchlist),
            )
        if claimed is False:
            logger.warning(
                "Duplicate durable scheduler run suppressed: run={} slot={}",
                resolved_run_id,
                slot_label or "scan",
            )
            return {
                "ok": False,
                "error": "duplicate_scheduler_run",
                "run_id": resolved_run_id,
                "started_at": started_at,
                "completed_at": started_at,
                "scanned": 0,
                "ranked_count": 0,
                "alert_count": 0,
                "filtered": [],
                "report": "",
                "telegram_sent": False,
                "telegram_ready": is_telegram_ready(cfg),
                "telegram_delivery_status": "duplicate_run_suppressed",
                "telegram_delivery": None,
                "slot_label": slot_label,
            }
        try:
            outcome = _run_scheduled_scan_once_unlocked(
                cfg,
                slot_label=slot_label,
                send=send,
                symbols=symbols,
                timeframe=timeframe,
                notify_on_empty=notify_on_empty,
                telegram_chat_ids=telegram_chat_ids,
                run_id=resolved_run_id,
            )
        except Exception:
            repository.finish(
                run_id=resolved_run_id,
                status="failed",
                summary={"result_code": "unhandled_exception"},
            )
            raise
        persisted = repository.finish(
            run_id=resolved_run_id,
            status="completed" if outcome.get("ok") else "failed",
            summary=_compact_run_summary(outcome),
            deliveries=list(outcome.get("delivery_audit") or []),
        )
        outcome["run_persisted"] = bool(persisted) if repository.enabled else None
        last_run = {
            "run_id": resolved_run_id,
            "status": "completed" if outcome.get("ok") else "failed",
            "result_code": outcome.get("result_code"),
            "started_at": outcome.get("started_at"),
            "completed_at": outcome.get("completed_at"),
            "symbols_requested": outcome.get("scanned"),
            "symbols_analyzed": outcome.get("analyzed_count"),
            "alert_count": outcome.get("alert_count"),
            "delivery_status": outcome.get("telegram_delivery_status"),
            "persisted": outcome.get("run_persisted"),
        }
        _status_update(last_run=last_run)
        if source == "scheduled":
            actual_at = _parse_utc_datetime(outcome.get("started_at"))
            expected_at = _parse_utc_datetime(scheduled_for)
            delay_seconds = (
                max(0.0, (actual_at - expected_at).total_seconds())
                if actual_at is not None and expected_at is not None else 0.0
            )
            _status_update(
                previous_expected_run_at=scheduled_for,
                previous_actual_run_at=outcome.get("started_at"),
                previous_run_status=last_run["status"],
                previous_misfire_status=(
                    "recovered_within_grace"
                    if delay_seconds > 1.0
                    else "on_time"
                ),
                previous_durable_run_id=resolved_run_id,
            )
        return outcome
    finally:
        if lock_acquired:
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
    times = list(cfg.scheduler.times or [])
    sessions = list(getattr(cfg.scheduler, "sessions", None) or [])
    tz_name = cfg.scheduler.timezone or "Africa/Lagos"
    _status_update(
        enabled=bool(cfg.scheduler.enabled),
        running=True,
        timezone=tz_name,
        times=times,
        sessions=sessions,
        active_windows=_active_scheduler_windows(cfg, sessions, times),
        private_beta_hourly_enabled=_private_beta_hourly_enabled(cfg),
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
        "Scheduler loop started: times={} timezone={} hourly_private_beta={} "
        "high-confidence-only=true",
        (
            [
                f"{session.get('name')} {session.get('time')} {session.get('timezone')}"
                for session in sessions
            ]
            if sessions
            else times
        ),
        tz_name,
        _private_beta_hourly_enabled(cfg),
    )
    try:
        if sessions or _private_beta_hourly_enabled(cfg):
            repository = SchedulerRunRepository(
                cfg.outcome_scoring.database_url
            )
            recovered = _recover_expected_scheduler_windows(cfg, repository)
            if recovered:
                logger.info(
                    "Scheduler startup recovery processed {} missing window(s)",
                    len(recovered),
                )
        while not stop.is_set():
            nxt, session_name = next_scheduler_datetime(
                cfg, sessions, times
            )
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
            label = _scheduled_slot_label(
                session_name,
                nxt,
                tz_name,
                sessions,
            )
            triggered_at = datetime.now(ZoneInfo("UTC")).isoformat()
            scheduled_for = nxt.astimezone(ZoneInfo("UTC")).isoformat()
            hourly_window = session_name == PRIVATE_BETA_HOURLY_SESSION_NAME
            _status_update(
                last_triggered_at=triggered_at,
                last_error=None,
                previous_expected_run_at=scheduled_for,
            )
            try:
                outcome = run_scheduled_scan_once(
                    cfg,
                    slot_label=label,
                    send=True,
                    scheduled_for=scheduled_for,
                    notify_on_empty=False if hourly_window else None,
                    skip_if_busy=hourly_window,
                    warn_on_misfire=not hourly_window,
                )
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
    repository = SchedulerRunRepository(cfg.outcome_scoring.database_url)
    latest_run = repository.latest()
    if latest_run:
        _status_update(last_run=latest_run)
    latest_scheduled = repository.latest_scheduled()
    if latest_scheduled:
        summary = dict(latest_scheduled.get("result_summary") or {})
        missed_window = bool(
            latest_scheduled.get("status") == "skipped"
            or summary.get("misfire_status") == "missed_beyond_grace"
            or summary.get("result_code") == "missed_beyond_grace"
        )
        _status_update(
            previous_expected_run_at=(
                _parse_utc_datetime(latest_scheduled.get("scheduled_for")).isoformat()
                if _parse_utc_datetime(latest_scheduled.get("scheduled_for"))
                else latest_scheduled.get("scheduled_for")
            ),
            previous_actual_run_at=(
                None
                if missed_window
                else (
                    _parse_utc_datetime(
                        latest_scheduled.get("started_at")
                    ).isoformat()
                    if _parse_utc_datetime(latest_scheduled.get("started_at"))
                    else latest_scheduled.get("started_at")
                )
            ),
            previous_run_status=latest_scheduled.get("status"),
            previous_misfire_status=summary.get("misfire_status") or "on_time",
            previous_durable_run_id=latest_scheduled.get("run_id"),
        )
    if not cfg.scheduler.enabled:
        sessions = list(getattr(cfg.scheduler, "sessions", None) or [])
        times = list(cfg.scheduler.times or [])
        _status_update(
            enabled=False,
            running=False,
            timezone=cfg.scheduler.timezone,
            times=list(cfg.scheduler.times),
            sessions=list(getattr(cfg.scheduler, "sessions", None) or []),
            active_windows=_active_scheduler_windows(cfg, sessions, times),
            private_beta_hourly_enabled=_private_beta_hourly_enabled(cfg),
            guarded_mode=True,
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
    sessions = list(getattr(cfg.scheduler, "sessions", None) or [])
    times = list(cfg.scheduler.times or [])
    _status_update(
        enabled=True,
        thread_alive=True,
        timezone=cfg.scheduler.timezone,
        times=list(cfg.scheduler.times),
        sessions=list(getattr(cfg.scheduler, "sessions", None) or []),
        active_windows=_active_scheduler_windows(cfg, sessions, times),
        private_beta_hourly_enabled=_private_beta_hourly_enabled(cfg),
        guarded_mode=False,
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
