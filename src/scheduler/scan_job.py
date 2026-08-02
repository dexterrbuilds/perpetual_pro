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
from src.notify.telegram import (
    format_signal_photo_caption,
    format_prop_scan_report,
    get_delivery_status,
    get_telegram_alert_chat_ids,
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
        material = f"{slot_label}|{scheduled_for}"
        return "sched_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return "run_" + uuid4().hex


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
    filtered = filter_high_confidence(
        ranked,
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
            getattr(
                cfg.analysis,
                "max_pre_entry_tp1_progress_pct",
                70.0,
            )
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
    suppressed_duplicates: List[Dict[str, Any]] = []
    if send and not manual_delivery and filtered:
        filtered, suppressed_duplicates = suppress_recent_scheduled_signals(filtered)
        if suppressed_duplicates:
            logger.info(
                "Suppressed {} duplicate scheduled signal(s) still inside "
                "their entry-validity window",
                len(suppressed_duplicates),
            )
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
                    accepted[index] = dict(validation.get("row") or original)
                else:
                    rejected = dict(original)
                    rejected["pre_delivery_rejection_reasons"] = list(
                        validation.get("reasons") or ["PRE_SEND_REJECTED"]
                    )
                    pre_delivery_rejected.append(rejected)
                    rejected["gate_evaluation"] = dict(
                        validation.get("gate_evaluation")
                        or rejected.get("gate_evaluation")
                        or {}
                    )
                    logger.warning(
                        "Telegram candidate failed final revalidation: symbol={} reasons={}",
                        original.get("symbol"),
                        "|".join(rejected["pre_delivery_rejection_reasons"]),
                    )
        filtered = [row for row in accepted if row is not None]

    scan_failed = not bool(result.get("ok"))
    report_summary = dict(result.get("rejection_analytics") or {})
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
            "⚠️ <b>SCAN UNAVAILABLE</b>\n\n"
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
    # Manual commands keep their exclusive per-request destination. Recurring
    # scans use the configured delivery mode; this routing layer does not alter
    # signal generation, eligibility, or lifecycle rules.
    destinations = get_telegram_alert_chat_ids(telegram_chat_ids)
    report_destinations = get_telegram_report_chat_ids(
        telegram_chat_ids if manual_delivery else None
    )
    routing_status = get_delivery_status()
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
                    if not signal.get("ok"):
                        item = {
                            "ok": False,
                            "symbol": symbol,
                            "mode": "photo",
                            "error": signal.get("error"),
                            "description": signal.get("description"),
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
                                "idempotency_key": f"{run_id}:{digest}:signal_photo:{symbol}",
                                "destination_hash": digest,
                                "delivery_type": f"signal_photo:{symbol}",
                                "ok": bool(item.get("ok")),
                                "message_id": item.get("message_id"),
                                "error": item.get("error"),
                            }
                        )
                    destination_items.append(item)
                    chart_items.append(item)
                    if item.get("ok"):
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
                        for signal_destinations in tracking_destinations:
                            if destination not in signal_destinations:
                                signal_destinations.append(destination)
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

            chart_sent = sum(1 for item in chart_items if item.get("ok"))
            chart_failed = len(chart_items) - chart_sent
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
            if photo_ok:
                sent = True
                delivery_status = "sent_chart_alerts"
                logger.info(
                    "Scheduled Telegram chart alerts succeeded: slot={} sent={} "
                    "destinations={}",
                    slot_label or "scan",
                    chart_sent,
                    len(destinations),
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
                        "telegram_manual"
                        if manual_delivery
                        else "telegram_scheduled"
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
            or routing_status["mode"] == "private_beta"
            or (
                bool(notify_on_empty)
                if notify_on_empty is not None
                else cfg.telegram.notify_on_empty
            )
        ):
            destination_results = []
            report_type = "scan_failure" if scan_failed else "no_quality_report"
            for destination in report_destinations:
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
            elif report_destinations:
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
                len(report_destinations),
                sent_count,
                max(0, len(report_destinations) - sent_count),
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
        "rejection_analytics": result.get("rejection_analytics") or {},
        "rejection_analytics_finalized": analytics_finalized,
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
    telegram_chat_ids: Optional[List[str]] = None,
    scheduled_for: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one scan without overlapping another scheduler, API, or bot request."""
    cfg = config or load_config()
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
        resolved_run_id = str(
            run_id or _durable_run_id(slot_label, scheduled_for)
        )
        source = _scheduler_run_source(
            slot_label,
            telegram_chat_ids,
            scheduled_for,
        )
        watchlist = list(symbols or cfg.scheduler.watchlist or []) or list(
            DEFAULT_WATCHLIST
        )
        repository = SchedulerRunRepository(cfg.outcome_scoring.database_url)
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
        if source == "scheduled" and repository.enabled and claimed is None:
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
        return outcome
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
    times = list(cfg.scheduler.times or [])
    sessions = list(getattr(cfg.scheduler, "sessions", None) or [])
    tz_name = cfg.scheduler.timezone or "Africa/Lagos"
    _status_update(
        enabled=bool(cfg.scheduler.enabled),
        running=True,
        timezone=tz_name,
        times=times,
        sessions=sessions,
        active_windows=(sessions if sessions else times),
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
                outcome = run_scheduled_scan_once(
                    cfg,
                    slot_label=label,
                    send=True,
                    scheduled_for=nxt.astimezone(ZoneInfo("UTC")).isoformat(),
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
    latest_run = SchedulerRunRepository(
        cfg.outcome_scoring.database_url
    ).latest()
    if latest_run:
        _status_update(last_run=latest_run)
    if not cfg.scheduler.enabled:
        _status_update(
            enabled=False,
            running=False,
            timezone=cfg.scheduler.timezone,
            times=list(cfg.scheduler.times),
            sessions=list(getattr(cfg.scheduler, "sessions", None) or []),
            active_windows=(
                list(getattr(cfg.scheduler, "sessions", None) or [])
                or list(cfg.scheduler.times)
            ),
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
    _status_update(
        enabled=True,
        thread_alive=True,
        timezone=cfg.scheduler.timezone,
        times=list(cfg.scheduler.times),
        sessions=list(getattr(cfg.scheduler, "sessions", None) or []),
        active_windows=(
            list(getattr(cfg.scheduler, "sessions", None) or [])
            or list(cfg.scheduler.times)
        ),
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
