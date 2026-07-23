"""Scheduled prop watchlist scan + Telegram delivery."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from loguru import logger

from src.api.service import AnalyzeRequest, scan_symbols
from src.notify.telegram import (
    format_prop_scan_report,
    is_telegram_ready,
    send_telegram_message,
)
from src.utils.config import AppConfig, load_config

# Fallback watchlist when scheduler.watchlist is empty
DEFAULT_WATCHLIST = [
    "BTC", "ETH", "SOL", "BNB", "AAVE", "ARB", "NEAR", "INJ", "SEI", "TIA",
    "SUI", "APT", "AVAX", "TRX", "UNI",
]


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
        "Scheduled scan: {} symbols · {} · {}",
        len(watchlist),
        req.timeframe,
        req.exchange,
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
    # Credentials from env only (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — never YAML
    tg_ready = is_telegram_ready(cfg)
    if send and tg_ready:
        if filtered or cfg.telegram.notify_on_empty:
            sent = send_telegram_message(
                report,
                parse_mode=cfg.telegram.parse_mode or "HTML",
            )
            if sent:
                logger.info(
                    "Telegram alert sent for {} high-confidence signal(s) ({})",
                    len(filtered),
                    slot_label or "scan",
                )
        else:
            logger.info("No high-confidence signals — Telegram quiet (notify_on_empty=false)")
    elif send and not tg_ready:
        logger.warning(
            "Telegram skip: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment"
        )
    return {
        "ok": bool(result.get("ok")),
        "scanned": len(watchlist),
        "ranked_count": len(ranked),
        "alert_count": len(filtered),
        "filtered": filtered,
        "report": report,
        "telegram_sent": sent,
        "telegram_ready": tg_ready,
        "slot_label": slot_label,
    }


def run_scheduler_loop(
    config: Optional[AppConfig] = None,
    *,
    once: bool = False,
    max_iterations: Optional[int] = None,
) -> None:
    """
    Sleep until next WAT slot, run scan, repeat.

    ``once=True`` runs a single scan immediately (for cron / CLI).
    """
    cfg = config or load_config()
    if once:
        run_scheduled_scan_once(cfg, slot_label="manual")
        return

    times = list(cfg.scheduler.times or ["05:00", "16:00", "20:00"])
    tz_name = cfg.scheduler.timezone or "Africa/Lagos"
    iterations = 0
    if not is_telegram_ready(cfg):
        logger.warning(
            "Scheduler running without Telegram credentials. "
            "Export TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to receive alerts."
        )
    else:
        logger.info("Telegram credentials loaded from environment (token redacted)")
    logger.info(
        "Scheduler loop started · times={} WAT · tz={} · high-confidence only",
        times,
        tz_name,
    )
    while True:
        nxt = next_slot_datetime(times, tz_name)
        now = datetime.now(ZoneInfo(tz_name))
        sleep_s = max(1.0, (nxt - now).total_seconds())
        logger.info("Next scan at {} (sleep {:.0f}s)", nxt.isoformat(), sleep_s)
        # Sleep in chunks so SIGINT is responsive
        end = time.time() + sleep_s
        while time.time() < end:
            time.sleep(min(30.0, end - time.time()))
        label = nxt.strftime("%H:%M WAT")
        try:
            run_scheduled_scan_once(cfg, slot_label=label, send=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scheduled scan failed: {}", exc)
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break
