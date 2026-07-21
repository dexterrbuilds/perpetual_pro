"""Telegram Bot API helpers for prop scan alerts.

Credentials MUST come from environment variables only:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

Never log the token; never read secrets from YAML.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from loguru import logger

from src.utils.config import AppConfig, TelegramConfig


def get_telegram_credentials(
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve credentials from explicit args or environment.

    Explicit args are only for tests; production path uses env vars.
    Never falls back to config.yaml.
    """
    token = (bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat = (chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    return token, chat


def send_telegram_message(
    text: str,
    *,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
    timeout: int = 20,
) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    token, chat = get_telegram_credentials(bot_token=bot_token, chat_id=chat_id)
    if not token or not chat:
        logger.warning(
            "Telegram not configured — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment"
        )
        return False
    # Use token only in the request URL; never include it in log messages
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": text[:4000],
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if not resp.ok:
            # Do not log response body if it might echo the token
            logger.warning(
                "Telegram API error status={} (token redacted)",
                resp.status_code,
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram send failed: {}", type(exc).__name__)
        return False


def format_prop_scan_report(
    ranked: List[Dict[str, Any]],
    *,
    slot_label: str = "",
    timezone: str = "Africa/Lagos",
    max_rows: int = 8,
) -> str:
    """Compact HTML-ish plain report for high-confidence prop signals."""
    try:
        now = datetime.now(ZoneInfo(timezone))
        when = now.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:  # noqa: BLE001
        when = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    header = f"📊 Prop Scan"
    if slot_label:
        header += f" · {slot_label}"
    lines = [header, when, ""]
    if not ranked:
        lines.append("No high-confidence prop-safe signals this window.")
        return "\n".join(lines)

    lines.append(f"{len(ranked)} high-confidence signal(s)\n")
    for i, row in enumerate(ranked[:max_rows], 1):
        sym = str(row.get("symbol") or "—")
        base = sym.split("/")[0].split(":")[0]
        price = row.get("price")
        if price is not None:
            try:
                p = float(price)
                price_s = f"${p:,.0f}" if p >= 1000 else f"${p:.4f}"
            except (TypeError, ValueError):
                price_s = "—"
        else:
            price_s = "—"
        direction = str(row.get("direction") or "flat").upper()
        llm = row.get("llm_confidence")
        llm_s = f"{float(llm):.0f}%" if llm is not None else "—"
        lev = row.get("leverage") or row.get("display_leverage") or "—"
        risk = row.get("risk_pct")
        risk_s = f"{float(risk):.2f}%" if risk is not None else "—"
        rank = row.get("rank_score")
        rank_s = f"{float(rank):.0f}" if rank is not None else "—"
        reason = (
            row.get("llm_confidence_reason")
            or row.get("reason")
            or ""
        )
        if len(reason) > 120:
            reason = reason[:117] + "…"
        flags = row.get("prop_flags") or []
        flag_s = ", ".join(flags) if flags else "none"
        lines.append(
            f"{i}. {base} {price_s} · {direction} · LLM {llm_s}\n"
            f"   Lev {lev}x · Risk {risk_s} · Rank {rank_s}\n"
            f"   ⚠ {flag_s}"
            + (f"\n   Why: {reason}" if reason else "")
        )
        lines.append("")
    lines.append("Prop rules: ≤5x · 0.5–1% risk · educational only")
    return "\n".join(lines).strip()


def telegram_from_config(config: AppConfig) -> TelegramConfig:
    """Return policy config (thresholds only). Secrets remain env-only."""
    return config.telegram


def is_telegram_ready(config: Optional[AppConfig] = None) -> bool:
    """True when env credentials exist (and policy not force-disabled)."""
    token, chat = get_telegram_credentials()
    if not token or not chat:
        return False
    if config is not None and not config.telegram.enabled:
        return False
    return True
