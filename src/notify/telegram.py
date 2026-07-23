"""Telegram Bot API helpers for prop scan alerts.

Credentials MUST come from environment variables only:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

Never log the token; never read secrets from YAML.
"""

from __future__ import annotations

import html
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
    max_rows: int = 6,
    min_signal_confidence: float = 68.0,
) -> str:
    """Compact Telegram report for actionable, prop-safe intraday signals."""
    try:
        now = datetime.now(ZoneInfo(timezone))
        when = now.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:  # noqa: BLE001
        when = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    header = "📊 <b>Perpetual Pro Prop Scan</b>"
    if slot_label:
        header += f" · {html.escape(slot_label)}"
    lines = [header, when, "15m execution · 1h/4h confirmation · ≤5x", ""]
    if not ranked:
        lines.append("No high-confidence prop-safe signals this window.")
        return "\n".join(lines)

    lines.append(f"<b>{len(ranked)} actionable signal(s)</b>\n")

    def fmt_price(value: Any) -> str:
        try:
            p = float(value)
        except (TypeError, ValueError):
            return "—"
        if p >= 1000:
            return f"${p:,.2f}"
        if p >= 1:
            return f"${p:.4f}"
        return f"${p:.8f}"

    for i, row in enumerate(ranked[:max_rows], 1):
        sym = str(row.get("symbol") or "—")
        base = sym.split("/")[0].split(":")[0]
        price = row.get("price")
        price_s = fmt_price(price)
        direction = str(row.get("direction") or "flat").upper()
        side_icon = "🟢" if direction == "LONG" else "🔴"
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
        flags = [f for f in (row.get("prop_flags") or []) if f != "LEV_CAPPED_5X"]
        entry_low, entry_high = row.get("entry_low"), row.get("entry_high")
        entry_s = (
            f"{fmt_price(entry_low)}–{fmt_price(entry_high)}"
            if entry_low is not None and entry_high is not None
            else price_s
        )
        entry_status = str(row.get("entry_status") or "ready").replace("_", " ").title()
        execution_score = row.get("execution_score")
        execution_s = (
            f"{float(execution_score):.0f}/100"
            if execution_score is not None
            else "—"
        )
        immediate_risk = row.get("immediate_sl_risk")
        immediate_risk_s = (
            f"{float(immediate_risk):.0f}%"
            if immediate_risk is not None
            else "—"
        )
        targets = list(row.get("take_profits") or [])
        target_line = ""
        if targets:
            shown = " · ".join(
                f"TP{j} {fmt_price(target)}"
                for j, target in enumerate(targets[:2], 1)
            )
            target_line = f"\n   {shown}"
        hold = html.escape(str(row.get("hold_label") or "intraday"))
        bt = row.get("backtest") or {}
        bt_line = ""
        if bt.get("sample_ok"):
            filled = int(bt.get("n_trades") or 0)
            signals = int(bt.get("n_signals") or filled)
            bt_line = (
                f"\n   BT {float(bt.get('win_rate') or 0):.0f}% WR · "
                f"PF {float(bt.get('profit_factor') or 0):.2f} · "
                f"{filled}/{signals} fills · "
                f"{float(bt.get('stop_out_rate') or 0):.0f}% stopped"
            )
        safe_reason = html.escape(str(reason))
        lines.append(
            f"{side_icon} <b>{i}. {html.escape(base)} {direction}</b> · LLM {llm_s}\n"
            f"   {entry_status} · execution {execution_s} · immediate-SL risk {immediate_risk_s}\n"
            f"   Price {price_s} · Entry {entry_s}{target_line}\n"
            f"   SL {fmt_price(row.get('stop_loss'))} · {lev}x · risk {risk_s} · {hold}\n"
            f"   Rank {rank_s}{bt_line}"
            + (f"\n   Why: {safe_reason}" if safe_reason else "")
            + (f"\n   ⚠ {html.escape(', '.join(flags))}" if flags else "")
        )
        lines.append("")
    lines.append(
        f"🛡 Prop gate: <b>≥{min_signal_confidence:.0f}% blended confidence</b> · "
        "execution ≥65 · 0.5–1% risk · ≤5x"
    )
    lines.append("Educational only · honor the stop · close within 24h")
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
