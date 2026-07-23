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

TELEGRAM_API_ROOT = "https://api.telegram.org"


def _masked_chat_id(chat_id: str) -> str:
    """Return a useful diagnostic identifier without exposing the full chat id."""
    value = (chat_id or "").strip()
    if not value:
        return "missing"
    if value.startswith("@"):
        return f"@…{value[-4:]}" if len(value) > 5 else "@…"
    return f"…{value[-4:]}" if len(value) > 4 else "…"


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


def _response_detail(response: requests.Response) -> Dict[str, Any]:
    """Extract Telegram's safe error fields without logging request URLs/tokens."""
    try:
        body = response.json()
    except (TypeError, ValueError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    parameters = body.get("parameters") if isinstance(body.get("parameters"), dict) else {}
    return {
        "http_status": response.status_code,
        "telegram_error_code": body.get("error_code"),
        "description": str(
            body.get("description") or f"Telegram HTTP {response.status_code}"
        )[:300],
        "retry_after": parameters.get("retry_after"),
        "response": body,
    }


def send_telegram_message_detailed(
    text: str,
    *,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
    timeout: int = 20,
) -> Dict[str, Any]:
    """Send a message and return redacted delivery diagnostics."""
    token, chat = get_telegram_credentials(bot_token=bot_token, chat_id=chat_id)
    masked_chat = _masked_chat_id(chat)
    if not token or not chat:
        missing = [
            name
            for name, value in (
                ("TELEGRAM_BOT_TOKEN", token),
                ("TELEGRAM_CHAT_ID", chat),
            )
            if not value
        ]
        description = f"Missing environment variable(s): {', '.join(missing)}"
        logger.error("Telegram delivery failed: {}", description)
        return {
            "ok": False,
            "error": "not_configured",
            "description": description,
            "chat_id_masked": masked_chat,
            "message_id": None,
        }
    if not (text or "").strip():
        logger.error("Telegram delivery failed: empty message")
        return {
            "ok": False,
            "error": "empty_message",
            "description": "Message text is empty",
            "chat_id_masked": masked_chat,
            "message_id": None,
        }

    # The token is used only in the request URL. Never log this URL.
    url = f"{TELEGRAM_API_ROOT}/bot{token}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id": chat,
        "text": text[:4000],
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    logger.info(
        "Telegram delivery attempt: chat={} chars={} parse_mode={}",
        masked_chat,
        len(payload["text"]),
        parse_mode or "none",
    )
    try:
        response = requests.post(url, json=payload, timeout=(5, timeout))
        detail = _response_detail(response)
        body = detail.pop("response")
        if response.ok and body.get("ok") is True:
            message = body.get("result") if isinstance(body.get("result"), dict) else {}
            message_id = message.get("message_id")
            logger.info(
                "Telegram delivery succeeded: chat={} message_id={}",
                masked_chat,
                message_id,
            )
            return {
                "ok": True,
                "error": None,
                "description": "Message delivered",
                "chat_id_masked": masked_chat,
                "message_id": message_id,
                **detail,
            }
        retry_note = (
            f" retry_after={detail['retry_after']}s"
            if detail.get("retry_after")
            else ""
        )
        logger.error(
            "Telegram delivery failed: chat={} http={} telegram_code={} description={}{}",
            masked_chat,
            detail.get("http_status"),
            detail.get("telegram_error_code"),
            detail.get("description"),
            retry_note,
        )
        return {
            "ok": False,
            "error": "telegram_api_error",
            "chat_id_masked": masked_chat,
            "message_id": None,
            **detail,
        }
    except requests.RequestException as exc:
        # Exception strings can contain the tokenized URL, so log only the type.
        logger.error(
            "Telegram delivery failed: chat={} network_error={} (token redacted)",
            masked_chat,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "error": "network_error",
            "description": type(exc).__name__,
            "chat_id_masked": masked_chat,
            "message_id": None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Telegram delivery failed unexpectedly: chat={} error_type={}",
            masked_chat,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "error": "unexpected_error",
            "description": type(exc).__name__,
            "chat_id_masked": masked_chat,
            "message_id": None,
        }


def send_telegram_message(
    text: str,
    *,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
    timeout: int = 20,
) -> bool:
    """Backward-compatible boolean wrapper around detailed delivery."""
    return bool(
        send_telegram_message_detailed(
            text,
            bot_token=bot_token,
            chat_id=chat_id,
            parse_mode=parse_mode,
            timeout=timeout,
        ).get("ok")
    )


def send_telegram_photo_detailed(
    photo: bytes,
    caption: str,
    *,
    filename: str = "perpetual-pro-signal.png",
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: str = "HTML",
    timeout: int = 35,
) -> Dict[str, Any]:
    """Upload a plotted signal chart with a concise Telegram caption."""
    token, chat = get_telegram_credentials(bot_token=bot_token, chat_id=chat_id)
    masked_chat = _masked_chat_id(chat)
    if not token or not chat:
        description = "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
        logger.error("Telegram photo delivery failed: {}", description)
        return {
            "ok": False,
            "error": "not_configured",
            "description": description,
            "chat_id_masked": masked_chat,
            "message_id": None,
        }
    if not photo:
        return {
            "ok": False,
            "error": "empty_photo",
            "description": "Rendered chart image is empty",
            "chat_id_masked": masked_chat,
            "message_id": None,
        }
    if len(caption) > 1024:
        logger.warning("Telegram photo caption truncated from {} characters", len(caption))
        caption = caption[:1000] + "\nEducational only."

    url = f"{TELEGRAM_API_ROOT}/bot{token}/sendPhoto"
    data: Dict[str, Any] = {
        "chat_id": chat,
        "caption": caption,
    }
    if parse_mode:
        data["parse_mode"] = parse_mode
    files = {"photo": (filename, photo, "image/png")}
    logger.info(
        "Telegram photo attempt: chat={} bytes={} caption_chars={}",
        masked_chat,
        len(photo),
        len(caption),
    )
    try:
        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=(5, timeout),
        )
        detail = _response_detail(response)
        body = detail.pop("response")
        if response.ok and body.get("ok") is True:
            message = body.get("result") if isinstance(body.get("result"), dict) else {}
            message_id = message.get("message_id")
            logger.info(
                "Telegram photo succeeded: chat={} message_id={}",
                masked_chat,
                message_id,
            )
            return {
                "ok": True,
                "error": None,
                "description": "Chart alert delivered",
                "chat_id_masked": masked_chat,
                "message_id": message_id,
                "photo_bytes": len(photo),
                **detail,
            }
        logger.error(
            "Telegram photo failed: chat={} http={} telegram_code={} description={}",
            masked_chat,
            detail.get("http_status"),
            detail.get("telegram_error_code"),
            detail.get("description"),
        )
        return {
            "ok": False,
            "error": "telegram_api_error",
            "chat_id_masked": masked_chat,
            "message_id": None,
            **detail,
        }
    except requests.RequestException as exc:
        logger.error(
            "Telegram photo failed: chat={} network_error={} (token redacted)",
            masked_chat,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "error": "network_error",
            "description": type(exc).__name__,
            "chat_id_masked": masked_chat,
            "message_id": None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Telegram photo failed unexpectedly: chat={} error_type={}",
            masked_chat,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "error": "unexpected_error",
            "description": type(exc).__name__,
            "chat_id_masked": masked_chat,
            "message_id": None,
        }


def format_signal_photo_caption(
    row: Dict[str, Any],
    *,
    slot_label: str = "",
) -> str:
    """Build a clean, actionable caption that stays within Telegram's limit."""
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    primary = (
        payload.get("primary_setup")
        if isinstance(payload.get("primary_setup"), dict)
        else {}
    )
    execution = (
        payload.get("execution")
        if isinstance(payload.get("execution"), dict)
        else {}
    )
    chart = payload.get("chart") if isinstance(payload.get("chart"), dict) else {}

    direction = str(row.get("direction") or "flat").upper()
    icon = "🟢" if direction == "LONG" else "🔴"
    symbol = str(row.get("symbol") or "—").split("/")[0].split(":")[0]
    confidence = _number(row.get("confidence"))
    technical = _number(row.get("technical_confidence"))
    llm = _number(row.get("llm_confidence"))
    execution_score = _number(row.get("execution_score") or execution.get("score"))
    status = str(row.get("entry_status") or execution.get("status") or "blocked")
    timeframe = str(row.get("primary_tf") or chart.get("timeframe") or "15m")
    setup_name = str(row.get("setup_name") or payload.get("setup_name") or "").strip()
    risk_rewards = list(primary.get("risk_reward") or [])
    confidence_label = (
        "VERY HIGH" if confidence >= 80 else ("HIGH" if confidence >= 72 else "QUALIFIED")
    )
    if status == "wait_retest":
        call = f"{direction} RETEST — DO NOT CHASE"
    elif status == "ready":
        call = f"{direction} — CONFIRMATION READY"
    else:
        call = f"{direction} — CONDITIONAL"

    reason = (
        row.get("llm_confidence_reason")
        or row.get("reason")
        or ((payload.get("key_reasons") or [""])[0])
        or "Multi-timeframe confluence passed"
    )
    reason = str(reason).strip()
    if len(reason) > 175:
        reason = reason[:172].rstrip() + "…"
    entry_reason = str(execution.get("entry_reason") or "").strip()
    if len(entry_reason) > 130:
        entry_reason = entry_reason[:127].rstrip() + "…"

    entry_low = row.get("entry_low")
    entry_high = row.get("entry_high")
    stop = row.get("stop_loss")
    targets = list(row.get("take_profits") or [])
    leverage = row.get("leverage") or row.get("display_leverage") or 5
    risk_pct = _number(row.get("risk_pct"), 1.0)
    immediate_risk = _number(row.get("immediate_sl_risk"))
    order_flow = _number(row.get("order_flow_score"))
    hold = str(row.get("hold_label") or primary.get("hold_detail") or "30m–24h")

    lines = [
        f"{icon} <b>{html.escape(symbol)} {call}</b>",
        (
            f"<b>{confidence_label} CONFIDENCE {confidence:.0f}%</b> · "
            f"Tech {technical:.0f}% · LLM {llm:.0f}% · Exec {execution_score:.0f}/100"
        ),
        f"⏱ {html.escape(timeframe)} entry · 1h/4h confirmation"
        + (f" · {html.escape(slot_label)}" if slot_label else ""),
    ]
    if setup_name or risk_rewards:
        setup_bits = []
        if setup_name:
            setup_bits.append(html.escape(setup_name))
        if len(risk_rewards) > 1:
            setup_bits.append(f"TP2 R:R {_number(risk_rewards[1]):.2f}")
        lines.append("📐 " + " · ".join(setup_bits))
    lines += [
        "",
        f"🎯 <b>Entry</b> {_caption_price(entry_low)} – {_caption_price(entry_high)}",
        f"🛑 <b>Stop</b> {_caption_price(stop)} · risk {risk_pct:.2f}% · ≤{leverage}x",
    ]
    if targets:
        lines.append(
            "✅ "
            + " · ".join(
                f"<b>TP{index}</b> {_caption_price(target)}"
                for index, target in enumerate(targets[:4], 1)
            )
        )
    lines += [
        f"📊 Immediate-SL risk {immediate_risk:.0f}% · flow {order_flow:+.2f}",
        f"🧠 <b>Why:</b> {html.escape(reason)}",
    ]
    if entry_reason:
        lines.append(f"📌 <b>Execution:</b> {html.escape(entry_reason)}")
    funding = row.get("funding_rate")
    oi_change = row.get("open_interest_change_pct_24h")
    derivatives = []
    if funding is not None:
        derivatives.append(f"funding {_number(funding) * 100:+.4f}%")
    if oi_change is not None:
        derivatives.append(f"OI 24h {_number(oi_change):+.2f}%")
    if derivatives:
        lines.append("⚙ " + " · ".join(derivatives))
    lines += [
        f"⌛ Hold {html.escape(hold)}",
        "<i>Wait for the stated entry condition. Educational only.</i>",
    ]
    caption = "\n".join(lines)
    # Optional details are removed before any hard truncation, preserving HTML.
    if len(caption) > 1024 and entry_reason:
        lines = [line for line in lines if not line.startswith("📌")]
        caption = "\n".join(lines)
    if len(caption) > 1024:
        lines = [line for line in lines if not line.startswith("⚙")]
        caption = "\n".join(lines)
    return caption


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _caption_price(value: Any) -> str:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return "—"
    if price >= 1000:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:.4f}".rstrip("0").rstrip(".")
    return f"${price:.8f}".rstrip("0").rstrip(".")


def diagnose_telegram(timeout: int = 10) -> Dict[str, Any]:
    """Validate token, chat access, and bot membership without sending a message."""
    token, chat = get_telegram_credentials()
    result: Dict[str, Any] = {
        "ok": False,
        "token_configured": bool(token),
        "chat_id_configured": bool(chat),
        "chat_id_masked": _masked_chat_id(chat),
        "bot_identity_ok": False,
        "chat_access_ok": False,
        "membership_ok": False,
        "can_send_inferred": False,
        "bot_username": None,
        "chat_type": None,
        "membership_status": None,
        "checks": [],
    }
    if not token or not chat:
        result["error"] = "not_configured"
        result["description"] = (
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the running process environment"
        )
        return result

    def telegram_get(method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            response = requests.get(
                f"{TELEGRAM_API_ROOT}/bot{token}/{method}",
                params=params,
                timeout=(5, timeout),
            )
            detail = _response_detail(response)
            body = detail.pop("response")
            if response.ok and body.get("ok") is True:
                return {"ok": True, "result": body.get("result"), **detail}
            return {"ok": False, **detail}
        except requests.RequestException as exc:
            return {
                "ok": False,
                "description": type(exc).__name__,
                "error": "network_error",
            }

    me = telegram_get("getMe")
    result["checks"].append(
        {
            "name": "getMe",
            "ok": bool(me.get("ok")),
            "description": me.get("description"),
        }
    )
    if not me.get("ok"):
        result["error"] = "invalid_token_or_network"
        result["description"] = me.get("description") or "Telegram getMe failed"
        logger.error(
            "Telegram diagnostics failed at getMe: {}",
            result["description"],
        )
        return result

    bot = me.get("result") if isinstance(me.get("result"), dict) else {}
    bot_id = bot.get("id")
    result["bot_identity_ok"] = True
    result["bot_username"] = bot.get("username")

    chat_result = telegram_get("getChat", {"chat_id": chat})
    result["checks"].append(
        {
            "name": "getChat",
            "ok": bool(chat_result.get("ok")),
            "description": chat_result.get("description"),
        }
    )
    if not chat_result.get("ok"):
        result["error"] = "chat_unavailable"
        result["description"] = chat_result.get("description") or "Telegram getChat failed"
        logger.error(
            "Telegram diagnostics failed at getChat: chat={} description={}",
            result["chat_id_masked"],
            result["description"],
        )
        return result

    chat_info = (
        chat_result.get("result")
        if isinstance(chat_result.get("result"), dict)
        else {}
    )
    result["chat_access_ok"] = True
    result["chat_type"] = chat_info.get("type")

    member_result = telegram_get(
        "getChatMember",
        {"chat_id": chat, "user_id": bot_id},
    )
    result["checks"].append(
        {
            "name": "getChatMember",
            "ok": bool(member_result.get("ok")),
            "description": member_result.get("description"),
        }
    )
    if member_result.get("ok"):
        membership = (
            member_result.get("result")
            if isinstance(member_result.get("result"), dict)
            else {}
        )
        status = str(membership.get("status") or "")
        result["membership_status"] = status
        result["membership_ok"] = status not in ("left", "kicked", "")
        if result["chat_type"] == "channel":
            result["can_send_inferred"] = bool(
                status == "creator" or membership.get("can_post_messages")
            )
        elif status == "restricted":
            result["can_send_inferred"] = bool(membership.get("can_send_messages"))
        else:
            result["can_send_inferred"] = result["membership_ok"]
    else:
        result["description"] = (
            member_result.get("description")
            or "Could not verify bot membership; sendMessage is the definitive test"
        )

    result["ok"] = bool(
        result["bot_identity_ok"]
        and result["chat_access_ok"]
        and result["membership_ok"]
        and result["can_send_inferred"]
    )
    if not result["ok"] and not result.get("error"):
        result["error"] = "permission_check_failed"
    logger.info(
        "Telegram diagnostics: chat={} bot=@{} chat_type={} membership={} can_send={}",
        result["chat_id_masked"],
        result.get("bot_username") or "unknown",
        result.get("chat_type") or "unknown",
        result.get("membership_status") or "unknown",
        result["can_send_inferred"],
    )
    return result


def send_test_telegram_alert(source: str = "manual") -> Dict[str, Any]:
    """Run permission diagnostics and send a fixed manual test message."""
    diagnostics = diagnose_telegram()
    if not diagnostics.get("token_configured") or not diagnostics.get("chat_id_configured"):
        return {
            "ok": False,
            "source": source,
            "diagnostics": diagnostics,
            "delivery": None,
        }
    when = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S UTC")
    message = (
        "✅ <b>Perpetual Pro Telegram test</b>\n"
        f"Source: {html.escape(source)}\n"
        f"Time: {when}\n"
        "Credentials, chat access, and message delivery are working."
    )
    delivery = send_telegram_message_detailed(message, parse_mode="HTML")
    return {
        "ok": bool(delivery.get("ok")),
        "source": source,
        "diagnostics": diagnostics,
        "delivery": delivery,
    }


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
