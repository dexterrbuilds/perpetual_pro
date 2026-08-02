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
from typing import Any, Dict, Iterable, List, Optional, Tuple
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


def get_telegram_alert_chat_ids(
    override_chat_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Resolve alert destinations without changing the primary credential contract.

    Scheduled scans use TELEGRAM_CHAT_ID plus optional comma/space-separated
    TELEGRAM_ADDITIONAL_ALERT_CHAT_IDS. A per-run override is exclusive and is
    used by private /scan commands so their results stay in the requesting DM.
    """
    if override_chat_ids is not None:
        sources = list(override_chat_ids)
    else:
        _, primary_chat = get_telegram_credentials()
        sources = [
            primary_chat,
            os.getenv("TELEGRAM_ADDITIONAL_ALERT_CHAT_IDS") or "",
        ]

    destinations: List[str] = []
    for source in sources:
        for candidate in str(source or "").replace(";", ",").split(","):
            for value in candidate.split():
                chat_id = value.strip()
                if chat_id and chat_id not in destinations:
                    destinations.append(chat_id)
    return destinations


def get_telegram_private_operator_chat_ids() -> List[str]:
    """Return explicit private operator destinations, never public fallbacks.

    Scheduled empty and operational reports must not fall back to
    ``TELEGRAM_CHAT_ID`` because that variable is the public signal channel in
    production. Private operator routing is intentionally opt-in through
    ``TELEGRAM_COMMAND_CHAT_IDS``.
    """
    sources = [os.getenv("TELEGRAM_COMMAND_CHAT_IDS") or ""]
    destinations: List[str] = []
    for source in sources:
        for candidate in str(source or "").replace(";", ",").split(","):
            for value in candidate.split():
                chat_id = value.strip()
                if chat_id and chat_id not in destinations:
                    destinations.append(chat_id)
    return destinations


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
        footer = "\n\nNFA · DYOR · Trade at your own risk"
        caption = caption[: 1024 - len(footer)].rstrip() + footer

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
    execution_score = _number(
        row.get("execution_quality")
        or row.get("execution_score")
        or execution.get("execution_quality")
        or execution.get("score")
    )
    components = dict(row.get("execution_components") or execution.get("components") or {})
    status = str(row.get("entry_status") or execution.get("status") or "blocked")
    timeframe = str(row.get("primary_tf") or chart.get("timeframe") or "15m")
    setup_name = str(row.get("setup_name") or payload.get("setup_name") or "").strip()
    risk_rewards = list(primary.get("risk_reward") or [])
    if status == "wait_retest":
        call = f"{direction} — RETEST"
        entry_mode = "RETEST ONLY — wait for Entry. Do not chase."
    elif status in ("ready", "confirmation_pending"):
        call = f"{direction} — CMP CONFIRMATION"
        entry_mode = (
            "CMP PENDING — price is inside Entry; wait for a confirming "
            "closed candle before the tracker records a fill."
        )
    else:
        call = f"{direction} — CONDITIONAL"
        entry_mode = "CONDITIONAL — do not enter until the stated condition is met."

    reason = (
        row.get("reason")
        or ((payload.get("key_reasons") or [""])[0])
        or row.get("llm_confidence_reason")
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
    scan_price = row.get("price")
    zone_relation = str(row.get("entry_zone_relation") or "").strip().lower()
    relation_label = {
        "inside": "inside Entry zone",
        "favorable_beyond": "already beyond Entry toward TP1",
        "adverse_side": "waiting on the opposite side of Entry",
    }.get(zone_relation, "")
    stop = row.get("stop_loss")
    targets = list(row.get("take_profits") or [])
    leverage = row.get("leverage") or row.get("display_leverage") or 5
    risk_pct = _number(row.get("risk_pct"), 1.0)
    hold_style = _caption_hold_style(
        row.get("hold_label") or primary.get("hold_label"),
        timeframe=timeframe,
        hold_hours_max=primary.get("hold_hours_max"),
    )
    entry_valid_minutes = int(
        _number(
            row.get("entry_valid_for_minutes")
            or primary.get("entry_valid_for_minutes")
        )
    )
    entry_valid_until = (
        row.get("entry_valid_until") or primary.get("entry_valid_until") or ""
    )
    hold_min = _number(
        row.get("hold_hours_min") or primary.get("hold_hours_min"),
        0.0,
    )
    hold_typical_max = _number(
        row.get("hold_hours_typical_max")
        or primary.get("hold_hours_typical_max"),
        0.0,
    )
    hold_hard_max = _number(
        row.get("hold_hours_max") or primary.get("hold_hours_max"),
        0.0,
    )
    rr_tp2 = _caption_target_rr(
        direction=direction,
        entry_low=entry_low,
        entry_high=entry_high,
        stop=stop,
        targets=targets,
        risk_rewards=risk_rewards,
    )
    session = _caption_session_label(slot_label)
    sl_risk = row.get("immediate_sl_risk")
    spread_bps = row.get("spread_bps")
    backtest = row.get("backtest") if isinstance(row.get("backtest"), dict) else {}
    outcome = (
        row.get("outcome_scoring")
        if isinstance(row.get("outcome_scoring"), dict)
        else {}
    )
    outcome_active = str(row.get("scoring_source") or "").startswith(
        "outcome_champion"
    )
    quality_bits = []
    if sl_risk is not None:
        quality_bits.append(f"immediate-SL risk index {_number(sl_risk):.0f}/100")
    if spread_bps is not None:
        quality_bits.append(f"spread {_number(spread_bps):.1f} bps")
    if backtest.get("sample_reliable") and backtest.get("expectancy_r") is not None:
        quality_bits.append(
            f"historical expectancy {_number(backtest.get('expectancy_r')):+.2f}R"
        )

    lines = [
        f"{icon} <b>{html.escape(symbol)} {call}</b>",
        (
            (
                f"<b>Calibrated TP1 probability {confidence:.0f}%</b> · "
                if outcome_active else f"<b>Overall Quality {confidence:.0f}/100</b> · "
            )
            + f"Technical Quality {technical:.0f}/100 · Execution Quality {execution_score:.0f}/100"
        ),
        "",
        (
            f"⏱ {html.escape(timeframe)} · 1h/4h · "
            f"{html.escape(hold_style)}"
            + (f" · {html.escape(session)}" if session else "")
        ),
        (
            f"⌛ <b>Entry valid:</b> {entry_valid_minutes}m"
            + (
                f" · until {html.escape(_caption_expiry_time(entry_valid_until))}"
                if entry_valid_until
                else ""
            )
        )
        if entry_valid_minutes
        else "",
        (
            f"🕒 <b>Hold after fill:</b> {hold_min:g}–{hold_typical_max:g}h"
            + (f" · hard max {hold_hard_max:g}h" if hold_hard_max else "")
        )
        if hold_min and hold_typical_max
        else "",
        (
            f"📍 <b>Price at scan:</b> {_caption_price(scan_price)}"
            + (f" · {html.escape(relation_label)}" if relation_label else "")
        )
        if scan_price is not None
        else "",
        f"🎯 <b>Entry:</b> {_caption_price(entry_low)} – {_caption_price(entry_high)}",
        f"🚦 <b>Entry mode:</b> {html.escape(entry_mode)}",
        f"🛑 <b>Stop:</b> {_caption_price(stop)}",
    ]
    lines = [line for line in lines if line]
    if targets:
        lines.extend(
            f"✅ <b>TP{index}:</b> {_caption_price(target)}"
            for index, target in enumerate(targets[:4], 1)
        )
    setup_label = setup_name or f"{direction.title()} {hold_style}"
    lines += [
        "",
        f"📐 <b>Setup:</b> {html.escape(setup_label)}",
        _telegram_rr_line(row, primary, rr_tp2, risk_pct, leverage),
        f"🧠 <b>Why:</b> {html.escape(reason)}",
    ]
    if components:
        component_bits = [
            f"Entry {_number(components.get('entry_accessibility')):.0f}",
            f"Stop {_number(components.get('stop_quality')):.0f}",
            f"Targets {_number(components.get('target_feasibility')):.0f}",
        ]
        freshness = str(row.get("data_freshness_state") or execution.get("data_freshness_state") or "unknown")
        lines.append(
            "🔎 <b>Execution:</b> "
            + " · ".join(component_bits)
            + f" · data {html.escape(freshness)}"
        )
    risks = list(execution.get("risks") or row.get("rejection_reasons") or [])
    if risks:
        risk_text = risks[0]
        if isinstance(risk_text, dict):
            risk_text = risk_text.get("detail") or risk_text.get("code") or "Execution uncertainty"
        lines.append(f"⚠️ <b>Primary risk:</b> {html.escape(str(risk_text)[:150])}")
    if outcome_active:
        lines.append(
            "🧮 <b>Calibrated EV:</b> "
            f"{_number(outcome.get('conservative_ev_r')):+.2f}R · "
            f"Rank {_number(outcome.get('rank_score')):.0f}/100"
        )
    if quality_bits:
        lines.append(f"🛡 <b>Quality:</b> {html.escape(' · '.join(quality_bits))}")
    execution_note = entry_reason
    if not execution_note and status == "wait_retest":
        execution_note = "Wait for retest of the zone. Do not chase."
    elif not execution_note and status in ("ready", "confirmation_pending"):
        execution_note = "Enter only after the confirmation candle closes."
    if execution_note:
        lines += ["", f"📌 {html.escape(execution_note)}"]
    invalidation_side = "below" if direction == "LONG" else "above"
    lines.append(
        "🧱 <b>Beginner rule:</b> Before fill: cancel at expiry, if TP1 trades, "
        f"or if a {html.escape(timeframe)} candle closes {invalidation_side} Stop; "
        "wicks alone do not count. After fill, honor Stop; "
        "after TP1, move it to breakeven."
    )
    lines += [
        "",
        "NFA · DYOR · Trade at your own risk",
    ]
    caption = "\n".join(lines)
    return caption


def _telegram_rr_line(
    row: Dict[str, Any],
    primary: Dict[str, Any],
    fallback_rr: float,
    risk_pct: float,
    leverage: Any,
) -> str:
    gross = list(row.get("gross_risk_reward") or primary.get("gross_risk_reward") or [])
    net = list(row.get("net_risk_reward") or primary.get("net_risk_reward") or [])
    index = 1 if (len(gross) > 1 or len(list(primary.get("risk_reward") or [])) > 1) else 0
    gross_value = _number(gross[index], fallback_rr) if gross else fallback_rr
    net_value = _number(net[index], gross_value) if net else gross_value
    rr_label = "TP2" if index == 1 else "TP1"
    rr_text = f"gross {gross_value:.2f}R"
    if abs(net_value - gross_value) >= 0.01:
        rr_text += f" · net {net_value:.2f}R"
    return (
        f"📊 <b>R:R ({rr_label}):</b> {rr_text} · "
        f"Risk {risk_pct:g}% · ≤{leverage}x"
    )


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


def _caption_expiry_time(value: Any) -> str:
    """Render an ISO signal deadline compactly in UTC."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo("UTC")).strftime("%H:%M UTC")
    except (TypeError, ValueError):
        return raw[:16]


def _caption_hold_style(
    hold_label: Any,
    *,
    timeframe: str,
    hold_hours_max: Any = None,
) -> str:
    """Normalize the actual plan horizon into a clear Telegram style."""
    label = str(hold_label or "").strip().lower()
    max_hours = _number(hold_hours_max, -1.0)
    if label == "scalp" or (0 < max_hours <= 2):
        return "Scalp"
    if label in {"intraday", "intra-day"} or (2 < max_hours <= 12):
        return "Intraday"
    if label in {"day", "day trade", "day-trade"} or max_hours > 12:
        return "Day"

    tf = str(timeframe or "").strip().lower()
    if tf in {"1m", "3m", "5m"}:
        return "Scalp"
    if tf in {"15m", "30m"}:
        return "Intraday"
    return "Day"


def _caption_target_rr(
    *,
    direction: str,
    entry_low: Any,
    entry_high: Any,
    stop: Any,
    targets: List[Any],
    risk_rewards: List[Any],
) -> float:
    if len(risk_rewards) > 1:
        return max(0.0, _number(risk_rewards[1]))
    if risk_rewards:
        return max(0.0, _number(risk_rewards[0]))
    target_index = 1 if len(targets) > 1 else 0
    if not targets:
        return 0.0
    entry = (_number(entry_low) + _number(entry_high)) / 2.0
    stop_value = _number(stop)
    target = _number(targets[target_index])
    risk = abs(entry - stop_value)
    if risk <= 0:
        return 0.0
    reward = target - entry if direction == "LONG" else entry - target
    return max(0.0, reward / risk)


def _caption_session_label(slot_label: str) -> str:
    label = str(slot_label or "").strip()
    lowered = label.lower()
    if "new york" in lowered or "ny " in lowered:
        return "NY open" if "open" in lowered else "NY session"
    if "london" in lowered:
        return "London open" if "open" in lowered else "London session"
    return label


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
    min_signal_confidence: float = 80.0,
    scanned_count: Optional[int] = None,
    ranked_count: Optional[int] = None,
    rejection_summary: Optional[Dict[str, Any]] = None,
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
        diagnostic = dict(rejection_summary or {})
        directional_count = int(
            diagnostic.get("directional_candidates")
            if diagnostic.get("directional_candidates") is not None
            else (ranked_count or 0)
        )
        lines += [
            "⏸ <b>NO QUALITY SETUP — STAND ASIDE</b>",
            "",
            (
                f"Scanned {scanned_count} symbols"
                if scanned_count is not None
                else "Scheduled scan completed"
            )
            + (
                f" · {directional_count} directional candidate(s)"
                if ranked_count is not None or diagnostic
                else ""
            )
            + ".",
            (
                f"Nothing passed ≥{min_signal_confidence:.0f}% confidence, "
                "execution/SL-risk, market-quality, R:R, and prop-safety gates."
            ),
        ]
        if diagnostic:
            labels = {
                "OVERALL_QUALITY_BELOW_MINIMUM": "Overall Quality below minimum",
                "EXECUTION_QUALITY_BELOW_MINIMUM": "Execution Quality below minimum",
                "CONFLUENCE_BELOW_MINIMUM": "Confluence below minimum",
                "AVOID_CHASE": "Avoid Chase",
                "ENTRY_BLOCKED": "Entry blocked",
                "FLAT_DIRECTION": "Flat/non-directional",
                "NO_FEASIBLE_TARGET": "No feasible target",
                "PROP_COMPATIBILITY_FAILED": "Prop compatibility",
            }
            primary = dict(diagnostic.get("primary_rejection_counts") or {})
            primary.pop("ELIGIBLE", None)
            if primary:
                lines += ["", "<b>Main blockers</b>"]
                for code, count in sorted(
                    primary.items(), key=lambda item: (-item[1], item[0])
                )[:4]:
                    label = labels.get(code, code.replace("_", " ").title())
                    lines.append(f"• {html.escape(label)}: {int(count)}")
            nearest = list(diagnostic.get("closest_rejected_candidates") or [])
            nearest = [
                row
                for row in nearest
                if str(row.get("direction") or "").lower() in {"long", "short"}
            ]
            if nearest:
                row = nearest[0]
                symbol = html.escape(str(row.get("symbol") or "—").split("/")[0])
                direction = html.escape(str(row.get("direction") or "").upper())
                overall = _number(row.get("overall_quality"))
                gate = str(row.get("closest_to_passing_gate") or "a required gate")
                distance = _number(row.get("distance_to_eligibility"))
                lines += [
                    "",
                    "<b>Closest rejected setup — NON-ACTIONABLE</b>",
                    f"{symbol} {direction} · Overall {overall:.1f}/100"
                    if overall is not None
                    else f"{symbol} {direction}",
                    (
                        f"Nearest gate: {html.escape(labels.get(gate, gate.replace('_', ' ').title()))}"
                        + (f" · diagnostic gap {distance:.3f}" if distance is not None else "")
                    ),
                ]
            lines += ["", "No gate was lowered. No rejected setup is a trade signal."]
        lines += [
            "No trade is the correct position until a clean entry appears.",
            "",
            "NFA · DYOR · Trade at your own risk",
        ]
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
        direction = str(row.get("direction") or "flat").upper()
        side_icon = "🟢" if direction == "LONG" else "🔴"
        confidence = _number(row.get("confidence"))
        technical = _number(row.get("technical_confidence"))
        lev = row.get("leverage") or row.get("display_leverage") or "—"
        risk = row.get("risk_pct")
        risk_s = f"{float(risk):.2f}%" if risk is not None else "—"
        reason = (
            row.get("reason")
            or row.get("llm_confidence_reason")
            or ""
        )
        if len(reason) > 120:
            reason = reason[:117] + "…"
        flags = [f for f in (row.get("prop_flags") or []) if f != "LEV_CAPPED_5X"]
        entry_low, entry_high = row.get("entry_low"), row.get("entry_high")
        entry_s = (
            f"{fmt_price(entry_low)}–{fmt_price(entry_high)}"
            if entry_low is not None and entry_high is not None
            else fmt_price(row.get("price"))
        )
        raw_entry_status = str(row.get("entry_status") or "ready")
        entry_status = {
            "ready": "CMP Confirmation",
            "confirmation_pending": "CMP Confirmation",
            "wait_retest": "Retest Only",
            "avoid_chase": "Avoid Chase",
            "blocked": "Blocked",
        }.get(raw_entry_status, raw_entry_status.replace("_", " ").title())
        execution_score = row.get("execution_quality", row.get("execution_score"))
        execution_s = (
            f"{float(execution_score):.0f}/100"
            if execution_score is not None
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
        valid_minutes = int(_number(row.get("entry_valid_for_minutes"), 0.0))
        hold_min = _number(row.get("hold_hours_min"), 0.0)
        hold_typical = _number(row.get("hold_hours_typical_max"), 0.0)
        safe_reason = html.escape(str(reason))
        outcome = (
            row.get("outcome_scoring")
            if isinstance(row.get("outcome_scoring"), dict)
            else {}
        )
        outcome_line = (
            f"\n   EV {_number(outcome.get('conservative_ev_r')):+.2f}R · "
            f"model rank {_number(outcome.get('rank_score')):.0f}/100"
            if str(row.get("scoring_source") or "").startswith(
                "outcome_champion"
            )
            else ""
        )
        lines.append(
            f"{side_icon} <b>{i}. {html.escape(base)} {direction}</b> · "
            f"<b>{'Calibrated TP1 probability ' + format(confidence, '.0f') + '%' if str(row.get('scoring_source') or '').startswith('outcome_champion') else 'Overall Quality ' + format(confidence, '.0f') + '/100'}</b>\n"
            f"   Technical Quality {technical:.0f}/100 · Execution Quality {execution_s} · {entry_status}\n"
            f"   Entry {entry_s}{target_line}\n"
            f"   SL {fmt_price(row.get('stop_loss'))} · {lev}x · risk {risk_s} · {hold}\n"
            + (
                f"   Entry expires {valid_minutes}m · hold {hold_min:g}–{hold_typical:g}h\n"
                if valid_minutes and hold_min and hold_typical
                else ""
            )
            + outcome_line
            + (f"\n   Why: {safe_reason}" if safe_reason else "")
            + (f"\n   ⚠ {html.escape(', '.join(flags))}" if flags else "")
        )
        lines.append("")
    displayed_rows = ranked[:max_rows]
    calibrated = bool(displayed_rows) and all(
        str(row.get("scoring_source") or "").startswith("outcome_champion")
        for row in displayed_rows
    )
    confidence_label = (
        "calibrated confidence" if calibrated else "signal confidence"
    )
    lines.append(
        f"🛡 Prop gate: <b>≥{min_signal_confidence:.0f}% {confidence_label}</b> · "
        "clean execution · TP2 ≥1.25R · 0.5–1% each · ≤2% total open risk · ≤5x"
    )
    lines.append("NFA · DYOR · Trade at your own risk")
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
