"""Inbound Telegram bot commands using a secured webhook."""

from __future__ import annotations

import hashlib
import html
import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from loguru import logger

from src.notify.telegram import (
    TELEGRAM_API_ROOT,
    get_telegram_credentials,
    is_telegram_ready,
    send_telegram_message_detailed,
)
from src.scheduler.scan_job import (
    get_scheduler_status,
    run_scheduled_scan_once,
    scan_in_progress,
)
from src.utils.config import AppConfig

_WEBHOOK_STATUS_LOCK = Lock()
_WEBHOOK_STATUS: Dict[str, Any] = {
    "configured": False,
    "public_url_configured": False,
    "host": None,
    "last_error": None,
}
_VALID_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9/_:.-]{0,29}$")
_VALID_TIMEFRAMES = {"15m", "1h", "4h"}


def telegram_webhook_secret(bot_token: Optional[str] = None) -> str:
    """Derive a stable Telegram-compatible secret without another env secret."""
    token, _ = get_telegram_credentials(bot_token=bot_token)
    if not token:
        return ""
    return hashlib.sha256(f"perpetual-pro:{token}".encode("utf-8")).hexdigest()


def resolve_telegram_webhook_url() -> str:
    """Use an explicit endpoint or Render's public service URL."""
    raw = (
        os.getenv("TELEGRAM_WEBHOOK_URL")
        or os.getenv("RENDER_EXTERNAL_URL")
        or ""
    ).strip()
    if not raw:
        return ""
    url = raw.rstrip("/")
    if not url.endswith("/telegram/webhook"):
        url += "/telegram/webhook"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def get_telegram_webhook_status() -> Dict[str, Any]:
    with _WEBHOOK_STATUS_LOCK:
        return dict(_WEBHOOK_STATUS)


def _set_webhook_status(**values: Any) -> None:
    with _WEBHOOK_STATUS_LOCK:
        _WEBHOOK_STATUS.update(values)


def configure_telegram_webhook(config: AppConfig, timeout: int = 15) -> bool:
    """Register the inbound command webhook and the bot command menu."""
    token, _ = get_telegram_credentials()
    webhook_url = resolve_telegram_webhook_url()
    host = urlparse(webhook_url).netloc if webhook_url else None
    if not is_telegram_ready(config):
        _set_webhook_status(
            configured=False,
            public_url_configured=bool(webhook_url),
            host=host,
            last_error="telegram_not_configured",
        )
        logger.warning("Telegram command webhook skipped: credentials are not ready")
        return False
    if not webhook_url:
        _set_webhook_status(
            configured=False,
            public_url_configured=False,
            host=None,
            last_error="public_url_missing",
        )
        logger.warning(
            "Telegram command webhook skipped: set TELEGRAM_WEBHOOK_URL "
            "(Render can use RENDER_EXTERNAL_URL automatically)"
        )
        return False

    api_url = f"{TELEGRAM_API_ROOT}/bot{token}/setWebhook"
    try:
        response = requests.post(
            api_url,
            json={
                "url": webhook_url,
                "secret_token": telegram_webhook_secret(token),
                "allowed_updates": ["message"],
                "drop_pending_updates": False,
            },
            timeout=(5, timeout),
        )
        try:
            body = response.json()
        except (TypeError, ValueError):
            body = {}
        if not response.ok or not isinstance(body, dict) or body.get("ok") is not True:
            description = (
                body.get("description")
                if isinstance(body, dict)
                else f"Telegram HTTP {response.status_code}"
            )
            _set_webhook_status(
                configured=False,
                public_url_configured=True,
                host=host,
                last_error=str(description or "setWebhook failed")[:200],
            )
            logger.error(
                "Telegram command webhook registration failed: host={} http={}",
                host,
                response.status_code,
            )
            return False

        # Command-menu failure should not disable an otherwise working webhook.
        try:
            requests.post(
                f"{TELEGRAM_API_ROOT}/bot{token}/setMyCommands",
                json={
                    "commands": [
                        {"command": "scan", "description": "Scan for qualified setups"},
                        {"command": "status", "description": "Show bot and scan status"},
                        {"command": "help", "description": "Show command examples"},
                        {"command": "chatid", "description": "Show this chat's numeric ID"},
                    ]
                },
                timeout=(5, timeout),
            )
        except requests.RequestException:
            logger.warning("Telegram command menu update failed; webhook remains active")

        _set_webhook_status(
            configured=True,
            public_url_configured=True,
            host=host,
            last_error=None,
        )
        logger.info("Telegram command webhook registered: host={}", host)
        return True
    except requests.RequestException as exc:
        _set_webhook_status(
            configured=False,
            public_url_configured=True,
            host=host,
            last_error=type(exc).__name__,
        )
        logger.error(
            "Telegram command webhook registration failed: host={} error_type={}",
            host,
            type(exc).__name__,
        )
        return False


def _command_help() -> str:
    return (
        "🤖 <b>Perpetual Pro commands</b>\n\n"
        "<code>/scan</code> — scan the configured watchlist\n"
        "<code>/scan BTC ETH SOL</code> — scan selected markets\n"
        "<code>/scan 1h BTC ETH</code> — scan selected markets on 1h\n"
        "<code>/status</code> — show scan availability\n"
        "<code>/chatid</code> — show this chat's numeric ID\n"
        "<code>/help</code> — show this guide\n\n"
        "Only qualified, prop-safe setups are returned. If none pass, "
        "the bot will tell you to stand aside."
    )


def parse_scan_command(text: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    """Parse optional timeframe and up to 20 symbols from a /scan command."""
    pieces = [part for part in re.split(r"[\s,]+", (text or "").strip()) if part]
    args = pieces[1:]
    timeframe: Optional[str] = None
    if args and args[0].lower() in _VALID_TIMEFRAMES:
        timeframe = args.pop(0).lower()
    symbols: List[str] = []
    for raw in args:
        symbol = raw.upper()
        if not _VALID_SYMBOL.fullmatch(symbol):
            return timeframe, [], f"Invalid symbol: {raw}"
        if symbol not in symbols:
            symbols.append(symbol)
    if len(symbols) > 20:
        return timeframe, [], "Use no more than 20 symbols per scan."
    return timeframe, symbols, None


def get_telegram_command_chat_ids(
    configured_chat_id: Optional[str] = None,
) -> List[str]:
    """Return explicit command chats, falling back to the alert destination."""
    raw = (os.getenv("TELEGRAM_COMMAND_CHAT_IDS") or "").strip()
    if raw:
        candidates = re.split(r"[\s,;]+", raw)
    else:
        fallback = configured_chat_id
        if fallback is None:
            _, fallback = get_telegram_credentials()
        candidates = [fallback or ""]

    command_chats: List[str] = []
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value and value not in command_chats:
            command_chats.append(value)
    return command_chats


def _message_from_update(update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    return message if isinstance(message.get("chat"), dict) else None


def _authorized_message(
    update: Dict[str, Any],
    configured_chat_id: str,
) -> Optional[Dict[str, Any]]:
    message = _message_from_update(update)
    if message is None:
        return None
    chat = message.get("chat")
    assert isinstance(chat, dict)
    incoming_id = str(chat.get("id") or "")
    username = str(chat.get("username") or "")
    allowed_chats = get_telegram_command_chat_ids(configured_chat_id)
    authorized = any(
        (
            allowed.startswith("@")
            and username
            and username.lower() == allowed[1:].lower()
        )
        or (not allowed.startswith("@") and incoming_id == allowed)
        for allowed in allowed_chats
    )
    if not authorized:
        masked = f"…{incoming_id[-4:]}" if incoming_id else "missing"
        logger.warning(
            "Ignored Telegram command from unauthorized chat={} allowed_count={}",
            masked,
            len(allowed_chats),
        )
        return None
    return message


def process_telegram_update(update: Dict[str, Any], config: AppConfig) -> Dict[str, Any]:
    """Handle one authorized Telegram update after the webhook response returns."""
    incoming_message = _message_from_update(update)
    if incoming_message is None:
        return {"ok": True, "handled": False, "reason": "unsupported_update"}
    incoming_chat = incoming_message.get("chat")
    assert isinstance(incoming_chat, dict)
    incoming_chat_id = str(incoming_chat.get("id") or "").strip()
    incoming_text = str(incoming_message.get("text") or "").strip()
    incoming_command = (
        incoming_text.split(maxsplit=1)[0].split("@", 1)[0].lower()
        if incoming_text.startswith("/")
        else ""
    )

    # Safe bootstrap command: it only reveals the requesting chat's own ID.
    if incoming_command == "/chatid" and incoming_chat_id:
        delivery = send_telegram_message_detailed(
            "🆔 This chat ID is:\n"
            f"<code>{html.escape(incoming_chat_id)}</code>\n\n"
            "Set it in <code>TELEGRAM_COMMAND_CHAT_IDS</code> to authorize commands.",
            chat_id=incoming_chat_id,
            parse_mode="HTML",
        )
        return {
            "ok": bool(delivery.get("ok")),
            "handled": True,
            "command": incoming_command,
        }

    _, configured_chat = get_telegram_credentials()
    message = _authorized_message(update, configured_chat)
    if message is None:
        return {"ok": True, "handled": False, "reason": "unauthorized_or_unsupported"}
    text = str(message.get("text") or "").strip()
    if not text.startswith("/"):
        return {"ok": True, "handled": False, "reason": "not_a_command"}
    command = text.split(maxsplit=1)[0].split("@", 1)[0].lower()

    if command in {"/start", "/help"}:
        delivery = send_telegram_message_detailed(
            _command_help(),
            chat_id=incoming_chat_id,
            parse_mode="HTML",
        )
        return {"ok": bool(delivery.get("ok")), "handled": True, "command": command}

    if command == "/status":
        scheduler = get_scheduler_status()
        webhook = get_telegram_webhook_status()
        state = "busy" if scan_in_progress() else "ready"
        next_run = scheduler.get("next_run_at") or "not scheduled"
        message_text = (
            "🟢 <b>Perpetual Pro bot is online</b>\n"
            f"Scan worker: <b>{state}</b>\n"
            f"Webhook: <b>{'active' if webhook.get('configured') else 'inactive'}</b>\n"
            f"Next scheduled scan: <code>{html.escape(str(next_run))}</code>"
        )
        delivery = send_telegram_message_detailed(
            message_text,
            chat_id=incoming_chat_id,
            parse_mode="HTML",
        )
        return {"ok": bool(delivery.get("ok")), "handled": True, "command": command}

    if command != "/scan":
        delivery = send_telegram_message_detailed(
            "Unknown command.\n\n" + _command_help(),
            chat_id=incoming_chat_id,
            parse_mode="HTML",
        )
        return {"ok": bool(delivery.get("ok")), "handled": True, "command": command}

    timeframe, symbols, error = parse_scan_command(text)
    if error:
        delivery = send_telegram_message_detailed(
            f"⚠️ {html.escape(error)}\n\n{_command_help()}",
            chat_id=incoming_chat_id,
            parse_mode="HTML",
        )
        return {"ok": bool(delivery.get("ok")), "handled": True, "command": command}
    if scan_in_progress():
        delivery = send_telegram_message_detailed(
            "⏳ A scan is already running. Try <code>/scan</code> again shortly.",
            chat_id=incoming_chat_id,
            parse_mode="HTML",
        )
        return {
            "ok": bool(delivery.get("ok")),
            "handled": True,
            "command": command,
            "busy": True,
        }

    requested = ", ".join(symbols) if symbols else "configured watchlist"
    selected_tf = timeframe or config.scheduler.timeframe or "15m"
    send_telegram_message_detailed(
        "🔎 <b>Scan started</b>\n"
        f"Markets: {html.escape(requested)}\n"
        f"Timeframe: <b>{html.escape(selected_tf)}</b>\n"
        "I’ll send qualified chart setups—or a stand-aside result—when it finishes.",
        chat_id=incoming_chat_id,
        parse_mode="HTML",
    )
    try:
        result = run_scheduled_scan_once(
            config,
            slot_label="Telegram on-demand scan",
            send=True,
            symbols=symbols or None,
            timeframe=timeframe,
            notify_on_empty=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Telegram on-demand scan failed: {}", type(exc).__name__)
        send_telegram_message_detailed(
            "❌ <b>Scan failed</b>\nThe market data request did not complete. "
            "Please try again shortly.",
            chat_id=incoming_chat_id,
            parse_mode="HTML",
        )
        return {
            "ok": False,
            "handled": True,
            "command": command,
            "error": type(exc).__name__,
        }
    if result.get("error") == "scan_in_progress":
        send_telegram_message_detailed(
            "⏳ Another scan started first. Try <code>/scan</code> again shortly.",
            chat_id=incoming_chat_id,
            parse_mode="HTML",
        )
    elif not result.get("ok") and not result.get("telegram_sent"):
        send_telegram_message_detailed(
            "❌ <b>Scan could not complete</b>\nPlease try again shortly.",
            chat_id=incoming_chat_id,
            parse_mode="HTML",
        )
    return {
        "ok": bool(result.get("ok")),
        "handled": True,
        "command": command,
        "scanned": result.get("scanned"),
        "alert_count": result.get("alert_count"),
        "delivery_status": result.get("telegram_delivery_status"),
    }
