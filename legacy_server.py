"""Isolated Railway entry point for Perpetual Pro Legacy comparison."""

from __future__ import annotations

import os


def _require(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required Legacy environment variable: {name}")
    return value


# Fail closed whenever delivery is enabled: Legacy never falls back to the
# strict/public Telegram identity. Guarded deployments may boot with Telegram
# disabled so readiness and database isolation can be verified first.
os.environ["BOT_VARIANT"] = "legacy"
os.environ.setdefault("BOT_NAMESPACE", "perpetual_pro_legacy_v1")
os.environ.setdefault("EXPERIMENT_ID", "strict_vs_legacy_v1")
os.environ["DELIVERY_MODE"] = "private_beta"
telegram_enabled = str(os.getenv("TELEGRAM_ENABLED", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
legacy_token = str(os.getenv("LEGACY_TELEGRAM_BOT_TOKEN") or "").strip()
legacy_recipients = str(os.getenv("LEGACY_BETA_CHAT_IDS") or "").strip()
if telegram_enabled:
    legacy_token = _require("LEGACY_TELEGRAM_BOT_TOKEN")
    legacy_recipients = _require("LEGACY_BETA_CHAT_IDS")
if legacy_token:
    os.environ["TELEGRAM_BOT_TOKEN"] = legacy_token
if legacy_recipients:
    os.environ["PRIVATE_BETA_CHAT_IDS"] = legacy_recipients
    os.environ["TELEGRAM_CHAT_ID"] = legacy_recipients.split(",")[0].strip()
if os.getenv("LEGACY_TELEGRAM_COMMAND_CHAT_IDS"):
    os.environ["TELEGRAM_COMMAND_CHAT_IDS"] = str(
        os.getenv("LEGACY_TELEGRAM_COMMAND_CHAT_IDS")
    )
if os.getenv("LEGACY_TELEGRAM_OPERATOR_CHAT_IDS"):
    os.environ["TELEGRAM_OPERATOR_CHAT_IDS"] = str(
        os.getenv("LEGACY_TELEGRAM_OPERATOR_CHAT_IDS")
    )
if os.getenv("LEGACY_TELEGRAM_WEBHOOK_URL"):
    os.environ["TELEGRAM_WEBHOOK_URL"] = str(
        os.getenv("LEGACY_TELEGRAM_WEBHOOK_URL")
    )

from main_server import app  # noqa: E402  (environment must be isolated first)

__all__ = ["app"]
