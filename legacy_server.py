"""Isolated Railway entry point for Perpetual Pro Legacy comparison."""

from __future__ import annotations

import os


def _require(name: str) -> str:
    value = str(os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required Legacy environment variable: {name}")
    return value


# Fail closed: Legacy never falls back to the strict/public Telegram identity.
os.environ["BOT_VARIANT"] = "legacy"
os.environ.setdefault("BOT_NAMESPACE", "perpetual_pro_legacy_v1")
os.environ.setdefault("EXPERIMENT_ID", "strict_vs_legacy_v1")
os.environ["DELIVERY_MODE"] = "private_beta"
os.environ["TELEGRAM_BOT_TOKEN"] = _require("LEGACY_TELEGRAM_BOT_TOKEN")
os.environ["PRIVATE_BETA_CHAT_IDS"] = _require("LEGACY_BETA_CHAT_IDS")
os.environ["TELEGRAM_CHAT_ID"] = os.environ["PRIVATE_BETA_CHAT_IDS"].split(",")[0].strip()
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
