"""Stable service identity and collision-proof experiment namespaces."""

from __future__ import annotations

import os
import re


STRICT_VARIANT = "strict"
LEGACY_VARIANT = "legacy"
VALID_VARIANTS = {STRICT_VARIANT, LEGACY_VARIANT}


def bot_variant() -> str:
    value = str(os.getenv("BOT_VARIANT") or STRICT_VARIANT).strip().lower()
    return value if value in VALID_VARIANTS else STRICT_VARIANT


def is_legacy_comparison() -> bool:
    return bot_variant() == LEGACY_VARIANT


def bot_namespace() -> str:
    fallback = (
        "perpetual_pro_legacy_v1"
        if is_legacy_comparison()
        else "perpetual_pro_strict"
    )
    raw = str(os.getenv("BOT_NAMESPACE") or fallback).strip().lower()
    normalized = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    return normalized or fallback


def experiment_id() -> str:
    value = str(os.getenv("EXPERIMENT_ID") or "strict_vs_legacy_v1").strip()
    return value or "strict_vs_legacy_v1"


def namespace_token() -> str:
    """Short stable token suitable for IDs and database unique keys."""
    if not is_legacy_comparison():
        return "strict"
    return re.sub(r"[^a-z0-9]+", "_", bot_namespace()).strip("_")[:32]


def namespaced_id(prefix: str, value: str) -> str:
    """Preserve strict historical IDs while isolating Legacy identifiers."""
    clean_prefix = str(prefix or "id").strip("_")
    clean_value = str(value or "").strip("_")
    if not is_legacy_comparison():
        return f"{clean_prefix}_{clean_value}"
    return f"legacy_{clean_prefix}_{clean_value}"


def identity_metadata() -> dict:
    return {
        "bot_variant": bot_variant(),
        "bot_namespace": bot_namespace(),
        "experiment_id": experiment_id(),
    }
