"""Trusted database relation selection for the comparison experiment.

The strict production build continues using ``public``.  Legacy uses a
dedicated additive schema so a paused strict build cannot recover Legacy
lifecycle state or scheduler cursors when it is resumed.
"""

from __future__ import annotations

from src.experiments.identity import is_legacy_comparison


def operational_schema() -> str:
    return "legacy_comparison" if is_legacy_comparison() else "public"


def operational_relation(table: str) -> str:
    allowed = {
        "tracked_signals",
        "signal_lifecycle_events",
        "telegram_notification_ledger",
        "scheduler_runs",
        "scheduler_run_deliveries",
    }
    if table not in allowed:
        raise ValueError(f"Unsupported operational table: {table}")
    return f"{operational_schema()}.{table}"
