#!/usr/bin/env python3
"""Private controlled verification of durable lifecycle recovery.

This script never places an order. It requires an explicitly configured private
Telegram command destination and creates a clearly marked verification signal.
Run ``seed``, restart Railway, then run ``progress <signal-id>``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.data.exchange import ExchangeClient
from src.notify.telegram_bot import get_telegram_command_chat_ids
from src.tracking.durable_repository import LifecycleRepository
from src.tracking.signal_tracker import SignalStore
from src.utils.config import load_config
from src.utils.helpers import safe_float


UTC = timezone.utc


def _private_destination() -> str:
    if not str(os.getenv("TELEGRAM_COMMAND_CHAT_IDS") or "").strip():
        raise RuntimeError("TELEGRAM_COMMAND_CHAT_IDS must explicitly name a private test chat")
    destinations = get_telegram_command_chat_ids("")
    if not destinations:
        raise RuntimeError("No private Telegram command destination is configured")
    return destinations[0]


def _repository(cfg):
    repository = LifecycleRepository(cfg.outcome_scoring.database_url)
    if not repository.check_ready():
        raise RuntimeError("Durable lifecycle migration is not ready")
    return repository


def seed() -> None:
    cfg = load_config()
    repository = _repository(cfg)
    destination = _private_destination()
    client = ExchangeClient(exchange_id=cfg.scheduler.exchange, config=cfg)
    try:
        ticker = client.fetch_ticker("BTC")
        price = safe_float(ticker.get("last") or ticker.get("close"))
    finally:
        client.close()
    if price <= 0:
        raise RuntimeError("Could not obtain a finite public BTC price")
    generated = datetime.now(UTC) + timedelta(minutes=30)
    row = {
        "symbol": "BTC/USDT:USDT",
        "exchange": cfg.scheduler.exchange,
        "direction": "long",
        "primary_tf": "15m",
        "confidence": 100,
        "entry_low": price * 0.998,
        "entry_high": price * 1.002,
        "stop_loss": price * 0.50,
        "take_profits": [price * 1.50, price * 1.60],
        "entry_status": "confirmation_pending",
        "entry_mode": "cmp_confirmation",
        "execution_setup_type": "cmp_confirmation",
        "price": price,
        "signal_generated_at": generated.isoformat(),
        "entry_valid_until": (generated + timedelta(hours=2)).isoformat(),
        "entry_valid_for_minutes": 120,
        "hold_hours_max": 4,
        "feature_schema_version": cfg.outcome_scoring.feature_schema_version,
        "execution_policy_version": cfg.analysis.execution_policy_version,
        "rank_policy_version": "deterministic_rank_v2a.1",
        "verification_fixture": True,
    }
    with tempfile.TemporaryDirectory(prefix="perpetual-pro-lifecycle-") as tmp:
        store = SignalStore(Path(tmp) / "fixture.db", target_allocations=[0.5, 0.5])
        signal, _ = store.register_signal(
            row, [destination], source="production_verification_seed", now=datetime.now(UTC)
        )
        if not repository.persist_state_and_events(
            signal, store.events_for_sync(signal["id"])
        ):
            raise RuntimeError("Could not persist controlled lifecycle seed")
        store.close()
    print(f"seeded signal_id={signal['id']} status={signal['status']} schema=1.0")


def progress(signal_id: str) -> None:
    cfg = load_config()
    repository = _repository(cfg)
    state = repository.load_signal(signal_id)
    if not state or not bool((state.get("row") or {}).get("verification_fixture")):
        raise RuntimeError("Signal is missing or is not a controlled verification fixture")
    generated = datetime.fromisoformat(str(state["generated_at"]).replace("Z", "+00:00"))
    entry = safe_float(state.get("entry_mid"))
    target = safe_float((state.get("take_profits") or [None])[0])
    with tempfile.TemporaryDirectory(prefix="perpetual-pro-lifecycle-") as tmp:
        store = SignalStore(Path(tmp) / "fixture.db", target_allocations=[0.5, 0.5])
        if not store.restore_signal(state):
            raise RuntimeError("Controlled state could not be restored")
        timeline = [
            ("confirmation", lambda: store.process_closed_candle(
                state["symbol"], entry, generated + timedelta(minutes=15),
                exchange_id=state["exchange_id"], high_price=entry, low_price=entry,
            )),
            ("tp1", lambda: store.process_price(
                state["symbol"], target, generated + timedelta(minutes=16),
                source="production_verification", exchange_id=state["exchange_id"],
            )),
            ("protected_close", lambda: store.process_price(
                state["symbol"], entry, generated + timedelta(minutes=17),
                source="production_verification", exchange_id=state["exchange_id"],
            )),
        ]
        for label, transition in timeline:
            if transition() != 1:
                raise RuntimeError(f"Controlled {label} transition did not occur exactly once")
            current = store.signals_for_sync(state["symbol"])[0]
            if not repository.persist_state_and_events(
                current, store.events_for_sync(signal_id)
            ):
                raise RuntimeError(f"Controlled {label} durable commit failed")
        store.close()
    final = repository.load_signal(signal_id) or {}
    print(
        "progressed "
        f"signal_id={signal_id} status={final.get('status')} "
        f"highest_tp={final.get('highest_tp')} ledger={repository.notification_summary(signal_id)}"
    )


def status(signal_id: str) -> None:
    cfg = load_config()
    repository = _repository(cfg)
    state = repository.load_signal(signal_id) or {}
    print(
        f"signal_id={signal_id} status={state.get('status')} "
        f"version={state.get('lifecycle_version')} "
        f"ledger={repository.notification_summary(signal_id)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed")
    for name in ("progress", "status"):
        command = sub.add_parser(name)
        command.add_argument("signal_id")
    args = parser.parse_args()
    if args.command == "seed":
        seed()
    elif args.command == "progress":
        progress(args.signal_id)
    else:
        status(args.signal_id)


if __name__ == "__main__":
    main()
