"""Durable Supabase lifecycle and Telegram delivery repository.

PostgreSQL is the production source of truth.  SQLite remains a process-local
working cache, but no lifecycle notification is eligible for delivery until
its event and per-destination ledger rows are committed here.
"""

from __future__ import annotations

import hashlib
import random
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from loguru import logger

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover
    psycopg = None
    dict_row = None
    Jsonb = None


UTC = timezone.utc
ACTIVE_STATES = ("pending", "entered")
LIFECYCLE_SCHEMA_VERSION = "1.0"
MAX_NOTIFICATION_ATTEMPTS = 6


def _now() -> datetime:
    return datetime.now(UTC)


def destination_hash(destination: Any) -> str:
    """Stable, non-reversible identifier used in logs and unique constraints."""
    raw = str(destination or "").strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def event_id_for(event: Mapping[str, Any]) -> str:
    existing = str(event.get("event_uid") or "").strip()
    if existing:
        return existing
    material = "|".join(
        (
            str(event.get("signal_id") or ""),
            str(event.get("lifecycle_version") or 0),
            str(event.get("event_type") or ""),
            str(event.get("occurred_at") or ""),
        )
    )
    return "evt_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class LifecycleRepository:
    """Connection-per-operation durable lifecycle repository.

    Short transactions tolerate Railway/Supabase connection recycling and
    avoid sharing connections across the tracker and notification worker.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = str(database_url or "").strip()
        self.enabled = bool(self.database_url and psycopg is not None)
        self._lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "configured": bool(self.database_url),
            "driver_available": psycopg is not None,
            "ready": False,
            "migration_required": False,
            "last_success_at": None,
            "last_error": None,
        }

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def _success(self) -> None:
        with self._lock:
            self._status.update(
                ready=True,
                migration_required=False,
                last_success_at=_now().isoformat(),
                last_error=None,
            )

    def _failure(self, exc: Exception, *, migration_required: bool = False) -> None:
        with self._lock:
            self._status.update(
                ready=False,
                migration_required=bool(migration_required),
                last_error=type(exc).__name__,
            )

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("Durable lifecycle database is not configured")
        return psycopg.connect(
            self.database_url,
            row_factory=dict_row,
            connect_timeout=8,
        )

    def check_ready(self) -> bool:
        if not self.enabled:
            return False
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select
                      to_regclass('public.tracked_signals') as signals,
                      to_regclass('public.signal_lifecycle_events') as events,
                      to_regclass('public.telegram_notification_ledger') as ledger
                    """
                )
                row = cursor.fetchone() or {}
                ready = all(row.get(name) for name in ("signals", "events", "ledger"))
            if ready:
                self._success()
            else:
                with self._lock:
                    self._status.update(ready=False, migration_required=True)
            return ready
        except Exception as exc:  # noqa: BLE001
            self._failure(exc)
            return False

    @staticmethod
    def _state_payload(signal: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(signal)
        payload.pop("_events", None)
        return payload

    def persist_state_and_events(
        self,
        signal: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        """Atomically persist a signal version, events, and delivery ledger."""
        if not self.enabled or not signal.get("id"):
            return False
        row = dict(signal.get("row") or {})
        setup_type = str(
            row.get("execution_setup_type") or row.get("setup_type") or "unknown"
        )
        entry_mode = str(row.get("entry_mode") or row.get("entry_status") or "retest")
        confirmation_pending = str(row.get("entry_status") or "").lower() in {
            "ready",
            "confirmation_pending",
        } and str(signal.get("status")) == "pending"
        targets = list(signal.get("take_profits") or [])
        allocations = [max(0.0, float(value)) for value in list(row.get("target_allocations") or [])]
        if len(allocations) != len(targets) or sum(allocations) <= 0:
            allocations = ([1.0 / len(targets)] * len(targets)) if targets else []
        else:
            total_allocation = sum(allocations)
            allocations = [value / total_allocation for value in allocations]
        highest_tp = int(signal.get("highest_tp") or 0)
        remaining_size = max(0.0, 1.0 - sum(allocations[:highest_tp]))
        params = {
            "signal_id": signal.get("id"),
            "candidate_id": row.get("candidate_id"),
            "fingerprint": signal.get("fingerprint"),
            "symbol": signal.get("symbol"),
            "exchange_id": signal.get("exchange_id"),
            "direction": signal.get("direction"),
            "timeframe": signal.get("timeframe"),
            "source": signal.get("source") or "telegram",
            "status": signal.get("status"),
            "setup_type": setup_type,
            "entry_mode": entry_mode,
            "confirmation_pending": confirmation_pending,
            "generated_at": signal.get("generated_at"),
            "valid_until": signal.get("valid_until"),
            "entered_at": signal.get("entered_at"),
            "terminal_at": signal.get("terminal_at"),
            "hold_until": signal.get("hold_until"),
            "entry_low": signal.get("entry_low"),
            "entry_high": signal.get("entry_high"),
            "entry_mid": signal.get("entry_mid"),
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "take_profits": Jsonb(targets),
            "target_allocations": Jsonb(allocations),
            "highest_tp": highest_tp,
            "remaining_size": remaining_size,
            "protected": bool(
                signal.get("protected") or highest_tp >= 1
            ),
            "realized_r": float(signal.get("realized_r") or 0),
            "mfe_r": float(signal.get("mfe_r") or 0),
            "mae_r": float(signal.get("mae_r") or 0),
            "slippage_bps": signal.get("slippage_bps"),
            "last_price": signal.get("last_price"),
            "last_price_at": signal.get("last_price_at"),
            "previous_price": signal.get("previous_price"),
            "previous_price_at": signal.get("previous_price_at"),
            "last_processed_candle_at": signal.get("last_processed_candle_at"),
            "ordering_policy": signal.get("ordering_policy") or "observed_segment_v1",
            "ambiguous_gap": str(signal.get("status")) == "ambiguous_gap",
            "technical_success": signal.get("technical_success"),
            "terminal_reason": signal.get("terminal_reason"),
            "lifecycle_version": int(signal.get("lifecycle_version") or 0),
            "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
            "feature_schema_version": row.get("feature_schema_version") or "3.0",
            "execution_policy_version": row.get("execution_policy_version"),
            "rank_policy_version": row.get("rank_policy_version"),
            "signal_payload": Jsonb(row),
            "lifecycle_state": Jsonb(self._state_payload(signal)),
        }
        upsert = """
            insert into public.tracked_signals (
              signal_id, candidate_id, fingerprint, symbol, exchange_id, direction,
              timeframe, source, status, setup_type, entry_mode,
              confirmation_pending, generated_at, valid_until, entered_at,
              terminal_at, hold_until, entry_low, entry_high, entry_mid,
              entry_price, stop_loss, take_profits, target_allocations, highest_tp,
              remaining_size, protected, realized_r, mfe_r, mae_r, slippage_bps,
              last_price, last_price_at, previous_price, previous_price_at,
              last_processed_candle_at, ordering_policy, ambiguous_gap,
              technical_success, terminal_reason, lifecycle_version,
              lifecycle_schema_version, feature_schema_version,
              execution_policy_version, rank_policy_version, signal_payload,
              lifecycle_state, updated_at
            ) values (
              %(signal_id)s, %(candidate_id)s, %(fingerprint)s, %(symbol)s,
              %(exchange_id)s, %(direction)s, %(timeframe)s, %(source)s,
              %(status)s, %(setup_type)s, %(entry_mode)s,
              %(confirmation_pending)s, %(generated_at)s, %(valid_until)s,
              %(entered_at)s, %(terminal_at)s, %(hold_until)s, %(entry_low)s,
              %(entry_high)s, %(entry_mid)s, %(entry_price)s, %(stop_loss)s,
              %(take_profits)s, %(target_allocations)s, %(highest_tp)s,
              %(remaining_size)s, %(protected)s, %(realized_r)s, %(mfe_r)s,
              %(mae_r)s, %(slippage_bps)s, %(last_price)s, %(last_price_at)s,
              %(previous_price)s, %(previous_price_at)s,
              %(last_processed_candle_at)s, %(ordering_policy)s, %(ambiguous_gap)s,
              %(technical_success)s, %(terminal_reason)s, %(lifecycle_version)s,
              %(lifecycle_schema_version)s, %(feature_schema_version)s,
              %(execution_policy_version)s, %(rank_policy_version)s,
              %(signal_payload)s, %(lifecycle_state)s, now()
            )
            on conflict (signal_id) do update set
              candidate_id=coalesce(excluded.candidate_id, tracked_signals.candidate_id),
              status=excluded.status, confirmation_pending=excluded.confirmation_pending,
              entered_at=excluded.entered_at, terminal_at=excluded.terminal_at,
              hold_until=excluded.hold_until, entry_price=excluded.entry_price,
              highest_tp=excluded.highest_tp, remaining_size=excluded.remaining_size,
              protected=excluded.protected, realized_r=excluded.realized_r,
              mfe_r=excluded.mfe_r, mae_r=excluded.mae_r,
              slippage_bps=excluded.slippage_bps, last_price=excluded.last_price,
              last_price_at=excluded.last_price_at,
              previous_price=excluded.previous_price,
              previous_price_at=excluded.previous_price_at,
              last_processed_candle_at=excluded.last_processed_candle_at,
              ambiguous_gap=excluded.ambiguous_gap,
              technical_success=excluded.technical_success,
              terminal_reason=excluded.terminal_reason,
              lifecycle_version=excluded.lifecycle_version,
              lifecycle_schema_version=excluded.lifecycle_schema_version,
              feature_schema_version=excluded.feature_schema_version,
              execution_policy_version=excluded.execution_policy_version,
              rank_policy_version=excluded.rank_policy_version,
              signal_payload=excluded.signal_payload,
              lifecycle_state=excluded.lifecycle_state, updated_at=now()
            where excluded.lifecycle_version >= tracked_signals.lifecycle_version
        """
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(upsert, params)
                for item in events:
                    event = dict(item)
                    event_uid = event_id_for(event)
                    cursor.execute(
                        """
                        insert into public.signal_lifecycle_events (
                          event_id, signal_id, lifecycle_version, event_type,
                          from_status, to_status, occurred_at, price, payload,
                          notification_required
                        ) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        on conflict (event_id) do nothing
                        """,
                        (
                            event_uid,
                            signal.get("id"),
                            int(event.get("lifecycle_version") or 0),
                            event.get("event_type"),
                            event.get("from_status"),
                            event.get("to_status") or signal.get("status"),
                            event.get("occurred_at"),
                            event.get("price"),
                            Jsonb(
                                {
                                    **dict(event.get("payload") or {}),
                                    "destinations": list(event.get("destinations") or []),
                                }
                            ),
                            bool(event.get("notify", True)),
                        ),
                    )
                    for destination in set(event.get("destinations") or []):
                            digest = destination_hash(destination)
                            if event.get("notify", True):
                                cursor.execute(
                                    """
                                insert into public.telegram_notification_ledger (
                                  idempotency_key, event_id, signal_id,
                                  destination_hash, status, attempt_count,
                                  next_retry_at
                                ) values (%s,%s,%s,%s,'queued',0,now())
                                on conflict (event_id, destination_hash) do nothing
                                """,
                                    (f"{event_uid}:{digest}", event_uid, signal.get("id"), digest),
                                )
                            else:
                                cursor.execute(
                                    """
                                    insert into public.telegram_notification_ledger (
                                      idempotency_key, event_id, signal_id,
                                      destination_hash, status, attempt_count,
                                      last_attempt_at, delivered_at, next_retry_at
                                    ) values (%s,%s,%s,%s,'delivered',1,now(),now(),now())
                                    on conflict (event_id, destination_hash) do nothing
                                    """,
                                    (f"{event_uid}:{digest}", event_uid, signal.get("id"), digest),
                                )
                if row.get("candidate_id"):
                    cursor.execute(
                        "update public.signal_candidates set delivered=true where id=%s",
                        (row.get("candidate_id"),),
                    )
                connection.commit()
            self._success()
            return True
        except Exception as exc:  # noqa: BLE001
            self._failure(exc, migration_required="UndefinedColumn" in type(exc).__name__)
            logger.error("Durable lifecycle commit failed: error_type={}", type(exc).__name__)
            return False

    def load_active(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select signal_id, lifecycle_state, lifecycle_schema_version,
                           feature_schema_version, execution_policy_version,
                           rank_policy_version
                    from public.tracked_signals
                    where status = any(%s)
                    order by generated_at
                    """,
                    (list(ACTIVE_STATES),),
                )
                rows = cursor.fetchall()
            self._success()
            records: List[Dict[str, Any]] = []
            for row in rows:
                state = dict(row.get("lifecycle_state") or {})
                state.setdefault("id", row.get("signal_id"))
                state["_lifecycle_schema_version"] = row.get(
                    "lifecycle_schema_version"
                )
                state["_feature_schema_version"] = row.get("feature_schema_version")
                state["_execution_policy_version"] = row.get(
                    "execution_policy_version"
                )
                state["_rank_policy_version"] = row.get("rank_policy_version")
                records.append(state)
            return records
        except Exception as exc:  # noqa: BLE001
            self._failure(exc)
            raise

    def load_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Load one durable state for controlled recovery verification."""
        if not self.enabled:
            return None
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select lifecycle_state, lifecycle_schema_version,
                           feature_schema_version, execution_policy_version,
                           rank_policy_version
                    from public.tracked_signals where signal_id=%s
                    """,
                    (str(signal_id),),
                )
                row = cursor.fetchone()
            self._success()
            if not row:
                return None
            state = dict(row.get("lifecycle_state") or {})
            state["_lifecycle_schema_version"] = row.get("lifecycle_schema_version")
            state["_feature_schema_version"] = row.get("feature_schema_version")
            state["_execution_policy_version"] = row.get("execution_policy_version")
            state["_rank_policy_version"] = row.get("rank_policy_version")
            return state
        except Exception as exc:  # noqa: BLE001
            self._failure(exc)
            return None

    def notification_summary(self, signal_id: str) -> Dict[str, int]:
        """Return non-sensitive ledger counts for deployment verification."""
        if not self.enabled:
            return {}
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select status, count(*) as count
                    from public.telegram_notification_ledger
                    where signal_id=%s group by status
                    """,
                    (str(signal_id),),
                )
                rows = cursor.fetchall()
            self._success()
            return {str(row["status"]): int(row["count"]) for row in rows}
        except Exception as exc:  # noqa: BLE001
            self._failure(exc)
            return {}

    def claim_notifications(self, *, limit: int = 25) -> List[Dict[str, Any]]:
        """Lease due deliveries so concurrent workers cannot send the same row."""
        if not self.enabled:
            return []
        lease_cutoff = _now() - timedelta(minutes=5)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    with due as (
                      select id from public.telegram_notification_ledger
                      where (
                        status in ('queued','retry') and next_retry_at <= now()
                      ) or (status='sending' and last_attempt_at < %s)
                      order by next_retry_at, id
                      for update skip locked
                      limit %s
                    )
                    update public.telegram_notification_ledger n
                    set status='sending', attempt_count=n.attempt_count+1,
                        last_attempt_at=now(), updated_at=now()
                    from due where n.id=due.id
                    returning n.*
                    """,
                    (lease_cutoff, int(limit)),
                )
                jobs = [dict(row) for row in cursor.fetchall()]
                for job in jobs:
                    cursor.execute(
                        """
                        select e.event_type, e.occurred_at, e.price, e.payload,
                               t.lifecycle_state
                        from public.signal_lifecycle_events e
                        join public.tracked_signals t on t.signal_id=e.signal_id
                        where e.event_id=%s
                        """,
                        (job["event_id"],),
                    )
                    context = cursor.fetchone() or {}
                    job.update(context)
                connection.commit()
            self._success()
            return jobs
        except Exception as exc:  # noqa: BLE001
            self._failure(exc)
            return []

    def finish_notification(
        self,
        ledger_id: int,
        *,
        delivered: bool,
        message_id: Optional[int] = None,
        error_category: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
    ) -> bool:
        if not self.enabled:
            return False
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "select attempt_count from public.telegram_notification_ledger where id=%s for update",
                    (int(ledger_id),),
                )
                row = cursor.fetchone()
                if not row:
                    return False
                attempts = int(row.get("attempt_count") or 0)
                if delivered:
                    status = "delivered"
                    next_retry = None
                elif attempts >= MAX_NOTIFICATION_ATTEMPTS:
                    status = "dead_letter"
                    next_retry = None
                else:
                    status = "retry"
                    base = max(float(retry_after_seconds or 0), min(1800.0, 30.0 * (2 ** max(0, attempts - 1))))
                    next_retry = _now() + timedelta(seconds=base + random.uniform(0, min(15.0, base * 0.1)))
                cursor.execute(
                    """
                    update public.telegram_notification_ledger
                    set status=%s, telegram_message_id=%s, delivered_at=%s,
                        next_retry_at=%s, error_category=%s, updated_at=now()
                    where id=%s and status='sending'
                    """,
                    (
                        status,
                        message_id if delivered else None,
                        _now() if delivered else None,
                        next_retry,
                        None if delivered else str(error_category or "unknown")[:80],
                        int(ledger_id),
                    ),
                )
                changed = cursor.rowcount == 1
                connection.commit()
            self._success()
            return changed
        except Exception as exc:  # noqa: BLE001
            self._failure(exc)
            return False
