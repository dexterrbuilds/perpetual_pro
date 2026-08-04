"""Durable scheduled-run audit and delivery ledger.

The repository stores compact operational summaries only. Candidate features
remain in the outcome journal and signal lifecycle state remains in the durable
tracker tables.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

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


class SchedulerRunRepository:
    """Connection-per-operation repository for low-frequency scheduler runs."""

    def __init__(self, database_url: str) -> None:
        self.database_url = str(database_url or "").strip()
        self.enabled = bool(self.database_url and psycopg is not None)

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("Scheduler run database is not configured")
        return psycopg.connect(
            self.database_url,
            row_factory=dict_row,
            connect_timeout=8,
        )

    def claim(
        self,
        *,
        run_id: str,
        source: str,
        slot_label: str,
        scheduled_for: Optional[str],
        symbols_requested: int,
    ) -> Optional[bool]:
        """Insert one run lease; False means this durable run already exists."""
        if not self.enabled:
            return None
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into public.scheduler_runs (
                      run_id, source, slot_label, scheduled_for, started_at,
                      status, symbols_requested
                    ) values (%s,%s,%s,%s,now(),'running',%s)
                    on conflict (run_id) do nothing
                    """,
                    (
                        str(run_id),
                        str(source),
                        str(slot_label or "scan"),
                        scheduled_for,
                        int(symbols_requested),
                    ),
                )
                claimed = cursor.rowcount == 1
                connection.commit()
            return claimed
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Scheduled-run durable claim failed: error_type={}",
                type(exc).__name__,
            )
            return None

    def finish(
        self,
        *,
        run_id: str,
        status: str,
        summary: Mapping[str, Any],
        deliveries: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        if not self.enabled:
            return False
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    update public.scheduler_runs set
                      completed_at=now(), status=%s,
                      symbols_analyzed=%s, symbol_failures=%s,
                      eligible_candidates=%s, revalidated_candidates=%s,
                      rejected_candidates=%s, scan_duration_seconds=%s,
                      max_ticker_age_seconds=%s, max_orderbook_age_seconds=%s,
                      result_code=%s, result_summary=%s, updated_at=now()
                    where run_id=%s and status='running'
                    """,
                    (
                        str(status),
                        int(summary.get("symbols_analyzed") or 0),
                        int(summary.get("symbol_failures") or 0),
                        int(summary.get("eligible_candidates") or 0),
                        int(summary.get("revalidated_candidates") or 0),
                        int(summary.get("rejected_candidates") or 0),
                        summary.get("scan_duration_seconds"),
                        summary.get("max_ticker_age_seconds"),
                        summary.get("max_orderbook_age_seconds"),
                        str(summary.get("result_code") or status),
                        Jsonb(dict(summary)),
                        str(run_id),
                    ),
                )
                for delivery in deliveries:
                    cursor.execute(
                        """
                        insert into public.scheduler_run_deliveries (
                          idempotency_key, run_id, destination_hash,
                          delivery_type, status, telegram_message_id,
                          attempted_at, delivered_at, error_category
                        ) values (%s,%s,%s,%s,%s,%s,now(),%s,%s)
                        on conflict (idempotency_key) do update set
                          status=excluded.status,
                          telegram_message_id=coalesce(
                            scheduler_run_deliveries.telegram_message_id,
                            excluded.telegram_message_id
                          ),
                          delivered_at=coalesce(
                            scheduler_run_deliveries.delivered_at,
                            excluded.delivered_at
                          ),
                          error_category=excluded.error_category,
                          updated_at=now()
                        """,
                        (
                            str(delivery.get("idempotency_key") or ""),
                            str(run_id),
                            str(delivery.get("destination_hash") or ""),
                            str(delivery.get("delivery_type") or "report"),
                            "delivered" if delivery.get("ok") else "failed",
                            delivery.get("message_id"),
                            datetime.now(UTC) if delivery.get("ok") else None,
                            None if delivery.get("ok") else str(
                                delivery.get("error") or "unknown"
                            )[:80],
                        ),
                    )
                connection.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Scheduled-run durable finish failed: run={} error_type={}",
                str(run_id)[:20],
                type(exc).__name__,
            )
            return False

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Return one durable run without exposing delivery destinations."""
        if not self.enabled:
            return None
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select run_id, source, slot_label, scheduled_for, started_at,
                           completed_at, status, symbols_requested,
                           symbols_analyzed, symbol_failures,
                           eligible_candidates, revalidated_candidates,
                           rejected_candidates, scan_duration_seconds,
                           result_code, result_summary
                    from public.scheduler_runs where run_id=%s
                    """,
                    (str(run_id),),
                )
                row = cursor.fetchone()
            return dict(row) if row else None
        except Exception:
            return None

    def latest_scheduled(self) -> Optional[Dict[str, Any]]:
        """Return the newest expected production window, excluding manual scans."""
        if not self.enabled:
            return None
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select run_id, source, slot_label, scheduled_for, started_at,
                           completed_at, status, symbols_requested,
                           symbols_analyzed, symbol_failures,
                           eligible_candidates, revalidated_candidates,
                           rejected_candidates, scan_duration_seconds,
                           result_code, result_summary
                    from public.scheduler_runs
                    where source='scheduled'
                    order by scheduled_for desc nulls last limit 1
                    """
                )
                row = cursor.fetchone()
            return dict(row) if row else None
        except Exception:
            return None

    def delivery_succeeded(self, idempotency_key: str) -> bool:
        """Whether this semantic signal already reached this destination."""
        if not self.enabled or not idempotency_key:
            return False
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select 1 from public.scheduler_run_deliveries
                    where idempotency_key=%s and status='delivered' limit 1
                    """,
                    (str(idempotency_key),),
                )
                return cursor.fetchone() is not None
        except Exception:
            return False

    def record_delivery(
        self,
        *,
        run_id: str,
        delivery: Mapping[str, Any],
    ) -> bool:
        """Persist one delivery immediately to narrow the restart retry window."""
        if not self.enabled or not delivery.get("idempotency_key"):
            return False
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into public.scheduler_run_deliveries (
                      idempotency_key, run_id, destination_hash,
                      delivery_type, status, telegram_message_id,
                      attempted_at, delivered_at, error_category
                    ) values (%s,%s,%s,%s,%s,%s,now(),%s,%s)
                    on conflict (idempotency_key) do update set
                      status=case
                        when scheduler_run_deliveries.status='delivered'
                          then scheduler_run_deliveries.status
                        else excluded.status
                      end,
                      telegram_message_id=coalesce(
                        scheduler_run_deliveries.telegram_message_id,
                        excluded.telegram_message_id
                      ),
                      delivered_at=coalesce(
                        scheduler_run_deliveries.delivered_at,
                        excluded.delivered_at
                      ),
                      error_category=case
                        when scheduler_run_deliveries.status='delivered'
                          then scheduler_run_deliveries.error_category
                        else excluded.error_category
                      end,
                      updated_at=now()
                    """,
                    (
                        str(delivery.get("idempotency_key")),
                        str(run_id),
                        str(delivery.get("destination_hash") or ""),
                        str(delivery.get("delivery_type") or "report"),
                        "delivered" if delivery.get("ok") else "failed",
                        delivery.get("message_id"),
                        datetime.now(UTC) if delivery.get("ok") else None,
                        None if delivery.get("ok") else str(
                            delivery.get("error") or "unknown"
                        )[:80],
                    ),
                )
                connection.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Immediate scheduler delivery persistence failed: run={} error_type={}",
                str(run_id)[:20],
                type(exc).__name__,
            )
            return False

    def latest(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select run_id, source, slot_label, scheduled_for, started_at,
                           completed_at, status, symbols_requested,
                           symbols_analyzed, symbol_failures,
                           eligible_candidates, revalidated_candidates,
                           rejected_candidates, scan_duration_seconds,
                           result_code
                    from public.scheduler_runs
                    order by started_at desc limit 1
                    """
                )
                row = cursor.fetchone()
            return dict(row) if row else None
        except Exception:
            return None
