"""Additive Supabase persistence for rejection analytics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
from typing import Any, Dict, Mapping, Optional, Sequence

from loguru import logger

from src.analytics.rejection import aggregate_rejection_rows

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover
    psycopg = None
    dict_row = None
    Jsonb = None


UTC = timezone.utc


class RejectionAnalyticsRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = str(database_url or "").strip()
        self.enabled = bool(self.database_url and psycopg is not None)
        self.last_error: Optional[str] = None
        self.last_latency_ms: Optional[float] = None

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("Rejection analytics database is not configured")
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
                    select to_regclass('public.scan_rejection_analytics') as scans,
                           to_regclass('public.candidate_rejection_analytics') as candidates
                    """
                )
                row = cursor.fetchone() or {}
            ready = bool(row.get("scans") and row.get("candidates"))
            self.last_error = None if ready else "migration_required"
            return ready
        except Exception as exc:  # noqa: BLE001
            self.last_error = type(exc).__name__
            return False

    def record_scan(
        self,
        scan: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Batch one scan and its candidates; failures never affect trading."""
        if not self.enabled or not scan.get("scan_id"):
            return False
        scan_payload = dict(scan)
        candidate_payloads = [dict(row) for row in candidates]
        started = time.monotonic()
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into public.scan_rejection_analytics (
                      scan_id, trigger_type, started_at, completed_at,
                      requested_symbols, analyzed_symbols, failed_symbols,
                      directional_candidates, eligible_candidates,
                      revalidated_candidates, scan_duration_seconds,
                      analytics_latency_ms, llm_calls, llm_rate_limit_events,
                      public_messages, private_messages, no_quality_result,
                      status, gate_policy_version, feature_schema_version,
                      execution_policy_version, rank_policy_version,
                      build_commit_sha, summary
                    ) values (
                      %(scan_id)s,%(trigger_type)s,%(started_at)s,%(completed_at)s,
                      %(requested_symbols)s,%(analyzed_symbols)s,%(failed_symbols)s,
                      %(directional_candidates)s,%(eligible_candidates)s,
                      %(revalidated_candidates)s,%(scan_duration_seconds)s,
                      %(analytics_latency_ms)s,%(llm_calls)s,%(llm_rate_limit_events)s,
                      %(public_messages)s,%(private_messages)s,%(no_quality_result)s,
                      %(status)s,%(gate_policy_version)s,%(feature_schema_version)s,
                      %(execution_policy_version)s,%(rank_policy_version)s,
                      %(build_commit_sha)s,%(summary)s
                    ) on conflict (scan_id) do update set
                      completed_at=excluded.completed_at,
                      analyzed_symbols=excluded.analyzed_symbols,
                      failed_symbols=excluded.failed_symbols,
                      directional_candidates=excluded.directional_candidates,
                      eligible_candidates=excluded.eligible_candidates,
                      revalidated_candidates=excluded.revalidated_candidates,
                      scan_duration_seconds=excluded.scan_duration_seconds,
                      analytics_latency_ms=excluded.analytics_latency_ms,
                      llm_calls=excluded.llm_calls,
                      llm_rate_limit_events=excluded.llm_rate_limit_events,
                      public_messages=excluded.public_messages,
                      private_messages=excluded.private_messages,
                      no_quality_result=excluded.no_quality_result,
                      status=excluded.status, summary=excluded.summary,
                      updated_at=now()
                    """,
                    {
                        **scan_payload,
                        "requested_symbols": Jsonb(list(scan_payload.get("requested_symbols") or [])),
                        "analyzed_symbols": Jsonb(list(scan_payload.get("analyzed_symbols") or [])),
                        "failed_symbols": Jsonb(list(scan_payload.get("failed_symbols") or [])),
                        "summary": Jsonb(dict(scan_payload.get("summary") or {})),
                    },
                )
                sql = """
                    insert into public.candidate_rejection_analytics (
                      candidate_id, scan_id, analyzed_at, symbol, exchange_id,
                      timeframe, direction, setup_type, feature_schema_version,
                      execution_policy_version, rank_policy_version,
                      gate_policy_version, technical_quality, execution_quality,
                      overall_quality, rank_score, immediate_sl_risk, gross_rr,
                      net_rr, spread_bps, ticker_age_seconds,
                      orderbook_age_seconds, prop_safe, entry_state, target_count,
                      target_feasibility, target_feasibility_summary,
                      stop_quality, data_quality_score, market_quality_ok,
                      chase_distance_atr, confluence_magnitude, lifecycle_state,
                      eligible, primary_rejection_reason, all_rejection_reasons,
                      closest_to_passing_gate, distance_to_eligibility,
                      proximity_label, gate_evaluation
                    ) values (
                      %(candidate_id)s,%(scan_id)s,%(analyzed_at)s,%(symbol)s,
                      %(exchange_id)s,%(timeframe)s,%(direction)s,%(setup_type)s,
                      %(feature_schema_version)s,%(execution_policy_version)s,
                      %(rank_policy_version)s,%(gate_policy_version)s,
                      %(technical_quality)s,%(execution_quality)s,
                      %(overall_quality)s,%(rank_score)s,%(immediate_sl_risk)s,
                      %(gross_rr)s,%(net_rr)s,%(spread_bps)s,
                      %(ticker_age_seconds)s,%(orderbook_age_seconds)s,
                      %(prop_safe)s,%(entry_state)s,%(target_count)s,
                      %(target_feasibility)s,%(target_feasibility_summary)s,
                      %(stop_quality)s,%(data_quality_score)s,%(market_quality_ok)s,
                      %(chase_distance_atr)s,%(confluence_magnitude)s,
                      %(lifecycle_state)s,%(eligible)s,
                      %(primary_rejection_reason)s,%(all_rejection_reasons)s,
                      %(closest_to_passing_gate)s,%(distance_to_eligibility)s,
                      %(proximity_label)s,%(gate_evaluation)s
                    ) on conflict (candidate_id) do update set
                      gate_evaluation=excluded.gate_evaluation,
                      primary_rejection_reason=excluded.primary_rejection_reason,
                      all_rejection_reasons=excluded.all_rejection_reasons,
                      closest_to_passing_gate=excluded.closest_to_passing_gate,
                      distance_to_eligibility=excluded.distance_to_eligibility,
                      proximity_label=excluded.proximity_label,
                      eligible=excluded.eligible, updated_at=now()
                """
                if candidate_payloads:
                    cursor.executemany(
                        sql,
                        [
                            {
                                **row,
                                "gross_rr": Jsonb(list(row.get("gross_rr") or [])),
                                "net_rr": Jsonb(list(row.get("net_rr") or [])),
                                "target_feasibility": Jsonb(list(row.get("target_feasibility") or [])),
                                "all_rejection_reasons": list(row.get("all_rejection_reasons") or []),
                                "gate_evaluation": Jsonb(dict(row.get("gate_evaluation") or {})),
                            }
                            for row in candidate_payloads
                        ],
                    )
                self.last_latency_ms = round(
                    (time.monotonic() - started) * 1000.0, 3
                )
                cursor.execute(
                    """
                    update public.scan_rejection_analytics
                    set analytics_latency_ms=%s, updated_at=now()
                    where scan_id=%s
                    """,
                    (self.last_latency_ms, str(scan.get("scan_id"))),
                )
                connection.commit()
            self.last_error = None
            return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = type(exc).__name__
            logger.error(
                "Rejection analytics persistence failed: scan={} error_type={}",
                str(scan.get("scan_id"))[:20],
                type(exc).__name__,
            )
            return False

    @staticmethod
    def _json_safe(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in dict(row).items()
        }

    def latest_scan(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "select * from public.scan_rejection_analytics order by started_at desc limit 1"
                )
                scan = cursor.fetchone()
                if not scan:
                    return None
                cursor.execute(
                    """
                    select * from public.candidate_rejection_analytics
                    where scan_id=%s order by eligible desc,
                      distance_to_eligibility asc, overall_quality desc
                    """,
                    (scan["scan_id"],),
                )
                candidates = [self._json_safe(row) for row in cursor.fetchall()]
            scan_row = self._json_safe(scan)
            return {
                "scan": scan_row,
                "analytics": aggregate_rejection_rows([scan_row], candidates),
                "candidates": candidates,
            }
        except Exception as exc:  # noqa: BLE001
            self.last_error = type(exc).__name__
            return None

    def finalize_scan(
        self,
        *,
        scan_id: str,
        revalidated_candidates: int,
        public_messages: int,
        private_messages: int,
        no_quality_result: bool,
        summary_patch: Mapping[str, Any],
        candidate_updates: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        """Add delivery/revalidation results without changing scan decisions."""
        if not self.enabled:
            return False
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    update public.scan_rejection_analytics set
                      revalidated_candidates=%s, public_messages=%s,
                      private_messages=%s, no_quality_result=%s,
                      summary=summary || %s, updated_at=now()
                    where scan_id=%s
                    """,
                    (
                        int(revalidated_candidates),
                        int(public_messages),
                        int(private_messages),
                        bool(no_quality_result),
                        Jsonb(dict(summary_patch)),
                        str(scan_id),
                    ),
                )
                for update in candidate_updates:
                    evaluation = dict(update.get("gate_evaluation") or {})
                    if not update.get("candidate_id") or not evaluation:
                        continue
                    cursor.execute(
                        """
                        update public.candidate_rejection_analytics set
                          eligible=%s, primary_rejection_reason=%s,
                          all_rejection_reasons=%s,
                          closest_to_passing_gate=%s,
                          distance_to_eligibility=%s, proximity_label=%s,
                          gate_evaluation=%s, updated_at=now()
                        where candidate_id=%s and scan_id=%s
                        """,
                        (
                            bool(evaluation.get("eligible")),
                            evaluation.get("primary_rejection_reason"),
                            list(evaluation.get("all_rejection_reasons") or []),
                            evaluation.get("closest_to_passing_gate"),
                            float(evaluation.get("distance_to_eligibility") or 0),
                            evaluation.get("proximity_label") or "FAR_FROM_ELIGIBLE",
                            Jsonb(evaluation),
                            str(update.get("candidate_id")),
                            str(scan_id),
                        ),
                    )
                connection.commit()
            self.last_error = None
            return True
        except Exception as exc:  # noqa: BLE001
            self.last_error = type(exc).__name__
            logger.error(
                "Rejection analytics finalization failed: scan={} error_type={}",
                str(scan_id)[:20],
                type(exc).__name__,
            )
            return False

    def range_summary(
        self,
        *,
        start: datetime,
        end: datetime,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return aggregate_rejection_rows([], [])
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select * from public.scan_rejection_analytics
                    where started_at >= %s and started_at < %s
                    order by started_at
                    """,
                    (start, end),
                )
                scans = [self._json_safe(row) for row in cursor.fetchall()]
                scan_ids = [row["scan_id"] for row in scans]
                candidates = []
                if scan_ids:
                    cursor.execute(
                        """
                        select * from public.candidate_rejection_analytics
                        where scan_id = any(%s)
                        order by analyzed_at
                        """,
                        (scan_ids,),
                    )
                    candidates = [self._json_safe(row) for row in cursor.fetchall()]
            return {
                "start": start.isoformat(),
                "end": end.isoformat(),
                **aggregate_rejection_rows(scans, candidates),
            }
        except Exception as exc:  # noqa: BLE001
            self.last_error = type(exc).__name__
            return {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "error": "analytics_query_failed",
            }

    def summary_for_hours(self, hours: int) -> Dict[str, Any]:
        end = datetime.now(UTC)
        return self.range_summary(
            start=end - timedelta(hours=max(1, min(int(hours), 24 * 31))),
            end=end,
        )

    def candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select * from public.candidate_rejection_analytics
                    where candidate_id=%s
                    """,
                    (str(candidate_id),),
                )
                row = cursor.fetchone()
            return self._json_safe(row) if row else None
        except Exception as exc:  # noqa: BLE001
            self.last_error = type(exc).__name__
            return None
