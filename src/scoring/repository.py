"""Supabase PostgreSQL persistence for candidates, outcomes, and model artifacts."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from loguru import logger

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError:  # pragma: no cover - dependency is installed in production
    psycopg = None
    dict_row = None
    Jsonb = None


TERMINAL_STATUSES = {
    "completed",
    "stopped",
    "missed",
    "expired",
    "invalidated",
    "time_exit",
    "ambiguous_gap",
}


class OutcomeRepository:
    """Small, thread-safe connection-per-operation Postgres repository.

    The scheduled workload is tiny, so short scoped connections are safer than
    retaining a stale connection across Railway/Supabase restarts.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = str(database_url or "").strip()
        self.enabled = bool(self.database_url and psycopg is not None)
        self._status_lock = threading.Lock()
        self._status: Dict[str, Any] = {
            "configured": bool(self.database_url),
            "driver_available": psycopg is not None,
            "ready": False,
            "migration_required": False,
            "last_success_at": None,
            "last_error": None,
        }

    def status(self) -> Dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def check_ready(self) -> bool:
        if not self.enabled:
            return False
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "select to_regclass('public.signal_candidates') as table_name"
                    )
                    row = cursor.fetchone()
                    ready = bool(row and row.get("table_name"))
            self._mark_success(ready=ready, migration_required=not ready)
            return ready
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            return False

    def record_candidates(self, candidates: Iterable[Mapping[str, Any]]) -> int:
        rows = [dict(item) for item in candidates if item.get("id")]
        if not rows or not self.enabled:
            return 0
        sql = """
            insert into public.signal_candidates (
                id, generated_at, symbol, exchange_id, timeframe, direction,
                setup_type, setup_name, feature_schema_version, features,
                decision, production_scores, production_eligible,
                production_rank, shadow_model_version, shadow_scores, source,
                is_directional_candidate
            ) values (
                %(id)s, %(generated_at)s, %(symbol)s, %(exchange_id)s,
                %(timeframe)s, %(direction)s, %(setup_type)s, %(setup_name)s,
                %(feature_schema_version)s, %(features)s, %(decision)s,
                %(production_scores)s, %(production_eligible)s,
                %(production_rank)s, %(shadow_model_version)s,
                %(shadow_scores)s, %(source)s, %(is_directional_candidate)s
            )
            on conflict (id) do update set
                features = excluded.features,
                decision = excluded.decision,
                production_scores = excluded.production_scores,
                production_eligible = excluded.production_eligible,
                production_rank = excluded.production_rank,
                shadow_model_version = excluded.shadow_model_version,
                shadow_scores = excluded.shadow_scores,
                is_directional_candidate = excluded.is_directional_candidate
        """
        payloads: List[Dict[str, Any]] = []
        for row in rows:
            payloads.append(
                {
                    **row,
                    "features": Jsonb(dict(row.get("features") or {})),
                    "decision": Jsonb(dict(row.get("decision") or {})),
                    "production_scores": Jsonb(
                        dict(row.get("production_scores") or {})
                    ),
                    "shadow_scores": (
                        Jsonb(dict(row.get("shadow_scores") or {}))
                        if row.get("shadow_scores") is not None
                        else None
                    ),
                    "shadow_model_version": row.get("shadow_model_version"),
                    "is_directional_candidate": bool(
                        row.get("is_directional_candidate")
                    ),
                }
            )
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.executemany(sql, payloads)
                connection.commit()
            self._mark_success(ready=True, migration_required=False)
            return len(payloads)
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            logger.warning(
                "Outcome journal write failed; live scan continues unblocked: {}",
                type(exc).__name__,
            )
            return 0

    def mark_candidates_delivered(self, candidate_ids: Iterable[str]) -> int:
        ids = [str(value) for value in candidate_ids if str(value or "").strip()]
        if not ids or not self.enabled:
            return 0
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        update public.signal_candidates
                        set delivered = true
                        where id = any(%s)
                        """,
                        (ids,),
                    )
                    count = int(cursor.rowcount or 0)
                connection.commit()
            self._mark_success(ready=True, migration_required=False)
            return count
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            return 0

    def upsert_tracked_signal(self, signal: Mapping[str, Any]) -> bool:
        if not self.enabled or not signal.get("id"):
            return False
        row_payload = dict(signal.get("row") or {})
        candidate_id = row_payload.get("candidate_id")
        targets = list(signal.get("take_profits") or [])
        sql = """
            insert into public.tracked_signals (
                signal_id, candidate_id, symbol, exchange_id, direction,
                timeframe, source, status, generated_at, valid_until,
                entered_at, terminal_at, entry_low, entry_high, entry_mid,
                entry_price, stop_loss, take_profits, highest_tp, realized_r,
                mfe_r, mae_r, slippage_bps, terminal_reason,
                outcome_classification, profitable, profitable_at, level_hits,
                signal_payload,
                updated_at
            ) values (
                %(signal_id)s, %(candidate_id)s, %(symbol)s, %(exchange_id)s,
                %(direction)s, %(timeframe)s, %(source)s, %(status)s,
                %(generated_at)s, %(valid_until)s, %(entered_at)s,
                %(terminal_at)s, %(entry_low)s, %(entry_high)s, %(entry_mid)s,
                %(entry_price)s, %(stop_loss)s, %(take_profits)s,
                %(highest_tp)s, %(realized_r)s, %(mfe_r)s, %(mae_r)s,
                %(slippage_bps)s, %(terminal_reason)s,
                %(outcome_classification)s, %(profitable)s, %(profitable_at)s,
                %(level_hits)s, %(signal_payload)s,
                now()
            )
            on conflict (signal_id) do update set
                candidate_id = coalesce(excluded.candidate_id, tracked_signals.candidate_id),
                status = excluded.status,
                entered_at = excluded.entered_at,
                terminal_at = excluded.terminal_at,
                entry_price = excluded.entry_price,
                highest_tp = excluded.highest_tp,
                realized_r = excluded.realized_r,
                mfe_r = excluded.mfe_r,
                mae_r = excluded.mae_r,
                slippage_bps = excluded.slippage_bps,
                terminal_reason = excluded.terminal_reason,
                outcome_classification = excluded.outcome_classification,
                profitable = excluded.profitable,
                profitable_at = excluded.profitable_at,
                level_hits = excluded.level_hits,
                signal_payload = excluded.signal_payload,
                updated_at = now()
        """
        params = {
            "signal_id": signal.get("id"),
            "candidate_id": candidate_id,
            "symbol": signal.get("symbol"),
            "exchange_id": signal.get("exchange_id"),
            "direction": signal.get("direction"),
            "timeframe": signal.get("timeframe"),
            "source": signal.get("source") or "telegram",
            "status": signal.get("status"),
            "generated_at": signal.get("generated_at"),
            "valid_until": signal.get("valid_until"),
            "entered_at": signal.get("entered_at"),
            "terminal_at": signal.get("terminal_at"),
            "entry_low": signal.get("entry_low"),
            "entry_high": signal.get("entry_high"),
            "entry_mid": signal.get("entry_mid"),
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "take_profits": Jsonb(targets),
            "highest_tp": int(signal.get("highest_tp") or 0),
            "realized_r": float(signal.get("realized_r") or 0),
            "mfe_r": float(signal.get("mfe_r") or 0),
            "mae_r": float(signal.get("mae_r") or 0),
            "slippage_bps": signal.get("slippage_bps"),
            "terminal_reason": signal.get("terminal_reason"),
            "outcome_classification": str(
                signal.get("outcome_classification")
                or (
                    "profitable"
                    if int(signal.get("highest_tp") or 0) >= 1
                    else "active"
                    if str(signal.get("status") or "") == "entered"
                    else "pending_entry"
                    if str(signal.get("status") or "") == "pending"
                    else "ambiguous"
                    if str(signal.get("status") or "") == "ambiguous_gap"
                    else "not_entered"
                    if not signal.get("entered_at")
                    else "not_profitable"
                )
            ),
            "profitable": bool(
                signal.get("profitable")
                or int(signal.get("highest_tp") or 0) >= 1
            ),
            "profitable_at": signal.get("profitable_at"),
            "level_hits": Jsonb(dict(signal.get("level_hits") or {})),
            "signal_payload": Jsonb(row_payload),
        }
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    if candidate_id:
                        cursor.execute(
                            """
                            update public.signal_candidates
                            set delivered = true
                            where id = %s
                            """,
                            (candidate_id,),
                        )
                connection.commit()
            self._mark_success(ready=True, migration_required=False)
            return True
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            logger.warning(
                "Durable tracked-signal mirror failed: {}", type(exc).__name__
            )
            return False

    def upsert_outcome_from_signal(self, signal: Mapping[str, Any]) -> bool:
        """Persist one terminal tracker state as an alert-level training label."""
        status = str(signal.get("status") or "")
        if status not in TERMINAL_STATUSES or not self.enabled:
            return False
        row_payload = dict(signal.get("row") or {})
        candidate_id = str(row_payload.get("candidate_id") or "")
        if not candidate_id:
            return False
        valid_fill = bool(signal.get("entered_at"))
        tp1_hit = int(signal.get("highest_tp") or 0) >= 1
        outcome_classification = str(
            signal.get("outcome_classification")
            or (
                "profitable"
                if valid_fill and tp1_hit
                else "ambiguous"
                if status == "ambiguous_gap"
                else "not_entered"
                if not valid_fill
                else "not_profitable"
            )
        )
        terminal_at = signal.get("terminal_at") or _now_iso()
        generated_at = _parse_datetime(signal.get("generated_at"))
        entered_at = _parse_datetime(signal.get("entered_at"))
        terminal_dt = _parse_datetime(terminal_at)
        entry_delay = (
            max(0.0, (entered_at - generated_at).total_seconds() / 60.0)
            if valid_fill and generated_at and entered_at
            else None
        )
        duration = (
            max(0.0, (terminal_dt - entered_at).total_seconds() / 60.0)
            if valid_fill and entered_at and terminal_dt
            else None
        )
        params = {
            "candidate_id": candidate_id,
            "signal_id": signal.get("id"),
            "valid_fill": valid_fill,
            "technical_success": (
                None
                if signal.get("technical_success") is None
                else bool(signal.get("technical_success"))
            ),
            "alert_success": bool(valid_fill and tp1_hit),
            "tp1_hit": tp1_hit,
            "tp2_hit": int(signal.get("highest_tp") or 0) >= 2,
            "tp3_hit": int(signal.get("highest_tp") or 0) >= 3,
            "tp4_hit": int(signal.get("highest_tp") or 0) >= 4,
            "outcome_classification": outcome_classification,
            "invalidated_before_fill": status == "invalidated" and not valid_fill,
            "missed_before_fill": status == "missed" and not valid_fill,
            "expired_before_fill": status == "expired" and not valid_fill,
            "entry_delay_minutes": entry_delay,
            "trade_duration_minutes": duration,
            "realized_r": float(signal.get("realized_r") or 0),
            "mfe_r": float(signal.get("mfe_r") or 0),
            "mae_r": float(signal.get("mae_r") or 0),
            "slippage_bps": signal.get("slippage_bps"),
            "terminal_status": status,
            "terminal_at": terminal_at,
            "ambiguity_policy": (
                "sparse_gap_unknown"
                if status == "ambiguous_gap"
                else "observed_sequence"
            ),
            "metadata": Jsonb(
                {
                    "terminal_reason": signal.get("terminal_reason"),
                    "highest_tp": int(signal.get("highest_tp") or 0),
                    "profitable_at": signal.get("profitable_at"),
                    "level_hits": dict(signal.get("level_hits") or {}),
                }
            ),
        }
        sql = """
            insert into public.signal_outcomes (
                candidate_id, signal_id, label_source, valid_fill,
                technical_success, alert_success, tp1_hit, tp2_hit, tp3_hit,
                tp4_hit, outcome_classification,
                invalidated_before_fill, missed_before_fill,
                expired_before_fill, entry_delay_minutes,
                trade_duration_minutes, realized_r, mfe_r, mae_r,
                slippage_bps, terminal_status, terminal_at,
                ambiguity_policy, metadata
            ) values (
                %(candidate_id)s, %(signal_id)s, 'forward_tracker',
                %(valid_fill)s, %(technical_success)s, %(alert_success)s,
                %(tp1_hit)s, %(tp2_hit)s, %(tp3_hit)s, %(tp4_hit)s,
                %(outcome_classification)s, %(invalidated_before_fill)s,
                %(missed_before_fill)s, %(expired_before_fill)s,
                %(entry_delay_minutes)s, %(trade_duration_minutes)s,
                %(realized_r)s, %(mfe_r)s, %(mae_r)s, %(slippage_bps)s,
                %(terminal_status)s, %(terminal_at)s,
                %(ambiguity_policy)s, %(metadata)s
            )
            on conflict (candidate_id) do update set
                signal_id = excluded.signal_id,
                label_source = excluded.label_source,
                valid_fill = excluded.valid_fill,
                technical_success = excluded.technical_success,
                alert_success = excluded.alert_success,
                tp1_hit = excluded.tp1_hit,
                tp2_hit = excluded.tp2_hit,
                tp3_hit = excluded.tp3_hit,
                tp4_hit = excluded.tp4_hit,
                outcome_classification = excluded.outcome_classification,
                invalidated_before_fill = excluded.invalidated_before_fill,
                missed_before_fill = excluded.missed_before_fill,
                expired_before_fill = excluded.expired_before_fill,
                entry_delay_minutes = excluded.entry_delay_minutes,
                trade_duration_minutes = excluded.trade_duration_minutes,
                realized_r = excluded.realized_r,
                mfe_r = excluded.mfe_r,
                mae_r = excluded.mae_r,
                slippage_bps = excluded.slippage_bps,
                terminal_status = excluded.terminal_status,
                terminal_at = excluded.terminal_at,
                ambiguity_policy = excluded.ambiguity_policy,
                metadata = excluded.metadata,
                updated_at = now()
        """
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, params)
                connection.commit()
            self._mark_success(ready=True, migration_required=False)
            return True
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            return False

    def record_replay_outcomes(
        self,
        outcomes: Iterable[Mapping[str, Any]],
    ) -> int:
        rows = [dict(item) for item in outcomes if item.get("candidate_id")]
        if not rows or not self.enabled:
            return 0
        sql = """
            insert into public.signal_outcomes (
                candidate_id, signal_id, label_source, valid_fill,
                technical_success, alert_success, tp1_hit, tp2_hit,
                invalidated_before_fill, missed_before_fill,
                expired_before_fill, entry_delay_minutes,
                trade_duration_minutes, realized_r, mfe_r, mae_r,
                slippage_bps, terminal_status, terminal_at,
                ambiguity_policy, metadata
            ) values (
                %(candidate_id)s, null, 'historical_replay', %(valid_fill)s,
                %(technical_success)s, %(alert_success)s, %(tp1_hit)s,
                %(tp2_hit)s, %(invalidated_before_fill)s,
                %(missed_before_fill)s, %(expired_before_fill)s,
                %(entry_delay_minutes)s, %(trade_duration_minutes)s,
                %(realized_r)s, %(mfe_r)s, %(mae_r)s, null,
                %(terminal_status)s, %(terminal_at)s,
                %(ambiguity_policy)s, %(metadata)s
            )
            on conflict (candidate_id) do update set
                label_source = excluded.label_source,
                valid_fill = excluded.valid_fill,
                technical_success = excluded.technical_success,
                alert_success = excluded.alert_success,
                tp1_hit = excluded.tp1_hit,
                tp2_hit = excluded.tp2_hit,
                invalidated_before_fill = excluded.invalidated_before_fill,
                missed_before_fill = excluded.missed_before_fill,
                expired_before_fill = excluded.expired_before_fill,
                entry_delay_minutes = excluded.entry_delay_minutes,
                trade_duration_minutes = excluded.trade_duration_minutes,
                realized_r = excluded.realized_r,
                mfe_r = excluded.mfe_r,
                mae_r = excluded.mae_r,
                terminal_status = excluded.terminal_status,
                terminal_at = excluded.terminal_at,
                ambiguity_policy = excluded.ambiguity_policy,
                metadata = excluded.metadata,
                updated_at = now()
        """
        payloads = [
            {
                **row,
                "metadata": Jsonb(dict(row.get("metadata") or {})),
            }
            for row in rows
        ]
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.executemany(sql, payloads)
                connection.commit()
            self._mark_success(ready=True, migration_required=False)
            return len(payloads)
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            return 0

    def load_model(self, stage: str = "shadow") -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        stages = [stage]
        if stage == "shadow":
            stages.append("champion")
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        select version, stage, feature_schema_version, trained_at,
                               artifact, metrics, validation
                        from public.scoring_model_versions
                        where stage = any(%s)
                        order by case when stage = %s then 0 else 1 end,
                                 trained_at desc
                        limit 1
                        """,
                        (stages, stage),
                    )
                    row = cursor.fetchone()
            self._mark_success(ready=True, migration_required=False)
            return dict(row) if row else None
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            return None

    def load_training_rows(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        select c.id, c.generated_at, c.symbol, c.exchange_id,
                               c.timeframe, c.setup_type,
                               c.direction, c.is_directional_candidate,
                               c.feature_schema_version, c.features,
                               c.production_scores, c.production_eligible,
                               o.valid_fill, o.technical_success,
                               o.alert_success, o.tp1_hit, o.tp2_hit,
                               o.invalidated_before_fill,
                               o.missed_before_fill,
                               o.expired_before_fill,
                               o.entry_delay_minutes,
                               o.trade_duration_minutes,
                               o.realized_r, o.mfe_r, o.mae_r,
                               o.terminal_status, o.terminal_at
                        from public.signal_candidates c
                        join public.signal_outcomes o
                          on o.candidate_id = c.id
                        where c.is_directional_candidate = true
                          and c.direction in ('long', 'short')
                          and o.terminal_status <> 'ambiguous_gap'
                          and coalesce(o.ambiguity_policy, '') <> 'sparse_gap_unknown'
                        order by c.generated_at
                        """
                    )
                    rows = [dict(row) for row in cursor.fetchall()]
            self._mark_success(ready=True, migration_required=False)
            return rows
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            return []

    def save_model(
        self,
        *,
        version: str,
        stage: str,
        feature_schema_version: str,
        training_start: Any,
        training_end: Any,
        training_samples: int,
        calibration_samples: int,
        artifact: Mapping[str, Any],
        metrics: Mapping[str, Any],
        validation: Mapping[str, Any],
        folds: Iterable[Mapping[str, Any]] = (),
    ) -> bool:
        if not self.enabled:
            return False
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        insert into public.scoring_model_versions (
                            version, stage, feature_schema_version, trained_at,
                            training_start, training_end, training_samples,
                            calibration_samples, artifact, metrics, validation
                        ) values (
                            %s, %s, %s, now(), %s, %s, %s, %s, %s, %s, %s
                        )
                        on conflict (version) do update set
                            stage = excluded.stage,
                            artifact = excluded.artifact,
                            metrics = excluded.metrics,
                            validation = excluded.validation
                        """,
                        (
                            version,
                            stage,
                            feature_schema_version,
                            training_start,
                            training_end,
                            training_samples,
                            calibration_samples,
                            Jsonb(dict(artifact)),
                            Jsonb(dict(metrics)),
                            Jsonb(dict(validation)),
                        ),
                    )
                    for fold in folds:
                        cursor.execute(
                            """
                            insert into public.scoring_validation_runs (
                                model_version, fold, train_start, train_end,
                                test_start, test_end, sample_count, metrics,
                                regime_metrics
                            ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            on conflict (model_version, fold) do update set
                                metrics = excluded.metrics,
                                regime_metrics = excluded.regime_metrics
                            """,
                            (
                                version,
                                int(fold.get("fold") or 0),
                                fold.get("train_start"),
                                fold.get("train_end"),
                                fold.get("test_start"),
                                fold.get("test_end"),
                                int(fold.get("sample_count") or 0),
                                Jsonb(dict(fold.get("metrics") or {})),
                                Jsonb(dict(fold.get("regime_metrics") or {})),
                            ),
                        )
                connection.commit()
            self._mark_success(ready=True, migration_required=False)
            return True
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            return False

    def promote_model(self, version: str) -> Dict[str, Any]:
        """Explicitly promote only a calibrated shadow that passed its gate."""
        if not self.enabled:
            return {"ok": False, "error": "database_not_configured"}
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        select version, stage, artifact, validation
                        from public.scoring_model_versions
                        where version = %s
                        for update
                        """,
                        (version,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        return {"ok": False, "error": "model_not_found"}
                    artifact = dict(row.get("artifact") or {})
                    validation = dict(row.get("validation") or {})
                    promotion = dict(validation.get("promotion_gate") or {})
                    if not bool(artifact.get("calibration_ready")):
                        return {
                            "ok": False,
                            "error": "calibration_not_ready",
                        }
                    if not bool(promotion.get("passed")):
                        return {
                            "ok": False,
                            "error": "walk_forward_gate_failed",
                            "checks": promotion.get("checks"),
                        }
                    cursor.execute(
                        """
                        update public.scoring_model_versions
                        set stage = 'retired'
                        where stage = 'champion' and version <> %s
                        """,
                        (version,),
                    )
                    cursor.execute(
                        """
                        update public.scoring_model_versions
                        set stage = 'champion', promoted_at = now()
                        where version = %s
                        """,
                        (version,),
                    )
                connection.commit()
            self._mark_success(ready=True, migration_required=False)
            return {"ok": True, "version": version, "stage": "champion"}
        except Exception as exc:  # noqa: BLE001
            self._mark_failure(exc)
            return {"ok": False, "error": type(exc).__name__}

    def _connect(self):
        if not self.enabled or psycopg is None:
            raise RuntimeError("PostgreSQL outcome repository is not configured")
        return psycopg.connect(
            self.database_url,
            connect_timeout=5,
            row_factory=dict_row,
        )

    def _mark_success(self, **values: Any) -> None:
        with self._status_lock:
            self._status.update(values)
            self._status["last_success_at"] = _now_iso()
            self._status["last_error"] = None

    def _mark_failure(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {str(exc)[:180]}"
        migration_required = "signal_candidates" in str(exc) and (
            "does not exist" in str(exc).lower()
        )
        with self._status_lock:
            self._status.update(
                {
                    "ready": False,
                    "migration_required": migration_required,
                    "last_error": message,
                }
            )


def _parse_datetime(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
