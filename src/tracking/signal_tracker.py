"""Persistent, low-overhead lifecycle tracking for delivered Telegram signals.

The tracker is deliberately account-independent: it observes public market
prices and records a standardized simulated fill. It never places orders and
does not claim to know a user's private exchange fill.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import queue
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock, RLock, Thread
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from loguru import logger

from src.data.exchange import ExchangeClient, normalize_exchange_id
from src.notify.telegram import send_telegram_message_detailed
from src.scoring.labels import technical_success_from_range
from src.scoring.repository import OutcomeRepository, TERMINAL_STATUSES as DURABLE_TERMINAL_STATUSES
from src.analysis.execution_policy import RANK_POLICY_VERSION
from src.tracking.durable_repository import (
    LIFECYCLE_SCHEMA_VERSION,
    LifecycleRepository,
    destination_hash,
)
from src.utils.config import AppConfig, load_config
from src.utils.helpers import safe_float

try:
    import websocket
except ImportError:  # pragma: no cover - production dependency, graceful fallback
    websocket = None


UTC = timezone.utc
ACTIVE_STATUSES = ("pending", "entered")
TERMINAL_STATUSES = (
    "completed",
    "stopped",
    "missed",
    "expired",
    "invalidated",
    "time_exit",
    "ambiguous_gap",
)

OUTCOME_PENDING_ENTRY = "pending_entry"
OUTCOME_ACTIVE = "active"
OUTCOME_PROFITABLE = "profitable"
OUTCOME_NOT_PROFITABLE = "not_profitable"
OUTCOME_NOT_ENTERED = "not_entered"
OUTCOME_AMBIGUOUS = "ambiguous"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _parse_datetime(value: Any, fallback: Optional[datetime] = None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value or "").strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            parsed = fallback or _utc_now()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _json_loads(value: Any, fallback: Any) -> Any:
    try:
        parsed = json.loads(str(value or ""))
    except (TypeError, ValueError):
        return fallback
    return parsed


def _outcome_classification(
    *,
    status: str,
    entered: bool,
    highest_tp: int,
) -> str:
    """Return the explicit trade outcome without changing lifecycle policy."""
    if highest_tp >= 1 and entered:
        return OUTCOME_PROFITABLE
    if status == "ambiguous_gap":
        return OUTCOME_AMBIGUOUS
    if status == "pending":
        return OUTCOME_PENDING_ENTRY
    if status == "entered":
        return OUTCOME_ACTIVE
    if not entered:
        return OUTCOME_NOT_ENTERED
    return OUTCOME_NOT_PROFITABLE


def _level_hit(
    *,
    level: float,
    observed_price: float,
    occurred_at: datetime,
    source: str,
) -> Dict[str, Any]:
    """Canonical exact-level observation stored in both cache and Supabase."""
    return {
        "level": float(level),
        "observed_price": float(observed_price),
        "hit_at": _iso(occurred_at),
        "source": str(source or "market_observation"),
    }


def _unique_strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in out:
            out.append(item)
    return out


def _within_bps(a: float, b: float, tolerance_bps: float) -> bool:
    reference = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / reference * 10_000.0 <= tolerance_bps


def _price_in_zone(price: float, low: float, high: float) -> bool:
    return min(low, high) <= price <= max(low, high)


def _target_hit(direction: str, price: float, target: float) -> bool:
    return price >= target if direction == "long" else price <= target


def _stop_hit(direction: str, price: float, stop: float) -> bool:
    return price <= stop if direction == "long" else price >= stop


def _segment_crossing_fraction(
    previous: float,
    current: float,
    low: float,
    high: Optional[float] = None,
) -> Optional[float]:
    """Earliest fraction along a linear observation segment touching a level/zone."""
    zone_low = min(low, high if high is not None else low)
    zone_high = max(low, high if high is not None else low)
    if zone_low <= previous <= zone_high:
        return 0.0
    delta = current - previous
    if delta == 0:
        return None
    boundary = zone_low if previous < zone_low else zone_high
    fraction = (boundary - previous) / delta
    if 0.0 <= fraction <= 1.0:
        point = previous + delta * fraction
        if zone_low - 1e-12 <= point <= zone_high + 1e-12:
            return fraction
    return None


def _pending_crossing_order(
    *,
    direction: str,
    previous: Optional[float],
    current: float,
    entry_low: float,
    entry_high: float,
    target: float,
) -> str:
    """Return entry_first, target_first, ambiguous, or none for sparse ticks."""
    if previous is None or previous <= 0:
        return "ambiguous" if _target_hit(direction, current, target) else "none"
    entry_fraction = _segment_crossing_fraction(
        previous, current, entry_low, entry_high
    )
    target_fraction = _segment_crossing_fraction(previous, current, target)
    if entry_fraction is not None and target_fraction is not None:
        if abs(entry_fraction - target_fraction) <= 1e-9:
            return "ambiguous"
        return "entry_first" if entry_fraction < target_fraction else "target_first"
    if entry_fraction is not None:
        return "entry_first"
    if target_fraction is not None:
        return "target_first"
    return "none"


def _r_at_price(signal: Dict[str, Any], price: float) -> float:
    entry = safe_float(signal.get("entry_price"))
    stop = safe_float(signal.get("stop_loss"))
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    move = price - entry
    if str(signal.get("direction")) == "short":
        move = -move
    return move / risk


def _normalize_allocations(
    configured: Sequence[float],
    target_count: int,
) -> List[float]:
    count = max(1, target_count)
    values = [max(0.0, safe_float(value)) for value in configured[:count]]
    if len(values) < count:
        values.extend([1.0] * (count - len(values)))
    total = sum(values)
    if total <= 0:
        return [1.0 / count] * count
    return [value / total for value in values]


def _resolve_database_path(config: AppConfig) -> Path:
    raw = str(config.signal_tracker.database_path or "./data/signal_tracker.db")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


class SignalStore:
    """Thread-safe SQLite journal for signals and notification events."""

    def __init__(
        self,
        path: Path,
        *,
        target_allocations: Sequence[float],
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._allocations_config = list(target_allocations)
        self._connection = sqlite3.connect(
            str(path),
            timeout=20,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_low REAL NOT NULL,
                    entry_high REAL NOT NULL,
                    entry_mid REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profits_json TEXT NOT NULL,
                    destinations_json TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    valid_until TEXT NOT NULL,
                    entered_at TEXT,
                    entry_price REAL,
                    hold_until TEXT,
                    highest_tp INTEGER NOT NULL DEFAULT 0,
                    realized_r REAL NOT NULL DEFAULT 0,
                    mfe_r REAL NOT NULL DEFAULT 0,
                    mae_r REAL NOT NULL DEFAULT 0,
                    slippage_bps REAL,
                    last_price REAL,
                    last_price_at TEXT,
                    terminal_at TEXT,
                    terminal_reason TEXT,
                    technical_success INTEGER,
                    outcome_classification TEXT NOT NULL DEFAULT 'pending_entry',
                    profitable INTEGER NOT NULL DEFAULT 0,
                    profitable_at TEXT,
                    level_hits_json TEXT NOT NULL DEFAULT '{}',
                    previous_price REAL,
                    previous_price_at TEXT,
                    last_processed_candle_at TEXT,
                    lifecycle_version INTEGER NOT NULL DEFAULT 0,
                    ordering_policy TEXT NOT NULL DEFAULT 'observed_segment_v1',
                    row_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_signals_active
                    ON signals(status, symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_generated
                    ON signals(generated_at);

                CREATE TABLE IF NOT EXISTS signal_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    price REAL,
                    payload_json TEXT NOT NULL,
                    destinations_json TEXT NOT NULL,
                    delivered_to_json TEXT NOT NULL DEFAULT '[]',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT,
                    notified_at TEXT,
                    event_uid TEXT,
                    lifecycle_version INTEGER NOT NULL DEFAULT 0,
                    from_status TEXT,
                    to_status TEXT,
                    notify INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(signal_id) REFERENCES signals(id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_pending
                    ON signal_events(notified_at, id);
                """
            )
            columns = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(signals)")
            }
            if "technical_success" not in columns:
                self._connection.execute(
                    "ALTER TABLE signals ADD COLUMN technical_success INTEGER"
                )
            additive_signal_columns = {
                "previous_price": "REAL",
                "previous_price_at": "TEXT",
                "last_processed_candle_at": "TEXT",
                "lifecycle_version": "INTEGER NOT NULL DEFAULT 0",
                "ordering_policy": "TEXT NOT NULL DEFAULT 'observed_segment_v1'",
                "outcome_classification": "TEXT NOT NULL DEFAULT 'pending_entry'",
                "profitable": "INTEGER NOT NULL DEFAULT 0",
                "profitable_at": "TEXT",
                "level_hits_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for name, declaration in additive_signal_columns.items():
                if name not in columns:
                    self._connection.execute(
                        f"ALTER TABLE signals ADD COLUMN {name} {declaration}"
                    )
            event_columns = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(signal_events)")
            }
            additive_event_columns = {
                "event_uid": "TEXT",
                "lifecycle_version": "INTEGER NOT NULL DEFAULT 0",
                "from_status": "TEXT",
                "to_status": "TEXT",
                "notify": "INTEGER NOT NULL DEFAULT 1",
            }
            for name, declaration in additive_event_columns.items():
                if name not in event_columns:
                    self._connection.execute(
                        f"ALTER TABLE signal_events ADD COLUMN {name} {declaration}"
                    )
            self._connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_uid ON signal_events(event_uid)"
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _decode_signal(row: sqlite3.Row) -> Dict[str, Any]:
        signal = dict(row)
        signal["take_profits"] = _json_loads(
            signal.pop("take_profits_json", "[]"),
            [],
        )
        signal["destinations"] = _json_loads(
            signal.pop("destinations_json", "[]"),
            [],
        )
        signal["row"] = _json_loads(signal.pop("row_json", "{}"), {})
        signal["level_hits"] = _json_loads(
            signal.pop("level_hits_json", "{}"),
            {},
        )
        signal["profitable"] = bool(signal.get("profitable"))
        return signal

    @staticmethod
    def _decode_event(row: sqlite3.Row) -> Dict[str, Any]:
        event = dict(row)
        event["payload"] = _json_loads(event.pop("payload_json", "{}"), {})
        event["destinations"] = _json_loads(
            event.pop("destinations_json", "[]"),
            [],
        )
        event["delivered_to"] = _json_loads(
            event.pop("delivered_to_json", "[]"),
            [],
        )
        return event

    def active_signals(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        sql = (
            "SELECT * FROM signals "
            "WHERE status IN ('pending', 'entered')"
        )
        params: List[Any] = []
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY created_at"
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [self._decode_signal(row) for row in rows]

    def signals_for_sync(
        self,
        symbol: Optional[str] = None,
        *,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Return lifecycle rows for durable PostgreSQL mirroring."""
        sql = "SELECT * FROM signals"
        params: List[Any] = []
        if symbol:
            sql += " WHERE symbol = ?"
            params.append(symbol)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        return [self._decode_signal(row) for row in rows]

    def events_for_sync(self, signal_id: str) -> List[Dict[str, Any]]:
        """Return immutable local events for atomic durable journaling."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM signal_events WHERE signal_id=? ORDER BY lifecycle_version",
                (signal_id,),
            ).fetchall()
        return [self._decode_event(row) for row in rows]

    def restore_signal(self, signal: Mapping[str, Any]) -> bool:
        """Restore a durable active record into the SQLite working cache."""
        signal_id = str(signal.get("id") or "").strip()
        if not signal_id or str(signal.get("status")) not in ACTIVE_STATUSES:
            return False
        targets = list(signal.get("take_profits") or [])
        destinations = list(signal.get("destinations") or [])
        row = dict(signal.get("row") or {})
        with self._lock:
            existing = self._connection.execute(
                "SELECT lifecycle_version FROM signals WHERE id=?", (signal_id,)
            ).fetchone()
            if existing and int(existing["lifecycle_version"] or 0) > int(
                signal.get("lifecycle_version") or 0
            ):
                return False
            self._connection.execute(
                """
                INSERT INTO signals (
                  id,fingerprint,symbol,exchange_id,direction,timeframe,source,status,
                  confidence,entry_low,entry_high,entry_mid,stop_loss,
                  take_profits_json,destinations_json,generated_at,valid_until,
                  entered_at,entry_price,hold_until,highest_tp,realized_r,mfe_r,
                  mae_r,slippage_bps,last_price,last_price_at,terminal_at,
                  terminal_reason,technical_success,previous_price,previous_price_at,
                  last_processed_candle_at,lifecycle_version,ordering_policy,
                  outcome_classification,profitable,profitable_at,level_hits_json,
                  row_json,created_at,updated_at
                ) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  status=excluded.status, entered_at=excluded.entered_at,
                  entry_price=excluded.entry_price, hold_until=excluded.hold_until,
                  highest_tp=excluded.highest_tp, realized_r=excluded.realized_r,
                  mfe_r=excluded.mfe_r, mae_r=excluded.mae_r,
                  last_price=excluded.last_price, last_price_at=excluded.last_price_at,
                  previous_price=excluded.previous_price,
                  previous_price_at=excluded.previous_price_at,
                  last_processed_candle_at=excluded.last_processed_candle_at,
                  lifecycle_version=excluded.lifecycle_version,
                  ordering_policy=excluded.ordering_policy,
                  outcome_classification=excluded.outcome_classification,
                  profitable=excluded.profitable,
                  profitable_at=excluded.profitable_at,
                  level_hits_json=excluded.level_hits_json,
                  destinations_json=excluded.destinations_json,
                  row_json=excluded.row_json, updated_at=excluded.updated_at
                """,
                (
                    signal_id, signal.get("fingerprint") or signal_id,
                    signal.get("symbol"), signal.get("exchange_id"),
                    signal.get("direction"), signal.get("timeframe") or "15m",
                    signal.get("source") or "telegram", signal.get("status"),
                    safe_float(signal.get("confidence")), signal.get("entry_low"),
                    signal.get("entry_high"), signal.get("entry_mid"),
                    signal.get("stop_loss"), json.dumps(targets, separators=(",", ":")),
                    json.dumps(destinations, separators=(",", ":")),
                    signal.get("generated_at"), signal.get("valid_until"),
                    signal.get("entered_at"), signal.get("entry_price"),
                    signal.get("hold_until"), int(signal.get("highest_tp") or 0),
                    safe_float(signal.get("realized_r")), safe_float(signal.get("mfe_r")),
                    safe_float(signal.get("mae_r")), signal.get("slippage_bps"),
                    signal.get("last_price"), signal.get("last_price_at"),
                    signal.get("terminal_at"), signal.get("terminal_reason"),
                    signal.get("technical_success"), signal.get("previous_price"),
                    signal.get("previous_price_at"), signal.get("last_processed_candle_at"),
                    int(signal.get("lifecycle_version") or 0),
                    signal.get("ordering_policy") or "observed_segment_v1",
                    signal.get("outcome_classification") or _outcome_classification(
                        status=str(signal.get("status") or "pending"),
                        entered=bool(signal.get("entered_at")),
                        highest_tp=int(signal.get("highest_tp") or 0),
                    ),
                    1 if bool(signal.get("profitable")) else 0,
                    signal.get("profitable_at"),
                    json.dumps(
                        dict(signal.get("level_hits") or {}),
                        separators=(",", ":"),
                        default=str,
                    ),
                    json.dumps(row, separators=(",", ":"), default=str),
                    signal.get("created_at") or signal.get("generated_at"),
                    signal.get("updated_at") or _iso(_utc_now()),
                ),
            )
            self._connection.commit()
        return True

    def counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM signals GROUP BY status"
            ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        counts["active"] = sum(counts.get(status, 0) for status in ACTIVE_STATUSES)
        return counts

    def reliability_summary(self) -> Dict[str, Any]:
        """Aggregate alert-level outcomes, including failures before entry."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT confidence, status, realized_r, generated_at, entered_at,
                       terminal_at, slippage_bps, mfe_r, mae_r, highest_tp,
                       outcome_classification, profitable
                FROM signals
                WHERE status IN (
                    'completed', 'stopped', 'time_exit',
                    'missed', 'expired', 'invalidated', 'ambiguous_gap'
                )
                  AND terminal_at IS NOT NULL
                ORDER BY terminal_at
                """
            ).fetchall()
        ambiguous_count = sum(
            1 for row in rows if str(row["status"]) == "ambiguous_gap"
        )
        evaluable_rows = [
            row for row in rows if str(row["status"]) != "ambiguous_gap"
        ]
        bands = [
            ("80–84%", 80.0, 85.0),
            ("85–89%", 85.0, 90.0),
            ("90%+", 90.0, float("inf")),
        ]
        grouped: List[Dict[str, Any]] = []
        brier_terms: List[float] = []
        for label, lower, upper in bands:
            selected = [
                row
                for row in evaluable_rows
                if lower <= safe_float(row["confidence"]) < upper
            ]
            sample = len(selected)
            successful = sum(
                1
                for row in selected
                if row["entered_at"] is not None
                and int(row["highest_tp"] or 0) >= 1
            )
            filled = sum(1 for row in selected if row["entered_at"] is not None)
            early_stops = 0
            entry_delays: List[float] = []
            slippage_values: List[float] = []
            for row in selected:
                confidence_probability = min(
                    1.0,
                    max(0.0, safe_float(row["confidence"]) / 100.0),
                )
                outcome = (
                    1.0
                    if row["entered_at"] is not None
                    and int(row["highest_tp"] or 0) >= 1
                    else 0.0
                )
                brier_terms.append((confidence_probability - outcome) ** 2)
                generated = _parse_datetime(row["generated_at"])
                entered = (
                    _parse_datetime(row["entered_at"])
                    if row["entered_at"] is not None
                    else None
                )
                terminal = _parse_datetime(row["terminal_at"])
                if entered is not None:
                    entry_delays.append(
                        max(0.0, (entered - generated).total_seconds() / 60.0)
                    )
                if row["slippage_bps"] is not None:
                    slippage_values.append(safe_float(row["slippage_bps"]))
                if (
                    row["status"] == "stopped"
                    and entered is not None
                    and (terminal - entered).total_seconds() <= 3600
                ):
                    early_stops += 1
            grouped.append(
                {
                    "confidence_band": label,
                    "sample": sample,
                    "successful_alerts": successful,
                    # Backward-compatible aliases retained for existing clients.
                    "profitable": successful,
                    "profitable_rate_pct": (
                        round(successful / sample * 100.0, 1)
                        if sample
                        else None
                    ),
                    "alert_success_rate_pct": (
                        round(successful / sample * 100.0, 1)
                        if sample
                        else None
                    ),
                    "valid_fill_rate_pct": (
                        round(filled / sample * 100.0, 1)
                        if sample
                        else None
                    ),
                    "pre_entry_failure_rate_pct": (
                        round((sample - filled) / sample * 100.0, 1)
                        if sample
                        else None
                    ),
                    "average_r": (
                        round(
                            sum(safe_float(row["realized_r"]) for row in selected)
                            / sample,
                            3,
                        )
                        if sample
                        else None
                    ),
                    "early_stop_rate_pct": (
                        round(early_stops / sample * 100.0, 1)
                        if sample
                        else None
                    ),
                    "average_entry_delay_minutes": (
                        round(sum(entry_delays) / len(entry_delays), 1)
                        if entry_delays
                        else None
                    ),
                    "average_slippage_proxy_bps": (
                        round(sum(slippage_values) / len(slippage_values), 2)
                        if slippage_values
                        else None
                    ),
                    "calibration_ready": sample >= 50,
                }
            )
        total = len(evaluable_rows)
        target_hits = {
            f"tp{target}_hit": sum(
                1 for row in evaluable_rows if int(row["highest_tp"] or 0) >= target
            )
            for target in range(1, 5)
        }
        classification_counts: Dict[str, int] = {}
        for row in rows:
            classification = str(
                row["outcome_classification"]
                or _outcome_classification(
                    status=str(row["status"]),
                    entered=bool(row["entered_at"]),
                    highest_tp=int(row["highest_tp"] or 0),
                )
            )
            classification_counts[classification] = (
                classification_counts.get(classification, 0) + 1
            )
        return {
            "status": (
                "enough_data_for_initial_calibration"
                if any(group["calibration_ready"] for group in grouped)
                else "collecting_forward_outcomes"
            ),
            "completed_outcomes": total,
            "ambiguous_outcomes": ambiguous_count,
            "outcome_classifications": classification_counts,
            "target_hits": target_hits,
            "minimum_samples_per_band": 50,
            "brier_score_diagnostic": (
                round(sum(brier_terms) / len(brier_terms), 4)
                if brier_terms
                else None
            ),
            "bands": grouped,
            "note": (
                "Alert success requires a valid fill and TP1 before Stop. "
                "Missed, expired, and invalidated entries count as failures. "
                "Displayed production confidence remains a confluence score "
                "until the shadow model passes unseen validation."
            ),
        }

    def _find_matching_active_locked(
        self,
        *,
        symbol: str,
        exchange_id: str,
        direction: str,
        entry_mid: float,
        stop_loss: float,
    ) -> Optional[sqlite3.Row]:
        rows = self._connection.execute(
            """
            SELECT * FROM signals
            WHERE symbol = ? AND exchange_id = ? AND direction = ?
              AND status IN ('pending', 'entered')
            ORDER BY created_at DESC
            """,
            (symbol, exchange_id, direction),
        ).fetchall()
        for row in rows:
            if _within_bps(entry_mid, float(row["entry_mid"]), 8.0) and _within_bps(
                stop_loss,
                float(row["stop_loss"]),
                12.0,
            ):
                return row
        return None

    def register_signal(
        self,
        row: Dict[str, Any],
        destinations: Sequence[str],
        *,
        source: str,
        now: Optional[datetime] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """Insert a delivered signal or merge destinations into an active match."""
        timestamp = now or _utc_now()
        row = dict(row)
        symbol = str(row.get("symbol") or "").upper().strip()
        direction = str(row.get("direction") or "").lower().strip()
        entry_low = safe_float(row.get("entry_low"))
        entry_high = safe_float(row.get("entry_high"))
        stop_loss = safe_float(row.get("stop_loss"))
        targets = [
            safe_float(value)
            for value in list(row.get("take_profits") or [])[:4]
            if safe_float(value) > 0
        ]
        row.setdefault(
            "target_allocations",
            _normalize_allocations(self._allocations_config, len(targets)),
        )
        chats = _unique_strings(destinations)
        if (
            not symbol
            or direction not in ("long", "short")
            or entry_low <= 0
            or entry_high <= 0
            or stop_loss <= 0
            or not targets
            or not chats
        ):
            raise ValueError("Signal is missing trackable levels or destinations")
        entry_low, entry_high = sorted((entry_low, entry_high))
        entry_mid = (entry_low + entry_high) / 2.0
        generated = _parse_datetime(row.get("signal_generated_at"), timestamp)
        validity_minutes = max(
            15.0,
            safe_float(row.get("entry_valid_for_minutes")) or 60.0,
        )
        valid_until = _parse_datetime(
            row.get("entry_valid_until"),
            generated + timedelta(minutes=validity_minutes),
        )
        exchange_id = normalize_exchange_id(
            str(row.get("exchange") or "okx")
        )
        timeframe = str(row.get("primary_tf") or "15m")
        level_blob = (
            f"{exchange_id}|{symbol}|{direction}|{entry_low:.12g}|{entry_high:.12g}|"
            f"{stop_loss:.12g}|{_iso(valid_until)}"
        )
        fingerprint = hashlib.sha256(level_blob.encode("utf-8")).hexdigest()[:24]
        signal_id = f"sig_{fingerprint}_{int(timestamp.timestamp())}"
        initial_price = safe_float(row.get("price"))
        # Publication inside a CMP zone is not execution evidence. All signals
        # begin pending; CMP requires a later closed-candle confirmation.
        status = "pending"
        entered_at = None
        entry_price = None
        hold_hours = max(0.5, safe_float(row.get("hold_hours_max")) or 12.0)
        hold_until = (
            entered_at + timedelta(hours=hold_hours)
            if entered_at is not None
            else None
        )
        slippage_bps = None
        if entry_price:
            direction_sign = 1.0 if direction == "long" else -1.0
            slippage_bps = (
                (entry_price - entry_mid) / entry_mid * 10_000.0 * direction_sign
            )
        now_iso = _iso(timestamp)

        with self._lock:
            existing = self._find_matching_active_locked(
                symbol=symbol,
                exchange_id=exchange_id,
                direction=direction,
                entry_mid=entry_mid,
                stop_loss=stop_loss,
            )
            if existing is not None:
                existing_signal = self._decode_signal(existing)
                previous_destinations = list(existing_signal["destinations"])
                merged = _unique_strings(
                    [*existing_signal["destinations"], *chats]
                )
                self._connection.execute(
                    """
                    UPDATE signals
                    SET destinations_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        json.dumps(merged, separators=(",", ":")),
                        now_iso,
                        existing_signal["id"],
                    ),
                )
                self._connection.commit()
                existing_signal["destinations"] = merged
                existing_signal["updated_at"] = now_iso
                newly_delivered = [chat for chat in chats if chat not in previous_destinations]
                if newly_delivered:
                    self._create_event_locked(
                        existing_signal["id"], "published", timestamp,
                        initial_price if initial_price > 0 else None,
                        {"source": source}, newly_delivered, notify=False,
                    )
                    self._connection.commit()
                refreshed = self._connection.execute(
                    "SELECT * FROM signals WHERE id=?", (existing_signal["id"],)
                ).fetchone()
                return self._decode_signal(refreshed), False

            self._connection.execute(
                """
                INSERT INTO signals (
                    id, fingerprint, symbol, exchange_id, direction, timeframe,
                    source, status, confidence, entry_low, entry_high, entry_mid,
                    stop_loss, take_profits_json, destinations_json, generated_at,
                    valid_until, entered_at, entry_price, hold_until, highest_tp,
                    realized_r, mfe_r, mae_r, slippage_bps, last_price,
                    last_price_at, outcome_classification, profitable,
                    profitable_at, level_hits_json, row_json, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    0, 0, 0, 0, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?
                )
                """,
                (
                    signal_id,
                    fingerprint,
                    symbol,
                    exchange_id,
                    direction,
                    timeframe,
                    str(source or "telegram"),
                    status,
                    safe_float(row.get("confidence")),
                    entry_low,
                    entry_high,
                    entry_mid,
                    stop_loss,
                    json.dumps(targets, separators=(",", ":")),
                    json.dumps(chats, separators=(",", ":")),
                    _iso(generated),
                    _iso(valid_until),
                    _iso(entered_at) if entered_at else None,
                    entry_price,
                    _iso(hold_until) if hold_until else None,
                    slippage_bps,
                    initial_price if initial_price > 0 else None,
                    now_iso if initial_price > 0 else None,
                    OUTCOME_PENDING_ENTRY,
                    json.dumps({}, separators=(",", ":")),
                    json.dumps(row, separators=(",", ":"), default=str),
                    now_iso,
                    now_iso,
                ),
            )
            self._connection.commit()
            inserted = self._connection.execute(
                "SELECT * FROM signals WHERE id = ?",
                (signal_id,),
            ).fetchone()
            self._create_event_locked(
                signal_id,
                "published",
                timestamp,
                initial_price if initial_price > 0 else None,
                {"source": source},
                chats,
                notify=False,
            )
            self._connection.commit()
            inserted = self._connection.execute(
                "SELECT * FROM signals WHERE id = ?",
                (signal_id,),
            ).fetchone()
        assert inserted is not None
        return self._decode_signal(inserted), True

    def _create_event_locked(
        self,
        signal_id: str,
        event_type: str,
        occurred_at: datetime,
        price: Optional[float],
        payload: Dict[str, Any],
        destinations: Sequence[str],
        *,
        notify: bool = True,
    ) -> int:
        chats = _unique_strings(destinations)
        delivered = [] if notify else chats
        notified_at = None if notify else _iso(occurred_at)
        state_row = self._connection.execute(
            "SELECT status, lifecycle_version, entered_at FROM signals WHERE id=?",
            (signal_id,),
        ).fetchone()
        previous_version = int(state_row["lifecycle_version"] or 0) if state_row else 0
        lifecycle_version = previous_version + 1
        to_status = str(state_row["status"] or "") if state_row else None
        if event_type in {"entered"}:
            from_status = "pending"
        elif event_type in {"target_hit", "completed", "protected_exit", "stopped", "time_exit"}:
            from_status = "entered"
        elif event_type in {"missed", "expired", "invalidated"}:
            from_status = "pending"
        elif event_type == "ambiguous_gap":
            from_status = "entered" if state_row and state_row["entered_at"] else "pending"
        else:
            from_status = to_status
        self._connection.execute(
            "UPDATE signals SET lifecycle_version=? WHERE id=?",
            (lifecycle_version, signal_id),
        )
        event_uid = f"{signal_id}:{lifecycle_version}:{event_type}"
        cursor = self._connection.execute(
            """
            INSERT INTO signal_events (
                signal_id, event_type, occurred_at, price, payload_json,
                destinations_json, delivered_to_json, notified_at, event_uid,
                lifecycle_version, from_status, to_status, notify
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                event_type,
                _iso(occurred_at),
                price,
                json.dumps(payload, separators=(",", ":"), default=str),
                json.dumps(chats, separators=(",", ":")),
                json.dumps(delivered, separators=(",", ":")),
                notified_at,
                event_uid,
                lifecycle_version,
                from_status,
                to_status,
                1 if notify else 0,
            ),
        )
        return int(cursor.lastrowid)

    def _terminal_transition_locked(
        self,
        signal: Dict[str, Any],
        *,
        status: str,
        event_type: str,
        timestamp: datetime,
        price: Optional[float],
        reason: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = dict(payload or {})
        details.setdefault("reason", reason)
        highest_tp = int(details.get("highest_tp") or signal.get("highest_tp") or 0)
        entered = bool(signal.get("entered_at"))
        classification = _outcome_classification(
            status=status,
            entered=entered,
            highest_tp=highest_tp,
        )
        profitable = classification == OUTCOME_PROFITABLE
        profitable_at = signal.get("profitable_at")
        if profitable and not profitable_at:
            profitable_at = _iso(timestamp)
        level_hits = dict(signal.get("level_hits") or {})
        if event_type == "stopped" and price is not None:
            level_hits.setdefault(
                "SL",
                _level_hit(
                    level=safe_float(signal.get("stop_loss")),
                    observed_price=price,
                    occurred_at=timestamp,
                    source="market_observation",
                ),
            )
            details.setdefault("stop_level", safe_float(signal.get("stop_loss")))
        elif event_type == "protected_exit" and price is not None:
            level_hits.setdefault(
                "PROTECTED_EXIT",
                _level_hit(
                    level=safe_float(signal.get("entry_price")),
                    observed_price=price,
                    occurred_at=timestamp,
                    source="market_observation",
                ),
            )
        elif event_type == "invalidated" and price is not None:
            level_hits.setdefault(
                "PRE_ENTRY_INVALIDATION",
                _level_hit(
                    level=safe_float(signal.get("stop_loss")),
                    observed_price=price,
                    occurred_at=timestamp,
                    source="closed_candle",
                ),
            )
        details.setdefault("outcome_classification", classification)
        details.setdefault("profitable", profitable)
        self._connection.execute(
            """
            UPDATE signals
            SET status = ?, terminal_at = ?, terminal_reason = ?,
                last_price = COALESCE(?, last_price),
                last_price_at = COALESCE(?, last_price_at),
                outcome_classification = ?, profitable = ?, profitable_at = ?,
                level_hits_json = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                status,
                _iso(timestamp),
                reason,
                price,
                _iso(timestamp) if price is not None else None,
                classification,
                1 if profitable else 0,
                profitable_at,
                json.dumps(level_hits, separators=(",", ":"), default=str),
                _iso(timestamp),
                signal["id"],
            ),
        )
        self._create_event_locked(
            signal["id"],
            event_type,
            timestamp,
            price,
            details,
            signal["destinations"],
        )

    def _record_entry_locked(
        self,
        signal: Dict[str, Any],
        *,
        price: float,
        timestamp: datetime,
        source: str,
    ) -> Dict[str, Any]:
        generated = _parse_datetime(signal["generated_at"])
        hold_hours = max(
            0.5,
            safe_float((signal.get("row") or {}).get("hold_hours_max")) or 12.0,
        )
        hold_until = timestamp + timedelta(hours=hold_hours)
        entry_mid = safe_float(signal["entry_mid"])
        direction_sign = 1.0 if signal["direction"] == "long" else -1.0
        slippage_bps = (
            (price - entry_mid)
            / max(entry_mid, 1e-12)
            * 10_000.0
            * direction_sign
        )
        level_hits = dict(signal.get("level_hits") or {})
        level_hits["ENTRY"] = _level_hit(
            level=price,
            observed_price=price,
            occurred_at=timestamp,
            source=source,
        )
        self._connection.execute(
            """
            UPDATE signals
            SET status = 'entered', entered_at = ?, entry_price = ?,
                hold_until = ?, slippage_bps = ?, mfe_r = 0, mae_r = 0,
                last_price = ?, last_price_at = ?,
                outcome_classification = ?, profitable = 0,
                level_hits_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                _iso(timestamp),
                price,
                _iso(hold_until),
                slippage_bps,
                price,
                _iso(timestamp),
                OUTCOME_ACTIVE,
                json.dumps(level_hits, separators=(",", ":"), default=str),
                _iso(timestamp),
                signal["id"],
            ),
        )
        self._create_event_locked(
            signal["id"],
            "entered",
            timestamp,
            price,
            {
                "entry_price": price,
                "entry_delay_minutes": max(
                    0.0, (timestamp - generated).total_seconds() / 60.0
                ),
                "slippage_bps": slippage_bps,
                "source": source,
            },
            signal["destinations"],
        )
        signal.update(
            {
                "status": "entered",
                "entered_at": _iso(timestamp),
                "entry_price": price,
                "hold_until": _iso(hold_until),
                "slippage_bps": slippage_bps,
                "mfe_r": 0.0,
                "mae_r": 0.0,
                "outcome_classification": OUTCOME_ACTIVE,
                "profitable": False,
                "level_hits": level_hits,
            }
        )
        return signal

    def _update_technical_success_locked(
        self,
        signal: Dict[str, Any],
        *,
        high: float,
        low: float,
    ) -> None:
        if signal.get("technical_success") is not None:
            return
        row = signal.get("row") or {}
        payload = row.get("payload") or {}
        primary = payload.get("primary_setup") or {}
        start_price = safe_float(row.get("price") or payload.get("price"))
        atr = safe_float(
            row.get("atr")
            or payload.get("atr")
            or primary.get("atr")
        )
        outcome = technical_success_from_range(
            direction=str(signal.get("direction") or ""),
            start_price=start_price,
            atr=atr,
            high=high,
            low=low,
        )
        if outcome is None:
            return
        self._connection.execute(
            "UPDATE signals SET technical_success = ?, updated_at = ? WHERE id = ?",
            (1 if outcome else 0, _iso(_utc_now()), signal["id"]),
        )
        signal["technical_success"] = 1 if outcome else 0

    def process_price(
        self,
        symbol: str,
        price: float,
        observed_at: datetime,
        *,
        source: str,
        exchange_id: Optional[str] = None,
    ) -> int:
        """Apply one ordered public-price observation to active signals."""
        if price <= 0:
            return 0
        timestamp = observed_at.astimezone(UTC)
        transition_count = 0
        with self._lock:
            sql = (
                "SELECT * FROM signals "
                "WHERE symbol = ? AND status IN ('pending', 'entered')"
            )
            params: List[Any] = [symbol]
            if exchange_id:
                sql += " AND exchange_id = ?"
                params.append(normalize_exchange_id(exchange_id))
            sql += " ORDER BY created_at"
            rows = self._connection.execute(sql, params).fetchall()
            for db_row in rows:
                signal = self._decode_signal(db_row)
                last_seen = (
                    _parse_datetime(signal["last_price_at"])
                    if signal.get("last_price_at")
                    else None
                )
                if last_seen and timestamp < last_seen:
                    continue
                previous_price = (
                    safe_float(signal.get("last_price"))
                    if signal.get("last_price") is not None
                    else None
                )
                previous_at = signal.get("last_price_at")
                self._update_technical_success_locked(
                    signal,
                    high=price,
                    low=price,
                )
                valid_until = _parse_datetime(signal["valid_until"])
                if signal["status"] == "pending" and timestamp > valid_until:
                    self._terminal_transition_locked(
                        signal,
                        status="expired",
                        event_type="expired",
                        timestamp=timestamp,
                        price=price,
                        reason="Entry window expired before the zone was reached",
                    )
                    transition_count += 1
                    continue

                self._connection.execute(
                    """
                    UPDATE signals
                    SET previous_price = ?, previous_price_at = ?,
                        last_price = ?, last_price_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        previous_price, previous_at, price, _iso(timestamp),
                        _iso(timestamp), signal["id"],
                    ),
                )

                if signal["status"] == "pending":
                    first_target = safe_float(signal["take_profits"][0])
                    row_entry_status = str(
                        (signal.get("row") or {}).get("entry_status") or "wait_retest"
                    ).lower()
                    confirmation_required = row_entry_status in (
                        "ready",
                        "confirmation_pending",
                    )
                    crossing = _pending_crossing_order(
                        direction=signal["direction"],
                        previous=previous_price,
                        current=price,
                        entry_low=safe_float(signal["entry_low"]),
                        entry_high=safe_float(signal["entry_high"]),
                        target=first_target,
                    )
                    if _target_hit(signal["direction"], price, first_target) and (
                        crossing == "ambiguous"
                    ):
                        self._terminal_transition_locked(
                            signal,
                            status="ambiguous_gap",
                            event_type="ambiguous_gap",
                            timestamp=timestamp,
                            price=price,
                            reason=(
                                "Sparse market-data gap crossed TP1 without enough "
                                "evidence to establish entry order"
                            ),
                        )
                        transition_count += 1
                        continue
                    if _target_hit(signal["direction"], price, first_target) and (
                        crossing == "target_first" or confirmation_required
                    ):
                        self._terminal_transition_locked(
                            signal,
                            status="missed",
                            event_type="missed",
                            timestamp=timestamp,
                            price=price,
                            reason="TP1 traded before a confirmed planned entry",
                        )
                        transition_count += 1
                        continue
                    if not confirmation_required and crossing == "entry_first":
                        fill_price = price
                        if previous_price is not None:
                            entry_fraction = _segment_crossing_fraction(
                                previous_price,
                                price,
                                safe_float(signal["entry_low"]),
                                safe_float(signal["entry_high"]),
                            )
                            if entry_fraction is not None:
                                fill_price = previous_price + (
                                    price - previous_price
                                ) * entry_fraction
                        signal = self._record_entry_locked(
                            signal,
                            price=fill_price,
                            timestamp=timestamp,
                            source=source,
                        )
                        transition_count += 1
                    else:
                        continue

                current_r = _r_at_price(signal, price)
                mfe_r = max(safe_float(signal.get("mfe_r")), current_r)
                mae_r = min(safe_float(signal.get("mae_r")), current_r)
                self._connection.execute(
                    """
                    UPDATE signals
                    SET mfe_r = ?, mae_r = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (mfe_r, mae_r, _iso(timestamp), signal["id"]),
                )
                highest_tp = int(signal.get("highest_tp") or 0)
                # TP1 changes management: the realized partial is a winning
                # outcome and the unclosed remainder is protected at entry.
                # A later reversal must never be reported as the original SL.
                if highest_tp >= 1 and _stop_hit(
                    signal["direction"],
                    price,
                    safe_float(signal["entry_price"]),
                ):
                    result_r = max(0.0, safe_float(signal.get("realized_r")))
                    self._connection.execute(
                        "UPDATE signals SET realized_r = ? WHERE id = ?",
                        (result_r, signal["id"]),
                    )
                    self._terminal_transition_locked(
                        signal,
                        status="completed",
                        event_type="protected_exit",
                        timestamp=timestamp,
                        price=price,
                        reason="Breakeven protection triggered after TP1",
                        payload={
                            "result_r": result_r,
                            "highest_tp": highest_tp,
                            "mfe_r": mfe_r,
                            "mae_r": mae_r,
                            "protected_price": safe_float(signal["entry_price"]),
                        },
                    )
                    transition_count += 1
                    continue
                if _stop_hit(
                    signal["direction"],
                    price,
                    safe_float(signal["stop_loss"]),
                ):
                    allocations = _normalize_allocations(
                        self._allocations_config,
                        len(signal["take_profits"]),
                    )
                    completed_fraction = sum(allocations[:highest_tp])
                    result_r = safe_float(signal.get("realized_r")) - (
                        1.0 - completed_fraction
                    )
                    self._connection.execute(
                        "UPDATE signals SET realized_r = ? WHERE id = ?",
                        (result_r, signal["id"]),
                    )
                    self._terminal_transition_locked(
                        signal,
                        status="stopped",
                        event_type="stopped",
                        timestamp=timestamp,
                        price=price,
                        reason="Hard stop traded after entry",
                        payload={
                            "result_r": result_r,
                            "highest_tp": highest_tp,
                            "mfe_r": mfe_r,
                            "mae_r": min(mae_r, -1.0),
                        },
                    )
                    transition_count += 1
                    continue

                old_highest = highest_tp
                new_highest = old_highest
                for index, target in enumerate(signal["take_profits"], 1):
                    if index > old_highest and _target_hit(
                        signal["direction"],
                        price,
                        safe_float(target),
                    ):
                        new_highest = index
                if new_highest > old_highest:
                    allocations = _normalize_allocations(
                        self._allocations_config,
                        len(signal["take_profits"]),
                    )
                    realized_r = 0.0
                    for index in range(new_highest):
                        realized_r += allocations[index] * _r_at_price(
                            signal,
                            safe_float(signal["take_profits"][index]),
                        )
                    completed = new_highest >= len(signal["take_profits"])
                    status = "completed" if completed else "entered"
                    level_hits = dict(signal.get("level_hits") or {})
                    newly_hit_records: List[Dict[str, Any]] = []
                    for index in range(old_highest + 1, new_highest + 1):
                        target_level = safe_float(signal["take_profits"][index - 1])
                        record = _level_hit(
                            level=target_level,
                            observed_price=price,
                            occurred_at=timestamp,
                            source=source,
                        )
                        level_hits[f"TP{index}"] = record
                        newly_hit_records.append({"tp_number": index, **record})
                    profitable_at = signal.get("profitable_at") or _iso(timestamp)
                    self._connection.execute(
                        """
                        UPDATE signals
                        SET highest_tp = ?, realized_r = ?, status = ?,
                            terminal_at = ?, terminal_reason = ?,
                            outcome_classification = ?, profitable = 1,
                            profitable_at = ?, level_hits_json = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            new_highest,
                            realized_r,
                            status,
                            _iso(timestamp) if completed else None,
                            "Final target traded" if completed else None,
                            OUTCOME_PROFITABLE,
                            profitable_at,
                            json.dumps(
                                level_hits,
                                separators=(",", ":"),
                                default=str,
                            ),
                            _iso(timestamp),
                            signal["id"],
                        ),
                    )
                    self._create_event_locked(
                        signal["id"],
                        "completed" if completed else "target_hit",
                        timestamp,
                        price,
                        {
                            "tp_number": new_highest,
                            "newly_hit": list(
                                range(old_highest + 1, new_highest + 1)
                            ),
                            "target_hits": newly_hit_records,
                            "target_price": safe_float(
                                signal["take_profits"][new_highest - 1]
                            ),
                            "realized_r": realized_r,
                            "mfe_r": mfe_r,
                            "mae_r": mae_r,
                            "outcome_classification": OUTCOME_PROFITABLE,
                            "profitable": True,
                            "profitable_at": profitable_at,
                        },
                        signal["destinations"],
                    )
                    transition_count += 1
            self._connection.commit()
        return transition_count

    def process_closed_candle(
        self,
        symbol: str,
        close_price: float,
        closed_at: datetime,
        *,
        exchange_id: Optional[str] = None,
        high_price: Optional[float] = None,
        low_price: Optional[float] = None,
    ) -> int:
        """Confirm CMP entries or cancel pending structure on a closed candle."""
        if close_price <= 0:
            return 0
        timestamp = closed_at.astimezone(UTC)
        count = 0
        with self._lock:
            sql = (
                "SELECT * FROM signals "
                "WHERE symbol = ? AND status = 'pending'"
            )
            params: List[Any] = [symbol]
            if exchange_id:
                sql += " AND exchange_id = ?"
                params.append(normalize_exchange_id(exchange_id))
            sql += " ORDER BY created_at"
            rows = self._connection.execute(sql, params).fetchall()
            for db_row in rows:
                signal = self._decode_signal(db_row)
                last_candle = (
                    _parse_datetime(signal.get("last_processed_candle_at"))
                    if signal.get("last_processed_candle_at")
                    else None
                )
                if last_candle and timestamp <= last_candle:
                    continue
                if timestamp <= _parse_datetime(signal["generated_at"]):
                    continue
                self._connection.execute(
                    "UPDATE signals SET last_processed_candle_at=?, updated_at=? WHERE id=?",
                    (_iso(timestamp), _iso(timestamp), signal["id"]),
                )
                self._update_technical_success_locked(
                    signal,
                    high=safe_float(high_price, close_price),
                    low=safe_float(low_price, close_price),
                )
                invalid = _stop_hit(
                    signal["direction"],
                    close_price,
                    safe_float(signal["stop_loss"]),
                )
                if invalid:
                    self._terminal_transition_locked(
                        signal,
                        status="invalidated",
                        event_type="invalidated",
                        timestamp=timestamp,
                        price=close_price,
                        reason=(
                            f"{signal['timeframe']} candle closed beyond the "
                            "structure invalidation level before entry"
                        ),
                    )
                    count += 1
                    continue
                row_entry_status = str(
                    (signal.get("row") or {}).get("entry_status") or ""
                ).lower()
                confirmation_required = row_entry_status in (
                    "ready",
                    "confirmation_pending",
                )
                high = safe_float(high_price, close_price)
                low = safe_float(low_price, close_price)
                first_target = safe_float(signal["take_profits"][0])
                target_touched = (
                    high >= first_target
                    if signal["direction"] == "long"
                    else low <= first_target
                )
                if confirmation_required and target_touched:
                    self._terminal_transition_locked(
                        signal,
                        status="missed",
                        event_type="missed",
                        timestamp=timestamp,
                        price=first_target,
                        reason="TP1 traded before CMP candle confirmation",
                    )
                    count += 1
                    continue
                if confirmation_required and _price_in_zone(
                    close_price,
                    safe_float(signal["entry_low"]),
                    safe_float(signal["entry_high"]),
                ):
                    self._record_entry_locked(
                        signal,
                        price=close_price,
                        timestamp=timestamp,
                        source="closed_candle_confirmation",
                    )
                    count += 1
            self._connection.commit()
        return count

    def apply_time_rules(self, now: Optional[datetime] = None) -> int:
        timestamp = now or _utc_now()
        count = 0
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM signals
                WHERE status IN ('pending', 'entered')
                ORDER BY created_at
                """
            ).fetchall()
            for db_row in rows:
                signal = self._decode_signal(db_row)
                if (
                    signal["status"] == "pending"
                    and timestamp >= _parse_datetime(signal["valid_until"])
                ):
                    self._terminal_transition_locked(
                        signal,
                        status="expired",
                        event_type="expired",
                        timestamp=timestamp,
                        price=signal.get("last_price"),
                        reason="Entry window expired before the zone was reached",
                    )
                    count += 1
                elif (
                    signal["status"] == "entered"
                    and signal.get("hold_until")
                    and timestamp >= _parse_datetime(signal["hold_until"])
                ):
                    last_price = safe_float(signal.get("last_price"))
                    allocations = _normalize_allocations(
                        self._allocations_config,
                        len(signal["take_profits"]),
                    )
                    completed_fraction = sum(
                        allocations[: int(signal.get("highest_tp") or 0)]
                    )
                    open_r = _r_at_price(signal, last_price) if last_price > 0 else 0.0
                    result_r = safe_float(signal.get("realized_r")) + (
                        1.0 - completed_fraction
                    ) * open_r
                    self._connection.execute(
                        "UPDATE signals SET realized_r = ? WHERE id = ?",
                        (result_r, signal["id"]),
                    )
                    self._terminal_transition_locked(
                        signal,
                        status="time_exit",
                        event_type="time_exit",
                        timestamp=timestamp,
                        price=last_price or None,
                        reason="Maximum day-trade hold window reached",
                        payload={
                            "result_r": result_r,
                            "highest_tp": int(signal.get("highest_tp") or 0),
                            "mfe_r": safe_float(signal.get("mfe_r")),
                            "mae_r": safe_float(signal.get("mae_r")),
                        },
                    )
                    count += 1
            self._connection.commit()
        return count

    def pending_events(
        self,
        *,
        retry_after_seconds: int,
        limit: int = 50,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        cutoff = (now or _utc_now()) - timedelta(seconds=retry_after_seconds)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT *
                FROM signal_events
                WHERE notified_at IS NULL
                  AND (last_attempt_at IS NULL OR last_attempt_at <= ?)
                ORDER BY id
                LIMIT ?
                """,
                (_iso(cutoff), int(limit)),
            ).fetchall()
            events: List[Dict[str, Any]] = []
            for row in rows:
                event = self._decode_event(row)
                signal_row = self._connection.execute(
                    "SELECT * FROM signals WHERE id = ?",
                    (event["signal_id"],),
                ).fetchone()
                if signal_row is None:
                    continue
                event["signal"] = self._decode_signal(signal_row)
                events.append(event)
        return events

    def record_delivery_attempt(
        self,
        event_id: int,
        successful_destinations: Sequence[str],
        *,
        all_delivered: bool,
        now: Optional[datetime] = None,
    ) -> None:
        timestamp = now or _utc_now()
        delivered = _unique_strings(successful_destinations)
        with self._lock:
            self._connection.execute(
                """
                UPDATE signal_events
                SET delivered_to_json = ?, attempts = attempts + 1,
                    last_attempt_at = ?, notified_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(delivered, separators=(",", ":")),
                    _iso(timestamp),
                    _iso(timestamp) if all_delivered else None,
                    int(event_id),
                ),
            )
            self._connection.commit()

def _base_symbol(symbol: str) -> str:
    raw = str(symbol or "").upper().strip()
    if "/" in raw:
        return raw.split("/", 1)[0]
    for quote in ("USDT", "USDC", "USD"):
        if raw.endswith(quote) and len(raw) > len(quote):
            return raw[: -len(quote)]
    return raw.split(":", 1)[0]


def _okx_instrument_id(symbol: str) -> str:
    return f"{_base_symbol(symbol)}-USDT-SWAP"


class OKXActiveTickerStream:
    """One dynamic public WebSocket connection for active OKX swaps only."""

    def __init__(
        self,
        url: str,
        on_price: Callable[[str, float, datetime], None],
        on_status: Callable[..., None],
    ) -> None:
        self.url = url
        self.on_price = on_price
        self.on_status = on_status
        self._lock = RLock()
        self._desired: Set[str] = set()
        self._subscribed: Set[str] = set()
        self._app: Any = None
        self._thread: Optional[Thread] = None
        self._shutdown = Event()
        self.connected = False

    def set_symbols(self, instruments: Iterable[str]) -> None:
        desired = {str(item) for item in instruments if str(item).strip()}
        with self._lock:
            previous = set(self._desired)
            self._desired = desired
            app = self._app
            connected = self.connected
        added = desired - previous
        removed = previous - desired
        if connected and app is not None:
            if removed:
                self._send_operation("unsubscribe", removed)
            if added:
                self._send_operation("subscribe", added)
        if desired:
            self._ensure_thread()
        elif app is not None:
            try:
                app.close()
            except Exception:  # noqa: BLE001
                pass

    def _ensure_thread(self) -> None:
        if websocket is None:
            self.on_status(
                websocket_connected=False,
                last_error="websocket-client is not installed; REST fallback active",
            )
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = Thread(
                target=self._run,
                name="perpetual-pro-okx-stream",
                daemon=True,
            )
            self._thread.start()

    def _send_operation(self, operation: str, instruments: Iterable[str]) -> None:
        args = [
            {"channel": "tickers", "instId": instrument}
            for instrument in sorted(set(instruments))
        ]
        if not args:
            return
        payload = json.dumps({"op": operation, "args": args})
        try:
            with self._lock:
                app = self._app
            if app is not None:
                app.send(payload)
                with self._lock:
                    if operation == "subscribe":
                        self._subscribed.update(item["instId"] for item in args)
                    else:
                        self._subscribed.difference_update(
                            item["instId"] for item in args
                        )
        except Exception as exc:  # noqa: BLE001
            self.on_status(last_error=f"WebSocket {operation}: {type(exc).__name__}")

    def _on_open(self, app: Any) -> None:
        with self._lock:
            self.connected = True
            desired = set(self._desired)
            self._subscribed.clear()
        self.on_status(
            websocket_connected=True,
            last_error=None,
            last_websocket_connected_at=_iso(_utc_now()),
        )
        self._send_operation("subscribe", desired)
        logger.info(
            "Signal tracker WebSocket connected: subscriptions={}",
            len(desired),
        )

    def _on_message(self, app: Any, message: Any) -> None:
        if message == "pong":
            return
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        if payload.get("event") == "error":
            code = str(payload.get("code") or "unknown")
            self.on_status(last_error=f"OKX WebSocket subscription error {code}")
            logger.error("OKX WebSocket subscription error: code={}", code)
            return
        arg = payload.get("arg") if isinstance(payload.get("arg"), dict) else {}
        instrument = str(arg.get("instId") or "")
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            price = safe_float(item.get("last"))
            if price <= 0:
                continue
            ts_ms = safe_float(item.get("ts"))
            observed = (
                datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
                if ts_ms > 0
                else _utc_now()
            )
            self.on_price(instrument, price, observed)

    def _on_error(self, app: Any, error: Any) -> None:
        error_type = type(error).__name__
        self.on_status(
            websocket_connected=False,
            last_error=f"WebSocket {error_type}",
        )
        logger.warning(
            "Signal tracker WebSocket error: error_type={}",
            error_type,
        )

    def _on_close(
        self,
        app: Any,
        status_code: Any,
        message: Any,
    ) -> None:
        with self._lock:
            self.connected = False
            self._subscribed.clear()
        self.on_status(websocket_connected=False)
        logger.info(
            "Signal tracker WebSocket disconnected: code={}",
            status_code,
        )

    def _run(self) -> None:
        try:
            while not self._shutdown.is_set():
                with self._lock:
                    if not self._desired:
                        return
                app = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                with self._lock:
                    self._app = app
                try:
                    app.run_forever(
                        ping_interval=20,
                        ping_timeout=10,
                        ping_payload="ping",
                    )
                except Exception as exc:  # noqa: BLE001
                    self.on_status(
                        websocket_connected=False,
                        last_error=f"WebSocket loop {type(exc).__name__}",
                    )
                finally:
                    with self._lock:
                        self._app = None
                        self.connected = False
                        self._subscribed.clear()
                if self._shutdown.wait(5):
                    return
        finally:
            with self._lock:
                self._thread = None

    def stop(self, timeout: float = 5.0) -> None:
        self._shutdown.set()
        with self._lock:
            app = self._app
            thread = self._thread
        if app is not None:
            try:
                app.close()
            except Exception:  # noqa: BLE001
                pass
        if thread and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))


def _format_price(value: Any) -> str:
    price = safe_float(value)
    if price <= 0:
        return "—"
    if price >= 1000:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:.4f}"
    return f"${price:.8f}"


def format_tracker_event(event: Dict[str, Any]) -> str:
    """Compact Telegram follow-up for a persisted lifecycle transition."""
    signal = event["signal"]
    payload = event.get("payload") or {}
    base = html.escape(_base_symbol(signal.get("symbol")))
    direction = str(signal.get("direction") or "").upper()
    event_type = str(event.get("event_type") or "")
    price = event.get("price")
    highest_tp = int(payload.get("tp_number") or payload.get("highest_tp") or 0)
    result_r = payload.get("result_r")

    if event_type == "entered":
        title = f"🟢 <b>{base} {direction} — ENTRY TRIGGERED</b>"
        body = [
            f"Observed entry: <b>{_format_price(price)}</b>",
            f"Signal delay: <b>{safe_float(payload.get('entry_delay_minutes')):.0f} min</b>",
            (
                "Fill proxy vs zone midpoint: "
                f"<b>{safe_float(payload.get('slippage_bps')):+.1f} bps</b>"
            ),
            "TP and hard Stop tracking is now active.",
        ]
    elif event_type in ("target_hit", "completed"):
        tp_number = int(payload.get("tp_number") or 0)
        complete = event_type == "completed"
        title = (
            f"🏁 <b>{base} {direction} — ALL TARGETS HIT</b>"
            if complete
            else f"✅ <b>{base} {direction} — TP{tp_number} HIT</b>"
        )
        body = [
            f"Target: <b>{_format_price(payload.get('target_price') or price)}</b>",
            f"Realized model result: <b>+{safe_float(payload.get('realized_r')):.2f}R</b>",
            (
                "Signal closed successfully."
                if complete
                else (
                    "Profit secured. Remaining targets stay active with the "
                    "remainder protected at breakeven."
                )
            ),
        ]
    elif event_type == "protected_exit":
        title = f"🟢 <b>{base} {direction} — PROFIT SECURED</b>"
        body = [
            (
                "Protected exit: "
                f"<b>{_format_price(payload.get('protected_price') or price)}</b>"
            ),
            f"Highest target reached: <b>TP{highest_tp}</b>",
            f"Realized model result: <b>+{safe_float(result_r):.2f}R</b>",
            "The remainder exited at breakeven protection; this is not a stop-loss.",
        ]
    elif event_type == "stopped":
        if highest_tp >= 1:
            # Compatibility for a queued event created by an older deployment.
            title = f"🟢 <b>{base} {direction} — PROFIT SECURED</b>"
            body = [
                f"Highest target reached: <b>TP{highest_tp}</b>",
                "TP1 had already locked a winning outcome.",
                "No stop-loss is recorded after TP1.",
            ]
        else:
            title = f"🛑 <b>{base} {direction} — STOP HIT</b>"
            body = [
                f"Stop event: <b>{_format_price(price)}</b>",
                f"Model result: <b>{safe_float(result_r):+.2f}R</b>",
                f"Targets reached first: <b>{highest_tp}</b>",
                (
                    f"MFE <b>{safe_float(payload.get('mfe_r')):+.2f}R</b> · "
                    f"MAE <b>{safe_float(payload.get('mae_r')):+.2f}R</b>"
                ),
            ]
    elif event_type == "missed":
        title = f"⚪ <b>{base} {direction} — SETUP MISSED</b>"
        body = [
            "TP1 traded before the planned entry zone.",
            "The old entry is cancelled. Do not chase or trade a later return automatically.",
            "A new scan must qualify a fresh setup.",
        ]
    elif event_type == "ambiguous_gap":
        title = f"⚪ <b>{base} {direction} — OUTCOME AMBIGUOUS</b>"
        body = [
            "A market-data gap crossed both execution and outcome levels.",
            "The event order cannot be proven, so it is not counted as a fill, win, or loss.",
            "The setup is closed and requires a fresh scan.",
        ]
    elif event_type == "invalidated":
        title = f"⚠️ <b>{base} {direction} — STRUCTURE INVALIDATED</b>"
        body = [
            f"Confirmed close: <b>{_format_price(price)}</b>",
            "A candle finished beyond the Stop before entry; the setup is cancelled.",
            "A brief wick alone would not have caused this cancellation.",
        ]
    elif event_type == "time_exit":
        title = f"⏰ <b>{base} {direction} — TIME EXIT</b>"
        body = [
            "The maximum day-trade hold window was reached.",
            f"Model result at last price: <b>{safe_float(result_r):+.2f}R</b>",
            f"Targets reached: <b>{highest_tp}</b>",
        ]
    else:
        title = f"⌛ <b>{base} {direction} — SETUP EXPIRED</b>"
        body = [
            "The entry zone was not reached inside its validity window.",
            "The old levels are cancelled. Run a new scan before considering the market.",
        ]

    message = "\n".join(
        [
            title,
            "",
            *body,
            "",
            "NFA · DYOR · Trade at your own risk",
        ]
    )
    if str(signal.get("source") or "").startswith("production_verification"):
        return "🧪 <b>CONTROLLED LIFECYCLE TEST — NOT A TRADE</b>\n\n" + message
    return message


class SignalTracker:
    """Coordinates the persistent store, WebSocket, REST, and Telegram."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.store = SignalStore(
            _resolve_database_path(config),
            target_allocations=config.signal_tracker.target_allocations,
        )
        self.outcome_repository = OutcomeRepository(
            config.outcome_scoring.database_url
        )
        self.lifecycle_repository = LifecycleRepository(
            config.outcome_scoring.database_url
        )
        self._stop = Event()
        self._queue: queue.Queue[
            Tuple[str, str, float, datetime, str]
        ] = queue.Queue(maxsize=5000)
        self._thread: Optional[Thread] = None
        self._exchange_clients: Dict[str, ExchangeClient] = {}
        self._closed = False
        self._instrument_symbols: Dict[str, List[str]] = {}
        self._last_observation_sync: Dict[str, float] = {}
        self._status_lock = Lock()
        self._status: Dict[str, Any] = {
            "enabled": bool(config.signal_tracker.enabled),
            "running": False,
            "thread_alive": False,
            "database_ready": True,
            "database_persistent_volume": bool(
                (os.getenv("RAILWAY_VOLUME_MOUNT_PATH") or "").strip()
                or (
                    (os.getenv("SIGNAL_TRACKER_DB_PATH") or "").strip().startswith(
                        "/data/"
                    )
                )
            ),
            "durable_outcome_database_configured": bool(
                config.outcome_scoring.database_url
            ),
            "durable_outcome_database_ready": False,
            "durable_lifecycle_required": bool(
                config.signal_tracker.durable_lifecycle_required
            ),
            "durable_lifecycle_ready": False,
            "recovery_completed": False,
            "recovery_failed": False,
            "recovery": {
                "active_found": 0,
                "restored": 0,
                "expired": 0,
                "reconciled": 0,
                "ambiguous": 0,
                "invalid": 0,
                "duplicates_skipped": 0,
            },
            "websocket_enabled": bool(
                config.signal_tracker.websocket_enabled
            ),
            "websocket_connected": False,
            "subscribed_symbols": [],
            "active_count": 0,
            "pending_count": 0,
            "entered_count": 0,
            "started_at": None,
            "last_price_at": None,
            "last_reconcile_at": None,
            "last_notification_at": None,
            "last_websocket_connected_at": None,
            "last_error": None,
        }
        self.stream = OKXActiveTickerStream(
            config.signal_tracker.websocket_url,
            self._on_stream_price,
            self._update_status,
        )
        self._refresh_counts_and_subscriptions()

    def _update_status(self, **values: Any) -> None:
        with self._status_lock:
            self._status.update(values)
            self._status["thread_alive"] = bool(
                self._thread and self._thread.is_alive()
            )

    def status(self) -> Dict[str, Any]:
        if not self._closed:
            self._refresh_counts_only()
        with self._status_lock:
            status = dict(self._status)
        status["thread_alive"] = bool(self._thread and self._thread.is_alive())
        durable = self.outcome_repository.status()
        status["durable_outcome_database"] = durable
        status["durable_outcome_database_ready"] = bool(durable.get("ready"))
        lifecycle = self.lifecycle_repository.status()
        status["durable_lifecycle_database"] = lifecycle
        status["durable_lifecycle_ready"] = bool(lifecycle.get("ready"))
        return status

    def _recover(self) -> bool:
        """Restore active Supabase state before any tracker worker can run."""
        required = bool(self.config.signal_tracker.durable_lifecycle_required)
        if not self.lifecycle_repository.check_ready():
            self._update_status(
                durable_lifecycle_ready=False,
                recovery_completed=False,
                recovery_failed=required,
                last_error="Durable lifecycle repository unavailable",
            )
            return not required
        stats = {
            "active_found": 0,
            "restored": 0,
            "expired": 0,
            "reconciled": 0,
            "ambiguous": 0,
            "invalid": 0,
            "duplicates_skipped": 0,
        }
        try:
            records = self.lifecycle_repository.load_active()
            stats["active_found"] = len(records)
            seen: Set[str] = set()
            for signal in records:
                signal_id = str(signal.get("id") or "")
                if not signal_id:
                    stats["invalid"] += 1
                    continue
                if signal_id in seen:
                    stats["duplicates_skipped"] += 1
                    continue
                seen.add(signal_id)
                if (
                    str(signal.get("_lifecycle_schema_version") or "")
                    not in ("", LIFECYCLE_SCHEMA_VERSION)
                    or str(
                        signal.get("_feature_schema_version")
                        or self.config.outcome_scoring.feature_schema_version
                    ) != self.config.outcome_scoring.feature_schema_version
                    or str(
                        signal.get("_execution_policy_version")
                        or self.config.analysis.execution_policy_version
                    ) != self.config.analysis.execution_policy_version
                    or str(
                        signal.get("_rank_policy_version") or RANK_POLICY_VERSION
                    ) != RANK_POLICY_VERSION
                    or
                    str(signal.get("status")) not in ACTIVE_STATUSES
                    or str(signal.get("direction")) not in ("long", "short")
                    or not list(signal.get("take_profits") or [])
                ):
                    stats["invalid"] += 1
                    continue
                if self.store.restore_signal(signal):
                    stats["restored"] += 1
            expired = self.store.apply_time_rules(_utc_now())
            stats["expired"] = int(expired)
            if expired:
                self._sync_all_signals()
            self._refresh_counts_and_subscriptions()
            if stats["invalid"] and required:
                self._update_status(
                    durable_lifecycle_ready=True,
                    recovery_completed=False,
                    recovery_failed=True,
                    recovery=stats,
                    last_error="Incompatible active durable lifecycle records",
                )
                logger.error(
                    "Lifecycle recovery blocked by incompatible active records: count={}",
                    stats["invalid"],
                )
                return False
            self._update_status(
                durable_lifecycle_ready=True,
                recovery_completed=not bool(self.store.active_signals()),
                recovery_failed=False,
                recovery=stats,
                last_error=None,
            )
            logger.info(
                "Lifecycle recovery complete: found={} restored={} expired={} "
                "invalid={} duplicates_skipped={}",
                stats["active_found"], stats["restored"], stats["expired"],
                stats["invalid"], stats["duplicates_skipped"],
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._update_status(
                durable_lifecycle_ready=False,
                recovery_completed=False,
                recovery_failed=True,
                recovery=stats,
                last_error=f"Lifecycle recovery {type(exc).__name__}",
            )
            logger.error("Lifecycle recovery failed: error_type={}", type(exc).__name__)
            return False

    def _refresh_counts_only(self) -> None:
        counts = self.store.counts()
        self._update_status(
            active_count=counts.get("active", 0),
            pending_count=counts.get("pending", 0),
            entered_count=counts.get("entered", 0),
        )

    def _refresh_counts_and_subscriptions(self) -> None:
        active = self.store.active_signals()
        instrument_symbols: Dict[str, List[str]] = {}
        for signal in active:
            if normalize_exchange_id(signal["exchange_id"]) != "okx":
                continue
            instrument = _okx_instrument_id(signal["symbol"])
            instrument_symbols.setdefault(instrument, []).append(signal["symbol"])
        self._instrument_symbols = instrument_symbols
        subscribed_symbols = sorted(
            {signal["symbol"] for signal in active if signal["exchange_id"] == "okx"}
        )
        self._update_status(
            active_count=len(active),
            pending_count=sum(1 for item in active if item["status"] == "pending"),
            entered_count=sum(1 for item in active if item["status"] == "entered"),
            subscribed_symbols=subscribed_symbols,
        )
        if self.config.signal_tracker.websocket_enabled:
            self.stream.set_symbols(instrument_symbols)

    def start(self) -> bool:
        if not self.config.signal_tracker.enabled:
            self._update_status(enabled=False, running=False)
            return False
        if self._thread and self._thread.is_alive():
            return True
        if not self._recover():
            return False
        active_before_reconcile = len(self.store.active_signals())
        if active_before_reconcile:
            reconciliation = self._reconcile()
            recovery = dict(self.status().get("recovery") or {})
            recovery["reconciled"] = max(
                0,
                int(reconciliation.get("attempted") or 0)
                - len(reconciliation.get("errors") or []),
            )
            recovery["ambiguous"] = sum(
                1
                for item in self.store.signals_for_sync()
                if str(item.get("status")) == "ambiguous_gap"
            )
            all_failed = bool(
                reconciliation.get("attempted")
                and len(reconciliation.get("errors") or [])
                >= int(reconciliation.get("attempted") or 0)
            )
            self._update_status(
                recovery=recovery,
                recovery_completed=not all_failed,
                recovery_failed=all_failed,
            )
            if all_failed and self.config.signal_tracker.durable_lifecycle_required:
                logger.error("Lifecycle startup reconciliation failed for every active market")
                return False
        self._stop.clear()
        self._thread = Thread(
            target=self._run,
            name="perpetual-pro-signal-tracker",
            daemon=True,
        )
        self._thread.start()
        self._update_status(
            enabled=True,
            running=True,
            started_at=_iso(_utc_now()),
        )
        logger.info(
            "Signal tracker started: active={} reconcile={}s websocket={}",
            len(self.store.active_signals()),
            self.config.signal_tracker.reconcile_interval_seconds,
            self.config.signal_tracker.websocket_enabled,
        )
        return True

    def register(
        self,
        row: Dict[str, Any],
        destinations: Sequence[str],
        *,
        source: str,
    ) -> Dict[str, Any]:
        signal, created = self.store.register_signal(
            row,
            destinations,
            source=source,
        )
        durable = self._sync_signals([signal])
        self._refresh_counts_and_subscriptions()
        logger.info(
            "Signal tracker {}: symbol={} direction={} status={} destinations={}",
            "registered" if created else "merged",
            signal["symbol"],
            signal["direction"],
            signal["status"],
            len(signal["destinations"]),
        )
        return {
            "ok": bool(durable or not self.config.signal_tracker.durable_lifecycle_required),
            "created": created,
            "signal_id": signal["id"],
            "status": signal["status"],
        }

    def _on_stream_price(
        self,
        instrument: str,
        price: float,
        observed_at: datetime,
    ) -> None:
        symbols = list(self._instrument_symbols.get(instrument) or [])
        for symbol in symbols:
            try:
                self._queue.put_nowait(
                    ("okx", symbol, price, observed_at, "websocket")
                )
            except queue.Full:
                self._update_status(last_error="Price queue full; event dropped")
                logger.error("Signal tracker price queue full; event dropped")
                break

    def _run(self) -> None:
        reconcile_interval = self.config.signal_tracker.reconcile_interval_seconds
        lifecycle_interval = self.config.signal_tracker.lifecycle_check_seconds
        next_reconcile = time.monotonic()
        next_lifecycle = time.monotonic()
        next_notification = time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    exchange_id, symbol, price, observed, source = self._queue.get(
                        timeout=1.0
                    )
                    transitions = self.store.process_price(
                        symbol,
                        price,
                        observed,
                        source=source,
                        exchange_id=exchange_id,
                    )
                    self._update_status(last_price_at=_iso(observed))
                    checkpoint_due = (
                        time.monotonic()
                        - self._last_observation_sync.get(symbol, 0.0)
                        >= 30.0
                    )
                    if transitions or checkpoint_due:
                        self._sync_symbol(symbol)
                        self._last_observation_sync[symbol] = time.monotonic()
                    if transitions:
                        self._refresh_counts_and_subscriptions()
                    self._queue.task_done()
                except queue.Empty:
                    pass
                except Exception as exc:  # noqa: BLE001
                    self._update_status(
                        last_error=f"Price processing {type(exc).__name__}"
                    )
                    logger.exception("Signal tracker price processing failed: {}", exc)

                now_mono = time.monotonic()
                if now_mono >= next_lifecycle:
                    try:
                        transitions = self.store.apply_time_rules()
                        if transitions:
                            self._sync_all_signals()
                            self._refresh_counts_and_subscriptions()
                    except Exception as exc:  # noqa: BLE001
                        self._update_status(
                            last_error=f"Lifecycle check {type(exc).__name__}"
                        )
                        logger.exception("Signal tracker lifecycle check failed: {}", exc)
                    next_lifecycle = now_mono + lifecycle_interval

                if now_mono >= next_notification:
                    self._flush_notifications()
                    next_notification = now_mono + 5

                if now_mono >= next_reconcile:
                    self._reconcile()
                    next_reconcile = time.monotonic() + reconcile_interval
        finally:
            self._update_status(running=False)

    def _client(self, exchange_id: str) -> ExchangeClient:
        normalized = normalize_exchange_id(exchange_id)
        if normalized not in self._exchange_clients:
            self._exchange_clients[normalized] = ExchangeClient(
                exchange_id=normalized,
                config=self.config,
            )
        return self._exchange_clients[normalized]

    def _reconcile(self) -> Dict[str, Any]:
        active = self.store.active_signals()
        if not active:
            self._update_status(last_reconcile_at=_iso(_utc_now()))
            return {"attempted": 0, "errors": []}
        markets = sorted(
            {
                (normalize_exchange_id(signal["exchange_id"]), signal["symbol"])
                for signal in active
            }
        )
        errors: List[str] = []
        for exchange_id, symbol in markets:
            try:
                client = self._client(exchange_id)
                ticker = client.fetch_ticker(symbol)
                price = safe_float(
                    ticker.get("last")
                    or ticker.get("close")
                    or ticker.get("mark")
                )
                timestamp_ms = safe_float(ticker.get("timestamp"))
                observed = (
                    datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)
                    if timestamp_ms > 0
                    else _utc_now()
                )
                timeframe = next(
                    (
                        item["timeframe"]
                        for item in active
                        if item["symbol"] == symbol
                        and normalize_exchange_id(item["exchange_id"])
                        == exchange_id
                    ),
                    "15m",
                )
                candles = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=5)
                if candles is not None and not candles.empty:
                    tf_minutes = _timeframe_minutes(timeframe)
                    now = _utc_now()
                    for index, candle in candles.iterrows():
                        opened = index.to_pydatetime()
                        if opened.tzinfo is None:
                            opened = opened.replace(tzinfo=UTC)
                        closed_at = opened.astimezone(UTC) + timedelta(
                            minutes=tf_minutes
                        )
                        if closed_at <= now:
                            self.store.process_closed_candle(
                                symbol,
                                safe_float(candle.get("close")),
                                closed_at,
                                exchange_id=exchange_id,
                                high_price=safe_float(candle.get("high")),
                                low_price=safe_float(candle.get("low")),
                            )
                # Closed-candle invalidations happened before this current
                # ticker observation, so reconcile them first.
                if price > 0:
                    self.store.process_price(
                        symbol,
                        price,
                        observed,
                        source="rest_reconcile",
                        exchange_id=exchange_id,
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{exchange_id}:{symbol}:{type(exc).__name__}")
                logger.warning(
                    "Signal tracker REST reconciliation failed: exchange={} "
                    "symbol={} error_type={}",
                    exchange_id,
                    symbol,
                    type(exc).__name__,
                )
        self._refresh_counts_and_subscriptions()
        self._sync_all_signals()
        self._update_status(
            last_reconcile_at=_iso(_utc_now()),
            last_error=(
                f"REST reconciliation errors: {', '.join(errors[:3])}"
                if errors
                else None
            ),
        )
        return {"attempted": len(markets), "errors": errors}

    def _sync_symbol(self, symbol: str) -> None:
        self._sync_signals(self.store.signals_for_sync(symbol))

    def _sync_all_signals(self) -> None:
        self._sync_signals(self.store.signals_for_sync())

    def _sync_signals(self, signals: Sequence[Dict[str, Any]]) -> bool:
        all_durable = True
        for signal in signals:
            events = self.store.events_for_sync(str(signal.get("id") or ""))
            durable = self.lifecycle_repository.persist_state_and_events(
                signal,
                events,
            )
            # Preserve the original Phase 1/2A mirror/outcome path for backward
            # compatibility. It is no longer the lifecycle source of truth.
            mirrored = self.outcome_repository.upsert_tracked_signal(signal)
            if (
                mirrored
                and str(signal.get("status") or "") in DURABLE_TERMINAL_STATUSES
            ):
                self.outcome_repository.upsert_outcome_from_signal(signal)
            if not durable and self.config.signal_tracker.durable_lifecycle_required:
                all_durable = False
                self._update_status(
                    durable_lifecycle_ready=False,
                    last_error="Durable lifecycle commit failed",
                )
        return all_durable

    def _flush_notifications(self) -> None:
        jobs = self.lifecycle_repository.claim_notifications(limit=25)
        for job in jobs:
            payload = dict(job.get("payload") or {})
            destinations = _unique_strings(payload.pop("destinations", []))
            destination = next(
                (
                    item for item in destinations
                    if destination_hash(item) == str(job.get("destination_hash") or "")
                ),
                None,
            )
            if destination is None:
                self.lifecycle_repository.finish_notification(
                    int(job["id"]),
                    delivered=False,
                    error_category="destination_unavailable",
                )
                continue
            event = {
                "event_type": job.get("event_type"),
                "occurred_at": job.get("occurred_at"),
                "price": job.get("price"),
                "payload": payload,
                "signal": dict(job.get("lifecycle_state") or {}),
            }
            result = send_telegram_message_detailed(
                format_tracker_event(event),
                chat_id=destination,
                parse_mode=self.config.telegram.parse_mode or "HTML",
            )
            success = bool(result.get("ok"))
            self.lifecycle_repository.finish_notification(
                int(job["id"]),
                delivered=success,
                message_id=result.get("message_id"),
                error_category=result.get("error"),
                retry_after_seconds=result.get("retry_after"),
            )
            if success:
                self._update_status(last_notification_at=_iso(_utc_now()))
            else:
                logger.error(
                    "Signal tracker Telegram update failed: event={} destination={} error={}",
                    job.get("event_type"),
                    str(job.get("destination_hash") or "")[:10],
                    result.get("error"),
                )

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        self.stream.stop(timeout=min(5.0, timeout))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, timeout))
        for client in self._exchange_clients.values():
            client.close()
        self._exchange_clients.clear()
        self.store.close()
        self._closed = True
        self._update_status(running=False, websocket_connected=False)
        logger.info("Signal tracker stopped")


def _timeframe_minutes(timeframe: str) -> int:
    raw = str(timeframe or "15m").strip().lower()
    try:
        if raw.endswith("m"):
            return max(1, int(raw[:-1]))
        if raw.endswith("h"):
            return max(1, int(raw[:-1]) * 60)
        if raw.endswith("d"):
            return max(1, int(raw[:-1]) * 1440)
    except ValueError:
        return 15
    return 15


_TRACKER_LOCK = Lock()
_TRACKER: Optional[SignalTracker] = None
_LAST_STATUS: Dict[str, Any] = {
    "enabled": False,
    "running": False,
    "thread_alive": False,
    "database_ready": False,
    "websocket_enabled": False,
    "websocket_connected": False,
    "subscribed_symbols": [],
    "active_count": 0,
    "pending_count": 0,
    "entered_count": 0,
    "last_error": None,
}


def start_signal_tracker_background(
    config: Optional[AppConfig] = None,
) -> bool:
    """Start one tracker worker per API process."""
    global _TRACKER, _LAST_STATUS
    cfg = config or load_config()
    if not cfg.signal_tracker.enabled:
        _LAST_STATUS = {**_LAST_STATUS, "enabled": False}
        return False
    with _TRACKER_LOCK:
        if _TRACKER is not None:
            return _TRACKER.start()
        try:
            tracker = SignalTracker(cfg)
            started = tracker.start()
            if not started:
                _LAST_STATUS = tracker.status()
                tracker.stop()
                return False
            _TRACKER = tracker
            _LAST_STATUS = tracker.status()
            return True
        except Exception as exc:  # noqa: BLE001
            _LAST_STATUS = {
                **_LAST_STATUS,
                "enabled": True,
                "database_ready": False,
                "last_error": f"{type(exc).__name__}: {exc}",
            }
            logger.exception("Signal tracker failed to start: {}", exc)
            return False


def stop_signal_tracker_background(timeout: float = 8.0) -> None:
    """Stop the process-local tracker worker."""
    global _TRACKER, _LAST_STATUS
    with _TRACKER_LOCK:
        tracker = _TRACKER
        _TRACKER = None
    if tracker is not None:
        previous = tracker.status()
        tracker.stop(timeout=timeout)
        _LAST_STATUS = {
            **previous,
            "running": False,
            "thread_alive": False,
            "websocket_connected": False,
        }


def get_signal_tracker_status() -> Dict[str, Any]:
    """Return a redacted JSON-safe tracker health snapshot."""
    if _TRACKER is not None:
        return _TRACKER.status()
    return dict(_LAST_STATUS)


def get_signal_reliability_summary() -> Dict[str, Any]:
    """Return forward performance bands from the process-local persistent store."""
    if _TRACKER is None:
        return {
            "status": "tracker_not_running",
            "completed_outcomes": 0,
            "minimum_samples_per_band": 50,
            "bands": [],
        }
    return _TRACKER.store.reliability_summary()


def register_delivered_signals(
    rows: Sequence[Dict[str, Any]],
    destinations_by_signal: Sequence[Sequence[str]],
    *,
    source: str,
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    """Register only signals that reached at least one Telegram destination."""
    cfg = config or load_config()
    if not cfg.signal_tracker.enabled:
        return {"ok": True, "enabled": False, "registered": 0}
    candidates: List[Tuple[int, Dict[str, Any], List[str]]] = []
    validation_errors: List[str] = []
    for index, row in enumerate(rows):
        destinations = (
            list(destinations_by_signal[index])
            if index < len(destinations_by_signal)
            else []
        )
        if not destinations:
            continue
        trackable = bool(
            row.get("symbol")
            and str(row.get("direction") or "").lower() in ("long", "short")
            and safe_float(row.get("entry_low")) > 0
            and safe_float(row.get("entry_high")) > 0
            and safe_float(row.get("stop_loss")) > 0
            and list(row.get("take_profits") or [])
        )
        if trackable:
            candidates.append((index, row, destinations))
        else:
            validation_errors.append(
                f"{str(row.get('symbol') or f'row-{index}')}:invalid_levels"
            )
    if not candidates:
        return {
            "ok": not validation_errors,
            "enabled": True,
            "registered": 0,
            "merged": 0,
            "errors": validation_errors,
        }
    if not start_signal_tracker_background(cfg):
        return {
            "ok": False,
            "enabled": True,
            "registered": 0,
            "error": get_signal_tracker_status().get("last_error"),
        }
    tracker = _TRACKER
    if tracker is None:
        return {"ok": False, "registered": 0, "error": "tracker_unavailable"}
    registered = 0
    merged = 0
    errors: List[str] = list(validation_errors)
    for index, row, destinations in candidates:
        try:
            result = tracker.register(row, destinations, source=source)
            if not result.get("ok"):
                errors.append(f"{str(row.get('symbol') or f'row-{index}')}:durable_commit_failed")
                continue
            if result.get("created"):
                registered += 1
            else:
                merged += 1
        except Exception as exc:  # noqa: BLE001
            symbol = str(row.get("symbol") or f"row-{index}")
            errors.append(f"{symbol}:{type(exc).__name__}")
            logger.error(
                "Delivered signal could not be tracked: symbol={} error_type={}",
                symbol,
                type(exc).__name__,
            )
    return {
        "ok": not errors,
        "enabled": True,
        "registered": registered,
        "merged": merged,
        "errors": errors,
    }
