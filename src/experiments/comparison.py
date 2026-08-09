"""Durable Strict-shadow versus Legacy performance reporting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover
    psycopg = None
    dict_row = None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_streak(values: Iterable[bool], wanted: bool) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value is wanted else 0
        best = max(best, current)
    return best


def build_comparison_report(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    candidates = [dict(row) for row in rows]
    legacy_signals = [
        row for row in candidates
        if bool((row.get("legacy_decision") or {}).get("qualified"))
    ]
    both = [
        row for row in legacy_signals
        if (row.get("strict_shadow_decision") or {}).get("decision") == "SIGNAL"
    ]
    legacy_only = [row for row in legacy_signals if row not in both]
    completed = [
        row for row in legacy_signals
        if str((row.get("outcome") or {}).get("terminal_status") or "")
        not in {"", "pending", "entered"}
    ]
    filled = [row for row in completed if bool((row.get("outcome") or {}).get("valid_fill"))]
    tp1 = [row for row in filled if bool((row.get("outcome") or {}).get("tp1_hit"))]
    realized = [_number((row.get("outcome") or {}).get("realized_r")) for row in completed]
    gains = sum(value for value in realized if value > 0)
    losses = abs(sum(value for value in realized if value < 0))
    equity = peak = max_drawdown = 0.0
    win_flags: List[bool] = []
    for value in realized:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        win_flags.append(value > 0)
    generated = sorted(
        row.get("generated_at") for row in candidates if row.get("generated_at")
    )
    days = 1.0
    if len(generated) >= 2:
        try:
            first = datetime.fromisoformat(str(generated[0]).replace("Z", "+00:00"))
            last = datetime.fromisoformat(str(generated[-1]).replace("Z", "+00:00"))
            days = max(1.0, (last - first).total_seconds() / 86400.0)
        except ValueError:
            days = 1.0
    return {
        "candidate_count": len(candidates),
        "legacy_signal_count": len(legacy_signals),
        "strict_shadow_signal_count": len(both),
        "strict_would_reject_count": len(legacy_only),
        "legacy_signals_per_day": round(len(legacy_signals) / days, 3),
        "completed_outcomes": len(completed),
        "fill_rate": round(len(filled) / len(completed), 4) if completed else None,
        "tp1_hit_rate": round(len(tp1) / len(filled), 4) if filled else None,
        "average_realized_r": round(sum(realized) / len(realized), 4) if realized else None,
        "profit_factor": round(gains / losses, 4) if losses else (None if not gains else "infinite"),
        "maximum_drawdown_r": round(max_drawdown, 4),
        "maximum_losing_streak": _max_streak(win_flags, False),
        "maximum_winning_streak": _max_streak(win_flags, True),
        "legacy_only": len(legacy_only),
        "both_policies": len(both),
        "sample_warning": (
            "Outcome conclusions are preliminary"
            if len(completed) < 30 else None
        ),
    }


class ComparisonRepository:
    def __init__(self, database_url: str) -> None:
        self.database_url = str(database_url or "").strip()
        self.enabled = bool(self.database_url and psycopg is not None)

    def report(self, *, days: int = 7) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "comparison_database_unavailable"}
        start = datetime.now(timezone.utc) - timedelta(days=max(1, min(days, 90)))
        try:
            with psycopg.connect(
                self.database_url, row_factory=dict_row, connect_timeout=8
            ) as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    select c.id, c.generated_at, c.symbol, c.direction,
                           c.setup_type, c.production_scores,
                           c.decision->'legacy_decision' as legacy_decision,
                           c.decision->'strict_shadow_decision' as strict_shadow_decision,
                           case when o.candidate_id is null then null else
                             jsonb_build_object(
                               'valid_fill', o.valid_fill,
                               'tp1_hit', o.tp1_hit,
                               'tp2_hit', o.tp2_hit,
                               'terminal_status', o.terminal_status,
                               'realized_r', o.realized_r,
                               'mfe_r', o.mfe_r,
                               'mae_r', o.mae_r
                             ) end as outcome
                    from public.signal_candidates c
                    left join public.signal_outcomes o on o.candidate_id=c.id
                    where c.generated_at >= %s
                      and c.id like 'legacy_cand_%%'
                      and c.decision->>'bot_variant' = 'legacy'
                    order by c.generated_at
                    """,
                    (start,),
                )
                rows = [dict(row) for row in cursor.fetchall()]
            return {"ok": True, "days": days, **build_comparison_report(rows)}
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": "comparison_query_failed",
                "error_type": type(exc).__name__,
            }
