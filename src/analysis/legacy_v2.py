"""Deterministic interim execution-aware scoring and outcome comparison.

Legacy V2 is deliberately not called probability-calibrated. It preserves the
technical score, then prevents overall confidence from materially exceeding the
quality of the published execution plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from src.utils.helpers import clamp


@dataclass(frozen=True)
class LegacyConfidenceComparison:
    technical_confidence: float
    execution_score: float
    legacy_confidence: float
    legacy_v2_confidence: float
    execution_confidence_cap: float
    execution_confidence_penalty: float
    immediate_sl_penalty: float
    data_quality_penalty: float

    def to_dict(self) -> Dict[str, float]:
        return {
            key: round(float(value), 4)
            for key, value in asdict(self).items()
        }


def execution_aware_legacy_confidence(
    *,
    technical_confidence: float,
    execution_score: float,
    immediate_sl_risk: float,
    data_quality_score: float,
    confidence_min: float,
    confidence_max: float,
    execution_confidence_buffer: float,
) -> LegacyConfidenceComparison:
    """Return exact legacy and execution-capped Legacy V2 confidence.

    Legacy formula:
        T + clip((E - 70) * 0.12, -10, +3) - SL_penalty - quality_penalty

    Legacy V2:
        min(T, E + buffer) - SL_penalty - quality_penalty

    Technical confidence ``T`` is never changed. The V2 confidence falls
    one-for-one once execution becomes the bottleneck and never receives an
    execution-derived bonus above the technical score.
    """
    technical = float(clamp(technical_confidence, 0.0, 100.0))
    execution = float(clamp(execution_score, 0.0, 100.0))
    immediate_sl_penalty = max(0.0, float(immediate_sl_risk) - 28.0) * 0.20
    data_quality_penalty = max(0.0, 75.0 - float(data_quality_score)) * 0.20
    legacy_adjustment = float(
        clamp((execution - 70.0) * 0.12, -10.0, 3.0)
    )
    legacy = float(
        clamp(
            technical
            + legacy_adjustment
            - immediate_sl_penalty
            - data_quality_penalty,
            confidence_min,
            confidence_max,
        )
    )
    execution_cap = float(
        clamp(
            execution + max(0.0, float(execution_confidence_buffer)),
            confidence_min,
            confidence_max,
        )
    )
    legacy_v2 = float(
        clamp(
            min(technical, execution_cap)
            - immediate_sl_penalty
            - data_quality_penalty,
            confidence_min,
            confidence_max,
        )
    )
    return LegacyConfidenceComparison(
        technical_confidence=technical,
        execution_score=execution,
        legacy_confidence=legacy,
        legacy_v2_confidence=legacy_v2,
        execution_confidence_cap=execution_cap,
        execution_confidence_penalty=max(0.0, legacy - legacy_v2),
        immediate_sl_penalty=immediate_sl_penalty,
        data_quality_penalty=data_quality_penalty,
    )


def compare_legacy_v2_outcomes(
    rows: Sequence[Mapping[str, Any]],
    *,
    alert_confidence_floor: float = 80.0,
) -> Dict[str, Any]:
    """Compare stored Legacy and Legacy V2 selections on identical outcomes."""
    ordered = sorted(rows, key=lambda row: str(row.get("generated_at") or ""))
    legacy_selected = [
        row
        for row in ordered
        if _selected(row, "legacy", alert_confidence_floor)
    ]
    v2_selected = [
        row
        for row in ordered
        if _selected(row, "legacy_v2", alert_confidence_floor)
    ]
    legacy_ids = {_row_id(row) for row in legacy_selected}
    v2_ids = {_row_id(row) for row in v2_selected}
    newly_rejected = [
        row for row in legacy_selected if _row_id(row) not in v2_ids
    ]
    return {
        "definition": (
            "Identical labeled candidates; alert confidence floor applied to "
            "stored pre-filter Legacy and Legacy V2 eligibility"
        ),
        "alert_confidence_floor": float(alert_confidence_floor),
        "labeled_candidates": len(ordered),
        "legacy": _outcome_metrics(legacy_selected),
        "legacy_v2": _outcome_metrics(v2_selected),
        "delta": {
            "selected_count": len(v2_selected) - len(legacy_selected),
            "fill_rate_pct": round(
                _rate(v2_selected, "valid_fill")
                - _rate(legacy_selected, "valid_fill"),
                3,
            ),
            "tp1_rate_pct": round(
                _rate(v2_selected, "alert_success")
                - _rate(legacy_selected, "alert_success"),
                3,
            ),
            "expectancy_r": round(
                _mean_r(v2_selected) - _mean_r(legacy_selected),
                4,
            ),
        },
        "v2_newly_rejected": {
            "count": len(newly_rejected),
            "avoided_failures": sum(
                1 for row in newly_rejected if not bool(row.get("alert_success"))
            ),
            "rejected_successes": sum(
                1 for row in newly_rejected if bool(row.get("alert_success"))
            ),
        },
        "same_selection_count": len(legacy_ids & v2_ids),
    }


def _selected(
    row: Mapping[str, Any],
    policy: str,
    floor: float,
) -> bool:
    scores = dict(row.get("production_scores") or {})
    confidence_key = (
        "legacy_confidence"
        if policy == "legacy"
        else "legacy_v2_confidence"
    )
    eligible_key = (
        "legacy_signal_eligible"
        if policy == "legacy"
        else "legacy_v2_signal_eligible"
    )
    return bool(
        scores.get(eligible_key, False)
        and _number(scores.get(confidence_key)) >= float(floor)
    )


def _outcome_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    realized = [_number(row.get("realized_r")) for row in rows]
    gains = sum(value for value in realized if value > 0)
    losses = abs(sum(value for value in realized if value < 0))
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in realized:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "selected_count": len(rows),
        "fill_rate_pct": round(_rate(rows, "valid_fill"), 3),
        "tp1_rate_pct": round(_rate(rows, "alert_success"), 3),
        "tp2_rate_pct": round(_rate(rows, "tp2_hit"), 3),
        "preentry_failure_rate_pct": round(
            100.0
            * sum(
                1
                for row in rows
                if bool(row.get("invalidated_before_fill"))
                or bool(row.get("missed_before_fill"))
                or bool(row.get("expired_before_fill"))
            )
            / max(1, len(rows)),
            3,
        ),
        "expectancy_r": round(_mean_r(rows), 4),
        "profit_factor": round(
            gains / losses if losses > 1e-12 else (999.0 if gains else 0.0),
            4,
        ),
        "maximum_drawdown_r": round(max_drawdown, 4),
    }


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return 100.0 * sum(1 for row in rows if bool(row.get(key))) / len(rows)


def _mean_r(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(_number(row.get("realized_r")) for row in rows) / len(rows)


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("candidate_id") or id(row))


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
