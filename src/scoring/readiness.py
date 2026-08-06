"""Read-only shadow-model promotion readiness assessment.

This module never saves or promotes a model.  It runs the same chronological
walk-forward and calibration gates used by the explicit promotion workflow and
reports whether the available forward/replay outcomes are sufficient.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from src.scoring.training import train_outcome_model
from src.utils.config import AppConfig


def assess_shadow_readiness(
    rows: Sequence[Mapping[str, Any]],
    config: AppConfig,
) -> Dict[str, Any]:
    """Evaluate replacement readiness without changing production authority."""
    scoring = config.outcome_scoring
    schema = str(scoring.feature_schema_version)
    usable = [
        row
        for row in rows
        if str(row.get("feature_schema_version") or "") == schema
        and row.get("is_directional_candidate") is not False
        and str(row.get("direction") or "").lower() in ("long", "short")
        and str(row.get("terminal_status") or "") != "ambiguous_gap"
        and str(row.get("ambiguity_policy") or "") != "sparse_gap_unknown"
        and bool(row.get("features"))
        and row.get("generated_at") is not None
    ]
    minimum_training = int(scoring.minimum_training_samples)
    minimum_calibration = int(scoring.minimum_calibration_samples)
    minimum_total = minimum_training + minimum_calibration
    base = {
        "ready_to_replace_current": False,
        "automatic_promotion": False,
        "feature_schema_version": schema,
        "rows_loaded": len(rows),
        "usable_labeled_rows": len(usable),
        "minimum_training_samples": minimum_training,
        "minimum_calibration_samples": minimum_calibration,
        "minimum_total_samples": minimum_total,
        "minimum_unseen_samples": int(scoring.promotion_minimum_unseen_samples),
        "maximum_ece": float(scoring.promotion_max_ece),
    }
    if len(usable) < minimum_total:
        shortfall = minimum_total - len(usable)
        return {
            **base,
            "status": "collecting_outcomes",
            "sample_shortfall": shortfall,
            "reasons": [
                f"Need {shortfall} more compatible labeled outcomes before training"
            ],
            "promotion_checks": {},
            "recommendation": "Remain in shadow mode",
        }
    try:
        result = train_outcome_model(
            usable,
            minimum_training_samples=minimum_training,
            minimum_calibration_samples=minimum_calibration,
            conservative_quantile=float(scoring.conservative_quantile),
            promotion_max_ece=float(scoring.promotion_max_ece),
            promotion_minimum_unseen_samples=int(
                scoring.promotion_minimum_unseen_samples
            ),
            feature_schema_version=schema,
        )
    except (TypeError, ValueError) as exc:
        return {
            **base,
            "status": "validation_unavailable",
            "sample_shortfall": 0,
            "reasons": [str(exc)],
            "promotion_checks": {},
            "recommendation": "Remain in shadow mode",
        }

    promotion = dict(result.validation.get("promotion_gate") or {})
    checks = dict(promotion.get("checks") or {})
    calibrated = bool(result.artifact.calibration_ready)
    ready = bool(calibrated and promotion.get("passed"))
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        **base,
        "ready_to_replace_current": ready,
        "status": "ready_for_explicit_approval" if ready else "remain_in_shadow",
        "sample_shortfall": 0,
        "model_version": result.artifact.version,
        "training_samples": int(result.artifact.training_samples),
        "calibration_samples": int(result.artifact.calibration_samples),
        "calibration_ready": calibrated,
        "unseen_samples": int(result.validation.get("unseen_samples") or 0),
        "walk_forward_folds": int(result.validation.get("fold_count") or 0),
        "promotion_checks": checks,
        "failed_checks": failed_checks,
        "metrics": dict(result.metrics or {}),
        "reasons": (
            []
            if ready
            else [
                "Calibration is not ready"
                if not calibrated
                else "One or more walk-forward promotion checks failed"
            ]
        ),
        "recommendation": (
            "Eligible for explicit champion review; do not auto-promote"
            if ready
            else "Remain in shadow mode"
        ),
    }
