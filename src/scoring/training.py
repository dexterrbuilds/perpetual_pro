"""Walk-forward training and validation for outcome-calibrated scoring."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from src.scoring.features import (
    FEATURE_SCHEMA_VERSION,
    all_model_feature_names,
    execution_model_feature_names,
    technical_model_feature_names,
    vector_from_features,
)
from src.scoring.model import (
    LinearProbabilityHead,
    LinearValueHead,
    OutcomeModelArtifact,
    SigmoidCalibration,
)
from src.utils.helpers import timeframe_to_minutes


LOGISTIC_C_CANDIDATES: Tuple[float, ...] = (0.03, 0.10, 0.30, 1.0)


@dataclass
class TrainingResult:
    artifact: OutcomeModelArtifact
    metrics: Dict[str, Any]
    validation: Dict[str, Any]
    folds: List[Dict[str, Any]]
    training_start: str
    training_end: str


def train_outcome_model(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_training_samples: int = 500,
    minimum_calibration_samples: int = 200,
    conservative_quantile: float = 0.10,
    promotion_max_ece: float = 0.05,
    promotion_minimum_unseen_samples: int = 200,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> TrainingResult:
    """Train a shadow model and evaluate only on later unseen observations."""
    prepared = _prepare_rows(rows, feature_schema_version)
    if len(prepared) < minimum_training_samples + minimum_calibration_samples:
        raise ValueError(
            "Insufficient labeled candidates: "
            f"{len(prepared)} available; need at least "
            f"{minimum_training_samples + minimum_calibration_samples}"
        )
    ordered = sorted(prepared, key=lambda item: item["generated_at"])
    folds = _walk_forward_folds(
        ordered,
        minimum_training_samples=minimum_training_samples,
        minimum_test_samples=max(50, minimum_calibration_samples // 2),
    )
    fold_results: List[Dict[str, Any]] = []
    all_new_prob: List[float] = []
    all_old_prob: List[float] = []
    all_outcomes: List[int] = []
    all_new_rank: List[float] = []
    all_old_rank: List[float] = []
    all_realized_r: List[float] = []
    all_valid_fill: List[int] = []
    all_tp2: List[int] = []
    all_invalidated: List[int] = []
    all_preentry_failure: List[int] = []
    all_duration: List[float] = []

    for fold_number, (train_rows, test_rows) in enumerate(folds, 1):
        artifact = _fit_artifact(
            train_rows,
            minimum_calibration_samples=min(
                minimum_calibration_samples,
                max(50, len(train_rows) // 5),
            ),
            conservative_quantile=conservative_quantile,
            version=f"fold-{fold_number}",
            feature_schema_version=feature_schema_version,
        )
        scored = [artifact.score(row["features"]) for row in test_rows]
        new_prob = [float(item["confidence"]) / 100.0 for item in scored]
        new_rank = [float(item["conservative_ev_r"]) for item in scored]
        old_prob = [
            _probability(
                (row.get("production_scores") or {}).get("confidence")
            )
            for row in test_rows
        ]
        old_rank = [
            _number((row.get("production_scores") or {}).get("rank"))
            for row in test_rows
        ]
        outcomes = [int(row["alert_success"]) for row in test_rows]
        realized = [_number(row["realized_r"]) for row in test_rows]
        valid_fill = [int(row["valid_fill"]) for row in test_rows]
        tp2 = [int(row["tp2_hit"]) for row in test_rows]
        invalidated = [
            int(row["invalidated_before_fill"]) for row in test_rows
        ]
        preentry_failure = [
            int(row["preentry_failure"]) for row in test_rows
        ]
        duration = [row["trade_duration_minutes"] for row in test_rows]
        fold_metrics = _comparison_metrics(
            outcomes=outcomes,
            realized_r=realized,
            valid_fill=valid_fill,
            tp2_hit=tp2,
            invalidated_before_fill=invalidated,
            preentry_failure=preentry_failure,
            trade_duration_minutes=duration,
            new_probability=new_prob,
            old_probability=old_prob,
            new_rank=new_rank,
            old_rank=old_rank,
        )
        fold_results.append(
            {
                "fold": fold_number,
                "train_start": train_rows[0]["generated_at"].isoformat(),
                "train_end": train_rows[-1]["generated_at"].isoformat(),
                "test_start": test_rows[0]["generated_at"].isoformat(),
                "test_end": test_rows[-1]["generated_at"].isoformat(),
                "sample_count": len(test_rows),
                "metrics": fold_metrics,
                "regime_metrics": _slice_metrics(test_rows, scored),
            }
        )
        all_new_prob.extend(new_prob)
        all_old_prob.extend(old_prob)
        all_outcomes.extend(outcomes)
        all_new_rank.extend(new_rank)
        all_old_rank.extend(old_rank)
        all_realized_r.extend(realized)
        all_valid_fill.extend(valid_fill)
        all_tp2.extend(tp2)
        all_invalidated.extend(invalidated)
        all_preentry_failure.extend(preentry_failure)
        all_duration.extend(duration)

    validation_metrics = _comparison_metrics(
        outcomes=all_outcomes,
        realized_r=all_realized_r,
        valid_fill=all_valid_fill,
        tp2_hit=all_tp2,
        invalidated_before_fill=all_invalidated,
        preentry_failure=all_preentry_failure,
        trade_duration_minutes=all_duration,
        new_probability=all_new_prob,
        old_probability=all_old_prob,
        new_rank=all_new_rank,
        old_rank=all_old_rank,
    )
    validation = {
        "method": "expanding_walk_forward",
        "fold_count": len(fold_results),
        "unseen_samples": len(all_outcomes),
        "metrics": validation_metrics,
        "promotion_gate": _promotion_gate(
            validation_metrics,
            fold_results,
            max_ece=promotion_max_ece,
            minimum_unseen_samples=promotion_minimum_unseen_samples,
        ),
    }
    version_seed = "|".join(
        [
            ordered[0]["generated_at"].isoformat(),
            ordered[-1]["generated_at"].isoformat(),
            str(len(ordered)),
            feature_schema_version,
        ]
    )
    version = (
        "outcome-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + hashlib.sha256(version_seed.encode("utf-8")).hexdigest()[:8]
    )
    final_artifact = _fit_artifact(
        ordered,
        minimum_calibration_samples=minimum_calibration_samples,
        conservative_quantile=conservative_quantile,
        version=version,
        feature_schema_version=feature_schema_version,
    )
    final_artifact.metrics = validation_metrics
    return TrainingResult(
        artifact=final_artifact,
        metrics=validation_metrics,
        validation=validation,
        folds=fold_results,
        training_start=ordered[0]["generated_at"].isoformat(),
        training_end=ordered[-1]["generated_at"].isoformat(),
    )


def _fit_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_calibration_samples: int,
    conservative_quantile: float,
    version: str,
    feature_schema_version: str,
) -> OutcomeModelArtifact:
    # Reserve the requested number of chronologically latest observations for
    # calibration whenever the dataset can still leave a viable fit window.
    # The prior quarter-split could reserve only 175 of 700 rows while requiring
    # 200, making a sufficiently large model permanently "not calibrated".
    calibration_count = min(
        max(50, minimum_calibration_samples),
        max(50, len(rows) - 100),
    )
    fit_rows = list(rows[:-calibration_count])
    calibration_rows = list(rows[-calibration_count:])
    if len(fit_rows) < 100:
        raise ValueError("Training window is too small after calibration split")

    technical_rows = [
        row for row in fit_rows if row.get("technical_success") is not None
    ]
    technical_cal = [
        row
        for row in calibration_rows
        if row.get("technical_success") is not None
    ]
    conditional_rows = [row for row in fit_rows if row.get("valid_fill")]
    conditional_cal = [row for row in calibration_rows if row.get("valid_fill")]
    if len(technical_rows) < 50 or len(conditional_rows) < 50:
        raise ValueError(
            "Need at least 50 technical-path labels and 50 valid-fill outcomes"
        )

    technical = _fit_probability_head(
        "technical",
        technical_rows,
        technical_cal,
        feature_names=technical_model_feature_names(),
        target="technical_success",
    )
    fill = _fit_probability_head(
        "fill",
        fit_rows,
        calibration_rows,
        feature_names=execution_model_feature_names(),
        target="execution_success",
    )
    conditional = _fit_probability_head(
        "conditional_win",
        conditional_rows,
        conditional_cal,
        feature_names=all_model_feature_names(),
        target="tp1_hit",
    )
    # Direct alert model: raw signal-time features only. Auxiliary probabilities
    # remain diagnostics, avoiding in-sample stacking leakage.
    alert = _fit_probability_head(
        "alert_success",
        fit_rows,
        calibration_rows,
        feature_names=all_model_feature_names(),
        target="alert_success",
    )
    expected_value = _fit_value_head(
        fit_rows,
        calibration_rows,
        feature_names=all_model_feature_names(),
        target="realized_r",
        conservative_quantile=conservative_quantile,
    )
    calibration_ready = bool(
        len(calibration_rows) >= minimum_calibration_samples
        and _has_two_classes(
            [int(row["alert_success"]) for row in calibration_rows]
        )
        and all(
            head.calibrated
            for head in (technical, fill, conditional, alert)
        )
    )
    return OutcomeModelArtifact(
        version=version,
        feature_schema_version=feature_schema_version,
        technical=technical,
        fill=fill,
        conditional_win=conditional,
        alert_success=alert,
        expected_value=expected_value,
        training_samples=len(fit_rows),
        calibration_samples=len(calibration_rows),
        calibration_ready=calibration_ready,
    )


def _fit_probability_head(
    name: str,
    train_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Iterable[str],
    target: str,
) -> LinearProbabilityHead:
    names = list(feature_names)
    x_train = _matrix(train_rows, names)
    y_train = np.asarray([int(bool(row[target])) for row in train_rows], dtype=int)
    if not _has_two_classes(y_train):
        raise ValueError(f"{name} target has only one class")
    selected_c = _select_logistic_regularization(x_train, y_train)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_train)
    estimator = LogisticRegression(
        penalty="l2",
        C=selected_c,
        solver="lbfgs",
        max_iter=2000,
        class_weight=None,
    )
    estimator.fit(scaled, y_train)
    calibration = SigmoidCalibration()
    calibrated = False
    calibration_sample_count = 0
    if calibration_rows:
        x_cal = _matrix(calibration_rows, names)
        y_cal = np.asarray(
            [int(bool(row[target])) for row in calibration_rows], dtype=int
        )
        if len(y_cal) >= 30 and _has_two_classes(y_cal):
            raw = estimator.predict_proba(scaler.transform(x_cal))[:, 1]
            calibration = _fit_sigmoid_calibration(raw, y_cal)
            calibrated = True
            calibration_sample_count = len(y_cal)
    return LinearProbabilityHead(
        name=name,
        feature_names=names,
        mean=scaler.mean_.astype(float).tolist(),
        scale=scaler.scale_.astype(float).tolist(),
        coefficients=estimator.coef_[0].astype(float).tolist(),
        intercept=float(estimator.intercept_[0]),
        regularization_c=selected_c,
        calibration_samples=calibration_sample_count,
        calibrated=calibrated,
        calibration=calibration,
    )


def _select_logistic_regularization(
    values: np.ndarray,
    target: np.ndarray,
) -> float:
    """Select L2 strength on the chronologically latest internal validation."""
    if len(target) < 150:
        return 0.10
    validation_count = max(30, len(target) // 5)
    split = len(target) - validation_count
    train_y = target[:split]
    validation_y = target[split:]
    if not _has_two_classes(train_y) or not _has_two_classes(validation_y):
        return 0.10
    best_c = LOGISTIC_C_CANDIDATES[0]
    best_loss = float("inf")
    for candidate_c in LOGISTIC_C_CANDIDATES:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(values[:split])
        validation_x = scaler.transform(values[split:])
        estimator = LogisticRegression(
            penalty="l2",
            C=candidate_c,
            solver="lbfgs",
            max_iter=2000,
            class_weight=None,
        )
        estimator.fit(train_x, train_y)
        probability = estimator.predict_proba(validation_x)[:, 1]
        loss = float(log_loss(validation_y, probability, labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_c = candidate_c
    return float(best_c)


def _fit_value_head(
    train_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Iterable[str],
    target: str,
    conservative_quantile: float,
) -> LinearValueHead:
    names = list(feature_names)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(_matrix(train_rows, names))
    y_train = np.asarray([_number(row[target]) for row in train_rows], dtype=float)
    estimator = HuberRegressor(
        epsilon=1.35,
        alpha=0.08,
        max_iter=3000,
        tol=1e-4,
    )
    estimator.fit(x_train, y_train)
    reference_rows = list(calibration_rows) or list(train_rows)
    x_reference = scaler.transform(_matrix(reference_rows, names))
    predictions = estimator.predict(x_reference)
    actual = np.asarray(
        [_number(row[target]) for row in reference_rows], dtype=float
    )
    overprediction_error = predictions - actual
    q = min(0.25, max(0.01, float(conservative_quantile)))
    penalty = max(0.0, float(np.quantile(overprediction_error, 1.0 - q)))
    conservative_reference = (predictions - penalty).astype(float).tolist()
    return LinearValueHead(
        feature_names=names,
        mean=scaler.mean_.astype(float).tolist(),
        scale=scaler.scale_.astype(float).tolist(),
        coefficients=estimator.coef_.astype(float).tolist(),
        intercept=float(estimator.intercept_),
        uncertainty_penalty_r=penalty,
        reference_ev=conservative_reference,
    )


def _fit_sigmoid_calibration(
    raw_probability: np.ndarray,
    target: np.ndarray,
) -> SigmoidCalibration:
    logits = np.asarray([_logit(value) for value in raw_probability]).reshape(-1, 1)
    calibrator = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    calibrator.fit(logits, target)
    return SigmoidCalibration(
        slope=float(calibrator.coef_[0][0]),
        intercept=float(calibrator.intercept_[0]),
    )


def _prepare_rows(
    rows: Sequence[Mapping[str, Any]],
    feature_schema_version: str,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for source in rows:
        if str(source.get("feature_schema_version") or "") != feature_schema_version:
            continue
        # Unknown event ordering is not a negative label and must not train the
        # execution or alert models in either direction.
        if str(source.get("terminal_status") or "") == "ambiguous_gap":
            continue
        generated = _parse_datetime(source.get("generated_at"))
        features = dict(source.get("features") or {})
        if generated is None or not features:
            continue
        prepared.append(
            {
                **dict(source),
                "generated_at": generated,
                "features": features,
                "valid_fill": bool(source.get("valid_fill")),
                "execution_success": _execution_success(source),
                "technical_success": (
                    None
                    if source.get("technical_success") is None
                    else bool(source.get("technical_success"))
                ),
                "alert_success": bool(source.get("alert_success")),
                "tp1_hit": bool(source.get("tp1_hit")),
                "tp2_hit": bool(source.get("tp2_hit")),
                "invalidated_before_fill": bool(
                    source.get("invalidated_before_fill")
                ),
                "preentry_failure": bool(
                    source.get("invalidated_before_fill")
                    or source.get("missed_before_fill")
                    or source.get("expired_before_fill")
                ),
                "entry_delay_minutes": _optional_number(
                    source.get("entry_delay_minutes")
                ),
                "trade_duration_minutes": _optional_number(
                    source.get("trade_duration_minutes")
                ),
                "realized_r": _number(source.get("realized_r")),
                "production_scores": dict(
                    source.get("production_scores") or {}
                ),
            }
        )
    return prepared


def _walk_forward_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_training_samples: int,
    minimum_test_samples: int,
) -> List[Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]]:
    available = len(rows) - minimum_training_samples
    fold_count = min(5, max(2, available // minimum_test_samples))
    test_size = max(minimum_test_samples, available // fold_count)
    folds: List[Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]] = []
    cursor = minimum_training_samples
    while cursor < len(rows) and len(folds) < fold_count:
        end = min(len(rows), cursor + test_size)
        test = list(rows[cursor:end])
        if len(test) < minimum_test_samples and folds:
            break
        test_start = test[0]["generated_at"]
        # Embargo all candidates whose lifecycle can overlap the first test
        # observation. Twenty-seven hours covers max 3h entry + 24h hold.
        train = [
            row
            for row in rows[:cursor]
            if (test_start - row["generated_at"]).total_seconds() >= 27 * 3600
        ]
        if len(train) >= 100 and test:
            folds.append((train, test))
        cursor = end
    if not folds:
        raise ValueError("Unable to create non-overlapping walk-forward folds")
    return folds


def _comparison_metrics(
    *,
    outcomes: Sequence[int],
    realized_r: Sequence[float],
    valid_fill: Sequence[int],
    tp2_hit: Sequence[int],
    invalidated_before_fill: Sequence[int],
    preentry_failure: Sequence[int],
    trade_duration_minutes: Sequence[Optional[float]],
    new_probability: Sequence[float],
    old_probability: Sequence[float],
    new_rank: Sequence[float],
    old_rank: Sequence[float],
) -> Dict[str, Any]:
    y = np.asarray(outcomes, dtype=int)
    r = np.asarray(realized_r, dtype=float)
    fill = np.asarray(valid_fill, dtype=int)
    tp2 = np.asarray(tp2_hit, dtype=int)
    invalidated = np.asarray(invalidated_before_fill, dtype=int)
    preentry = np.asarray(preentry_failure, dtype=int)
    duration = np.asarray(
        [
            float(value) if value is not None and math.isfinite(float(value)) else np.nan
            for value in trade_duration_minutes
        ],
        dtype=float,
    )
    new_p = np.clip(np.asarray(new_probability, dtype=float), 1e-6, 1 - 1e-6)
    old_p = np.clip(np.asarray(old_probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "new": {
            "brier": round(float(brier_score_loss(y, new_p)), 6),
            "log_loss": round(float(log_loss(y, new_p, labels=[0, 1])), 6),
            "ece": round(_ece(y, new_p), 6),
            **_rank_metrics(
                r,
                np.asarray(new_rank, dtype=float),
                outcomes=y,
                valid_fill=fill,
                tp2_hit=tp2,
                invalidated_before_fill=invalidated,
                preentry_failure=preentry,
                trade_duration_minutes=duration,
            ),
        },
        "old": {
            "brier": round(float(brier_score_loss(y, old_p)), 6),
            "log_loss": round(float(log_loss(y, old_p, labels=[0, 1])), 6),
            "ece": round(_ece(y, old_p), 6),
            **_rank_metrics(
                r,
                np.asarray(old_rank, dtype=float),
                outcomes=y,
                valid_fill=fill,
                tp2_hit=tp2,
                invalidated_before_fill=invalidated,
                preentry_failure=preentry,
                trade_duration_minutes=duration,
            ),
        },
        "sample_count": int(len(y)),
        "base_success_rate": round(float(np.mean(y)) if len(y) else 0.0, 6),
        "overall_expectancy_r": round(float(np.mean(r)) if len(r) else 0.0, 6),
    }


def _rank_metrics(
    realized_r: np.ndarray,
    rank: np.ndarray,
    *,
    outcomes: np.ndarray,
    valid_fill: np.ndarray,
    tp2_hit: np.ndarray,
    invalidated_before_fill: np.ndarray,
    preentry_failure: np.ndarray,
    trade_duration_minutes: np.ndarray,
) -> Dict[str, Any]:
    if len(realized_r) == 0:
        return {
            "top_quintile_expectancy_r": 0.0,
            "top_quintile_profit_factor": 0.0,
            "top_quintile_max_drawdown_r": 0.0,
        }
    cutoff = float(np.quantile(rank, 0.80))
    mask = rank >= cutoff
    selected = realized_r[mask]
    gains = float(np.sum(selected[selected > 0]))
    losses = abs(float(np.sum(selected[selected < 0])))
    profit_factor = gains / losses if losses > 1e-12 else (999.0 if gains > 0 else 0.0)
    equity = np.cumsum(selected)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    drawdown = np.concatenate(([0.0], equity)) - peaks
    selected_duration = trade_duration_minutes[mask]
    finite_duration = selected_duration[np.isfinite(selected_duration)]
    return {
        "top_quintile_count": int(len(selected)),
        "top_quintile_win_rate": round(
            float(np.mean(selected > 0)) if len(selected) else 0.0, 6
        ),
        "top_quintile_tp1_rate": round(
            float(np.mean(outcomes[mask])) if len(selected) else 0.0, 6
        ),
        "top_quintile_tp2_rate": round(
            float(np.mean(tp2_hit[mask])) if len(selected) else 0.0, 6
        ),
        "top_quintile_fill_rate": round(
            float(np.mean(valid_fill[mask])) if len(selected) else 0.0, 6
        ),
        "top_quintile_invalidation_before_fill_rate": round(
            float(np.mean(invalidated_before_fill[mask]))
            if len(selected)
            else 0.0,
            6,
        ),
        "top_quintile_preentry_failure_rate": round(
            float(np.mean(preentry_failure[mask])) if len(selected) else 0.0,
            6,
        ),
        "top_quintile_average_duration_minutes": round(
            float(np.mean(finite_duration)) if len(finite_duration) else 0.0,
            4,
        ),
        "top_quintile_expectancy_r": round(float(np.mean(selected)), 6),
        "top_quintile_profit_factor": round(float(profit_factor), 6),
        "top_quintile_max_drawdown_r": round(
            abs(float(np.min(drawdown))), 6
        ),
    }


def _promotion_gate(
    metrics: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    *,
    max_ece: float,
    minimum_unseen_samples: int,
) -> Dict[str, Any]:
    new = dict(metrics.get("new") or {})
    old = dict(metrics.get("old") or {})
    checks = {
        "brier_improved": _number(new.get("brier")) < _number(old.get("brier")),
        "log_loss_not_worse": _number(new.get("log_loss"))
        <= _number(old.get("log_loss")),
        "calibration_not_worse": _number(new.get("ece"))
        <= _number(old.get("ece")),
        "absolute_calibration_quality": _number(new.get("ece"))
        <= float(max_ece),
        "minimum_unseen_sample": int(metrics.get("sample_count") or 0)
        >= int(minimum_unseen_samples),
        "top_ev_improved": _number(new.get("top_quintile_expectancy_r"))
        > _number(old.get("top_quintile_expectancy_r")),
        "profit_factor_not_worse": _number(
            new.get("top_quintile_profit_factor")
        )
        >= _number(old.get("top_quintile_profit_factor")),
        "all_folds_positive_top_ev": all(
            _number(((fold.get("metrics") or {}).get("new") or {}).get(
                "top_quintile_expectancy_r"
            ))
            > 0
            for fold in folds
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "automatic_promotion": False,
        "reason": (
            "Eligible for explicit champion approval"
            if all(checks.values())
            else "Remain in shadow mode"
        ),
    }


def _slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    scored: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    setup_types = sorted({str(row.get("setup_type") or "") for row in rows})
    for setup in setup_types:
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row.get("setup_type") or "") == setup
        ]
        if len(indices) < 20:
            continue
        output[f"setup:{setup}"] = {
            "sample_count": len(indices),
            "success_rate": round(
                float(np.mean([bool(rows[index]["alert_success"]) for index in indices])),
                6,
            ),
            "mean_confidence": round(
                float(np.mean([_number(scored[index]["confidence"]) for index in indices])),
                4,
            ),
            "expectancy_r": round(
                float(np.mean([_number(rows[index]["realized_r"]) for index in indices])),
                6,
            ),
        }
    return output


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [vector_from_features(row["features"], names) for row in rows],
        dtype=float,
    )


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(target))
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (
            (probability >= lower) & (probability < upper)
            if upper < 1.0
            else (probability >= lower) & (probability <= upper)
        )
        if not np.any(selected):
            continue
        value += (
            float(np.sum(selected))
            / total
            * abs(float(np.mean(target[selected])) - float(np.mean(probability[selected])))
        )
    return float(value)


def _has_two_classes(values: Iterable[Any]) -> bool:
    return len(set(int(value) for value in values)) >= 2


def _probability(value: Any) -> float:
    number = _number(value)
    if number > 1.0:
        number /= 100.0
    return min(1.0 - 1e-6, max(1e-6, number))


def _logit(probability: float) -> float:
    value = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(value / (1.0 - value))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _optional_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = _number(value, float("nan"))
    return number if math.isfinite(number) else None


def _execution_success(source: Mapping[str, Any]) -> bool:
    """A fill is executable only if it survives the immediate sweep window."""
    if not bool(source.get("valid_fill")):
        return False
    if str(source.get("terminal_status") or "") != "stopped":
        return True
    duration = _optional_number(source.get("trade_duration_minutes"))
    timeframe = str(source.get("timeframe") or "15m")
    try:
        confirmation_minutes = max(1, timeframe_to_minutes(timeframe))
    except (TypeError, ValueError):
        confirmation_minutes = 15
    return duration is None or duration > confirmation_minutes


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
