"""Portable JSON model artifact and fully explainable inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.scoring.features import vector_from_features


AUXILIARY_FEATURES: Tuple[str, ...] = (
    "aux_technical_logit",
    "aux_fill_logit",
    "aux_conditional_logit",
)
OOD_WARNING_Z = 4.0
OOD_HARD_STOP_Z = 8.0
OOD_MAX_FEATURE_FRACTION = 0.10


@dataclass
class SigmoidCalibration:
    slope: float = 1.0
    intercept: float = 0.0

    def apply(self, probability: float) -> float:
        logit = _logit(probability)
        return _sigmoid(self.slope * logit + self.intercept)

    def to_dict(self) -> Dict[str, float]:
        return {"slope": float(self.slope), "intercept": float(self.intercept)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SigmoidCalibration":
        slope = data.get("slope")
        intercept = data.get("intercept")
        return cls(
            slope=float(1.0 if slope is None else slope),
            intercept=float(0.0 if intercept is None else intercept),
        )


@dataclass
class LinearProbabilityHead:
    name: str
    feature_names: List[str]
    mean: List[float]
    scale: List[float]
    coefficients: List[float]
    intercept: float
    regularization_c: float = 0.0
    calibration_samples: int = 0
    calibrated: bool = False
    calibration: SigmoidCalibration = field(default_factory=SigmoidCalibration)

    def raw_probability(self, values: Mapping[str, Any]) -> float:
        vector = np.asarray(
            vector_from_features(values, self.feature_names), dtype=float
        )
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        coefs = np.asarray(self.coefficients, dtype=float)
        normalized = (vector - mean) / np.where(np.abs(scale) > 1e-12, scale, 1.0)
        return _sigmoid(float(self.intercept + np.dot(normalized, coefs)))

    def probability(self, values: Mapping[str, Any]) -> float:
        return self.calibration.apply(self.raw_probability(values))

    def contributions(self, values: Mapping[str, Any]) -> Dict[str, float]:
        vector = np.asarray(
            vector_from_features(values, self.feature_names), dtype=float
        )
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        coefs = np.asarray(self.coefficients, dtype=float)
        normalized = (vector - mean) / np.where(np.abs(scale) > 1e-12, scale, 1.0)
        return {
            name: float(value)
            for name, value in zip(self.feature_names, normalized * coefs)
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_names": list(self.feature_names),
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
            "coefficients": [float(value) for value in self.coefficients],
            "intercept": float(self.intercept),
            "regularization_c": float(self.regularization_c),
            "calibration_samples": int(self.calibration_samples),
            "calibrated": bool(self.calibrated),
            "calibration": self.calibration.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LinearProbabilityHead":
        return cls(
            name=str(data.get("name") or ""),
            feature_names=[str(value) for value in data.get("feature_names") or []],
            mean=[float(value) for value in data.get("mean") or []],
            scale=[float(value) for value in data.get("scale") or []],
            coefficients=[
                float(value) for value in data.get("coefficients") or []
            ],
            intercept=float(data.get("intercept") or 0.0),
            regularization_c=float(data.get("regularization_c") or 0.0),
            calibration_samples=int(data.get("calibration_samples") or 0),
            calibrated=bool(data.get("calibrated", False)),
            calibration=SigmoidCalibration.from_dict(
                dict(data.get("calibration") or {})
            ),
        )


@dataclass
class LinearValueHead:
    feature_names: List[str]
    mean: List[float]
    scale: List[float]
    coefficients: List[float]
    intercept: float
    uncertainty_penalty_r: float
    reference_ev: List[float]

    def expected_r(self, values: Mapping[str, Any]) -> float:
        vector = np.asarray(
            vector_from_features(values, self.feature_names), dtype=float
        )
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        coefs = np.asarray(self.coefficients, dtype=float)
        normalized = (vector - mean) / np.where(np.abs(scale) > 1e-12, scale, 1.0)
        return float(self.intercept + np.dot(normalized, coefs))

    def contributions(self, values: Mapping[str, Any]) -> Dict[str, float]:
        vector = np.asarray(
            vector_from_features(values, self.feature_names), dtype=float
        )
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        coefs = np.asarray(self.coefficients, dtype=float)
        normalized = (vector - mean) / np.where(np.abs(scale) > 1e-12, scale, 1.0)
        return {
            name: float(value)
            for name, value in zip(self.feature_names, normalized * coefs)
        }

    def rank_percentile(self, conservative_ev: float) -> float:
        reference = np.asarray(self.reference_ev, dtype=float)
        if reference.size == 0:
            return 0.0
        return float(
            np.searchsorted(np.sort(reference), conservative_ev, side="right")
            / reference.size
            * 100.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
            "coefficients": [float(value) for value in self.coefficients],
            "intercept": float(self.intercept),
            "uncertainty_penalty_r": float(self.uncertainty_penalty_r),
            "reference_ev": [float(value) for value in self.reference_ev],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LinearValueHead":
        return cls(
            feature_names=[str(value) for value in data.get("feature_names") or []],
            mean=[float(value) for value in data.get("mean") or []],
            scale=[float(value) for value in data.get("scale") or []],
            coefficients=[
                float(value) for value in data.get("coefficients") or []
            ],
            intercept=float(data.get("intercept") or 0.0),
            uncertainty_penalty_r=float(
                data.get("uncertainty_penalty_r") or 0.0
            ),
            reference_ev=[float(value) for value in data.get("reference_ev") or []],
        )


@dataclass
class OutcomeModelArtifact:
    version: str
    feature_schema_version: str
    technical: LinearProbabilityHead
    fill: LinearProbabilityHead
    conditional_win: LinearProbabilityHead
    alert_success: LinearProbabilityHead
    expected_value: LinearValueHead
    training_samples: int
    calibration_samples: int
    calibration_ready: bool
    metrics: Dict[str, Any] = field(default_factory=dict)

    def score(self, features: Mapping[str, Any]) -> Dict[str, Any]:
        p_technical = self.technical.probability(features)
        p_fill = self.fill.probability(features)
        p_conditional = self.conditional_win.probability(features)
        augmented = dict(features)
        augmented.update(
            {
                "aux_technical_logit": _logit(p_technical),
                "aux_fill_logit": _logit(p_fill),
                "aux_conditional_logit": _logit(p_conditional),
            }
        )
        p_alert = self.alert_success.probability(augmented)
        product_diagnostic = p_fill * p_conditional
        expected_r = self.expected_value.expected_r(augmented)
        conservative_ev = expected_r - self.expected_value.uncertainty_penalty_r
        rank = self.expected_value.rank_percentile(conservative_ev)

        technical_contrib = self.technical.contributions(features)
        execution_contrib = self.fill.contributions(features)
        confidence_contrib = self.alert_success.contributions(augmented)
        ev_contrib = self.expected_value.contributions(augmented)
        distribution = _distribution_diagnostics(
            self.alert_success,
            augmented,
        )
        positives, negatives = _top_factors(
            _merge_contributions(
                technical_contrib,
                execution_contrib,
                confidence_contrib,
                ev_contrib,
            )
        )
        return {
            "model_version": self.version,
            "feature_schema_version": self.feature_schema_version,
            "calibration_ready": bool(self.calibration_ready),
            "technical_score": round(p_technical * 100.0, 2),
            "execution_score": round(p_fill * 100.0, 2),
            "conditional_tp1_probability": round(p_conditional * 100.0, 2),
            "confidence": round(p_alert * 100.0, 2),
            "execution_times_conditional_diagnostic": round(
                product_diagnostic * 100.0, 2
            ),
            # Backward-compatible diagnostic key retained for stored consumers.
            "fill_times_conditional_diagnostic": round(
                product_diagnostic * 100.0, 2
            ),
            "expected_value_r": round(expected_r, 4),
            "uncertainty_penalty_r": round(
                self.expected_value.uncertainty_penalty_r, 4
            ),
            "conservative_ev_r": round(conservative_ev, 4),
            "rank_score": round(rank, 2),
            "model_applicable": bool(distribution["applicable"]),
            "distribution_diagnostics": distribution,
            "technical_breakdown": _ordered_contributions(technical_contrib),
            "execution_breakdown": _ordered_contributions(execution_contrib),
            "confidence_breakdown": _ordered_contributions(confidence_contrib),
            "rank_breakdown": _ordered_contributions(ev_contrib),
            "calculation": {
                "technical": _probability_calculation(
                    self.technical,
                    "P(favorable 1 ATR before adverse 1 ATR)",
                ),
                "execution": _probability_calculation(
                    self.fill,
                    "P(valid fill and survives immediate stop-sweep window)",
                ),
                "confidence": _probability_calculation(
                    self.alert_success,
                    "P(valid executable entry and TP1 before Stop Loss)",
                ),
                "rank": {
                    "definition": (
                        "percentile of conservative expected R against held-out "
                        "reference outcomes"
                    ),
                    "ev_intercept": round(self.expected_value.intercept, 6),
                    "uncertainty_penalty_r": round(
                        self.expected_value.uncertainty_penalty_r, 6
                    ),
                    "reference_sample_count": len(
                        self.expected_value.reference_ev
                    ),
                },
            },
            "top_positive_factors": positives,
            "top_negative_factors": negatives,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "feature_schema_version": self.feature_schema_version,
            "technical": self.technical.to_dict(),
            "fill": self.fill.to_dict(),
            "conditional_win": self.conditional_win.to_dict(),
            "alert_success": self.alert_success.to_dict(),
            "expected_value": self.expected_value.to_dict(),
            "training_samples": int(self.training_samples),
            "calibration_samples": int(self.calibration_samples),
            "calibration_ready": bool(self.calibration_ready),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutcomeModelArtifact":
        return cls(
            version=str(data.get("version") or ""),
            feature_schema_version=str(
                data.get("feature_schema_version") or ""
            ),
            technical=LinearProbabilityHead.from_dict(
                dict(data.get("technical") or {})
            ),
            fill=LinearProbabilityHead.from_dict(dict(data.get("fill") or {})),
            conditional_win=LinearProbabilityHead.from_dict(
                dict(data.get("conditional_win") or {})
            ),
            alert_success=LinearProbabilityHead.from_dict(
                dict(data.get("alert_success") or {})
            ),
            expected_value=LinearValueHead.from_dict(
                dict(data.get("expected_value") or {})
            ),
            training_samples=int(data.get("training_samples") or 0),
            calibration_samples=int(data.get("calibration_samples") or 0),
            calibration_ready=bool(data.get("calibration_ready", False)),
            metrics=dict(data.get("metrics") or {}),
        )


def _merge_contributions(*groups: Mapping[str, float]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for group in groups:
        for name, value in group.items():
            if name.startswith("setup__") or "_x_setup__" in name:
                label = name
            elif name.startswith("aux_"):
                label = name
            else:
                label = name
            merged[label] = merged.get(label, 0.0) + float(value)
    return merged


def _probability_calculation(
    head: LinearProbabilityHead,
    definition: str,
) -> Dict[str, Any]:
    return {
        "definition": definition,
        "standardized_logit_intercept": round(head.intercept, 6),
        "selected_l2_regularization_c": round(head.regularization_c, 6),
        "calibrated": bool(head.calibrated),
        "calibration_samples": int(head.calibration_samples),
        "calibration": {
            "method": "sigmoid",
            "slope": round(head.calibration.slope, 6),
            "intercept": round(head.calibration.intercept, 6),
        },
    }


def _distribution_diagnostics(
    head: LinearProbabilityHead,
    values: Mapping[str, Any],
) -> Dict[str, Any]:
    vector = np.asarray(
        vector_from_features(values, head.feature_names), dtype=float
    )
    mean = np.asarray(head.mean, dtype=float)
    scale = np.asarray(head.scale, dtype=float)
    z = np.abs(
        (vector - mean) / np.where(np.abs(scale) > 1e-12, scale, 1.0)
    )
    warning_mask = z > OOD_WARNING_Z
    fraction = float(np.mean(warning_mask)) if z.size else 1.0
    max_z = float(np.max(z)) if z.size else float("inf")
    applicable = bool(
        z.size
        and max_z <= OOD_HARD_STOP_Z
        and fraction <= OOD_MAX_FEATURE_FRACTION
    )
    extremes = sorted(
        (
            (name, float(value))
            for name, value in zip(head.feature_names, z)
            if value > OOD_WARNING_Z
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    return {
        "applicable": applicable,
        "warning_z": OOD_WARNING_Z,
        "hard_stop_z": OOD_HARD_STOP_Z,
        "outlier_feature_fraction": round(fraction, 4),
        "max_abs_z": round(max_z, 4) if math.isfinite(max_z) else None,
        "extreme_features": [
            {"factor": name, "abs_z": round(value, 3)}
            for name, value in extremes
        ],
    }


def _top_factors(
    contributions: Mapping[str, float],
    count: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ordered = sorted(contributions.items(), key=lambda item: item[1])
    negative = [
        {"factor": name, "contribution": round(value, 4)}
        for name, value in ordered[:count]
        if value < 0
    ]
    positive = [
        {"factor": name, "contribution": round(value, 4)}
        for name, value in reversed(ordered[-count:])
        if value > 0
    ]
    return positive, negative


def _ordered_contributions(
    contributions: Mapping[str, float],
) -> List[Dict[str, Any]]:
    return [
        {"factor": name, "contribution": round(float(value), 5)}
        for name, value in sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        if abs(float(value)) > 1e-8
    ]


def _sigmoid(value: float) -> float:
    clipped = min(35.0, max(-35.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def _logit(probability: float) -> float:
    value = min(1.0 - 1e-6, max(1e-6, float(probability)))
    return math.log(value / (1.0 - value))
