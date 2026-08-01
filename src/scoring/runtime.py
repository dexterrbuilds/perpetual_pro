"""Low-overhead runtime facade for journaling and shadow inference."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Iterable, Mapping, Optional

from loguru import logger

from src.scoring.model import OutcomeModelArtifact
from src.scoring.repository import OutcomeRepository
from src.utils.config import AppConfig, load_config


class OutcomeScoringRuntime:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.repository = OutcomeRepository(config.outcome_scoring.database_url)
        self._lock = threading.Lock()
        self._model: Optional[OutcomeModelArtifact] = None
        self._model_loaded_at = 0.0
        self._model_error: Optional[str] = None

    def score(self, candidate: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if (
            not self.config.outcome_scoring.enabled
            or self.config.outcome_scoring.mode == "off"
        ):
            return None
        model = self._load_model_if_needed()
        if model is None:
            return None
        if (
            model.feature_schema_version
            != str(candidate.get("feature_schema_version") or "")
        ):
            self._model_error = "feature_schema_mismatch"
            return None
        return model.score(dict(candidate.get("features") or {}))

    def journal(self, candidates: Iterable[Mapping[str, Any]]) -> int:
        if (
            not self.config.outcome_scoring.enabled
            or self.config.outcome_scoring.mode == "off"
        ):
            return 0
        return self.repository.record_candidates(candidates)

    def status(self) -> Dict[str, Any]:
        model = self._model
        return {
            "enabled": bool(self.config.outcome_scoring.enabled),
            "mode": self.config.outcome_scoring.mode,
            "feature_schema_version": self.config.outcome_scoring.feature_schema_version,
            "database": self.repository.status(),
            "model_loaded": model is not None,
            "model_version": model.version if model else None,
            "calibration_ready": bool(model.calibration_ready) if model else False,
            "training_samples": int(model.training_samples) if model else 0,
            "calibration_samples": int(model.calibration_samples) if model else 0,
            "last_model_error": self._model_error,
            "production_activation_allowed": bool(
                self.config.outcome_scoring.mode == "production"
                and model is not None
                and model.calibration_ready
            ),
        }

    def _load_model_if_needed(self) -> Optional[OutcomeModelArtifact]:
        now = time.monotonic()
        ttl = max(30, self.config.outcome_scoring.model_refresh_seconds)
        if self._model is not None and now - self._model_loaded_at < ttl:
            return self._model
        with self._lock:
            if self._model is not None and now - self._model_loaded_at < ttl:
                return self._model
            stage = (
                "champion"
                if self.config.outcome_scoring.mode == "production"
                else "shadow"
            )
            row = self.repository.load_model(stage=stage)
            self._model_loaded_at = now
            if not row:
                self._model_error = "no_model_available"
                return self._model
            try:
                artifact = OutcomeModelArtifact.from_dict(
                    dict(row.get("artifact") or {})
                )
            except Exception as exc:  # noqa: BLE001
                self._model_error = f"invalid_artifact:{type(exc).__name__}"
                logger.error(
                    "Outcome scoring artifact rejected: {}", type(exc).__name__
                )
                return self._model
            self._model = artifact
            self._model_error = None
            logger.info(
                "Outcome scoring model loaded: version={} stage={} calibrated={}",
                artifact.version,
                row.get("stage"),
                artifact.calibration_ready,
            )
            return self._model


_RUNTIME_LOCK = threading.Lock()
_RUNTIME: Optional[OutcomeScoringRuntime] = None
_RUNTIME_KEY: Optional[tuple] = None


def get_outcome_scoring_runtime(
    config: Optional[AppConfig] = None,
) -> OutcomeScoringRuntime:
    global _RUNTIME, _RUNTIME_KEY
    cfg = config or load_config()
    key = (
        cfg.outcome_scoring.database_url,
        cfg.outcome_scoring.mode,
        cfg.outcome_scoring.enabled,
        cfg.outcome_scoring.feature_schema_version,
    )
    with _RUNTIME_LOCK:
        if _RUNTIME is None or _RUNTIME_KEY != key:
            _RUNTIME = OutcomeScoringRuntime(cfg)
            _RUNTIME_KEY = key
        return _RUNTIME


def score_candidate_shadow(
    candidate: Mapping[str, Any],
    config: Optional[AppConfig] = None,
) -> Optional[Dict[str, Any]]:
    return get_outcome_scoring_runtime(config).score(candidate)


def journal_scan_candidates(
    candidates: Iterable[Mapping[str, Any]],
    config: Optional[AppConfig] = None,
) -> int:
    return get_outcome_scoring_runtime(config).journal(candidates)


def get_outcome_scoring_status(
    config: Optional[AppConfig] = None,
) -> Dict[str, Any]:
    return get_outcome_scoring_runtime(config).status()
