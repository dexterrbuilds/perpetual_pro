"""Non-secret immutable deployment identity."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict

from src import __version__
from src.analysis.execution_policy import EXECUTION_POLICY_VERSION, RANK_POLICY_VERSION
from src.scoring.features import FEATURE_SCHEMA_VERSION


def get_build_identity() -> Dict[str, Any]:
    commit = str(
        os.getenv("GIT_COMMIT_SHA")
        or os.getenv("RAILWAY_GIT_COMMIT_SHA")
        or "unknown"
    ).strip()
    timestamp = str(os.getenv("BUILD_TIMESTAMP") or "unknown").strip()
    environment = str(
        os.getenv("RAILWAY_ENVIRONMENT_NAME")
        or os.getenv("APP_ENVIRONMENT")
        or "local"
    ).strip()
    return {
        "application_version": __version__,
        "git_commit_sha": commit,
        "build_timestamp": timestamp,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "execution_policy": EXECUTION_POLICY_VERSION,
        "rank_policy": RANK_POLICY_VERSION,
        "environment": environment,
        "identity_complete": bool(commit != "unknown" and timestamp != "unknown"),
        "reported_at": datetime.now(timezone.utc).isoformat(),
    }

