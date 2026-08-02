"""Shared rejection analytics runtime."""

from __future__ import annotations

import threading
from typing import Optional

from src.analytics.repository import RejectionAnalyticsRepository
from src.utils.config import AppConfig, load_config


_LOCK = threading.Lock()
_REPOSITORY: Optional[RejectionAnalyticsRepository] = None
_KEY: Optional[str] = None


def get_rejection_repository(
    config: Optional[AppConfig] = None,
) -> RejectionAnalyticsRepository:
    global _REPOSITORY, _KEY
    cfg = config or load_config()
    key = str(cfg.outcome_scoring.database_url or "")
    with _LOCK:
        if _REPOSITORY is None or _KEY != key:
            _REPOSITORY = RejectionAnalyticsRepository(key)
            _KEY = key
        return _REPOSITORY
