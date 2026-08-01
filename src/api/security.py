"""Authentication and resource budgets for expensive analysis endpoints."""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Deque, Dict, Iterable, List, Optional

from fastapi import HTTPException, Request


ALLOWED_TIMEFRAMES = frozenset({"15m", "1h", "4h"})
ALLOWED_EXCHANGES = frozenset({"okx", "bybit", "bitget", "binanceusdm"})
MAX_SCAN_SYMBOLS = 20
MAX_SCAN_CONCURRENCY = 2
SCAN_REQUESTS_PER_MINUTE = 6
SCAN_TIMEOUT_SECONDS = 240
SCAN_FALLBACK_EXCHANGES = 2
SCAN_BUDGET_SECONDS = 210


class ScanAccessController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        self._slots = threading.BoundedSemaphore(MAX_SCAN_CONCURRENCY)

    @staticmethod
    def configured_key() -> str:
        return str(os.getenv("SCAN_API_KEY") or "").strip()

    def authorize(self, request: Request, provided: Optional[str]) -> str:
        expected = self.configured_key()
        production = bool(os.getenv("RAILWAY_ENVIRONMENT_NAME"))
        if not expected:
            if production:
                raise HTTPException(status_code=503, detail={"error": "scan_auth_not_configured"})
            # Tests/local development remain usable without silently opening a
            # deployed Railway endpoint.
            return "local-development"
        bearer = str(request.headers.get("authorization") or "").strip()
        if bearer.lower().startswith("bearer "):
            provided = bearer[7:].strip()
        if not provided or not hmac.compare_digest(str(provided), expected):
            raise HTTPException(status_code=401, detail={"error": "unauthorized"})
        identity = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:16]
        now = time.monotonic()
        with self._lock:
            bucket = self._requests[identity]
            while bucket and now - bucket[0] >= 60.0:
                bucket.popleft()
            if len(bucket) >= SCAN_REQUESTS_PER_MINUTE:
                raise HTTPException(
                    status_code=429,
                    detail={"error": "rate_limited", "retry_after_seconds": 60},
                    headers={"Retry-After": "60"},
                )
            bucket.append(now)
        return identity

    @contextmanager
    def slot(self):
        if not self._slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail={"error": "scan_concurrency_limit"})
        try:
            yield
        finally:
            self._slots.release()


SCAN_ACCESS = ScanAccessController()


def normalize_requested_symbols(
    symbols: Iterable[str],
    *,
    approved_bases: Iterable[str],
) -> List[str]:
    values = [str(item or "").strip().upper() for item in symbols if str(item or "").strip()]
    if not values:
        raise HTTPException(status_code=422, detail={"error": "no_symbols"})
    if len(values) > MAX_SCAN_SYMBOLS:
        raise HTTPException(
            status_code=422,
            detail={"error": "symbol_limit_exceeded", "maximum": MAX_SCAN_SYMBOLS},
        )
    approved = {str(item).upper().strip() for item in approved_bases}
    normalized: List[str] = []
    unsupported: List[str] = []
    for value in values:
        base = value.split("/", 1)[0].split(":", 1)[0]
        for quote in ("USDT", "USDC", "USD"):
            if base.endswith(quote) and len(base) > len(quote):
                base = base[: -len(quote)]
                break
        if base not in approved:
            unsupported.append(base)
        elif base not in normalized:
            normalized.append(base)
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail={"error": "unsupported_symbol", "symbols": sorted(set(unsupported))},
        )
    return normalized


def validate_timeframe(value: Optional[str], default: str) -> str:
    timeframe = str(value or default).strip().lower()
    if timeframe not in ALLOWED_TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail={"error": "unsupported_timeframe", "allowed": sorted(ALLOWED_TIMEFRAMES)},
        )
    return timeframe


def validate_exchange(value: Optional[str], default: str) -> str:
    exchange = str(value or default).strip().lower().replace("-", "").replace("_", "")
    if exchange not in ALLOWED_EXCHANGES:
        raise HTTPException(status_code=422, detail={"error": "unsupported_exchange"})
    return exchange

