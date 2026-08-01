"""Multi-timeframe OHLCV orchestration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.data.exchange import (
    ExchangeClient,
    MarketSnapshot,
    build_exchange_attempt_order,
    is_permanent_exchange_access_error,
    normalize_exchange_id,
)
from src.utils.config import AppConfig
from src.utils.helpers import timeframe_to_minutes


_VENUE_HEALTH_LOCK = RLock()
_VENUE_BLOCKED_UNTIL: Dict[str, float] = {}
_VENUE_BLOCK_COOLDOWN_SECONDS = 30 * 60


def _venue_is_blocked(exchange_id: str) -> bool:
    """Return whether a venue recently produced a systemic access failure."""
    now = time.monotonic()
    with _VENUE_HEALTH_LOCK:
        until = _VENUE_BLOCKED_UNTIL.get(exchange_id, 0.0)
        if until <= now:
            _VENUE_BLOCKED_UNTIL.pop(exchange_id, None)
            return False
        return True


def _block_venue(exchange_id: str, reason: Any) -> None:
    """Circuit-break a geo/permission-blocked venue for later scan symbols."""
    with _VENUE_HEALTH_LOCK:
        _VENUE_BLOCKED_UNTIL[exchange_id] = (
            time.monotonic() + _VENUE_BLOCK_COOLDOWN_SECONDS
        )
    logger.warning(
        "Venue {} circuit-breaker opened for {} minutes: {}",
        exchange_id,
        _VENUE_BLOCK_COOLDOWN_SECONDS // 60,
        str(reason)[:240],
    )


@dataclass
class MultiTimeframeData:
    """OHLCV frames keyed by timeframe plus optional market snapshot."""

    symbol: str
    exchange_id: str
    primary_tf: str
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    snapshot: Optional[MarketSnapshot] = None
    errors: List[str] = field(default_factory=list)
    quality: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def primary(self) -> pd.DataFrame:
        return self.frames.get(self.primary_tf, pd.DataFrame())

    def higher_frames(self) -> Dict[str, pd.DataFrame]:
        primary_m = timeframe_to_minutes(self.primary_tf)
        return {
            tf: df
            for tf, df in self.frames.items()
            if timeframe_to_minutes(tf) > primary_m and not df.empty
        }

    def all_timeframes(self) -> List[str]:
        return sorted(self.frames.keys(), key=timeframe_to_minutes)


def closed_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Exclude the exchange's still-forming candle from signal calculations."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    last_ts = df.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    closes_at = last_ts + pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
    now = pd.Timestamp(datetime.now(timezone.utc))
    if closes_at > now and len(df) > 1:
        logger.debug("Excluded incomplete {} candle at {}", timeframe, last_ts)
        return df.iloc[:-1].copy()
    return df


def assess_candle_quality(
    df: pd.DataFrame,
    timeframe: str,
    *,
    now: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """Score live-candle freshness and continuity without rejecting old test data."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return {
            "ok": False,
            "score": 0.0,
            "reason": "missing_or_unindexed_candles",
            "bars": 0,
        }

    minutes = max(1, timeframe_to_minutes(timeframe))
    expected = pd.Timedelta(minutes=minutes)
    current = now or pd.Timestamp(datetime.now(timezone.utc))
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    last_open = df.index[-1]
    if last_open.tzinfo is None:
        last_open = last_open.tz_localize("UTC")
    last_close = last_open + expected
    age_intervals = max(
        0.0,
        (current - last_close).total_seconds() / max(expected.total_seconds(), 1.0),
    )

    recent = df.tail(min(160, len(df)))
    diffs = recent.index.to_series().diff().dropna()
    gap_ratio = (
        float((diffs > expected * 1.5).mean())
        if not diffs.empty
        else 0.0
    )
    volumes = pd.to_numeric(recent.get("volume"), errors="coerce")
    zero_volume_ratio = (
        float((volumes.fillna(0) <= 0).mean())
        if volumes is not None and len(volumes)
        else 0.0
    )
    invalid_ohlc = (
        (recent["high"] < recent[["open", "close"]].max(axis=1))
        | (recent["low"] > recent[["open", "close"]].min(axis=1))
        | (recent["low"] <= 0)
        | (recent["high"] <= 0)
    )
    invalid_ratio = float(invalid_ohlc.mean()) if len(invalid_ohlc) else 0.0

    score = 100.0
    score -= min(55.0, age_intervals * 28.0)
    score -= min(30.0, gap_ratio * 300.0)
    score -= min(20.0, zero_volume_ratio * 100.0)
    score -= min(50.0, invalid_ratio * 500.0)
    score = max(0.0, min(100.0, score))
    ok = bool(
        len(df) >= 60
        and age_intervals <= 1.5
        and gap_ratio <= 0.08
        and zero_volume_ratio <= 0.15
        and invalid_ratio == 0.0
    )
    reasons: List[str] = []
    if age_intervals > 1.5:
        reasons.append(f"stale_{age_intervals:.1f}_intervals")
    if gap_ratio > 0.08:
        reasons.append(f"gaps_{gap_ratio:.0%}")
    if zero_volume_ratio > 0.15:
        reasons.append(f"zero_volume_{zero_volume_ratio:.0%}")
    if invalid_ratio > 0:
        reasons.append(f"invalid_ohlc_{invalid_ratio:.0%}")
    if len(df) < 60:
        reasons.append(f"only_{len(df)}_bars")
    return {
        "ok": ok,
        "score": round(score, 1),
        "reason": ",".join(reasons) if reasons else "fresh_continuous_closed_candles",
        "bars": len(df),
        "last_open": last_open.isoformat(),
        "last_close": last_close.isoformat(),
        "age_intervals": round(age_intervals, 3),
        "gap_ratio": round(gap_ratio, 4),
        "zero_volume_ratio": round(zero_volume_ratio, 4),
        "invalid_ratio": round(invalid_ratio, 4),
    }


def fetch_multi_timeframe(
    client: ExchangeClient,
    symbol: str,
    primary_tf: str,
    higher_tfs: Optional[List[str]] = None,
    limit: int = 500,
    include_snapshot: bool = True,
    config: Optional[AppConfig] = None,
) -> MultiTimeframeData:
    """
    Fetch primary + higher timeframe OHLCV (and optional derivatives snapshot).

    Higher TFs default to the day-trade stack: 1h drive, 4h confirmation.
    """
    if higher_tfs is None:
        higher_tfs = (
            list(config.timeframes.higher)
            if config
            else ["1h", "4h"]
        )
    # Day-trade bar depth (enough for micro-structure without HTF bloat)
    if config:
        limit = max(limit or 0, int(config.timeframes.ohlcv_limit or 500))
    else:
        limit = max(limit or 300, 500)

    # Deduplicate while preserving order; always include primary
    ordered: List[str] = []
    for tf in [primary_tf, *higher_tfs]:
        if tf and tf not in ordered:
            ordered.append(tf)

    # Sort ascending so primary context is clear in logs
    ordered = sorted(ordered, key=timeframe_to_minutes)

    result = MultiTimeframeData(
        symbol=symbol,
        exchange_id=client.exchange_id,
        primary_tf=primary_tf,
    )

    workers = max(
        1,
        min(
            len(ordered) + (1 if include_snapshot else 0),
            int(getattr(getattr(config, "timeframes", None), "fetch_workers", 4) or 4),
        ),
    )
    futures: Dict[Future, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for tf in ordered:
            future = pool.submit(client.fetch_ohlcv, symbol, timeframe=tf, limit=limit)
            futures[future] = ("ohlcv", tf)
        if include_snapshot:
            futures[pool.submit(client.fetch_market_snapshot, symbol)] = ("snapshot", "snapshot")

        for future in as_completed(futures):
            kind, label = futures[future]
            try:
                value = future.result()
                if kind == "ohlcv":
                    value = closed_candles(value, label)
                    result.frames[label] = value
                    result.quality[label] = assess_candle_quality(value, label)
                    logger.info("Loaded {} · {} · {} closed bars", symbol, label, len(value))
                else:
                    result.snapshot = value
                    if value and value.symbol:
                        result.symbol = value.symbol
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{label}: {exc}")
                if kind == "ohlcv":
                    result.frames[label] = pd.DataFrame()
                    result.quality[label] = {
                        "ok": False,
                        "score": 0.0,
                        "reason": f"fetch_failed:{type(exc).__name__}",
                        "bars": 0,
                    }
                    logger.error("Failed multi-tf fetch {}: {}", label, exc)
                else:
                    logger.warning("Market snapshot failed: {}", exc)

    # Preserve a deterministic frame order despite completion order.
    result.frames = {tf: result.frames.get(tf, pd.DataFrame()) for tf in ordered}

    return result


@dataclass
class FallbackFetchResult:
    """Result of multi-exchange OHLCV fetch with automatic venue fallback."""

    mtf: MultiTimeframeData
    client: ExchangeClient
    requested_exchange: str
    exchange_used: str
    fallback_used: bool
    attempted_exchanges: List[str] = field(default_factory=list)


def fetch_multi_timeframe_with_fallback(
    symbol: str,
    primary_tf: str,
    preferred_exchange: str,
    higher_tfs: Optional[List[str]] = None,
    limit: int = 500,
    include_snapshot: bool = True,
    config: Optional[AppConfig] = None,
    auto_fallback: Optional[bool] = None,
    max_exchanges: Optional[int] = None,
    deadline_monotonic: Optional[float] = None,
) -> FallbackFetchResult:
    """
    Try preferred exchange first; on missing symbol/OHLCV, iterate fallbacks.

    Returns the first exchange that yields non-empty primary OHLCV, or the last
    empty attempt if all venues fail (caller may fall back to vision-only mode).

    Caller owns the returned ``client`` and must call ``client.close()``.
    Failed intermediate clients are always closed inside this function.
    """
    requested = normalize_exchange_id(preferred_exchange)
    exchanges = build_exchange_attempt_order(
        requested, config, auto_fallback=auto_fallback
    )
    if max_exchanges is not None:
        exchanges = exchanges[: max(1, int(max_exchanges))]
    healthy = [ex for ex in exchanges if not _venue_is_blocked(ex)]
    cooled_down = [ex for ex in exchanges if ex not in healthy]
    if healthy and cooled_down:
        exchanges = [*healthy, *cooled_down]
        logger.info(
            "Deferring temporarily blocked venues to the end: {}",
            ", ".join(cooled_down),
        )
    attempted: List[str] = []
    last_client: Optional[ExchangeClient] = None
    last_mtf: Optional[MultiTimeframeData] = None

    for ex_id in exchanges:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            logger.warning("Exchange fallback budget exhausted before venue={}", ex_id)
            break
        attempted.append(ex_id)
        client: Optional[ExchangeClient] = None
        try:
            client = ExchangeClient(exchange_id=ex_id, config=config)
            mtf = fetch_multi_timeframe(
                client,
                symbol=symbol,
                primary_tf=primary_tf,
                higher_tfs=higher_tfs,
                limit=limit,
                include_snapshot=include_snapshot,
                config=config,
            )
            if not mtf.primary.empty:
                # Success — release any prior empty-result client first
                if last_client is not None:
                    last_client.close()
                    last_client = None
                fallback_used = ex_id != requested
                if fallback_used:
                    logger.info(
                        "Symbol {} unavailable on {} — using {} for data "
                        "(tried: {})",
                        symbol,
                        requested,
                        ex_id,
                        " → ".join(attempted),
                    )
                else:
                    logger.info("Using {} for {} market data", ex_id, symbol)
                return FallbackFetchResult(
                    mtf=mtf,
                    client=client,
                    requested_exchange=requested,
                    exchange_used=ex_id,
                    fallback_used=fallback_used,
                    attempted_exchanges=list(attempted),
                )

            logger.warning(
                "Empty OHLCV for {} on {} — trying next exchange",
                symbol,
                ex_id,
            )
            failure_detail = " | ".join(mtf.errors)
            if is_permanent_exchange_access_error(failure_detail):
                _block_venue(ex_id, failure_detail)
            # Keep this client as last-resort shell; drop previous empty one
            if last_client is not None:
                last_client.close()
            last_client = client
            last_mtf = mtf
            client = None  # ownership transferred to last_client
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exchange {} failed for {}: {}", ex_id, symbol, exc)
            if is_permanent_exchange_access_error(exc):
                _block_venue(ex_id, exc)
            if client is not None:
                client.close()
                client = None

    if last_client is not None and last_mtf is not None:
        used = last_client.exchange_id
        logger.warning(
            "All exchanges failed for {}; returning last empty attempt on {} "
            "(tried: {})",
            symbol,
            used,
            " → ".join(attempted) if attempted else requested,
        )
        return FallbackFetchResult(
            mtf=last_mtf,
            client=last_client,
            requested_exchange=requested,
            exchange_used=used,
            fallback_used=used != requested,
            attempted_exchanges=list(attempted),
        )

    # Every bounded attempt raised. Do not silently repeat the preferred venue;
    # that used to double timeout cost for unsupported symbols.
    client = ExchangeClient(exchange_id=requested, config=config)
    mtf = MultiTimeframeData(
        symbol=symbol,
        exchange_id=requested,
        primary_tf=primary_tf,
        errors=["bounded_exchange_fallback_exhausted"],
    )
    return FallbackFetchResult(
        mtf=mtf,
        client=client,
        requested_exchange=requested,
        exchange_used=client.exchange_id,
        fallback_used=False,
        attempted_exchanges=list(attempted) if attempted else [requested],
    )
