"""Multi-timeframe OHLCV orchestration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.data.exchange import (
    ExchangeClient,
    MarketSnapshot,
    build_exchange_attempt_order,
    normalize_exchange_id,
)
from src.utils.config import AppConfig
from src.utils.helpers import timeframe_to_minutes


@dataclass
class MultiTimeframeData:
    """OHLCV frames keyed by timeframe plus optional market snapshot."""

    symbol: str
    exchange_id: str
    primary_tf: str
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    snapshot: Optional[MarketSnapshot] = None
    errors: List[str] = field(default_factory=list)

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
                    result.frames[label] = value
                    logger.info("Loaded {} · {} · {} bars", symbol, label, len(value))
                else:
                    result.snapshot = value
                    if value and value.symbol:
                        result.symbol = value.symbol
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{label}: {exc}")
                if kind == "ohlcv":
                    result.frames[label] = pd.DataFrame()
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
    attempted: List[str] = []
    last_client: Optional[ExchangeClient] = None
    last_mtf: Optional[MultiTimeframeData] = None

    for ex_id in exchanges:
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
            # Keep this client as last-resort shell; drop previous empty one
            if last_client is not None:
                last_client.close()
            last_client = client
            last_mtf = mtf
            client = None  # ownership transferred to last_client
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exchange {} failed for {}: {}", ex_id, symbol, exc)
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

    # Every attempt raised — open preferred so caller still has a client to close
    client = ExchangeClient(exchange_id=requested, config=config)
    try:
        mtf = fetch_multi_timeframe(
            client,
            symbol=symbol,
            primary_tf=primary_tf,
            higher_tfs=higher_tfs,
            limit=limit,
            include_snapshot=include_snapshot,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Final preferred-exchange fetch failed for {} on {}: {}",
            symbol,
            requested,
            exc,
        )
        mtf = MultiTimeframeData(
            symbol=symbol,
            exchange_id=requested,
            primary_tf=primary_tf,
            errors=[str(exc)],
        )
    return FallbackFetchResult(
        mtf=mtf,
        client=client,
        requested_exchange=requested,
        exchange_used=client.exchange_id,
        fallback_used=False,
        attempted_exchanges=list(attempted) if attempted else [requested],
    )
