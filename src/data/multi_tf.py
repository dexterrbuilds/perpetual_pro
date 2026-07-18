"""Multi-timeframe OHLCV orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import pandas_ta as ta
from loguru import logger

from src.data.exchange import ExchangeClient, MarketSnapshot
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

    Higher TFs default from config or common prop stack (1h/4h/1d).
    """
    if higher_tfs is None:
        higher_tfs = (
            list(config.timeframes.higher)
            if config
            else ["5m", "1h", "4h", "1d"]
        )
    # Max signal quality default stack (includes lower TF context when useful)
    if config:
        limit = max(limit or 0, int(config.timeframes.ohlcv_limit or 1000))
    else:
        limit = max(limit or 500, 1000)

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

    for tf in ordered:
        try:
            df = client.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            result.frames[tf] = df
            logger.info("Loaded {} · {} · {} bars", symbol, tf, len(df))
        except Exception as exc:  # noqa: BLE001
            msg = f"{tf}: {exc}"
            result.errors.append(msg)
            logger.error("Failed multi-tf fetch {}: {}", tf, exc)
            result.frames[tf] = pd.DataFrame()

    if include_snapshot:
        try:
            result.snapshot = client.fetch_market_snapshot(symbol)
            if result.snapshot and result.snapshot.symbol:
                result.symbol = result.snapshot.symbol
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"snapshot: {exc}")
            logger.warning("Market snapshot failed: {}", exc)

    return result
