"""JSON-safe chart payloads for the Streamlit webapp."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.analysis.indicators import IndicatorSuite
from src.analysis.market_structure import StructureReport
from src.analysis.patterns import PatternReport
from src.analysis.risk import TradePlan


def build_market_chart_payload(
    df: pd.DataFrame,
    indicators: Optional[IndicatorSuite],
    structure: Optional[StructureReport],
    patterns: Optional[PatternReport],
    plan: Optional[TradePlan],
    *,
    timeframe: str,
    limit: int = 140,
) -> Dict[str, Any]:
    """Return candles, overlays, levels, and plan lines without binary images."""
    if df is None or df.empty:
        return {}
    source = indicators.df if indicators is not None and indicators.df is not None else df
    source = source.tail(limit).copy()
    candles: List[Dict[str, Any]] = []
    for ts, row in source.iterrows():
        candles.append(
            {
                "t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "open": _num(row.get("open")),
                "high": _num(row.get("high")),
                "low": _num(row.get("low")),
                "close": _num(row.get("close")),
                "volume": _num(row.get("volume")),
                "ema_fast": _num(row.get("ema_fast")),
                "ema_mid": _num(row.get("ema_mid")),
                "vwap": _num(row.get("vwap")),
            }
        )

    levels: List[Dict[str, Any]] = []
    if structure is not None:
        for level in structure.levels:
            if level.kind not in ("order_block", "fvg", "liquidity", "volume_poc"):
                continue
            levels.append(
                {
                    "kind": level.kind,
                    "side": level.side,
                    "low": level.price_low,
                    "high": level.price_high,
                    "mid": level.mid,
                    "label": level.note or level.kind,
                }
            )
        for name, value in (
            ("POC", structure.volume_profile_poc),
            ("VAH", structure.volume_profile_vah),
            ("VAL", structure.volume_profile_val),
        ):
            if value is not None:
                levels.append(
                    {
                        "kind": "volume_profile",
                        "side": "neutral",
                        "low": value,
                        "high": value,
                        "mid": value,
                        "label": name,
                    }
                )

    pattern_rows = []
    if patterns is not None:
        pattern_rows = [
            {
                "name": hit.name,
                "bias": hit.bias,
                "confidence": hit.confidence,
                "note": hit.note,
            }
            for hit in patterns.top_hits[:6]
        ]

    trade = None
    if plan is not None:
        trade = {
            "direction": plan.direction,
            "entry_low": plan.entry_low,
            "entry_high": plan.entry_high,
            "stop_loss": plan.stop_loss,
            "take_profits": list(plan.take_profits),
            "entry_status": getattr(plan, "entry_status", "blocked"),
            "execution_score": getattr(plan, "execution_score", 0.0),
        }
    return {
        "timeframe": timeframe,
        "candles": candles,
        "levels": levels[:20],
        "patterns": pattern_rows,
        "trade": trade,
    }


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None

