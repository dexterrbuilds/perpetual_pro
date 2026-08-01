"""Leakage-safe historical replay through the production analysis engine."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
from loguru import logger

from src.analysis.confluence import ConfluenceEngine
from src.data.multi_tf import MultiTimeframeData
from src.scoring.features import build_candidate_record
from src.scoring.labels import label_candidate_from_ohlcv
from src.utils.config import AppConfig
from src.utils.helpers import timeframe_to_minutes


def replay_historical_candidates(
    *,
    symbol: str,
    exchange_id: str,
    primary_tf: str,
    frames: Mapping[str, pd.DataFrame],
    config: AppConfig,
    step: int = 3,
    warmup: int = 240,
    max_candidates: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Replay the exact confluence/execution engine using as-of candle slices.

    Historical derivatives/order-book fields are intentionally left missing
    unless a future data archive provides timestamped snapshots.
    """
    normalized = {
        timeframe: _utc_frame(frame)
        for timeframe, frame in frames.items()
        if frame is not None and not frame.empty
    }
    primary = normalized.get(primary_tf)
    if primary is None or len(primary) < warmup + 40:
        raise ValueError("Insufficient primary candles for historical replay")
    primary_minutes = timeframe_to_minutes(primary_tf)
    engine = ConfluenceEngine(config)
    candidates: List[Dict[str, Any]] = []
    outcomes: List[Dict[str, Any]] = []
    # Do not label a recent candidate with an incomplete future window. Reserve
    # the full maximum lifecycle: three hours pending entry plus 24 hours hold.
    lifecycle_bars = int(math.ceil((27 * 60) / primary_minutes)) + 1
    final_index = len(primary) - lifecycle_bars
    if final_index <= warmup:
        raise ValueError(
            "Insufficient primary candles after reserving the 27h outcome window"
        )

    for index in range(warmup, final_index, max(1, int(step))):
        candle_open = primary.index[index]
        decision_time = candle_open + pd.Timedelta(minutes=primary_minutes)
        sliced: Dict[str, pd.DataFrame] = {}
        for timeframe, frame in normalized.items():
            close_delta = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
            available = frame[(frame.index + close_delta) <= decision_time]
            if not available.empty:
                sliced[timeframe] = available.copy()
        if primary_tf not in sliced or len(sliced[primary_tf]) < warmup:
            continue

        mtf = MultiTimeframeData(
            symbol=symbol,
            exchange_id=exchange_id,
            primary_tf=primary_tf,
            frames=sliced,
            snapshot=None,
            quality={},
        )
        try:
            analysis = engine.analyze(mtf, news=None, use_llm=False)
            bias = str(analysis.bias or "").lower()
            inferred_direction = (
                "long" if bias == "bullish" else (
                    "short" if bias == "bearish" else "flat"
                )
            )
            if inferred_direction not in ("long", "short"):
                continue
            plan = analysis.trade_plan
            execution = analysis.execution.to_dict() if analysis.execution else {}
            if plan is None or not plan.take_profits:
                continue
            generated_iso = decision_time.to_pydatetime().isoformat()
            validity_minutes = max(15, int(plan.entry_valid_for_minutes or 60))
            valid_until = (
                decision_time + pd.Timedelta(minutes=validity_minutes)
            ).to_pydatetime().isoformat()
            row = {
                "symbol": symbol,
                "exchange": exchange_id,
                "primary_tf": primary_tf,
                "direction": analysis.direction,
                "bias": analysis.bias,
                "confidence": analysis.confidence,
                "legacy_confidence": (analysis.meta or {}).get(
                    "legacy_confidence",
                    analysis.confidence,
                ),
                "legacy_v2_confidence": (analysis.meta or {}).get(
                    "legacy_v2_confidence",
                    analysis.confidence,
                ),
                "technical_confidence": analysis.technical_confidence,
                "llm_confidence": analysis.llm_confidence,
                "rank_score": analysis.rank_score,
                "confluence_score": analysis.confluence_total,
                "setup_name": analysis.setup_name,
                "strategy_tags": list(analysis.strategy_tags),
                "signal_eligible": bool(
                    (analysis.meta or {}).get("signal_eligible", False)
                ),
                "legacy_signal_eligible": bool(
                    (analysis.meta or {}).get(
                        "legacy_signal_eligible",
                        False,
                    )
                ),
                "legacy_v2_signal_eligible": bool(
                    (analysis.meta or {}).get(
                        "legacy_v2_signal_eligible",
                        False,
                    )
                ),
                "rejection_reasons": list(
                    (analysis.meta or {}).get("rejection_reasons") or []
                ),
                "prop_safe": bool(plan.prop_safe),
                "historical_edge_ok": True,
                "data_quality_ok": True,
                "market_quality_ok": bool(
                    execution.get("market_quality_ok", True)
                ),
                "price": float((analysis.meta or {}).get("price") or 0),
                "entry_low": plan.entry_low,
                "entry_high": plan.entry_high,
                "stop_loss": plan.stop_loss,
                "take_profits": list(plan.take_profits),
                "risk_reward": list(plan.risk_reward),
                "entry_status": execution.get("status", "blocked"),
                "execution_score": execution.get("score", 0),
                "immediate_sl_risk": execution.get("immediate_sl_risk", 100),
                "chase_distance_atr": execution.get("chase_distance_atr", 0),
                "order_flow_score": execution.get("order_flow_score", 0),
                "spread_bps": None,
                "orderbook_imbalance": None,
                "orderbook_alignment": 0,
                "entry_valid_for_minutes": validity_minutes,
                "entry_valid_until": valid_until,
                "hold_hours_max": plan.hold_hours_max,
                "signal_generated_at": generated_iso,
                "risk_pct": plan.risk_pct,
                "funding_rate": None,
                "open_interest_change_pct_24h": None,
                "execution": execution,
            }
            analysis.generated_at = generated_iso
            candidate = build_candidate_record(
                analysis,
                row,
                source="historical_replay",
                feature_schema_version=config.outcome_scoring.feature_schema_version,
            )
            # Ensure historical timestamps override lifecycle fields generated by
            # the live wall clock inside RiskManager.
            candidate["generated_at"] = generated_iso
            candidate["decision"]["entry_valid_until"] = valid_until
            candidate["decision"]["entry_valid_for_minutes"] = validity_minutes
            candidate["decision"]["hold_hours_max"] = plan.hold_hours_max
            future_end = decision_time + pd.Timedelta(
                hours=max(1.0, float(plan.hold_hours_max or 12.0)) + 3.0
            )
            future = primary[
                (primary.index >= decision_time)
                & (primary.index <= future_end)
            ]
            if future.empty:
                continue
            outcome = label_candidate_from_ohlcv(candidate, future).to_dict()
            outcome.update(
                {
                    "candidate_id": candidate["id"],
                    "metadata": {
                        "symbol": symbol,
                        "exchange_id": exchange_id,
                        "feature_availability": (
                            "closed_ohlcv_only; historical orderbook/funding/OI "
                            "not synthesized"
                        ),
                    },
                }
            )
            candidates.append(candidate)
            outcomes.append(outcome)
            if max_candidates and len(candidates) >= max_candidates:
                break
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Historical replay candidate skipped: symbol={} index={} error={}",
                symbol,
                index,
                type(exc).__name__,
            )
    return candidates, outcomes


def _utc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy().sort_index()
    if not isinstance(output.index, pd.DatetimeIndex):
        output.index = pd.to_datetime(output.index, utc=True)
    elif output.index.tz is None:
        output.index = output.index.tz_localize("UTC")
    else:
        output.index = output.index.tz_convert("UTC")
    return output
