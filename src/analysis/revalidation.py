"""Fail-closed final validation immediately before Telegram delivery."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from loguru import logger

from src.data.exchange import ExchangeClient
from src.data.multi_tf import assess_candle_quality, closed_candles
from src.utils.config import AppConfig
from src.utils.helpers import clamp, safe_float


def _age_seconds(timestamp_ms: Any) -> Optional[float]:
    source = safe_float(timestamp_ms)
    if source <= 0:
        return None
    if source < 10_000_000_000:
        source *= 1000.0
    return max(0.0, time.time() - source / 1000.0)


def evaluate_pre_delivery_candidate(
    row: Mapping[str, Any],
    *,
    current_price: float,
    spread_bps: Optional[float],
    ticker_age_seconds: Optional[float],
    orderbook_age_seconds: Optional[float],
    latest_closed_price: float,
    candle_data_ok: bool,
    config: AppConfig,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Re-evaluate published levels without changing strategy calculations."""
    updated = dict(row)
    reasons: list[str] = []
    direction = str(row.get("direction") or "").lower()
    entry_low, entry_high = sorted(
        (safe_float(row.get("entry_low")), safe_float(row.get("entry_high")))
    )
    stop = safe_float(row.get("stop_loss"))
    targets = [safe_float(value) for value in row.get("take_profits") or []]
    if (
        direction not in ("long", "short")
        or current_price <= 0
        or entry_low <= 0
        or entry_high <= 0
        or stop <= 0
        or not targets
    ):
        reasons.append("PRE_SEND_LEVELS_INVALID")
        return {"ok": False, "row": updated, "reasons": reasons}

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    valid_until_raw = str(row.get("entry_valid_until") or "").strip()
    if valid_until_raw:
        try:
            valid_until = datetime.fromisoformat(valid_until_raw.replace("Z", "+00:00"))
            if valid_until.tzinfo is None:
                valid_until = valid_until.replace(tzinfo=timezone.utc)
            if current.astimezone(timezone.utc) >= valid_until.astimezone(timezone.utc):
                reasons.append("PRE_SEND_ENTRY_EXPIRED")
        except ValueError:
            reasons.append("PRE_SEND_EXPIRY_INVALID")

    ticker_limit = float(config.analysis.max_ticker_age_seconds)
    book_limit = float(config.analysis.max_orderbook_age_seconds)
    if ticker_age_seconds is None or ticker_age_seconds > ticker_limit:
        reasons.append("PRE_SEND_TICKER_STALE")
    if orderbook_age_seconds is None or orderbook_age_seconds > book_limit:
        reasons.append("PRE_SEND_ORDERBOOK_STALE")
    if spread_bps is None:
        reasons.append("PRE_SEND_SPREAD_UNAVAILABLE")
    elif spread_bps > float(config.analysis.max_spread_bps):
        reasons.append("PRE_SEND_SPREAD_TOO_WIDE")
    if not candle_data_ok or latest_closed_price <= 0:
        reasons.append("PRE_SEND_CANDLES_STALE")

    stop_traded = current_price <= stop if direction == "long" else current_price >= stop
    close_invalid = (
        latest_closed_price <= stop
        if direction == "long"
        else latest_closed_price >= stop
    )
    if stop_traded:
        reasons.append("PRE_SEND_STOP_ALREADY_TRADED")
    if close_invalid:
        reasons.append("PRE_SEND_STRUCTURE_INVALIDATED")

    tp1 = targets[0]
    tp1_traded = current_price >= tp1 if direction == "long" else current_price <= tp1
    if tp1_traded:
        reasons.append("PRE_SEND_TP1_ALREADY_TRADED")

    inside = entry_low <= current_price <= entry_high
    favorable_beyond = (
        current_price > entry_high if direction == "long" else current_price < entry_low
    )
    relation = "inside" if inside else (
        "favorable_beyond" if favorable_beyond else "adverse_side"
    )
    entry_mid = (entry_low + entry_high) / 2.0
    tp_distance = abs(tp1 - entry_mid)
    favorable_move = (
        current_price - entry_mid
        if direction == "long"
        else entry_mid - current_price
    )
    progress = float(
        clamp(favorable_move / max(tp_distance, 1e-12) * 100.0, 0.0, 200.0)
    )
    if favorable_beyond and progress >= float(
        config.analysis.max_pre_entry_tp1_progress_pct
    ):
        reasons.append("PRE_SEND_ENTRY_MOVE_MOSTLY_MISSED")

    updated.update(
        {
            "price": current_price,
            "spread_bps": spread_bps,
            "ticker_age_seconds": ticker_age_seconds,
            "orderbook_age_seconds": orderbook_age_seconds,
            "entry_zone_relation": relation,
            "tp1_progress_pct": round(progress, 1),
            "entry_status": (
                "confirmation_pending" if inside else "wait_retest"
            ),
            "market_quality_ok": not reasons,
            "pre_delivery_revalidated_at": current.isoformat(),
        }
    )
    return {"ok": not reasons, "row": updated, "reasons": reasons}


def revalidate_candidate_for_delivery(
    row: Mapping[str, Any],
    config: AppConfig,
) -> Dict[str, Any]:
    """Force-refresh the execution sources and apply the pure final checks."""
    exchange_id = str(row.get("exchange") or config.exchange.default)
    symbol = str(row.get("symbol") or "")
    timeframe = str(row.get("primary_tf") or config.timeframes.primary)
    client = ExchangeClient(exchange_id=exchange_id, config=config)
    try:
        ticker = client.fetch_ticker(symbol, force_refresh=True)
        book = client.fetch_order_book_summary(symbol, 25, force_refresh=True)
        candles = client.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=8,
            force_refresh=True,
        )
        candles = closed_candles(candles, timeframe)
        quality = assess_candle_quality(candles, timeframe)
        current_price = 0.0
        for key in ("last", "close", "mark", "index", "ask", "bid"):
            candidate = safe_float(ticker.get(key))
            if candidate > 0:
                current_price = candidate
                break
        ticker_timestamp = (
            ticker.get("_perpetual_pro_source_timestamp_ms")
            or ticker.get("_perpetual_pro_observed_at_ms")
        )
        result = evaluate_pre_delivery_candidate(
            row,
            current_price=current_price,
            spread_bps=(
                safe_float(book.get("spread_bps"))
                if book.get("spread_bps") is not None
                else None
            ),
            ticker_age_seconds=_age_seconds(ticker_timestamp),
            orderbook_age_seconds=_age_seconds(book.get("timestamp")),
            latest_closed_price=(
                safe_float(candles["close"].iloc[-1]) if not candles.empty else 0.0
            ),
            candle_data_ok=bool(quality.get("ok")),
            config=config,
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Pre-Telegram revalidation failed: symbol={} error_type={}",
            symbol,
            type(exc).__name__,
        )
        return {
            "ok": False,
            "row": dict(row),
            "reasons": [f"PRE_SEND_DATA_FAILURE:{type(exc).__name__}"],
        }
    finally:
        client.close()
