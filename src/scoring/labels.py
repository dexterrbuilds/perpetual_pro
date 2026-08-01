"""Conservative event labels from post-signal OHLCV.

The same-candle policy is intentionally pessimistic: adverse/stop events win
when lower-resolution data cannot establish the true sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional

import pandas as pd


@dataclass
class CandidateOutcome:
    valid_fill: bool
    technical_success: Optional[bool]
    alert_success: bool
    tp1_hit: bool
    tp2_hit: bool
    invalidated_before_fill: bool
    missed_before_fill: bool
    expired_before_fill: bool
    entry_delay_minutes: Optional[float]
    trade_duration_minutes: Optional[float]
    realized_r: float
    mfe_r: float
    mae_r: float
    terminal_status: str
    terminal_at: str
    ambiguity_policy: str = "stop_first"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def technical_success_from_range(
    *,
    direction: str,
    start_price: float,
    atr: float,
    high: float,
    low: float,
) -> Optional[bool]:
    """Resolve the canonical ±1 ATR directional label, adverse event first."""
    if direction not in ("long", "short") or start_price <= 0 or atr <= 0:
        return None
    favorable = start_price + atr if direction == "long" else start_price - atr
    adverse = start_price - atr if direction == "long" else start_price + atr
    adverse_hit = low <= adverse if direction == "long" else high >= adverse
    favorable_hit = high >= favorable if direction == "long" else low <= favorable
    if adverse_hit:
        return False
    if favorable_hit:
        return True
    return None


def label_candidate_from_ohlcv(
    candidate: Mapping[str, Any],
    future: pd.DataFrame,
) -> CandidateOutcome:
    """Resolve one published plan against time-ordered future candles."""
    decision = dict(candidate.get("decision") or {})
    direction = str(candidate.get("direction") or "").lower()
    if direction not in ("long", "short"):
        raise ValueError("Candidate direction must be long or short")
    if future is None or future.empty:
        raise ValueError("Future OHLCV is required")

    frame = future.sort_index().copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")

    generated = _parse_time(candidate.get("generated_at"))
    valid_until = _parse_time(decision.get("entry_valid_until"))
    if valid_until is None:
        valid_minutes = max(
            15.0, _number(decision.get("entry_valid_for_minutes"), 60.0)
        )
        valid_until = generated + timedelta(minutes=valid_minutes)
    hold_hours = max(0.5, _number(decision.get("hold_hours_max"), 12.0))
    entry_low = _number(decision.get("entry_low"))
    entry_high = _number(decision.get("entry_high"))
    entry_low, entry_high = min(entry_low, entry_high), max(entry_low, entry_high)
    entry_mid = (entry_low + entry_high) / 2.0
    stop = _number(decision.get("stop_loss"))
    targets = [_number(value) for value in decision.get("take_profits") or []]
    if not entry_mid or not stop or not targets:
        raise ValueError("Candidate is missing entry, stop, or target levels")

    start_price = _number(decision.get("price"), entry_mid)
    atr = max(
        _number(decision.get("atr")),
        start_price * _number(decision.get("atr_pct")) / 100.0,
        start_price * 0.001,
    )
    technical_success: Optional[bool] = None
    confirmation_required = bool(
        str(decision.get("entry_status") or "").lower()
        in ("ready", "confirmation_pending")
        and entry_low <= start_price <= entry_high
    )
    # CMP publication is not a fill. It requires a later closed candle to
    # confirm inside the published zone, matching the forward tracker.
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    highest_tp = 0
    mfe_r = 0.0
    mae_r = 0.0
    terminal_status = "expired"
    terminal_at = min(valid_until, frame.index[-1].to_pydatetime())

    for index, candle in frame.iterrows():
        timestamp = index.to_pydatetime().astimezone(timezone.utc)
        if timestamp <= generated:
            continue
        high = _number(candle.get("high"))
        low = _number(candle.get("low"))
        close = _number(candle.get("close"))

        if technical_success is None:
            technical_success = technical_success_from_range(
                direction=direction,
                start_price=start_price,
                atr=atr,
                high=high,
                low=low,
            )

        if entry_time is None:
            adverse_close = (
                close <= stop if direction == "long" else close >= stop
            )
            confirmation_target_touched = confirmation_required and (
                high >= targets[0] if direction == "long" else low <= targets[0]
            )
            target_without_fill = (
                high >= targets[0] and low > entry_high
                if direction == "long"
                else low <= targets[0] and high < entry_low
            )
            if adverse_close:
                terminal_status = "invalidated"
                terminal_at = timestamp
                break
            if confirmation_target_touched:
                terminal_status = "missed"
                terminal_at = timestamp
                break
            if target_without_fill:
                terminal_status = "missed"
                terminal_at = timestamp
                break
            if timestamp > valid_until:
                terminal_status = "expired"
                terminal_at = timestamp
                break
            touched = low <= entry_high and high >= entry_low
            confirmed = confirmation_required and entry_low <= close <= entry_high
            first_target_touched = (
                high >= targets[0] if direction == "long" else low <= targets[0]
            )
            if confirmed and first_target_touched:
                # Confirmation exists only at this candle's close, so TP1 was
                # observed before the fill could be confirmed.
                terminal_status = "missed"
                terminal_at = timestamp
                break
            if confirmed:
                entry_time = timestamp
                entry_price = close
            elif touched and not confirmation_required:
                entry_time = timestamp
                entry_price = entry_mid
            else:
                continue

        assert entry_price is not None and entry_time is not None
        risk = max(abs(entry_price - stop), entry_price * 1e-9)
        favorable = (
            (high - entry_price) / risk
            if direction == "long"
            else (entry_price - low) / risk
        )
        adverse = (
            (low - entry_price) / risk
            if direction == "long"
            else (entry_price - high) / risk
        )
        mfe_r = max(mfe_r, favorable)
        mae_r = min(mae_r, adverse)
        protected_exit = bool(
            highest_tp >= 1
            and (
                low <= entry_price
                if direction == "long"
                else high >= entry_price
            )
        )
        if protected_exit:
            # TP1 is a realized win and moves the unclosed remainder to
            # breakeven. A later reversal is not an original-stop loss.
            terminal_status = "completed"
            terminal_at = timestamp
            break
        stop_hit = low <= stop if direction == "long" else high >= stop
        if stop_hit:
            terminal_status = "stopped"
            terminal_at = timestamp
            break
        for target_number, target in enumerate(targets, 1):
            target_hit = high >= target if direction == "long" else low <= target
            if target_number > highest_tp and target_hit:
                highest_tp = target_number
        if highest_tp >= len(targets):
            terminal_status = "completed"
            terminal_at = timestamp
            break
        if timestamp >= entry_time + timedelta(hours=hold_hours):
            terminal_status = "time_exit"
            terminal_at = timestamp
            break
    else:
        if entry_time is None:
            terminal_status = "expired"
        else:
            terminal_status = "time_exit"
        terminal_at = frame.index[-1].to_pydatetime().astimezone(timezone.utc)

    valid_fill = entry_time is not None
    tp1_hit = highest_tp >= 1
    tp2_hit = highest_tp >= 2
    realized_r = _realized_r(
        highest_tp=highest_tp,
        target_count=len(targets),
        targets=targets,
        entry=entry_mid,
        stop=stop,
        direction=direction,
        stopped=terminal_status == "stopped",
    )
    return CandidateOutcome(
        valid_fill=valid_fill,
        technical_success=technical_success,
        alert_success=bool(valid_fill and tp1_hit),
        tp1_hit=tp1_hit,
        tp2_hit=tp2_hit,
        invalidated_before_fill=terminal_status == "invalidated" and not valid_fill,
        missed_before_fill=terminal_status == "missed" and not valid_fill,
        expired_before_fill=terminal_status == "expired" and not valid_fill,
        entry_delay_minutes=(
            (entry_time - generated).total_seconds() / 60.0
            if entry_time is not None
            else None
        ),
        trade_duration_minutes=(
            (terminal_at - entry_time).total_seconds() / 60.0
            if entry_time is not None
            else None
        ),
        realized_r=realized_r,
        mfe_r=mfe_r,
        mae_r=mae_r,
        terminal_status=terminal_status,
        terminal_at=terminal_at.isoformat(),
    )


def _realized_r(
    *,
    highest_tp: int,
    target_count: int,
    targets: list[float],
    entry: float,
    stop: float,
    direction: str,
    stopped: bool,
) -> float:
    if target_count <= 0:
        return -1.0 if stopped else 0.0
    allocation = 1.0 / target_count
    risk = max(abs(entry - stop), entry * 1e-9)
    realized = 0.0
    for index in range(min(highest_tp, target_count)):
        move = (
            targets[index] - entry
            if direction == "long"
            else entry - targets[index]
        )
        realized += allocation * move / risk
    if stopped:
        realized -= 1.0 - allocation * min(highest_tp, target_count)
    return float(realized)


def _parse_time(value: Any) -> datetime:
    raw = str(value or "").strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
