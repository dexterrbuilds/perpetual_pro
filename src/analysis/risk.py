"""Risk management: intraday perp leverage, multi-TP plans, hold windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config import AppConfig, RiskConfig
from src.utils.helpers import clamp, format_price, safe_float


@dataclass
class TradePlan:
    direction: str  # long | short | flat
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profits: List[float] = field(default_factory=list)
    risk_reward: List[float] = field(default_factory=list)
    # Alternative / secondary entry (limit pullback or breakout add)
    alternative_entry_low: Optional[float] = None
    alternative_entry_high: Optional[float] = None
    alternative_entry_note: str = ""
    # Simulation (not a real account)
    simulated_capital: float = 1000.0
    risk_pct: float = 1.0
    risk_amount: float = 0.0
    position_size_units: float = 0.0
    position_size_notional: float = 0.0
    margin_required: float = 0.0
    leverage_suggested: float = 5.0
    leverage_reasoning: str = ""
    potential_profits: List[float] = field(default_factory=list)  # $ at each TP
    potential_profit_pcts: List[float] = field(default_factory=list)  # % of capital
    atr: float = 0.0
    atr_pct: float = 0.0
    quality: str = "poor"
    notes: List[str] = field(default_factory=list)
    invalidation: str = ""
    # Hold window (scalp / day-trade biased)
    hold_label: str = ""
    hold_detail: str = ""
    hold_hours_max: float = 24.0
    # Explicit: no hardcoded exchange leverage / balance claims
    is_simulation: bool = True
    # Prop account management
    prop_mode: bool = True
    prop_safe: bool = True
    prop_flags: List[str] = field(default_factory=list)
    max_leverage_allowed: float = 5.0
    # Execution quality: prevents treating a directional bias as a market order.
    entry_status: str = "blocked"  # ready | wait_retest | avoid_chase | blocked
    entry_reason: str = ""
    execution_score: float = 0.0
    immediate_sl_risk: float = 100.0
    chase_distance_atr: float = 0.0
    order_flow_score: float = 0.0
    candle_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def entry_mid(self) -> float:
        return (self.entry_low + self.entry_high) / 2.0

    @property
    def primary_rr(self) -> float:
        return self.risk_reward[0] if self.risk_reward else 0.0

    def setup_headline(self, symbol: str) -> str:
        """Pro-style alert: 🚨 BTC SHORT SETUP"""
        base = _symbol_base(symbol)
        if self.direction == "long":
            return f"🚨 {base} LONG SETUP"
        if self.direction == "short":
            return f"🚨 {base} SHORT SETUP"
        return f"⏸ {base} NO TRADE — STAND ASIDE"

    def to_primary_setup(self) -> Dict[str, Any]:
        """Compact PRIMARY SETUP block for API / extension reports."""
        tps = self.take_profits
        rrs = self.risk_reward
        return {
            "headline": None,  # filled by report layer with symbol
            "direction": self.direction,
            "direction_label": self.direction.upper() if self.direction != "flat" else "FLAT",
            "entry_zone": {
                "low": self.entry_low,
                "high": self.entry_high,
                "mid": self.entry_mid,
            },
            "alternative_entry": {
                "low": self.alternative_entry_low,
                "high": self.alternative_entry_high,
                "note": self.alternative_entry_note,
            }
            if self.alternative_entry_low is not None
            else None,
            "stop_loss": self.stop_loss,
            "tp1": tps[0] if len(tps) > 0 else None,
            "tp2": tps[1] if len(tps) > 1 else None,
            "tp3": tps[2] if len(tps) > 2 else None,
            "tp4": tps[3] if len(tps) > 3 else None,
            "take_profits": list(tps),
            "rr_tp1": rrs[0] if len(rrs) > 0 else None,
            "rr_tp2": rrs[1] if len(rrs) > 1 else None,
            "rr_tp3": rrs[2] if len(rrs) > 2 else None,
            "rr_tp4": rrs[3] if len(rrs) > 3 else None,
            "risk_reward": list(rrs),
            "leverage_suggested": self.leverage_suggested,
            "hold_label": self.hold_label,
            "hold_detail": self.hold_detail,
            "hold_hours_max": self.hold_hours_max,
            "invalidation": self.invalidation,
            "quality": self.quality,
            "prop_mode": self.prop_mode,
            "prop_safe": self.prop_safe,
            "prop_flags": list(self.prop_flags),
            "max_leverage_allowed": self.max_leverage_allowed,
            "risk_pct": self.risk_pct,
            "entry_status": self.entry_status,
            "entry_reason": self.entry_reason,
            "execution_score": self.execution_score,
            "immediate_sl_risk": self.immediate_sl_risk,
            "chase_distance_atr": self.chase_distance_atr,
            "order_flow_score": self.order_flow_score,
            "candle_context": dict(self.candle_context),
        }

    def to_position_simulation(self) -> Dict[str, Any]:
        return {
            "is_simulation": True,
            "simulated_capital": self.simulated_capital,
            "risk_pct": self.risk_pct,
            "risk_amount": self.risk_amount,
            "position_size_units": self.position_size_units,
            "position_size_notional": self.position_size_notional,
            "margin_required": self.margin_required,
            "leverage_suggested": self.leverage_suggested,
            "leverage_reasoning": self.leverage_reasoning,
            "potential_profits_usd": self.potential_profits,
            "potential_profit_pct_of_capital": self.potential_profit_pcts,
            "atr": self.atr,
            "atr_pct": self.atr_pct,
            "notes": self.notes,
            "prop_mode": self.prop_mode,
            "prop_safe": self.prop_safe,
            "prop_flags": list(self.prop_flags),
            "max_leverage_allowed": self.max_leverage_allowed,
            "entry_status": self.entry_status,
            "entry_reason": self.entry_reason,
            "execution_score": self.execution_score,
            "immediate_sl_risk": self.immediate_sl_risk,
            "chase_distance_atr": self.chase_distance_atr,
            "order_flow_score": self.order_flow_score,
        }

    def to_pro_lines(self, symbol: str, price: Optional[float] = None) -> List[str]:
        """Human-readable pro trader card lines (terminal / markdown)."""
        ref = price if price is not None else self.entry_mid
        lines = [
            self.setup_headline(symbol),
            f"📍 Direction: {self.direction.upper() if self.direction != 'flat' else 'FLAT'}",
        ]
        if self.direction == "flat":
            lines.append("⛔ No entry — wait for cleaner structure / bias.")
            return lines

        lines.append(
            f"🎯 Entry Zone: {format_price(self.entry_low, ref)} – {format_price(self.entry_high, ref)}"
        )
        lines.append(f"🚦 Entry Status: {self.entry_status.upper()} — {self.entry_reason}")
        if self.alternative_entry_low is not None and self.alternative_entry_high is not None:
            lines.append(
                f"🔄 Alternative Entry: {format_price(self.alternative_entry_low, ref)} – "
                f"{format_price(self.alternative_entry_high, ref)}"
                + (f" ({self.alternative_entry_note})" if self.alternative_entry_note else "")
            )
        lines.append(f"🛑 Stop-Loss: {format_price(self.stop_loss, ref)}")
        for i, (tp, rr) in enumerate(zip(self.take_profits, self.risk_reward), 1):
            emoji = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣"}.get(i, f"T{i}")
            lines.append(
                f"{emoji} TP{i}: {format_price(tp, ref)}  ·  R:R {rr:.2f}"
            )
        lines.append(f"⚡ Suggested Leverage: {self.leverage_suggested:.0f}x (max {self.max_leverage_allowed:.0f}x)")
        lines.append(f"📉 Risk: {self.risk_pct:.2f}% of capital (${self.risk_amount:,.2f})")
        if self.prop_mode:
            safe = "YES" if self.prop_safe else "NO — review flags"
            lines.append(f"🛡 Prop-safe: {safe}")
            if self.prop_flags:
                lines.append(f"⚠ Flags: {', '.join(self.prop_flags)}")
        if self.hold_detail:
            lines.append(f"⏱ {self.hold_detail}")
        if self.invalidation:
            lines.append(f"❌ Invalidation: {self.invalidation}")
        lines.append(f"⭐ Quality: {self.quality.upper()}")
        return lines


@dataclass
class ScenarioSet:
    bullish: dict
    base: dict
    bearish: dict


class RiskManager:
    """
    Build crypto-perp trade plans with prop-aware defaults:
    - structure/ATR stops & TP1–TP4
    - prop mode: risk 0.5–1%, leverage ≤5x, drawdown flags
    - day-trade mode (prop_mode=false): dynamic leverage 10x–30x
    - simulated capital + fixed risk %
    - scalp / day-trade hold windows (max 12–24h unless strong swing)
    """

    # Aggressive band (when prop_mode is false)
    AGGRESSIVE_LEV_FLOOR = 10.0
    AGGRESSIVE_LEV_CEILING = 30.0
    # Prop band
    PROP_LEV_FLOOR = 1.0
    PROP_LEV_CEILING = 5.0
    MIN_PROP_SIGNAL_CONFIDENCE = 60.0
    # Short-term TP stack (ATR multiples) — scalp / day-trade realistic
    DEFAULT_TP_MULTS = [0.7, 1.3, 2.0, 3.0]

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        risk_cfg: Optional[RiskConfig] = None,
        simulated_capital: Optional[float] = None,
        risk_pct: Optional[float] = None,
    ) -> None:
        self.config = config
        self.risk = risk_cfg or (config.risk if config else RiskConfig())
        if simulated_capital is not None:
            self.risk.simulated_capital = float(simulated_capital)
        elif getattr(self.risk, "simulated_capital", None) is None:
            bal = getattr(self.risk, "account_balance", 1000.0)
            self.risk.simulated_capital = float(bal)
        if risk_pct is not None:
            self.risk.risk_per_trade_pct = float(risk_pct)

    @property
    def prop_mode(self) -> bool:
        return bool(getattr(self.risk, "prop_mode", True))

    def _clamp_risk_pct(self, risk_pct: float) -> float:
        """Enforce 0.5–1% band in prop mode."""
        risk_pct = float(risk_pct or 1.0)
        if not self.prop_mode:
            return max(0.1, min(5.0, risk_pct))
        lo = float(getattr(self.risk, "risk_per_trade_min_pct", 0.5) or 0.5)
        hi = float(getattr(self.risk, "risk_per_trade_max_pct", 1.0) or 1.0)
        if lo > hi:
            lo, hi = 0.5, 1.0
        return float(clamp(risk_pct, lo, hi))

    def build_plan(
        self,
        direction: str,
        price: float,
        atr: float,
        structure_stops: Optional[Tuple[Optional[float], Optional[float]]] = None,
        confidence: float = 50.0,
        funding_rate: Optional[float] = None,
        min_rr: Optional[float] = None,
        primary_tf: Optional[str] = None,
        setup_name: str = "",
        strategy_tags: Optional[List[str]] = None,
        execution: Optional[Dict[str, Any]] = None,
    ) -> TradePlan:
        min_rr = min_rr if min_rr is not None else self.risk.min_rr
        atr = max(safe_float(atr), price * 0.001)
        price = safe_float(price)
        capital = float(getattr(self.risk, "simulated_capital", None) or 1000.0)
        risk_pct = self._clamp_risk_pct(float(self.risk.risk_per_trade_pct or 1.0))
        atr_pct = (atr / price * 100.0) if price else 0.0
        tags = list(strategy_tags or [])
        max_lev = (
            float(getattr(self.risk, "max_leverage", None) or self.PROP_LEV_CEILING)
            if self.prop_mode
            else float(getattr(self.risk, "leverage_ceiling", None) or self.AGGRESSIVE_LEV_CEILING)
        )
        if self.prop_mode:
            max_lev = min(max_lev, self.PROP_LEV_CEILING)

        if direction not in ("long", "short"):
            hold_label, hold_detail, hold_max = suggest_hold_window(
                primary_tf, setup_name, tags, direction, confidence
            )
            floor = self.PROP_LEV_FLOOR if self.prop_mode else self.AGGRESSIVE_LEV_FLOOR
            return TradePlan(
                direction="flat",
                entry_low=price,
                entry_high=price,
                stop_loss=price,
                simulated_capital=capital,
                risk_pct=risk_pct,
                atr=atr,
                atr_pct=atr_pct,
                notes=["No trade — wait for clearer bias / structure."],
                quality="poor",
                invalidation="N/A",
                leverage_suggested=floor,
                leverage_reasoning="Flat bias — no leverage recommendation.",
                hold_label=hold_label,
                hold_detail=hold_detail,
                hold_hours_max=hold_max,
                prop_mode=self.prop_mode,
                prop_safe=True,
                prop_flags=["FLAT_NO_TRADE"],
                max_leverage_allowed=max_lev,
                entry_status="blocked",
                entry_reason="No directional setup passed the signal gate.",
            )

        # Tight scalp / day-trade stop (ATR multiples)
        stop_mult = float(self.risk.default_stop_atr_mult or 1.0)
        cfg_tps = list(getattr(self.risk, "default_tp_atr_mults", None) or [])
        if len(cfg_tps) >= 4:
            tp_mults = cfg_tps[:4]
        elif len(cfg_tps) == 3:
            tp_mults = cfg_tps + [cfg_tps[-1] * 1.4]
        else:
            tp_mults = list(self.DEFAULT_TP_MULTS)

        # High conviction → slightly tighter stop; low conf → slightly wider
        if confidence >= 75:
            stop_mult *= 0.85
        elif confidence < 45:
            stop_mult *= 1.1

        support, resistance = (None, None)
        if structure_stops:
            support, resistance = structure_stops

        buffer = atr * 0.08  # tighter micro-structure buffer
        alt_low = alt_high = None
        alt_note = ""
        execution = dict(execution or {})
        use_execution = bool(
            execution.get("entry_low") is not None
            and execution.get("entry_high") is not None
            and execution.get("stop_loss") is not None
        )

        if use_execution:
            entry_low = safe_float(execution["entry_low"])
            entry_high = safe_float(execution["entry_high"])
            stop = safe_float(execution["stop_loss"])
            tps = [
                safe_float(x)
                for x in (execution.get("targets") or [])
                if safe_float(x) > 0
            ]
            if len(tps) < 4:
                entry_ref = (entry_low + entry_high) / 2.0
                sign = 1.0 if direction == "long" else -1.0
                risk_hint = max(abs(entry_ref - stop), atr * 0.7)
                for rr in (0.8, 1.3, 2.0, 2.8):
                    candidate = entry_ref + sign * risk_hint * rr
                    if len(tps) >= 4:
                        break
                    if not tps or (
                        candidate > tps[-1] if direction == "long" else candidate < tps[-1]
                    ):
                        tps.append(candidate)
            tps = tps[:4]
            alt_note = str(execution.get("entry_reason") or "")
            invalidation = str(execution.get("invalidation_reason") or "")
        elif direction == "long":
            # Tight primary zone around price (breakout / momentum entry)
            entry_low = price - atr * 0.06
            entry_high = price + atr * 0.04
            # Alt: retest of demand / OB (not deep swing)
            if support and support < price and (price - support) <= atr * 2.2:
                alt_low = support
                alt_high = min(price - atr * 0.02, support + atr * 0.25)
                if alt_high <= alt_low:
                    alt_high = alt_low + atr * 0.12
                alt_note = "retest of demand / bullish OB"
            else:
                alt_low = price - atr * 0.35
                alt_high = price - atr * 0.12
                alt_note = "shallow pullback retest if primary missed"

            # Prefer micro stop under recent structure; cap distance for scalps
            if support and support < price and (price - support) <= atr * 1.6:
                stop = support - buffer
            else:
                stop = price - atr * stop_mult
            # Enforce min/max stop distance for high-leverage perps
            if price - stop < atr * 0.35:
                stop = price - atr * 0.45
            if price - stop > atr * 1.8:
                stop = price - atr * 1.4
            tps = [price + atr * m for m in tp_mults]
            if resistance and resistance > price:
                near = resistance - buffer
                if near > price:
                    tps[0] = min(tps[0], near)
                tps = sorted(set(tps))
                while len(tps) < 4:
                    tps.append(tps[-1] + atr * 0.55)
            invalidation = f"Close below {format_price(stop, price)} (scalp invalidation)"
        else:
            entry_low = price - atr * 0.04
            entry_high = price + atr * 0.06
            if resistance and resistance > price and (resistance - price) <= atr * 2.2:
                alt_low = max(price + atr * 0.02, resistance - atr * 0.25)
                alt_high = resistance
                if alt_high <= alt_low:
                    alt_low = alt_high - atr * 0.12
                alt_note = "retest of supply / bearish OB"
            else:
                alt_low = price + atr * 0.12
                alt_high = price + atr * 0.35
                alt_note = "shallow premium retest if primary missed"

            if resistance and resistance > price and (resistance - price) <= atr * 1.6:
                stop = resistance + buffer
            else:
                stop = price + atr * stop_mult
            if stop - price < atr * 0.35:
                stop = price + atr * 0.45
            if stop - price > atr * 1.8:
                stop = price + atr * 1.4
            tps = [price - atr * m for m in tp_mults]
            if support and support < price:
                near = support + buffer
                if near < price:
                    tps[0] = max(tps[0], near)
                tps = sorted(set(tps), reverse=True)
                while len(tps) < 4:
                    tps.append(tps[-1] - atr * 0.55)
            invalidation = f"Close above {format_price(stop, price)} (scalp invalidation)"

        entry_reference = (entry_low + entry_high) / 2.0
        risk_per_unit = max(abs(entry_reference - stop), entry_reference * 1e-6)
        rrs = [abs(tp - entry_reference) / risk_per_unit for tp in tps]

        lev, lev_reason = self._dynamic_leverage(
            atr_pct=atr_pct,
            confidence=confidence,
            funding_rate=funding_rate,
            direction=direction,
        )

        risk_amount = capital * (risk_pct / 100.0)
        if confidence < 40:
            cut = risk_pct * 0.5
            if self.prop_mode:
                cut = max(float(getattr(self.risk, "risk_per_trade_min_pct", 0.5) or 0.5), cut)
            risk_amount = capital * (cut / 100.0)
            effective_risk_pct = cut
        else:
            effective_risk_pct = risk_pct
        effective_risk_pct = self._clamp_risk_pct(effective_risk_pct)
        risk_amount = capital * (effective_risk_pct / 100.0)

        units = risk_amount / risk_per_unit
        notional = units * entry_reference
        margin_required = notional / max(lev, 1e-9)

        if margin_required > capital:
            margin_required = capital
            notional = capital * lev
            units = notional / entry_reference
            risk_amount = units * risk_per_unit
            effective_risk_pct = (risk_amount / capital) * 100.0
            if self.prop_mode:
                hi = float(getattr(self.risk, "risk_per_trade_max_pct", 1.0) or 1.0)
                if effective_risk_pct > hi:
                    effective_risk_pct = hi
                    risk_amount = capital * (hi / 100.0)
                    units = risk_amount / risk_per_unit
                    notional = units * entry_reference
                    margin_required = notional / max(lev, 1e-9)

        potential_profits = [units * abs(tp - entry_reference) for tp in tps]
        potential_pcts = [(p / capital) * 100.0 for p in potential_profits]

        primary_rr = rrs[0] if rrs else 0.0
        # TP1 is a partial/de-risk target. Prop quality is evaluated at TP2,
        # where the planned trade is expected to meet its minimum R:R.
        evaluation_rr = rrs[1] if len(rrs) > 1 else primary_rr
        quality = self._quality(evaluation_rr, confidence, min_rr)

        hold_label, hold_detail, hold_max = suggest_hold_window(
            primary_tf, setup_name, tags, direction, confidence
        )

        prop_flags = self._prop_flags(
            atr_pct=atr_pct,
            primary_rr=evaluation_rr,
            min_rr=float(min_rr or 1.2),
            effective_risk_pct=effective_risk_pct,
            confidence=confidence,
            lev=lev,
            entry_status=str(execution.get("status") or "ready"),
            execution_score=safe_float(execution.get("score"), 100.0),
        )
        prop_safe = not any(
            f in prop_flags
            for f in (
                "HIGH_DRAWDOWN_RISK",
                "WIDE_STOP",
                "LOW_RR",
                "LOW_CONFIDENCE",
                "POOR_EXECUTION",
                "AVOID_CHASE",
            )
        )

        mode_note = (
            f"PROP mode: risk {effective_risk_pct:.2f}% (band 0.5–1%), leverage ≤{max_lev:.0f}x."
            if self.prop_mode
            else (
                f"Aggressive perp leverage ~{lev:.0f}x from ATR {atr_pct:.2f}%, "
                f"confidence {confidence:.0f}% (day-trade band 10x–30x)."
            )
        )
        notes: List[str] = [
            f"SIMULATION: ${capital:,.2f} capital, risking {effective_risk_pct:.2f}% "
            f"(${risk_amount:,.2f}) at stop.",
            mode_note,
            f"{hold_detail}",
            f"Position ≈ {units:.6g} units · notional ${notional:,.2f} · margin ${margin_required:,.2f}.",
        ]
        for i, (tp, rr, profit) in enumerate(zip(tps, rrs, potential_profits), 1):
            notes.append(
                f"TP{i} {format_price(tp, price)} → R:R {rr:.2f} · "
                f"sim P/L ~${profit:,.2f} ({potential_pcts[i-1]:+.2f}% of capital)."
            )
        if evaluation_rr < min_rr:
            notes.append(
                f"R:R to TP2 is {evaluation_rr:.2f} < min {min_rr:.2f} — prefer tighter entry or skip."
            )
        if confidence < 40:
            notes.append("Low confidence — size reduced; consider standing aside.")
        if prop_flags:
            notes.append(f"Prop flags: {', '.join(prop_flags)}.")
        if not self.prop_mode and lev >= 50:
            notes.append(
                "High leverage — size small, trail hard after TP1, and respect the stop without exception."
            )

        return TradePlan(
            direction=direction,
            entry_low=float(min(entry_low, entry_high)),
            entry_high=float(max(entry_low, entry_high)),
            stop_loss=float(stop),
            take_profits=[float(x) for x in tps],
            risk_reward=[float(x) for x in rrs],
            alternative_entry_low=float(min(alt_low, alt_high)) if alt_low is not None else None,
            alternative_entry_high=float(max(alt_low, alt_high)) if alt_high is not None else None,
            alternative_entry_note=alt_note,
            simulated_capital=float(capital),
            risk_pct=float(effective_risk_pct),
            risk_amount=float(risk_amount),
            position_size_units=float(units),
            position_size_notional=float(notional),
            margin_required=float(margin_required),
            leverage_suggested=float(lev),
            leverage_reasoning=lev_reason,
            potential_profits=[float(x) for x in potential_profits],
            potential_profit_pcts=[float(x) for x in potential_pcts],
            atr=float(atr),
            atr_pct=float(atr_pct),
            quality=quality,
            notes=notes,
            invalidation=invalidation,
            hold_label=hold_label,
            hold_detail=hold_detail,
            hold_hours_max=hold_max,
            is_simulation=True,
            prop_mode=self.prop_mode,
            prop_safe=prop_safe,
            prop_flags=prop_flags,
            max_leverage_allowed=float(max_lev),
            entry_status=str(execution.get("status") or "ready"),
            entry_reason=str(execution.get("entry_reason") or "Entry zone validated."),
            execution_score=safe_float(execution.get("score"), 100.0),
            immediate_sl_risk=safe_float(execution.get("immediate_sl_risk"), 0.0),
            chase_distance_atr=safe_float(execution.get("chase_distance_atr"), 0.0),
            order_flow_score=safe_float(execution.get("order_flow_score"), 0.0),
            candle_context=dict(execution.get("candle") or {}),
        )

    def apply_prop_confidence_gate(
        self,
        plan: TradePlan,
        confidence: float,
        *,
        minimum: float = MIN_PROP_SIGNAL_CONFIDENCE,
    ) -> TradePlan:
        """Apply the final blended-confidence gate to a drafted trade plan."""
        if not plan.prop_mode or plan.direction not in ("long", "short"):
            return plan
        plan.prop_flags = [flag for flag in plan.prop_flags if flag != "LOW_CONFIDENCE"]
        if confidence < minimum:
            plan.prop_flags.append("LOW_CONFIDENCE")
            plan.notes.append(
                f"Prop gate: {confidence:.0f}% confidence is below the {minimum:.0f}% minimum — no prop signal."
            )
        unsafe = {
            "HIGH_DRAWDOWN_RISK",
            "WIDE_STOP",
            "LOW_RR",
            "LOW_CONFIDENCE",
            "POOR_EXECUTION",
            "AVOID_CHASE",
        }
        plan.prop_flags = list(dict.fromkeys(plan.prop_flags))
        plan.prop_safe = not any(flag in unsafe for flag in plan.prop_flags)
        return plan

    def _prop_flags(
        self,
        *,
        atr_pct: float,
        primary_rr: float,
        min_rr: float,
        effective_risk_pct: float,
        confidence: float,
        lev: float,
        entry_status: str = "ready",
        execution_score: float = 100.0,
    ) -> List[str]:
        flags: List[str] = []
        if not self.prop_mode:
            return flags
        day_warn = float(getattr(self.risk, "daily_drawdown_warn_pct", 3.0) or 3.0)
        if atr_pct > 2.5:
            flags.append("WIDE_STOP")
        if atr_pct > 3.5 or effective_risk_pct >= day_warn * 0.5:
            flags.append("HIGH_DRAWDOWN_RISK")
        if primary_rr < min_rr:
            flags.append("LOW_RR")
        if confidence < self.MIN_PROP_SIGNAL_CONFIDENCE:
            flags.append("LOW_CONFIDENCE")
        if execution_score < 65:
            flags.append("POOR_EXECUTION")
        if entry_status in ("avoid_chase", "blocked"):
            flags.append("AVOID_CHASE")
        if lev >= self.PROP_LEV_CEILING - 1e-9:
            flags.append("LEV_CAPPED_5X")
        return flags

    def _dynamic_leverage(
        self,
        atr_pct: float,
        confidence: float,
        funding_rate: Optional[float],
        direction: str,
    ) -> Tuple[float, str]:
        """Prop: [1x, 5x]. Standard day-trade mode: [10x, 30x]."""
        if self.prop_mode:
            return self._prop_leverage(atr_pct, confidence, funding_rate, direction)

        raw_ceil = float(getattr(self.risk, "leverage_ceiling", None) or self.AGGRESSIVE_LEV_CEILING)
        raw_floor = float(getattr(self.risk, "leverage_floor", None) or self.AGGRESSIVE_LEV_FLOOR)
        ceiling = max(self.AGGRESSIVE_LEV_FLOOR, min(30.0, raw_ceil if raw_ceil >= 10 else 30.0))
        floor = max(self.AGGRESSIVE_LEV_FLOOR, min(ceiling, raw_floor if raw_floor >= 10 else 10.0))
        if floor > ceiling:
            floor, ceiling = self.AGGRESSIVE_LEV_FLOOR, self.AGGRESSIVE_LEV_CEILING

        if atr_pct <= 0.4:
            base = 30.0
        elif atr_pct <= 0.7:
            base = 27.0
        elif atr_pct <= 1.0:
            base = 24.0
        elif atr_pct <= 1.5:
            base = 20.0
        elif atr_pct <= 2.5:
            base = 16.0
        elif atr_pct <= 4.0:
            base = 13.0
        else:
            base = 10.0

        conf_mult = 0.80 + (clamp(confidence, 0, 100) / 100.0) * 0.30
        fund_mult = 1.0
        fund_note = "funding neutral"
        if funding_rate is not None:
            fr_pct = funding_rate * 100.0
            if direction == "long" and fr_pct > 0.05:
                fund_mult = 0.75
                fund_note = f"elevated long funding ({fr_pct:+.4f}%) — cut leverage"
            elif direction == "long" and fr_pct < -0.03:
                fund_mult = 1.12
                fund_note = f"shorts paying ({fr_pct:+.4f}%) — squeeze tailwind"
            elif direction == "short" and fr_pct < -0.05:
                fund_mult = 0.75
                fund_note = f"elevated short funding ({fr_pct:+.4f}%) — cut leverage"
            elif direction == "short" and fr_pct > 0.03:
                fund_mult = 1.12
                fund_note = f"longs crowded ({fr_pct:+.4f}%) — short tailwind"
            else:
                fund_note = f"funding {fr_pct:+.4f}%"

        lev = base * conf_mult * fund_mult
        lev = float(clamp(lev, floor, ceiling))
        lev = max(floor, min(ceiling, round(lev / 2.0) * 2.0))
        if lev < floor:
            lev = floor

        reason = (
            f"Perp leverage base {base:.0f}x from ATR {atr_pct:.2f}% of price; "
            f"×{conf_mult:.2f} confidence; {fund_note}; "
            f"clamped to [{floor:.0f}x–{ceiling:.0f}x] → {lev:.0f}x. "
            f"Intraday 10x–30x band; always size risk first."
        )
        return lev, reason

    def _prop_leverage(
        self,
        atr_pct: float,
        confidence: float,
        funding_rate: Optional[float],
        direction: str,
    ) -> Tuple[float, str]:
        """Conservative prop leverage in [1x, 5x]."""
        ceiling = min(
            self.PROP_LEV_CEILING,
            float(getattr(self.risk, "leverage_ceiling", None) or self.PROP_LEV_CEILING),
            float(getattr(self.risk, "max_leverage", None) or self.PROP_LEV_CEILING),
        )
        floor = max(
            self.PROP_LEV_FLOOR,
            min(ceiling, float(getattr(self.risk, "leverage_floor", None) or self.PROP_LEV_FLOOR)),
        )
        if atr_pct <= 0.5:
            base = 5.0
        elif atr_pct <= 1.0:
            base = 4.0
        elif atr_pct <= 1.8:
            base = 3.0
        elif atr_pct <= 2.5:
            base = 2.5
        else:
            base = 2.0
        conf_mult = 0.85 + (clamp(confidence, 0, 100) / 100.0) * 0.25
        fund_note = "funding neutral"
        if funding_rate is not None:
            fr_pct = funding_rate * 100.0
            if (direction == "long" and fr_pct > 0.05) or (direction == "short" and fr_pct < -0.05):
                conf_mult *= 0.85
                fund_note = f"hostile funding ({fr_pct:+.4f}%)"
            else:
                fund_note = f"funding {fr_pct:+.4f}%"
        lev = float(clamp(base * conf_mult, floor, ceiling))
        lev = max(floor, min(ceiling, round(lev)))
        reason = (
            f"Prop leverage ~{lev:.0f}x (max {ceiling:.0f}x) from ATR {atr_pct:.2f}%, "
            f"confidence {confidence:.0f}%, {fund_note}. Risk first — never max margin."
        )
        return lev, reason

    @staticmethod
    def _quality(rr: float, confidence: float, min_rr: float) -> str:
        if rr < min_rr * 0.8 or confidence < 35:
            return "poor"
        if rr >= min_rr * 1.8 and confidence >= 70:
            return "excellent"
        if rr >= min_rr * 1.3 and confidence >= 55:
            return "good"
        if rr >= min_rr:
            return "acceptable"
        return "poor"

    def scenarios(
        self,
        price: float,
        atr: float,
        bias: str,
        confidence: float,
        support: Optional[float] = None,
        resistance: Optional[float] = None,
    ) -> ScenarioSet:
        atr = max(atr, price * 0.001)
        bull = {
            "name": "Bullish",
            "probability": _scenario_prob(bias, confidence, "bullish"),
            "trigger": f"Hold above {format_price(support or price - atr, price)} and reclaim momentum",
            "target": format_price(resistance or price + 2 * atr, price),
            "invalidation": format_price((support or price) - atr * 0.8, price),
            "narrative": "Trend continuation / breakout if higher TFs align and funding not extreme long.",
        }
        base = {
            "name": "Base",
            "probability": _scenario_prob(bias, confidence, "base"),
            "trigger": "Range between nearest demand/supply; mean reversion dominates",
            "target": format_price(price, price),
            "invalidation": f"ATR expansion > {format_price(atr * 2, price)} range break",
            "narrative": "Chop until BOS; trade edges only with tight risk.",
        }
        bear = {
            "name": "Bearish",
            "probability": _scenario_prob(bias, confidence, "bearish"),
            "trigger": f"Lose {format_price(support or price - atr * 0.5, price)} with volume",
            "target": format_price(support - atr if support else price - 2 * atr, price),
            "invalidation": format_price((resistance or price) + atr * 0.8, price),
            "narrative": "Breakdown / distribution if OI rises into selloff and HTF resistance holds.",
        }
        return ScenarioSet(bullish=bull, base=base, bearish=bear)


def suggest_hold_window(
    primary_tf: Optional[str],
    setup_name: str = "",
    strategy_tags: Optional[List[str]] = None,
    direction: str = "",
    confidence: float = 50.0,
) -> Tuple[str, str, float]:
    """
    Crypto perp day-trade / scalp holds.
    Every suggested window stays inside the 30-minute to 24-hour mandate.
    """
    t = (primary_tf or "").strip().lower()
    tags = [str(x).lower() for x in (strategy_tags or [])]
    blob = f"{setup_name or ''} {' '.join(tags)} {direction or ''}".lower()

    if t in ("1m", "3m", "5m") or "scalp" in blob or "momentum_scalp" in blob:
        return "Scalp", "Suggested hold: 30–90 minutes (scalp — max 2h)", 2.0
    if t in ("15m", "30m") or "mean_reversion" in blob:
        return "Intraday", "Suggested hold: 1–8 hours (day trade — max 12h)", 12.0
    if t in ("1h", "2h"):
        return "Day trade", "Suggested hold: 4–12 hours (day trade — max 24h)", 24.0
    if t in ("4h", "6h", "8h", "12h"):
        return "Day trade", "Suggested hold: 8–24 hours (flat by next session if thesis weak)", 24.0
    if t in ("1d", "3d", "1w") or t in ("d", "w"):
        return "Day trade", "Suggested hold: 12–24 hours — reassess; not a default swing tool", 24.0
    return "Day trade", "Suggested hold: 4–24 hours (default day-trade window)", 24.0


def _symbol_base(symbol: str) -> str:
    s = (symbol or "").upper().replace(" ", "")
    if "/" in s:
        return s.split("/")[0]
    for q in ("USDT", "USDC", "USD", "BUSD"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s or "PAIR"


def _scenario_prob(bias: str, confidence: float, which: str) -> float:
    conf = clamp(confidence, 0, 100) / 100.0
    if bias in ("bullish", "long"):
        weights = {
            "bullish": 0.45 + 0.25 * conf,
            "base": 0.35 - 0.1 * conf,
            "bearish": 0.2 - 0.15 * conf,
        }
    elif bias in ("bearish", "short"):
        weights = {
            "bearish": 0.45 + 0.25 * conf,
            "base": 0.35 - 0.1 * conf,
            "bullish": 0.2 - 0.15 * conf,
        }
    else:
        weights = {"base": 0.5, "bullish": 0.25, "bearish": 0.25}
    s = sum(max(0.05, w) for w in weights.values())
    return round(100.0 * max(0.05, weights[which]) / s, 1)
