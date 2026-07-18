"""Risk management: dynamic leverage, simulated capital, R:R, TP profits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    # Simulation (not a real account)
    simulated_capital: float = 1000.0
    risk_pct: float = 1.0
    risk_amount: float = 0.0
    position_size_units: float = 0.0
    position_size_notional: float = 0.0
    margin_required: float = 0.0
    leverage_suggested: float = 1.0
    leverage_reasoning: str = ""
    potential_profits: List[float] = field(default_factory=list)  # $ at each TP
    potential_profit_pcts: List[float] = field(default_factory=list)  # % of capital
    atr: float = 0.0
    atr_pct: float = 0.0
    quality: str = "poor"
    notes: List[str] = field(default_factory=list)
    invalidation: str = ""
    # Explicit: no hardcoded exchange leverage / balance claims
    is_simulation: bool = True

    @property
    def entry_mid(self) -> float:
        return (self.entry_low + self.entry_high) / 2.0

    @property
    def primary_rr(self) -> float:
        return self.risk_reward[0] if self.risk_reward else 0.0

    def to_primary_setup(self) -> Dict[str, Any]:
        """Compact PRIMARY SETUP block for API reports."""
        tps = self.take_profits
        rrs = self.risk_reward
        profits = self.potential_profits
        return {
            "direction": self.direction,
            "entry_zone": {
                "low": self.entry_low,
                "high": self.entry_high,
                "mid": self.entry_mid,
            },
            "stop_loss": self.stop_loss,
            "tp1": tps[0] if len(tps) > 0 else None,
            "tp2": tps[1] if len(tps) > 1 else None,
            "tp3": tps[2] if len(tps) > 2 else None,
            "rr_tp1": rrs[0] if len(rrs) > 0 else None,
            "rr_tp2": rrs[1] if len(rrs) > 1 else None,
            "rr_tp3": rrs[2] if len(rrs) > 2 else None,
            "invalidation": self.invalidation,
            "quality": self.quality,
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
        }


@dataclass
class ScenarioSet:
    bullish: dict
    base: dict
    bearish: dict


class RiskManager:
    """
    Build trade plans with:
    - structure/ATR stops & TPs
    - dynamic leverage from ATR%, confidence, funding
    - simulated capital + fixed risk % (default 1%)
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        risk_cfg: Optional[RiskConfig] = None,
        simulated_capital: Optional[float] = None,
        risk_pct: Optional[float] = None,
    ) -> None:
        self.risk = risk_cfg or (config.risk if config else RiskConfig())
        # Prefer simulated_capital field; fall back to account_balance alias
        if simulated_capital is not None:
            self.risk.simulated_capital = float(simulated_capital)
        elif getattr(self.risk, "simulated_capital", None) is None:
            # migrate old field
            bal = getattr(self.risk, "account_balance", 1000.0)
            self.risk.simulated_capital = float(bal)
        if risk_pct is not None:
            self.risk.risk_per_trade_pct = float(risk_pct)

    def build_plan(
        self,
        direction: str,
        price: float,
        atr: float,
        structure_stops: Optional[Tuple[Optional[float], Optional[float]]] = None,
        confidence: float = 50.0,
        funding_rate: Optional[float] = None,
        min_rr: Optional[float] = None,
    ) -> TradePlan:
        min_rr = min_rr if min_rr is not None else self.risk.min_rr
        atr = max(safe_float(atr), price * 0.001)
        price = safe_float(price)
        capital = float(getattr(self.risk, "simulated_capital", None) or 1000.0)
        risk_pct = float(self.risk.risk_per_trade_pct or 1.0)
        atr_pct = (atr / price * 100.0) if price else 0.0

        if direction not in ("long", "short"):
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
                leverage_reasoning="Flat bias — no leverage recommendation.",
            )

        stop_mult = float(self.risk.default_stop_atr_mult)
        tp_mults = list(self.risk.default_tp_atr_mults or [1.5, 2.5, 4.0])

        if confidence >= 75:
            stop_mult *= 0.9
        elif confidence < 45:
            stop_mult *= 1.15

        support, resistance = (None, None)
        if structure_stops:
            support, resistance = structure_stops

        buffer = atr * 0.15

        if direction == "long":
            entry_low = price - atr * 0.15
            entry_high = price + atr * 0.05
            if support and support < price:
                stop = support - buffer
            else:
                stop = price - atr * stop_mult
            if price - stop < atr * 0.5:
                stop = price - atr * 0.75
            risk_per_unit = price - stop
            tps = [price + atr * m for m in tp_mults]
            if resistance and resistance > price:
                tps[0] = (
                    min(tps[0], resistance - buffer) if resistance - buffer > price else tps[0]
                )
                tps = sorted(set([tps[0], *tps[1:]]))
            invalidation = f"Close below {format_price(stop, price)} (structure/ATR stop)"
        else:
            entry_low = price - atr * 0.05
            entry_high = price + atr * 0.15
            if resistance and resistance > price:
                stop = resistance + buffer
            else:
                stop = price + atr * stop_mult
            if stop - price < atr * 0.5:
                stop = price + atr * 0.75
            risk_per_unit = stop - price
            tps = [price - atr * m for m in tp_mults]
            if support and support < price:
                tps[0] = max(tps[0], support + buffer) if support + buffer < price else tps[0]
                tps = sorted(set([tps[0], *tps[1:]]), reverse=True)
            invalidation = f"Close above {format_price(stop, price)} (structure/ATR stop)"

        risk_per_unit = max(risk_per_unit, price * 1e-6)
        rrs = [abs(tp - price) / risk_per_unit for tp in tps]

        # --- Dynamic leverage (not hardcoded account leverage) ---
        lev, lev_reason = self._dynamic_leverage(
            atr_pct=atr_pct,
            confidence=confidence,
            funding_rate=funding_rate,
            direction=direction,
        )

        # --- 1% (configurable) risk on simulated capital ---
        # Size so loss at stop ≈ risk_amount, then check margin vs leverage
        risk_amount = capital * (risk_pct / 100.0)
        if confidence < 40:
            risk_amount *= 0.5
            effective_risk_pct = risk_pct * 0.5
        else:
            effective_risk_pct = risk_pct

        units = risk_amount / risk_per_unit
        notional = units * price
        margin_required = notional / max(lev, 1e-9)

        # Cap notional so margin ≤ capital (can't use more than full sim capital as margin)
        if margin_required > capital:
            margin_required = capital
            notional = capital * lev
            units = notional / price
            risk_amount = units * risk_per_unit
            effective_risk_pct = (risk_amount / capital) * 100.0

        # Potential profit at each TP (linear, same size)
        potential_profits = [units * abs(tp - price) for tp in tps]
        potential_pcts = [(p / capital) * 100.0 for p in potential_profits]

        primary_rr = rrs[0] if rrs else 0.0
        quality = self._quality(primary_rr, confidence, min_rr)

        notes: List[str] = [
            f"SIMULATION: ${capital:,.2f} capital, risking {effective_risk_pct:.2f}% "
            f"(${risk_amount:,.2f}) at stop.",
            f"Dynamic leverage ~{lev:.1f}x from ATR {atr_pct:.2f}%, confidence {confidence:.0f}%, "
            f"funding context.",
            f"Position ≈ {units:.6g} units · notional ${notional:,.2f} · margin ${margin_required:,.2f}.",
        ]
        for i, (tp, rr, profit) in enumerate(zip(tps, rrs, potential_profits), 1):
            notes.append(
                f"TP{i} {format_price(tp, price)} → R:R {rr:.2f} · "
                f"sim P/L ~${profit:,.2f} ({potential_pcts[i-1]:+.2f}% of capital)."
            )
        if primary_rr < min_rr:
            notes.append(
                f"R:R to TP1 is {primary_rr:.2f} < min {min_rr:.2f} — prefer tighter entry or skip."
            )
        if confidence < 40:
            notes.append("Low confidence — size halved in simulation.")

        return TradePlan(
            direction=direction,
            entry_low=float(min(entry_low, entry_high)),
            entry_high=float(max(entry_low, entry_high)),
            stop_loss=float(stop),
            take_profits=[float(x) for x in tps],
            risk_reward=[float(x) for x in rrs],
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
            is_simulation=True,
        )

    def _dynamic_leverage(
        self,
        atr_pct: float,
        confidence: float,
        funding_rate: Optional[float],
        direction: str,
    ) -> Tuple[float, str]:
        """
        Map volatility + confidence + funding → suggested leverage.

        High ATR% → lower lev; high confidence → higher lev; extreme funding against
        position → reduce lev.
        Ceiling from config.leverage_ceiling (not a fixed exchange max).
        """
        ceiling = float(getattr(self.risk, "leverage_ceiling", None) or 20.0)
        floor = float(getattr(self.risk, "leverage_floor", None) or 1.0)

        # Base from ATR%: ~1% ATR → 8x, ~3% → 3x, ~5%+ → 1.5x
        if atr_pct <= 0.5:
            base = 12.0
        elif atr_pct <= 1.0:
            base = 8.0
        elif atr_pct <= 2.0:
            base = 5.0
        elif atr_pct <= 3.5:
            base = 3.0
        elif atr_pct <= 5.0:
            base = 2.0
        else:
            base = 1.5

        # Confidence scale 0.55 .. 1.25
        conf_mult = 0.55 + (clamp(confidence, 0, 100) / 100.0) * 0.7

        # Funding: if longs pay a lot and we're long → crowded → cut; etc.
        fund_mult = 1.0
        fund_note = "funding neutral"
        if funding_rate is not None:
            fr_pct = funding_rate * 100.0
            if direction == "long" and fr_pct > 0.05:
                fund_mult = 0.7
                fund_note = f"elevated long funding ({fr_pct:+.4f}%) — cut leverage"
            elif direction == "long" and fr_pct < -0.03:
                fund_mult = 1.1
                fund_note = f"shorts paying ({fr_pct:+.4f}%) — mild squeeze tailwind"
            elif direction == "short" and fr_pct < -0.05:
                fund_mult = 0.7
                fund_note = f"elevated short funding ({fr_pct:+.4f}%) — cut leverage"
            elif direction == "short" and fr_pct > 0.03:
                fund_mult = 1.1
                fund_note = f"longs crowded ({fr_pct:+.4f}%) — mild short tailwind"
            else:
                fund_note = f"funding {fr_pct:+.4f}%"

        lev = base * conf_mult * fund_mult
        lev = float(clamp(lev, floor, ceiling))
        # Nice steps
        lev = round(lev * 2) / 2.0  # 0.5 increments
        lev = max(floor, min(ceiling, lev))

        reason = (
            f"Base {base:.1f}x from ATR {atr_pct:.2f}% of price; "
            f"×{conf_mult:.2f} confidence scale; {fund_note}; "
            f"clamped to [{floor:.1f}x, {ceiling:.1f}x] → {lev:.1f}x."
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
