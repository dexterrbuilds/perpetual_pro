"""Risk management: stops, targets, position sizing, R:R quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    position_size_units: float = 0.0
    position_size_notional: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    leverage_suggested: float = 1.0
    quality: str = "poor"  # poor | acceptable | good | excellent
    notes: List[str] = field(default_factory=list)
    invalidation: str = ""

    @property
    def entry_mid(self) -> float:
        return (self.entry_low + self.entry_high) / 2.0

    @property
    def primary_rr(self) -> float:
        return self.risk_reward[0] if self.risk_reward else 0.0


@dataclass
class ScenarioSet:
    bullish: dict
    base: dict
    bearish: dict


class RiskManager:
    """Build trade plans and position sizes from structure + ATR."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        risk_cfg: Optional[RiskConfig] = None,
        account_balance: Optional[float] = None,
        risk_pct: Optional[float] = None,
    ) -> None:
        self.risk = risk_cfg or (config.risk if config else RiskConfig())
        if account_balance is not None:
            self.risk.account_balance = account_balance
        if risk_pct is not None:
            self.risk.risk_per_trade_pct = risk_pct

    def build_plan(
        self,
        direction: str,
        price: float,
        atr: float,
        structure_stops: Optional[Tuple[Optional[float], Optional[float]]] = None,
        confidence: float = 50.0,
        min_rr: Optional[float] = None,
    ) -> TradePlan:
        """
        structure_stops: optional (support_level, resistance_level) for SL placement.
        direction: long | short | flat
        """
        min_rr = min_rr if min_rr is not None else self.risk.min_rr
        atr = max(safe_float(atr), price * 0.001)
        price = safe_float(price)

        if direction not in ("long", "short"):
            return TradePlan(
                direction="flat",
                entry_low=price,
                entry_high=price,
                stop_loss=price,
                notes=["No trade — wait for clearer bias / structure."],
                quality="poor",
                invalidation="N/A",
            )

        stop_mult = self.risk.default_stop_atr_mult
        tp_mults = list(self.risk.default_tp_atr_mults)

        # Tighter stops on higher confidence, wider on low
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
            # Prefer structure stop under support
            if support and support < price:
                stop = support - buffer
            else:
                stop = price - atr * stop_mult
            # Ensure minimum distance
            if price - stop < atr * 0.5:
                stop = price - atr * 0.75
            risk_per_unit = price - stop
            tps = [price + atr * m for m in tp_mults]
            if resistance and resistance > price:
                # Align TP1 toward resistance if closer than TP2
                tps[0] = min(tps[0] * 1.0, resistance - buffer) if resistance - buffer > price else tps[0]
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

        risk_amount = self.risk.account_balance * (self.risk.risk_per_trade_pct / 100.0)
        units = risk_amount / risk_per_unit
        notional = units * price
        lev = notional / max(self.risk.account_balance, 1e-9)
        lev = float(clamp(lev, 1.0, float(self.risk.max_leverage)))
        # Recalc units if leverage capped
        max_notional = self.risk.account_balance * lev
        if notional > max_notional:
            notional = max_notional
            units = notional / price
            risk_amount = units * risk_per_unit

        primary_rr = rrs[0] if rrs else 0.0
        quality = self._quality(primary_rr, confidence, min_rr)

        notes: List[str] = []
        notes.append(
            f"Risking {self.risk.risk_per_trade_pct:.2f}% (${risk_amount:,.2f}) of "
            f"${self.risk.account_balance:,.2f} account."
        )
        if primary_rr < min_rr:
            notes.append(
                f"R:R to TP1 is {primary_rr:.2f} < min {min_rr:.2f} — consider passing or tighter entry."
            )
        else:
            notes.append(f"Primary R:R {primary_rr:.2f} meets minimum {min_rr:.2f}.")
        notes.append(f"Suggested leverage cap ~{lev:.1f}x (config max {self.risk.max_leverage}x).")
        if confidence < 40:
            notes.append("Low confidence — half-size or skip recommended.")
            units *= 0.5
            notional *= 0.5
            risk_amount *= 0.5

        return TradePlan(
            direction=direction,
            entry_low=float(min(entry_low, entry_high)),
            entry_high=float(max(entry_low, entry_high)),
            stop_loss=float(stop),
            take_profits=[float(x) for x in tps],
            risk_reward=[float(x) for x in rrs],
            position_size_units=float(units),
            position_size_notional=float(notional),
            risk_amount=float(risk_amount),
            risk_pct=float(self.risk.risk_per_trade_pct if confidence >= 40 else self.risk.risk_per_trade_pct * 0.5),
            leverage_suggested=float(lev),
            quality=quality,
            notes=notes,
            invalidation=invalidation,
        )

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
        """Bullish / base / bearish narrative scenarios for the report."""
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
    """Soft probabilities for three scenarios that sum ~100."""
    conf = clamp(confidence, 0, 100) / 100.0
    if bias in ("bullish", "long"):
        weights = {"bullish": 0.45 + 0.25 * conf, "base": 0.35 - 0.1 * conf, "bearish": 0.2 - 0.15 * conf}
    elif bias in ("bearish", "short"):
        weights = {"bearish": 0.45 + 0.25 * conf, "base": 0.35 - 0.1 * conf, "bullish": 0.2 - 0.15 * conf}
    else:
        weights = {"base": 0.5, "bullish": 0.25, "bearish": 0.25}
    # normalize
    s = sum(max(0.05, w) for w in weights.values())
    return round(100.0 * max(0.05, weights[which]) / s, 1)
