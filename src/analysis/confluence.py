"""Weighted multi-factor confluence engine — senior prop trader scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.indicators import IndicatorSuite, compute_indicators
from src.analysis.market_structure import MarketStructureAnalyzer, StructureReport
from src.analysis.patterns import PatternDetector, PatternReport
from src.analysis.risk import RiskManager, ScenarioSet, TradePlan
from src.data.exchange import MarketSnapshot
from src.data.multi_tf import MultiTimeframeData
from src.data.news import NewsBundle
from src.utils.config import AppConfig
from src.utils.helpers import clamp, format_price, safe_float, utc_now_iso


@dataclass
class FactorScore:
    name: str
    score: float  # -1 .. +1
    weight: float
    detail: str = ""

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class FullAnalysis:
    symbol: str
    exchange_id: str
    primary_tf: str
    generated_at: str = field(default_factory=utc_now_iso)

    # Core outputs
    bias: str = "neutral"  # bullish | bearish | neutral
    direction: str = "flat"  # long | short | flat
    confidence: float = 0.0  # 0-100
    setup_name: str = ""
    strategy_tags: List[str] = field(default_factory=list)

    factors: List[FactorScore] = field(default_factory=list)
    confluence_total: float = 0.0  # weighted -1..+1

    indicators: Optional[IndicatorSuite] = None
    patterns: Optional[PatternReport] = None
    structure: Optional[StructureReport] = None
    trade_plan: Optional[TradePlan] = None
    scenarios: Optional[ScenarioSet] = None
    news: Optional[NewsBundle] = None
    snapshot: Optional[MarketSnapshot] = None

    multi_tf_notes: List[str] = field(default_factory=list)
    derivatives_notes: List[str] = field(default_factory=list)
    key_levels: List[Dict[str, Any]] = field(default_factory=list)
    trader_commentary: str = ""
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def factor_breakdown(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": f.name,
                "score": round(f.score, 3),
                "weight": f.weight,
                "weighted": round(f.weighted, 3),
                "detail": f.detail,
            }
            for f in self.factors
        ]


class ConfluenceEngine:
    """
    Orchestrates indicator, structure, pattern, derivatives, MTF, and news
    into a single directional bias with trade plan.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.patterns = PatternDetector()
        self.structure = MarketStructureAnalyzer()
        self.risk = RiskManager(config=config)

    def analyze(
        self,
        mtf: MultiTimeframeData,
        news: Optional[NewsBundle] = None,
        account_balance: Optional[float] = None,
        risk_pct: Optional[float] = None,
    ) -> FullAnalysis:
        if account_balance is not None or risk_pct is not None:
            self.risk = RiskManager(
                config=self.config,
                account_balance=account_balance,
                risk_pct=risk_pct,
            )

        primary = mtf.primary
        result = FullAnalysis(
            symbol=mtf.symbol,
            exchange_id=mtf.exchange_id,
            primary_tf=mtf.primary_tf,
            snapshot=mtf.snapshot,
            news=news,
        )

        if primary is None or primary.empty:
            result.warnings.append("No primary timeframe data — cannot analyze.")
            result.trader_commentary = "Data unavailable. Check symbol/exchange and retry."
            return result

        # --- Indicators ---
        ind = compute_indicators(primary, config=self.config)
        result.indicators = ind
        if ind.errors:
            result.warnings.extend(ind.errors)

        atr = safe_float(ind.summary.get("atr"), primary["high"].iloc[-14:].sub(primary["low"].iloc[-14:]).mean())
        price = safe_float(ind.summary.get("close"), primary["close"].iloc[-1])
        if mtf.snapshot and mtf.snapshot.last:
            price = mtf.snapshot.last

        # --- Patterns & structure ---
        pat = self.patterns.detect(primary)
        result.patterns = pat
        struct = self.structure.analyze(primary, atr=atr)
        result.structure = struct

        # --- Higher TF ---
        htf_score, htf_notes = self._multi_tf_score(mtf)
        result.multi_tf_notes = htf_notes

        # --- Derivatives ---
        der_score, der_notes = self._derivatives_score(mtf.snapshot)
        result.derivatives_notes = der_notes

        # --- News ---
        news_score = 0.0
        news_detail = "No news"
        if news and news.items:
            news_score = float(clamp(news.aggregate_sentiment, -1, 1))
            news_detail = news.summary
        elif news:
            news_detail = news.summary or "No headlines"

        w = self.config.analysis.weights
        factors = [
            FactorScore("trend", float(ind.summary.get("trend_score", 0)), w.trend, self._trend_detail(ind, struct)),
            FactorScore("momentum", float(ind.summary.get("momentum_score", 0)), w.momentum, self._mom_detail(ind)),
            FactorScore("structure", float(struct.structure_score), w.structure, struct.summary),
            FactorScore(
                "patterns",
                float(clamp(0.55 * pat.candle_bias + 0.45 * pat.classical_bias, -1, 1)),
                w.patterns,
                pat.summary,
            ),
            FactorScore("derivatives", der_score, w.derivatives, "; ".join(der_notes) or "n/a"),
            FactorScore("multi_tf", htf_score, w.multi_tf, "; ".join(htf_notes[:3]) or "n/a"),
            FactorScore("volume", float(ind.summary.get("volume_bias", 0)), w.volume, self._vol_detail(ind)),
            FactorScore("news", news_score, w.news, news_detail[:200]),
        ]

        # Divergence adjustment inside momentum factor
        if ind.divergences:
            div_adj = 0.0
            for d in ind.divergences:
                div_adj += 0.12 if d.kind == "bullish" else -0.12
            factors[1].score = float(clamp(factors[1].score + div_adj, -1, 1))
            factors[1].detail += f" | Divergences: {len(ind.divergences)}"

        result.factors = factors
        total = sum(f.weighted for f in factors)
        # Normalize if weights don't sum to 1
        wsum = sum(f.weight for f in factors) or 1.0
        total = total / wsum
        result.confluence_total = float(clamp(total, -1, 1))

        # Bias & confidence
        bias, direction, conf = self._bias_from_score(result.confluence_total, ind, struct)
        conf = float(
            clamp(
                conf,
                self.config.analysis.min_confidence,
                self.config.analysis.max_confidence,
            )
        )
        result.bias = bias
        result.direction = direction
        result.confidence = conf

        # Strategy tags (prop-style playbook)
        result.strategy_tags = self._strategy_tags(ind, struct, pat, mtf, direction, conf)
        result.setup_name = self._setup_name(result.strategy_tags, direction, struct)

        # Key levels for plan
        support, resistance = self._nearest_levels(struct, price)
        result.key_levels = self._collect_levels(struct, price)

        # Trade plan
        plan = self.risk.build_plan(
            direction=direction,
            price=price,
            atr=atr,
            structure_stops=(support, resistance),
            confidence=conf,
        )
        result.trade_plan = plan
        result.scenarios = self.risk.scenarios(
            price=price,
            atr=atr,
            bias=bias,
            confidence=conf,
            support=support,
            resistance=resistance,
        )

        result.trader_commentary = self._commentary(result)
        result.meta = {
            "price": price,
            "atr": atr,
            "indicator_count": ind.indicator_count,
            "bars_primary": len(primary),
            "timeframes": mtf.all_timeframes(),
        }
        logger.info(
            "Analysis {} {} bias={} conf={:.1f}% score={:.3f}",
            result.symbol,
            result.primary_tf,
            result.bias,
            result.confidence,
            result.confluence_total,
        )
        return result

    # ------------------------------------------------------------------
    def _multi_tf_score(self, mtf: MultiTimeframeData) -> Tuple[float, List[str]]:
        notes: List[str] = []
        scores: List[float] = []
        weights: List[float] = []

        for tf, df in mtf.frames.items():
            if df is None or df.empty or len(df) < 30:
                continue
            ind = compute_indicators(df, config=self.config)
            s = float(ind.summary.get("trend_score", 0))
            # Weight higher TFs more
            from src.utils.helpers import timeframe_to_minutes

            mins = timeframe_to_minutes(tf)
            w = 1.0 + np.log1p(mins / 15.0)
            scores.append(s)
            weights.append(w)
            rsi = ind.summary.get("rsi")
            rsi_s = f" RSI={rsi:.1f}" if rsi is not None else ""
            notes.append(f"{tf}: trend={s:+.2f}{rsi_s}")

        if not scores:
            return 0.0, ["No multi-TF data"]
        total_w = sum(weights) or 1.0
        agg = sum(s * w for s, w in zip(scores, weights)) / total_w

        # Alignment bonus/penalty
        signs = [np.sign(s) for s in scores if abs(s) > 0.1]
        if signs and all(x == signs[0] for x in signs):
            notes.insert(0, "All timeframes aligned")
            agg = clamp(agg * 1.1, -1, 1)
        elif signs:
            notes.insert(0, "Timeframe conflict — reduce size")
            agg *= 0.7

        return float(clamp(agg, -1, 1)), notes

    def _derivatives_score(self, snap: Optional[MarketSnapshot]) -> Tuple[float, List[str]]:
        notes: List[str] = []
        if not snap:
            return 0.0, ["Derivatives snapshot unavailable"]

        score = 0.0
        fr = snap.funding_rate
        if fr is not None:
            fr_pct = fr * 100
            notes.append(f"Funding {fr_pct:+.4f}%")
            # Extreme positive funding → crowded long → mild bearish lean
            if fr_pct > 0.05:
                score -= 0.35
                notes.append("Elevated long funding — squeeze risk")
            elif fr_pct > 0.02:
                score -= 0.15
            elif fr_pct < -0.05:
                score += 0.35
                notes.append("Elevated short funding — short-squeeze risk")
            elif fr_pct < -0.02:
                score += 0.15
        else:
            notes.append("Funding n/a")

        if snap.open_interest is not None:
            notes.append(f"OI {snap.open_interest:,.0f}")
            # Without history we only note level; mild neutrality
        if snap.long_short_ratio is not None:
            lsr = snap.long_short_ratio
            notes.append(f"L/S ratio {lsr:.3f}")
            if lsr > 1.8:
                score -= 0.2
                notes.append("Crowded longs (L/S high)")
            elif lsr < 0.7:
                score += 0.2
                notes.append("Crowded shorts (L/S low)")

        if snap.percentage_24h is not None:
            notes.append(f"24h {snap.percentage_24h:+.2f}%")
            # Fade extreme extensions slightly
            if snap.percentage_24h > 12:
                score -= 0.1
            elif snap.percentage_24h < -12:
                score += 0.1

        return float(clamp(score, -1, 1)), notes

    def _bias_from_score(
        self,
        total: float,
        ind: IndicatorSuite,
        struct: StructureReport,
    ) -> Tuple[str, str, float]:
        # Soft thresholds
        if total >= 0.12:
            bias, direction = "bullish", "long"
        elif total <= -0.12:
            bias, direction = "bearish", "short"
        else:
            bias, direction = "neutral", "flat"

        # Confidence from magnitude + ADX (trend clarity)
        adx = ind.summary.get("adx")
        clarity = 1.0
        if adx is not None:
            if adx < 15:
                clarity = 0.75
            elif adx > 25:
                clarity = 1.1

        conf = min(92.0, abs(total) * 100 * 1.15 * clarity + 20)
        if direction == "flat":
            conf = min(conf, 48)

        # Structure conflict penalty
        if struct.trend == "up" and direction == "short":
            conf *= 0.85
        if struct.trend == "down" and direction == "long":
            conf *= 0.85

        return bias, direction, float(conf)

    def _strategy_tags(
        self,
        ind: IndicatorSuite,
        struct: StructureReport,
        pat: PatternReport,
        mtf: MultiTimeframeData,
        direction: str,
        conf: float,
    ) -> List[str]:
        tags: List[str] = []
        adx = ind.summary.get("adx") or 0
        rsi = ind.summary.get("rsi")
        trend = float(ind.summary.get("trend_score", 0))
        bb_pos = float(ind.summary.get("bb_position", 0.5))

        if adx >= 25 and abs(trend) > 0.25:
            tags.append("trend_following")
        if adx < 20 and 0.2 < bb_pos < 0.8:
            tags.append("mean_reversion")
        if struct.last_bos:
            tags.append("breakout" if struct.last_bos == "bullish" else "breakdown")
        if abs(float(ind.summary.get("momentum_score", 0))) > 0.35:
            tags.append("momentum")
        if any(p.bias in ("bullish", "bearish") and p.kind == "candlestick" for p in pat.hits[:3]):
            if any(x in (p.name.lower() for p in pat.hits[:3]) for x in ("engulf", "star", "hammer", "shooting")):
                tags.append("reversal")
        if "accumulation" in struct.wyckoff_phase or "distribution" in struct.wyckoff_phase:
            tags.append("wyckoff")
        if "impulse" in struct.elliott_notes.lower():
            tags.append("elliott_impulse")
        elif "corrective" in struct.elliott_notes.lower():
            tags.append("elliott_corrective")

        from src.utils.helpers import timeframe_to_minutes

        if timeframe_to_minutes(mtf.primary_tf) <= 15:
            tags.append("scalping")
        if timeframe_to_minutes(mtf.primary_tf) >= 60:
            tags.append("swing")
        if timeframe_to_minutes(mtf.primary_tf) >= 240:
            tags.append("position")

        if direction == "flat":
            tags.append("wait")
        if conf >= 70:
            tags.append("high_conviction")

        # Volume profile context
        if struct.volume_profile_poc:
            tags.append("volume_profile")

        # Unique preserve order
        seen = set()
        out = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _setup_name(self, tags: List[str], direction: str, struct: StructureReport) -> str:
        if direction == "flat":
            return "No Trade / Observe"
        side = "Long" if direction == "long" else "Short"
        if "breakout" in tags or "breakdown" in tags:
            return f"{side} Break of Structure"
        if "wyckoff" in tags and "accumulation" in struct.wyckoff_phase:
            return f"{side} Wyckoff Accumulation Continuation"
        if "wyckoff" in tags and "distribution" in struct.wyckoff_phase:
            return f"{side} Wyckoff Distribution Continuation"
        if "mean_reversion" in tags:
            return f"{side} Mean Reversion to Value"
        if "trend_following" in tags:
            return f"{side} Trend Pullback"
        if "reversal" in tags:
            return f"{side} Reversal Setup"
        return f"{side} Confluence Setup"

    def _nearest_levels(
        self, struct: StructureReport, price: float
    ) -> Tuple[Optional[float], Optional[float]]:
        supports: List[float] = []
        resistances: List[float] = []
        for lvl in struct.levels:
            mid = lvl.mid
            if lvl.side == "bullish" or lvl.kind in ("order_block",) and lvl.side != "bearish":
                if mid < price:
                    supports.append(mid)
            if lvl.side == "bearish" or (lvl.kind == "order_block" and lvl.side == "bearish"):
                if mid > price:
                    resistances.append(mid)
            if lvl.kind == "liquidity" and mid > price:
                resistances.append(mid)
            if lvl.kind == "liquidity" and mid < price:
                supports.append(mid)

        for s in struct.swing_lows:
            if s < price:
                supports.append(s)
        for s in struct.swing_highs:
            if s > price:
                resistances.append(s)

        if struct.volume_profile_val and struct.volume_profile_val < price:
            supports.append(struct.volume_profile_val)
        if struct.volume_profile_vah and struct.volume_profile_vah > price:
            resistances.append(struct.volume_profile_vah)

        support = max(supports) if supports else None
        resistance = min(resistances) if resistances else None
        return support, resistance

    def _collect_levels(self, struct: StructureReport, price: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for lvl in struct.levels[:20]:
            rows.append(
                {
                    "kind": lvl.kind,
                    "side": lvl.side,
                    "low": lvl.price_low,
                    "high": lvl.price_high,
                    "mid": lvl.mid,
                    "confidence": lvl.confidence,
                    "note": lvl.note,
                    "distance_pct": (lvl.mid - price) / price * 100 if price else 0,
                }
            )
        rows.sort(key=lambda r: abs(r["distance_pct"]))
        return rows[:12]

    def _trend_detail(self, ind: IndicatorSuite, struct: StructureReport) -> str:
        s = ind.summary
        return (
            f"EMA stack score {s.get('trend_score', 0):+.2f}; "
            f"structure trend={struct.trend}; ADX={s.get('adx')}"
        )

    def _mom_detail(self, ind: IndicatorSuite) -> str:
        s = ind.summary
        return f"RSI={s.get('rsi')} MACD_hist={s.get('macd_hist')} score={s.get('momentum_score')}"

    def _vol_detail(self, ind: IndicatorSuite) -> str:
        s = ind.summary
        return f"vol_ratio={s.get('vol_ratio')} CMF={s.get('cmf')} MFI={s.get('mfi')}"

    def _commentary(self, a: FullAnalysis) -> str:
        """Senior trader style narrative."""
        parts: List[str] = []
        price = a.meta.get("price")
        parts.append(
            f"On {a.primary_tf}, {a.symbol} prints a {a.bias} lean "
            f"({a.confidence:.0f}% confidence) via weighted confluence "
            f"{a.confluence_total:+.3f}."
        )
        if a.structure:
            parts.append(a.structure.summary + ".")
            if a.structure.wyckoff_notes:
                parts.append(a.structure.wyckoff_notes)
            if a.structure.elliott_notes:
                parts.append(a.structure.elliott_notes)
        if a.patterns and a.patterns.hits:
            top = a.patterns.hits[0]
            parts.append(
                f"Pattern focus: {top.name} ({top.bias}, {top.confidence:.0f}% conf)."
            )
        if a.derivatives_notes:
            parts.append("Derivatives: " + "; ".join(a.derivatives_notes[:4]) + ".")
        if a.multi_tf_notes:
            parts.append("MTF: " + a.multi_tf_notes[0] + ".")
        if a.news and a.news.bias != "neutral":
            parts.append(f"News flow tilts {a.news.bias}.")
        if a.trade_plan and a.trade_plan.direction != "flat":
            tp = a.trade_plan
            parts.append(
                f"Playbook: {a.setup_name}. Entry {format_price(tp.entry_low, price)}–"
                f"{format_price(tp.entry_high, price)}, SL {format_price(tp.stop_loss, price)}, "
                f"R:R TP1 {tp.primary_rr:.2f} ({tp.quality})."
            )
        else:
            parts.append("No clean asymmetric trade — stand aside or scalp only with tiny size.")
        parts.append("This is analytical output, not financial advice.")
        return " ".join(parts)
