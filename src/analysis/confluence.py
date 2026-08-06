"""Weighted multi-factor confluence engine — senior prop trader scoring."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.analytics.rejection import evaluate_analysis_gates
from src.analysis.execution import ExecutionProfile, build_execution_profile
from src.analysis.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    RANK_POLICY_VERSION,
    deterministic_rank_score,
)
from src.analysis.indicators import IndicatorSuite, compute_indicators
from src.analysis.legacy_v2 import execution_aware_legacy_confidence
from src.analysis.llm import (
    LLMNarrative,
    NarrativeLLM,
    heuristic_llm_confidence,
)
from src.analysis.market_structure import MarketStructureAnalyzer, StructureReport
from src.analysis.patterns import PatternDetector, PatternReport
from src.analysis.risk import RiskManager, ScenarioSet, TradePlan
from src.analysis.setups import detect_strict_setup
from src.data.exchange import MarketSnapshot
from src.data.multi_tf import MultiTimeframeData
from src.data.news import NewsBundle
from src.utils.config import AppConfig
from src.utils.helpers import clamp, format_price, safe_float, timeframe_to_minutes, utc_now_iso


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
    confidence: float = 0.0  # 0-100 (technical, pre-LLM blend)
    technical_confidence: float = 0.0  # pure technical score
    llm_confidence: float = 0.0  # narrative context score; never alert authority
    llm_confidence_reason: str = ""
    llm_confidence_detail: Dict[str, Any] = field(default_factory=dict)
    rank_score: float = 0.0  # deterministic technical/confluence/execution rank
    setup_name: str = ""
    strategy_tags: List[str] = field(default_factory=list)

    factors: List[FactorScore] = field(default_factory=list)
    confluence_total: float = 0.0  # weighted -1..+1

    indicators: Optional[IndicatorSuite] = None
    patterns: Optional[PatternReport] = None
    structure: Optional[StructureReport] = None
    trade_plan: Optional[TradePlan] = None
    execution: Optional[ExecutionProfile] = None
    scenarios: Optional[ScenarioSet] = None
    news: Optional[NewsBundle] = None
    snapshot: Optional[MarketSnapshot] = None

    multi_tf_notes: List[str] = field(default_factory=list)
    derivatives_notes: List[str] = field(default_factory=list)
    key_levels: List[Dict[str, Any]] = field(default_factory=list)
    trader_commentary: str = ""
    key_reasons: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    llm: Optional[LLMNarrative] = None
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


def deterministic_narrative_eligible(
    *,
    use_llm: bool,
    direction: str,
    technical_quality: float,
    overall_quality: float,
    confluence_total: float,
    execution: ExecutionProfile,
    data_quality_ok: bool,
    prop_safe: bool,
    has_feasible_target: bool,
    confidence_floor: float,
    score_floor: float,
    execution_floor: float,
    max_immediate_sl_risk: float,
    min_net_rr: float = 1.25,
) -> bool:
    """Whether a deterministic signal merits optional narrative enrichment.

    This is deliberately the production signal gate plus explicit hard-failure
    and target checks. It cannot alter any deterministic score or eligibility;
    it only prevents external LLM work for candidates already rejected by the
    trading engine.
    """
    decision = evaluate_analysis_gates(
        direction=direction,
        technical_quality=technical_quality,
        execution_quality=execution.score,
        overall_quality=overall_quality,
        confluence_magnitude=abs(confluence_total),
        entry_status=execution.status,
        immediate_sl_risk=execution.immediate_sl_risk,
        market_quality_ok=execution.market_quality_ok,
        data_quality_ok=data_quality_ok,
        prop_safe=prop_safe,
        confidence_floor=confidence_floor,
        score_floor=score_floor,
        execution_floor=execution_floor,
        max_immediate_sl_risk=max_immediate_sl_risk,
        hard_failures=execution.hard_failures,
        net_rr=(
            execution.net_risk_reward[1]
            if len(execution.net_risk_reward) > 1
            else (execution.net_risk_reward[0] if execution.net_risk_reward else None)
        ),
        min_net_rr=min_net_rr,
    )
    return bool(
        use_llm
        and decision.eligible
        and technical_quality >= confidence_floor
        and has_feasible_target
        and not execution.hard_failures
    )


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
        simulated_capital: Optional[float] = None,
        risk_pct: Optional[float] = None,
        account_balance: Optional[float] = None,  # legacy alias
        use_llm: bool = True,
    ) -> FullAnalysis:
        capital = simulated_capital if simulated_capital is not None else account_balance
        if capital is not None or risk_pct is not None:
            self.risk = RiskManager(
                config=self.config,
                simulated_capital=capital,
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
        # Compute every timeframe once. The primary suite is reused by the live
        # score and watchlist backtest; higher suites feed MTF alignment.
        indicator_suites: Dict[str, IndicatorSuite] = {}
        for tf, frame in mtf.frames.items():
            if frame is not None and not frame.empty:
                indicator_suites[tf] = compute_indicators(frame, config=self.config)
        ind = indicator_suites.get(mtf.primary_tf) or compute_indicators(
            primary, config=self.config
        )
        result.indicators = ind
        if ind.errors:
            result.warnings.extend(ind.errors)

        atr = safe_float(ind.summary.get("atr"), primary["high"].iloc[-14:].sub(primary["low"].iloc[-14:]).mean())
        closed_price = safe_float(ind.summary.get("close"), primary["close"].iloc[-1])
        price = closed_price
        if mtf.snapshot and mtf.snapshot.last:
            price = mtf.snapshot.last
        live_move_atr = abs(price - closed_price) / max(atr, closed_price * 1e-9)
        live_move_pct = (
            abs(price - closed_price) / closed_price * 100.0
            if closed_price > 0
            else 0.0
        )
        live_price_ok = bool(live_move_atr <= 4.0 and live_move_pct <= 4.0)
        if not live_price_ok:
            result.warnings.append(
                "Live price moved too far from the latest closed candle "
                f"({live_move_atr:.1f} ATR / {live_move_pct:.1f}%) — signal blocked."
            )

        # --- Patterns & structure ---
        pat = self.patterns.detect(primary)
        result.patterns = pat
        struct = self.structure.analyze(primary, atr=atr)
        result.structure = struct

        # --- Higher TF ---
        htf_score, htf_notes = self._multi_tf_score(mtf, indicator_suites)
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
            FactorScore(
                "volatility",
                float(ind.summary.get("volatility_score", 0)),
                w.volatility,
                self._volatility_detail(ind),
            ),
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

        total = sum(f.weighted for f in result.factors)
        # Normalize if weights don't sum to 1
        wsum = sum(f.weight for f in result.factors) or 1.0
        total = total / wsum

        # A genuine confluence score must discount disagreement between the core
        # directional systems. Context (news/funding) cannot overpower EMA,
        # momentum, structure, and MTF conflicts.
        core = [
            f for f in result.factors
            if f.name in ("trend", "momentum", "structure", "multi_tf")
            and abs(f.score) >= 0.12
        ]
        if core and abs(total) >= 0.05:
            total_sign = 1 if total > 0 else -1
            aligned = sum(1 for f in core if np.sign(f.score) == total_sign)
            opposed = sum(1 for f in core if np.sign(f.score) == -total_sign)
            if opposed >= 2:
                total *= 0.68
                result.multi_tf_notes.insert(0, "Core-signal conflict: confidence discounted")
            elif aligned >= 3 and opposed == 0:
                total *= 1.08
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
        result.technical_confidence = conf
        result.confidence = conf

        # Strategy tags and strict closed-candle setup recognition. Detection
        # classifies an already-directional thesis; it cannot create direction
        # or bypass any downstream quality/safety gate.
        result.strategy_tags = self._strategy_tags(ind, struct, pat, mtf, direction, conf)
        strict_setup = detect_strict_setup(
            primary,
            ind,
            struct,
            direction=direction,
            price=closed_price,
            atr=atr,
        )
        if strict_setup is not None:
            result.strategy_tags.extend(
                [strict_setup.setup_type, "strict_setup_confirmed"]
            )
            result.strategy_tags = list(dict.fromkeys(result.strategy_tags))
            result.setup_name = strict_setup.label
        else:
            result.setup_name = self._setup_name(result.strategy_tags, direction, struct)

        # Key levels for plan
        support, resistance = self._nearest_levels(struct, price)
        result.key_levels = self._collect_levels(struct, price)
        execution = build_execution_profile(
            primary,
            ind,
            struct,
            direction=direction,
            price=price,
            atr=atr,
            snapshot=mtf.snapshot,
            max_pre_entry_tp1_progress_pct=float(
                getattr(
                    self.config.analysis,
                    "max_pre_entry_tp1_progress_pct",
                    70.0,
                )
            ),
            setup_name=result.setup_name,
            strategy_tags=result.strategy_tags,
            remaining_expiry_minutes=float(
                min(
                    getattr(self.config.analysis, "max_entry_valid_minutes", 180),
                    getattr(self.config.analysis, "retest_entry_expiry_bars", 6)
                    * timeframe_to_minutes(mtf.primary_tf),
                )
            ),
            expected_hold_hours=(
                4.0 if "scalping" in result.strategy_tags else 8.0
            ),
            policy=replace(
                DEFAULT_EXECUTION_POLICY,
                default_taker_fee_bps_per_side=float(
                    self.config.analysis.execution_taker_fee_bps_per_side
                ),
                default_slippage_bps_per_side=float(
                    self.config.analysis.execution_slippage_bps_per_side
                ),
                default_funding_bps_per_8h=float(
                    self.config.analysis.execution_funding_bps_per_8h
                ),
                reference_impact_notional_usd=float(
                    self.config.analysis.execution_impact_notional_usd
                ),
                max_spread_bps=float(self.config.analysis.max_spread_bps),
                max_ticker_age_seconds=float(
                    self.config.analysis.max_ticker_age_seconds
                ),
                max_orderbook_age_seconds=float(
                    self.config.analysis.max_orderbook_age_seconds
                ),
                minimum_execution_quality=float(
                    self.config.analysis.execution_min_score
                ),
            ),
        )
        result.execution = execution
        quality_map = dict(mtf.quality or {})
        primary_quality = dict(quality_map.get(mtf.primary_tf) or {})
        required_quality = {
            tf: dict(quality_map.get(tf) or {})
            for tf in [mtf.primary_tf, "1h", "4h"]
            if tf in mtf.frames
        }
        # Historical/unit callers may not supply live-quality metadata. When
        # the live fetcher does supply it, all requested confirmation frames
        # must be fresh and continuous before a directional signal can pass.
        mtf_data_quality_ok = bool(
            not quality_map
            or (
                required_quality
                and all(q.get("ok", False) for q in required_quality.values())
            )
        )
        data_quality_ok = mtf_data_quality_ok and live_price_ok
        data_quality_score = min(
            [
                safe_float(q.get("score"), 0.0)
                for q in required_quality.values()
            ]
            or [100.0]
        )

        # Preserve the existing technical score. Legacy V2 changes only overall
        # trade confidence by making execution a bottleneck instead of a small
        # rank adjustment.
        legacy_comparison = execution_aware_legacy_confidence(
            technical_confidence=conf,
            execution_score=execution.score,
            immediate_sl_risk=execution.immediate_sl_risk,
            data_quality_score=data_quality_score,
            confidence_min=self.config.analysis.min_confidence,
            confidence_max=self.config.analysis.max_confidence,
            execution_confidence_buffer=getattr(
                self.config.analysis,
                "execution_confidence_buffer",
                5.0,
            ),
        )
        legacy_conf = legacy_comparison.legacy_confidence
        legacy_v2_conf = legacy_comparison.legacy_v2_confidence
        calibrated_conf = (
            legacy_v2_conf
            if bool(getattr(self.config.analysis, "legacy_v2_enabled", True))
            else legacy_conf
        )
        if not data_quality_ok:
            calibrated_conf = min(calibrated_conf, 45.0)
            legacy_conf = min(legacy_conf, 45.0)
            legacy_v2_conf = min(legacy_v2_conf, 45.0)
        result.confidence = calibrated_conf

        # Trade plan — structure-clustered retest entry, stop, and targets.
        funding = mtf.snapshot.funding_rate if mtf.snapshot else None
        plan = self.risk.build_plan(
            direction=direction,
            price=price,
            atr=atr,
            structure_stops=(support, resistance),
            confidence=calibrated_conf,
            funding_rate=funding,
            primary_tf=mtf.primary_tf,
            setup_name=result.setup_name,
            strategy_tags=result.strategy_tags,
            execution=execution.to_dict(),
        )
        result.trade_plan = plan
        result.scenarios = self.risk.scenarios(
            price=price,
            atr=atr,
            bias=bias,
            confidence=calibrated_conf,
            support=support,
            resistance=resistance,
        )

        # Deterministic reasons first
        result.key_reasons = self._key_reasons(result, ind, struct, pat)
        result.key_risks = self._key_risks(result, atr, price, funding)
        result.key_reasons.extend(execution.reasons)
        result.key_risks.extend(execution.risks)

        funding_pct = (funding * 100.0) if funding is not None else None

        # Resolve deterministic production gates before optional narrative work.
        confidence_floor = float(
            getattr(self.config.analysis, "directional_confidence_threshold", 68.0)
        )
        score_floor = float(
            getattr(self.config.analysis, "directional_score_threshold", 0.20)
        )
        execution_floor = float(
            getattr(self.config.analysis, "execution_min_score", 72.0)
        )
        legacy_execution_floor = float(
            getattr(self.config.analysis, "legacy_execution_min_score", 65.0)
        )
        max_immediate_sl_risk = float(
            getattr(self.config.analysis, "max_immediate_sl_risk", 32.0)
        )
        plan_prop_safe_before_confidence_gate = bool(
            getattr(plan, "prop_safe", True)
        )
        minimum_net_rr = float(getattr(self.config.risk, "min_rr", 1.25))
        execution_net_rr = (
            execution.net_risk_reward[1]
            if len(execution.net_risk_reward) > 1
            else (execution.net_risk_reward[0] if execution.net_risk_reward else None)
        )
        analysis_gate_evaluation = evaluate_analysis_gates(
            direction=direction,
            technical_quality=result.technical_confidence,
            execution_quality=execution.score,
            overall_quality=result.confidence,
            confluence_magnitude=abs(result.confluence_total),
            entry_status=execution.status,
            immediate_sl_risk=execution.immediate_sl_risk,
            market_quality_ok=execution.market_quality_ok,
            data_quality_ok=data_quality_ok,
            prop_safe=plan_prop_safe_before_confidence_gate,
            confidence_floor=confidence_floor,
            score_floor=score_floor,
            execution_floor=execution_floor,
            max_immediate_sl_risk=max_immediate_sl_risk,
            hard_failures=execution.hard_failures,
            net_rr=execution_net_rr,
            min_net_rr=minimum_net_rr,
        )

        # LLM narrative is optional and narrative-only. External providers are
        # contacted only after every deterministic eligibility and safety gate.
        llm_candidate = deterministic_narrative_eligible(
            use_llm=use_llm,
            direction=direction,
            technical_quality=result.technical_confidence,
            overall_quality=calibrated_conf,
            confluence_total=result.confluence_total,
            execution=execution,
            data_quality_ok=data_quality_ok,
            prop_safe=plan_prop_safe_before_confidence_gate,
            has_feasible_target=bool(plan.take_profits),
            confidence_floor=confidence_floor,
            score_floor=score_floor,
            execution_floor=execution_floor,
            max_immediate_sl_risk=max_immediate_sl_risk,
            min_net_rr=minimum_net_rr,
        )
        llm_invocation_status = (
            "eligible_not_requested"
            if not use_llm and llm_candidate
            else (
                "pending"
                if llm_candidate
                else "skipped_deterministic_reject"
            )
        )
        if llm_candidate:
            try:
                llm_ctx = {
                    "symbol": result.symbol,
                    "exchange": result.exchange_id,
                    "primary_tf": result.primary_tf,
                    "bias": bias,
                    "direction": direction,
                    "confidence": calibrated_conf,
                    "price": price,
                    "setup_name": result.setup_name,
                    "confluence_total": result.confluence_total,
                    "top_factors": result.factor_breakdown(),
                    "structure": struct.summary,
                    "patterns": [h.name for h in (pat.hits[:5] if pat else [])],
                    "derivatives": der_notes,
                    "multi_tf": htf_notes[:4],
                    "leverage_suggested": plan.leverage_suggested,
                    "leverage_engine": plan.leverage_reasoning,
                    "atr_pct": plan.atr_pct,
                    "funding_rate_pct": funding_pct,
                    "primary_setup": plan.to_primary_setup(),
                    "position_simulation": plan.to_position_simulation(),
                    "execution": execution.to_dict(),
                }
                narrative = NarrativeLLM(self.config).generate(llm_ctx)
                llm_invocation_status = (
                    f"completed:{str(narrative.provider or 'unknown').lower()}"
                )
                result.llm = narrative
                if narrative.signal_narrative:
                    result.trader_commentary = narrative.signal_narrative
                else:
                    result.trader_commentary = self._commentary(result)
                if narrative.key_reasons:
                    result.key_reasons = list(
                        dict.fromkeys(
                            [*result.key_reasons, *narrative.key_reasons]
                        )
                    )[:8]
                if narrative.key_risks:
                    result.key_risks = list(
                        dict.fromkeys([*result.key_risks, *narrative.key_risks])
                    )[:8]
                # Enrich scenario narratives if provided
                if result.scenarios and narrative.scenarios:
                    for key in ("bullish", "base", "bearish"):
                        text = narrative.scenarios.get(key)
                        if text and hasattr(result.scenarios, key):
                            sc = getattr(result.scenarios, key)
                            if isinstance(sc, dict):
                                sc["narrative"] = text
                if plan.leverage_reasoning and narrative.leverage_reasoning:
                    plan.leverage_reasoning = (
                        f"{plan.leverage_reasoning} LLM: {narrative.leverage_reasoning}"
                    )
                # LLM play-out confidence
                result.llm_confidence = float(narrative.llm_confidence or 0.0)
                result.llm_confidence_reason = (narrative.confidence_reason or "").strip()
                result.llm_confidence_detail = dict(getattr(narrative, "confidence_detail", None) or {})
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM narrative layer failed: {}", exc)
                llm_invocation_status = f"fallback:{type(exc).__name__}"
                result.trader_commentary = self._commentary(result)
                result.llm_confidence, result.llm_confidence_reason = heuristic_llm_confidence(
                    direction=direction,
                    technical_confidence=calibrated_conf,
                    confluence_total=result.confluence_total,
                    atr_pct=plan.atr_pct,
                    funding_rate_pct=funding_pct,
                )
                from src.analysis.llm import build_heuristic_confidence_detail

                result.llm_confidence_detail = build_heuristic_confidence_detail(
                    direction=direction,
                    llm_confidence=result.llm_confidence,
                    conf_reason=result.llm_confidence_reason,
                    factors=result.factor_breakdown(),
                    confluence_total=result.confluence_total,
                    technical_confidence=calibrated_conf,
                )
        else:
            if not use_llm:
                llm_invocation_status = "disabled"
            result.trader_commentary = self._commentary(result)
            result.llm_confidence, result.llm_confidence_reason = heuristic_llm_confidence(
                direction=direction,
                technical_confidence=calibrated_conf,
                confluence_total=result.confluence_total,
                atr_pct=plan.atr_pct,
                funding_rate_pct=funding_pct,
            )
            from src.analysis.llm import build_heuristic_confidence_detail

            result.llm_confidence_detail = build_heuristic_confidence_detail(
                direction=direction,
                llm_confidence=result.llm_confidence,
                conf_reason=result.llm_confidence_reason,
                factors=result.factor_breakdown(),
                confluence_total=result.confluence_total,
                technical_confidence=calibrated_conf,
            )

        if not result.llm_confidence_reason:
            result.llm_confidence_reason = (
                f"Play-out score {result.llm_confidence:.0f}% for {direction}."
            )

        # LLM confidence is a narrative/veto field only. It cannot raise the
        # deterministic execution-calibrated confidence.
        if direction not in ("long", "short"):
            # Do not present flat setups as high-confidence trades
            result.llm_confidence = min(float(result.llm_confidence or 0.0), 25.0)
            result.confidence = min(calibrated_conf, 30.0)

        # Prop eligibility is based on deterministic confidence calibrated by
        # execution and data quality.
        common_legacy_gate = bool(
            direction in ("long", "short")
            and abs(result.confluence_total) >= score_floor
            and execution.status in ("confirmation_pending", "wait_retest")
            and execution.immediate_sl_risk <= max_immediate_sl_risk
            and execution.market_quality_ok
            and data_quality_ok
            and execution_net_rr is not None
            and execution_net_rr >= minimum_net_rr
        )
        legacy_signal_eligible = bool(
            common_legacy_gate
            and legacy_conf >= confidence_floor
            and execution.score >= legacy_execution_floor
        )
        legacy_v2_signal_eligible = bool(
            common_legacy_gate
            and legacy_v2_conf >= confidence_floor
            and execution.score >= execution_floor
        )
        self.risk.apply_prop_confidence_gate(
            plan, result.confidence, minimum=confidence_floor
        )
        signal_eligible = bool(analysis_gate_evaluation.eligible)
        universal_eligible = bool(analysis_gate_evaluation.universal_eligible)
        production_qualified = bool(analysis_gate_evaluation.production_qualified)
        structured_rejections: List[Dict[str, Any]] = [
            {
                "code": gate.code,
                "detail": gate.explanation,
                "value": gate.actual_value,
                "threshold": gate.required_value,
                "distance": gate.distance,
                "severity": gate.severity,
                "stage": gate.stage,
                "authoritative": gate.authoritative,
            }
            for gate in analysis_gate_evaluation.gates
            if not gate.passed
        ]
        if direction in ("long", "short") and not signal_eligible:
            gate_reasons = [
                gate.explanation
                for gate in analysis_gate_evaluation.gates
                if not gate.passed and gate.authoritative
            ]
            logger.info(
                "Signal rejected: symbol={} timeframe={} bias={} "
                "technical={:.1f} legacy_conf={:.1f} v2_conf={:.1f} "
                "execution={:.1f} reasons={}",
                result.symbol,
                result.primary_tf,
                result.bias,
                result.technical_confidence,
                legacy_conf,
                legacy_v2_conf,
                execution.score,
                "|".join(
                    str(item.get("code") or "") for item in structured_rejections
                ),
            )
            result.direction = "flat"
            result.setup_name = "No Trade / Wait for Confirmation"
            plan.direction = "flat"
            plan.entry_status = "blocked"
            plan.entry_reason = "Signal blocked: " + ", ".join(gate_reasons)
            result.warnings.append(plan.entry_reason)
            result.trader_commentary = (
                f"{result.bias.title()} bias only — no trade. {plan.entry_reason}. "
                f"{execution.entry_reason}"
            )

        for reason in execution.reasons:
            if reason not in result.key_reasons:
                result.key_reasons.append(reason)
        for risk in execution.risks:
            if risk not in result.key_risks:
                result.key_risks.append(risk)

        # Rank the evaluated directional thesis before publication gates rewrite
        # a rejected signal to ``flat``.  The old post-gate condition produced
        # a synthetic 0 for every rejected directional candidate, even though
        # all deterministic rank inputs were available.
        rank_available = direction in ("long", "short")
        if rank_available:
            result.rank_score, rank_breakdown = deterministic_rank_score(
                overall_quality=result.confidence,
                execution_quality=execution.score,
                target_feasibility=execution.components.get("target_feasibility", 0.0),
                stop_quality=execution.components.get("stop_quality", 0.0),
                net_rr=(
                    execution.net_risk_reward[1]
                    if len(execution.net_risk_reward) > 1
                    else (execution.net_risk_reward[0] if execution.net_risk_reward else 0.0)
                ),
                market_data_quality=execution.components.get("data_market_quality", 0.0),
                setup_validity=100.0 if not execution.hard_failures else 0.0,
                uncertainty_penalty=min(20.0, len(execution.uncertainties) * 3.0),
            )
        else:
            result.rank_score, rank_breakdown = 0.0, {}

        result.meta = {
            "price": price,
            "atr": atr,
            "atr_pct": plan.atr_pct,
            "indicator_count": ind.indicator_count,
            "bars_primary": len(primary),
            "timeframes": mtf.all_timeframes(),
            "volume_profile": {
                "poc": struct.volume_profile_poc,
                "vah": struct.volume_profile_vah,
                "val": struct.volume_profile_val,
            },
            "funding_rate": funding,
            "open_interest": mtf.snapshot.open_interest if mtf.snapshot else None,
            "long_short_ratio": mtf.snapshot.long_short_ratio if mtf.snapshot else None,
            "simulated_capital": plan.simulated_capital,
            "is_simulation": True,
            "technical_confidence": result.technical_confidence,
            "legacy_confidence": legacy_conf,
            "legacy_v2_confidence": legacy_v2_conf,
            "legacy_v2_enabled": bool(
                getattr(self.config.analysis, "legacy_v2_enabled", True)
            ),
            "legacy_confidence_comparison": legacy_comparison.to_dict(),
            "llm_confidence": result.llm_confidence,
            "llm_confidence_reason": result.llm_confidence_reason,
            "llm_confidence_detail": result.llm_confidence_detail,
            "llm_invocation_status": llm_invocation_status,
            "llm_provider": (
                str(result.llm.provider) if result.llm is not None else "none"
            ),
            "llm_rate_limit_events": int(
                getattr(result.llm, "rate_limit_events", 0)
                if result.llm is not None
                else 0
            ),
            "llm_provider_errors": list(
                getattr(result.llm, "provider_errors", [])
                if result.llm is not None
                else []
            ),
            "rank_score": result.rank_score,
            "rank_available": rank_available,
            "rank_policy_version": RANK_POLICY_VERSION,
            "rank_breakdown": rank_breakdown,
            "prop_safe": bool(getattr(plan, "prop_safe", True)),
            "prop_flags": list(getattr(plan, "prop_flags", None) or []),
            "prop_guidance": dict(getattr(plan, "prop_guidance", None) or {}),
            "universal_eligible": universal_eligible,
            "production_qualified": production_qualified,
            "signal_eligible": signal_eligible,
            "legacy_signal_eligible": legacy_signal_eligible,
            "legacy_v2_signal_eligible": legacy_v2_signal_eligible,
            "rejection_reasons": structured_rejections,
            "gate_evaluation": analysis_gate_evaluation.to_dict(),
            "gate_policy_version": analysis_gate_evaluation.policy_version,
            "evaluated_direction": direction,
            "execution": execution.to_dict(),
            "data_quality": dict(mtf.quality or {}),
            "primary_data_quality_ok": data_quality_ok,
            "live_price_ok": live_price_ok,
            "live_move_atr": round(live_move_atr, 3),
            "live_move_pct": round(live_move_pct, 3),
            "holding_window": "30m–24h",
            "indicator_timeframes_computed": sorted(indicator_suites),
            "strict_setup_detection": (
                strict_setup.to_dict() if strict_setup is not None else None
            ),
        }
        logger.info(
            "Analysis {} {} bias={} tech={:.1f}% llm={:.1f}% rank={:.1f} score={:.3f} lev={:.1f}x",
            result.symbol,
            result.primary_tf,
            result.bias,
            result.technical_confidence,
            result.llm_confidence,
            result.rank_score,
            result.confluence_total,
            plan.leverage_suggested,
        )
        return result

    # ------------------------------------------------------------------
    def _multi_tf_score(
        self,
        mtf: MultiTimeframeData,
        indicator_suites: Optional[Dict[str, IndicatorSuite]] = None,
    ) -> Tuple[float, List[str]]:
        """
        Day-trade MTF blend: prioritize 15m, 1h, and 4h (4h as confirmation).
        Mix trend + momentum on LTF for intraday alignment; weight volume and momentum higher.
        """
        from src.utils.helpers import timeframe_to_minutes

        notes: List[str] = []
        scores: List[float] = []
        weights: List[float] = []

        # Day-trade TF weights: emphasize 15m and 1h, keep 4h confirmation
        def _tf_weight(mins: int) -> float:
            if mins <= 15:
                return 1.65  # primary intraday drive
            if mins <= 60:
                return 1.45  # session bias
            if mins <= 240:
                return 1.15  # confirmation TF
            return 0.45  # de-emphasize daily+

        for tf, df in mtf.frames.items():
            if df is None or df.empty or len(df) < 30:
                continue
            quality = dict((mtf.quality or {}).get(tf) or {})
            if quality and not quality.get("ok", False):
                notes.append(
                    f"{tf}: excluded for data quality ({quality.get('reason', 'invalid')})"
                )
                continue
            ind = (indicator_suites or {}).get(tf)
            if ind is None:
                ind = compute_indicators(df, config=self.config)
            trend_s = float(ind.summary.get("trend_score", 0))
            mom_s = float(ind.summary.get("momentum_score", 0))
            mins = timeframe_to_minutes(tf)
            # For intraday TFs (<=1h) favor momentum slightly; for 4h favor trend/structure
            if mins <= 60:
                s = 0.40 * trend_s + 0.60 * mom_s
            else:
                s = 0.70 * trend_s + 0.30 * mom_s
            w = _tf_weight(mins)
            # Boost score slightly when volume surge present on that TF
            vol_ratio = float(ind.summary.get("vol_ratio") or 1.0)
            if vol_ratio >= 1.3:
                s = clamp(s * 1.08, -1, 1)
                notes.append(f"{tf}: volume spike (vol_ratio={vol_ratio:.2f})")
            scores.append(s)
            weights.append(w)
            rsi = ind.summary.get("rsi")
            rsi_s = f" RSI={rsi:.1f}" if rsi is not None else ""
            notes.append(f"{tf}: score={s:+.2f} (t={trend_s:+.2f}/m={mom_s:+.2f}){rsi_s}")

        if not scores:
            return 0.0, ["No multi-TF data"]
        total_w = sum(weights) or 1.0
        agg = sum(s * w for s, w in zip(scores, weights)) / total_w

        signs = [np.sign(s) for s in scores if abs(s) > 0.1]
        if signs and all(x == signs[0] for x in signs):
            notes.insert(0, "Day-trade stack aligned (15m/1h/4h)")
            agg = clamp(agg * 1.10, -1, 1)
        elif signs:
            # Soft conflict: reduce aggression and prefer waiting for 1h/4h confirmation
            notes.insert(0, "MTF conflict — reduce size / wait for 1h or 4h confirmation")
            agg *= 0.75

        return float(clamp(agg, -1, 1)), notes

    def _derivatives_score(self, snap: Optional[MarketSnapshot]) -> Tuple[float, List[str]]:
        """Funding-first for high-leverage perps; fade crowded positioning."""
        notes: List[str] = []
        if not snap:
            return 0.0, ["Derivatives snapshot unavailable"]

        score = 0.0
        fr = snap.funding_rate
        if fr is not None:
            fr_pct = fr * 100
            notes.append(f"Funding {fr_pct:+.4f}%")
            # Stronger fade signals for day-trade funding extremes
            if fr_pct > 0.08:
                score -= 0.45
                notes.append("Very elevated long funding — mean-revert / short bias")
            elif fr_pct > 0.035:
                score -= 0.28
                notes.append("Elevated long funding — squeeze risk")
            elif fr_pct > 0.015:
                score -= 0.12
            elif fr_pct < -0.08:
                score += 0.45
                notes.append("Very elevated short funding — squeeze / long bias")
            elif fr_pct < -0.035:
                score += 0.28
                notes.append("Elevated short funding — short-squeeze risk")
            elif fr_pct < -0.015:
                score += 0.12
        else:
            notes.append("Funding n/a")

        if snap.open_interest is not None:
            notes.append(f"OI {snap.open_interest:,.0f}")
        if snap.open_interest_change_pct_24h is not None:
            oi_change = snap.open_interest_change_pct_24h
            notes.append(f"OI 24h {oi_change:+.2f}%")
            price_change = snap.percentage_24h or 0.0
            if oi_change >= 4.0:
                if price_change > 1.0:
                    score += 0.16
                    notes.append("Rising OI confirms upside participation")
                elif price_change < -1.0:
                    score -= 0.16
                    notes.append("Rising OI confirms downside participation")
            elif oi_change <= -4.0:
                score *= 0.82
                notes.append("Falling OI weakens continuation conviction")
        if snap.funding_average_24h is not None:
            notes.append(f"Avg funding 24h {snap.funding_average_24h * 100:+.4f}%")
        if snap.long_short_ratio is not None:
            lsr = snap.long_short_ratio
            notes.append(f"L/S ratio {lsr:.3f}")
            if lsr > 1.6:
                score -= 0.22
                notes.append("Crowded longs (L/S high) — fade bias")
            elif lsr < 0.75:
                score += 0.22
                notes.append("Crowded shorts (L/S low) — squeeze bias")

        if snap.percentage_24h is not None:
            notes.append(f"24h {snap.percentage_24h:+.2f}%")
            # Intraday extension fade (mean reversion fuel)
            if snap.percentage_24h > 8:
                score -= 0.15
                notes.append("Extended up 24h — MR short fuel")
            elif snap.percentage_24h < -8:
                score += 0.15
                notes.append("Extended down 24h — MR long fuel")

        return float(clamp(score, -1, 1)), notes

    def _bias_from_score(
        self,
        total: float,
        ind: IndicatorSuite,
        struct: StructureReport,
    ) -> Tuple[str, str, float]:
        # Directional calls require a meaningful edge; weaker scores remain bias-only.
        threshold = float(
            getattr(self.config.analysis, "directional_score_threshold", 0.20)
        )
        if total >= threshold:
            bias, direction = "bullish", "long"
        elif total <= -threshold:
            bias, direction = "bearish", "short"
        else:
            bias, direction = "neutral", "flat"

        adx = ind.summary.get("adx")
        mom = abs(float(ind.summary.get("momentum_score", 0) or 0))
        clarity = 1.0
        if adx is not None:
            if adx < 15:
                clarity = 0.8  # range → mean reversion ok, slightly less conf
            elif adx > 22:
                clarity = 1.12  # trend day — momentum scalps favored
        if mom > 0.4:
            clarity *= 1.05

        conf = min(92.0, abs(total) * 100 * 1.2 * clarity + 18)
        if direction == "flat":
            conf = min(conf, 48)

        # Softer structure conflict on LTF (can scalp against HTF briefly)
        if struct.trend == "up" and direction == "short":
            conf *= 0.90
        if struct.trend == "down" and direction == "long":
            conf *= 0.90

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
        """Tags optimized for intraday perp day-trading playbooks (15m–4h focus)."""
        from src.utils.helpers import timeframe_to_minutes

        tags: List[str] = []
        adx = ind.summary.get("adx") or 0
        rsi = ind.summary.get("rsi")
        trend = float(ind.summary.get("trend_score", 0))
        mom = float(ind.summary.get("momentum_score", 0))
        bb_pos = float(ind.summary.get("bb_position", 0.5))
        vol_ratio = float(ind.summary.get("vol_ratio") or 1.0)
        mins = timeframe_to_minutes(mtf.primary_tf)

        # Style tags
        if abs(mom) > 0.35 and adx >= 18:
            tags.append("momentum")
        if adx < 24 and (bb_pos <= 0.22 or bb_pos >= 0.78 or (rsi is not None and (rsi < 35 or rsi > 65))):
            tags.append("mean_reversion")
        bos_agrees = (
            direction == "long" and struct.last_bos == "bullish"
        ) or (
            direction == "short" and struct.last_bos == "bearish"
        )
        if bos_agrees:
            tags.append("breakout" if struct.last_bos == "bullish" else "breakdown")
        if abs(mom) > 0.25:
            tags.append("momentum")
        if vol_ratio >= 1.25:
            tags.append("volume_surge")
        if any(p.bias in ("bullish", "bearish") and p.kind == "candlestick" for p in pat.hits[:3]):
            names = " ".join(p.name.lower() for p in pat.hits[:3])
            if any(x in names for x in ("engulf", "star", "hammer", "shooting", "marubozu")):
                tags.append("reversal_pattern")

        # Horizon — prefer intraday horizons: 15m, 1h, 4h
        if mins <= 5:
            tags.append("scalping")
        elif mins <= 240:
            tags.append("day_trade")
        else:
            tags.append("swing")  # de-emphasized for day-traders

        if direction == "flat":
            tags.append("wait")
        if conf >= 70:
            tags.append("high_conviction")
        if struct.last_choch:
            tags.append("choch")
        if struct.volume_profile_poc:
            tags.append("micro_structure")

        seen = set()
        out = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def _setup_name(self, tags: List[str], direction: str, struct: StructureReport) -> str:
        """Short-term setup labels for aggressive perp cards."""
        if direction == "flat":
            return "No Trade / Stand Aside"
        side = "Long" if direction == "long" else "Short"
        wanted = "bullish" if direction == "long" else "bearish"
        if "confirmed_reversal" in tags and struct.last_choch == wanted:
            return f"{side} Confirmed Reversal"
        if "mean_reversion" in tags and struct.trend == "range":
            return f"{side} Range Mean Reversion"
        if "breakout" in tags or "breakdown" in tags:
            if "momentum" in tags and "volume_surge" in tags:
                return f"{side} Breakout Continuation"
            return f"{side} Breakout Retest"
        if "momentum" in tags:
            return f"{side} Momentum"
        if "volume_surge" in tags and "momentum" in tags:
            return f"{side} Volume Momentum Burst"
        if "choch" in tags:
            return f"{side} CHoCH Continuation"
        return f"{side} Day-Trade Confluence"

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

    def _volatility_detail(self, ind: IndicatorSuite) -> str:
        s = ind.summary
        return (
            f"BB position={s.get('bb_position')} width={s.get('bb_width_pct')}%; "
            f"ATR={s.get('atr_pct')}%"
        )

    def _commentary(self, a: FullAnalysis) -> str:
        """Day-trader narrative for selective, prop-aware crypto perps."""
        parts: List[str] = []
        price = a.meta.get("price") if a.meta else None
        parts.append(
            f"Day-trade read on {a.primary_tf} {a.symbol}: {a.bias} lean "
            f"({a.confidence:.0f}% conf, confluence {a.confluence_total:+.3f}). "
            f"Engine is weighted for trend, momentum, structure, volume, and execution quality."
        )
        if a.structure:
            parts.append(a.structure.summary + ".")
        if a.patterns and a.patterns.hits:
            top = a.patterns.hits[0]
            parts.append(
                f"Candle/pattern trigger: {top.name} ({top.bias}, {top.confidence:.0f}%)."
            )
        if a.derivatives_notes:
            parts.append("Funding/OI: " + "; ".join(a.derivatives_notes[:3]) + ".")
        if a.multi_tf_notes:
            parts.append("Stack: " + a.multi_tf_notes[0] + ".")
        if a.news and a.news.bias != "neutral":
            parts.append(f"News tilt {a.news.bias} — treat as secondary.")
        if a.trade_plan and a.trade_plan.direction != "flat":
            tp = a.trade_plan
            hold = getattr(tp, "hold_detail", "") or "hold ~30min–12h"
            parts.append(
                f"Playbook: {a.setup_name}. Tight entry "
                f"{format_price(tp.entry_low, price)}–{format_price(tp.entry_high, price)}, "
                f"SL {format_price(tp.stop_loss, price)}, TP1 R:R {tp.primary_rr:.2f} ({tp.quality}). "
                f"Suggested lev ~{tp.leverage_suggested:.0f}x "
                f"({'prop max 5x' if tp.prop_mode else 'day-trade band 10–30x'}). {hold}."
            )
        else:
            parts.append("No clean scalp — stand aside; do not force high leverage.")
        parts.append("Research / simulation only — not financial advice.")
        return " ".join(parts)

    def _key_reasons(
        self,
        a: FullAnalysis,
        ind: IndicatorSuite,
        struct: StructureReport,
        pat: PatternReport,
    ) -> List[str]:
        reasons: List[str] = []
        for f in sorted(a.factors, key=lambda x: abs(x.weighted), reverse=True)[:5]:
            if abs(f.score) < 0.05:
                continue
            side = "supports long" if f.score > 0 else "supports short"
            reasons.append(f"{f.name.title()} {side} ({f.score:+.2f}): {f.detail[:100]}")
        if struct.last_bos:
            reasons.append(f"Market structure BOS: {struct.last_bos}")
        if struct.last_choch:
            reasons.append(f"CHoCH: {struct.last_choch}")
        if pat.hits:
            reasons.append(f"Top pattern: {pat.hits[0].name} ({pat.hits[0].bias})")
        return reasons[:8]

    def _key_risks(
        self,
        a: FullAnalysis,
        atr: float,
        price: float,
        funding: Optional[float],
    ) -> List[str]:
        risks = [
            "Leveraged perps can liquidate quickly — risk % of capital at the stop, not max margin.",
            "Day-trade / scalp: thesis should resolve intraday (30min–12h); do not turn losers into swings.",
            "Simulation only — not live account advice.",
        ]
        atr_pct = (atr / price * 100) if price else 0
        if atr_pct > 2.0:
            risks.append(f"Elevated ATR ({atr_pct:.2f}% of price) — reduce leverage and wait for a wider structural invalidation.")
        if funding is not None and abs(funding) > 0.0003:
            risks.append(f"Funding not neutral ({funding*100:+.4f}%) — squeeze / funding bleed risk.")
        if a.confidence < 45:
            risks.append("Confidence below 45% — skip or micro size only.")
        if a.trade_plan and a.trade_plan.quality == "poor":
            risks.append("Plan quality poor — do not force the trade.")
        return risks[:8]
