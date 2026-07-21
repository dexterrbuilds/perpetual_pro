"""Optional free LLM narratives via Groq or Google Gemini (HTTP APIs)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger

from src.utils.config import AppConfig


@dataclass
class LLMNarrative:
    # Short human narrative
    signal_narrative: str = ""
    # Structured trade card fields for high-leverage perp scalps
    direction: str = ""  # long | short | flat
    entry_zones: List[Dict[str, Any]] = field(default_factory=list)  # [{"low": 123, "high": 125, "note": "tight"}]
    alternative_entries: List[Dict[str, Any]] = field(default_factory=list)
    stop_loss: Dict[str, Any] = field(default_factory=dict)  # {"price": 122, "reason": "invalidates structure"}
    tps: List[Dict[str, Any]] = field(default_factory=list)  # [{"label":"TP1","price":126,"rr":0.6}]
    max_hold_hours: int = 24
    suggested_leverage: float = 20.0
    suggested_leverage_range: str = "20x-100x"
    leverage_reasoning: str = ""
    funding_impact: str = ""
    volume_confidence: float = 0.0
    # Play-out likelihood for the proposed signal (0–100)
    llm_confidence: float = 0.0
    confidence_reason: str = ""
    # Structured why-confident / why-not explanation
    confidence_detail: Dict[str, Any] = field(default_factory=dict)

    # Reasons / risks / scenarios
    key_reasons: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    scenarios: Dict[str, str] = field(default_factory=dict)  # bullish/base/bearish text

    # Meta
    provider: str = "none"
    model: str = ""
    raw_ok: bool = False
    error: str = ""
    # Compact trader-style card (human readable, emojis ok)
    trade_card: str = ""


class NarrativeLLM:
    """
    Best-effort narrative layer.
    Priority: Groq → Gemini → deterministic local fallback.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config
        llm_cfg = getattr(config, "llm", None) if config else None
        self.enabled = True if llm_cfg is None else bool(getattr(llm_cfg, "enabled", True))
        self.groq_key = os.getenv("GROQ_API_KEY", "") or (
            getattr(llm_cfg, "groq_api_key", "") if llm_cfg else ""
        )
        self.gemini_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "") or (
            getattr(llm_cfg, "gemini_api_key", "") if llm_cfg else ""
        )
        self.groq_model = (
            getattr(llm_cfg, "groq_model", None) if llm_cfg else None
        ) or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.gemini_model = (
            getattr(llm_cfg, "gemini_model", None) if llm_cfg else None
        ) or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.timeout = int(getattr(llm_cfg, "timeout_s", 25) if llm_cfg else 25)

    def generate(self, context: Dict[str, Any]) -> LLMNarrative:
        if not self.enabled:
            return self._fallback(context, provider="disabled")

        prompt = self._build_prompt(context)
        if self.groq_key:
            try:
                text = self._call_groq(prompt)
                parsed = self._parse_json_response(text)
                if parsed:
                    return self._from_parsed(parsed, "groq", self.groq_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Groq narrative failed: {}", exc)

        if self.gemini_key:
            try:
                text = self._call_gemini(prompt)
                parsed = self._parse_json_response(text)
                if parsed:
                    return self._from_parsed(parsed, "gemini", self.gemini_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gemini narrative failed: {}", exc)

        return self._fallback(context, provider="local_fallback")

    def _build_prompt(self, ctx: Dict[str, Any]) -> str:
        # Updated prompt to request a professional trader-style trade card optimized
        # for high-leverage crypto perpetual day-trading and scalping.
        return (
            "You are a senior crypto perpetual futures day trader. "
            "Prioritize 15m, 1h, and 4h (4h for confirmation). Hold time typically 30min–12 hours (max 24h). "
            "Leverage should be aggressive but responsible (20x–100x); avoid extreme >150x except in rare, very high conviction micro-scalps. "
            "Signal styles: short-term momentum, breakout retests, mean reversion, and intraday structure. "
            "Entries must be tight zones with clear invalidation; include alternative entries on retests. "
            "CRITICAL: Score llm_confidence (0-100) as how likely THIS directional signal will play out "
            "over the suggested hold window. Flat/no-trade → llm_confidence ≤ 25. "
            "Do not inflate confidence; be skeptical of weak confluence or conflicting factors. "
            "Respond with ONLY valid JSON (no markdown) matching the schema below. Be concise, professional, non-hype, and risk-aware. Not financial advice.\n\n"
            "JSON_SCHEMA:\n"
            "{\n"
            '  "direction": "long|short|flat",\n'
            '  "llm_confidence": <0-100 integer>,  /* play-out likelihood for this signal */\n'
            '  "confidence_reason": "one short sentence justifying llm_confidence",\n'
            '  "confidence_detail": {\n'
            '    "summary": "2-3 sentences explaining why confident or not (prop risk-aware)",\n'
            '    "supporting": ["factor that increases confidence", "..."],\n'
            '    "opposing": ["factor that reduces confidence", "..."],\n'
            '    "verdict": "high|medium|low|skip"\n'
            "  },\n"
            '  "signal_narrative": "3-5 concise sentences explaining the trade",\n'
            '  "entry_zones": [{"low": <num>, "high": <num>, "note": "tight|retention"}],\n'
            '  "alternative_entries": [{"low": <num>, "high": <num>, "note": "retest/alt"}],\n'
            '  "stop_loss": {"price": <num>, "reason": "text"},\n'
            '  "tps": [{"label":"TP1","price":<num>,"rr":<num>,"note":"short-term"}, ... up to 4],\n'
            '  "max_hold_hours": <int>,  /* typically 12 or 24 */\n'
            '  "suggested_leverage": <num>,\n'
            '  "suggested_leverage_range": "20x-100x (or narrower)",\n'
            '  "leverage_reasoning": "short justification for leverage choice",\n'
            '  "funding_impact": "how funding rate affects this trade (short)",\n'
            '  "volume_confidence": <0.0-1.0>,\n'
            '  "key_reasons": ["reason 1", "reason 2"],\n'
            '  "key_risks": ["risk 1", "risk 2"],\n'
            '  "scenarios": {"bullish":"...","base":"...","bearish":"..."},\n'
            '  "trade_card": "Human-friendly single-line card with emojis and clear fields"\n'
            "}\n\n"
            f"ANALYSIS:\n{json.dumps(ctx, default=str)[:12000]}"
        )

    def _call_groq(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.groq_model,
            "temperature": 0.35,
            "max_tokens": 900,
            "messages": [
                {
                    "role": "system",
                    "content": "You output only compact JSON for trading analysis narratives.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str) -> str:
        model = self.gemini_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.gemini_key}"
        )
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.35,
                "maxOutputTokens": 900,
                "responseMimeType": "application/json",
            },
        }
        resp = requests.post(url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            if t.lower().startswith("json"):
                t = t[4:].strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            # try extract first {...}
            start, end = t.find("{"), t.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(t[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def _from_parsed(self, parsed: Dict[str, Any], provider: str, model: str) -> LLMNarrative:
        scenarios = parsed.get("scenarios") or {}

        # Helper to coerce numeric fields safely
        def _safe_num(v, default=None):
            try:
                return float(v) if v is not None else default
            except Exception:
                return default

        entry_z = parsed.get("entry_zones") or parsed.get("entry_zones", [])
        alt_z = parsed.get("alternative_entries") or []
        stop = parsed.get("stop_loss") or {}
        tps = parsed.get("tps") or []
        llm_conf = _clamp_confidence(
            parsed.get("llm_confidence")
            if parsed.get("llm_confidence") is not None
            else parsed.get("confidence")
        )
        conf_reason = str(
            parsed.get("confidence_reason")
            or parsed.get("llm_confidence_reason")
            or ""
        ).strip()
        if not conf_reason and llm_conf is not None:
            conf_reason = f"Model play-out score {llm_conf:.0f}% for proposed direction."

        direction = str(parsed.get("direction") or "").strip().lower()
        # Flat / no-trade should not look high-conviction
        if direction in ("flat", "neutral", "none", "") and llm_conf is not None and llm_conf > 35:
            llm_conf = min(llm_conf, 25.0)
            if "flat" not in conf_reason.lower() and "no trade" not in conf_reason.lower():
                conf_reason = (conf_reason + " " if conf_reason else "") + "Capped: flat / no-trade setup."

        conf_detail = _normalize_confidence_detail(
            parsed.get("confidence_detail"),
            llm_conf=float(llm_conf if llm_conf is not None else 0.0),
            conf_reason=conf_reason,
            direction=direction,
        )

        return LLMNarrative(
            signal_narrative=str(parsed.get("signal_narrative") or ""),
            direction=direction,
            entry_zones=[{
                "low": _safe_num(e.get("low"), None) if isinstance(e, dict) else None,
                "high": _safe_num(e.get("high"), None) if isinstance(e, dict) else None,
                "note": e.get("note") if isinstance(e, dict) else str(e),
            } for e in entry_z][:3],
            alternative_entries=[{
                "low": _safe_num(e.get("low"), None) if isinstance(e, dict) else None,
                "high": _safe_num(e.get("high"), None) if isinstance(e, dict) else None,
                "note": e.get("note") if isinstance(e, dict) else str(e),
            } for e in alt_z][:3],
            stop_loss={
                "price": _safe_num(stop.get("price"), None) if isinstance(stop, dict) else None,
                "reason": str(stop.get("reason") or "") if isinstance(stop, dict) else str(stop),
            },
            tps=[{
                "label": str(t.get("label") or f"TP{i+1}"),
                "price": _safe_num(t.get("price"), None),
                "rr": _safe_num(t.get("rr"), None),
                "note": str(t.get("note") or "")
            } for i, t in enumerate(tps)][:4],
            max_hold_hours=int(parsed.get("max_hold_hours") or parsed.get("max_hold") or 24),
            suggested_leverage=_safe_num(parsed.get("suggested_leverage"), 20.0) or 20.0,
            suggested_leverage_range=str(parsed.get("suggested_leverage_range") or "20x-100x"),
            leverage_reasoning=str(parsed.get("leverage_reasoning") or ""),
            funding_impact=str(parsed.get("funding_impact") or ""),
            volume_confidence=_safe_num(parsed.get("volume_confidence"), 0.0) or 0.0,
            llm_confidence=float(llm_conf if llm_conf is not None else 0.0),
            confidence_reason=conf_reason[:280],
            confidence_detail=conf_detail,
            key_reasons=[str(x) for x in (parsed.get("key_reasons") or [])][:8],
            key_risks=[str(x) for x in (parsed.get("key_risks") or [])][:8],
            scenarios={
                "bullish": str(scenarios.get("bullish") or ""),
                "base": str(scenarios.get("base") or ""),
                "bearish": str(scenarios.get("bearish") or ""),
            },
            trade_card=str(parsed.get("trade_card") or parsed.get("card") or ""),
            provider=provider,
            model=model,
            raw_ok=True,
        )

    def _fallback(self, ctx: Dict[str, Any], provider: str = "local_fallback") -> LLMNarrative:
        bias = ctx.get("bias", "neutral")
        conf = float(ctx.get("confidence") or 0)
        setup = ctx.get("setup_name", "Setup")
        lev = float(ctx.get("leverage_suggested") or 20)
        atr_pct = float(ctx.get("atr_pct") or 0)
        funding = ctx.get("funding_rate_pct")
        factors = ctx.get("top_factors") or []
        direction = str(ctx.get("direction") or "flat").lower()
        price = ctx.get("price") or None
        confluence = float(ctx.get("confluence_total") or 0)

        # Heuristic play-out score when Groq/Gemini unavailable
        llm_conf, conf_reason = heuristic_llm_confidence(
            direction=direction,
            technical_confidence=conf,
            confluence_total=confluence,
            atr_pct=atr_pct,
            funding_rate_pct=float(funding) if funding is not None else None,
        )

        # Build compact reasons from top factors
        reasons = []
        for f in factors[:5]:
            if isinstance(f, dict):
                reasons.append(f"{f.get('name')}: score {f.get('score')} — {str(f.get('detail', ''))[:80]}")
        if not reasons:
            reasons = [f"Weighted confluence supports {bias} lean at {conf:.0f}% confidence."]

        risks = [
            "Crypto perps can gap; stops may slip in liquidation cascades.",
            "High leverage (20x–100x band) amplifies liquidation risk — risk the plan, not max margin.",
            "Funding and crowded positioning can reverse quickly around event risk.",
        ]
        if atr_pct and atr_pct > 2.5:
            risks.append(f"Elevated ATR (~{atr_pct:.2f}% of price) — volatility can invalidate levels fast.")
        if funding is not None and abs(float(funding)) > 0.03:
            risks.append(f"Funding extreme ({funding}%) — squeeze risk against crowded side.")

        # Heuristic entry / tp / sl generation when LLM not available
        entry_zones = []
        alt_entries = []
        stop = {}
        tps = []
        if price:
            # Tight entry band ±0.25–0.75% depending on confidence
            band_pct = 0.25 if conf >= 70 else 0.5 if conf >= 50 else 0.8
            low = price * (1 - band_pct / 100) if direction == "long" else price * (1 + band_pct / 100)
            high = price * (1 + band_pct / 100) if direction == "long" else price * (1 - band_pct / 100)
            entry_zones = [{"low": low, "high": high, "note": "primary tight zone"}]
            # alternative on retest ~1-1.5x band
            alt_band = band_pct * 1.8
            alt_low = price * (1 - alt_band / 100) if direction == "long" else price * (1 + alt_band / 100)
            alt_high = price * (1 + alt_band / 100) if direction == "long" else price * (1 - alt_band / 100)
            alt_entries = [{"low": alt_low, "high": alt_high, "note": "retest/alt"}]
            # stop ~ ATR-based or a fixed percent
            stop_loss_price = price * (1 - 0.6 / 100) if direction == "long" else price * (1 + 0.6 / 100)
            stop = {"price": stop_loss_price, "reason": "Invalidates structure / tight scalp stop"}

            # TP stack realistic short-term: TP1 0.6–1.0%, TP2 1.2–2.0%, TP3 2.5–4.0%, TP4 5%+ (only if strong)
            tp_moves = [0.6, 1.6, 3.0, 6.0]
            for i, move in enumerate(tp_moves):
                if direction == "long":
                    tp_price = price * (1 + move / 100)
                else:
                    tp_price = price * (1 - move / 100)
                tps.append({"label": f"TP{i+1}", "price": tp_price, "rr": None, "note": f"~{move:.2f}%"})

        narrative = (
            f"{setup}: {direction} bias {bias} ({conf:.0f}% conf). "
            f"Confluence from short-term momentum, volume, funding and micro-structure. "
            f"Suggested leverage ~{lev:.0f}x within 20x–100x band; tighten to ~20x if funding/ATR hostile. "
            f"Hold typically 30min–12h (max 24h); scale out at TP1–TP4."
        )
        lev_reason = (
            f"Leverage chosen ~{lev:.0f}x based on confidence {conf:.0f}%, ATR {atr_pct:.2f}%, funding {funding}. "
            "Higher conf + low funding → push toward upper band; hostile funding → reduce toward 20x."
        )
        scenarios = {
            "bullish": "Continuation if short-term momentum and volume expand and funding stabilizes.",
            "base": "Range / retest — prefer edges, micro-size on weak conviction.",
            "bearish": "Fail structure with rising volume and funding flip against position.",
        }

        # Compact trader card with emojis
        card_lines = []
        card_lines.append(f"{ '🟢' if direction=='long' else '🔴' if direction=='short' else '⚪️'} {setup} — {direction.upper()} ({conf:.0f}% )")
        if entry_zones:
            ez = entry_zones[0]
            card_lines.append(f"Entry: {ez['low']:.4f}–{ez['high']:.4f} {ez['note']}")
        if stop:
            card_lines.append(f"Stop: {stop['price']:.4f} — {stop['reason']}")
        for tp in (tps[:4] if tps else []):
            card_lines.append(f"{tp['label']}: {tp['price']:.4f} ({tp['note']})")
        card_lines.append(f"Lev: ~{lev:.0f}x (20x–100x). Hold ~30min–12h (max 24h)")
        trade_card = " | ".join(card_lines)

        conf_detail = build_heuristic_confidence_detail(
            direction=direction,
            llm_confidence=llm_conf,
            conf_reason=conf_reason,
            factors=factors,
            confluence_total=confluence,
            technical_confidence=conf,
        )

        return LLMNarrative(
            signal_narrative=narrative,
            direction=direction,
            entry_zones=entry_zones,
            alternative_entries=alt_entries,
            stop_loss=stop,
            tps=tps,
            max_hold_hours=24,
            suggested_leverage=lev,
            suggested_leverage_range="20x-100x",
            leverage_reasoning=lev_reason,
            funding_impact=(f"Funding {funding}%" if funding is not None else "n/a"),
            volume_confidence=float(ctx.get("volume_confidence") or 0.0),
            llm_confidence=llm_conf,
            confidence_reason=conf_reason,
            confidence_detail=conf_detail,
            key_reasons=reasons,
            key_risks=risks,
            scenarios=scenarios,
            provider=provider,
            model="heuristic",
            raw_ok=False,
            error="No LLM keys or provider failed — used local narrative engine.",
            trade_card=trade_card,
        )


def _clamp_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    # Allow 0–1 fraction inputs
    if 0.0 <= conf <= 1.0:
        conf *= 100.0
    return max(0.0, min(100.0, conf))


def heuristic_llm_confidence(
    *,
    direction: str,
    technical_confidence: float,
    confluence_total: float = 0.0,
    atr_pct: float = 0.0,
    funding_rate_pct: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Deterministic play-out score when remote LLM is unavailable.

    Returns (llm_confidence 0–100, short reason).
    """
    direction = (direction or "flat").lower()
    tech = max(0.0, min(100.0, float(technical_confidence or 0.0)))
    conf_mag = abs(float(confluence_total or 0.0))  # 0..1
    conf_score = conf_mag * 100.0

    if direction in ("flat", "neutral", ""):
        score = min(25.0, tech * 0.35 + conf_score * 0.15)
        reason = (
            f"Flat/neutral bias — low play-out priority "
            f"(tech {tech:.0f}%, confluence {confluence_total:+.3f})."
        )
        return round(score, 1), reason

    # Directional: blend technical confidence with |confluence|
    score = 0.55 * tech + 0.45 * conf_score

    # Soft penalties
    if atr_pct and atr_pct > 3.5:
        score *= 0.9
    if funding_rate_pct is not None:
        fr = float(funding_rate_pct)
        # Funding against long (positive) or against short (negative)
        if direction == "long" and fr > 0.05:
            score *= 0.92
        elif direction == "short" and fr < -0.05:
            score *= 0.92

    score = max(15.0, min(92.0, score))
    reason = (
        f"Heuristic play-out from technical {tech:.0f}% and "
        f"|confluence| {conf_mag:.3f} for {direction.upper()}."
    )
    return round(score, 1), reason


def _normalize_confidence_detail(
    raw: Any,
    *,
    llm_conf: float,
    conf_reason: str,
    direction: str,
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    supporting = [str(x) for x in (raw.get("supporting") or []) if x][:6]
    opposing = [str(x) for x in (raw.get("opposing") or []) if x][:6]
    summary = str(raw.get("summary") or conf_reason or "").strip()
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict not in ("high", "medium", "low", "skip"):
        if direction in ("flat", "neutral", ""):
            verdict = "skip"
        elif llm_conf >= 70:
            verdict = "high"
        elif llm_conf >= 50:
            verdict = "medium"
        else:
            verdict = "low"
    if not summary:
        summary = conf_reason or f"Play-out score {llm_conf:.0f}% for {direction or 'flat'}."
    return {
        "summary": summary[:500],
        "supporting": supporting,
        "opposing": opposing,
        "verdict": verdict,
    }


def build_heuristic_confidence_detail(
    *,
    direction: str,
    llm_confidence: float,
    conf_reason: str,
    factors: Any,
    confluence_total: float,
    technical_confidence: float,
) -> Dict[str, Any]:
    supporting: List[str] = []
    opposing: List[str] = []
    for f in (factors or [])[:6]:
        if not isinstance(f, dict):
            continue
        name = str(f.get("name") or "factor")
        score = float(f.get("score") or 0)
        detail = str(f.get("detail") or "")[:80]
        line = f"{name}: {score:+.2f}" + (f" — {detail}" if detail else "")
        if score >= 0.15:
            supporting.append(line)
        elif score <= -0.15:
            opposing.append(line)
    if abs(confluence_total) >= 0.2:
        supporting.append(f"Confluence magnitude {confluence_total:+.3f}")
    else:
        opposing.append(f"Weak confluence ({confluence_total:+.3f})")
    if technical_confidence < 45:
        opposing.append(f"Technical confidence only {technical_confidence:.0f}%")
    elif technical_confidence >= 65:
        supporting.append(f"Technical confidence {technical_confidence:.0f}%")
    if direction in ("flat", "neutral", ""):
        opposing.append("No clear directional bias — prefer stand aside")
    return _normalize_confidence_detail(
        {
            "summary": conf_reason,
            "supporting": supporting[:5],
            "opposing": opposing[:5],
        },
        llm_conf=llm_confidence,
        conf_reason=conf_reason,
        direction=direction,
    )


def combined_rank_score(
    *,
    direction: str,
    llm_confidence: float,
    technical_confidence: float,
    confluence_total: float = 0.0,
) -> float:
    """
    Rank score for directional setups only.

    Flat/neutral → 0 (excluded from leaderboard).
    Else: 60% LLM confidence + 40% technical (conf + |confluence|).
    """
    direction = (direction or "flat").lower()
    if direction not in ("long", "short"):
        return 0.0
    llm_c = max(0.0, min(100.0, float(llm_confidence or 0.0)))
    tech_c = max(0.0, min(100.0, float(technical_confidence or 0.0)))
    conf_boost = abs(float(confluence_total or 0.0)) * 100.0
    technical_blend = 0.7 * tech_c + 0.3 * conf_boost
    return round(0.6 * llm_c + 0.4 * technical_blend, 3)
