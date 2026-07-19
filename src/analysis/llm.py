"""Optional free LLM narratives via Groq or Google Gemini (HTTP APIs)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
            "Respond with ONLY valid JSON (no markdown) matching the schema below. Be concise, professional, non-hype, and risk-aware. Not financial advice.\n\n"
            "JSON_SCHEMA:\n"
            "{\n"
            '  "direction": "long|short|flat",\n'
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

        return LLMNarrative(
            signal_narrative=str(parsed.get("signal_narrative") or ""),
            direction=str(parsed.get("direction") or ""),
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
        conf = ctx.get("confidence", 0)
        setup = ctx.get("setup_name", "Setup")
        lev = float(ctx.get("leverage_suggested") or 20)
        atr_pct = float(ctx.get("atr_pct") or 0)
        funding = ctx.get("funding_rate_pct")
        factors = ctx.get("top_factors") or []
        direction = ctx.get("direction", "flat")
        price = ctx.get("price") or None

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
            key_reasons=reasons,
            key_risks=risks,
            scenarios=scenarios,
            provider=provider,
            model="heuristic",
            raw_ok=False,
            error="No LLM keys or provider failed — used local narrative engine.",
            trade_card=trade_card,
        )
