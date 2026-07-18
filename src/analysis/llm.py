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
    signal_narrative: str = ""
    leverage_reasoning: str = ""
    key_reasons: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    scenarios: Dict[str, str] = field(default_factory=dict)  # bullish/base/bearish text
    provider: str = "none"
    model: str = ""
    raw_ok: bool = False
    error: str = ""


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
        return (
            "You are a senior crypto perpetual futures prop trader. "
            "Given the analysis JSON, respond with ONLY valid JSON (no markdown) matching:\n"
            "{\n"
            '  "signal_narrative": "3-5 sentences explaining the trade signal",\n'
            '  "leverage_reasoning": "2-3 sentences on why suggested leverage is appropriate",\n'
            '  "key_reasons": ["bullet reason 1", "bullet reason 2", "..."],\n'
            '  "key_risks": ["risk 1", "risk 2", "..."],\n'
            '  "scenarios": {\n'
            '    "bullish": "1-2 sentences",\n'
            '    "base": "1-2 sentences",\n'
            '    "bearish": "1-2 sentences"\n'
            "  }\n"
            "}\n"
            "Be precise, non-hype, risk-aware. Not financial advice.\n\n"
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
        return LLMNarrative(
            signal_narrative=str(parsed.get("signal_narrative") or ""),
            leverage_reasoning=str(parsed.get("leverage_reasoning") or ""),
            key_reasons=[str(x) for x in (parsed.get("key_reasons") or [])][:8],
            key_risks=[str(x) for x in (parsed.get("key_risks") or [])][:8],
            scenarios={
                "bullish": str(scenarios.get("bullish") or ""),
                "base": str(scenarios.get("base") or ""),
                "bearish": str(scenarios.get("bearish") or ""),
            },
            provider=provider,
            model=model,
            raw_ok=True,
        )

    def _fallback(self, ctx: Dict[str, Any], provider: str = "local_fallback") -> LLMNarrative:
        bias = ctx.get("bias", "neutral")
        conf = ctx.get("confidence", 0)
        setup = ctx.get("setup_name", "Setup")
        lev = ctx.get("leverage_suggested", 1)
        atr_pct = ctx.get("atr_pct", 0)
        funding = ctx.get("funding_rate_pct")
        factors = ctx.get("top_factors") or []
        direction = ctx.get("direction", "flat")

        reasons = []
        for f in factors[:5]:
            if isinstance(f, dict):
                reasons.append(f"{f.get('name')}: score {f.get('score')} — {str(f.get('detail', ''))[:80]}")
        if not reasons:
            reasons = [f"Weighted confluence supports {bias} lean at {conf:.0f}% confidence."]

        risks = [
            "Crypto perps can gap; stops may slip in liquidation cascades.",
            "Funding and crowded positioning can reverse quickly around event risk.",
        ]
        if atr_pct and atr_pct > 2.5:
            risks.append(f"Elevated ATR (~{atr_pct:.2f}% of price) — volatility can invalidate levels fast.")
        if funding is not None and abs(float(funding)) > 0.03:
            risks.append(f"Funding extreme ({funding}%) — squeeze risk against crowded side.")

        narrative = (
            f"{setup}: bias is {bias} ({conf:.0f}% confidence) with directional lean {direction}. "
            f"Structure, multi-timeframe trend, and derivatives context were combined into a "
            f"weighted confluence score. Prefer planned risk of ~1% of simulated capital; "
            f"do not chase if entry zone is missed."
        )
        lev_reason = (
            f"Suggested leverage ~{lev:.1f}x is scaled from ATR volatility ({atr_pct:.2f}% of price), "
            f"signal confidence ({conf:.0f}%), and funding pressure. "
            f"Lower confidence or higher vol → lower leverage; this is a simulation, not a mandate."
        )
        scenarios = {
            "bullish": "Continuation if HTF demand holds and momentum expands with rising volume.",
            "base": "Range / mean-reversion until BOS; trade edges only or stand aside.",
            "bearish": "Breakdown if support fails on volume and funding flips against longs.",
        }
        if bias == "bullish":
            scenarios["bullish"] = "Primary path: hold structure, reclaim mid-range, push toward TP stack."
        elif bias == "bearish":
            scenarios["bearish"] = "Primary path: fail resistance, lose mid-range, extend toward TP stack."

        return LLMNarrative(
            signal_narrative=narrative,
            leverage_reasoning=lev_reason,
            key_reasons=reasons,
            key_risks=risks,
            scenarios=scenarios,
            provider=provider,
            model="heuristic",
            raw_ok=False,
            error="No LLM keys or provider failed — used local narrative engine.",
        )
