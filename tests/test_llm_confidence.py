"""LLM confidence scoring + rank helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.llm import (
    NarrativeLLM,
    combined_rank_score,
    heuristic_llm_confidence,
)


def test_heuristic_flat_is_low_priority():
    score, reason = heuristic_llm_confidence(
        direction="flat",
        technical_confidence=80.0,
        confluence_total=0.4,
    )
    assert score <= 25
    assert "flat" in reason.lower() or "neutral" in reason.lower()


def test_heuristic_directional_blends_tech_and_confluence():
    score, reason = heuristic_llm_confidence(
        direction="long",
        technical_confidence=70.0,
        confluence_total=0.5,
    )
    assert 30 <= score <= 92
    assert "long" in reason.lower() or "LONG" in reason


def test_combined_rank_score_zeros_flat():
    assert combined_rank_score(
        direction="flat",
        llm_confidence=90,
        technical_confidence=90,
        confluence_total=0.8,
    ) == 0.0


def test_combined_rank_score_prefers_higher_llm():
    low = combined_rank_score(
        direction="long",
        llm_confidence=40,
        technical_confidence=70,
        confluence_total=0.3,
    )
    high = combined_rank_score(
        direction="long",
        llm_confidence=85,
        technical_confidence=70,
        confluence_total=0.3,
    )
    assert high > low


def test_from_parsed_reads_llm_confidence():
    llm = NarrativeLLM(config=None)
    narrative = llm._from_parsed(
        {
            "direction": "short",
            "llm_confidence": 72,
            "confidence_reason": "HTF resistance holding with weak bounce.",
            "signal_narrative": "Short bias into supply.",
            "key_reasons": ["Rejection wick"],
            "key_risks": ["Squeeze risk"],
            "scenarios": {},
        },
        provider="test",
        model="unit",
    )
    assert narrative.llm_confidence == 72
    assert "HTF resistance" in narrative.confidence_reason
    assert narrative.raw_ok is True


def test_from_parsed_caps_flat_confidence():
    llm = NarrativeLLM(config=None)
    narrative = llm._from_parsed(
        {
            "direction": "flat",
            "llm_confidence": 80,
            "confidence_reason": "Chop.",
            "signal_narrative": "No trade.",
        },
        provider="test",
        model="unit",
    )
    assert narrative.llm_confidence <= 25
