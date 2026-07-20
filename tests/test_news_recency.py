"""News lookback / recency helpers."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.news import NewsAnalyzer, NewsItem, _parse_published_at
from src.utils.config import NewsConfig


def test_parse_published_at_iso_and_unix():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    assert _parse_published_at(now.isoformat()) is not None
    assert _parse_published_at(int(now.timestamp())) is not None
    assert _parse_published_at("") is None
    assert _parse_published_at(None) is None


def test_is_within_hours_filters_old_items():
    analyzer = NewsAnalyzer(news_cfg=NewsConfig(lookback_hours=4, enabled=True))
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    assert analyzer._is_within_hours(recent, 4) is True
    assert analyzer._is_within_hours(old, 4) is False


def test_analyze_prefers_recent_headlines(monkeypatch):
    analyzer = NewsAnalyzer(news_cfg=NewsConfig(lookback_hours=4, max_articles=5, enabled=True))
    recent_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=20)).isoformat()

    def fake_public(base: str):
        return [
            NewsItem(title=f"{base} old dump", published_at=old_ts, source="test"),
            NewsItem(title=f"{base} fresh rally breakout", published_at=recent_ts, source="test"),
        ]

    monkeypatch.setattr(analyzer, "_fetch_public_headlines", fake_public)
    monkeypatch.setattr(analyzer, "_fetch_cryptopanic", lambda base: [])
    # no cryptopanic token
    analyzer.news_cfg.cryptopanic_token = ""

    bundle = analyzer.analyze("BTC")
    assert bundle.items
    assert all("old dump" not in i.title for i in bundle.items)
    assert any("fresh rally" in i.title for i in bundle.items)
    assert "h" in (bundle.summary or "")
