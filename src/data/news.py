"""Real-time crypto news fetch + lightweight sentiment scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from loguru import logger

from src.utils.config import AppConfig, NewsConfig
from src.utils.helpers import clamp, symbol_base, utc_now_iso


@dataclass
class NewsItem:
    title: str
    url: str = ""
    source: str = ""
    published_at: str = ""
    currencies: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1 .. +1
    kind: str = "news"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsBundle:
    symbol: str
    items: List[NewsItem] = field(default_factory=list)
    aggregate_sentiment: float = 0.0  # -1 .. +1
    bias: str = "neutral"  # bullish | bearish | neutral
    summary: str = ""
    fetched_at: str = field(default_factory=utc_now_iso)
    errors: List[str] = field(default_factory=list)

    def top(self, n: int = 5) -> List[NewsItem]:
        return self.items[:n]


class NewsAnalyzer:
    """Fetch headlines and score sentiment for a perpetual symbol."""

    CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
    # Public fallback RSS-style endpoints (no key)
    ALTERNATE_SOURCES = [
        # CoinGecko trending status (not full news, used as soft signal)
    ]

    def __init__(self, config: Optional[AppConfig] = None, news_cfg: Optional[NewsConfig] = None):
        self.config = config
        self.news_cfg = news_cfg or (config.news if config else NewsConfig())
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "perpetual_pro/1.0 (+https://local; research tool)",
                "Accept": "application/json,text/plain,*/*",
            }
        )

    def analyze(self, symbol: str) -> NewsBundle:
        base = symbol_base(symbol)
        bundle = NewsBundle(symbol=base)

        if not self.news_cfg.enabled:
            bundle.summary = "News disabled in config."
            return bundle

        items: List[NewsItem] = []

        # 1) CryptoPanic if token present
        if self.news_cfg.cryptopanic_token:
            try:
                items.extend(self._fetch_cryptopanic(base))
            except Exception as exc:  # noqa: BLE001
                bundle.errors.append(f"cryptopanic: {exc}")
                logger.warning("CryptoPanic fetch failed: {}", exc)

        # 2) Free public fallbacks
        if len(items) < 3:
            try:
                items.extend(self._fetch_public_headlines(base))
            except Exception as exc:  # noqa: BLE001
                bundle.errors.append(f"public_news: {exc}")
                logger.debug("Public news fallback failed: {}", exc)

        # Score & sort
        scored: List[NewsItem] = []
        for it in items:
            it.sentiment_score = self._score_text(f"{it.title} {it.kind}")
            scored.append(it)

        # Deduplicate by title similarity
        seen = set()
        unique: List[NewsItem] = []
        for it in scored:
            key = re.sub(r"\W+", "", it.title.lower())[:80]
            if key in seen:
                continue
            seen.add(key)
            unique.append(it)

        unique.sort(key=lambda x: abs(x.sentiment_score), reverse=True)
        max_n = self.news_cfg.max_articles
        bundle.items = unique[:max_n]

        if bundle.items:
            # Weight more extreme headlines slightly higher
            weights = [0.5 + abs(i.sentiment_score) for i in bundle.items]
            total_w = sum(weights) or 1.0
            agg = sum(i.sentiment_score * w for i, w in zip(bundle.items, weights)) / total_w
            bundle.aggregate_sentiment = clamp(agg, -1.0, 1.0)
        else:
            bundle.aggregate_sentiment = 0.0
            bundle.summary = f"No recent headlines found for {base}."
            return bundle

        if bundle.aggregate_sentiment >= 0.15:
            bundle.bias = "bullish"
        elif bundle.aggregate_sentiment <= -0.15:
            bundle.bias = "bearish"
        else:
            bundle.bias = "neutral"

        top_titles = "; ".join(i.title[:90] for i in bundle.items[:3])
        bundle.summary = (
            f"News bias {bundle.bias} (score {bundle.aggregate_sentiment:+.2f}) "
            f"from {len(bundle.items)} headlines. Top: {top_titles}"
        )
        return bundle

    def _fetch_cryptopanic(self, base: str) -> List[NewsItem]:
        params = {
            "auth_token": self.news_cfg.cryptopanic_token,
            "currencies": base,
            "public": "true",
            "kind": "news",
            "filter": "hot",
        }
        resp = self.session.get(self.CRYPTOPANIC_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        items: List[NewsItem] = []
        for row in results:
            title = (row.get("title") or "").strip()
            if not title:
                continue
            currencies = [
                c.get("code", "")
                for c in (row.get("currencies") or [])
                if isinstance(c, dict)
            ]
            items.append(
                NewsItem(
                    title=title,
                    url=(row.get("url") or ""),
                    source=(row.get("source") or {}).get("title", "CryptoPanic")
                    if isinstance(row.get("source"), dict)
                    else "CryptoPanic",
                    published_at=str(row.get("published_at") or ""),
                    currencies=currencies,
                    kind=str(row.get("kind") or "news"),
                    raw=row,
                )
            )
        return items

    def _fetch_public_headlines(self, base: str) -> List[NewsItem]:
        """
        Best-effort free sources without API keys.
        Uses CryptoCompare news (public) filtered by category/symbol keywords.
        """
        items: List[NewsItem] = []
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("Data") or []
            base_l = base.lower()
            for row in rows:
                title = (row.get("title") or "").strip()
                body = (row.get("body") or "")[:280]
                cats = (row.get("categories") or "").lower()
                tags = " ".join(row.get("tags") or []).lower() if isinstance(row.get("tags"), list) else str(row.get("tags") or "").lower()
                blob = f"{title} {body} {cats} {tags}".lower()
                # Keep if symbol mentioned or major market news
                if base_l not in blob and base_l not in cats and "market" not in cats:
                    # Still keep top market-wide items lightly
                    if "BTC" not in base and "bitcoin" not in blob and "crypto" not in blob:
                        continue
                items.append(
                    NewsItem(
                        title=title,
                        url=str(row.get("url") or row.get("guid") or ""),
                        source=str((row.get("source_info") or {}).get("name") or row.get("source") or "CryptoCompare"),
                        published_at=_ts_to_iso(row.get("published_on")),
                        currencies=[base] if base_l in blob else [],
                        kind="news",
                        raw={"id": row.get("id")},
                    )
                )
                if len(items) >= self.news_cfg.max_articles:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("CryptoCompare news failed: {}", exc)

        # Secondary: Gecko trending as soft macro signal (not classic news)
        try:
            url = "https://api.coingecko.com/api/v3/search/trending"
            resp = self.session.get(url, timeout=12)
            if resp.ok:
                coins = (resp.json() or {}).get("coins") or []
                names = []
                hit = False
                for c in coins[:10]:
                    item = c.get("item") or {}
                    sym = (item.get("symbol") or "").upper()
                    names.append(sym)
                    if sym == base.upper():
                        hit = True
                if names:
                    title = f"CoinGecko trending: {', '.join(names[:8])}"
                    if hit:
                        title = f"{base} is in CoinGecko trending list"
                    items.append(
                        NewsItem(
                            title=title,
                            url="https://www.coingecko.com/en/highlights/trending-crypto",
                            source="CoinGecko",
                            published_at=utc_now_iso(),
                            currencies=[base] if hit else [],
                            kind="trending",
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("CoinGecko trending failed: {}", exc)

        return items

    def _score_text(self, text: str) -> float:
        """Lexicon + config keyword sentiment in [-1, 1]."""
        t = text.lower()
        score = 0.0

        bullish_extra = self.news_cfg.bullish_keywords or []
        bearish_extra = self.news_cfg.bearish_keywords or []

        default_bull = [
            "surge", "soar", "rally", "breakout", "all-time high", "ath", "bull",
            "adoption", "inflow", "approve", "approval", "etf", "record high",
            "partnership", "upgrade", "accumulate", "institutional", "listing",
            "pump", "moon", "gain", "growth", "optimistic", "support",
        ]
        default_bear = [
            "crash", "plunge", "hack", "exploit", "lawsuit", "ban", "sec charge",
            "fraud", "bear", "dump", "liquidat", "outflow", "delist", "outage",
            "investigation", "probe", "sell-off", "selloff", "collapse", "fear",
            "warning", "risk", "reject", "rejection", "bankrupt", "insolvent",
        ]

        for kw in default_bull + list(bullish_extra):
            if kw.lower() in t:
                score += 0.18
        for kw in default_bear + list(bearish_extra):
            if kw.lower() in t:
                score -= 0.2

        # Mild polarity from common verbs
        if re.search(r"\b(rises?|rising|jumps?|climbs?)\b", t):
            score += 0.1
        if re.search(r"\b(falls?|falling|drops?|slides?|tumbles?)\b", t):
            score -= 0.1

        return clamp(score, -1.0, 1.0)


def _ts_to_iso(ts: Any) -> str:
    try:
        if ts is None:
            return ""
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except Exception:  # noqa: BLE001
        return str(ts or "")
