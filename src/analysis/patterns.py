"""Candlestick and classical chart pattern detection with confidence scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import safe_float


@dataclass
class PatternHit:
    name: str
    kind: str  # candlestick | classical
    bias: str  # bullish | bearish | neutral
    confidence: float  # 0-100
    note: str = ""
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None


@dataclass
class PatternReport:
    hits: List[PatternHit] = field(default_factory=list)
    candle_bias: float = 0.0  # -1..+1
    classical_bias: float = 0.0
    summary: str = ""

    @property
    def top_hits(self) -> List[PatternHit]:
        return sorted(self.hits, key=lambda h: h.confidence, reverse=True)[:8]


class PatternDetector:
    """Detect candlestick + classical chart patterns on OHLCV."""

    def detect(self, df: pd.DataFrame) -> PatternReport:
        if df is None or len(df) < 20:
            return PatternReport(summary="Insufficient data for patterns")

        work = df.copy()
        for c in ("open", "high", "low", "close"):
            if c not in work.columns:
                # try lower
                cols = {x.lower(): x for x in work.columns}
                if c in cols:
                    work[c] = work[cols[c]]
                else:
                    return PatternReport(summary=f"Missing {c}")

        hits: List[PatternHit] = []
        hits.extend(self._candlestick_patterns(work))
        hits.extend(self._classical_patterns(work))

        candle = [h for h in hits if h.kind == "candlestick"]
        classical = [h for h in hits if h.kind == "classical"]

        def bias_of(group: List[PatternHit]) -> float:
            if not group:
                return 0.0
            score = 0.0
            wsum = 0.0
            for h in group:
                sign = 1.0 if h.bias == "bullish" else (-1.0 if h.bias == "bearish" else 0.0)
                w = h.confidence / 100.0
                score += sign * w
                wsum += w
            return float(np.clip(score / max(wsum, 1e-9), -1, 1))

        report = PatternReport(
            hits=sorted(hits, key=lambda h: h.confidence, reverse=True),
            candle_bias=bias_of(candle),
            classical_bias=bias_of(classical),
        )
        if report.hits:
            top = report.hits[0]
            report.summary = (
                f"{len(report.hits)} patterns; top: {top.name} "
                f"({top.bias}, {top.confidence:.0f}%)"
            )
        else:
            report.summary = "No high-confidence patterns detected"
        return report

    # ------------------------------------------------------------------
    # Candlesticks
    # ------------------------------------------------------------------
    def _candlestick_patterns(self, df: pd.DataFrame) -> List[PatternHit]:
        hits: List[PatternHit] = []
        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        n = len(df)
        i = n - 1
        if i < 3:
            return hits

        body = np.abs(c - o)
        full = np.maximum(h - l, 1e-12)
        upper_wick = h - np.maximum(c, o)
        lower_wick = np.minimum(c, o) - l
        avg_body = pd.Series(body).rolling(14).mean().values

        def bull(name: str, conf: float, note: str = "") -> PatternHit:
            return PatternHit(name, "candlestick", "bullish", conf, note, i, i)

        def bear(name: str, conf: float, note: str = "") -> PatternHit:
            return PatternHit(name, "candlestick", "bearish", conf, note, i, i)

        # Doji
        if body[i] / full[i] < 0.1:
            hits.append(PatternHit("Doji", "candlestick", "neutral", 55, "Indecision", i, i))

        # Hammer / Hanging man
        if lower_wick[i] >= 2 * body[i] and upper_wick[i] <= body[i] * 0.4 and body[i] > 0:
            # context: after decline → hammer; after rally → hanging man
            prior = c[i - 5 : i].mean() if i >= 5 else c[i - 1]
            if c[i] < prior:
                hits.append(bull("Hammer", 68, "Long lower wick after decline"))
            else:
                hits.append(bear("Hanging Man", 60, "Long lower wick after advance"))

        # Inverted hammer / shooting star
        if upper_wick[i] >= 2 * body[i] and lower_wick[i] <= body[i] * 0.4 and body[i] > 0:
            prior = c[i - 5 : i].mean() if i >= 5 else c[i - 1]
            if c[i] > prior:
                hits.append(bear("Shooting Star", 66, "Long upper wick after advance"))
            else:
                hits.append(bull("Inverted Hammer", 58, "Long upper wick after decline"))

        # Marubozu
        if body[i] / full[i] > 0.88:
            if c[i] > o[i]:
                hits.append(bull("Bullish Marubozu", 62, "Strong close near high"))
            else:
                hits.append(bear("Bearish Marubozu", 62, "Strong close near low"))

        # Engulfing
        if i >= 1:
            prev_bear = c[i - 1] < o[i - 1]
            prev_bull = c[i - 1] > o[i - 1]
            curr_bull = c[i] > o[i]
            curr_bear = c[i] < o[i]
            if prev_bear and curr_bull and o[i] <= c[i - 1] and c[i] >= o[i - 1]:
                conf = 72 if body[i] > body[i - 1] * 1.1 else 64
                hits.append(bull("Bullish Engulfing", conf))
            if prev_bull and curr_bear and o[i] >= c[i - 1] and c[i] <= o[i - 1]:
                conf = 72 if body[i] > body[i - 1] * 1.1 else 64
                hits.append(bear("Bearish Engulfing", conf))

        # Harami
        if i >= 1 and body[i - 1] > 0:
            inside = max(o[i], c[i]) <= max(o[i - 1], c[i - 1]) and min(o[i], c[i]) >= min(
                o[i - 1], c[i - 1]
            )
            if inside and body[i] < body[i - 1] * 0.6:
                if c[i - 1] < o[i - 1] and c[i] > o[i]:
                    hits.append(bull("Bullish Harami", 58))
                if c[i - 1] > o[i - 1] and c[i] < o[i]:
                    hits.append(bear("Bearish Harami", 58))

        # Morning / Evening star (3-candle)
        if i >= 2:
            b0, b1, b2 = body[i - 2], body[i - 1], body[i]
            # Morning star
            if (
                c[i - 2] < o[i - 2]
                and b1 < b0 * 0.5
                and c[i] > o[i]
                and c[i] > (o[i - 2] + c[i - 2]) / 2
            ):
                hits.append(bull("Morning Star", 74, "3-candle bullish reversal"))
            # Evening star
            if (
                c[i - 2] > o[i - 2]
                and b1 < b0 * 0.5
                and c[i] < o[i]
                and c[i] < (o[i - 2] + c[i - 2]) / 2
            ):
                hits.append(bear("Evening Star", 74, "3-candle bearish reversal"))

        # Three soldiers / crows
        if i >= 2:
            if all(c[j] > o[j] for j in (i - 2, i - 1, i)) and c[i] > c[i - 1] > c[i - 2]:
                if all(body[j] > avg_body[j] * 0.7 for j in (i - 2, i - 1, i) if not np.isnan(avg_body[j])):
                    hits.append(bull("Three White Soldiers", 70))
            if all(c[j] < o[j] for j in (i - 2, i - 1, i)) and c[i] < c[i - 1] < c[i - 2]:
                if all(body[j] > avg_body[j] * 0.7 for j in (i - 2, i - 1, i) if not np.isnan(avg_body[j])):
                    hits.append(bear("Three Black Crows", 70))

        # Piercing / Dark cloud
        if i >= 1:
            mid_prev = (o[i - 1] + c[i - 1]) / 2
            if c[i - 1] < o[i - 1] and c[i] > o[i] and o[i] < l[i - 1] and c[i] > mid_prev and c[i] < o[i - 1]:
                hits.append(bull("Piercing Line", 65))
            if c[i - 1] > o[i - 1] and c[i] < o[i] and o[i] > h[i - 1] and c[i] < mid_prev and c[i] > o[i - 1]:
                hits.append(bear("Dark Cloud Cover", 65))

        # Tweezer tops/bottoms
        if i >= 1:
            tol = full[i] * 0.05
            if abs(l[i] - l[i - 1]) <= tol and c[i] > o[i] and c[i - 1] < o[i - 1]:
                hits.append(bull("Tweezer Bottom", 60))
            if abs(h[i] - h[i - 1]) <= tol and c[i] < o[i] and c[i - 1] > o[i - 1]:
                hits.append(bear("Tweezer Top", 60))

        return hits

    # ------------------------------------------------------------------
    # Classical chart patterns (approximate geometric)
    # ------------------------------------------------------------------
    def _classical_patterns(self, df: pd.DataFrame) -> List[PatternHit]:
        hits: List[PatternHit] = []
        if len(df) < 40:
            return hits

        window = df.tail(80)
        h = window["high"].values.astype(float)
        l = window["low"].values.astype(float)
        c = window["close"].values.astype(float)
        n = len(window)

        piv_h = self._pivots(h, "high", 3)
        piv_l = self._pivots(l, "low", 3)

        # Double top / bottom
        if len(piv_h) >= 2:
            i1, i2 = piv_h[-2], piv_h[-1]
            level1, level2 = h[i1], h[i2]
            if abs(level1 - level2) / max(level1, 1e-12) < 0.008 and i2 - i1 >= 5:
                trough = l[i1:i2].min()
                conf = 68.0
                if c[-1] < trough:
                    conf = 78.0
                hits.append(
                    PatternHit(
                        "Double Top",
                        "classical",
                        "bearish",
                        conf,
                        f"Peaks ~{level1:.4g} / {level2:.4g}",
                        int(i1),
                        int(i2),
                    )
                )

        if len(piv_l) >= 2:
            i1, i2 = piv_l[-2], piv_l[-1]
            level1, level2 = l[i1], l[i2]
            if abs(level1 - level2) / max(level1, 1e-12) < 0.008 and i2 - i1 >= 5:
                peak = h[i1:i2].max()
                conf = 68.0
                if c[-1] > peak:
                    conf = 78.0
                hits.append(
                    PatternHit(
                        "Double Bottom",
                        "classical",
                        "bullish",
                        conf,
                        f"Troughs ~{level1:.4g} / {level2:.4g}",
                        int(i1),
                        int(i2),
                    )
                )

        # Head & shoulders (simplified 3 peaks)
        if len(piv_h) >= 3:
            a, b, cidx = piv_h[-3], piv_h[-2], piv_h[-1]
            la, lb, lc = h[a], h[b], h[cidx]
            if lb > la and lb > lc and abs(la - lc) / max(lb, 1e-12) < 0.03:
                neck = min(l[a:b].min(), l[b:cidx].min())
                conf = 72.0 if c[-1] < neck else 64.0
                hits.append(
                    PatternHit(
                        "Head & Shoulders",
                        "classical",
                        "bearish",
                        conf,
                        f"Head {lb:.4g}, neck ~{neck:.4g}",
                        int(a),
                        int(cidx),
                    )
                )

        if len(piv_l) >= 3:
            a, b, cidx = piv_l[-3], piv_l[-2], piv_l[-1]
            la, lb, lc = l[a], l[b], l[cidx]
            if lb < la and lb < lc and abs(la - lc) / max(abs(lb), 1e-12) < 0.03:
                neck = max(h[a:b].max(), h[b:cidx].max())
                conf = 72.0 if c[-1] > neck else 64.0
                hits.append(
                    PatternHit(
                        "Inverse Head & Shoulders",
                        "classical",
                        "bullish",
                        conf,
                        f"Head {lb:.4g}, neck ~{neck:.4g}",
                        int(a),
                        int(cidx),
                    )
                )

        # Triangle / squeeze via converging highs & lows
        if len(piv_h) >= 3 and len(piv_l) >= 3:
            recent_h = piv_h[-3:]
            recent_l = piv_l[-3:]
            h_slope = (h[recent_h[-1]] - h[recent_h[0]]) / max(recent_h[-1] - recent_h[0], 1)
            l_slope = (l[recent_l[-1]] - l[recent_l[0]]) / max(recent_l[-1] - recent_l[0], 1)
            range_now = h[-5:].max() - l[-5:].min()
            range_prev = h[-30:-20].max() - l[-30:-20].min() if n >= 30 else range_now
            contracting = range_now < range_prev * 0.75
            if contracting and h_slope < 0 and l_slope > 0:
                hits.append(
                    PatternHit(
                        "Symmetrical Triangle",
                        "classical",
                        "neutral",
                        63,
                        "Converging swings — breakout watch",
                    )
                )
            elif contracting and abs(h_slope) < abs(l_slope) * 0.4 and l_slope > 0:
                hits.append(
                    PatternHit(
                        "Ascending Triangle",
                        "classical",
                        "bullish",
                        66,
                        "Flat resistance, rising support",
                    )
                )
            elif contracting and abs(l_slope) < abs(h_slope) * 0.4 and h_slope < 0:
                hits.append(
                    PatternHit(
                        "Descending Triangle",
                        "classical",
                        "bearish",
                        66,
                        "Flat support, falling resistance",
                    )
                )

        # Flag / pennant after impulse
        if n >= 30:
            impulse = c[-30] 
            move = (c[-15] - impulse) / max(abs(impulse), 1e-12)
            cons_range = (h[-10:].max() - l[-10:].min()) / max(abs(c[-1]), 1e-12)
            if abs(move) > 0.04 and cons_range < 0.025:
                bias = "bullish" if move > 0 else "bearish"
                hits.append(
                    PatternHit(
                        "Bull Flag" if move > 0 else "Bear Flag",
                        "classical",
                        bias,
                        67,
                        f"Impulse {move*100:.1f}% then tight coil",
                    )
                )

        # Channel (parallel-ish highs/lows regression)
        x = np.arange(n)
        try:
            coef_c = np.polyfit(x[-40:], c[-40:], 1)
            slope = coef_c[0]
            resid = c[-40:] - (coef_c[0] * x[-40:] + coef_c[1])
            if np.std(resid) / max(abs(c[-1]), 1e-12) < 0.012 and abs(slope) * 40 / max(abs(c[-1]), 1e-12) > 0.01:
                bias = "bullish" if slope > 0 else "bearish"
                hits.append(
                    PatternHit(
                        "Rising Channel" if slope > 0 else "Falling Channel",
                        "classical",
                        bias,
                        60,
                        f"Linear slope {slope:.6g}/bar",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Channel fit failed: {}", exc)

        # Cup & handle (very approximate U shape)
        if n >= 60:
            left = c[-60:-50].mean()
            bottom = c[-45:-25].min()
            right = c[-20:-10].mean()
            handle = c[-10:].min()
            if (
                bottom < left * 0.97
                and abs(right - left) / max(left, 1e-12) < 0.03
                and handle < right
                and handle > bottom
            ):
                hits.append(
                    PatternHit(
                        "Cup & Handle (approx)",
                        "classical",
                        "bullish",
                        58,
                        "U-shape base with shallow handle",
                    )
                )

        return hits

    @staticmethod
    def _pivots(arr: np.ndarray, mode: str, order: int = 3) -> List[int]:
        idxs: List[int] = []
        n = len(arr)
        for i in range(order, n - order):
            left, right = arr[i - order : i], arr[i + 1 : i + 1 + order]
            if mode == "high" and arr[i] >= left.max() and arr[i] >= right.max():
                idxs.append(i)
            if mode == "low" and arr[i] <= left.min() and arr[i] <= right.min():
                idxs.append(i)
        return idxs
