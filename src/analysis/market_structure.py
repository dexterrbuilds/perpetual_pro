"""Market structure: BOS/CHoCH, Order Blocks, FVG, liquidity, Wyckoff, Elliott basics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.helpers import safe_float


@dataclass
class StructureLevel:
    kind: str  # order_block | fvg | liquidity | bos | choch | support | resistance
    side: str  # bullish | bearish | neutral
    price_low: float
    price_high: float
    confidence: float
    note: str = ""
    index: Optional[int] = None

    @property
    def mid(self) -> float:
        return (self.price_low + self.price_high) / 2.0


@dataclass
class StructureReport:
    levels: List[StructureLevel] = field(default_factory=list)
    trend: str = "range"  # up | down | range
    last_bos: Optional[str] = None  # bullish | bearish
    last_choch: Optional[str] = None
    structure_score: float = 0.0  # -1..+1
    wyckoff_phase: str = "unknown"
    wyckoff_notes: str = ""
    elliott_notes: str = ""
    volume_profile_poc: Optional[float] = None
    volume_profile_vah: Optional[float] = None
    volume_profile_val: Optional[float] = None
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)
    summary: str = ""


class MarketStructureAnalyzer:
    """Approximate SMC / ICT-style structure + Wyckoff / Elliott heuristics."""

    def analyze(self, df: pd.DataFrame, atr: Optional[float] = None) -> StructureReport:
        if df is None or len(df) < 30:
            return StructureReport(summary="Insufficient data for structure")

        work = df.copy()
        for c in ("open", "high", "low", "close", "volume"):
            if c not in work.columns:
                cols = {x.lower(): x for x in work.columns}
                if c in cols:
                    work[c] = work[cols[c]]

        h = work["high"].values.astype(float)
        l = work["low"].values.astype(float)
        c = work["close"].values.astype(float)
        o = work["open"].values.astype(float)
        v = work["volume"].values.astype(float) if "volume" in work.columns else np.ones(len(work))

        if atr is None or atr <= 0:
            atr = float(np.mean(h[-14:] - l[-14:])) if len(h) >= 14 else float(np.mean(h - l))

        report = StructureReport()
        swing_h_idx = self._swing_indices(h, "high", order=3)
        swing_l_idx = self._swing_indices(l, "low", order=3)
        report.swing_highs = [float(h[i]) for i in swing_h_idx[-5:]]
        report.swing_lows = [float(l[i]) for i in swing_l_idx[-5:]]

        # Trend from swing structure
        report.trend = self._structure_trend(h, l, swing_h_idx, swing_l_idx)

        # BOS / CHoCH
        bos, choch, events = self._bos_choch(c, h, l, swing_h_idx, swing_l_idx, atr)
        report.last_bos = bos
        report.last_choch = choch
        report.levels.extend(events)

        # Order blocks
        report.levels.extend(self._order_blocks(o, h, l, c, atr))

        # Fair value gaps
        report.levels.extend(self._fair_value_gaps(h, l, c, atr))

        # Liquidity pools (equal highs/lows, recent swing extremes)
        report.levels.extend(self._liquidity_pools(h, l, swing_h_idx, swing_l_idx, atr))

        # Volume profile approximation
        poc, vah, val = self._volume_profile(c, v, bins=24)
        report.volume_profile_poc = poc
        report.volume_profile_vah = vah
        report.volume_profile_val = val
        if poc is not None:
            report.levels.append(
                StructureLevel(
                    kind="volume_poc",
                    side="neutral",
                    price_low=poc * 0.999,
                    price_high=poc * 1.001,
                    confidence=70,
                    note="Volume Profile POC (approx)",
                )
            )

        # Wyckoff heuristic
        report.wyckoff_phase, report.wyckoff_notes = self._wyckoff(c, v, h, l, atr)

        # Elliott basics
        report.elliott_notes = self._elliott_basics(c, swing_h_idx, swing_l_idx)

        # Structure score
        report.structure_score = self._score(report, c[-1], atr)
        report.summary = self._summarize(report, c[-1])
        return report

    @staticmethod
    def _swing_indices(arr: np.ndarray, mode: str, order: int = 3) -> List[int]:
        idxs: List[int] = []
        n = len(arr)
        for i in range(order, n - order):
            left, right = arr[i - order : i], arr[i + 1 : i + 1 + order]
            if mode == "high" and arr[i] >= left.max() and arr[i] >= right.max():
                idxs.append(i)
            if mode == "low" and arr[i] <= left.min() and arr[i] <= right.min():
                idxs.append(i)
        return idxs

    def _structure_trend(
        self,
        h: np.ndarray,
        l: np.ndarray,
        sh: List[int],
        sl: List[int],
    ) -> str:
        if len(sh) < 2 or len(sl) < 2:
            return "range"
        hh = h[sh[-1]] > h[sh[-2]]
        hl = l[sl[-1]] > l[sl[-2]]
        lh = h[sh[-1]] < h[sh[-2]]
        ll = l[sl[-1]] < l[sl[-2]]
        if hh and hl:
            return "up"
        if lh and ll:
            return "down"
        return "range"

    def _bos_choch(
        self,
        c: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        sh: List[int],
        sl: List[int],
        atr: float,
    ) -> Tuple[Optional[str], Optional[str], List[StructureLevel]]:
        levels: List[StructureLevel] = []
        last_bos = None
        last_choch = None
        if len(sh) < 2 or len(sl) < 2:
            return last_bos, last_choch, levels

        # Last swing levels
        last_swing_high = float(h[sh[-1]])
        prev_swing_high = float(h[sh[-2]])
        last_swing_low = float(l[sl[-1]])
        prev_swing_low = float(l[sl[-2]])
        price = float(c[-1])

        prior_trend = self._structure_trend(h, l, sh[:-1] or sh, sl[:-1] or sl)

        # Bullish BOS: close above last significant swing high in up/range
        if price > last_swing_high + atr * 0.05:
            last_bos = "bullish"
            levels.append(
                StructureLevel(
                    "bos",
                    "bullish",
                    last_swing_high,
                    last_swing_high,
                    75,
                    f"Close above swing high {last_swing_high:.6g}",
                    sh[-1],
                )
            )
            if prior_trend == "down":
                last_choch = "bullish"
                levels.append(
                    StructureLevel(
                        "choch",
                        "bullish",
                        last_swing_high,
                        last_swing_high,
                        80,
                        "Change of Character: break of bearish structure",
                        sh[-1],
                    )
                )

        # Bearish BOS
        if price < last_swing_low - atr * 0.05:
            last_bos = "bearish"
            levels.append(
                StructureLevel(
                    "bos",
                    "bearish",
                    last_swing_low,
                    last_swing_low,
                    75,
                    f"Close below swing low {last_swing_low:.6g}",
                    sl[-1],
                )
            )
            if prior_trend == "up":
                last_choch = "bearish"
                levels.append(
                    StructureLevel(
                        "choch",
                        "bearish",
                        last_swing_low,
                        last_swing_low,
                        80,
                        "Change of Character: break of bullish structure",
                        sl[-1],
                    )
                )

        # Internal structure notes via consecutive swings
        if len(sh) >= 2 and h[sh[-1]] > h[sh[-2]] and len(sl) >= 2 and l[sl[-1]] > l[sl[-2]]:
            if last_bos is None:
                levels.append(
                    StructureLevel(
                        "structure",
                        "bullish",
                        prev_swing_low,
                        last_swing_high,
                        60,
                        "HH + HL sequence intact",
                    )
                )
        if len(sh) >= 2 and h[sh[-1]] < h[sh[-2]] and len(sl) >= 2 and l[sl[-1]] < l[sl[-2]]:
            if last_bos is None:
                levels.append(
                    StructureLevel(
                        "structure",
                        "bearish",
                        last_swing_low,
                        prev_swing_high,
                        60,
                        "LH + LL sequence intact",
                    )
                )

        return last_bos, last_choch, levels

    def _order_blocks(
        self,
        o: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        atr: float,
    ) -> List[StructureLevel]:
        """
        Order block approx: last opposing candle before impulsive displacement.
        """
        levels: List[StructureLevel] = []
        n = len(c)
        look = min(n - 2, 60)
        for i in range(n - look, n - 1):
            body = abs(c[i] - o[i])
            move_next = c[i + 1] - c[i]
            # Bullish OB: down candle then strong up displacement
            if c[i] < o[i] and move_next > atr * 0.8:
                # displacement continuation
                if i + 3 < n and c[i + 2] > c[i + 1]:
                    levels.append(
                        StructureLevel(
                            kind="order_block",
                            side="bullish",
                            price_low=float(min(o[i], c[i], l[i])),
                            price_high=float(max(o[i], c[i], h[i])),
                            confidence=66,
                            note="Bullish OB before displacement",
                            index=i,
                        )
                    )
            # Bearish OB
            if c[i] > o[i] and move_next < -atr * 0.8:
                if i + 3 < n and c[i + 2] < c[i + 1]:
                    levels.append(
                        StructureLevel(
                            kind="order_block",
                            side="bearish",
                            price_low=float(min(o[i], c[i], l[i])),
                            price_high=float(max(o[i], c[i], h[i])),
                            confidence=66,
                            note="Bearish OB before displacement",
                            index=i,
                        )
                    )

        # Keep most recent few of each side
        bulls = [x for x in levels if x.side == "bullish"][-3:]
        bears = [x for x in levels if x.side == "bearish"][-3:]
        return bulls + bears

    def _fair_value_gaps(
        self,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        atr: float,
    ) -> List[StructureLevel]:
        levels: List[StructureLevel] = []
        n = len(c)
        start = max(2, n - 80)
        for i in range(start, n):
            # Bullish FVG: low[i] > high[i-2]
            if l[i] > h[i - 2] and (l[i] - h[i - 2]) > atr * 0.15:
                gap_low, gap_high = float(h[i - 2]), float(l[i])
                # Unfilled if current price still above gap mid or gap not fully traded through
                filled = c[-1] < gap_low
                if not filled:
                    levels.append(
                        StructureLevel(
                            kind="fvg",
                            side="bullish",
                            price_low=gap_low,
                            price_high=gap_high,
                            confidence=64,
                            note="Bullish FVG (imbalance)",
                            index=i,
                        )
                    )
            # Bearish FVG: high[i] < low[i-2]
            if h[i] < l[i - 2] and (l[i - 2] - h[i]) > atr * 0.15:
                gap_low, gap_high = float(h[i]), float(l[i - 2])
                filled = c[-1] > gap_high
                if not filled:
                    levels.append(
                        StructureLevel(
                            kind="fvg",
                            side="bearish",
                            price_low=gap_low,
                            price_high=gap_high,
                            confidence=64,
                            note="Bearish FVG (imbalance)",
                            index=i,
                        )
                    )
        return levels[-6:]

    def _liquidity_pools(
        self,
        h: np.ndarray,
        l: np.ndarray,
        sh: List[int],
        sl: List[int],
        atr: float,
    ) -> List[StructureLevel]:
        levels: List[StructureLevel] = []
        # Equal highs
        for a, b in zip(sh[-6:], sh[-5:]):
            if abs(h[a] - h[b]) <= atr * 0.15:
                lvl = float((h[a] + h[b]) / 2)
                levels.append(
                    StructureLevel(
                        kind="liquidity",
                        side="bearish",
                        price_low=lvl,
                        price_high=lvl,
                        confidence=62,
                        note="Equal highs liquidity (buy-side stop pool above)",
                        index=b,
                    )
                )
        for a, b in zip(sl[-6:], sl[-5:]):
            if abs(l[a] - l[b]) <= atr * 0.15:
                lvl = float((l[a] + l[b]) / 2)
                levels.append(
                    StructureLevel(
                        kind="liquidity",
                        side="bullish",
                        price_low=lvl,
                        price_high=lvl,
                        confidence=62,
                        note="Equal lows liquidity (sell-side stop pool below)",
                        index=b,
                    )
                )

        # Recent swing high/low as external liquidity
        if sh:
            levels.append(
                StructureLevel(
                    "liquidity",
                    "bearish",
                    float(h[sh[-1]]),
                    float(h[sh[-1]]),
                    55,
                    "External buy-side liquidity (recent swing high)",
                    sh[-1],
                )
            )
        if sl:
            levels.append(
                StructureLevel(
                    "liquidity",
                    "bullish",
                    float(l[sl[-1]]),
                    float(l[sl[-1]]),
                    55,
                    "External sell-side liquidity (recent swing low)",
                    sl[-1],
                )
            )
        return levels

    def _volume_profile(
        self, closes: np.ndarray, volumes: np.ndarray, bins: int = 24
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if len(closes) < 10:
            return None, None, None
        try:
            hist, edges = np.histogram(closes[-120:], bins=bins, weights=volumes[-120:])
            if hist.sum() <= 0:
                return None, None, None
            poc_idx = int(np.argmax(hist))
            poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2)
            # Value area ~70% volume
            total = hist.sum()
            order = list(np.argsort(hist)[::-1])
            acc = 0.0
            selected = set()
            for idx in order:
                selected.add(idx)
                acc += hist[idx]
                if acc >= 0.7 * total:
                    break
            sel = sorted(selected)
            val = float(edges[sel[0]])
            vah = float(edges[sel[-1] + 1])
            return poc, vah, val
        except Exception as exc:  # noqa: BLE001
            logger.debug("Volume profile failed: {}", exc)
            return None, None, None

    def _wyckoff(
        self,
        c: np.ndarray,
        v: np.ndarray,
        h: np.ndarray,
        l: np.ndarray,
        atr: float,
    ) -> Tuple[str, str]:
        """Heuristic Wyckoff phase classification on recent range."""
        n = len(c)
        if n < 40:
            return "unknown", "Not enough history for Wyckoff read"

        window = 50
        seg_c = c[-window:]
        seg_v = v[-window:]
        seg_h = h[-window:]
        seg_l = l[-window:]
        rng = seg_h.max() - seg_l.min()
        pos = (seg_c[-1] - seg_l.min()) / max(rng, 1e-12)
        vol_early = seg_v[: window // 3].mean()
        vol_late = seg_v[-window // 3 :].mean()
        slope = (seg_c[-1] - seg_c[0]) / max(atr * window, 1e-12)

        # Spring: dip below range then reclaim with volume
        range_low = seg_l[: -5].min()
        spring = seg_l[-8:].min() < range_low - atr * 0.1 and seg_c[-1] > range_low
        upthrust = seg_h[-8:].max() > seg_h[: -5].max() + atr * 0.1 and seg_c[-1] < seg_h[: -5].max()

        if spring and vol_late > vol_early:
            return (
                "accumulation (spring)",
                "Potential Wyckoff spring: liquidity grab below range then reclaim on rising volume.",
            )
        if upthrust and vol_late > vol_early:
            return (
                "distribution (UTAD)",
                "Potential upthrust after distribution: false breakout above range then reject.",
            )
        if abs(slope) < 0.15 and rng < atr * 8:
            if pos < 0.4 and vol_late < vol_early:
                return "accumulation", "Sideways base lower-half with declining volume — possible accumulation."
            if pos > 0.6 and vol_late >= vol_early:
                return "distribution", "Sideways range upper-half with sustained volume — possible distribution."
            return "ranging / re-accumulation", "Balanced range; wait for SOS/SOW confirmation."
        if slope > 0.25:
            return "markup", "Advancing trend — markup phase; buy pullbacks to demand."
        if slope < -0.25:
            return "markdown", "Declining trend — markdown phase; sell rallies to supply."
        return "transition", "Mixed structure; no clean Wyckoff phase."

    def _elliott_basics(
        self, c: np.ndarray, sh: List[int], sl: List[int]
    ) -> str:
        """Very rough impulse vs corrective labeling from last 5 swings."""
        if len(sh) < 2 or len(sl) < 2:
            return "Elliott: insufficient swings"
        # Build alternating swing sequence from end
        points: List[Tuple[int, float, str]] = []
        for i in sh[-4:]:
            points.append((i, float(c[i]) if i < len(c) else float(c[-1]), "H"))
        for i in sl[-4:]:
            points.append((i, float(c[i]) if i < len(c) else float(c[-1]), "L"))
        points.sort(key=lambda x: x[0])
        if len(points) < 4:
            return "Elliott: building structure"

        # Net direction of last swings
        net = points[-1][1] - points[0][1]
        zigzag_amp = sum(abs(points[i][1] - points[i - 1][1]) for i in range(1, len(points)))
        efficiency = abs(net) / max(zigzag_amp, 1e-12)
        if efficiency > 0.55 and abs(net) / max(abs(points[0][1]), 1e-12) > 0.02:
            direction = "impulsive up" if net > 0 else "impulsive down"
            return (
                f"Elliott (approx): {direction} — progressive swings with high efficiency "
                f"({efficiency:.2f}). Prefer with-trend entries; avoid catching 'wave 3' knives."
            )
        return (
            f"Elliott (approx): corrective / choppy (efficiency {efficiency:.2f}). "
            "Expect overlapping swings; mean-reversion or wait for break of structure."
        )

    def _score(self, report: StructureReport, price: float, atr: float) -> float:
        score = 0.0
        if report.trend == "up":
            score += 0.35
        elif report.trend == "down":
            score -= 0.35

        if report.last_bos == "bullish":
            score += 0.25
        elif report.last_bos == "bearish":
            score -= 0.25
        if report.last_choch == "bullish":
            score += 0.2
        elif report.last_choch == "bearish":
            score -= 0.2

        # Nearby OBs / FVGs alignment
        for lvl in report.levels:
            if lvl.kind not in ("order_block", "fvg"):
                continue
            dist = abs(price - lvl.mid) / max(atr, 1e-12)
            if dist < 1.5:
                w = 0.08 * (lvl.confidence / 100.0)
                score += w if lvl.side == "bullish" else -w

        if "accumulation" in report.wyckoff_phase:
            score += 0.1
        if "distribution" in report.wyckoff_phase:
            score -= 0.1
        if report.wyckoff_phase == "markup":
            score += 0.15
        if report.wyckoff_phase == "markdown":
            score -= 0.15

        return float(np.clip(score, -1, 1))

    def _summarize(self, report: StructureReport, price: float) -> str:
        parts = [f"Trend structure: {report.trend}"]
        if report.last_bos:
            parts.append(f"BOS={report.last_bos}")
        if report.last_choch:
            parts.append(f"CHoCH={report.last_choch}")
        parts.append(f"Wyckoff≈{report.wyckoff_phase}")
        obs = [l for l in report.levels if l.kind == "order_block"]
        fvgs = [l for l in report.levels if l.kind == "fvg"]
        parts.append(f"OBs={len(obs)} FVGs={len(fvgs)}")
        if report.volume_profile_poc:
            parts.append(f"POC≈{report.volume_profile_poc:.6g}")
        return " | ".join(parts)
