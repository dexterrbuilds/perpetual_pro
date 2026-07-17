"""Computer vision: candlesticks, trendlines, S/R, volume bars (+ optional Ollama)."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from loguru import logger
from PIL import Image

from src.utils.config import AppConfig, VisionConfig
from src.vision.preprocess import extract_chart_region, pil_to_cv, preprocess_chart_image


@dataclass
class CandleBlob:
    x: int
    y_open: int
    y_close: int
    y_high: int
    y_low: int
    bullish: bool


@dataclass
class VisionChartResult:
    candles_detected: int = 0
    trendline_slopes: List[float] = field(default_factory=list)
    horizontal_levels_y: List[int] = field(default_factory=list)
    approx_support_ys: List[int] = field(default_factory=list)
    approx_resistance_ys: List[int] = field(default_factory=list)
    volume_bars: int = 0
    trend_guess: str = "unknown"  # up | down | range | unknown
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)
    ollama_summary: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.candles_detected >= 5 or self.confidence >= 0.35


class ChartVision:
    """Detect structural chart elements from a screenshot."""

    def __init__(self, config: Optional[AppConfig] = None, vision_cfg: Optional[VisionConfig] = None):
        self.config = config
        self.vision_cfg = vision_cfg or (config.vision if config else VisionConfig())

    def analyze(self, image: Image.Image, dark_theme: Optional[bool] = None) -> VisionChartResult:
        bgr_raw = pil_to_cv(image)
        chart = extract_chart_region(bgr_raw)
        processed, meta = preprocess_chart_image(chart, dark_theme=dark_theme, for_ocr=False, scale=1.0)
        result = VisionChartResult(meta={"preprocess": meta, "shape": processed.shape[:2]})

        try:
            candles = self._detect_candles(processed)
            result.candles_detected = len(candles)
            if candles:
                bulls = sum(1 for c in candles if c.bullish)
                bears = len(candles) - bulls
                result.notes.append(f"Candles≈{len(candles)} (bull {bulls} / bear {bears})")
                # Micro trend from last third closes
                last = candles[-max(5, len(candles) // 3) :]
                closes = [c.y_close for c in last]
                # y increases downward → decreasing y = rising price
                if closes[0] - closes[-1] > 5:
                    result.trend_guess = "up"
                elif closes[-1] - closes[0] > 5:
                    result.trend_guess = "down"
                else:
                    result.trend_guess = "range"
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"candle_detect_error: {exc}")
            logger.debug("Candle detect failed: {}", exc)

        try:
            slopes, h_ys = self._detect_lines(processed)
            result.trendline_slopes = slopes
            result.horizontal_levels_y = h_ys
            if h_ys:
                mid = processed.shape[0] / 2
                result.approx_support_ys = sorted([y for y in h_ys if y > mid])[:5]
                result.approx_resistance_ys = sorted([y for y in h_ys if y <= mid])[:5]
                result.notes.append(f"Horizontal levels≈{len(h_ys)}")
            if slopes:
                avg = float(np.mean(slopes))
                result.notes.append(f"Trendline avg slope={avg:.4f}")
                if result.trend_guess == "unknown":
                    # image slope: positive slope in x-y (y down) often means falling price
                    result.trend_guess = "down" if avg > 0.05 else ("up" if avg < -0.05 else "range")
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"line_detect_error: {exc}")

        try:
            result.volume_bars = self._detect_volume_bars(processed)
            if result.volume_bars:
                result.notes.append(f"Volume bars≈{result.volume_bars}")
        except Exception as exc:  # noqa: BLE001
            result.notes.append(f"volume_detect_error: {exc}")

        # Confidence
        conf = 0.0
        conf += min(0.5, result.candles_detected / 80.0)
        conf += min(0.2, len(result.horizontal_levels_y) / 20.0)
        conf += 0.1 if result.volume_bars > 5 else 0.0
        conf += 0.1 if result.trend_guess != "unknown" else 0.0
        result.confidence = float(min(0.95, conf))

        # Optional Ollama vision
        if self.vision_cfg.use_ollama:
            summary = self._ollama_describe(image)
            if summary:
                result.ollama_summary = summary
                result.notes.append("Ollama vision available")
                result.confidence = min(0.95, result.confidence + 0.1)

        if not result.ok:
            result.notes.append("Vision weak — prefer data-mode fallback with OCR symbol")

        return result

    def _detect_candles(self, bgr: np.ndarray) -> List[CandleBlob]:
        """
        Detect vertical candle-like blobs via color masks (green/red/white/gray).
        Approximate — works best on standard TradingView color schemes.
        """
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Green-ish bodies
        mask_g1 = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
        # Red-ish bodies (wrap hue)
        mask_r1 = cv2.inRange(hsv, (0, 50, 40), (15, 255, 255))
        mask_r2 = cv2.inRange(hsv, (165, 50, 40), (180, 255, 255))
        mask_r = cv2.bitwise_or(mask_r1, mask_r2)

        candles: List[CandleBlob] = []
        for mask, bullish in ((mask_g1, True), (mask_r, False)):
            # Clean
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if ch < 4 or cw < 1:
                    continue
                if ch > h * 0.6 or cw > w * 0.05:
                    continue
                aspect = ch / max(cw, 1)
                if aspect < 0.8:
                    continue
                # Approximate wick extension via vertical scan in gray
                candles.append(
                    CandleBlob(
                        x=x + cw // 2,
                        y_open=y + ch if bullish else y,
                        y_close=y if bullish else y + ch,
                        y_high=y,
                        y_low=y + ch,
                        bullish=bullish,
                    )
                )

        candles.sort(key=lambda c: c.x)
        # Deduplicate by x proximity
        deduped: List[CandleBlob] = []
        for c in candles:
            if deduped and abs(c.x - deduped[-1].x) < 3:
                continue
            deduped.append(c)
        return deduped

    def _detect_lines(self, bgr: np.ndarray) -> Tuple[List[float], List[int]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=bgr.shape[1] // 8, maxLineGap=12
        )
        slopes: List[float] = []
        horizontals: List[int] = []
        if lines is None:
            return slopes, horizontals

        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 2:
                continue
            slope = dy / dx
            length = np.hypot(dx, dy)
            if abs(slope) < 0.03 and length > bgr.shape[1] * 0.15:
                horizontals.append(int((y1 + y2) / 2))
            elif 0.03 < abs(slope) < 1.5 and length > bgr.shape[1] * 0.12:
                slopes.append(float(slope))

        # Cluster horizontal y's
        horizontals = self._cluster_positions(horizontals, tol=6)
        return slopes[:20], horizontals[:15]

    def _detect_volume_bars(self, bgr: np.ndarray) -> int:
        """Count vertical bars in bottom ~20% of image."""
        h, w = bgr.shape[:2]
        roi = bgr[int(h * 0.78) :, :]
        if roi.size == 0:
            return 0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Threshold bright-ish bars on dark / dark bars on light
        mean = float(np.mean(gray))
        if mean < 100:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch > 3 and cw < w * 0.05 and ch > cw:
                count += 1
        return count

    @staticmethod
    def _cluster_positions(vals: List[int], tol: int = 5) -> List[int]:
        if not vals:
            return []
        vals = sorted(vals)
        clusters: List[List[int]] = [[vals[0]]]
        for v in vals[1:]:
            if abs(v - clusters[-1][-1]) <= tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [int(np.mean(c)) for c in clusters]

    def _ollama_describe(self, image: Image.Image) -> str:
        """Optional local LLaVA/vision model via Ollama HTTP API."""
        url = self.vision_cfg.ollama_base_url.rstrip("/") + "/api/generate"
        try:
            # Quick health check
            tags = requests.get(
                self.vision_cfg.ollama_base_url.rstrip("/") + "/api/tags",
                timeout=2,
            )
            if not tags.ok:
                return ""
        except Exception:
            logger.debug("Ollama not reachable — skipping vision LLM")
            return ""

        buf = io.BytesIO()
        # Shrink for speed
        img = image.copy()
        img.thumbnail((1280, 720))
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        prompt = (
            "You are a professional crypto futures trader. Look at this chart screenshot. "
            "In 4-6 concise sentences: identify likely symbol/timeframe if visible, "
            "trend direction, key support/resistance, candlestick or chart patterns, "
            "and a high-level trade bias (long/short/neutral) with reasoning. "
            "Be precise; if unsure say so."
        )
        payload = {
            "model": self.vision_cfg.ollama_model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.vision_cfg.ollama_timeout_s)
            if not resp.ok:
                logger.debug("Ollama generate failed: {}", resp.status_code)
                return ""
            data = resp.json()
            return (data.get("response") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ollama describe error: {}", exc)
            return ""
