"""Image preprocessing optimized for TradingView / exchange chart OCR + CV."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL RGB → OpenCV BGR."""
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    """OpenCV BGR or gray → PIL RGB."""
    if img.ndim == 2:
        return Image.fromarray(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def estimate_is_dark(img_bgr: np.ndarray) -> bool:
    """Heuristic: mean luminance below mid-gray → dark theme."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return float(np.mean(gray)) < 100.0


def preprocess_chart_image(
    image: Image.Image | np.ndarray,
    dark_theme: Optional[bool] = None,
    for_ocr: bool = True,
    scale: float = 1.5,
) -> Tuple[np.ndarray, dict]:
    """
    Heavy preprocessing pipeline for charts.

    Returns (processed_bgr_or_gray, meta).
    When for_ocr=True, returns a high-contrast grayscale optimized for text.
    When for_ocr=False, returns BGR suitable for candle/line detection.
    """
    if isinstance(image, Image.Image):
        bgr = pil_to_cv(image)
    else:
        bgr = image.copy()
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    meta: dict = {"orig_shape": bgr.shape[:2]}

    # Upscale small captures for better OCR/CV
    h, w = bgr.shape[:2]
    if scale and scale != 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        meta["scale"] = scale

    if dark_theme is None:
        dark_theme = estimate_is_dark(bgr)
    meta["dark_theme"] = dark_theme

    # Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 6, 6, 7, 21)

    # Mild contrast via CLAHE on L channel
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    meta["clahe"] = True

    if not for_ocr:
        # Slight sharpen for structure detection
        blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharp = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)
        meta["mode"] = "cv"
        return sharp, meta

    # --- OCR path ---
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Invert dark themes so text is dark on light (Tesseract prefers this)
    if dark_theme:
        gray = cv2.bitwise_not(gray)
        meta["inverted"] = True

    # Adaptive threshold for variable chart backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    # Morph open to remove pepper noise
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optional Otsu blend for large text regions
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mixed = cv2.bitwise_and(binary, otsu)

    meta["mode"] = "ocr"
    return mixed, meta


def extract_chart_region(bgr: np.ndarray) -> np.ndarray:
    """
    Attempt to crop to the main candlestick pane (largest dark/content region).
    Falls back to original if detection is weak.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Edge map
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr
    # Largest contour by area with reasonable aspect
    best = None
    best_area = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < (h * w * 0.15):
            continue
        aspect = cw / max(ch, 1)
        if 0.8 < aspect < 6 and area > best_area:
            best_area = area
            best = (x, y, cw, ch)
    if best is None:
        # Default: drop top 8% (symbol bar) and bottom 12% (time axis / volume labels)
        y0, y1 = int(h * 0.06), int(h * 0.92)
        x0, x1 = int(w * 0.02), int(w * 0.92)
        return bgr[y0:y1, x0:x1]
    x, y, cw, ch = best
    pad = 4
    y0 = max(0, y - pad)
    x0 = max(0, x - pad)
    y1 = min(h, y + ch + pad)
    x1 = min(w, x + cw + pad)
    return bgr[y0:y1, x0:x1]


def annotate_levels(
    image: Image.Image | np.ndarray,
    levels: list,
    entry: Optional[tuple] = None,
    stop: Optional[float] = None,
    tps: Optional[list] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
) -> Image.Image:
    """
    Draw horizontal levels on chart image.

    levels: list of dicts with 'mid' or 'price' and optional 'side'/'kind'
    price_min/max map prices to y if known; else distribute heuristically.
    """
    if isinstance(image, Image.Image):
        bgr = pil_to_cv(image)
    else:
        bgr = image.copy()
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    h, w = bgr.shape[:2]
    prices = []
    for lv in levels or []:
        p = lv.get("mid") or lv.get("price") or lv.get("high")
        if p is not None:
            prices.append(float(p))
    if stop is not None:
        prices.append(float(stop))
    if tps:
        prices.extend(float(t) for t in tps)
    if entry:
        prices.extend([float(entry[0]), float(entry[1])])

    if not prices:
        return cv_to_pil(bgr)

    pmin = price_min if price_min is not None else min(prices) * 0.995
    pmax = price_max if price_max is not None else max(prices) * 1.005
    if pmax <= pmin:
        pmax = pmin + 1.0

    def y_for(price: float) -> int:
        # Screen coords: higher price → lower y
        t = (price - pmin) / (pmax - pmin)
        return int(np.clip((1.0 - t) * (h * 0.85) + h * 0.05, 0, h - 1))

    def draw_line(price: float, color: tuple, label: str, thickness: int = 1) -> None:
        y = y_for(price)
        cv2.line(bgr, (0, y), (w, y), color, thickness, cv2.LINE_AA)
        cv2.putText(
            bgr,
            f"{label} {price:.6g}",
            (8, max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    for lv in levels or []:
        p = lv.get("mid") or lv.get("price")
        if p is None:
            continue
        side = (lv.get("side") or "").lower()
        color = (80, 200, 80) if side == "bullish" else ((80, 80, 220) if side == "bearish" else (180, 180, 80))
        draw_line(float(p), color, str(lv.get("kind", "lvl"))[:8])

    if entry:
        mid = (float(entry[0]) + float(entry[1])) / 2
        draw_line(mid, (255, 200, 50), "ENTRY", 2)
    if stop is not None:
        draw_line(float(stop), (50, 50, 255), "SL", 2)
    if tps:
        for i, tp in enumerate(tps, 1):
            draw_line(float(tp), (50, 220, 50), f"TP{i}", 1)

    return cv_to_pil(bgr)
