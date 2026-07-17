"""Dual OCR engines (Tesseract + EasyOCR) for chart metadata extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from src.utils.config import AppConfig, OCRConfig
from src.utils.helpers import normalize_symbol, safe_float
from src.vision.preprocess import cv_to_pil, preprocess_chart_image


@dataclass
class OCRResult:
    raw_text: str = ""
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    prices: List[float] = field(default_factory=list)
    indicators_mentioned: List[str] = field(default_factory=list)
    confidence: float = 0.0
    engine_notes: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class OCREngine:
    """Run Tesseract and/or EasyOCR and parse trading chart text."""

    TIMEFRAME_RE = re.compile(
        r"\b(\d+[mhdw]|1M|3M|45m|2h|6h|8h|12h|1D|4H|15m|5m|1h|1d)\b",
        re.IGNORECASE,
    )
    SYMBOL_RE = re.compile(
        r"\b([A-Z]{2,10})[/\-_]?(USDT|USDC|USD|PERP)?\b"
        r"|\b(1000[A-Z]{2,8})(USDT)?\b"
    )
    PRICE_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+\.\d+\b")
    INDICATOR_KEYWORDS = [
        "RSI",
        "MACD",
        "EMA",
        "SMA",
        "VWAP",
        "ATR",
        "BB",
        "BOLL",
        "STOCH",
        "VOLUME",
        "OI",
        "FUNDING",
        "SUPER TREND",
        "ICHIMOKU",
        "CVD",
        "OBV",
    ]

    def __init__(self, config: Optional[AppConfig] = None, ocr_cfg: Optional[OCRConfig] = None):
        self.config = config
        self.ocr_cfg = ocr_cfg or (config.ocr if config else OCRConfig())
        self._easyocr_reader = None
        self._tesseract_ready = None

        if self.ocr_cfg.tesseract_cmd:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = self.ocr_cfg.tesseract_cmd
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not set tesseract_cmd: {}", exc)

    def extract(self, image: Image.Image, dark_theme: Optional[bool] = None) -> OCRResult:
        processed, meta = preprocess_chart_image(image, dark_theme=dark_theme, for_ocr=True)
        pil_img = cv_to_pil(processed) if isinstance(processed, np.ndarray) else processed

        engine = (self.ocr_cfg.engine or "dual").lower()
        texts: List[Tuple[str, float, str]] = []  # text, conf, engine

        if engine in ("tesseract", "dual"):
            t_text, t_conf = self._run_tesseract(pil_img)
            if t_text.strip():
                texts.append((t_text, t_conf, "tesseract"))

        if engine in ("easyocr", "dual"):
            e_text, e_conf = self._run_easyocr(processed)
            if e_text.strip():
                texts.append((e_text, e_conf, "easyocr"))

        if not texts:
            # Last resort: OCR on original without preprocess
            t_text, t_conf = self._run_tesseract(image)
            if t_text.strip():
                texts.append((t_text, t_conf, "tesseract_raw"))

        combined = "\n".join(t[0] for t in texts)
        conf = float(np.mean([t[1] for t in texts])) if texts else 0.0
        result = OCRResult(
            raw_text=combined,
            confidence=conf,
            engine_notes=[f"{t[2]}:{t[1]:.2f}" for t in texts],
            lines=[ln.strip() for ln in combined.splitlines() if ln.strip()],
            meta={"preprocess": meta},
        )
        self._parse(result)
        logger.info(
            "OCR symbol={} tf={} conf={:.2f} engines={}",
            result.symbol,
            result.timeframe,
            result.confidence,
            result.engine_notes,
        )
        return result

    def _run_tesseract(self, image: Image.Image) -> Tuple[str, float]:
        try:
            import pytesseract

            # PSM 6 = assume uniform block of text; also try 11 sparse
            config = "--psm 6"
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
            words = []
            confs = []
            for txt, conf in zip(data.get("text", []), data.get("conf", [])):
                if not txt or not str(txt).strip():
                    continue
                try:
                    c = float(conf)
                except ValueError:
                    c = 0.0
                if c < 0:
                    continue
                words.append(str(txt).strip())
                confs.append(c / 100.0)
            text = " ".join(words)
            if not text:
                text = pytesseract.image_to_string(image, config=config) or ""
            avg = float(np.mean(confs)) if confs else (0.4 if text.strip() else 0.0)
            return text, avg
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tesseract OCR failed: {}", exc)
            return "", 0.0

    def _run_easyocr(self, image: np.ndarray | Image.Image) -> Tuple[str, float]:
        try:
            reader = self._get_easyocr()
            if reader is None:
                return "", 0.0
            if isinstance(image, Image.Image):
                arr = np.array(image.convert("RGB"))
            else:
                arr = image
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
            # EasyOCR expects RGB
            results = reader.readtext(arr, detail=1, paragraph=False)
            parts = []
            confs = []
            for item in results:
                # (bbox, text, conf)
                if len(item) >= 3:
                    parts.append(str(item[1]))
                    confs.append(float(item[2]))
            text = " ".join(parts)
            avg = float(np.mean(confs)) if confs else 0.0
            return text, avg
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR failed: {}", exc)
            return "", 0.0

    def _get_easyocr(self):
        if self._easyocr_reader is not None:
            return self._easyocr_reader
        try:
            import easyocr

            langs = self.ocr_cfg.languages or ["en"]
            self._easyocr_reader = easyocr.Reader(langs, gpu=self.ocr_cfg.easyocr_gpu, verbose=False)
            return self._easyocr_reader
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR init failed: {}", exc)
            return None

    def _parse(self, result: OCRResult) -> None:
        text = result.raw_text
        upper = text.upper()

        # Timeframe
        tf_matches = self.TIMEFRAME_RE.findall(text)
        if tf_matches:
            # Prefer common chart TFs
            priority = ["15m", "5m", "1h", "4h", "1d", "1D", "4H", "1H", "30m", "1m"]
            chosen = None
            for p in priority:
                for m in tf_matches:
                    if str(m).lower() == p.lower():
                        chosen = str(m).lower().replace("d", "d").replace("h", "h")
                        break
                if chosen:
                    break
            result.timeframe = (chosen or str(tf_matches[0])).lower()
            # normalize 1D -> 1d
            result.timeframe = result.timeframe.replace("d", "d")
            if result.timeframe.endswith("d") and result.timeframe[0].isdigit():
                result.timeframe = result.timeframe.lower()
            if result.timeframe.endswith("h"):
                result.timeframe = result.timeframe.lower()

        # Symbol — look for BTCUSDT style and reject common UI words
        blacklist = {
            "USD", "USDT", "USDC", "PERP", "SPOT", "LONG", "SHORT", "BUY", "SELL",
            "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "PRICE", "CHART", "TIME",
            "INDICATOR", "INTERVAL", "BINANCE", "BYBIT", "OKX", "BITGET",
            "UTC", "GMT", "CROSS", "ISOLATED", "MARKET", "LIMIT", "THE", "AND",
            "FOR", "RSI", "MACD", "EMA", "SMA", "ATR", "PnL".upper(), "ROE",
        }
        candidates: List[str] = []
        for m in self.SYMBOL_RE.finditer(upper):
            raw = m.group(0).replace(" ", "")
            base = m.group(1) or m.group(3)
            if not base or base in blacklist:
                continue
            if len(base) < 2:
                continue
            candidates.append(raw)

        # Also scan for known majors explicitly
        for major in ("BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "AVAX", "LINK", "DOT"):
            if re.search(rf"\b{major}(USDT|USD|PERP)?\b", upper):
                candidates.insert(0, f"{major}USDT")

        if candidates:
            try:
                result.symbol = normalize_symbol(candidates[0])
            except ValueError:
                result.symbol = candidates[0]

        # Prices
        prices = []
        for m in self.PRICE_RE.findall(text.replace(",", "")):
            val = safe_float(m.replace(",", ""), default=float("nan"))
            if val != val:  # nan
                continue
            if val <= 0:
                continue
            # filter out years and pure integers that look like dates
            if 1900 < val < 2100 and val == int(val):
                continue
            prices.append(val)
        # Unique sorted
        result.prices = sorted(set(round(p, 10) for p in prices))[:40]

        # Indicators mentioned
        inds = []
        for kw in self.INDICATOR_KEYWORDS:
            if kw in upper:
                inds.append(kw)
        result.indicators_mentioned = inds
