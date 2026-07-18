"""Screen capture, OCR, and chart computer vision.

Desktop-only modules (mss capture) are imported lazily so the API can boot
on servers without those packages.
"""

from .chart_detect import ChartVision, VisionChartResult
from .ocr import OCREngine, OCRResult
from .preprocess import preprocess_chart_image
from .url_symbol import UrlHints, parse_chart_url

__all__ = [
    "ScreenCapture",
    "CaptureResult",
    "ChartVision",
    "VisionChartResult",
    "OCREngine",
    "OCRResult",
    "preprocess_chart_image",
    "UrlHints",
    "parse_chart_url",
]


def __getattr__(name: str):
    """Lazy-load desktop capture helpers (require mss)."""
    if name in ("ScreenCapture", "CaptureResult"):
        from .capture import CaptureResult, ScreenCapture

        return ScreenCapture if name == "ScreenCapture" else CaptureResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
