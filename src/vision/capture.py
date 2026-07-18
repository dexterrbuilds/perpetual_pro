"""Screen capture via mss with interactive or region modes.

``mss`` is optional — only required for local screen capture CLI features.
The FastAPI service never imports this at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

from loguru import logger
from PIL import Image

from src.utils.helpers import ensure_dir, utc_now_iso

try:
    import mss
    import mss.tools  # noqa: F401
except ImportError:  # pragma: no cover
    mss = None  # type: ignore


@dataclass
class CaptureResult:
    image: Image.Image
    region: dict  # mon / left / top / width / height
    path: Optional[Path] = None
    captured_at: str = field(default_factory=utc_now_iso)
    mode: str = "full"


def _require_mss() -> None:
    if mss is None:
        raise RuntimeError(
            "Screen capture requires the 'mss' package. "
            "Install with: pip install mss"
        )


class ScreenCapture:
    """Capture full screen, monitor, or explicit region."""

    def __init__(self, output_dir: Union[str, Path] = "./output") -> None:
        self.output_dir = ensure_dir(output_dir)

    def list_monitors(self) -> list:
        _require_mss()
        with mss.mss() as sct:
            return list(sct.monitors)

    def capture_full(self, monitor: int = 0, save: bool = True) -> CaptureResult:
        """
        monitor: 0 = all monitors virtual, 1+ = specific monitor
        """
        _require_mss()
        with mss.mss() as sct:
            monitors = sct.monitors
            if monitor < 0 or monitor >= len(monitors):
                raise ValueError(
                    f"Invalid monitor index {monitor}; available 0..{len(monitors)-1}"
                )
            mon = monitors[monitor]
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            result = CaptureResult(image=img, region=dict(mon), mode="full")
            if save:
                result.path = self._save(img, "full")
            logger.info("Captured full screen {}x{}", img.width, img.height)
            return result

    def capture_region(
        self,
        left: int,
        top: int,
        width: int,
        height: int,
        save: bool = True,
    ) -> CaptureResult:
        _require_mss()
        region = {
            "left": int(left),
            "top": int(top),
            "width": int(width),
            "height": int(height),
        }
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        with mss.mss() as sct:
            shot = sct.grab(region)
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            result = CaptureResult(image=img, region=region, mode="region")
            if save:
                result.path = self._save(img, "region")
            logger.info("Captured region {}x{} at ({},{})", width, height, left, top)
            return result

    def capture_interactive(self, save: bool = True) -> CaptureResult:
        """
        Interactive region selection.

        Tries tkinter drag-select; falls back to full primary monitor with
        instructions if GUI is unavailable.
        """
        try:
            region = self._tk_select_region()
            if region:
                return self.capture_region(*region, save=save)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Interactive selection failed ({}). Falling back to full screen.", exc
            )

        logger.info(
            "Using full primary monitor. Tip: pass --region L,T,W,H for precise crop."
        )
        return self.capture_full(
            monitor=1 if len(self.list_monitors()) > 1 else 0, save=save
        )

    def capture_from_file(self, path: Union[str, Path]) -> CaptureResult:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        img = Image.open(p).convert("RGB")
        return CaptureResult(
            image=img,
            region={
                "left": 0,
                "top": 0,
                "width": img.width,
                "height": img.height,
                "file": str(p),
            },
            path=p,
            mode="file",
        )

    def _save(self, img: Image.Image, tag: str) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"capture_{tag}_{ts}.png"
        img.save(path)
        logger.debug("Saved capture to {}", path)
        return path

    def _tk_select_region(self) -> Optional[Tuple[int, int, int, int]]:
        """Fullscreen translucent overlay to drag a rectangle."""
        import tkinter as tk

        result: dict = {}

        root = tk.Tk()
        root.attributes("-alpha", 0.3)
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        root.configure(bg="black")
        root.title("perpetual_pro — drag to select chart region (ESC cancel)")

        canvas = tk.Canvas(root, cursor="cross", bg="gray20", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        state = {"x0": 0, "y0": 0, "rect": None}

        def on_press(event):
            state["x0"], state["y0"] = event.x, event.y
            if state["rect"]:
                canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="lime", width=2
            )

        def on_drag(event):
            if state["rect"]:
                canvas.coords(state["rect"], state["x0"], state["y0"], event.x, event.y)

        def on_release(event):
            x0, y0 = state["x0"], state["y0"]
            x1, y1 = event.x, event.y
            left, top = min(x0, x1), min(y0, y1)
            width, height = abs(x1 - x0), abs(y1 - y0)
            sx = root.winfo_rootx()
            sy = root.winfo_rooty()
            if width > 10 and height > 10:
                result["region"] = (left + sx, top + sy, width, height)
            root.destroy()

        def on_escape(_event):
            root.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.bind("<Escape>", on_escape)

        canvas.create_text(
            root.winfo_screenwidth() // 2,
            40,
            text="Drag to select chart · ESC to cancel",
            fill="white",
            font=("Helvetica", 18, "bold"),
        )
        root.mainloop()
        return result.get("region")
