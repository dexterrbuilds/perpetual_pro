"""JSON chart payloads plus a headless PNG renderer for Telegram alerts."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.analysis.indicators import IndicatorSuite
from src.analysis.market_structure import StructureReport
from src.analysis.patterns import PatternReport
from src.analysis.risk import TradePlan


def build_market_chart_payload(
    df: pd.DataFrame,
    indicators: Optional[IndicatorSuite],
    structure: Optional[StructureReport],
    patterns: Optional[PatternReport],
    plan: Optional[TradePlan],
    *,
    timeframe: str,
    limit: int = 140,
) -> Dict[str, Any]:
    """Return candles, overlays, levels, and plan lines without binary images."""
    if df is None or df.empty:
        return {}
    source = indicators.df if indicators is not None and indicators.df is not None else df
    source = source.tail(limit).copy()
    candles: List[Dict[str, Any]] = []
    for ts, row in source.iterrows():
        candles.append(
            {
                "t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "open": _num(row.get("open")),
                "high": _num(row.get("high")),
                "low": _num(row.get("low")),
                "close": _num(row.get("close")),
                "volume": _num(row.get("volume")),
                "ema_fast": _num(row.get("ema_fast")),
                "ema_mid": _num(row.get("ema_mid")),
                "vwap": _num(row.get("vwap")),
            }
        )

    levels: List[Dict[str, Any]] = []
    if structure is not None:
        for level in structure.levels:
            if level.kind not in ("order_block", "fvg", "liquidity", "volume_poc"):
                continue
            levels.append(
                {
                    "kind": level.kind,
                    "side": level.side,
                    "low": level.price_low,
                    "high": level.price_high,
                    "mid": level.mid,
                    "label": level.note or level.kind,
                }
            )
        for name, value in (
            ("POC", structure.volume_profile_poc),
            ("VAH", structure.volume_profile_vah),
            ("VAL", structure.volume_profile_val),
        ):
            if value is not None:
                levels.append(
                    {
                        "kind": "volume_profile",
                        "side": "neutral",
                        "low": value,
                        "high": value,
                        "mid": value,
                        "label": name,
                    }
                )

    pattern_rows = []
    if patterns is not None:
        pattern_rows = [
            {
                "name": hit.name,
                "bias": hit.bias,
                "confidence": hit.confidence,
                "note": hit.note,
            }
            for hit in patterns.top_hits[:6]
        ]

    trade = None
    if plan is not None:
        trade = {
            "direction": plan.direction,
            "entry_low": plan.entry_low,
            "entry_high": plan.entry_high,
            "stop_loss": plan.stop_loss,
            "take_profits": list(plan.take_profits),
            "entry_status": getattr(plan, "entry_status", "blocked"),
            "execution_score": getattr(plan, "execution_score", 0.0),
        }
    return {
        "timeframe": timeframe,
        "candles": candles,
        "levels": levels[:20],
        "patterns": pattern_rows,
        "trade": trade,
    }


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def render_signal_chart_png(
    row: Dict[str, Any],
    *,
    width: int = 1280,
    height: int = 720,
    candle_limit: int = 90,
) -> bytes:
    """Render a compact closed-candle execution chart without a GUI backend."""
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    chart = payload.get("chart") if isinstance(payload.get("chart"), dict) else {}
    candles = list(chart.get("candles") or [])[-max(20, candle_limit) :]
    if len(candles) < 5:
        raise ValueError("Insufficient chart candles for Telegram image")

    image = Image.new("RGB", (width, height), "#07111f")
    draw = ImageDraw.Draw(image)
    font_small = _font(17)
    font_axis = _font(18)
    font_body = _font(20)
    font_title = _font(28, bold=True)

    left, right = 82, width - 176
    top, price_bottom = 82, height - 148
    volume_top, volume_bottom = height - 122, height - 50
    plot_width = right - left
    plot_height = price_bottom - top

    direction = str(row.get("direction") or "flat").upper()
    direction_color = "#25d695" if direction == "LONG" else "#ff5c73"
    symbol = str(row.get("symbol") or "—").split("/")[0]
    timeframe = str(row.get("primary_tf") or chart.get("timeframe") or "15m")
    confidence = _finite(row.get("confidence"), 0.0)
    status = str(row.get("entry_status") or "blocked").replace("_", " ").upper()
    draw.text(
        (left, 26),
        f"{symbol}  {direction}  •  {timeframe} CLOSED CANDLES",
        fill=direction_color,
        font=font_title,
    )
    draw.text(
        (right - 280, 32),
        f"CONF {confidence:.0f}%   {status}",
        fill="#dbeafe",
        font=font_body,
    )

    numeric_candles = []
    for candle in candles:
        values = [_finite(candle.get(k), np.nan) for k in ("open", "high", "low", "close")]
        if all(np.isfinite(values)):
            numeric_candles.append(candle)
    candles = numeric_candles
    if len(candles) < 5:
        raise ValueError("Chart candles contain invalid OHLC values")

    trade = chart.get("trade") if isinstance(chart.get("trade"), dict) else {}
    plan_values = [
        trade.get("entry_low"),
        trade.get("entry_high"),
        trade.get("stop_loss"),
        *(trade.get("take_profits") or [])[:4],
    ]
    price_values = [
        *[_finite(candle.get("high"), np.nan) for candle in candles],
        *[_finite(candle.get("low"), np.nan) for candle in candles],
        *[_finite(value, np.nan) for value in plan_values],
    ]
    price_values = [value for value in price_values if np.isfinite(value) and value > 0]
    if not price_values:
        raise ValueError("No finite chart prices")
    price_min, price_max = min(price_values), max(price_values)
    padding = max((price_max - price_min) * 0.08, price_max * 0.001)
    price_min -= padding
    price_max += padding

    def y_of(value: Any) -> float:
        price = _finite(value, price_min)
        return price_bottom - (price - price_min) / max(price_max - price_min, 1e-12) * plot_height

    # Grid and price labels.
    for i in range(6):
        y = top + plot_height * i / 5
        price = price_max - (price_max - price_min) * i / 5
        draw.line((left, y, right, y), fill="#18283b", width=1)
        draw.text(
            (right + 10, y - 10),
            _price(price),
            fill="#91a4bb",
            font=font_axis,
        )
    for i in range(7):
        x = left + plot_width * i / 6
        draw.line((x, top, x, volume_bottom), fill="#112235", width=1)

    step = plot_width / max(len(candles), 1)
    candle_width = max(3, min(11, int(step * 0.62)))
    max_volume = max(_finite(candle.get("volume"), 0.0) for candle in candles) or 1.0
    for index, candle in enumerate(candles):
        x = left + (index + 0.5) * step
        open_ = _finite(candle.get("open"))
        high = _finite(candle.get("high"))
        low = _finite(candle.get("low"))
        close = _finite(candle.get("close"))
        bullish = close >= open_
        color = "#20c997" if bullish else "#f35b72"
        draw.line((x, y_of(high), x, y_of(low)), fill=color, width=2)
        body_top, body_bottom = sorted((y_of(open_), y_of(close)))
        if body_bottom - body_top < 2:
            body_bottom = body_top + 2
        draw.rectangle(
            (
                x - candle_width / 2,
                body_top,
                x + candle_width / 2,
                body_bottom,
            ),
            fill=color,
        )
        volume = _finite(candle.get("volume"), 0.0)
        volume_height = volume / max_volume * (volume_bottom - volume_top)
        draw.rectangle(
            (
                x - candle_width / 2,
                volume_bottom - volume_height,
                x + candle_width / 2,
                volume_bottom,
            ),
            fill="#147f69" if bullish else "#8f3348",
        )

    legend_index = 0
    for key, label, color in (
        ("ema_fast", "EMA 9", "#f6b94a"),
        ("ema_mid", "EMA 21", "#54bffc"),
        ("vwap", "VWAP", "#b59cff"),
    ):
        points = []
        for index, candle in enumerate(candles):
            value = _finite(candle.get(key), np.nan)
            if np.isfinite(value):
                points.append((left + (index + 0.5) * step, y_of(value)))
        if len(points) >= 2:
            draw.line(points, fill=color, width=2)
            draw.text(
                (left + 8, top + 8 + 23 * legend_index),
                label,
                fill=color,
                font=font_small,
            )
            legend_index += 1

    # Nearby structure levels stay subtle so the actual trade plan is dominant.
    for level in list(chart.get("levels") or [])[:10]:
        value = _finite(level.get("mid"), np.nan)
        if not np.isfinite(value) or not (price_min <= value <= price_max):
            continue
        side = str(level.get("side") or "neutral")
        color = "#236b59" if side == "bullish" else ("#713849" if side == "bearish" else "#40566f")
        _dashed_line(draw, left, right, y_of(value), color, dash=8, gap=8, width=1)

    entry_low = _finite(trade.get("entry_low"), np.nan)
    entry_high = _finite(trade.get("entry_high"), np.nan)
    if np.isfinite(entry_low) and np.isfinite(entry_high):
        y0, y1 = sorted((y_of(entry_low), y_of(entry_high)))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle((left, y0, right, y1), fill=(246, 185, 74, 42))
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)
        _trade_line(
            draw,
            left,
            right,
            (entry_low + entry_high) / 2,
            y_of,
            "ENTRY",
            "#f6b94a",
            font_body,
        )

    stop = _finite(trade.get("stop_loss"), np.nan)
    if np.isfinite(stop):
        _trade_line(draw, left, right, stop, y_of, "SL", "#ff4967", font_body, width=3)
    target_rows = []
    for index, target in enumerate((trade.get("take_profits") or [])[:4], 1):
        value = _finite(target, np.nan)
        if np.isfinite(value):
            target_rows.append((index, value, y_of(value)))
    label_positions = _spread_label_positions(
        [row[2] for row in target_rows],
        top + 14,
        price_bottom - 14,
        min_gap=28,
    )
    for (index, value, _), label_y in zip(target_rows, label_positions):
        _trade_line(
            draw,
            left,
            right,
            value,
            y_of,
            f"TP{index}",
            "#20c997",
            font_body,
            label_y=label_y,
        )

    draw.text((left, volume_top - 25), "VOLUME", fill="#91a4bb", font=font_small)
    draw.text(
        (left, height - 28),
        "Perpetual Pro • 15m execution / 1h + 4h confirmation • educational only",
        fill="#60758e",
        font=font_small,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    left: float,
    right: float,
    y: float,
    color: str,
    *,
    dash: int,
    gap: int,
    width: int,
) -> None:
    x = left
    while x < right:
        draw.line((x, y, min(x + dash, right), y), fill=color, width=width)
        x += dash + gap


def _trade_line(
    draw: ImageDraw.ImageDraw,
    left: float,
    right: float,
    value: float,
    y_of: Any,
    label: str,
    color: str,
    font: ImageFont.ImageFont,
    width: int = 2,
    label_y: Optional[float] = None,
) -> None:
    y = y_of(value)
    draw.line((left, y, right, y), fill=color, width=width)
    text_y = y if label_y is None else label_y
    if abs(text_y - y) > 1:
        draw.line((right - 22, y, right + 2, text_y), fill=color, width=1)
    draw.rectangle(
        (right + 2, text_y - 13, right + 162, text_y + 13),
        fill="#07111f",
    )
    draw.text(
        (right + 8, text_y - 12),
        f"{label} {_price(value)}",
        fill=color,
        font=font,
    )


def _spread_label_positions(
    positions: List[float],
    minimum: float,
    maximum: float,
    *,
    min_gap: float,
) -> List[float]:
    """Prevent nearby TP labels from becoming unreadable on mobile."""
    if not positions:
        return []
    indexed = sorted(enumerate(positions), key=lambda item: item[1])
    spread: List[Tuple[int, float]] = []
    previous = minimum - min_gap
    for original_index, position in indexed:
        placed = max(minimum, position, previous + min_gap)
        spread.append((original_index, placed))
        previous = placed
    overflow = max(0.0, spread[-1][1] - maximum)
    if overflow:
        spread = [(index, value - overflow) for index, value in spread]
        for idx in range(len(spread) - 2, -1, -1):
            next_value = spread[idx + 1][1]
            spread[idx] = (
                spread[idx][0],
                min(spread[idx][1], next_value - min_gap),
            )
    mapped = {index: max(minimum, min(maximum, value)) for index, value in spread}
    return [mapped[index] for index in range(len(positions))]
