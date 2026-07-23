from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.api.service import AnalyzeRequest, SCAN_LEVERAGE_CAP
from src.utils.config import load_config
from src.utils.helpers import format_price


st.set_page_config(page_title="Perpetual Pro", page_icon="📈", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Default watchlist (matches extension / prior web app coins)
DEFAULT_WATCHLIST = (
    "BTC,ETH,SOL,BNB,AAVE,ARB,NEAR,INJ,SEI,TIA,SUI,APT,AVAX,TRX,UNI"
)

EXCHANGE_OPTIONS = [
    "bybit",
    "binanceusdm",
    "okx",
    "bitget",
    "mexc",
    "bingx",
    "bitmart",
    "gate",
    "htx",
    "weex",
]

SIM_BASE_USD = 100.0


def _app_css() -> str:
    """Theme-aware visual system shared by Streamlit light and dark modes."""
    return """
<style>
    :root {
        --pp-border: color-mix(in srgb, var(--text-color) 14%, transparent);
        --pp-muted: color-mix(in srgb, var(--text-color) 66%, transparent);
        --pp-primary-soft: color-mix(in srgb, var(--primary-color) 13%, transparent);
        --pp-surface: color-mix(
            in srgb,
            var(--secondary-background-color) 86%,
            var(--background-color)
        );
    }

    .stApp {
        background:
            radial-gradient(
                circle at 74% -12%,
                color-mix(in srgb, var(--primary-color) 9%, transparent),
                transparent 34rem
            ),
            var(--background-color);
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--pp-border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
    }

    .pp-hero {
        padding: 1.7rem 1.85rem;
        margin-bottom: 1.15rem;
        border: 1px solid var(--pp-border);
        border-radius: 1.15rem;
        background:
            linear-gradient(135deg, var(--pp-primary-soft), transparent 52%),
            var(--pp-surface);
        box-shadow: 0 14px 36px color-mix(in srgb, var(--text-color) 7%, transparent);
    }

    .pp-eyebrow {
        color: var(--primary-color);
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .pp-title {
        color: var(--text-color);
        font-size: clamp(1.75rem, 4vw, 2.55rem);
        font-weight: 760;
        letter-spacing: -0.04em;
        line-height: 1.04;
    }

    .pp-subtitle {
        color: var(--pp-muted);
        font-size: 0.98rem;
        line-height: 1.55;
        margin-top: 0.65rem;
        max-width: 48rem;
    }

    .pp-section-label {
        color: var(--pp-muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0.15rem 0 0.35rem;
    }

    div[data-testid="stMetric"] {
        min-height: 7rem;
        padding: 1rem 1.05rem;
        border: 1px solid var(--pp-border);
        border-radius: 0.9rem;
        background: var(--pp-surface);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--pp-muted);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--pp-border);
        border-radius: 1rem;
        background: color-mix(in srgb, var(--pp-surface) 78%, transparent);
    }

    div[data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--pp-border);
        border-radius: 0.9rem;
    }

    .stButton > button,
    .stDownloadButton > button {
        min-height: 2.7rem;
        border-radius: 0.7rem;
        font-weight: 680;
    }

    [data-baseweb="tab-list"] {
        gap: 0.45rem;
        border-bottom: 1px solid var(--pp-border);
    }

    [data-baseweb="tab"] {
        min-height: 3rem;
        padding-inline: 1rem;
        font-weight: 650;
    }

    [data-testid="stExpander"] {
        border-color: var(--pp-border);
        border-radius: 0.85rem;
        overflow: hidden;
    }

    hr {
        border-color: var(--pp-border) !important;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1rem;
        }

        .pp-hero {
            padding: 1.3rem;
        }
    }
</style>
"""


def inject_app_styles() -> None:
    st.markdown(_app_css(), unsafe_allow_html=True)


def _is_private_enabled() -> bool:
    return bool(os.getenv("STREAMLIT_PASSWORD"))


def _check_access() -> None:
    if not _is_private_enabled():
        return
    password = os.getenv("STREAMLIT_PASSWORD", "")
    if st.session_state.get("streamlit_auth") == password:
        return
    entered = st.text_input("Password", type="password", key="streamlit_password")
    if entered == password:
        st.session_state["streamlit_auth"] = entered
        st.rerun()
    else:
        st.warning("Enter the configured password to continue.")
        st.stop()


def normalize_symbol_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [x.strip().upper() for x in raw.split(",") if x and x.strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _plan_bundle(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    plan = payload.get("trade_plan") or {}
    primary = payload.get("primary_setup") or {}
    sim = payload.get("position_simulation") or {}
    if not isinstance(plan, dict):
        plan = {}
    if not isinstance(primary, dict):
        primary = {}
    if not isinstance(sim, dict):
        sim = {}
    return plan, primary, sim


def _entry_zone(plan: Dict[str, Any], primary: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    low = plan.get("entry_low")
    high = plan.get("entry_high")
    if low is None and isinstance(primary.get("entry_zone"), dict):
        low = primary["entry_zone"].get("low")
        high = primary["entry_zone"].get("high")
    if low is None:
        low = plan.get("entry") or plan.get("entry_price")
        high = high if high is not None else low
    return (
        _safe_float(low) if low is not None else None,
        _safe_float(high) if high is not None else None,
    )


def _take_profits(plan: Dict[str, Any], primary: Dict[str, Any]) -> List[float]:
    tps = plan.get("take_profits")
    if isinstance(tps, list) and tps:
        return [_safe_float(x) for x in tps[:4]]
    out: List[float] = []
    for key in ("tp1", "tp2", "tp3", "tp4"):
        val = primary.get(key)
        if val is not None:
            out.append(_safe_float(val))
    if not out and plan.get("take_profit") is not None:
        out.append(_safe_float(plan.get("take_profit")))
    return out


def _risk_rewards(plan: Dict[str, Any], primary: Dict[str, Any], n: int) -> List[Optional[float]]:
    rrs = plan.get("risk_reward")
    if isinstance(rrs, list) and rrs:
        vals = [_safe_float(x) for x in rrs[:n]]
        while len(vals) < n:
            vals.append(None)
        return vals
    out: List[Optional[float]] = []
    for key in ("rr_tp1", "rr_tp2", "rr_tp3", "rr_tp4")[:n]:
        val = primary.get(key)
        out.append(_safe_float(val) if val is not None else None)
    while len(out) < n:
        out.append(None)
    return out


def _display_leverage(payload: Dict[str, Any], plan: Dict[str, Any], primary: Dict[str, Any]) -> int:
    if payload.get("display_leverage") is not None:
        return int(payload["display_leverage"])
    raw = (
        plan.get("leverage_suggested")
        if plan.get("leverage_suggested") is not None
        else primary.get("leverage_suggested")
    )
    try:
        lev = int(round(float(raw or SCAN_LEVERAGE_CAP)))
    except (TypeError, ValueError):
        lev = SCAN_LEVERAGE_CAP
    return min(SCAN_LEVERAGE_CAP, max(1, lev))


def _model_leverage(payload: Dict[str, Any], plan: Dict[str, Any]) -> Optional[int]:
    if payload.get("model_leverage") is not None:
        try:
            return int(payload["model_leverage"])
        except (TypeError, ValueError):
            pass
    if plan.get("leverage_suggested") is not None:
        try:
            return int(round(float(plan["leverage_suggested"])))
        except (TypeError, ValueError):
            return None
    return None


def _scale_sim_to_base(
    plan: Dict[str, Any],
    sim: Dict[str, Any],
    base_usd: float = SIM_BASE_USD,
) -> Dict[str, Any]:
    """Rescale position simulation from model capital to a $100 example."""
    capital = _safe_float(
        plan.get("simulated_capital") or sim.get("simulated_capital"),
        1000.0,
    )
    scale = (base_usd / capital) if capital > 0 else 1.0
    profits = plan.get("potential_profits") or sim.get("potential_profits_usd") or []
    profit_pcts = plan.get("potential_profit_pcts") or sim.get("potential_profit_pct_of_capital") or []
    return {
        "base_usd": base_usd,
        "risk_amount": _safe_float(plan.get("risk_amount") or sim.get("risk_amount")) * scale,
        "risk_pct": _safe_float(plan.get("risk_pct") or sim.get("risk_pct"), 1.0),
        "notional": _safe_float(plan.get("position_size_notional") or sim.get("position_size_notional")) * scale,
        "margin": _safe_float(plan.get("margin_required") or sim.get("margin_required")) * scale,
        "units": _safe_float(plan.get("position_size_units") or sim.get("position_size_units")) * scale,
        "profits": [_safe_float(p) * scale for p in (profits or [])[:4]],
        "profit_pcts": [_safe_float(p) for p in (profit_pcts or [])[:4]],
    }


def build_report_markdown(payload: Dict[str, Any]) -> str:
    """Rich markdown report for download / export."""
    symbol = payload.get("symbol") or "—"
    direction = payload.get("direction") or payload.get("bias") or "flat"
    confidence = payload.get("confidence") or payload.get("signal", {}).get("confidence_pct") or 0
    setup = payload.get("setup_name") or payload.get("signal", {}).get("setup_name") or "—"
    score = payload.get("signal", {}).get("confluence_score") or payload.get("confluence_total") or "—"
    exchange = payload.get("exchange") or "—"
    plan, primary, sim = _plan_bundle(payload)
    entry_low, entry_high = _entry_zone(plan, primary)
    stop = plan.get("stop_loss") or plan.get("sl") or primary.get("stop_loss")
    tps = _take_profits(plan, primary)
    rrs = _risk_rewards(plan, primary, len(tps) or 4)
    lev = _display_leverage(payload, plan, primary)
    model_lev = _model_leverage(payload, plan)
    hold = plan.get("hold_detail") or primary.get("hold_detail") or payload.get("signal", {}).get("hold_detail") or "—"
    hold_label = plan.get("hold_label") or primary.get("hold_label") or ""
    invalidation = plan.get("invalidation") or primary.get("invalidation") or "—"
    quality = plan.get("quality") or primary.get("quality") or "—"
    price = None
    if isinstance(payload.get("meta"), dict):
        price = payload["meta"].get("price")
    if price is None and isinstance(payload.get("snapshot"), dict):
        price = payload["snapshot"].get("last")
    ref = _safe_float(price or entry_low or stop or 0)

    def fp(v: Any) -> str:
        if v is None:
            return "—"
        return format_price(_safe_float(v), ref if ref else None)

    llm_conf = payload.get("llm_confidence")
    if llm_conf is None and isinstance(payload.get("signal"), dict):
        llm_conf = payload["signal"].get("llm_confidence")
    llm_reason = (
        payload.get("llm_confidence_reason")
        or (payload.get("llm_narrative") or {}).get("confidence_reason")
        or (payload.get("signal") or {}).get("llm_confidence_reason")
        or ""
    )
    tech_conf = payload.get("technical_confidence")
    if tech_conf is None and isinstance(payload.get("signal"), dict):
        tech_conf = payload["signal"].get("technical_confidence")
    rank_score = payload.get("rank_score")
    if rank_score is None and isinstance(payload.get("signal"), dict):
        rank_score = payload["signal"].get("rank_score")

    lines = [
        f"# {symbol}",
        "",
        f"- **Direction:** {str(direction).upper()}",
        f"- **Confidence:** {confidence}",
        f"- **LLM Confidence:** {llm_conf if llm_conf is not None else '—'}%",
        f"- **LLM reason:** {llm_reason or '—'}",
        f"- **Technical confidence:** {tech_conf if tech_conf is not None else '—'}",
        f"- **Rank score:** {rank_score if rank_score is not None else '—'}",
        f"- **Setup:** {setup}",
        f"- **Confluence:** {score}",
        f"- **Exchange used:** {exchange}",
    ]
    if payload.get("fallback_used"):
        lines.append(
            f"- **Fallback:** requested {payload.get('exchange_requested')} → used {exchange}"
        )
    execution = payload.get("execution") or (payload.get("meta") or {}).get("execution") or {}
    if execution:
        lines += [
            "",
            "## Entry quality",
            "",
            f"- **Status:** {str(execution.get('status') or 'blocked').replace('_', ' ').title()}",
            f"- **Execution score:** {_safe_float(execution.get('score')):.1f}/100",
            f"- **Immediate-SL risk:** {_safe_float(execution.get('immediate_sl_risk')):.1f}%",
            f"- **Distance to entry:** {_safe_float(execution.get('chase_distance_atr')):.2f} ATR",
            f"- **Order-flow approximation:** {_safe_float(execution.get('order_flow_score')):+.2f}",
            f"- **Rule:** {execution.get('entry_reason') or '—'}",
        ]
        for note in ((execution.get("candle") or {}).get("notes") or [])[:5]:
            lines.append(f"- {note}")
    lines += [
        "",
        "## Trade setup",
        "",
    ]
    if entry_low is not None and entry_high is not None:
        lines.append(f"- **Entry zone:** {fp(entry_low)} – {fp(entry_high)}")
    else:
        lines.append(f"- **Entry:** {fp(entry_low or plan.get('entry'))}")
    alt_low = plan.get("alternative_entry_low")
    alt_high = plan.get("alternative_entry_high")
    if alt_low is not None and alt_high is not None:
        note = plan.get("alternative_entry_note") or ""
        lines.append(
            f"- **Alternative entry:** {fp(alt_low)} – {fp(alt_high)}"
            + (f" ({note})" if note else "")
        )
    lines.append(f"- **Stop loss:** {fp(stop)}")
    for i, tp in enumerate(tps, 1):
        rr = rrs[i - 1] if i - 1 < len(rrs) else None
        rr_s = f"{rr:.2f}" if rr is not None else "—"
        lines.append(f"- **TP{i}:** {fp(tp)}  ·  R:R {rr_s}")
    lines += [
        f"- **Suggested leverage (display priority):** {lev}x (cap {SCAN_LEVERAGE_CAP}x)",
    ]
    if model_lev is not None and model_lev != lev:
        lines.append(f"- **Model leverage:** {model_lev}x")
    lines += [
        f"- **Hold:** {hold_label + ' — ' if hold_label else ''}{hold}",
        f"- **Invalidation:** {invalidation}",
        f"- **Quality:** {str(quality).upper()}",
        "",
        f"## Simulation example (${SIM_BASE_USD:.0f} base)",
        "",
    ]
    scaled = _scale_sim_to_base(plan, sim, SIM_BASE_USD)
    lines.append(
        f"- Risk ~${scaled['risk_amount']:.2f} ({scaled['risk_pct']:.2f}% of ${SIM_BASE_USD:.0f}) "
        f"at {lev}x display leverage priority"
    )
    lines.append(
        f"- Notional ~${scaled['notional']:.2f} · margin ~${scaled['margin']:.2f}"
    )
    for i, profit in enumerate(scaled["profits"], 1):
        pct = scaled["profit_pcts"][i - 1] if i - 1 < len(scaled["profit_pcts"]) else None
        pct_s = f" ({pct:+.2f}% of base)" if pct is not None else ""
        lines.append(f"- At TP{i} → sim P/L ~${profit:,.2f}{pct_s}")
    if not scaled["profits"]:
        lines.append("- Profit targets unavailable for this plan.")
    lines += ["", "_Simulation only — not a live order. NOT FINANCIAL ADVICE._", ""]
    reasons = payload.get("key_reasons") or []
    if reasons:
        lines += ["## Key reasons", ""]
        for r in reasons[:5]:
            lines.append(f"- {r}")
        lines.append("")
    risks = payload.get("key_risks") or []
    if risks:
        lines += ["## Key risks", ""]
        for r in risks[:5]:
            lines.append(f"- {r}")
        lines.append("")
    return "\n".join(lines)


def call_backend(endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BACKEND_URL.rstrip('/')}{endpoint}"
    try:
        if payload is None:
            response = requests.get(url, timeout=60)
        else:
            response = requests.post(url, data=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def send_manual_telegram_test() -> Dict[str, Any]:
    """Prefer the secured API test endpoint, with a local Streamlit fallback."""
    test_key = (os.getenv("TELEGRAM_TEST_KEY") or "").strip()
    if test_key:
        url = f"{BACKEND_URL.rstrip('/')}/telegram/test"
        try:
            response = requests.post(
                url,
                headers={"X-Telegram-Test-Key": test_key},
                timeout=45,
            )
            try:
                result = response.json()
            except (TypeError, ValueError):
                result = {
                    "ok": False,
                    "error": "invalid_backend_response",
                    "description": f"Backend returned HTTP {response.status_code}",
                }
            result["test_path"] = "backend"
            return result
        except requests.RequestException:
            # Local-only Streamlit deployments may not run FastAPI.
            pass

    from src.notify.telegram import send_test_telegram_alert

    result = send_test_telegram_alert(source="Streamlit test button")
    result["test_path"] = "streamlit_process"
    return result


def run_manual_telegram_scan_test() -> Dict[str, Any]:
    """Run the real scheduled scan now and send chart/no-setup Telegram output."""
    test_key = (os.getenv("TELEGRAM_TEST_KEY") or "").strip()
    if test_key:
        url = f"{BACKEND_URL.rstrip('/')}/telegram/test-scan"
        try:
            response = requests.post(
                url,
                headers={"X-Telegram-Test-Key": test_key},
                timeout=900,
            )
            try:
                result = response.json()
            except (TypeError, ValueError):
                result = {
                    "ok": False,
                    "error": "invalid_backend_response",
                    "description": f"Backend returned HTTP {response.status_code}",
                }
            result["test_path"] = "backend"
            return result
        except requests.RequestException:
            # Local-only Streamlit deployments can run the same workflow directly.
            pass

    from src.scheduler.scan_job import run_scheduled_scan_once

    result = run_scheduled_scan_once(
        load_config(),
        slot_label="Manual test scan",
        send=True,
    )
    return {
        "ok": bool(result.get("ok") and result.get("telegram_sent")),
        "scan_ok": bool(result.get("ok")),
        "telegram_sent": bool(result.get("telegram_sent")),
        "telegram_ready": bool(result.get("telegram_ready")),
        "delivery_status": result.get("telegram_delivery_status"),
        "delivery": result.get("telegram_delivery"),
        "scanned": result.get("scanned"),
        "ranked_count": result.get("ranked_count"),
        "alert_count": result.get("alert_count"),
        "started_at": result.get("started_at"),
        "completed_at": result.get("completed_at"),
        "slot_label": result.get("slot_label"),
        "test_path": "streamlit_process",
    }


def analyze_symbol(symbol: str, timeframe: str, exchange: str, no_news: bool) -> Dict[str, Any]:
    """Run a fresh single-symbol analysis (no Streamlit cache)."""
    from src.api.service import analyze_market_data

    req = AnalyzeRequest(
        symbol=symbol,
        timeframe=timeframe,
        exchange=exchange,
        no_news=no_news,
        simulated_capital=SIM_BASE_USD,
        risk_pct=1.0,
        use_llm=True,
    )
    cfg = load_config()
    return analyze_market_data(symbol, request=req, config=cfg)


def scan_symbols(symbols: List[str], timeframe: str, exchange: str, no_news: bool) -> Dict[str, Any]:
    """
    Always run a fresh multi-symbol scan (no Streamlit cache).

    Re-fetches market data + news on every click so results are not stale.
    """
    from src.api.service import scan_symbols as scan_backend

    req = AnalyzeRequest(
        timeframe=timeframe,
        exchange=exchange,
        no_news=no_news,
        simulated_capital=SIM_BASE_USD,
        risk_pct=1.0,
        use_llm=True,
    )
    cfg = load_config()
    return scan_backend(symbols, request=req, config=cfg)


def format_ticker_price(symbol: Any, price: Any) -> str:
    """Human label like ``BTC $112,450`` for scan rows / expanders."""
    raw = str(symbol or "—")
    base = raw.split("/")[0].split(":")[0].upper() if raw != "—" else "—"
    if price is None or price == "":
        return f"{base} —"
    try:
        p = float(price)
    except (TypeError, ValueError):
        return f"{base} —"
    if p >= 1000:
        return f"{base} ${p:,.0f}"
    if p >= 1:
        return f"{base} ${p:,.4f}".rstrip("0").rstrip(".")
    if p >= 0.01:
        return f"{base} ${p:.6f}".rstrip("0").rstrip(".")
    return f"{base} ${p:.8f}".rstrip("0").rstrip(".")


def _resolved_chart_height(requested: int, expanded: bool) -> int:
    """Keep charts readable by default and substantially larger on demand."""
    base_height = max(600, int(requested or 600))
    return max(840, base_height + 240) if expanded else base_height


def render_market_chart(
    chart: Dict[str, Any],
    *,
    title: str,
    key: str,
    height: int = 600,
) -> None:
    """Render an expandable interactive execution chart."""
    candles = (chart or {}).get("candles") or []
    if not candles:
        st.caption("Candle plot unavailable.")
        return
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:  # noqa: BLE001
        st.caption("Install plotly to display candle charts.")
        return

    frame = pd.DataFrame(candles)
    frame["t"] = pd.to_datetime(frame["t"], utc=True)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.76, 0.24],
    )
    fig.add_trace(
        go.Candlestick(
            x=frame["t"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Closed candles",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        ),
        row=1,
        col=1,
    )
    for column, label, color in (
        ("ema_fast", "EMA 9", "#f59e0b"),
        ("ema_mid", "EMA 21", "#38bdf8"),
        ("vwap", "VWAP", "#a78bfa"),
    ):
        if column in frame and frame[column].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=frame["t"],
                    y=frame[column],
                    mode="lines",
                    name=label,
                    line={"width": 1.6, "color": color},
                    hovertemplate=f"{label}: %{{y:,.6g}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    volume_colors = [
        "#16a34a" if close >= open_ else "#dc2626"
        for open_, close in zip(frame["open"], frame["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=frame["t"],
            y=frame["volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.55,
            hovertemplate="Volume: %{y:,.4~s}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    annotated_level_types = set()
    annotation_positions = ("top left", "bottom left", "top right", "bottom right")
    for level_index, level in enumerate(((chart or {}).get("levels") or [])[:12]):
        value = level.get("mid")
        if value is None:
            continue
        side = level.get("side")
        kind = str(level.get("kind") or "level").replace("_", " ").title()
        color = "#22c55e" if side == "bullish" else ("#ef4444" if side == "bearish" else "#94a3b8")
        level_signature = (kind, side)
        show_label = level_signature not in annotated_level_types
        if show_label:
            annotated_level_types.add(level_signature)
        level_line = {
            "y": float(value),
            "line_width": 1.05,
            "line_dash": "dot",
            "line_color": color,
            "row": 1,
            "col": 1,
        }
        if show_label:
            fig.add_hline(
                **level_line,
                annotation_text=kind,
                annotation_position=annotation_positions[
                    level_index % len(annotation_positions)
                ],
                annotation_font_color=color,
                annotation_font_size=10,
            )
        else:
            fig.add_hline(**level_line)

    trade = (chart or {}).get("trade") or {}
    if trade.get("entry_low") is not None and trade.get("entry_high") is not None:
        fig.add_hrect(
            y0=float(trade["entry_low"]),
            y1=float(trade["entry_high"]),
            fillcolor="#eab308",
            opacity=0.2,
            line_color="#facc15",
            line_width=1.4,
            annotation_text=f"Entry · {trade.get('entry_status', '')}",
            annotation_font_color="#111827",
            annotation_font_size=11,
            annotation_bgcolor="rgba(250, 204, 21, 0.92)",
            annotation_borderpad=4,
            row=1,
            col=1,
        )
    if trade.get("stop_loss") is not None:
        fig.add_hline(
            y=float(trade["stop_loss"]),
            line_color="#f43f5e",
            line_width=2.2,
            annotation_text="SL",
            annotation_font_color="#ffffff",
            annotation_font_size=11,
            annotation_bgcolor="rgba(244, 63, 94, 0.92)",
            annotation_borderpad=4,
            row=1,
            col=1,
        )
    for index, target in enumerate((trade.get("take_profits") or [])[:4], 1):
        fig.add_hline(
            y=float(target),
            line_color="#10b981",
            line_width=1.55,
            line_dash="dash",
            annotation_text=f"TP{index}",
            annotation_font_color="#ffffff",
            annotation_font_size=11,
            annotation_bgcolor="rgba(16, 185, 129, 0.9)",
            annotation_borderpad=4,
            row=1,
            col=1,
        )
    with st.container(border=True):
        control_text, control_toggle = st.columns([4, 1])
        with control_text:
            st.caption("Drag to pan · scroll to zoom · hover for values · double-click to reset")
        with control_toggle:
            expanded = st.toggle(
                "Large view",
                value=False,
                key=f"{key}_large_view",
                help="Increase chart height for detailed structure and candle review.",
            )
        chart_height = _resolved_chart_height(height, expanded)
        fig.update_layout(
            title={
                "text": title,
                "x": 0.01,
                "xanchor": "left",
                "font": {"size": 17},
            },
            height=chart_height,
            xaxis_rangeslider_visible=False,
            margin={"l": 14, "r": 28, "t": 58, "b": 76},
            legend={
                "orientation": "h",
                "y": -0.12,
                "x": 0,
                "xanchor": "left",
                "yanchor": "top",
            },
            hovermode="x unified",
            dragmode="pan",
            hoverlabel={"namelength": -1},
            uirevision=f"{key}:{'large' if expanded else 'standard'}",
        )
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="dot",
            showline=True,
            fixedrange=False,
        )
        fig.update_yaxes(
            title_text="Price",
            showspikes=True,
            spikemode="across",
            spikedash="dot",
            fixedrange=False,
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", fixedrange=False, row=2, col=1)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            theme="streamlit",
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{key}_chart",
                    "scale": 2,
                },
            },
        )
        patterns = (chart or {}).get("patterns") or []
        if patterns:
            st.caption(
                "Candle patterns · "
                + " · ".join(
                    f"{p.get('name')} ({p.get('bias')}, "
                    f"{float(p.get('confidence') or 0):.0f}%)"
                    for p in patterns[:4]
                )
            )


def render_scan_results(payload: Dict[str, Any], *, key_prefix: str = "scan") -> None:
    """Render a concise leaderboard followed by the strongest setup cards."""
    if payload.get("ok") is False and (payload.get("error") or payload.get("message")):
        st.error(payload.get("error") or payload.get("message"))
        return
    rows = payload.get("ranked_results", [])
    if not rows:
        st.info(
            "No high-quality setup is available right now. Staying flat is a valid trade decision."
        )
        return

    display_rows: List[Dict[str, Any]] = []
    for row in rows:
        row_payload = row.get("payload") or {}
        plan = row_payload.get("trade_plan") or {}
        entry_zone = plan.get("entry_zone") or {}
        entry_low = row.get("entry_low")
        if entry_low is None:
            entry_low = plan.get("entry_low", entry_zone.get("low"))
        entry_high = row.get("entry_high")
        if entry_high is None:
            entry_high = plan.get("entry_high", entry_zone.get("high"))
        stop = row.get("stop_loss")
        if stop is None:
            stop = plan.get("stop_loss")
        target = plan.get("tp1")
        if target is None:
            targets = plan.get("take_profits") or []
            target = targets[0] if targets else None
        rr = plan.get("rr_tp1")
        if rr is None:
            risk_rewards = plan.get("risk_reward") or []
            rr = risk_rewards[0] if risk_rewards else None
        confidence = row.get("confidence")
        if confidence is None:
            confidence = row.get("llm_confidence")
        entry_state = str(
            row.get("entry_status")
            or (row_payload.get("execution") or {}).get("status")
            or "wait"
        ).replace("_", " ").title()
        display_rows.append(
            {
                "Market": format_ticker_price(row.get("symbol"), row.get("price")),
                "Side": str(row.get("direction") or "flat").upper(),
                "Confidence": round(_safe_float(confidence), 1),
                "Entry": (
                    f"{format_price(_safe_float(entry_low))} – "
                    f"{format_price(_safe_float(entry_high))}"
                    if entry_low is not None and entry_high is not None
                    else "—"
                ),
                "Stop": format_price(_safe_float(stop)) if stop is not None else "—",
                "TP1": format_price(_safe_float(target)) if target is not None else "—",
                "R:R": round(_safe_float(rr), 2) if rr is not None else "—",
                "Status": entry_state,
                "Lev.": f"{row.get('leverage', '—')}x",
            }
        )

    df = pd.DataFrame(display_rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        key=f"{key_prefix}_results_table",
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0,
                max_value=100,
                format="%.0f%%",
            ),
            "Side": st.column_config.TextColumn("Side", width="small"),
            "Status": st.column_config.TextColumn("Entry status", width="medium"),
        },
    )
    st.caption(
        f"{len(rows)} qualified setup{'s' if len(rows) != 1 else ''} · "
        f"{payload.get('flat_count', 0)} neutral market"
        f"{'s' if payload.get('flat_count', 0) != 1 else ''} filtered out"
    )

    for i, row in enumerate(rows[:5]):
        ticker = format_ticker_price(row.get("symbol"), row.get("price"))
        confidence = row.get("confidence")
        if confidence is None:
            confidence = row.get("llm_confidence")
        title = (
            f"#{i + 1} {ticker} · {str(row.get('direction', 'flat')).upper()} · "
            f"{_safe_float(confidence):.0f}% confidence"
        )
        with st.expander(title, expanded=(i == 0)):
            row_payload = row.get("payload") or {}
            execution = row_payload.get("execution") or {}
            status = str(
                execution.get("status") or row.get("entry_status") or "wait"
            ).replace("_", " ").upper()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Side", str(row.get("direction") or "flat").upper())
            c2.metric("Confidence", f"{_safe_float(confidence):.0f}%")
            c3.metric("Entry status", status)
            c4.metric("Leverage", f"{row.get('leverage', '—')}x")
            if row.get("llm_confidence_reason"):
                st.info(row["llm_confidence_reason"])
            elif row.get("reason"):
                st.info(row["reason"])
            if execution.get("entry_reason"):
                if status == "READY":
                    st.success(execution["entry_reason"])
                else:
                    st.warning(execution["entry_reason"])
            render_market_chart(
                row_payload.get("chart") or {},
                title=f"{ticker} · execution chart",
                key=f"{key_prefix}_chart_{i}",
                height=600,
            )
            render_trade_setup_card(row_payload)

    if rows:
        export_payload = {
            "symbol": rows[0].get("symbol"),
            "direction": rows[0].get("direction"),
            "confidence": rows[0].get("confidence"),
            "technical_confidence": rows[0].get("technical_confidence"),
            "llm_confidence": rows[0].get("llm_confidence"),
            "llm_confidence_reason": rows[0].get("llm_confidence_reason"),
            "rank_score": rows[0].get("rank_score"),
            "setup_name": rows[0].get("setup_name"),
            "confluence_total": rows[0].get("confluence_score"),
            "exchange": rows[0].get("exchange"),
            "fallback_used": rows[0].get("fallback_used"),
            "exchange_requested": rows[0].get("exchange_requested"),
            "display_leverage": rows[0].get("leverage"),
            "model_leverage": rows[0].get("model_leverage"),
            "trade_plan": (rows[0].get("payload") or {}).get("trade_plan") or {},
            "primary_setup": (rows[0].get("payload") or {}).get("primary_setup") or {},
            "position_simulation": (rows[0].get("payload") or {}).get("position_simulation") or {},
            "key_reasons": (rows[0].get("payload") or {}).get("key_reasons") or [],
        }
        # Flatten entry zone for markdown builder if nested
        tp = export_payload["trade_plan"]
        if isinstance(tp, dict) and "entry_zone" in tp and "entry_low" not in tp:
            ez = tp.get("entry_zone") or {}
            tp = {
                **tp,
                "entry_low": ez.get("low"),
                "entry_high": ez.get("high"),
                "take_profits": [
                    tp.get(k) for k in ("tp1", "tp2", "tp3", "tp4") if tp.get(k) is not None
                ],
                "risk_reward": [
                    tp.get(k) for k in ("rr_tp1", "rr_tp2", "rr_tp3", "rr_tp4") if tp.get(k) is not None
                ],
            }
            export_payload["trade_plan"] = tp
        st.download_button(
            label="Download top scan report",
            data=build_report_markdown(export_payload),
            file_name="scan_report.md",
            mime="text/markdown",
            key=f"{key_prefix}_download_scan_report",
        )


def render_trade_setup_card(payload: Dict[str, Any]) -> None:
    plan, primary, _ = _plan_bundle(payload)
    direction = (plan.get("direction") or payload.get("direction") or "flat").lower()
    if direction == "flat":
        st.info("No directional trade plan. Wait for clearer structure.")
        return

    entry_low, entry_high = _entry_zone(plan, primary)
    stop = plan.get("stop_loss") or primary.get("stop_loss")
    tps = _take_profits(plan, primary)
    rrs = _risk_rewards(plan, primary, max(len(tps), 4))
    lev = _display_leverage(payload, plan, primary)
    hold = plan.get("hold_detail") or primary.get("hold_detail") or "—"
    hold_label = plan.get("hold_label") or primary.get("hold_label") or ""
    invalidation = plan.get("invalidation") or primary.get("invalidation") or "—"
    quality = (plan.get("quality") or primary.get("quality") or "—").upper()
    price = None
    if isinstance(payload.get("meta"), dict):
        price = payload["meta"].get("price")
    if price is None and isinstance(payload.get("snapshot"), dict):
        price = payload["snapshot"].get("last")
    ref = _safe_float(price or entry_low or stop or 0)

    def fp(v: Any) -> str:
        if v is None:
            return "—"
        return format_price(_safe_float(v), ref if ref else None)

    with st.container(border=True):
        st.markdown('<div class="pp-section-label">Trade plan</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Side", direction.upper())
        m2.metric("Entry zone", f"{fp(entry_low)} – {fp(entry_high)}")
        m3.metric("Stop", fp(stop))
        m4.metric("Leverage", f"{lev}x")

        st.markdown("**Profit targets**")
        if tps:
            tp_cols = st.columns(len(tps))
            for i, tp in enumerate(tps):
                rr = rrs[i] if i < len(rrs) else None
                rr_s = f"R:R {rr:.2f}" if rr is not None else "R:R —"
                with tp_cols[i]:
                    st.metric(f"TP{i + 1}", fp(tp), delta=rr_s)
        else:
            st.caption("Targets are unavailable for this setup.")

        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.markdown(f"**Hold**  \n{hold_label or 'Day trade'} · {hold}")
        with detail_right:
            st.markdown(f"**Invalidation**  \n{invalidation}")

        alt_low = plan.get("alternative_entry_low")
        alt_high = plan.get("alternative_entry_high")
        if alt_low is not None and alt_high is not None:
            note = plan.get("alternative_entry_note") or ""
            st.caption(
                f"Alternative entry: {fp(alt_low)} – {fp(alt_high)}"
                + (f" · {note}" if note else "")
            )
        st.caption(
            f"{quality.title()} setup · prop-account leverage capped at {SCAN_LEVERAGE_CAP}x"
        )


def render_simulation_card(payload: Dict[str, Any]) -> None:
    plan, primary, sim = _plan_bundle(payload)
    direction = (plan.get("direction") or payload.get("direction") or "flat").lower()
    if direction == "flat":
        st.caption("No simulation — flat / neutral bias.")
        return

    lev = _display_leverage(payload, plan, primary)
    scaled = _scale_sim_to_base(plan, sim, SIM_BASE_USD)

    with st.expander(f"Position sizing example · ${SIM_BASE_USD:.0f} account"):
        st.caption("Illustration only. Size every trade to your own daily loss limits.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk at stop", f"${scaled['risk_amount']:.2f}")
        c2.metric("Notional", f"${scaled['notional']:.2f}")
        c3.metric(f"Margin @ {lev}x", f"${scaled['margin']:.2f}")
        if scaled["profits"]:
            profit_rows = []
            for i, profit in enumerate(scaled["profits"], 1):
                pct = (
                    scaled["profit_pcts"][i - 1]
                    if i - 1 < len(scaled["profit_pcts"])
                    else None
                )
                profit_rows.append(
                    {
                        "Target": f"TP{i}",
                        "Estimated P/L": f"${profit:,.2f}",
                        "Account return": f"{pct:+.2f}%" if pct is not None else "—",
                    }
                )
            st.dataframe(
                pd.DataFrame(profit_rows),
                use_container_width=True,
                hide_index=True,
            )


def render_single_analysis(
    payload: Dict[str, Any],
    *,
    symbol: str,
    exchange: str,
) -> None:
    """Render one completed analysis without exposing model/backend internals."""
    if not payload.get("ok"):
        st.error(payload.get("error") or payload.get("message") or "Analysis failed")
        return

    direction = str(payload.get("bias") or payload.get("direction") or "flat").upper()
    confidence = payload.get("confidence")
    if confidence is None:
        confidence = payload.get("llm_confidence")
    if confidence is None:
        confidence = (payload.get("signal") or {}).get("llm_confidence")
    execution = payload.get("execution") or (payload.get("meta") or {}).get("execution") or {}
    status = str(execution.get("status") or "wait").replace("_", " ").upper()
    ticker = format_ticker_price(
        payload.get("symbol") or symbol,
        (payload.get("meta") or {}).get("price"),
    )
    st.subheader(ticker)
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Side", direction)
    h2.metric("Confidence", f"{_safe_float(confidence):.0f}%")
    h3.metric("Setup", payload.get("setup_name") or "—")
    h4.metric("Entry status", status)

    reason = (
        payload.get("llm_confidence_reason")
        or (payload.get("llm_narrative") or {}).get("confidence_reason")
        or payload.get("reason")
        or ""
    )
    plan = payload.get("trade_plan") or {}
    if plan.get("prop_safe") is False or payload.get("prop_safe") is False:
        st.warning("This setup does not pass prop-account guardrails. Do not treat it as actionable.")
    elif reason:
        st.info(reason)

    if execution.get("entry_reason"):
        if status == "READY":
            st.success(execution["entry_reason"])
        else:
            st.warning(execution["entry_reason"])

    render_trade_setup_card(payload)

    render_market_chart(
        payload.get("chart") or {},
        title=f"{ticker} · structure and execution",
        key=f"single_chart_{symbol.replace('/', '_').replace(':', '_')}",
    )
    render_simulation_card(payload)

    reasons = payload.get("key_reasons") or []
    risks = payload.get("key_risks") or []
    if reasons or risks:
        r1, r2 = st.columns(2)
        with r1:
            with st.container(border=True):
                st.markdown("**Why this setup**")
                if reasons:
                    for item in reasons[:5]:
                        st.write(f"- {item}")
                else:
                    st.caption("No additional confirmations.")
        with r2:
            with st.container(border=True):
                st.markdown("**What could invalidate it**")
                if risks:
                    for item in risks[:6]:
                        st.write(f"- {item}")
                else:
                    st.caption("No additional risk notes.")

    has_more = any(
        (
            payload.get("key_levels"),
            payload.get("trader_commentary"),
            payload.get("warnings"),
        )
    )
    if has_more:
        with st.expander("More market context"):
            if payload.get("trader_commentary"):
                st.write(payload["trader_commentary"])
            if payload.get("key_levels"):
                st.markdown("**Key levels**")
                for level in payload.get("key_levels")[:6]:
                    if isinstance(level, dict):
                        label = level.get("label") or level.get("kind") or "Level"
                        value = level.get("price", level.get("value", level.get("mid")))
                        st.write(
                            f"- {str(label).replace('_', ' ').title()}: "
                            + (
                                format_price(_safe_float(value))
                                if value is not None
                                else str(level.get("note") or "—")
                            )
                        )
                    else:
                        st.write(f"- {level}")
            if payload.get("warnings"):
                st.markdown("**Warnings**")
                for w in payload["warnings"]:
                    st.write(f"- {w}")

    safe_sym = symbol.replace("/", "_").replace(":", "_") or "symbol"
    st.download_button(
        label="Download trade report",
        data=build_report_markdown(payload),
        file_name=f"{safe_sym}_report.md",
        mime="text/markdown",
        key=f"download_single_report_{safe_sym}",
    )


def render_backtest_panel(symbol: str, timeframe: str, exchange: str) -> None:
    st.markdown("### Backtest a setup")
    st.caption("Validate the current market and timeframe with prop-safe sizing.")
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            bars = st.slider(
                "History length",
                min_value=150,
                max_value=1000,
                value=500,
                step=50,
                key="bt_bars",
                help="More candles improve context but take longer to process.",
            )
        with right:
            st.markdown('<div class="pp-section-label">Current test</div>', unsafe_allow_html=True)
            st.write(f"**{symbol.upper()} · {timeframe}**")
        if st.button(
            "Run backtest",
            key="btn_run_backtest",
            type="primary",
            use_container_width=True,
        ):
            from src.api.service import run_symbol_backtest

            with st.spinner(f"Backtesting {symbol.upper()} on {timeframe}…"):
                result = run_symbol_backtest(
                    symbol,
                    timeframe=timeframe,
                    bars=int(bars),
                    exchange=exchange,
                    config=load_config(),
                )
            st.session_state["last_backtest"] = result
            st.session_state["last_backtest_label"] = f"{symbol.upper()} · {timeframe}"

    result = st.session_state.get("last_backtest")
    if not result:
        st.info("Run a backtest to see performance, drawdown, and the equity curve.")
        return
    if not result.get("ok", True) and result.get("error"):
        st.error(result.get("error"))
        return

    result_label = st.session_state.get("last_backtest_label")
    if result_label:
        st.markdown(f"#### Results · {result_label}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Win rate", f"{_safe_float(result.get('win_rate')):.1f}%")
    m2.metric("Profit factor", f"{_safe_float(result.get('profit_factor')):.2f}")
    m3.metric("Max drawdown", f"{_safe_float(result.get('max_drawdown_pct')):.2f}%")
    m4.metric("Net P&L", f"${_safe_float(result.get('net_pnl')):+,.2f}")
    st.caption(
        f"{result.get('n_trades', 0)} trades · "
        f"{_safe_float(result.get('stop_out_rate')):.1f}% stop-out rate · "
        f"${_safe_float(result.get('final_equity')):,.2f} final equity"
    )
    curve = result.get("equity_curve") or []
    if curve:
        eq_df = pd.DataFrame(curve)
        if "t" in eq_df.columns and "equity" in eq_df.columns:
            eq_df = eq_df.set_index("t")
            st.markdown("**Equity curve**")
            st.line_chart(eq_df["equity"], height=260)
    trades = result.get("trades") or []
    if trades:
        with st.expander(f"Trade history · {len(trades)}"):
            st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)


def main() -> None:
    _check_access()
    inject_app_styles()
    try:
        cfg = load_config()
        prop_on = bool(getattr(cfg.risk, "prop_mode", True))
    except Exception:  # noqa: BLE001
        cfg = None
        prop_on = True

    st.markdown(
        """
<div class="pp-hero">
    <div class="pp-eyebrow">Crypto perps · intraday intelligence</div>
    <div class="pp-title">Perpetual Pro</div>
    <div class="pp-subtitle">
        Find selective, structure-led setups with precise entries, realistic targets,
        and prop-account guardrails.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Market setup")
        if prop_on:
            st.success("Prop guardrails active")
            st.caption("Maximum 5x leverage · 0.5–1% risk")
        else:
            st.warning("Prop guardrails are off")
        symbol = st.text_input(
            "Symbol",
            value="BTC",
            placeholder="BTC or BTC/USDT:USDT",
            key="input_symbol",
        )
        timeframe = st.selectbox(
            "Timeframe",
            ["15m", "1h", "4h"],
            index=0,
            key="input_timeframe",
        )
        try:
            default_ex = (cfg.exchange.default if cfg else None) or "bybit"
        except Exception:  # noqa: BLE001
            default_ex = "bybit"
        ex_index = EXCHANGE_OPTIONS.index(default_ex) if default_ex in EXCHANGE_OPTIONS else 0
        exchange = st.selectbox(
            "Preferred exchange",
            EXCHANGE_OPTIONS,
            index=ex_index,
            key="input_exchange",
        )
        include_news = st.toggle(
            "Use news filter",
            value=True,
            key="input_no_news_v2",
            help="Include recent market headlines when validating a setup.",
        )
        no_news = not include_news

    tab_scan, tab_bt = st.tabs(["Market scanner", "Backtest"])

    with tab_scan:
        single_col, scan_col = st.columns([0.85, 1.15], gap="large")
        with single_col:
            with st.container(border=True):
                st.markdown("### Analyze one market")
                st.caption(f"{symbol.upper()} · {timeframe} · {exchange}")
                run_single = st.button(
                    "Analyze market",
                    type="primary",
                    key="btn_run_single",
                    use_container_width=True,
                )
        with scan_col:
            with st.container(border=True):
                st.markdown("### Scan watchlist")
                watchlist = st.text_area(
                    "Markets",
                    value=DEFAULT_WATCHLIST,
                    height=86,
                    key="input_watchlist",
                    label_visibility="collapsed",
                    placeholder="BTC, ETH, SOL",
                )
                run_scan = st.button(
                    "Find qualified setups",
                    key="btn_scan_watchlist",
                    use_container_width=True,
                )
        if run_single:
            clean_symbol = symbol.strip().upper()
            if not clean_symbol:
                st.warning("Enter a market symbol first.")
            else:
                with st.spinner(
                    f"Analyzing {clean_symbol} across 15m, 1h, and 4h structure…"
                ):
                    single_payload = analyze_symbol(
                        clean_symbol,
                        timeframe,
                        exchange,
                        no_news,
                    )
                st.session_state["last_single"] = single_payload
                st.session_state["last_single_symbol"] = clean_symbol
                st.session_state["last_single_at"] = datetime.now(timezone.utc).strftime(
                    "%H:%M UTC"
                )

        if run_scan:
            symbols = normalize_symbol_list(watchlist)
            if not symbols:
                st.warning("Add at least one symbol to scan.")
            else:
                with st.spinner(f"Scanning {len(symbols)} markets for qualified setups…"):
                    payload = scan_symbols(symbols, timeframe, exchange, no_news)
                st.session_state["last_scan"] = payload
                st.session_state["last_scan_at"] = datetime.now(timezone.utc).strftime(
                    "%H:%M UTC"
                )

        if st.session_state.get("last_single"):
            st.divider()
            single_label = st.session_state.get("last_single_symbol") or symbol.upper()
            single_at = st.session_state.get("last_single_at")
            st.markdown(f"## Individual analysis · {single_label}")
            if single_at:
                st.caption(f"Updated {single_at}")
            render_single_analysis(
                st.session_state["last_single"],
                symbol=single_label,
                exchange=exchange,
            )

        st.divider()
        st.markdown("## Ranked signals")
        if st.session_state.get("last_scan"):
            scanned_at = st.session_state.get("last_scan_at")
            if scanned_at:
                st.caption(f"Updated {scanned_at}")
            render_scan_results(st.session_state["last_scan"], key_prefix="scan")
        else:
            st.info("Scan your watchlist to rank only the setups that pass current filters.")

    with tab_bt:
        render_backtest_panel(symbol, timeframe, exchange)


if __name__ == "__main__":
    main()
