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
    "BTC,ETH,SOL,BNB,AAVE,ARB,OP,NEAR,INJ,SEI,BSV,TIA,ADA,SUI,APT,AVAX,TRXV,VIRTUAL,KAITO,UNI,JTO"
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


def render_scan_results(payload: Dict[str, Any], *, key_prefix: str = "scan") -> None:
    """Render ranked scan table + detail cards. ``key_prefix`` keeps widget IDs unique."""
    rows = payload.get("ranked_results", [])
    if not rows:
        st.info("No ranked setups returned yet. Try a broader watchlist or a different exchange.")
        return

    # Enrich rows with ticker + market price for display
    display_rows: List[Dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["ticker"] = format_ticker_price(row.get("symbol"), row.get("price"))
        display_rows.append(enriched)

    cols = [
        c
        for c in [
            "ticker",
            "direction",
            "price",
            "llm_confidence",
            "rank_score",
            "confidence",
            "technical_confidence",
            "confluence_score",
            "setup_name",
            "leverage",
            "model_leverage",
            "exchange",
            "entry_low",
            "entry_high",
            "stop_loss",
            "hold_label",
        ]
        if c in (display_rows[0] if display_rows else {})
    ]
    df = pd.DataFrame(display_rows)
    # Format price column for readability in the table
    if "price" in df.columns:
        df = df.copy()
        df["price"] = df.apply(
            lambda r: format_ticker_price(r.get("symbol"), r.get("price")).split(" ", 1)[-1]
            if r.get("price") is not None
            else "—",
            axis=1,
        )
    st.caption(
        f"Ranked by LLM confidence + technical confluence "
        f"({payload.get('ranking') or 'directional only'}). "
        f"Flat/neutral excluded"
        + (
            f" ({payload.get('flat_count', 0)} skipped)."
            if payload.get("flat_count")
            else "."
        )
        + f" Display leverage capped at {payload.get('leverage_display_cap', SCAN_LEVERAGE_CAP)}x. "
        f"Prices are live from the exchange used for each symbol."
    )
    st.dataframe(
        df[cols],
        use_container_width=True,
        hide_index=True,
        key=f"{key_prefix}_results_table",
    )

    skipped = payload.get("skipped_flat") or []
    if skipped:
        with st.expander(f"Skipped flat/neutral ({len(skipped)})", expanded=False):
            st.dataframe(pd.DataFrame(skipped), use_container_width=True, hide_index=True)

    # Expandable detail cards for top results (unique titles avoid expander ID clashes)
    for i, row in enumerate(rows[:5]):
        sym = str(row.get("symbol") or f"row{i}")
        ticker = format_ticker_price(row.get("symbol"), row.get("price"))
        llm_c = row.get("llm_confidence")
        title = (
            f"#{i + 1} {ticker} · {str(row.get('direction', 'flat')).upper()} · "
            f"LLM {llm_c if llm_c is not None else '—'}% · lev {row.get('leverage', '—')}x"
        )
        with st.expander(title, expanded=(i == 0)):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Price · {sym}", ticker.split(" ", 1)[-1] if " " in ticker else ticker)
            c2.metric(
                f"LLM Confidence · {sym}",
                f"{_safe_float(llm_c):.0f}%" if llm_c is not None else "—",
            )
            c3.metric(f"Rank score · {sym}", f"{_safe_float(row.get('rank_score')):.1f}")
            c4.metric(f"Leverage (priority) · {sym}", f"{row.get('leverage', '—')}x")
            if row.get("llm_confidence_reason"):
                st.info(f"**LLM Confidence:** {_safe_float(llm_c):.0f}% — {row['llm_confidence_reason']}")
            elif row.get("reason"):
                st.caption(f"Why: {row['reason']}")
            if row.get("fallback_used"):
                st.caption(
                    f"Fallback used: requested {row.get('exchange_requested')} → {row.get('exchange')}"
                )
            plan = (row.get("payload") or {}).get("trade_plan") or {}
            if plan:
                ez = plan.get("entry_zone") or {}
                st.write(
                    f"**Entry:** {ez.get('low', '—')} – {ez.get('high', '—')}  ·  "
                    f"**SL:** {plan.get('stop_loss', '—')}  ·  "
                    f"**Hold:** {plan.get('hold_detail') or plan.get('hold_label') or '—'}"
                )
                tp_bits = []
                for n in range(1, 5):
                    tp = plan.get(f"tp{n}")
                    rr = plan.get(f"rr_tp{n}")
                    if tp is not None:
                        rr_s = f" (R:R {float(rr):.2f})" if rr is not None else ""
                        tp_bits.append(f"TP{n}: {tp}{rr_s}")
                if tp_bits:
                    st.write(" · ".join(tp_bits))

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
    plan, primary, sim = _plan_bundle(payload)
    direction = (plan.get("direction") or payload.get("direction") or "flat").lower()
    if direction == "flat":
        st.info("No directional trade plan — stand aside or wait for clearer structure.")
        return

    entry_low, entry_high = _entry_zone(plan, primary)
    stop = plan.get("stop_loss") or primary.get("stop_loss")
    tps = _take_profits(plan, primary)
    rrs = _risk_rewards(plan, primary, max(len(tps), 4))
    lev = _display_leverage(payload, plan, primary)
    model_lev = _model_leverage(payload, plan)
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

    st.subheader("🚨 Trade setup")
    dir_emoji = "🟢" if direction == "long" else ("🔴" if direction == "short" else "⚪")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Direction", f"{dir_emoji} {direction.upper()}")
    m2.metric("Leverage (priority)", f"{lev}x")
    m3.metric("Quality", quality)
    m4.metric("Hold", hold_label or "Day trade")

    st.markdown(
        f"""
| Field | Value |
|---|---|
| 🎯 **Entry zone** | **{fp(entry_low)} – {fp(entry_high)}** |
| 🛑 **Stop loss** | **{fp(stop)}** |
| ⚡ **Display leverage** | **{lev}x** (priority cap {SCAN_LEVERAGE_CAP}x) |
| ⏱ **Hold** | {hold} |
| ❌ **Invalidation** | {invalidation} |
"""
    )
    if model_lev is not None and model_lev != lev:
        st.caption(f"Model suggested leverage: {model_lev}x (aggressive band 20–100x).")

    alt_low = plan.get("alternative_entry_low")
    alt_high = plan.get("alternative_entry_high")
    if alt_low is not None and alt_high is not None:
        note = plan.get("alternative_entry_note") or ""
        st.markdown(
            f"🔄 **Alternative entry:** {fp(alt_low)} – {fp(alt_high)}"
            + (f" · _{note}_" if note else "")
        )

    st.markdown("#### Take profits")
    tp_cols = st.columns(max(len(tps), 1))
    for i, tp in enumerate(tps):
        rr = rrs[i] if i < len(rrs) else None
        rr_s = f"{rr:.2f}" if rr is not None else "—"
        with tp_cols[i]:
            st.metric(f"TP{i + 1}", fp(tp), delta=f"R:R {rr_s}")


def render_simulation_card(payload: Dict[str, Any]) -> None:
    plan, primary, sim = _plan_bundle(payload)
    direction = (plan.get("direction") or payload.get("direction") or "flat").lower()
    if direction == "flat":
        st.caption("No simulation — flat / neutral bias.")
        return

    lev = _display_leverage(payload, plan, primary)
    scaled = _scale_sim_to_base(plan, sim, SIM_BASE_USD)

    st.subheader(f"Simulation example (${SIM_BASE_USD:.0f} base)")
    st.caption("For illustration only — not a real balance or order.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk at stop", f"${scaled['risk_amount']:.2f}")
    c2.metric("Notional", f"${scaled['notional']:.2f}")
    c3.metric("Margin @ priority lev", f"${scaled['margin']:.2f}")
    st.write(
        f"If you trade this with **${SIM_BASE_USD:.0f}** at **{lev}x** display leverage priority "
        f"(risk ~{scaled['risk_pct']:.2f}%):"
    )
    if scaled["profits"]:
        for i, profit in enumerate(scaled["profits"], 1):
            pct = scaled["profit_pcts"][i - 1] if i - 1 < len(scaled["profit_pcts"]) else None
            pct_s = f" ({pct:+.2f}% of base)" if pct is not None else ""
            st.write(f"- At **TP{i}** → sim P/L **${profit:,.2f}**{pct_s}")
    else:
        st.write("- Profit targets unavailable for this plan.")
    lev_note = plan.get("leverage_reasoning") or sim.get("leverage_reasoning")
    if lev_note:
        st.caption(f"Leverage logic: {lev_note}")


def render_single_analysis(symbol: str, timeframe: str, exchange: str, no_news: bool) -> None:
    if not symbol:
        st.info("Enter a symbol to run a single analysis.")
        return
    with st.spinner(f"Analyzing {symbol}…"):
        payload = analyze_symbol(symbol, timeframe, exchange, no_news)
    if not payload.get("ok"):
        st.error(payload.get("error") or payload.get("message") or "Analysis failed")
        if payload.get("attempted_exchanges"):
            st.caption(f"Tried exchanges: {', '.join(payload['attempted_exchanges'])}")
        return

    st.success("Analysis complete")
    if payload.get("fallback_used"):
        st.warning(
            f"Auto-fallback: requested **{payload.get('exchange_requested')}** → "
            f"using **{payload.get('exchange')}**"
        )

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Bias", str(payload.get("bias") or payload.get("direction") or "flat").upper())
    llm_c = payload.get("llm_confidence")
    if llm_c is None:
        llm_c = (payload.get("signal") or {}).get("llm_confidence")
    h2.metric(
        "LLM Confidence",
        f"{_safe_float(llm_c):.0f}%" if llm_c is not None else "—",
    )
    h3.metric("Setup", payload.get("setup_name") or "—")
    h4.metric("Exchange", payload.get("exchange") or exchange)

    llm_reason = (
        payload.get("llm_confidence_reason")
        or (payload.get("llm_narrative") or {}).get("confidence_reason")
        or ""
    )
    if llm_c is not None or llm_reason:
        st.info(
            f"**LLM Confidence: {_safe_float(llm_c):.0f}%**"
            + (f" — {llm_reason}" if llm_reason else "")
        )

    conf_total = payload.get("confluence_total")
    tech_c = payload.get("technical_confidence")
    rank_s = payload.get("rank_score")
    meta_bits = []
    if conf_total is not None:
        meta_bits.append(f"Confluence {float(conf_total):+.3f}")
    if tech_c is not None:
        meta_bits.append(f"Technical conf {_safe_float(tech_c):.1f}%")
    if rank_s is not None:
        meta_bits.append(f"Rank score {_safe_float(rank_s):.1f}")
    if payload.get("confidence") is not None:
        meta_bits.append(f"Blended conf {_safe_float(payload.get('confidence')):.1f}%")
    if meta_bits:
        st.caption(" · ".join(meta_bits))

    render_trade_setup_card(payload)
    st.divider()
    render_simulation_card(payload)

    reasons = payload.get("key_reasons") or []
    risks = payload.get("key_risks") or []
    if reasons or risks:
        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Why this signal")
            if reasons:
                for item in reasons[:6]:
                    st.write(f"- {item}")
            else:
                st.caption("—")
        with r2:
            st.subheader("Risk notes")
            if risks:
                for item in risks[:6]:
                    st.write(f"- {item}")
            else:
                st.caption("—")

    if payload.get("key_levels"):
        with st.expander("Key levels"):
            st.json(payload.get("key_levels")[:6])

    if payload.get("trader_commentary"):
        with st.expander("Trader commentary"):
            st.write(payload["trader_commentary"])

    if payload.get("warnings"):
        with st.expander("Warnings"):
            for w in payload["warnings"]:
                st.write(f"- {w}")

    safe_sym = symbol.replace("/", "_").replace(":", "_") or "symbol"
    st.download_button(
        label="Export full report",
        data=build_report_markdown(payload),
        file_name=f"{safe_sym}_report.md",
        mime="text/markdown",
        key=f"download_single_report_{safe_sym}",
    )


def main() -> None:
    _check_access()
    st.title("Perpetual Pro — Web App")
    st.caption(
        "Private-friendly web companion with multi-exchange fallback, "
        f"{SCAN_LEVERAGE_CAP}x display leverage priority, and full trade cards."
    )

    with st.sidebar:
        st.header("Inputs")
        symbol = st.text_input(
            "Symbol",
            value="BTC",
            placeholder="BTC or BTC/USDT:USDT",
            key="input_symbol",
        )
        timeframe = st.selectbox(
            "Timeframe",
            ["1m", "5m", "15m", "1h", "4h"],
            index=2,
            key="input_timeframe",
        )
        try:
            default_ex = load_config().exchange.default or "bybit"
        except Exception:  # noqa: BLE001
            default_ex = "bybit"
        ex_index = EXCHANGE_OPTIONS.index(default_ex) if default_ex in EXCHANGE_OPTIONS else 0
        exchange = st.selectbox(
            "Preferred exchange",
            EXCHANGE_OPTIONS,
            index=ex_index,
            key="input_exchange",
        )
        st.caption("If the symbol is missing on the preferred venue, others are tried automatically.")
        # Default: fetch news (recent lookback). Key bumped so old sessions reset to unchecked.
        no_news = st.checkbox(
            "Skip news",
            value=False,
            key="input_no_news_v2",
            help="When unchecked (default), analysis fetches recent headlines (last ~1–4 hours).",
        )
        st.caption(f"Backend: {BACKEND_URL}")
        if st.button("Ping backend", key="btn_ping_backend"):
            result = call_backend("/health")
            st.json(result)

    col1, col2 = st.columns(2)
    with col1:
        run_single = st.button("Run single analysis", type="primary", key="btn_run_single")
    with col2:
        watchlist = st.text_area(
            "Symbols (comma-separated)",
            value=DEFAULT_WATCHLIST,
            height=100,
            key="input_watchlist",
        )
        run_scan = st.button("Scan watchlist", key="btn_scan_watchlist")

    if run_single:
        render_single_analysis(symbol, timeframe, exchange, no_news)

    if run_scan:
        symbols = normalize_symbol_list(watchlist)
        if not symbols:
            st.warning("Add at least one symbol to scan.")
        else:
            # Always start clean: drop prior ranked table before re-fetching
            st.session_state.pop("last_scan", None)
            with st.spinner(f"Scanning {len(symbols)} symbols (fresh data + news)…"):
                payload = scan_symbols(symbols, timeframe, exchange, no_news)
            st.session_state["last_scan"] = payload
            st.session_state["last_scan_at"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

    st.divider()
    st.subheader("Ranked signals")
    # Render scan results only once (from session state) so widget IDs stay unique
    if st.session_state.get("last_scan"):
        scanned_at = st.session_state.get("last_scan_at")
        if scanned_at:
            st.caption(f"Last fresh scan: {scanned_at}")
        render_scan_results(st.session_state["last_scan"], key_prefix="scan")
    else:
        st.caption("Run a watchlist scan to populate ranked signals.")


if __name__ == "__main__":
    main()
