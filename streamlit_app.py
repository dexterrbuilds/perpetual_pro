from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.api.service import AnalyzeRequest
from src.utils.config import load_config


st.set_page_config(page_title="Perpetual Pro", page_icon="📈", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


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


def build_report_markdown(payload: Dict[str, Any]) -> str:
    symbol = payload.get("symbol") or "—"
    direction = payload.get("direction") or payload.get("bias") or "flat"
    confidence = payload.get("confidence") or payload.get("signal", {}).get("confidence_pct") or 0
    setup = payload.get("setup_name") or payload.get("signal", {}).get("setup_name") or "—"
    plan = payload.get("trade_plan") or {}
    entry = plan.get("entry") or plan.get("entry_price") or "—"
    stop = plan.get("stop_loss") or plan.get("sl") or "—"
    tp = plan.get("take_profit") or plan.get("tp") or "—"
    score = payload.get("signal", {}).get("confluence_score") or payload.get("confluence_total") or "—"
    return f"""# {symbol}

- Direction: {direction}
- Confidence: {confidence}
- Setup: {setup}
- Confluence: {score}

## Trade plan
- Entry: {entry}
- Stop Loss: {stop}
- Take Profit: {tp}
"""


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


@st.cache_data(show_spinner=False)
def analyze_symbol(symbol: str, timeframe: str, exchange: str, no_news: bool) -> Dict[str, Any]:
    from src.api.service import analyze_from_image

    req = AnalyzeRequest(symbol=symbol, timeframe=timeframe, exchange=exchange, no_news=no_news)
    cfg = load_config()
    payload = analyze_from_image(
        image=b"",
        request=req,
        config=cfg,
    )
    return payload


@st.cache_data(show_spinner=False)
def scan_symbols(symbols: List[str], timeframe: str, exchange: str, no_news: bool) -> Dict[str, Any]:
    from src.api.service import scan_symbols as scan_backend

    req = AnalyzeRequest(timeframe=timeframe, exchange=exchange, no_news=no_news)
    cfg = load_config()
    return scan_backend(symbols, request=req, config=cfg)


def render_scan_results(payload: Dict[str, Any]) -> None:
    rows = payload.get("ranked_results", [])
    if not rows:
        st.info("No ranked setups returned yet. Try a broader watchlist or a different exchange.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(
        df[["symbol", "direction", "confidence", "confluence_score", "setup_name", "leverage", "price"]],
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Export scan report"):
        st.download_button(
            label="Download scan report",
            data=build_report_markdown(rows[0]),
            file_name="scan_report.md",
            mime="text/markdown",
        )


def render_single_analysis(symbol: str, timeframe: str, exchange: str, no_news: bool) -> None:
    if not symbol:
        st.info("Enter a symbol to run a single analysis.")
        return
    with st.spinner(f"Analyzing {symbol}…"):
        payload = analyze_symbol(symbol, timeframe, exchange, no_news)
    if not payload.get("ok"):
        st.error(payload.get("error") or payload.get("message") or "Analysis failed")
        return
    st.success("Analysis complete")
    st.metric("Bias", payload.get("bias") or payload.get("direction") or "flat")
    st.metric("Confidence", f"{payload.get('confidence', 0):.1f}%")
    st.metric("Setup", payload.get("setup_name") or "—")
    st.write(payload.get("trade_plan") or {})
    if payload.get("key_levels"):
        st.json(payload.get("key_levels")[:6])
    st.download_button(
        label="Export report",
        data=build_report_markdown(payload),
        file_name=f"{symbol.replace('/', '_')}_report.md",
        mime="text/markdown",
    )


def main() -> None:
    _check_access()
    st.title("Perpetual Pro — Web App")
    st.caption("Private-friendly web companion to the Chrome extension with shared analysis logic.")

    with st.sidebar:
        st.header("Inputs")
        symbol = st.text_input("Symbol", value="BTC", placeholder="BTC or BTC/USDT:USDT")
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h"], index=2)
        exchange = st.selectbox("Exchange", ["binanceusdm", "bybit", "okx", "bitget", "mexc", "bingx"], index=0)
        no_news = st.checkbox("Skip news")
        st.caption(f"Backend: {BACKEND_URL}")
        if st.button("Ping backend"):
            result = call_backend("/health")
            st.json(result)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run single analysis"):
            render_single_analysis(symbol, timeframe, exchange, no_news)
    with col2:
        if st.button("Scan watchlist"):
            symbols = normalize_symbol_list(st.text_area("Symbols (comma-separated)", value="BTC,ETH,SOL,BNB,AAVE,ADA,AERO,ALGO,ARB,ATOM,AXS,BCH,BERA,BGB,BRETT,CAKE,BSV,CHZ,CRO,DASH,DOT,FIL,GRASS,HBAR,ICP,INJ,JUP,XLM,XMR,XTZ,ZEC,TIA,S,SEI,RAY,OP,NEAR,MNT,KAS,KAIA"))
            if not symbols:
                st.warning("Add at least one symbol to scan.")
            else:
                with st.spinner("Scanning watchlist…"):
                    payload = scan_symbols(symbols, timeframe, exchange, no_news)
                st.session_state["last_scan"] = payload
                render_scan_results(payload)

    st.divider()
    st.subheader("Ranked signals")
    if st.session_state.get("last_scan"):
        render_scan_results(st.session_state["last_scan"])


if __name__ == "__main__":
    main()
