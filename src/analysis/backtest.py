"""Simple prop-oriented historical signal backtest.

Rule-based EMA/RSI/ATR system (fast approximation of the live stack).
Uses 0.5–1% risk and ≤5x leverage when prop_mode is enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.analysis.indicators import compute_indicators
from src.data.exchange import ExchangeClient
from src.utils.config import AppConfig, RiskConfig, load_config
from src.utils.helpers import normalize_symbol, safe_float


@dataclass
class BacktestTrade:
    entry_time: str
    exit_time: str
    direction: str
    entry: float
    exit: float
    stop: float
    tp: float
    pnl: float
    pnl_pct: float
    bars_held: int
    reason: str


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    n_bars: int
    n_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    net_pnl: float
    net_pnl_pct: float
    final_equity: float
    starting_equity: float
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    prop_settings: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_backtest(
    symbol: str,
    *,
    timeframe: str = "15m",
    bars: int = 500,
    config: Optional[AppConfig] = None,
    exchange: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    step: int = 4,
    warmup: int = 80,
    max_hold_bars: int = 24,
) -> BacktestResult:
    """
    Run a simple long/short backtest with prop risk rules.

    If ``df`` is provided, no network fetch is performed (tests).
    """
    cfg = config or load_config()
    risk: RiskConfig = cfg.risk
    prop = bool(getattr(risk, "prop_mode", True))
    risk_pct = float(risk.risk_per_trade_pct or 1.0)
    if prop:
        lo = float(getattr(risk, "risk_per_trade_min_pct", 0.5) or 0.5)
        hi = float(getattr(risk, "risk_per_trade_max_pct", 1.0) or 1.0)
        risk_pct = max(lo, min(hi, risk_pct))
    max_lev = min(5.0, float(getattr(risk, "max_leverage", 5) or 5)) if prop else float(
        getattr(risk, "leverage_ceiling", 20) or 20
    )
    stop_atr = float(getattr(risk, "default_stop_atr_mult", 1.0) or 1.0)
    tp_mults = list(getattr(risk, "default_tp_atr_mults", None) or [0.7, 1.3, 2.0, 3.0])
    tp_atr = float(tp_mults[0] if tp_mults else 0.7)
    capital0 = float(getattr(risk, "simulated_capital", None) or 1000.0)

    sym = normalize_symbol(symbol)
    if df is None:
        client = ExchangeClient(exchange_id=exchange or cfg.exchange.default, config=cfg)
        try:
            raw = client.fetch_ohlcv(sym, timeframe=timeframe, limit=max(bars, warmup + 50))
        finally:
            client.close()
        df = raw
    if df is None or df.empty or len(df) < warmup + 10:
        return BacktestResult(
            symbol=sym,
            timeframe=timeframe,
            n_bars=0,
            n_trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            net_pnl=0.0,
            net_pnl_pct=0.0,
            final_equity=capital0,
            starting_equity=capital0,
            notes=["Insufficient OHLCV for backtest."],
            prop_settings={"prop_mode": prop, "risk_pct": risk_pct, "max_leverage": max_lev},
        )

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index, utc=True)
    try:
        suite = compute_indicators(work, cfg)
        work = suite.df if suite is not None and suite.df is not None else work
    except Exception as exc:  # noqa: BLE001
        logger.warning("Indicator compute failed in backtest: {}", exc)

    equity = capital0
    peak = capital0
    max_dd = 0.0
    curve: List[Dict[str, Any]] = []
    trades: List[BacktestTrade] = []
    open_pos: Optional[Dict[str, Any]] = None

    def _col(*names: str) -> Optional[str]:
        for n in names:
            if n in work.columns:
                return n
        return None

    atr_col = _col("atr", "ATR")
    ema_f = _col("ema_fast", "EMA_fast", "ema_9", "EMA_9")
    ema_m = _col("ema_mid", "EMA_mid", "ema_21", "EMA_21")
    rsi_col = _col("rsi", "RSI", "rsi_14", "RSI_14")

    closes = work["close"].astype(float)
    highs = work["high"].astype(float)
    lows = work["low"].astype(float)

    i = warmup
    while i < len(work) - 1:
        ts = work.index[i]
        price = float(closes.iloc[i])
        # ATR proxy
        if atr_col:
            atr = float(work[atr_col].iloc[i] or price * 0.01)
        else:
            atr = float((highs.iloc[i - 14 : i + 1] - lows.iloc[i - 14 : i + 1]).mean() or price * 0.01)
        atr = max(atr, price * 0.001)

        # Manage open trade
        if open_pos is not None:
            direction = open_pos["direction"]
            hit = None
            hi = float(highs.iloc[i])
            lo = float(lows.iloc[i])
            if direction == "long":
                if lo <= open_pos["stop"]:
                    hit = ("sl", open_pos["stop"])
                elif hi >= open_pos["tp"]:
                    hit = ("tp", open_pos["tp"])
            else:
                if hi >= open_pos["stop"]:
                    hit = ("sl", open_pos["stop"])
                elif lo <= open_pos["tp"]:
                    hit = ("tp", open_pos["tp"])
            held = i - open_pos["entry_i"]
            if hit is None and held >= max_hold_bars:
                hit = ("time", price)
            if hit is not None:
                exit_px = float(hit[1])
                units = open_pos["units"]
                if direction == "long":
                    pnl = units * (exit_px - open_pos["entry"])
                else:
                    pnl = units * (open_pos["entry"] - exit_px)
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
                trades.append(
                    BacktestTrade(
                        entry_time=str(open_pos["entry_time"]),
                        exit_time=str(ts),
                        direction=direction,
                        entry=open_pos["entry"],
                        exit=exit_px,
                        stop=open_pos["stop"],
                        tp=open_pos["tp"],
                        pnl=float(pnl),
                        pnl_pct=float(pnl / capital0 * 100.0),
                        bars_held=held,
                        reason=hit[0],
                    )
                )
                open_pos = None
                curve.append({"t": str(ts), "equity": round(equity, 4)})
            i += 1
            continue

        # New signal every `step` bars
        if (i - warmup) % max(1, step) != 0:
            i += 1
            continue

        direction = _signal_direction(work, i, ema_f, ema_m, rsi_col)
        if direction not in ("long", "short"):
            i += 1
            continue

        if direction == "long":
            stop = price - atr * stop_atr
            tp = price + atr * tp_atr
            risk_per_unit = max(price - stop, price * 1e-6)
        else:
            stop = price + atr * stop_atr
            tp = price - atr * tp_atr
            risk_per_unit = max(stop - price, price * 1e-6)

        risk_amount = equity * (risk_pct / 100.0)
        units = risk_amount / risk_per_unit
        notional = units * price
        margin = notional / max(max_lev, 1e-9)
        if margin > equity:
            notional = equity * max_lev
            units = notional / price

        open_pos = {
            "direction": direction,
            "entry": price,
            "stop": stop,
            "tp": tp,
            "units": units,
            "entry_i": i,
            "entry_time": ts,
        }
        curve.append({"t": str(ts), "equity": round(equity, 4)})
        i += 1

    # Force-close open pos at last bar
    if open_pos is not None:
        last_i = len(work) - 1
        exit_px = float(closes.iloc[last_i])
        units = open_pos["units"]
        if open_pos["direction"] == "long":
            pnl = units * (exit_px - open_pos["entry"])
        else:
            pnl = units * (open_pos["entry"] - exit_px)
        equity += pnl
        trades.append(
            BacktestTrade(
                entry_time=str(open_pos["entry_time"]),
                exit_time=str(work.index[last_i]),
                direction=open_pos["direction"],
                entry=open_pos["entry"],
                exit=exit_px,
                stop=open_pos["stop"],
                tp=open_pos["tp"],
                pnl=float(pnl),
                pnl_pct=float(pnl / capital0 * 100.0),
                bars_held=last_i - open_pos["entry_i"],
                reason="eod",
            )
        )
        curve.append({"t": str(work.index[last_i]), "equity": round(equity, 4)})

    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    pf = (gross_win / gross_loss) if gross_loss > 1e-12 else (999.0 if gross_win > 0 else 0.0)
    n = len(trades)
    wr = (wins / n * 100.0) if n else 0.0
    net = equity - capital0

    # Rebuild max DD from curve if empty
    if curve:
        eq = np.array([c["equity"] for c in curve], dtype=float)
        peaks = np.maximum.accumulate(eq)
        dds = np.where(peaks > 0, (peaks - eq) / peaks * 100.0, 0.0)
        max_dd = float(np.max(dds)) if len(dds) else max_dd

    return BacktestResult(
        symbol=sym,
        timeframe=timeframe,
        n_bars=len(work),
        n_trades=n,
        wins=wins,
        losses=losses,
        win_rate=round(wr, 2),
        profit_factor=round(float(pf), 3),
        max_drawdown_pct=round(float(max_dd), 3),
        net_pnl=round(float(net), 4),
        net_pnl_pct=round(float(net / capital0 * 100.0), 3),
        final_equity=round(float(equity), 4),
        starting_equity=capital0,
        equity_curve=curve,
        trades=[asdict(t) for t in trades[-50:]],
        prop_settings={
            "prop_mode": prop,
            "risk_pct": risk_pct,
            "max_leverage": max_lev,
            "stop_atr_mult": stop_atr,
            "tp_atr_mult": tp_atr,
        },
        notes=[
            "Rule-based EMA/RSI/ATR backtest — approximates live confluence stack.",
            f"Prop rules: risk {risk_pct}% · max lev {max_lev:.0f}x · one position at a time.",
        ],
    )


def _signal_direction(
    df: pd.DataFrame,
    i: int,
    ema_f: Optional[str],
    ema_m: Optional[str],
    rsi_col: Optional[str],
) -> str:
    """Lightweight directional bias at bar i."""
    close = float(df["close"].iloc[i])
    score = 0.0
    if ema_f and ema_m and pd.notna(df[ema_f].iloc[i]) and pd.notna(df[ema_m].iloc[i]):
        ef = float(df[ema_f].iloc[i])
        em = float(df[ema_m].iloc[i])
        if ef > em and close > ef:
            score += 1.0
        elif ef < em and close < ef:
            score -= 1.0
    if rsi_col and pd.notna(df[rsi_col].iloc[i]):
        rsi = float(df[rsi_col].iloc[i])
        if rsi >= 55:
            score += 0.5
        elif rsi <= 45:
            score -= 0.5
    # Fallback: short momentum
    if abs(score) < 0.5 and i >= 5:
        ret = close / float(df["close"].iloc[i - 5]) - 1.0
        if ret > 0.004:
            score += 0.8
        elif ret < -0.004:
            score -= 0.8
    if score >= 0.8:
        return "long"
    if score <= -0.8:
        return "short"
    return "flat"
