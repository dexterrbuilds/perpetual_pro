"""URL → symbol parsing tests (TradingView priority)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.url_symbol import parse_chart_url


def test_tradingview_query_symbol():
    h = parse_chart_url(
        "https://www.tradingview.com/chart/?symbol=BINANCE%3ABTCUSDT.P&interval=15"
    )
    assert h.symbol and "BTC" in h.symbol
    assert h.timeframe == "15m"
    assert h.exchange_hint == "binanceusdm"
    assert h.confidence >= 0.9


def test_tradingview_symbols_path():
    h = parse_chart_url("https://www.tradingview.com/symbols/ETHUSDT/")
    assert h.symbol and "ETH" in h.symbol


def test_bybit_url():
    h = parse_chart_url("https://www.bybit.com/trade/usdt/SOLUSDT")
    assert h.symbol and "SOL" in h.symbol
    assert h.exchange_hint == "bybit"


def test_tradingview_exchange_aliases():
    for url, expected in [
        ("https://www.tradingview.com/chart/?symbol=MEXC:BTCUSDT.P", "mexc"),
        ("https://www.tradingview.com/chart/?symbol=BINGX:ETHUSDT.P", "bingx"),
        ("https://www.tradingview.com/chart/?symbol=BITFINEX:SOLUSDT.P", "bitfinex"),
        ("https://www.tradingview.com/chart/?symbol=BITGET:BTCUSDT.P", "bitget"),
        ("https://www.tradingview.com/chart/?symbol=GATEIO:BTCUSDT.P", "gateio"),
        ("https://www.tradingview.com/chart/?symbol=HUOBI:BTCUSDT.P", "huobi"),
        ("https://www.tradingview.com/chart/?symbol=WEEX:BTCUSDT.P", "weex"),
    ]:
        h = parse_chart_url(url)
        assert h.exchange_hint == expected


def test_empty_url():
    h = parse_chart_url("")
    assert h.symbol is None
