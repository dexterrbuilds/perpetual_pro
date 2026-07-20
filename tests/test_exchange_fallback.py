"""Unit tests for multi-exchange fallback + client lifecycle."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.multi_tf import (
    FallbackFetchResult,
    MultiTimeframeData,
    fetch_multi_timeframe_with_fallback,
)
from src.utils.config import AppConfig, ExchangeConfig


class _FakeClient:
    """Tracks open/close for leak detection."""

    open_ids: List[str] = []
    closed_ids: List[str] = []

    def __init__(self, exchange_id=None, config=None, exchange_cfg=None):
        self.exchange_id = exchange_id or "bybit"
        _FakeClient.open_ids.append(self.exchange_id)

    def close(self) -> None:
        _FakeClient.closed_ids.append(self.exchange_id)


def _empty_mtf(symbol: str, ex: str, primary_tf: str) -> MultiTimeframeData:
    return MultiTimeframeData(
        symbol=symbol,
        exchange_id=ex,
        primary_tf=primary_tf,
        frames={primary_tf: pd.DataFrame()},
    )


def _filled_mtf(symbol: str, ex: str, primary_tf: str) -> MultiTimeframeData:
    idx = pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        },
        index=idx,
    )
    return MultiTimeframeData(
        symbol=symbol,
        exchange_id=ex,
        primary_tf=primary_tf,
        frames={primary_tf: df},
    )


@pytest.fixture(autouse=True)
def _reset_fake_client():
    _FakeClient.open_ids = []
    _FakeClient.closed_ids = []
    yield
    _FakeClient.open_ids = []
    _FakeClient.closed_ids = []


def test_fallback_uses_first_successful_exchange(monkeypatch):
    import src.data.multi_tf as multi_tf
    import src.data.exchange as exchange_mod

    order = ["bybit", "okx", "mexc"]

    def fake_order(preferred, config=None, auto_fallback=None):
        return list(order)

    def fake_fetch(client, symbol, primary_tf, higher_tfs=None, limit=500, include_snapshot=True, config=None):
        if client.exchange_id == "bybit":
            return _empty_mtf(symbol, "bybit", primary_tf)
        if client.exchange_id == "okx":
            return _filled_mtf(symbol, "okx", primary_tf)
        return _empty_mtf(symbol, client.exchange_id, primary_tf)

    monkeypatch.setattr(multi_tf, "build_exchange_attempt_order", fake_order)
    monkeypatch.setattr(multi_tf, "ExchangeClient", _FakeClient)
    monkeypatch.setattr(multi_tf, "fetch_multi_timeframe", fake_fetch)
    monkeypatch.setattr(exchange_mod, "ExchangeClient", _FakeClient)

    result = fetch_multi_timeframe_with_fallback(
        symbol="ABC",
        primary_tf="15m",
        preferred_exchange="bybit",
        config=AppConfig(exchange=ExchangeConfig(default="bybit", auto_fallback=True)),
    )

    assert isinstance(result, FallbackFetchResult)
    assert result.exchange_used == "okx"
    assert result.fallback_used is True
    assert result.requested_exchange == "bybit"
    assert result.attempted_exchanges == ["bybit", "okx"]
    assert not result.mtf.primary.empty
    # Failed empty client (bybit) must be closed; successful okx stays open for caller
    assert "bybit" in _FakeClient.closed_ids
    assert "okx" not in _FakeClient.closed_ids
    result.client.close()
    assert "okx" in _FakeClient.closed_ids


def test_fallback_closes_clients_on_exception(monkeypatch):
    import src.data.multi_tf as multi_tf

    def fake_order(preferred, config=None, auto_fallback=None):
        return ["bybit", "binanceusdm"]

    def fake_fetch(client, symbol, primary_tf, higher_tfs=None, limit=500, include_snapshot=True, config=None):
        if client.exchange_id == "bybit":
            raise RuntimeError("boom")
        return _filled_mtf(symbol, client.exchange_id, primary_tf)

    monkeypatch.setattr(multi_tf, "build_exchange_attempt_order", fake_order)
    monkeypatch.setattr(multi_tf, "ExchangeClient", _FakeClient)
    monkeypatch.setattr(multi_tf, "fetch_multi_timeframe", fake_fetch)

    result = fetch_multi_timeframe_with_fallback(
        symbol="XYZ",
        primary_tf="15m",
        preferred_exchange="bybit",
    )
    assert result.exchange_used == "binanceusdm"
    assert result.fallback_used is True
    assert "bybit" in _FakeClient.closed_ids
    assert "binanceusdm" not in _FakeClient.closed_ids
    result.client.close()


def test_build_exchange_attempt_order_prefers_requested():
    from src.data.exchange import build_exchange_attempt_order

    order = build_exchange_attempt_order(
        "okx",
        AppConfig(
            exchange=ExchangeConfig(
                default="bybit",
                auto_fallback=True,
                fallback_exchanges=["bybit", "binanceusdm", "okx"],
            )
        ),
    )
    assert order[0] == "okx"
    assert "bybit" in order
    assert len(order) == len(set(order))
