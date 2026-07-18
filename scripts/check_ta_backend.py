#!/usr/bin/env python3
"""Smoke-test indicator backend (pandas-ta-classic)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.indicators import compute_indicators
from src.analysis.ta_backend import get_ta, ta_available


def main() -> int:
    ta, name = get_ta()
    print(f"backend={name} available={ta_available()}")
    n = 250
    rng = np.random.default_rng(1)
    close = 50000 + np.cumsum(rng.normal(0.1, 30, n))
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 10, n),
            "high": close + 40,
            "low": close - 40,
            "close": close,
            "volume": rng.uniform(100, 500, n),
        }
    )
    suite = compute_indicators(df)
    print(f"indicator_count={suite.indicator_count}")
    print(f"errors={suite.errors}")
    print(f"ta_backend={suite.summary.get('ta_backend')}")
    print(f"rsi={suite.summary.get('rsi')}")
    print(f"atr={suite.summary.get('atr')}")
    print(f"macd_hist={suite.summary.get('macd_hist')}")
    print(f"trend_score={suite.summary.get('trend_score')}")
    for key in ("ema_fast", "rsi", "atr"):
        assert key in suite.df.columns, f"missing {key}"
    assert suite.summary.get("rsi") is not None
    assert suite.summary.get("atr") is not None
    assert suite.indicator_count >= 20
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
