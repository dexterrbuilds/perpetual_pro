#!/usr/bin/env python3
"""
perpetual_pro — professional crypto perpetual futures analysis CLI.

Usage:
    python main.py BTC/USDT:USDT
    python main.py ETH --exchange bybit --timeframe 15m
    python main.py --screen
    python main.py --screen --image chart.png --symbol SOL
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
