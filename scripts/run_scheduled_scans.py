#!/usr/bin/env python3
"""Run prop watchlist scans on a WAT schedule (or once).

Sends prop-safe Telegram chart alerts at DST-aware market sessions:
  London open · New York open · New York 3 p.m. liquidity window

Usage:
  export TELEGRAM_BOT_TOKEN=...   # required for alerts (never commit)
  export TELEGRAM_CHAT_ID=...
  python scripts/run_scheduled_scans.py --once
  python scripts/run_scheduled_scans.py          # loop WAT slots

Credentials are read ONLY from the environment — never from config.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduler.scan_job import run_scheduler_loop
from src.utils.config import load_config, setup_logging


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prop scheduled watchlist scans + Telegram")
    p.add_argument("--once", action="store_true", help="Run one scan now and exit")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg)
    run_scheduler_loop(cfg, once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
