"""Administrative CLI for the durable outcome scoring system."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from src.analysis.legacy_v2 import compare_legacy_v2_outcomes
from src.data.multi_tf import fetch_multi_timeframe_with_fallback
from src.scoring.replay import replay_historical_candidates
from src.scoring.readiness import assess_shadow_readiness
from src.scoring.repository import OutcomeRepository
from src.scoring.training import train_outcome_model
from src.utils.config import DEFAULT_CRYPTO_WATCHLIST, load_config


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Perpetual Pro outcome scorer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Check Supabase schema and model state")
    subparsers.add_parser(
        "readiness",
        help=(
            "Run read-only chronological shadow replacement gates; never saves "
            "or promotes a model"
        ),
    )
    compare_parser = subparsers.add_parser(
        "compare-legacy-v2",
        help="Compare execution-aware Legacy V2 against stored legacy outcomes",
    )
    compare_parser.add_argument(
        "--confidence-floor",
        type=float,
        default=80.0,
    )
    train_parser = subparsers.add_parser(
        "train-shadow",
        help="Train and save a shadow model from labeled candidates",
    )
    train_parser.add_argument(
        "--allow-small-sample",
        action="store_true",
        help="Development only: lower sample requirements; never promotes a model",
    )
    replay_parser = subparsers.add_parser(
        "backfill-replay",
        help="Replay recent closed candles through the live engine and persist labels",
    )
    replay_parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_CRYPTO_WATCHLIST),
    )
    replay_parser.add_argument("--bars", type=int, default=700)
    replay_parser.add_argument("--step", type=int, default=3)
    promote_parser = subparsers.add_parser(
        "promote",
        help="Explicitly promote a calibrated shadow that passed every gate",
    )
    promote_parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    config = load_config()
    repository = OutcomeRepository(config.outcome_scoring.database_url)
    if args.command == "status":
        ready = repository.check_ready()
        print(json.dumps({"ready": ready, **repository.status()}, indent=2))
        return 0 if ready else 2
    if args.command == "compare-legacy-v2":
        rows = repository.load_training_rows()
        comparison = compare_legacy_v2_outcomes(
            rows,
            alert_confidence_floor=float(args.confidence_floor),
        )
        print(json.dumps(comparison, indent=2))
        return 0 if comparison["labeled_candidates"] else 6
    if args.command == "readiness":
        rows = repository.load_training_rows()
        readiness = assess_shadow_readiness(rows, config)
        print(json.dumps(readiness, indent=2))
        return 0 if readiness.get("ready_to_replace_current") else 7
    if args.command == "backfill-replay":
        symbols = [
            value.strip()
            for value in str(args.symbols or "").split(",")
            if value.strip()
        ]
        total_candidates = 0
        total_outcomes = 0
        errors = []
        for symbol in symbols:
            try:
                fetch = fetch_multi_timeframe_with_fallback(
                    symbol=symbol,
                    primary_tf=config.timeframes.primary,
                    preferred_exchange=config.exchange.default,
                    higher_tfs=list(config.timeframes.higher),
                    limit=max(300, int(args.bars)),
                    include_snapshot=False,
                    config=config,
                )
                try:
                    candidates, outcomes = replay_historical_candidates(
                        symbol=fetch.mtf.symbol,
                        exchange_id=fetch.exchange_used,
                        primary_tf=config.timeframes.primary,
                        frames=fetch.mtf.frames,
                        config=config,
                        step=max(1, int(args.step)),
                    )
                finally:
                    fetch.client.close()
                written_candidates = repository.record_candidates(candidates)
                written_outcomes = repository.record_replay_outcomes(outcomes)
                total_candidates += written_candidates
                total_outcomes += written_outcomes
                print(
                    json.dumps(
                        {
                            "symbol": symbol,
                            "candidates": written_candidates,
                            "outcomes": written_outcomes,
                        }
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}:{type(exc).__name__}")
        print(
            json.dumps(
                {
                    "candidate_rows": total_candidates,
                    "outcome_rows": total_outcomes,
                    "errors": errors,
                },
                indent=2,
            )
        )
        return 0 if total_outcomes else 4
    if args.command == "promote":
        result = repository.promote_model(str(args.version))
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 5

    rows = repository.load_training_rows()
    minimum_training = config.outcome_scoring.minimum_training_samples
    minimum_calibration = config.outcome_scoring.minimum_calibration_samples
    if args.allow_small_sample:
        minimum_training = 100
        minimum_calibration = 50
    result = train_outcome_model(
        rows,
        minimum_training_samples=minimum_training,
        minimum_calibration_samples=minimum_calibration,
        conservative_quantile=config.outcome_scoring.conservative_quantile,
        promotion_max_ece=config.outcome_scoring.promotion_max_ece,
        promotion_minimum_unseen_samples=(
            config.outcome_scoring.promotion_minimum_unseen_samples
        ),
        feature_schema_version=config.outcome_scoring.feature_schema_version,
    )
    saved = repository.save_model(
        version=result.artifact.version,
        stage="shadow",
        feature_schema_version=result.artifact.feature_schema_version,
        training_start=result.training_start,
        training_end=result.training_end,
        training_samples=result.artifact.training_samples,
        calibration_samples=result.artifact.calibration_samples,
        artifact=result.artifact.to_dict(),
        metrics=result.metrics,
        validation=result.validation,
        folds=result.folds,
    )
    print(
        json.dumps(
            {
                "saved": saved,
                "version": result.artifact.version,
                "validation": result.validation,
            },
            indent=2,
        )
    )
    return 0 if saved else 3


if __name__ == "__main__":
    sys.exit(main())
