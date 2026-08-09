"""Legacy comparison qualification over current corrected analysis outputs.

The policy deliberately changes publication selectivity only. It does not
recalculate Technical Quality, Execution Quality, Overall Quality, Rank,
entries, stops, targets, or trading costs.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.analysis.qualification import evaluate_private_beta_qualification
from src.experiments.identity import identity_metadata


LEGACY_POLICY_VERSION = "legacy_comparison_v1"
LEGACY_MIN_OVERALL_DEFAULT = 80.0
LEGACY_MIN_EXECUTION_DEFAULT = 65.0
LEGACY_MIN_NET_RR_DEFAULT = 0.75
LEGACY_PREFERRED_RR_DEFAULT = 1.25
LEGACY_ABSOLUTE_CHASE_ATR_DEFAULT = 1.35

RELAXABLE_EXECUTION_FAILURES = frozenset(
    {
        "stop_beyond_maximum_structure_distance",
        "reversal_confirmation_insufficient",
        "cmp_position_excessively_extended_inside_zone",
        "setup_confirmation_insufficient",
    }
)

UNIVERSAL_EXECUTION_FAILURES = frozenset(
    {
        "stale_execution_data",
        "spread_above_hard_limit",
        "no_feasible_target",
        "tp1_blocked_by_nearby_structure",
        "tp1_reward_does_not_cover_cost_buffer",
        "range_setup_during_trend_expansion",
        "adverse_structure_change_before_confirmation",
    }
)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def legacy_thresholds() -> Dict[str, float]:
    return {
        "minimum_overall_quality": _env_float(
            "LEGACY_MIN_OVERALL_QUALITY", LEGACY_MIN_OVERALL_DEFAULT
        ),
        "minimum_execution_quality": _env_float(
            "LEGACY_MIN_EXECUTION_QUALITY", LEGACY_MIN_EXECUTION_DEFAULT
        ),
        "minimum_net_rr": _env_float(
            "LEGACY_MIN_NET_RR", LEGACY_MIN_NET_RR_DEFAULT
        ),
        "preferred_net_rr": _env_float(
            "LEGACY_PREFERRED_NET_RR", LEGACY_PREFERRED_RR_DEFAULT
        ),
        "preferred_gross_rr": _env_float(
            "LEGACY_PREFERRED_GROSS_RR", LEGACY_PREFERRED_RR_DEFAULT
        ),
        "preferred_immediate_sl_risk": _env_float(
            "LEGACY_PREFERRED_IMMEDIATE_SL_RISK", 32.0
        ),
        "preferred_chase_distance_atr": _env_float(
            "LEGACY_PREFERRED_CHASE_ATR", 1.0
        ),
        "absolute_chase_distance_atr": _env_float(
            "LEGACY_ABSOLUTE_CHASE_ATR", LEGACY_ABSOLUTE_CHASE_ATR_DEFAULT
        ),
    }


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rr(row: Mapping[str, Any], key: str) -> Optional[float]:
    values = list(row.get(key) or [])
    if not values:
        payload = dict(row.get("payload") or {})
        execution = dict(payload.get("execution") or {})
        setup = dict(payload.get("primary_setup") or payload.get("trade_plan") or {})
        values = list(execution.get(key) or setup.get(key) or [])
    if not values:
        return None
    return _number(values[1 if len(values) > 1 else 0])


def _direction(row: Mapping[str, Any]) -> str:
    return str(
        row.get("evaluated_direction") or row.get("direction") or "flat"
    ).lower()


def _entry_expired(row: Mapping[str, Any], now: Optional[datetime]) -> bool:
    raw = str(row.get("entry_valid_until") or "").strip()
    if not raw:
        return False
    try:
        expiry = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return expiry <= (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _valid_levels(row: Mapping[str, Any], direction: str) -> bool:
    low = _number(row.get("entry_low"))
    high = _number(row.get("entry_high"))
    stop = _number(row.get("stop_loss"))
    targets = [_number(value) for value in list(row.get("take_profits") or [])]
    targets = [value for value in targets if value is not None and value > 0]
    if low is None or high is None or stop is None or min(low, high, stop) <= 0:
        return False
    entry_mid = (low + high) / 2.0
    if direction == "long":
        return stop < entry_mid and any(target > entry_mid for target in targets)
    return stop > entry_mid and any(target < entry_mid for target in targets)


def _failure_item(code: str, explanation: str, actual: Any = None, required: Any = None) -> Dict[str, Any]:
    return {
        "code": code,
        "explanation": explanation,
        "actual_value": actual,
        "required_value": required,
    }


def _quality_tier(overall: float, execution: float) -> Dict[str, str]:
    if overall >= 85.0 and execution >= 80.0:
        return {"code": "a_plus", "label": "A+ SETUP", "badge": "💎"}
    if overall >= 82.0 and execution >= 72.0:
        return {"code": "a", "label": "A SETUP", "badge": "💚"}
    return {"code": "b", "label": "B SETUP", "badge": "🟡"}


def evaluate_legacy_qualification(
    row: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    thresholds = legacy_thresholds()
    direction = _direction(row)
    overall = _number(row.get("confidence", row.get("overall_quality")))
    execution = _number(
        row.get("execution_quality", row.get("execution_score"))
    )
    net_rr = _rr(row, "net_risk_reward")
    gross_rr = _rr(row, "gross_risk_reward")
    spread = _number(row.get("spread_bps"))
    ticker_age = _number(row.get("ticker_age_seconds"))
    orderbook_age = _number(row.get("orderbook_age_seconds"))
    chase = _number(row.get("chase_distance_atr"))
    progress = _number(row.get("tp1_progress_pct"))
    entry_status = str(row.get("entry_status") or "blocked").lower()
    hard_failures = [str(value) for value in list(row.get("hard_failures") or [])]

    failures: List[Dict[str, Any]] = []
    caveats: List[Dict[str, Any]] = []

    if direction not in {"long", "short"}:
        failures.append(_failure_item("INVALID_DIRECTION", "A valid LONG or SHORT direction is required", direction, ["long", "short"]))
    if row.get("market_supported") is False:
        failures.append(_failure_item("UNSUPPORTED_MARKET", "The configured venue does not support this perpetual market"))
    if row.get("data_quality_ok") is False:
        failures.append(_failure_item("INVALID_MARKET_DATA", "Candle data is malformed, incomplete, discontinuous, or stale"))
    if ticker_age is not None and ticker_age > 45.0:
        failures.append(_failure_item("STALE_TICKER", "Ticker data is stale", ticker_age, 45.0))
    if orderbook_age is not None and orderbook_age > 30.0:
        failures.append(_failure_item("STALE_ORDER_BOOK", "Order-book data is stale", orderbook_age, 30.0))
    if spread is not None and spread > 12.0:
        failures.append(_failure_item("UNUSABLE_SPREAD", "Spread exceeds the existing hard market limit", spread, 12.0))
    if not _valid_levels(row, direction):
        failures.append(_failure_item("INVALID_TRADE_LEVELS", "Entry, stop, and target geometry is not coherent"))
    if _entry_expired(row, now):
        failures.append(_failure_item("ENTRY_EXPIRED", "The entry window has expired"))
    if entry_status in {"blocked", "expired", "invalidated"}:
        failures.append(_failure_item("ENTRY_INVALID", f"Entry state is {entry_status}"))
    elif entry_status == "avoid_chase":
        caveats.append(
            _failure_item(
                "ENTRY_PATIENCE_REQUIRED",
                "The setup is valid only at the published zone; do not chase current price",
                chase,
                thresholds["absolute_chase_distance_atr"],
            )
        )
    if chase is not None and chase > thresholds["absolute_chase_distance_atr"]:
        failures.append(_failure_item("EXTREME_CHASE", "Price is beyond the historical absolute chase boundary", chase, thresholds["absolute_chase_distance_atr"]))
    if progress is not None and progress >= 70.0 and str(row.get("entry_zone_relation") or "") == "favorable_beyond":
        failures.append(_failure_item("TP1_MOVE_MOSTLY_COMPLETED", "Too much of the TP1 move occurred before entry", progress, 70.0))
    for failure in hard_failures:
        if failure in UNIVERSAL_EXECUTION_FAILURES:
            failures.append(_failure_item(f"EXECUTION_{failure.upper()}", failure.replace("_", " ")))
        elif failure in RELAXABLE_EXECUTION_FAILURES:
            caveats.append(_failure_item(f"CAVEAT_{failure.upper()}", failure.replace("_", " ")))
        else:
            failures.append(_failure_item("UNKNOWN_EXECUTION_FAILURE", failure.replace("_", " ")))

    overall_passed = overall is not None and overall >= thresholds["minimum_overall_quality"]
    execution_passed = execution is not None and execution >= thresholds["minimum_execution_quality"]
    net_rr_passed = net_rr is not None and net_rr >= thresholds["minimum_net_rr"]
    if not overall_passed:
        failures.append(_failure_item("OVERALL_QUALITY_BELOW_LEGACY_MINIMUM", "Overall Quality is below the Legacy minimum", overall, thresholds["minimum_overall_quality"]))
    if not execution_passed:
        failures.append(_failure_item("EXECUTION_QUALITY_BELOW_LEGACY_MINIMUM", "Execution Quality is below the Legacy minimum", execution, thresholds["minimum_execution_quality"]))
    if not net_rr_passed:
        failures.append(_failure_item("NET_RR_BELOW_LEGACY_ABSOLUTE_FLOOR", "Net R:R is below the absolute Legacy economics floor", net_rr, thresholds["minimum_net_rr"]))

    immediate = _number(row.get("immediate_sl_risk"))
    confluence = _number(row.get("confluence_score"))
    rank = _number(row.get("authoritative_rank", row.get("rank_score")))
    stop_quality = _number(row.get("stop_quality"))
    target_values = [_number(value) for value in list(row.get("target_feasibility") or [])]
    target_quality = max((value for value in target_values if value is not None), default=None)
    if net_rr_passed and net_rr is not None and net_rr < thresholds["preferred_net_rr"]:
        caveats.append(_failure_item("NET_RR_BELOW_PREFERENCE", "Reward efficiency is below the preferred level", net_rr, thresholds["preferred_net_rr"]))
    if gross_rr is not None and gross_rr < thresholds["preferred_gross_rr"]:
        caveats.append(_failure_item("GROSS_RR_BELOW_PREFERENCE", "Gross R:R is below the preferred level", gross_rr, thresholds["preferred_gross_rr"]))
    if immediate is not None and immediate > thresholds["preferred_immediate_sl_risk"]:
        caveats.append(_failure_item("IMMEDIATE_SL_ABOVE_PREFERENCE", "Immediate-SL risk is elevated but the stop remains structurally valid", immediate, thresholds["preferred_immediate_sl_risk"]))
    if chase is not None and thresholds["preferred_chase_distance_atr"] < chase <= thresholds["absolute_chase_distance_atr"]:
        caveats.append(_failure_item("MARGINAL_ENTRY_EXTENSION", "Entry requires patience; do not chase the published zone", chase, thresholds["preferred_chase_distance_atr"]))
    if confluence is not None and confluence < 0.20:
        caveats.append(_failure_item("CONFLUENCE_BELOW_PREFERENCE", "Directional confluence is below the historical preference", confluence, 0.20))
    if rank is not None and rank < 50.0:
        caveats.append(_failure_item("RANK_BELOW_PREFERENCE", "Deterministic Rank is below the preferred level", rank, 50.0))
    if stop_quality is not None and stop_quality < 55.0:
        caveats.append(_failure_item("MODERATE_STOP_QUALITY", "Stop geometry is valid but lower quality", stop_quality, 55.0))
    if target_quality is not None and target_quality < 55.0:
        caveats.append(_failure_item("MODERATE_TARGET_FEASIBILITY", "A valid target remains but feasibility is moderate", target_quality, 55.0))
    if row.get("market_quality_ok") is False and not any(item["code"] == "UNUSABLE_SPREAD" for item in failures):
        caveats.append(_failure_item("MODERATE_MARKET_QUALITY", "Market quality is weaker than preferred; use conservative sizing"))

    pre_delivery = row.get("pre_delivery_revalidation_ok")
    if pre_delivery is False:
        failures.append(_failure_item("FINAL_REVALIDATION_FAILED", "Final force-refreshed validation failed"))
    qualifies_before_revalidation = not failures
    qualifies = bool(qualifies_before_revalidation and pre_delivery is not False)
    tier = _quality_tier(overall or 0.0, execution or 0.0) if qualifies else None
    return {
        **identity_metadata(),
        "qualification_policy_version": LEGACY_POLICY_VERSION,
        "qualified": qualifies,
        "qualified_before_revalidation": qualifies_before_revalidation,
        "final_revalidation_observed": pre_delivery is not None,
        "quality_tier": tier,
        "overall_quality": overall,
        "execution_quality": execution,
        "net_rr": net_rr,
        "gross_rr": gross_rr,
        "thresholds": thresholds,
        "hard_failures": failures,
        "caveats": caveats,
        "qualification_reason": (
            "Passed Legacy universal correctness, score, execution, and absolute economics requirements"
            if qualifies
            else "; ".join(item["explanation"] for item in failures)
        ),
    }


def strict_shadow_decision(row: Mapping[str, Any]) -> Dict[str, Any]:
    qualification = evaluate_private_beta_qualification(
        row, row.get("gate_evaluation")
    )
    return {
        "policy_version": qualification.get("qualification_policy_version"),
        "decision": (
            "SIGNAL" if qualification.get("private_beta_qualified") else "REJECTED"
        ),
        "qualification_type": qualification.get("qualification_type"),
        "failed_hard_checks": [
            item.get("canonical_check_id")
            for item in qualification.get("failed_hard_checks") or []
        ],
        "failed_important_checks": [
            item.get("canonical_check_id")
            for item in qualification.get("failed_important_soft_checks") or []
        ],
        "overall_quality_passed": bool(qualification.get("overall_quality_passed")),
        "execution_quality_passed": bool(qualification.get("execution_quality_passed")),
        "net_rr_floor_passed": bool(qualification.get("net_rr_floor_passed")),
        "reason": (
            "Current strict private-beta policy would publish this candidate"
            if qualification.get("private_beta_qualified")
            else "Current strict private-beta policy rejected one or more qualification requirements"
        ),
    }


def qualify_legacy_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int = 2,
) -> List[Dict[str, Any]]:
    qualified: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        decision = evaluate_legacy_qualification(row)
        row["legacy_decision"] = decision
        row["strict_shadow_decision"] = strict_shadow_decision(row)
        row["quality_tier"] = decision.get("quality_tier")
        row["caveats"] = list(decision.get("caveats") or [])
        row["bot_variant"] = "legacy"
        row["qualification_policy_version"] = LEGACY_POLICY_VERSION
        if decision["qualified"]:
            qualified.append(row)
    qualified.sort(
        key=lambda row: (
            float((row.get("legacy_decision") or {}).get("overall_quality") or 0.0),
            float((row.get("legacy_decision") or {}).get("execution_quality") or 0.0),
            float((row.get("legacy_decision") or {}).get("net_rr") or 0.0),
            float(row.get("authoritative_rank") or row.get("rank_score") or 0.0),
        ),
        reverse=True,
    )
    return qualified[: max(0, int(limit))]
