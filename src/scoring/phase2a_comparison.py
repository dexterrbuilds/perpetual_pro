"""Pure shadow-comparison helpers for Legacy execution vs Phase 2A quality."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping


def build_phase2a_shadow_report(
    candidates: Iterable[Mapping[str, Any]],
    *,
    minimum_quality: float = 72.0,
) -> Dict[str, Any]:
    rows = [dict(row) for row in candidates]

    def score(row: Mapping[str, Any], key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    eligible_statuses = {"confirmation_pending", "wait_retest"}
    old_pass = [
        row for row in rows
        if score(row, "legacy_execution_score") >= minimum_quality
        and str(
            row.get("legacy_status")
            or row.get("legacy_execution_status")
            or row.get("status")
            or "wait_retest"
        ) in eligible_statuses
    ]
    new_pass = [
        row for row in rows
        if score(row, "execution_quality") >= minimum_quality
        and not list(row.get("hard_failures") or [])
        and str(row.get("status") or "wait_retest") in eligible_statuses
    ]
    rejection_counts: Counter[str] = Counter()
    for row in rows:
        if row not in new_pass:
            reasons = list(row.get("hard_failures") or row.get("rejection_reasons") or ["quality_below_minimum"])
            for reason in reasons:
                if isinstance(reason, Mapping):
                    reason = reason.get("code") or reason.get("detail") or "unknown"
                rejection_counts[str(reason)] += 1

    def grouped(key: str) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(key) or "unknown")].append(row)
        return {
            name: {
                "count": len(items),
                "old_pass": sum(item in old_pass for item in items),
                "new_pass": sum(item in new_pass for item in items),
                "old_mean": round(mean(score(item, "legacy_execution_score") for item in items), 3),
                "new_mean": round(mean(score(item, "execution_quality") for item in items), 3),
            }
            for name, items in sorted(groups.items())
        }

    ranking_changes = sorted(
        [
            {
                "symbol": row.get("symbol"),
                "setup_type": row.get("setup_type"),
                "direction": row.get("direction"),
                "old_execution": score(row, "legacy_execution_score"),
                "new_execution": score(row, "execution_quality"),
                "change": round(score(row, "execution_quality") - score(row, "legacy_execution_score"), 3),
                "hard_failures": list(row.get("hard_failures") or []),
            }
            for row in rows
        ],
        key=lambda item: abs(item["change"]),
        reverse=True,
    )
    return {
        "candidate_count": len(rows),
        "old_signal_count": len(old_pass),
        "new_signal_count": len(new_pass),
        "rejection_reasons": dict(rejection_counts.most_common()),
        "score_distribution": {
            "legacy_mean": round(mean([score(row, "legacy_execution_score") for row in rows]) if rows else 0.0, 3),
            "phase2a_mean": round(mean([score(row, "execution_quality") for row in rows]) if rows else 0.0, 3),
        },
        "by_setup_type": grouped("setup_type"),
        "by_asset": grouped("symbol"),
        "by_direction": grouped("direction"),
        "ranking_changes": ranking_changes,
        "geometry_changes": [
            {
                "symbol": row.get("symbol"),
                "stop_distance_atr": row.get("stop_distance_atr"),
                "target_count": len(list(row.get("targets") or row.get("take_profits") or [])),
                "gross_rr": list(row.get("gross_risk_reward") or []),
                "net_rr": list(row.get("net_risk_reward") or []),
            }
            for row in rows
        ],
    }
