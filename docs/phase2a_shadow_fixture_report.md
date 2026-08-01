# Phase 2A shadow fixture comparison

This report uses six deterministic synthetic edge-case fixtures. It validates
policy behavior; it is not evidence of live profitability.

| Fixture | Setup | Side | Legacy execution | Phase 2A quality | Result |
|---|---|---:|---:|---:|---|
| BTC clean structural retest | Retest continuation | Long | 89.39 | 81.73 | Accepted; three feasible targets |
| ETH extended CMP | Breakout continuation | Long | 89.39 | 79.57 | Rejected: CMP position too extended inside zone |
| SOL clean structural retest | Retest continuation | Short | 84.21 | 81.09 | Accepted; long/short behavior remains symmetric |
| ARB mitigated-anchor mix | Retest continuation | Long | 86.39 | 80.85 | Accepted using the remaining fresh anchor |
| WIF wide spread | Retest continuation | Long | 61.39 | 74.82 | Rejected by the preserved hard spread gate |
| LINK late retest | Retest continuation | Long | 76.61 | 77.36 | Rejected as `avoid_chase` despite clearing score |

At the unchanged 72 quality floor, status and hard gates included:

- Legacy baseline: 4 of 6 fixtures eligible.
- Phase 2A: 3 of 6 fixtures eligible.
- Mean legacy execution score: 81.23.
- Mean Phase 2A Execution Quality: 79.24.
- New rejection reasons: excessive CMP position, spread hard limit, and late/avoid-chase lifecycle state.

The clean BTC fixture produced three structural targets rather than a forced
four. Gross R:R was 1.84R, 2.81R and 3.68R; conservative net R:R was 1.42R,
2.25R and 2.99R. The SOL short fixture produced the corresponding mirrored
geometry (1.84R/2.81R/3.68R gross), with slightly lower net values due to its
wider spread.

A separate strong-volume closed-candle CMP probe passed as
`confirmation_pending` (never immediate fill) at 81.97 Execution Quality. Its
component values were Entry 85.7, Zone 100.0, Pre-entry Survival 73.0,
Confirmation 74.8, Stop 100.0, Targets 73.4, Liquidity/Cost 63.8 and Data 100.0;
net target R:R values were 0.80R, 1.63R and 2.37R.

The comparison helper also groups counts and score changes by setup, asset and
direction and reports target count, stop distance, gross R:R and net R:R. It is
implemented in `src/scoring/phase2a_comparison.py` for use with future stored
schema 3.0 candidate/outcome rows.
