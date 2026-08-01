# Phase 2A execution and trade-plan policy

Policy: `execution_quality_v2a.1`
Rank policy: `deterministic_rank_v2a.1`

These scores are deterministic quality ratings. They are not probabilities of
fill, TP1, or profit.

## Preserved baseline

The pre-Phase-2A execution formula is retained as `legacy_execution_score` for
shadow comparison. Legacy V2 still caps overall quality at `Execution Quality +
5`, then applies the existing immediate-SL and data-quality penalties. The +5
buffer and the 72 alert floor are unchanged.

## Execution Quality

Every component is bounded to 0–100. The final score is the normalized weighted
sum. Hard failures reject the candidate regardless of its weighted score.

| Component | Base weight | Purpose |
|---|---:|---|
| Entry Accessibility | 18% | Distance, zone width, spread precision, expiry, velocity and TP1 progress |
| Entry-Zone Quality | 10% | Structural cluster strength, age, touches and mitigation |
| Pre-Entry Survival | 16% | Room to invalidation, BOS/CHoCH, candle risk, expiry and zone health |
| Confirmation Quality | 10% | Closed-candle, signed-volume, book and structure agreement |
| Stop Quality | 14% | ATR noise, wick/sweep risk, friction, structural quality and excessive width |
| Target Feasibility | 14% | Structure and time/ATR reachability, led by TP1 |
| Liquidity and Cost Quality | 10% | Spread, quote-notional depth, impact and estimated total costs |
| Data Freshness and Market Quality | 8% | Ticker/order-book ages and source completeness |

Setup-specific weight maps are defined in
`src/analysis/execution_policy.py`. Retests emphasize accessibility and
pre-entry survival; breakouts emphasize confirmation and targets; ranges
emphasize zone location; reversals require stronger confirmation.

## Costs and R:R

`gross_RR_i = abs(TP_i - entry) / abs(entry - stop)`

`total_cost_bps = spread + 2*fee_per_side + 2*slippage_per_side + impact + funding`

`cost_price = entry * total_cost_bps / 10,000`

`net_RR_i = max(0, abs(TP_i-entry)-cost_price) / (abs(entry-stop)+cost_price)`

The conservative fallback inputs are configurable in `config.yaml`. Target
feasibility never receives a bonus merely because a constructed target clears
minimum R:R. Targets are selected from live structure within documented ATR and
hold-time bounds. A volatility projection is allowed only for TP1 when no real
level is available; it is never moved to manufacture the minimum-R gate.

## Immediate-SL risk index

`0.38*(100-Stop Quality) + 0.27*(100-Confirmation Quality) +
0.20*(100-Pre-Entry Survival) + 0.15*(100-Liquidity/Cost Quality)`, bounded
to 0–100. This remains a deterministic risk index, not a probability.

## Interim deterministic rank

The rank is a quality ordering, not expected value:

`0.25*LegacyV2Overall + 0.25*ExecutionQuality + 0.13*TargetFeasibility +
0.12*StopQuality + 0.10*NetRRQuality + 0.08*MarketDataQuality +
0.07*SetupValidity - UncertaintyPenalty`

`NetRRQuality = clamp((net_RR - 0.5) / 2.0 * 100, 0, 100)`.

Raw confluence is intentionally absent because its evidence is already present
in Technical/Legacy V2 quality.

## Hard gates

The existing spread, freshness, minimum quality, immediate-SL, chase,
data-quality, prop-safety, confidence and minimum-R gates remain. Phase 2A also
blocks stale source data, no feasible target, stop beyond the maximum structural
distance, a nearby obstacle that blocks TP1, TP1 reward below the cost buffer,
range logic during trend expansion, unconfirmed reversals, unacceptable CMP
zone position, and adverse structure change before CMP confirmation.

Missing required data rejects. Missing important execution data receives a
strong component penalty and explicit uncertainty. Missing optional venue
metadata receives a mild rank uncertainty penalty.

## Quick backtest isolation

The lightweight backtest is labeled
`diagnostic_proxy_not_production_engine`. Its score and edge flag cannot modify
rank or live alert eligibility.
