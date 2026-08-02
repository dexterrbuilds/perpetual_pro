# Rejection analytics policy 1.0

Rejection analytics is an observability layer. It mirrors the production
analysis, alert-filter, and pre-delivery gates; it never changes a score,
threshold, trade plan, ranking, or eligibility decision.

## Evaluation order

The first failed authoritative hard gate in actual evaluation order is the
`primary_rejection_reason`. Every deterministic check whose inputs are already
available is retained in `all_rejection_reasons`. Non-authoritative diagnostic
checks are marked with `authoritative=false`; they cannot reject a candidate.
Pre-delivery checks append to the existing gate record and never promote an
upstream reject.

## Diagnostic distance

For numeric minimum gates, distance is `max(required - actual, 0)`. For numeric
maximum gates, it is `max(actual - required, 0)`. Normalized distance divides
that value by `max(abs(required), 1)`. `distance_to_eligibility` is the sum of
normalized distances for failed authoritative hard gates. It is not a trading
score or probability.

Labels are deterministic:

- `NEAR_PASS`: at most one failed hard gate and normalized distance at most 0.10.
- `MODERATE_GAP`: at most two failed hard gates and distance at most 0.40.
- `FAR_FROM_ELIGIBLE`: all other directional rejects.
- `NON_DIRECTIONAL`: flat candidates.
- `DATA_OR_MARKET_FAILURE`: data, freshness, analysis, or market availability failures.

## Persistence and access

Migration `004_rejection_analytics.sql` adds scan and candidate analytics
tables. Writes are batched per scan and are non-authoritative: a persistence
failure alerts the operator but cannot block an otherwise valid signal.

Protected endpoints:

- `GET /admin/rejections/latest`
- `GET /admin/rejections/summary?hours=24`
- `GET /admin/rejections/candidates/{candidate_id}`

They use the same private scan/admin authentication and rate limiting. The
authorized Telegram commands `/rejections`, `/rejections 24h`, and
`/rejections 72h` reply only to the requesting operator chat.

Rollback is code-only. The additive tables should remain for audit continuity.
