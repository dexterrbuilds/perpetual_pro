# Perpetual Pro Legacy comparison

The Legacy service is an isolated comparison deployment built from the current
corrected engine. It changes publication selectivity, not market data,
scoring, trade geometry, lifecycle, or outcome definitions.

## Policy

A Legacy signal requires a valid LONG/SHORT direction, valid and fresh market
data, coherent live entry/stop/target geometry, no expiry or completed move,
Overall Quality at least 80, Execution Quality at least 65, Net R:R at least
0.75, and successful force-refreshed pre-delivery revalidation. Later
selectivity preferences are retained as visible caveats.

The current Strict private-beta policy is evaluated in shadow against the same
candidate. It cannot deliver, register a lifecycle, or influence Legacy.

## Isolation

- Strict remains on `main`; Legacy runs from `codex/legacy-comparison`.
- Start Legacy with `uvicorn legacy_server:app --host 0.0.0.0 --port $PORT`.
- Legacy Telegram credentials are mandatory and separate.
- Candidate/outcome IDs use `legacy_cand_` / `legacy_sig_` prefixes.
- Operational lifecycle, event, notification, and scheduler tables live in the
  additive `legacy_comparison` Supabase schema created by migration 006.
- Legacy candidates are excluded from Strict directional training datasets.

## Safe switch back to Strict

1. Set Legacy `SCHEDULER_ENABLED=0` and `TELEGRAM_ENABLED=0`.
2. Stop the Legacy deployment after its scan lock is idle.
3. Redeploy the preserved Strict commit to the existing `backend` service with
   delivery and scheduling still disabled.
4. Verify readiness and durable Strict lifecycle recovery.
5. Set Strict `TELEGRAM_ENABLED=1` and `SCHEDULER_ENABLED=1`.
6. Run one private Strict scan and verify only one scheduler instance.

No code or database rollback is required. Legacy experiment tables remain as
non-destructive audit history.
