# Outcome-calibrated scoring

Perpetual Pro now has a shadow-first scoring pipeline that learns from complete
alert outcomes. It does not replace the current production scorer merely because
a model can be trained.

## Immediate Legacy V2 protection

While the learned model remains in shadow, production uses an execution-aware
Legacy V2 policy. All existing Technical Score calculations remain unchanged.
The old confidence formula is retained alongside the new value for comparison,
but production confidence now follows:

```text
Legacy V2 Confidence =
    min(Technical Confidence, Execution Score + 5)
    - immediate-SL penalty
    - data-quality penalty
```

The default minimum Execution Score is 72. Consequently, an alert requiring 80%
confidence generally needs at least 75 Execution because execution may not trail
overall confidence by more than the five-point buffer. Clean execution cannot
inflate a weak Technical Score.

Every analysis-gate rejection is logged with structured reason codes and stored
with the candidate in PostgreSQL. Telegram delivery filters also log their
rejection codes.

The defaults live in `config.yaml`. Railway backend overrides are available as
`LEGACY_V2_ENABLED`, `LEGACY_V2_EXECUTION_MIN_SCORE`, and
`LEGACY_V2_CONFIDENCE_BUFFER`. Setting `LEGACY_V2_ENABLED=0` provides an
immediate rollback to the exact prior confidence formula.

## What each score means

- **Technical Score** is a calibrated estimate of the probability that the
  predicted direction reaches a one-ATR favorable barrier before a one-ATR
  adverse barrier. It describes chart direction quality, not executability.
- **Execution Score** is the estimated probability that the published entry is
  validly filled before the setup is invalidated, missed, or expires **and**
  survives the immediate one-candle stop-sweep window.
- **Confidence** is a separately calibrated estimate of the complete alert event:
  a valid fill followed by TP1 before Stop Loss. It is not a weighted average of
  Technical and Execution.
- **Expected Value** is a robust regression estimate of realized R after the
  published management rules. A one-sided residual penalty is subtracted to
  produce conservative EV.
- **Rank Score** is the 0–100 percentile of conservative EV relative to the
  model's held-out reference distribution.

Setup-type one-hot features and learned setup interactions let historical
evidence determine whether execution, trend, timing, or structure matters more
for momentum, breakout, retest, pullback, range, mean-reversion, and reversal
setups. There are no hand-written Technical-versus-Execution weights in the new
model.

## Label policy

Every directional candidate is journaled, including blocked and unfilled
candidates. Only recording delivered winners would create selection bias.

Historical replay uses the same confluence engine on closed, as-of candles. It
does not synthesize historical funding, order-book, or news data when those
features are unavailable. For ambiguous OHLCV candles, Stop/adverse movement is
assumed to happen first. This deliberately avoids optimistic backtest results.

Forward tracker outcomes take precedence as real alerts accumulate. A complete
alert is successful only when:

1. the entry fills before invalidation, missing TP1, or expiry; and
2. TP1 trades before Stop Loss.

## Validation and promotion

Training uses expanding walk-forward validation with a 27-hour embargo between
train and test observations. That covers the maximum three-hour pending-entry
window plus the maximum 24-hour hold and prevents overlapping lifecycle leakage.

The old and new systems are compared on unseen data using:

- win, TP1, TP2, and fill rates;
- invalidation and total pre-entry failure rates;
- average trade duration;
- expectancy in R, profit factor, and maximum drawdown in R;
- Brier score, log loss, and expected calibration error;
- top-quintile trade performance; and
- per-setup slices where the sample is large enough.

A shadow can be promoted only when all configured gates pass. Promotion is never
automatic. The first production stage is **veto-only**: the model may reject and
reorder an otherwise eligible setup, but it cannot resurrect a setup blocked by
the deterministic safety engine.

The default promotion policy also requires at least 200 truly unseen
walk-forward observations and expected calibration error no greater than 0.05.
These policy thresholds are centralized in `outcome_scoring` configuration.

At inference time the champion also checks standardized feature drift. A
candidate outside the training distribution is marked `model_applicable=false`
and cannot pass the production outcome gate, even if the extrapolated probability
would otherwise look high.

## Supabase setup

Supabase Storage is not required. This feature uses only the Supabase PostgreSQL
database. Model artifacts are compact JSON stored in PostgreSQL.

1. Create a Supabase project.
2. Open **SQL Editor**, paste
   [`migrations/001_outcome_scoring.sql`](../migrations/001_outcome_scoring.sql),
   and run it once.
3. Open **Connect → Session pooler** and copy the PostgreSQL connection string.
   Use the session pooler for a long-running Railway backend and keep
   `sslmode=require`. If the database password contains URL-reserved characters,
   use the already encoded connection string shown by Supabase or URL-encode the
   password.
4. Add these variables to the Railway **backend service only**:

   ```text
   DATABASE_URL=postgresql://...supabase.../postgres?sslmode=require
   OUTCOME_SCORING_ENABLED=1
   OUTCOME_SCORING_MODE=shadow
   ```

   Do not put `DATABASE_URL` in Streamlit, browser-visible variables, Git, or
   `config.yaml`.
5. Redeploy the backend, then check:

   ```bash
   curl https://YOUR-BACKEND-DOMAIN/outcome-scoring/status
   ```

   `repository.ready` should be `true` and `migration_required` should be
   `false`.

## Backfill, train, and activate

Run administrative commands from the repository with Railway's backend
environment injected:

```bash
~/.railway/bin/railway run --service backend --environment production \
  python -m src.scoring.cli status

~/.railway/bin/railway run --service backend --environment production \
  python -m src.scoring.cli backfill-replay --bars 700 --step 3

~/.railway/bin/railway run --service backend --environment production \
  python -m src.scoring.cli compare-legacy-v2 --confidence-floor 80

~/.railway/bin/railway run --service backend --environment production \
  python -m src.scoring.cli train-shadow
```

The production minimum is 500 fit candidates plus 200 calibration candidates.
Do not use `--allow-small-sample` for a production decision; that option is only
for verifying the pipeline.

The training command prints a model version and the full promotion gate. If any
gate is false, leave the model in shadow, collect more forward outcomes, diagnose
the failing market/setup slices, and retrain.

If every gate passes, promotion still requires an explicit command:

```bash
~/.railway/bin/railway run --service backend --environment production \
  python -m src.scoring.cli promote --version outcome-YYYYMMDDTHHMMSSZ-XXXXXXXX
```

Then set `OUTCOME_SCORING_MODE=production` on the backend and redeploy. The
backend reloads model state every five minutes.

## Safe rollback

Set:

```text
OUTCOME_SCORING_MODE=shadow
```

and redeploy. Candidate and outcome collection continues, but the champion stops
affecting signal confidence, eligibility, and ordering.

## Operational notes

- PostgreSQL failures are fail-open for the existing signal engine: scans and
  Telegram still work, while status/logs clearly report the durable journal
  failure.
- A promoted champion does not use LLM confidence as a numeric gate. LLM output
  remains explanation/context only.
- The existing spread, data-quality, structure, execution, R:R, prop leverage,
  and portfolio-risk gates remain in force.
- No score can make signals perfect. The purpose of this pipeline is to make
  confidence empirically honest and reject redesigns that fail on unseen trading
  outcomes.
