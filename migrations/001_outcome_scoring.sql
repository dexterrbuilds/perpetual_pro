-- Durable candidate, outcome, and model registry for Perpetual Pro.
-- Run once in Supabase SQL Editor before setting DATABASE_URL on Railway.

create table if not exists public.signal_candidates (
    id text primary key,
    generated_at timestamptz not null,
    symbol text not null,
    exchange_id text not null,
    timeframe text not null,
    direction text not null,
    setup_type text not null,
    setup_name text not null default '',
    feature_schema_version text not null,
    features jsonb not null,
    decision jsonb not null,
    production_scores jsonb not null,
    production_eligible boolean not null default false,
    production_rank double precision,
    shadow_model_version text,
    shadow_scores jsonb,
    delivered boolean not null default false,
    source text not null default 'watchlist',
    created_at timestamptz not null default now()
);

create index if not exists signal_candidates_generated_idx
    on public.signal_candidates (generated_at desc);
create index if not exists signal_candidates_symbol_idx
    on public.signal_candidates (symbol, generated_at desc);
create index if not exists signal_candidates_setup_idx
    on public.signal_candidates (setup_type, generated_at desc);
create index if not exists signal_candidates_unlabeled_idx
    on public.signal_candidates (generated_at)
    where delivered = false;

create table if not exists public.tracked_signals (
    signal_id text primary key,
    candidate_id text references public.signal_candidates(id) on delete set null,
    symbol text not null,
    exchange_id text not null,
    direction text not null,
    timeframe text not null,
    source text not null,
    status text not null,
    generated_at timestamptz not null,
    valid_until timestamptz not null,
    entered_at timestamptz,
    terminal_at timestamptz,
    entry_low double precision not null,
    entry_high double precision not null,
    entry_mid double precision not null,
    entry_price double precision,
    stop_loss double precision not null,
    take_profits jsonb not null,
    highest_tp integer not null default 0,
    realized_r double precision not null default 0,
    mfe_r double precision not null default 0,
    mae_r double precision not null default 0,
    slippage_bps double precision,
    terminal_reason text,
    signal_payload jsonb not null,
    updated_at timestamptz not null default now()
);

create index if not exists tracked_signals_candidate_idx
    on public.tracked_signals (candidate_id);
create index if not exists tracked_signals_status_idx
    on public.tracked_signals (status, updated_at desc);

create table if not exists public.signal_outcomes (
    candidate_id text primary key
        references public.signal_candidates(id) on delete cascade,
    signal_id text references public.tracked_signals(signal_id) on delete set null,
    label_source text not null,
    valid_fill boolean not null,
    technical_success boolean,
    alert_success boolean not null,
    tp1_hit boolean not null,
    tp2_hit boolean not null default false,
    invalidated_before_fill boolean not null default false,
    missed_before_fill boolean not null default false,
    expired_before_fill boolean not null default false,
    entry_delay_minutes double precision,
    trade_duration_minutes double precision,
    realized_r double precision not null default 0,
    mfe_r double precision not null default 0,
    mae_r double precision not null default 0,
    slippage_bps double precision,
    terminal_status text not null,
    terminal_at timestamptz not null,
    ambiguity_policy text not null default 'stop_first',
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists signal_outcomes_terminal_idx
    on public.signal_outcomes (terminal_at desc);

create table if not exists public.scoring_model_versions (
    version text primary key,
    stage text not null check (stage in ('candidate', 'shadow', 'champion', 'retired')),
    feature_schema_version text not null,
    trained_at timestamptz not null,
    training_start timestamptz,
    training_end timestamptz,
    training_samples integer not null,
    calibration_samples integer not null,
    artifact jsonb not null,
    metrics jsonb not null,
    validation jsonb not null,
    promoted_at timestamptz,
    created_at timestamptz not null default now()
);

create unique index if not exists one_scoring_champion_idx
    on public.scoring_model_versions (stage)
    where stage = 'champion';
create index if not exists scoring_models_stage_idx
    on public.scoring_model_versions (stage, trained_at desc);

create table if not exists public.scoring_validation_runs (
    id bigserial primary key,
    model_version text not null
        references public.scoring_model_versions(version) on delete cascade,
    fold integer not null,
    train_start timestamptz,
    train_end timestamptz,
    test_start timestamptz,
    test_end timestamptz,
    sample_count integer not null,
    metrics jsonb not null,
    regime_metrics jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (model_version, fold)
);

-- These tables are backend-only. The Railway database role is granted access
-- through the Postgres connection string; browser-facing anon/publishable roles
-- receive no table privileges.
alter table public.signal_candidates enable row level security;
alter table public.tracked_signals enable row level security;
alter table public.signal_outcomes enable row level security;
alter table public.scoring_model_versions enable row level security;
alter table public.scoring_validation_runs enable row level security;

revoke all on public.signal_candidates from anon, authenticated;
revoke all on public.tracked_signals from anon, authenticated;
revoke all on public.signal_outcomes from anon, authenticated;
revoke all on public.scoring_model_versions from anon, authenticated;
revoke all on public.scoring_validation_runs from anon, authenticated;
revoke all on sequence public.scoring_validation_runs_id_seq from anon, authenticated;
