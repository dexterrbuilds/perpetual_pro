-- Persistent, non-authoritative gate and rejection analytics.
-- Additive only. Apply after 003_scheduler_runs.sql.

create table if not exists public.scan_rejection_analytics (
  scan_id text primary key,
  trigger_type text not null check (trigger_type in ('scheduled','manual','telegram','api')),
  started_at timestamptz not null,
  completed_at timestamptz not null,
  requested_symbols jsonb not null default '[]'::jsonb,
  analyzed_symbols jsonb not null default '[]'::jsonb,
  failed_symbols jsonb not null default '[]'::jsonb,
  directional_candidates integer not null default 0,
  eligible_candidates integer not null default 0,
  revalidated_candidates integer not null default 0,
  scan_duration_seconds double precision,
  analytics_latency_ms double precision,
  llm_calls integer not null default 0,
  llm_rate_limit_events integer not null default 0,
  public_messages integer not null default 0,
  private_messages integer not null default 0,
  no_quality_result boolean not null default false,
  status text not null check (status in ('completed','failed','partial')),
  gate_policy_version text not null,
  feature_schema_version text not null,
  execution_policy_version text not null,
  rank_policy_version text not null,
  build_commit_sha text,
  summary jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists scan_rejection_started_idx
  on public.scan_rejection_analytics (started_at desc);
create index if not exists scan_rejection_trigger_idx
  on public.scan_rejection_analytics (trigger_type, started_at desc);

create table if not exists public.candidate_rejection_analytics (
  candidate_id text primary key,
  scan_id text not null references public.scan_rejection_analytics(scan_id) on delete cascade,
  analyzed_at timestamptz not null,
  symbol text not null,
  exchange_id text,
  timeframe text not null,
  direction text not null check (direction in ('long','short','flat')),
  setup_type text,
  feature_schema_version text not null,
  execution_policy_version text,
  rank_policy_version text,
  gate_policy_version text not null,
  technical_quality double precision,
  execution_quality double precision,
  overall_quality double precision,
  rank_score double precision,
  immediate_sl_risk double precision,
  gross_rr jsonb not null default '[]'::jsonb,
  net_rr jsonb not null default '[]'::jsonb,
  spread_bps double precision,
  ticker_age_seconds double precision,
  orderbook_age_seconds double precision,
  prop_safe boolean,
  entry_state text,
  target_count integer not null default 0,
  target_feasibility jsonb not null default '[]'::jsonb,
  target_feasibility_summary double precision,
  stop_quality double precision,
  data_quality_score double precision,
  market_quality_ok boolean,
  chase_distance_atr double precision,
  confluence_magnitude double precision,
  lifecycle_state text not null default 'not_published',
  eligible boolean not null default false,
  primary_rejection_reason text,
  all_rejection_reasons text[] not null default '{}',
  closest_to_passing_gate text,
  distance_to_eligibility double precision not null default 0,
  proximity_label text not null,
  gate_evaluation jsonb not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists candidate_rejection_scan_idx
  on public.candidate_rejection_analytics (scan_id, distance_to_eligibility);
create index if not exists candidate_rejection_time_idx
  on public.candidate_rejection_analytics (analyzed_at desc);
create index if not exists candidate_rejection_primary_idx
  on public.candidate_rejection_analytics (primary_rejection_reason, analyzed_at desc);
create index if not exists candidate_rejection_symbol_idx
  on public.candidate_rejection_analytics (symbol, analyzed_at desc);
create index if not exists candidate_rejection_setup_idx
  on public.candidate_rejection_analytics (setup_type, analyzed_at desc);
create index if not exists candidate_rejection_codes_gin_idx
  on public.candidate_rejection_analytics using gin (all_rejection_reasons);

alter table public.scan_rejection_analytics enable row level security;
alter table public.candidate_rejection_analytics enable row level security;
revoke all on public.scan_rejection_analytics from anon, authenticated;
revoke all on public.candidate_rejection_analytics from anon, authenticated;

-- Rollback is code-only. Keep additive analytics tables for audit continuity;
-- no trading, lifecycle, candidate, or outcome record is changed or removed.
