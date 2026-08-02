-- Durable scheduled-run and redacted per-destination delivery audit.
-- Additive only. Apply after 002_durable_lifecycle.sql.

create table if not exists public.scheduler_runs (
  run_id text primary key,
  source text not null check (source in ('scheduled','manual','telegram')),
  slot_label text not null,
  scheduled_for timestamptz,
  started_at timestamptz not null default now(),
  completed_at timestamptz,
  status text not null default 'running'
    check (status in ('running','completed','failed','skipped')),
  symbols_requested integer not null default 0,
  symbols_analyzed integer not null default 0,
  symbol_failures integer not null default 0,
  eligible_candidates integer not null default 0,
  revalidated_candidates integer not null default 0,
  rejected_candidates integer not null default 0,
  scan_duration_seconds double precision,
  max_ticker_age_seconds double precision,
  max_orderbook_age_seconds double precision,
  result_code text,
  result_summary jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists scheduler_runs_started_idx
  on public.scheduler_runs (started_at desc);
create index if not exists scheduler_runs_status_idx
  on public.scheduler_runs (status, scheduled_for);

create table if not exists public.scheduler_run_deliveries (
  id bigserial primary key,
  idempotency_key text not null unique,
  run_id text not null references public.scheduler_runs(run_id) on delete cascade,
  destination_hash text not null,
  delivery_type text not null,
  status text not null check (status in ('delivered','failed')),
  telegram_message_id bigint,
  attempted_at timestamptz not null default now(),
  delivered_at timestamptz,
  error_category text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (run_id, destination_hash, delivery_type)
);

create index if not exists scheduler_run_deliveries_run_idx
  on public.scheduler_run_deliveries (run_id, status);

alter table public.scheduler_runs enable row level security;
alter table public.scheduler_run_deliveries enable row level security;
revoke all on public.scheduler_runs from anon, authenticated;
revoke all on public.scheduler_run_deliveries from anon, authenticated;
revoke all on sequence public.scheduler_run_deliveries_id_seq from anon, authenticated;

-- Rollback is code-only: deploy the previous application. Keep these additive
-- audit tables so completed run and delivery history remains traceable.
