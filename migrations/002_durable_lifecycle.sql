-- Perpetual Pro durable lifecycle, notification idempotency, and training isolation.
-- Additive only. Apply after 001_outcome_scoring.sql.

alter table public.signal_candidates
  add column if not exists is_directional_candidate boolean not null default false;

alter table public.tracked_signals
  add column if not exists fingerprint text,
  add column if not exists setup_type text not null default 'unknown',
  add column if not exists entry_mode text not null default 'retest',
  add column if not exists confirmation_pending boolean not null default false,
  add column if not exists hold_until timestamptz,
  add column if not exists target_allocations jsonb not null default '[]'::jsonb,
  add column if not exists remaining_size double precision not null default 1.0,
  add column if not exists protected boolean not null default false,
  add column if not exists last_price double precision,
  add column if not exists last_price_at timestamptz,
  add column if not exists previous_price double precision,
  add column if not exists previous_price_at timestamptz,
  add column if not exists last_processed_candle_at timestamptz,
  add column if not exists ordering_policy text not null default 'observed_segment_v1',
  add column if not exists ambiguous_gap boolean not null default false,
  add column if not exists technical_success boolean,
  add column if not exists lifecycle_version bigint not null default 0,
  add column if not exists lifecycle_schema_version text not null default 'legacy',
  add column if not exists feature_schema_version text,
  add column if not exists execution_policy_version text,
  add column if not exists rank_policy_version text,
  add column if not exists lifecycle_state jsonb not null default '{}'::jsonb;

create unique index if not exists tracked_signals_fingerprint_active_idx
  on public.tracked_signals (fingerprint)
  where status in ('pending', 'entered') and fingerprint is not null;
create index if not exists tracked_signals_recovery_idx
  on public.tracked_signals (status, lifecycle_schema_version, generated_at);

do $$ begin
  alter table public.tracked_signals add constraint tracked_signals_status_check
    check (status in (
      'pending','entered','completed','stopped','missed','expired',
      'invalidated','time_exit','ambiguous_gap'
    )) not valid;
exception when duplicate_object then null;
end $$;
alter table public.tracked_signals validate constraint tracked_signals_status_check;

create or replace function public.enforce_signal_lifecycle_transition()
returns trigger language plpgsql as $$
begin
  if old.status = new.status then return new; end if;
  if old.status = 'pending' and new.status in (
      'entered','missed','expired','invalidated','ambiguous_gap'
    ) then return new; end if;
  if old.status = 'entered' and new.status in (
      'completed','stopped','time_exit','ambiguous_gap'
    ) then return new; end if;
  raise exception 'invalid lifecycle transition % -> %', old.status, new.status
    using errcode = '23514';
end $$;

do $$ begin
  if not exists (
    select 1 from pg_trigger
    where tgname = 'tracked_signals_transition_guard' and not tgisinternal
  ) then
    create trigger tracked_signals_transition_guard
    before update of status on public.tracked_signals
    for each row execute function public.enforce_signal_lifecycle_transition();
  end if;
end $$;

create table if not exists public.signal_lifecycle_events (
  event_id text primary key,
  signal_id text not null references public.tracked_signals(signal_id) on delete cascade,
  lifecycle_version bigint not null,
  event_type text not null,
  from_status text,
  to_status text not null,
  occurred_at timestamptz not null,
  price double precision,
  payload jsonb not null default '{}'::jsonb,
  notification_required boolean not null default true,
  created_at timestamptz not null default now(),
  unique (signal_id, lifecycle_version)
);

create index if not exists lifecycle_events_signal_idx
  on public.signal_lifecycle_events (signal_id, lifecycle_version);

create table if not exists public.telegram_notification_ledger (
  id bigserial primary key,
  idempotency_key text not null unique,
  event_id text not null references public.signal_lifecycle_events(event_id) on delete cascade,
  signal_id text not null references public.tracked_signals(signal_id) on delete cascade,
  destination_hash text not null,
  status text not null default 'queued'
    check (status in ('queued','sending','retry','delivered','dead_letter')),
  telegram_message_id bigint,
  attempt_count integer not null default 0,
  last_attempt_at timestamptz,
  next_retry_at timestamptz not null default now(),
  delivered_at timestamptz,
  error_category text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (event_id, destination_hash)
);

create index if not exists telegram_ledger_due_idx
  on public.telegram_notification_ledger (status, next_retry_at)
  where status in ('queued','sending','retry');
create index if not exists telegram_ledger_signal_idx
  on public.telegram_notification_ledger (signal_id, created_at);

alter table public.signal_lifecycle_events enable row level security;
alter table public.telegram_notification_ledger enable row level security;
revoke all on public.signal_lifecycle_events from anon, authenticated;
revoke all on public.telegram_notification_ledger from anon, authenticated;
revoke all on sequence public.telegram_notification_ledger_id_seq from anon, authenticated;

-- Rollback is code-only: deploy the prior application.  Retain these additive
-- columns/tables so lifecycle and audit history are never destroyed.
