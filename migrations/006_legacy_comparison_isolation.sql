-- Perpetual Pro Legacy comparison operational isolation.
-- Additive only. Apply after 005_lifecycle_outcome_detail.sql.
--
-- Candidate and outcome rows remain in public with namespaced IDs so A/B
-- reports can join them. Active lifecycle, event, delivery, and scheduler
-- state live here so the preserved strict build cannot recover Legacy work.

create schema if not exists legacy_comparison;

create table if not exists legacy_comparison.tracked_signals
  (like public.tracked_signals including all);
create table if not exists legacy_comparison.signal_lifecycle_events
  (like public.signal_lifecycle_events including all);
create table if not exists legacy_comparison.telegram_notification_ledger
  (like public.telegram_notification_ledger including all);
create table if not exists legacy_comparison.scheduler_runs
  (like public.scheduler_runs including all);
create table if not exists legacy_comparison.scheduler_run_deliveries
  (like public.scheduler_run_deliveries including all);

create sequence if not exists legacy_comparison.telegram_notification_ledger_id_seq;
alter sequence legacy_comparison.telegram_notification_ledger_id_seq
  owned by legacy_comparison.telegram_notification_ledger.id;
alter table legacy_comparison.telegram_notification_ledger
  alter column id set default nextval(
    'legacy_comparison.telegram_notification_ledger_id_seq'::regclass
  );

create sequence if not exists legacy_comparison.scheduler_run_deliveries_id_seq;
alter sequence legacy_comparison.scheduler_run_deliveries_id_seq
  owned by legacy_comparison.scheduler_run_deliveries.id;
alter table legacy_comparison.scheduler_run_deliveries
  alter column id set default nextval(
    'legacy_comparison.scheduler_run_deliveries_id_seq'::regclass
  );

-- LIKE does not copy foreign keys. Add schema-local lifecycle integrity while
-- keeping public candidates/outcomes independently queryable for comparison.
do $$ begin
  alter table legacy_comparison.signal_lifecycle_events
    add constraint legacy_events_signal_fk
    foreign key (signal_id)
    references legacy_comparison.tracked_signals(signal_id) on delete cascade;
exception when duplicate_object then null;
end $$;

do $$ begin
  alter table legacy_comparison.telegram_notification_ledger
    add constraint legacy_ledger_event_fk
    foreign key (event_id)
    references legacy_comparison.signal_lifecycle_events(event_id) on delete cascade;
exception when duplicate_object then null;
end $$;

do $$ begin
  alter table legacy_comparison.telegram_notification_ledger
    add constraint legacy_ledger_signal_fk
    foreign key (signal_id)
    references legacy_comparison.tracked_signals(signal_id) on delete cascade;
exception when duplicate_object then null;
end $$;

do $$ begin
  alter table legacy_comparison.scheduler_run_deliveries
    add constraint legacy_delivery_run_fk
    foreign key (run_id)
    references legacy_comparison.scheduler_runs(run_id) on delete cascade;
exception when duplicate_object then null;
end $$;

do $$ begin
  if not exists (
    select 1 from pg_trigger
    where tgname = 'legacy_tracked_signals_transition_guard'
      and tgrelid = 'legacy_comparison.tracked_signals'::regclass
      and not tgisinternal
  ) then
    create trigger legacy_tracked_signals_transition_guard
    before update of status on legacy_comparison.tracked_signals
    for each row execute function public.enforce_signal_lifecycle_transition();
  end if;
end $$;

alter table legacy_comparison.tracked_signals enable row level security;
alter table legacy_comparison.signal_lifecycle_events enable row level security;
alter table legacy_comparison.telegram_notification_ledger enable row level security;
alter table legacy_comparison.scheduler_runs enable row level security;
alter table legacy_comparison.scheduler_run_deliveries enable row level security;

revoke all on schema legacy_comparison from anon, authenticated;
revoke all on all tables in schema legacy_comparison from anon, authenticated;
revoke all on all sequences in schema legacy_comparison from anon, authenticated;

-- Rollback is code-only: stop the Legacy service and resume Strict. Retain the
-- schema as immutable experiment history; no Strict table is changed.
