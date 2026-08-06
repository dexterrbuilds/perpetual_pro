-- Explicit lifecycle profitability and per-level outcome detail.
-- Additive only; apply after 002_durable_lifecycle.sql.

alter table public.tracked_signals
  add column if not exists outcome_classification text,
  add column if not exists profitable boolean,
  add column if not exists profitable_at timestamptz,
  add column if not exists level_hits jsonb not null default '{}'::jsonb;

-- Deterministic compatibility backfill from fields already recorded by the
-- tracker.  Historical TP1 timestamps remain NULL when the exact event time is
-- unavailable; no timestamp is fabricated from terminal time.
update public.tracked_signals
set profitable = (entered_at is not null and highest_tp >= 1),
    outcome_classification = case
      when entered_at is not null and highest_tp >= 1 then 'profitable'
      when status = 'ambiguous_gap' then 'ambiguous'
      when status = 'pending' then 'pending_entry'
      when status = 'entered' then 'active'
      when entered_at is null then 'not_entered'
      else 'not_profitable'
    end;

alter table public.tracked_signals
  alter column outcome_classification set default 'pending_entry',
  alter column outcome_classification set not null,
  alter column profitable set default false,
  alter column profitable set not null;

do $$ begin
  alter table public.tracked_signals
    add constraint tracked_signals_outcome_classification_check
    check (outcome_classification in (
      'pending_entry','active','profitable','not_profitable',
      'not_entered','ambiguous'
    )) not valid;
exception when duplicate_object then null;
end $$;
alter table public.tracked_signals
  validate constraint tracked_signals_outcome_classification_check;

create index if not exists tracked_signals_profitable_idx
  on public.tracked_signals (profitable, profitable_at desc);

alter table public.signal_outcomes
  add column if not exists tp3_hit boolean not null default false,
  add column if not exists tp4_hit boolean not null default false,
  add column if not exists outcome_classification text;

-- TP3/TP4 were previously retained in outcome metadata as highest_tp.  Restore
-- those explicit flags where the information exists, then classify TP1 wins.
update public.signal_outcomes
set tp3_hit = coalesce((metadata->>'highest_tp')::integer >= 3, false),
    tp4_hit = coalesce((metadata->>'highest_tp')::integer >= 4, false),
    outcome_classification = case
      when valid_fill and tp1_hit then 'profitable'
      when terminal_status = 'ambiguous_gap'
        or ambiguity_policy = 'sparse_gap_unknown' then 'ambiguous'
      when not valid_fill then 'not_entered'
      else 'not_profitable'
    end;

alter table public.signal_outcomes
  alter column outcome_classification set default 'not_profitable',
  alter column outcome_classification set not null;

do $$ begin
  alter table public.signal_outcomes
    add constraint signal_outcomes_classification_check
    check (outcome_classification in (
      'profitable','not_profitable','not_entered','ambiguous'
    )) not valid;
exception when duplicate_object then null;
end $$;
alter table public.signal_outcomes
  validate constraint signal_outcomes_classification_check;

create index if not exists signal_outcomes_profitability_idx
  on public.signal_outcomes (outcome_classification, terminal_at desc);

-- Rollback is code-only. Retain these additive columns and audit data.
