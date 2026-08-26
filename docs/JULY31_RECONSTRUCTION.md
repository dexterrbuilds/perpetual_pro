# July 31, 2026 reconstruction

This branch is a best-effort reconstruction of Railway deployment
`9e75bb83-8b8e-4bd3-8d59-3c5429cb1727`.

The deployment was a Railway CLI upload from an uncommitted working tree, so
Git cannot reproduce its container image byte-for-byte. The branch therefore
starts from the last preceding commit, `aeb724f69de3b6e68eb4f1779b171a8863472222`,
and carries only the independently identifiable changes requested before the
August 1 Phase 1/Phase 2A release:

- the 22-symbol July production watchlist;
- CMP publication remains pending until a later closed candle confirms entry;
- sparse entry/TP crossings use the prior observation and become
  `ambiguous_gap` when order cannot be established;
- TP1 after a valid entry secures a profitable outcome and prevents a later
  original-stop alert;
- Telegram lifecycle labels reflect those states.

It intentionally excludes the August 1 Phase 1/Phase 2A scoring, execution
policy, outcome-model, and pre-delivery revalidation expansion.

