# Row-count retention archives doomed rows to JSONL before deleting

Reflexio caps hot storage tables by row count (`retention.py`; deployments may
lower caps aggressively — claude-smart runs `interactions` at 500 rows), so raw
interaction evidence was silently destroyed within days, making longitudinal
evaluation (recall rate, proven memory wins, told-it-twice) impossible to
compute. We decided that the row-count retention path appends every doomed row
— including cascade-deleted dependent rows — as JSON lines to
`<data_dir>/archive/<table>.jsonl` (embedding columns dropped) before issuing
the DELETE, gated by an env flag that is **off by default**.

## Considered options

- **Client-side daily export script** (claude-smart reads the SQLite DB on a
  timer): no OSS change needed, but leaves a loss window between exports and
  every consumer must rebuild it. Rejected in favor of fixing it where the
  deletion happens.
- **Raising the row caps**: stops the bleeding but grows the hot table
  unboundedly (embeddings dominate at ~15KB/row) and produces no join-ready
  eval records. Kept only as a local stopgap until this change ships.
- **Separate archive SQLite DB**: SQL-queryable but requires schema mirroring
  and migrations for the same information.

## Consequences

- The archive hooks **only** the row-count retention path. Governance purge
  (right-to-erasure) uses its own deletion path and never archives; however,
  rows archived *before* a later purge request survive in archive files. That
  is why the flag defaults to off: enabling the archive is an explicit
  operator decision that the deployment's erasure guarantees do not extend to
  archive files.
- Completeness invariant for consumers: live table ∪ archive = full history
  (per table, since the flag was enabled).
- Archive files grow without bound while enabled (~7KB per interaction row
  without embeddings); the operator owns rotation/disposal.
