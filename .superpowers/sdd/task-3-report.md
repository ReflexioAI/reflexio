# Task 3 Report — LineageGCScheduler + Lifespan Wiring

## Status
DONE

## Files changed
- NEW: `reflexio/server/services/lineage/gc_scheduler.py`
- NEW: `tests/server/services/lineage/test_gc_scheduler.py`
- MOD: `reflexio/server/api.py` (lifespan wiring)

## Deviations from spec
- `_discover_org_ids` uses `storage.list_org_ids()` (a generic cross-org sweep), falling back to bootstrap-only on `NotImplementedError`/`AttributeError`. The resume scheduler uses `list_resumable_work_org_ids` which is work-filtered. For GC, an unfiltered list is correct because we want to sweep all orgs regardless of whether they have pending work.
- Added `if ctx.storage is None: continue` guard in `_gc_tick` — pyright requires it because `create_storage()` returns `BaseStorage | None`. This is a correct, minimal guard.

## Tests
8 unit tests, all green. No threads used in tests — `_gc_tick` tested directly.
