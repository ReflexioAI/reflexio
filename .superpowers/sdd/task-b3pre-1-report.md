# Task B3-pre T1 Report: Dedup Soft-Delete + Set-Based Lineage

## STATUS: COMPLETE

## Summary

Converted the profile dedup removal path from hard-delete to soft-delete behind `is_dedup_soft_delete_enabled` flag.

## Changes

### OSS (open_source/reflexio)

**`reflexio/server/services/storage/storage_base/_profiles.py`**
- Added abstract method `supersede_profiles_by_ids(user_id, profile_ids, request_id) -> int`

**`reflexio/server/services/storage/sqlite_storage/_profiles.py`**
- Added `supersede_profiles_by_ids` implementation: user_id-scoped, eligible = NULL|PENDING, one `conn.commit()` per call, `status_change` lineage event per updated id with shared `request_id`, guarded on `rowcount > 0`. No FTS/vec deletion.

**`reflexio/server/services/profile/profile_generation_service.py`**
- Branched `_finalize_extracted_items`: flag ON + non-empty `request_id` → `supersede_profiles_by_ids`; flag OFF or empty `request_id` → existing `delete_user_profile` loop (byte-for-byte unchanged).

**`tests/server/services/profile/test_dedup_soft_delete_integration.py`** (new)
- 13 tests: storage-level (8) + service-level (5). All pass.

### Enterprise (reflexio_ext)

**`reflexio_ext/server/services/storage/supabase_storage/_profiles.py`**
- Added `supersede_profiles_by_ids` + `_rpc_status_change_with_request_id` to satisfy the new abstract method. Calls `lineage_status_change_and_log` RPC with caller-supplied `request_id` for set-based lineage. Covers NULL and PENDING eligible statuses in two RPC calls.

## Test Summary

13 new integration tests, all green. 590 total tests pass (profile + storage + feature_flags).

## generated_from_request_id Verification

CONFIRMED. `profile_deduplicator.py:618` explicitly sets `generated_from_request_id=request_id` on all new (merged + unique) profiles. The add side uses the existing column — no new add event needed.

## Concerns

None. Atomicity: one commit at end, guarded on rowcount. User_id-scoped predicate confirmed. Flag OFF = unchanged behavior. From_status derived from actual row. FTS/vec rows NOT deleted.
