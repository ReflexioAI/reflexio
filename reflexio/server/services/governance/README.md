# server/services/governance
Description: Governance service layer for user data export, erasure, audit events, and idempotent purge operation tracking.

## Main Entry Points

| File | Purpose |
|------|---------|
| `service.py` | `GovernanceService` orchestrates user export and user erasure workflows against `request_context.storage`. |
| `subject_refs.py` | Stable privacy-safe reference helpers (`subject_ref`, `request_ref`, `stable_id`) used in audit and purge records. |

## Purpose

1. **User export** - Build a bundle of a user's profiles, interactions, requests, sessions, and user playbooks, then append an audit event.
2. **User erasure** - Prepare idempotent purge targets, delete user-owned rows, hide/rebuild impacted agent playbooks, and complete or fail the purge with audit state.
3. **Privacy-safe bookkeeping** - Store hashed subject/request references instead of raw user or request identifiers in governance audit/purge rows.

## Architecture Pattern

`GovernanceService` is a focused orchestration layer over storage primitives. It does not own persistence directly; backend-neutral contracts live in `storage/storage_base/_governance.py` and the SQLite implementation lives in `storage/sqlite_storage/_governance.py`.

```
GovernanceService
  -> storage.begin_purge_operation / prepare_governance_erase_targets
  -> storage.apply_governance_user_data_delete
  -> storage.apply_governance_agent_playbook_rebuild
  -> storage.complete_purge_operation_with_audit
```

## Key Contracts

- `export_user(user_id, request_id) -> UserExportResult`
- `erase_user(user_id, request_id) -> UserEraseResult`
- Domain schemas: `models/api_schema/domain/governance.py`
- Retention config: `Config.governance_retention` / `GovernanceRetentionConfig`

## Requirements / Problems to Avoid

- **Do not store raw identifiers in audit/purge references** — use `subject_refs.py` helpers.
- **Do not bypass storage governance methods** — purge target preparation, deletion, rebuild, and audit completion are idempotency-sensitive.
- **Keep governance storage backend-neutral** — add abstract methods to `storage_base/_governance.py` before implementation-specific code.
