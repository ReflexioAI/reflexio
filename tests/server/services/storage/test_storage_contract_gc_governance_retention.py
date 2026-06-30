"""Storage contract tests for gc_governance_retention.

Parametrized over locally-testable backends via the shared ``storage`` fixture
in conftest.py (currently SQLite only).  When the shared fixture is extended to
include a Supabase/Postgres backend the tests here will cover it automatically
with no changes required.

Design
------
``gc_governance_retention`` deletes audit_events rows older than
``audit_events_retention_days`` for the storage's own org, up to
``audit_events_delete_batch_limit`` rows per call.  The method returns 0
immediately when ``audit_events_retention_enabled`` is False.

We seed audit events with ``created_at=1`` (Unix epoch 1970-01-01) so they are
always older than any realistic retention window.  Recent events use the default
``created_at`` (now).

Cross-org scoping note
-----------------------
``append_audit_event`` enforces ``event.org_id == storage.org_id``, so it is not
possible to insert another org's events through the abstract ``BaseStorage`` API.
The org-scoping invariant (gc only deletes rows for its own org) is therefore
covered by the SQLite-specific test in
``tests/server/services/storage/sqlite_storage/test_governance_storage.py``
(``test_gc_governance_retention_deletes_expired_audit_rows_in_batches``), which
uses two ``SQLiteStorage`` instances on the same DB file.

Supabase/Postgres follow-up
-----------------------------
The shared ``storage`` fixture currently parametrizes SQLite only (see
``tests/server/services/storage/conftest.py``).  To run the supabase branch:
  1. Export ``DATA_SUPABASE_URL``, ``DATA_SUPABASE_KEY``, and
     ``DATA_SUPABASE_SERVICE_ROLE_KEY`` from the local Supabase stack (ports
     54321/54322).
  2. Add a ``"supabase"`` branch in the parametrized ``storage`` fixture in
     conftest.py.
Until then, supabase coverage is deferred; this file is structured so the tests
run automatically once the param is added.
"""

import pytest

from reflexio.models.api_schema.domain.governance import AuditEvent
from reflexio.models.config_schema import GovernanceRetentionConfig

pytestmark = pytest.mark.integration

# Pre-formatted refs that satisfy governance validation
# (pattern: {prefix}[0-9a-f]{32})
_SUBJECT_REF = "subref_v1_" + "a" * 32
_REQUEST_REF = "reqref_v1_" + "b" * 32

# Unix epoch 1 — always older than any realistic retention cutoff
_VERY_OLD_CREATED_AT = 1


def _aged_event(storage, idempotency_key: str) -> AuditEvent:
    """Return an audit event with a very old created_at for the storage's org."""
    return AuditEvent(
        org_id=storage.org_id,
        operation="EXPORT",
        entity_type="request",
        subject_ref=_SUBJECT_REF,
        request_ref=_REQUEST_REF,
        idempotency_key=idempotency_key,
        created_at=_VERY_OLD_CREATED_AT,
    )


def _recent_event(storage, idempotency_key: str) -> AuditEvent:
    """Return an audit event with a current created_at for the storage's org."""
    return AuditEvent(
        org_id=storage.org_id,
        operation="EXPORT",
        entity_type="request",
        subject_ref=_SUBJECT_REF,
        request_ref=_REQUEST_REF,
        idempotency_key=idempotency_key,
        # created_at defaults to now via _now_epoch()
    )


def _retention_config(**kwargs: object) -> GovernanceRetentionConfig:
    """Return an enabled retention config with 1-day window, overriding with kwargs."""
    defaults: dict[str, object] = {
        "audit_events_retention_enabled": True,
        "audit_events_retention_days": 1,
    }
    defaults.update(kwargs)
    return GovernanceRetentionConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Case 1: Aged event is deleted; recent event survives.
# ---------------------------------------------------------------------------


def test_gc_governance_retention_deletes_aged_event_keeps_recent(storage) -> None:
    """Aged audit event (created_at=1) is deleted; a recent event is untouched."""
    storage.append_audit_event(_aged_event(storage, "gc-aged"))
    storage.append_audit_event(_recent_event(storage, "gc-recent"))

    deleted = storage.gc_governance_retention(config=_retention_config())

    assert deleted == 1
    remaining = [e.idempotency_key for e in storage.list_audit_events()]
    assert "gc-recent" in remaining
    assert "gc-aged" not in remaining


# ---------------------------------------------------------------------------
# Case 2: No-op when retention is disabled.
# ---------------------------------------------------------------------------


def test_gc_governance_retention_noops_when_disabled(storage) -> None:
    """When audit_events_retention_enabled=False, gc_governance_retention returns 0 and deletes nothing."""
    storage.append_audit_event(_aged_event(storage, "gc-disabled-aged"))

    deleted = storage.gc_governance_retention(config=GovernanceRetentionConfig())

    assert deleted == 0
    assert len(storage.list_audit_events()) == 1


# ---------------------------------------------------------------------------
# Case 3: Batch limit caps deletions per call.
# ---------------------------------------------------------------------------


def test_gc_governance_retention_respects_batch_limit(storage) -> None:
    """With 3 eligible aged events and batch_limit=2, only 2 are deleted per call."""
    for i in range(3):
        storage.append_audit_event(_aged_event(storage, f"gc-batch-{i}"))

    first_call = storage.gc_governance_retention(
        config=_retention_config(audit_events_delete_batch_limit=2)
    )
    assert first_call == 2

    second_call = storage.gc_governance_retention(
        config=_retention_config(audit_events_delete_batch_limit=2)
    )
    assert second_call == 1

    third_call = storage.gc_governance_retention(
        config=_retention_config(audit_events_delete_batch_limit=2)
    )
    assert third_call == 0
    assert storage.list_audit_events() == []


# ---------------------------------------------------------------------------
# Case 4: Idempotent — second call on empty table returns 0.
# ---------------------------------------------------------------------------


def test_gc_governance_retention_idempotent(storage) -> None:
    """After all eligible events are deleted, a second call returns 0."""
    storage.append_audit_event(_aged_event(storage, "gc-idem"))

    first = storage.gc_governance_retention(config=_retention_config())
    assert first == 1

    second = storage.gc_governance_retention(config=_retention_config())
    assert second == 0
    assert storage.list_audit_events() == []
