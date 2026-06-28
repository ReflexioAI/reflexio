from __future__ import annotations

from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.governance import AuditEvent
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id="org1", db_path=str(tmp_path / "g.db"))


def test_audit_event_idempotency(storage):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref="subref_abc",
        request_ref="reqref_r1",
        idempotency_key="export_1",
        detail={"requests": 1},
    )

    assert storage.append_audit_event(event) is True
    assert storage.append_audit_event(event) is False
    rows = storage.list_audit_events(subject_ref="subref_abc")
    assert len(rows) == 1
    assert rows[0].idempotency_key == "export_1"


def test_purge_targets_require_snapshot_marker(storage):
    purge = storage.begin_purge_operation(
        purge_id="purge_1",
        idempotency_key="idem_1",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref="subref_abc",
        request_ref="reqref_r1",
    )
    storage.record_purge_target(
        purge_id=purge.purge_id,
        target_name="request",
        target_ref="req1",
        phase="delete",
        status="complete",
        deleted_count=1,
    )

    assert storage.purge_targets_prepared(purge.purge_id) is False
    with pytest.raises(ValueError, match="target snapshot"):
        storage.complete_purge_operation_with_audit(
            purge.purge_id,
            AuditEvent(
                org_id="org1",
                operation="ERASE",
                entity_type="request",
                subject_ref="subref_abc",
                request_ref="reqref_r1",
                idempotency_key=purge.purge_id,
            ),
        )


def test_complete_purge_operation_with_audit_is_atomic_success_path(storage):
    purge = storage.begin_purge_operation(
        purge_id="purge_2",
        idempotency_key="idem_2",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref="subref_abc",
        request_ref="reqref_r2",
    )
    storage.record_purge_target(
        purge_id=purge.purge_id,
        target_name="target_snapshot",
        target_ref="all",
        phase="prepare_targets",
        status="complete",
    )
    complete = storage.complete_purge_operation_with_audit(
        purge.purge_id,
        AuditEvent(
            org_id="org1",
            operation="ERASE",
            entity_type="request",
            subject_ref="subref_abc",
            request_ref="reqref_r2",
            idempotency_key=purge.purge_id,
        ),
    )

    assert complete.status == "complete"
    rows = storage.list_audit_events(subject_ref="subref_abc")
    assert [row.operation for row in rows] == ["ERASE"]
    same = storage.complete_purge_operation_with_audit(
        purge.purge_id,
        AuditEvent(
            org_id="org1",
            operation="ERASE",
            entity_type="request",
            subject_ref="subref_abc",
            request_ref="reqref_r2",
            idempotency_key=purge.purge_id,
        ),
    )
    assert same.status == "complete"
    assert len(storage.list_audit_events(subject_ref="subref_abc")) == 1
