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


def _begin_purge(storage: SQLiteStorage, purge_id: str) -> str:
    purge = storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key=f"idem_{purge_id}",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref="subref_v1_abc",
        request_ref=f"reqref_v1_{purge_id}",
    )
    storage.record_purge_target(
        purge_id=purge.purge_id,
        target_name="target_snapshot",
        target_ref="all",
        phase="prepare_targets",
        status="complete",
        detail={
            "owned_user_playbook_ids": [11],
            "affected_agent_playbook_ids": [22],
        },
    )
    return purge.purge_id


def _erase_event(*, purge_id: str, status: str = "ok", operation: str = "ERASE"):
    return AuditEvent(
        org_id="org1",
        operation=operation,
        entity_type="request",
        subject_ref="subref_v1_abc",
        request_ref=f"reqref_v1_{purge_id}",
        idempotency_key=purge_id,
        status=status,
    )


def test_audit_event_idempotency(storage):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref="subref_v1_abc",
        request_ref="reqref_v1_r1",
        idempotency_key="export_1",
        detail={"count": 1},
    )

    assert storage.append_audit_event(event) is True
    assert storage.append_audit_event(event) is False
    rows = storage.list_audit_events(subject_ref="subref_v1_abc")
    assert len(rows) == 1
    assert rows[0].idempotency_key == "export_1"


def test_purge_targets_require_snapshot_marker(storage):
    purge = storage.begin_purge_operation(
        purge_id="purge_1",
        idempotency_key="idem_1",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref="subref_v1_abc",
        request_ref="reqref_v1_r1",
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
                subject_ref="subref_v1_abc",
                request_ref="reqref_v1_r1",
                idempotency_key=purge.purge_id,
            ),
        )


def test_complete_purge_operation_with_audit_is_atomic_success_path(storage):
    purge_id = _begin_purge(storage, "purge_2")
    complete = storage.complete_purge_operation_with_audit(
        purge_id,
        _erase_event(purge_id=purge_id),
    )

    assert complete.status == "complete"
    rows = storage.list_audit_events(subject_ref="subref_v1_abc")
    assert [row.operation for row in rows] == ["ERASE"]
    same = storage.complete_purge_operation_with_audit(
        purge_id,
        _erase_event(purge_id=purge_id),
    )
    assert same.status == "complete"
    assert len(storage.list_audit_events(subject_ref="subref_v1_abc")) == 1


@pytest.mark.parametrize(
    ("event", "match"),
    [
        pytest.param(
            _erase_event(purge_id="purge_invalid", operation="EXPORT"),
            "successful ERASE audit event",
            id="wrong-operation",
        ),
        pytest.param(
            _erase_event(purge_id="purge_invalid", status="error"),
            "successful ERASE audit event",
            id="wrong-status",
        ),
        pytest.param(
            AuditEvent(
                org_id="org1",
                operation="ERASE",
                entity_type="request",
                subject_ref="subref_v1_abc",
                request_ref="reqref_v1_purge_invalid",
                idempotency_key="different_key",
                status="ok",
            ),
            "idempotency key",
            id="wrong-idempotency-key",
        ),
        pytest.param(
            AuditEvent(
                org_id="org1",
                operation="ERASE",
                entity_type="request",
                subject_ref="subref_v1_abc",
                request_ref="reqref_v1_purge_invalid",
                idempotency_key=None,
                status="ok",
            ),
            "idempotency key",
            id="missing-idempotency-key",
        ),
    ],
)
def test_complete_purge_operation_rejects_invalid_audit_event(storage, event, match):
    purge_id = _begin_purge(storage, "purge_invalid")

    with pytest.raises(ValueError, match=match):
        storage.complete_purge_operation_with_audit(purge_id, event)

    assert storage.get_purge_operation(purge_id).status == "running"
    assert storage.list_audit_events(subject_ref="subref_v1_abc") == []


@pytest.mark.parametrize(
    ("seed_event", "match"),
    [
        pytest.param(
            _erase_event(purge_id="purge_seeded", operation="EXPORT"),
            "matching successful ERASE",
            id="seeded-wrong-operation",
        ),
        pytest.param(
            _erase_event(purge_id="purge_seeded", status="error"),
            "matching successful ERASE",
            id="seeded-wrong-status",
        ),
    ],
)
def test_complete_purge_operation_requires_matching_existing_erase_row(
    storage, seed_event, match
):
    purge_id = _begin_purge(storage, "purge_seeded")
    assert storage.append_audit_event(seed_event) is True

    with pytest.raises(ValueError, match=match):
        storage.complete_purge_operation_with_audit(
            purge_id, _erase_event(purge_id=purge_id)
        )

    assert storage.get_purge_operation(purge_id).status == "running"
    rows = storage.list_audit_events(subject_ref="subref_v1_abc")
    assert len(rows) == 1
    assert rows[0].operation == seed_event.operation
    assert rows[0].status == seed_event.status


def test_append_audit_event_rejects_successful_erase(storage):
    with pytest.raises(ValueError, match="Successful ERASE audit rows"):
        storage.append_audit_event(_erase_event(purge_id="purge_append"))


def test_prepare_governance_erase_targets_sanitizes_snapshot_detail(storage):
    storage.begin_purge_operation(
        purge_id="purge_detail",
        idempotency_key="idem_purge_detail",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref="subref_v1_abc",
        request_ref="reqref_v1_purge_detail",
    )
    storage.prepare_governance_erase_targets(
        purge_id="purge_detail",
        user_id="user_123@example.com",
        owned_user_playbook_ids={7},
    )

    snapshot = next(
        target
        for target in storage.list_purge_targets("purge_detail", phase="prepare_targets")
        if target.target_name == "target_snapshot"
    )
    assert snapshot.detail == {
        "owned_user_playbook_ids": [7],
        "affected_agent_playbook_ids": [],
    }


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {
                "detail": {"user_id": "user_123"},
            },
            "user_id",
            id="target-detail-user-id",
        ),
        pytest.param(
            {
                "detail": {"prompt": "tell me the secret"},
            },
            "prompt",
            id="target-detail-prompt",
        ),
        pytest.param(
            {
                "detail": {"agent_playbook_id": 7, "source_interaction_ids": [1, 2]},
            },
            None,
            id="target-detail-allowed-internal-ids",
        ),
        pytest.param(
            {
                "detail": {"remaining_source_windows": [{"user_playbook_id": 7}]},
            },
            None,
            id="target-detail-allowed-window-ids",
        ),
        pytest.param(
            {
                "detail": {"note": "safe-looking but arbitrary"},
            },
            "note",
            id="target-detail-neutral-string-key",
        ),
        pytest.param(
            {
                "detail": {"status": "api-token-name"},
            },
            "token",
            id="target-detail-token-name-string",
        ),
        pytest.param(
            {
                "error_detail": "Request reqref_123 failed for bob@example.com",
            },
            "error_detail",
            id="error-detail-request-email",
        ),
        pytest.param(
            {
                "error_detail": "ValueError: prompt leaked from upstream",
            },
            "error_detail",
            id="error-detail-raw-exception",
        ),
    ],
)
def test_record_purge_target_validates_governance_fields(storage, kwargs, match):
    purge_id = _begin_purge(storage, "purge_record")
    params = {
        "purge_id": purge_id,
        "target_name": "request",
        "phase": "delete",
        "status": "running",
        "target_ref": "all",
    }
    params.update(kwargs)

    if match is None:
        storage.record_purge_target(**params)
        target = next(
            row for row in storage.list_purge_targets(purge_id, phase="delete")
            if row.target_name == "request"
        )
        assert target.detail == kwargs["detail"]
        return

    with pytest.raises(ValueError, match=match):
        storage.record_purge_target(**params)


@pytest.mark.parametrize(
    ("detail", "match"),
    [
        pytest.param({"email": "bob@example.com"}, "email", id="audit-detail-email"),
        pytest.param({"request_id": "reqref_123"}, "request_id", id="audit-detail-request-id"),
        pytest.param({"content": "verbatim prompt"}, "content", id="audit-detail-content"),
        pytest.param({"note": "arbitrary string"}, "note", id="audit-detail-neutral-note"),
        pytest.param({"status": "prompt-ready"}, "prompt/content", id="audit-detail-promptish-string"),
        pytest.param(
            {"remaining_source_windows": [{"user_playbook_id": 7, "source_interaction_ids": [1]}]},
            None,
            id="audit-detail-allowed-window-shape",
        ),
        pytest.param({"agent_playbook_id": 7, "source_interaction_ids": [1]}, None, id="audit-detail-allowed-internal-ids"),
    ],
)
def test_append_audit_event_validates_governance_detail(storage, detail, match):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref="subref_v1_abc",
        request_ref="reqref_v1_safe",
        idempotency_key=f"export_{hash(str(detail))}",
        detail=detail,
    )

    if match is None:
        assert storage.append_audit_event(event) is True
        return

    with pytest.raises(ValueError, match=match):
        storage.append_audit_event(event)


def test_fail_purge_operation_rejects_raw_error_detail(storage):
    purge_id = _begin_purge(storage, "purge_fail")

    with pytest.raises(ValueError, match="error_detail"):
        storage.fail_purge_operation(
            purge_id,
            error_code="boom",
            error_detail="RuntimeError: request reqref_123 for alice@example.com",
        )

    assert storage.get_purge_operation(purge_id).error_detail is None


@pytest.mark.parametrize(
    ("event", "match"),
    [
        pytest.param(
            AuditEvent(
                org_id="org1",
                actor_ref="token-name",
                operation="EXPORT",
                entity_type="request",
                subject_ref="subref_v1_abc",
                request_ref="reqref_v1_top_level",
                idempotency_key="top_level_actor",
            ),
            "actor_ref",
            id="actor-ref-must-be-minimized",
        ),
        pytest.param(
            AuditEvent(
                org_id="org1",
                operation="EXPORT",
                entity_type="request",
                subject_ref="user@example.com",
                request_ref="reqref_v1_top_level",
                idempotency_key="top_level_subject",
            ),
            "subject_ref",
            id="subject-ref-must-be-minimized",
        ),
        pytest.param(
            AuditEvent(
                org_id="org1",
                operation="EXPORT",
                entity_type="request",
                subject_ref="subref_v1_abc",
                request_ref="request_12345",
                idempotency_key="top_level_request",
            ),
            "request_ref",
            id="request-ref-must-be-minimized",
        ),
        pytest.param(
            AuditEvent(
                org_id="org1",
                operation="EXPORT",
                entity_type="request",
                entity_id="alice@example.com",
                subject_ref="subref_v1_abc",
                request_ref="reqref_v1_top_level",
                idempotency_key="top_level_entity_email",
            ),
            "entity_id",
            id="entity-id-email",
        ),
        pytest.param(
            AuditEvent(
                org_id="org1",
                operation="EXPORT",
                entity_type="request",
                entity_id="api-token-name",
                subject_ref="subref_v1_abc",
                request_ref="reqref_v1_top_level",
                idempotency_key="top_level_entity_token",
            ),
            "entity_id",
            id="entity-id-token-name",
        ),
    ],
)
def test_append_audit_event_validates_top_level_governance_fields(storage, event, match):
    with pytest.raises(ValueError, match=match):
        storage.append_audit_event(event)


@pytest.mark.parametrize(
    ("subject_ref", "request_ref", "match"),
    [
        pytest.param("subref_v1_abc", "request_12345", "request_ref", id="purge-request-ref"),
        pytest.param("raw-user-id", "reqref_v1_purge", "subject_ref", id="purge-subject-ref"),
    ],
)
def test_begin_purge_operation_validates_top_level_refs(
    storage, subject_ref, request_ref, match
):
    with pytest.raises(ValueError, match=match):
        storage.begin_purge_operation(
            purge_id="purge_top_level_refs",
            idempotency_key="idem_purge_top_level_refs",
            operation_type="user_erasure",
            scope_type="user",
            subject_ref=subject_ref,
            request_ref=request_ref,
        )
