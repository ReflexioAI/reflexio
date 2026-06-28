from __future__ import annotations

import json
import sqlite3
from typing import Any, cast
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    AuditOperation,
    AuditStatus,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._governance import (
    init_governance_tables,
)

pytestmark = pytest.mark.integration

SUBJECT_REF = "subref_v1_" + "a" * 32
OTHER_SUBJECT_REF = "subref_v1_" + "c" * 32
REQUEST_REF = "reqref_v1_" + "b" * 32
OTHER_REQUEST_REF = "reqref_v1_" + "d" * 32
ACTOR_REF = "actref_v1_" + "e" * 32


@pytest.fixture
def storage(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id="org1", db_path=str(tmp_path / "g.db"))


@pytest.fixture
def storage_factory(tmp_path):
    def _make_storage(org_id: str) -> SQLiteStorage:
        return SQLiteStorage(org_id=org_id, db_path=str(tmp_path / "shared-g.db"))

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield _make_storage


def _begin_purge(storage: SQLiteStorage, purge_id: str) -> str:
    purge = storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key=f"idem_{purge_id}",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
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


def _erase_event(
    *,
    purge_id: str,
    status: AuditStatus = "ok",
    operation: AuditOperation = "ERASE",
):
    return AuditEvent(
        org_id="org1",
        operation=operation,
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key=purge_id,
        status=status,
    )


def _seed_user_scoped_rows(storage: SQLiteStorage, *, user_id: str) -> None:
    created_at = "2026-01-01T00:00:00.000Z"
    storage.conn.execute(
        """INSERT INTO requests (
               request_id, user_id, created_at, source, agent_version, session_id, evaluation_only
           ) VALUES (?, ?, ?, '', '', ?, 0)""",
        ("request_seed", user_id, created_at, "session_seed"),
    )
    storage.conn.execute(
        """INSERT INTO interactions (
               user_id, content, request_id, created_at, role, user_action,
               user_action_description, interacted_image_url, image_encoding,
               shadow_content, expert_content, tools_used, citations, embedding
           ) VALUES (?, '', ?, ?, 'User', 'none', '', '', '', '', '', '[]', '[]', '[]')""",
        (user_id, "request_seed", created_at),
    )
    storage.conn.execute(
        """INSERT INTO profiles (
               profile_id, user_id, content, last_modified_timestamp,
               generated_from_request_id, profile_time_to_live, expiration_timestamp,
               embedding, source_interaction_ids, created_at
           ) VALUES (?, ?, ?, ?, ?, 'infinity', ?, '[]', '[]', ?)""",
        ("profile_seed", user_id, "profile-content", 1, "request_seed", 4102444800, created_at),
    )
    storage.conn.execute(
        """INSERT INTO user_playbooks (
               user_id, playbook_name, created_at, request_id, agent_version,
               content, source_interaction_ids, embedding
           ) VALUES (?, '', ?, ?, '', ?, '[]', '[]')""",
        (user_id, created_at, "request_seed", "playbook-content"),
    )
    storage.conn.commit()


def _seed_agent_playbook(
    storage: SQLiteStorage,
    *,
    status: Status | None = Status.ARCHIVED,
    source_windows: list[AgentPlaybookSourceWindow] | None = None,
) -> int:
    playbook = AgentPlaybook(
        playbook_name="governance-rebuild",
        agent_version="test-agent",
        content="original content",
        trigger="original trigger",
        rationale="original rationale",
        status=status,
        tags=["seed"],
    )
    saved = storage.save_agent_playbooks([playbook])[0]
    storage.set_source_windows_for_agent_playbook(
        saved.agent_playbook_id,
        source_windows
        or [AgentPlaybookSourceWindow(user_playbook_id=7, source_interaction_ids=[101])],
    )
    return saved.agent_playbook_id


def test_audit_event_idempotency(storage):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="export_1",
        detail={"count": 1},
    )

    assert storage.append_audit_event(event) is True
    assert storage.append_audit_event(event) is False
    rows = storage.list_audit_events(subject_ref=SUBJECT_REF)
    assert len(rows) == 1
    assert rows[0].idempotency_key == "export_1"


def test_append_audit_event_rejects_mismatched_org_id(storage):
    event = AuditEvent(
        org_id="org2",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="export_wrong_org",
    )

    with pytest.raises(ValueError, match="org_id"):
        storage.append_audit_event(event)

    assert storage.list_audit_events(subject_ref=SUBJECT_REF) == []


def test_purge_targets_require_snapshot_marker(storage):
    purge = storage.begin_purge_operation(
        purge_id="purge_1",
        idempotency_key="idem_1",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    storage.record_purge_target(
        purge_id=purge.purge_id,
        target_name="request",
        target_ref=REQUEST_REF,
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
                subject_ref=SUBJECT_REF,
                request_ref=REQUEST_REF,
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
    rows = storage.list_audit_events(subject_ref=SUBJECT_REF)
    assert [row.operation for row in rows] == ["ERASE"]
    same = storage.complete_purge_operation_with_audit(
        purge_id,
        _erase_event(purge_id=purge_id),
    )
    assert same.status == "complete"
    assert len(storage.list_audit_events(subject_ref=SUBJECT_REF)) == 1


def test_complete_purge_operation_with_audit_accepts_planned_success_detail(storage):
    purge_id = _begin_purge(storage, "purge_success_detail")
    deleted_counts = {
        "interactions": 3,
        "user_playbooks": 2,
        "profiles": 1,
        "requests": 1,
        "purged_profiles": 0,
        "purged_user_playbooks": 0,
    }
    rebuilt_ids = [17, 21]

    complete = storage.complete_purge_operation_with_audit(
        purge_id,
        AuditEvent(
            org_id="org1",
            operation="ERASE",
            entity_type="request",
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
            idempotency_key=purge_id,
            detail={
                "deleted_counts": deleted_counts,
                "rebuilt_agent_playbook_ids": rebuilt_ids,
            },
        ),
    )

    assert complete.status == "complete"
    audit_rows = storage.list_audit_events(subject_ref=SUBJECT_REF)
    assert len(audit_rows) == 1
    assert audit_rows[0].detail == {
        "deleted_counts": deleted_counts,
        "rebuilt_agent_playbook_ids": rebuilt_ids,
    }


@pytest.mark.parametrize(
    ("retry_kwargs", "match"),
    [
        pytest.param({"purge_id": "purge_begin_retry_other"}, "purge_id", id="purge-id"),
        pytest.param({"request_ref": OTHER_REQUEST_REF}, "request_ref", id="request-ref"),
        pytest.param({"scope_type": "org"}, "scope_type", id="scope-type"),
    ],
)
def test_begin_purge_operation_rejects_mismatched_idempotent_retry(
    storage, retry_kwargs, match
):
    storage.begin_purge_operation(
        purge_id="purge_begin_retry",
        idempotency_key="idem_begin_retry",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )

    with pytest.raises(ValueError, match=match):
        storage.begin_purge_operation(
            purge_id=retry_kwargs.get("purge_id", "purge_begin_retry"),
            idempotency_key="idem_begin_retry",
            operation_type=retry_kwargs.get("operation_type", "user_erasure"),
            scope_type=retry_kwargs.get("scope_type", "user"),
            subject_ref=retry_kwargs.get("subject_ref", SUBJECT_REF),
            request_ref=retry_kwargs.get("request_ref", REQUEST_REF),
        )

    purge = storage.get_purge_operation("purge_begin_retry")
    assert purge.request_ref == REQUEST_REF
    assert purge.scope_type == "user"
    assert purge.purge_id == "purge_begin_retry"


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
                subject_ref=SUBJECT_REF,
                request_ref=REQUEST_REF,
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
                subject_ref=SUBJECT_REF,
                request_ref=REQUEST_REF,
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
    assert storage.list_audit_events(subject_ref=SUBJECT_REF) == []


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
    rows = storage.list_audit_events(subject_ref=SUBJECT_REF)
    assert len(rows) == 1
    assert rows[0].operation == seed_event.operation
    assert rows[0].status == seed_event.status


@pytest.mark.parametrize(
    ("field_name", "seed_kwargs"),
    [
        pytest.param("entity_type", {"entity_type": "session"}, id="entity-type"),
        pytest.param("subject_ref", {"subject_ref": OTHER_SUBJECT_REF}, id="subject-ref"),
        pytest.param("request_ref", {"request_ref": OTHER_REQUEST_REF}, id="request-ref"),
        pytest.param("actor_type", {"actor_type": "jwt"}, id="actor-type"),
        pytest.param("actor_ref", {"actor_ref": ACTOR_REF}, id="actor-ref"),
        pytest.param("entity_id", {"entity_id": "17"}, id="entity-id"),
        pytest.param("detail", {"detail": {"count": 2}}, id="detail"),
    ],
)
def test_complete_purge_operation_rejects_mismatched_existing_erase_row(
    storage, field_name, seed_kwargs
):
    purge_id = _begin_purge(storage, "purge_seeded_mismatch")
    seeded_event = _erase_event(purge_id=purge_id).model_copy(update=seed_kwargs)
    storage.conn.execute(
        """INSERT INTO audit_events (
               org_id, actor_type, actor_ref, operation, entity_type, entity_id,
               subject_ref, request_ref, idempotency_key, status, detail, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            seeded_event.org_id,
            seeded_event.actor_type,
            seeded_event.actor_ref,
            seeded_event.operation,
            seeded_event.entity_type,
            seeded_event.entity_id,
            seeded_event.subject_ref,
            seeded_event.request_ref,
            seeded_event.idempotency_key,
            seeded_event.status,
            json.dumps(seeded_event.detail) if seeded_event.detail is not None else None,
            seeded_event.created_at,
        ),
    )
    storage.conn.commit()

    with pytest.raises(ValueError, match="matching successful ERASE"):
        storage.complete_purge_operation_with_audit(
            purge_id, _erase_event(purge_id=purge_id)
        )

    assert storage.get_purge_operation(purge_id).status == "running"
    rows = storage.list_audit_events(subject_ref=seeded_event.subject_ref)
    assert len(rows) == 1
    assert getattr(rows[0], field_name) == getattr(seeded_event, field_name)


def test_append_audit_event_rejects_successful_erase(storage):
    with pytest.raises(ValueError, match="Successful ERASE audit rows"):
        storage.append_audit_event(_erase_event(purge_id="purge_append"))


def test_prepare_governance_erase_targets_sanitizes_snapshot_detail(storage):
    storage.begin_purge_operation(
        purge_id="purge_detail",
        idempotency_key="idem_purge_detail",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
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


def test_prepare_governance_erase_targets_persists_rebuild_source_windows(storage):
    storage.begin_purge_operation(
        purge_id="purge_rebuild_windows",
        idempotency_key="idem_purge_rebuild_windows",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    agent_playbook_id = _seed_agent_playbook(
        storage,
        source_windows=[
            AgentPlaybookSourceWindow(user_playbook_id=7, source_interaction_ids=[101, 102]),
            AgentPlaybookSourceWindow(user_playbook_id=9, source_interaction_ids=[201]),
        ],
    )

    storage.prepare_governance_erase_targets(
        purge_id="purge_rebuild_windows",
        user_id="user-rebuild-windows",
        owned_user_playbook_ids={7},
    )

    rebuild_target = next(
        target
        for target in storage.list_purge_targets(
            "purge_rebuild_windows", phase="rebuild_without_erased_sources"
        )
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    assert rebuild_target.status == "pending"
    assert rebuild_target.detail == {
        "original_source_windows": [
            {"user_playbook_id": 7, "source_interaction_ids": [101, 102]},
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
        "remaining_source_windows": [
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
    }


def test_hide_governance_agent_playbooks_for_rebuild_sets_archive_in_progress_and_hide_marker(
    storage,
):
    purge_id = "purge_hide_rebuild"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_hide_rebuild",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    agent_playbook_id = _seed_agent_playbook(
        storage,
        status=None,
        source_windows=[
            AgentPlaybookSourceWindow(user_playbook_id=7, source_interaction_ids=[101]),
            AgentPlaybookSourceWindow(user_playbook_id=9, source_interaction_ids=[201]),
        ],
    )
    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id="user-hide-rebuild",
        owned_user_playbook_ids={7},
    )
    expected_detail = {
        "original_source_windows": [
            {"user_playbook_id": 7, "source_interaction_ids": [101]},
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
        "remaining_source_windows": [
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
    }

    hidden_ids = storage.hide_governance_agent_playbooks_for_rebuild(purge_id)

    assert hidden_ids == [agent_playbook_id]
    status = storage.conn.execute(
        "SELECT status FROM agent_playbooks WHERE agent_playbook_id = ?",
        (agent_playbook_id,),
    ).fetchone()[0]
    assert status == Status.ARCHIVE_IN_PROGRESS.value
    hide_target = next(
        target
        for target in storage.list_purge_targets(purge_id, phase="hide_for_rebuild")
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    assert hide_target.status == "complete"
    rebuild_target = next(
        target
        for target in storage.list_purge_targets(
            purge_id, phase="rebuild_without_erased_sources"
        )
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    assert rebuild_target.status == "running"
    assert rebuild_target.detail == expected_detail


def test_apply_governance_agent_playbook_rebuild_completes_planned_phase(storage):
    purge_id = "purge_rebuild_complete"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_rebuild_complete",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    agent_playbook_id = _seed_agent_playbook(
        storage,
        status=Status.ARCHIVE_IN_PROGRESS,
        source_windows=[
            AgentPlaybookSourceWindow(user_playbook_id=7, source_interaction_ids=[101]),
            AgentPlaybookSourceWindow(user_playbook_id=9, source_interaction_ids=[201]),
        ],
    )
    storage.record_purge_target(
        purge_id=purge_id,
        target_name="agent_playbook",
        target_ref=str(agent_playbook_id),
        phase="rebuild_without_erased_sources",
        status="running",
        detail={
            "original_source_windows": [
                {"user_playbook_id": 7, "source_interaction_ids": [101]},
                {"user_playbook_id": 9, "source_interaction_ids": [201]},
            ],
            "remaining_source_windows": [
                {"user_playbook_id": 9, "source_interaction_ids": [201]},
            ],
        },
    )
    expected_detail = {
        "original_source_windows": [
            {"user_playbook_id": 7, "source_interaction_ids": [101]},
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
        "remaining_source_windows": [
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
    }

    storage.apply_governance_agent_playbook_rebuild(
        purge_id=purge_id,
        agent_playbook_id=agent_playbook_id,
        remaining_source_windows=[
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
        content="rebuilt content",
        trigger="rebuilt trigger",
        rationale="rebuilt rationale",
        blocking_issue=None,
        expanded_terms="rebuilt terms",
        tags=["rebuilt"],
    )

    rebuild_target = next(
        target
        for target in storage.list_purge_targets(
            purge_id, phase="rebuild_without_erased_sources"
        )
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    assert rebuild_target.status == "complete"
    assert rebuild_target.detail == expected_detail


def test_purge_targets_are_scoped_by_org_for_same_purge_id(storage_factory):
    storage_org1 = storage_factory("org1")
    storage_org2 = storage_factory("org2")
    purge_id = "purge_shared_scope"

    for storage_instance, request_ref in (
        (storage_org1, REQUEST_REF),
        (storage_org2, OTHER_REQUEST_REF),
    ):
        storage_instance.begin_purge_operation(
            purge_id=purge_id,
            idempotency_key=f"idem_{storage_instance.org_id}_{purge_id}",
            operation_type="user_erasure",
            scope_type="user",
            subject_ref=SUBJECT_REF,
            request_ref=request_ref,
        )

    storage_org1.record_purge_target(
        purge_id=purge_id,
        target_name="target_snapshot",
        target_ref="all",
        phase="prepare_targets",
        status="complete",
        detail={"prepared": True},
    )
    storage_org1.record_purge_target(
        purge_id=purge_id,
        target_name="request",
        target_ref=REQUEST_REF,
        phase="delete",
        status="pending",
        detail={"count": 1},
    )
    storage_org2.record_purge_target(
        purge_id=purge_id,
        target_name="request",
        target_ref=OTHER_REQUEST_REF,
        phase="delete",
        status="complete",
        detail={"count": 2},
        deleted_count=2,
    )

    org1_targets = storage_org1.list_purge_targets(purge_id)
    org2_targets = storage_org2.list_purge_targets(purge_id)

    assert {(target.phase, target.target_ref, target.status) for target in org1_targets} == {
        ("delete", REQUEST_REF, "pending"),
        ("prepare_targets", "all", "complete"),
    }
    assert {(target.phase, target.target_ref, target.status) for target in org2_targets} == {
        ("delete", OTHER_REQUEST_REF, "complete"),
    }
    assert storage_org1.purge_targets_prepared(purge_id) is True
    assert storage_org2.purge_targets_prepared(purge_id) is False

    storage_org2.record_purge_target(
        purge_id=purge_id,
        target_name="request",
        target_ref=REQUEST_REF,
        phase="delete",
        status="running",
        detail={"count": 3},
    )

    org1_request_target = next(
        target
        for target in storage_org1.list_purge_targets(purge_id, phase="delete")
        if target.target_ref == REQUEST_REF
    )
    org2_delete_targets = storage_org2.list_purge_targets(purge_id, phase="delete")

    assert org1_request_target.status == "pending"
    assert org1_request_target.detail == {"count": 1}
    assert {(target.target_ref, target.status, target.deleted_count) for target in org2_delete_targets} == {
        (OTHER_REQUEST_REF, "complete", 2),
        (REQUEST_REF, "running", 0),
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
                "error_detail": "stable failure detail",
            },
            "error_detail",
            id="error-detail-freeform-prose",
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
    ("deleted_count", "match"),
    [
        pytest.param(cast(Any, True), "deleted_count", id="bool"),
        pytest.param(cast(Any, 1.5), "deleted_count", id="float"),
        pytest.param(-1, "deleted_count", id="negative"),
    ],
)
def test_record_purge_target_rejects_invalid_deleted_count(storage, deleted_count, match):
    purge_id = _begin_purge(storage, "purge_deleted_count")

    with pytest.raises(ValueError, match=match):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="request",
            target_ref="all",
            phase="delete",
            status="complete",
            deleted_count=deleted_count,
        )


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
    detail_key = next(iter(detail))
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key=f"export_detail_{detail_key}",
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


def test_fail_purge_operation_rejects_freeform_error_detail(storage):
    purge_id = _begin_purge(storage, "purge_fail_freeform")

    with pytest.raises(ValueError, match="error_detail"):
        storage.fail_purge_operation(
            purge_id,
            error_code="PURGE_TARGET_FAILED",
            error_detail="stable failure detail",
        )

    assert storage.get_purge_operation(purge_id).error_detail is None


def test_fail_purge_operation_persists_code_shaped_error_detail(storage):
    purge_id = _begin_purge(storage, "purge_fail_code_detail")

    failed = storage.fail_purge_operation(
        purge_id,
        error_code="PURGE_TARGET_FAILED",
        error_detail="target_delete_failed",
    )

    assert failed.status == "failed"
    assert failed.error_detail == "target_delete_failed"


@pytest.mark.parametrize(
    ("error_code", "match"),
    [
        pytest.param("PURGE_TARGET_FAILED", None, id="stable-code"),
        pytest.param("alice@example.com", "error_code", id="email"),
        pytest.param("request_12345", "error_code", id="request-id"),
        pytest.param("user_123", "error_code", id="user-like"),
    ],
)
def test_fail_purge_operation_validates_error_code(storage, error_code, match):
    purge_id = _begin_purge(storage, f"purge_error_code_{error_code.replace('@', '_')}")

    if match is None:
        failed = storage.fail_purge_operation(
            purge_id,
            error_code=error_code,
            error_detail="target_delete_failed",
        )
        assert failed.status == "failed"
        assert failed.error_code == error_code
        return

    with pytest.raises(ValueError, match=match):
        storage.fail_purge_operation(
            purge_id,
            error_code=error_code,
            error_detail="target_delete_failed",
        )

    assert storage.get_purge_operation(purge_id).error_code is None


@pytest.mark.parametrize(
    ("event", "match"),
    [
        pytest.param(
            AuditEvent(
                org_id="org1",
                actor_ref=ACTOR_REF[:-1],
                operation="EXPORT",
                entity_type="request",
                subject_ref=SUBJECT_REF,
                request_ref=REQUEST_REF,
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
                request_ref=REQUEST_REF,
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
                subject_ref=SUBJECT_REF,
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
                subject_ref=SUBJECT_REF,
                request_ref=REQUEST_REF,
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
                subject_ref=SUBJECT_REF,
                request_ref=REQUEST_REF,
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
    ("field_name", "value"),
    [
        pytest.param("actor_type", "person", id="actor-type"),
        pytest.param("operation", "PURGE", id="operation"),
        pytest.param("entity_type", "message", id="entity-type"),
        pytest.param("status", "done", id="status"),
    ],
)
def test_append_audit_event_rejects_invalid_top_level_enum_values(
    storage, field_name, value
):
    event = AuditEvent.model_construct(
        org_id="org1",
        actor_type="system",
        actor_ref=None,
        operation="EXPORT",
        entity_type="request",
        entity_id=None,
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key=f"invalid_{field_name}",
        status="ok",
        detail=None,
        created_at=1,
    )
    setattr(event, field_name, value)

    with pytest.raises(ValueError, match=field_name):
        storage.append_audit_event(event)


def test_append_audit_event_requires_minimized_request_ref(storage):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=None,
        idempotency_key="missing_request_ref",
    )

    with pytest.raises(ValueError, match="request_ref"):
        storage.append_audit_event(event)


@pytest.mark.parametrize(
    ("subject_ref", "request_ref", "match"),
    [
        pytest.param(SUBJECT_REF, "request_12345", "request_ref", id="purge-request-ref"),
        pytest.param("raw-user-id", REQUEST_REF, "subject_ref", id="purge-subject-ref"),
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


@pytest.mark.parametrize(
    ("operation_type", "scope_type", "match"),
    [
        pytest.param(cast(Any, "erase_user"), "user", "operation_type", id="operation-type"),
        pytest.param("user_erasure", cast(Any, "workspace"), "scope_type", id="scope-type"),
    ],
)
def test_begin_purge_operation_rejects_invalid_enum_values(
    storage, operation_type, scope_type, match
):
    with pytest.raises(ValueError, match=match):
        storage.begin_purge_operation(
            purge_id="purge_invalid_enum",
            idempotency_key="idem_purge_invalid_enum",
            operation_type=operation_type,
            scope_type=scope_type,
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
        )


@pytest.mark.parametrize(
    "purge_id",
    [
        "alice@example.com",
        "request_12345",
        "alice",
        SUBJECT_REF,
    ],
)
def test_begin_purge_operation_rejects_unsafe_purge_id(storage, purge_id):
    with pytest.raises(ValueError, match="purge_id"):
        storage.begin_purge_operation(
            purge_id=purge_id,
            idempotency_key="idem_purge_invalid_id",
            operation_type="user_erasure",
            scope_type="user",
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
        )


@pytest.mark.parametrize("detail_key", ["remaining_source_windows", "original_source_windows"])
def test_append_audit_event_rejects_mixed_case_window_keys(storage, detail_key):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key=f"mixed_case_{detail_key}",
        detail={detail_key: [{"User_Playbook_Id": "alice@example.com"}]},
    )

    with pytest.raises(ValueError, match="user_playbook_id"):
        storage.append_audit_event(event)


@pytest.mark.parametrize("detail_key", ["remaining_source_windows", "original_source_windows"])
def test_record_purge_target_rejects_mixed_case_window_keys(storage, detail_key):
    purge_id = _begin_purge(storage, f"purge_{detail_key}")

    with pytest.raises(ValueError, match="user_playbook_id"):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="agent_playbook",
            target_ref="7",
            phase="rebuild",
            status="running",
            detail={detail_key: [{"User_Playbook_Id": "alice@example.com"}]},
        )


@pytest.mark.parametrize("detail_key", ["remaining_source_windows", "original_source_windows"])
def test_append_audit_event_requires_window_user_playbook_id(storage, detail_key):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key=f"missing_upb_{detail_key}",
        detail={detail_key: [{"source_interaction_ids": [1, 2]}]},
    )

    with pytest.raises(ValueError, match="user_playbook_id"):
        storage.append_audit_event(event)


@pytest.mark.parametrize("detail_key", ["remaining_source_windows", "original_source_windows"])
def test_record_purge_target_requires_window_user_playbook_id(storage, detail_key):
    purge_id = _begin_purge(storage, f"purge_missing_upb_{detail_key}")

    with pytest.raises(ValueError, match="user_playbook_id"):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="agent_playbook",
            target_ref="7",
            phase="rebuild",
            status="running",
            detail={detail_key: [{"source_interaction_ids": [1, 2]}]},
        )


@pytest.mark.parametrize(
    ("target_ref", "match"),
    [
        pytest.param("all", None, id="marker-all"),
        pytest.param("17", None, id="internal-numeric-id"),
        pytest.param(REQUEST_REF, None, id="minimized-request-ref"),
        pytest.param(SUBJECT_REF, None, id="minimized-subject-ref"),
        pytest.param("", None, id="empty-default"),
        pytest.param("alice@example.com", "target_ref", id="raw-email"),
        pytest.param("request_12345", "target_ref", id="raw-request-id"),
        pytest.param("alice", "target_ref", id="raw-user-like"),
    ],
)
def test_record_purge_target_validates_target_ref_contract(storage, target_ref, match):
    purge_id = _begin_purge(storage, "purge_target_ref")

    if match is None:
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="request",
            target_ref=target_ref,
            phase="delete",
            status="running",
        )
        return

    with pytest.raises(ValueError, match=match):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="request",
            target_ref=target_ref,
            phase="delete",
            status="running",
        )


@pytest.mark.parametrize("purge_id", ["alice@example.com", "request_12345", "alice", SUBJECT_REF])
def test_persistence_paths_reject_unsafe_purge_id(storage, purge_id):
    now = 1
    storage.conn.execute(
        """INSERT INTO purge_operations (
               purge_id, org_id, operation_type, scope_type, subject_ref, request_ref,
               idempotency_key, status, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)""",
        (
            purge_id,
            "org1",
            "user_erasure",
            "user",
            SUBJECT_REF,
            REQUEST_REF,
            "idem_seeded_invalid_purge_id",
            now,
            now,
        ),
    )
    storage.conn.commit()

    with pytest.raises(ValueError, match="purge_id"):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="request",
            target_ref="all",
            phase="delete",
            status="running",
        )

    with pytest.raises(ValueError, match="purge_id"):
        storage.complete_purge_operation_with_audit(
            purge_id,
            AuditEvent(
                org_id="org1",
                operation="ERASE",
                entity_type="request",
                subject_ref=SUBJECT_REF,
                request_ref=REQUEST_REF,
                idempotency_key=purge_id,
            ),
        )

    with pytest.raises(ValueError, match="purge_id"):
        storage.list_purge_targets(purge_id)
    assert storage.list_audit_events(subject_ref=SUBJECT_REF) == []


def test_apply_governance_user_data_delete_rejects_unsafe_purge_id_before_side_effects(
    storage,
):
    user_id = "user-delete-seed"
    _seed_user_scoped_rows(storage, user_id=user_id)

    with pytest.raises(ValueError, match="purge_id"):
        storage.apply_governance_user_data_delete(
            purge_id="alice@example.com",
            user_id=user_id,
        )

    remaining = {
        "requests": storage.conn.execute(
            "SELECT COUNT(*) FROM requests WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0],
        "interactions": storage.conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0],
        "profiles": storage.conn.execute(
            "SELECT COUNT(*) FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0],
        "user_playbooks": storage.conn.execute(
            "SELECT COUNT(*) FROM user_playbooks WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0],
    }
    assert remaining == {
        "requests": 1,
        "interactions": 1,
        "profiles": 1,
        "user_playbooks": 1,
    }


def test_apply_governance_user_data_delete_rejects_unexpected_target_name_from_internal_counts(
    storage, monkeypatch
):
    purge_id = _begin_purge(storage, "purge_internal_target_name")

    def _stub_clear_user_data(self: SQLiteStorage, user_id: str) -> dict[str, int]:
        del self, user_id
        return {"requests": 1, "surprise_target": 2}

    monkeypatch.setattr(
        SQLiteStorage,
        "clear_user_data",
        _stub_clear_user_data,
    )

    with pytest.raises(ValueError, match="target_name"):
        storage.apply_governance_user_data_delete(
            purge_id=purge_id,
            user_id="user-delete-seed",
        )

    delete_targets = storage.list_purge_targets(purge_id, phase="delete")
    assert all(target.target_name != "surprise_target" for target in delete_targets)


def test_apply_governance_agent_playbook_rebuild_rejects_unsafe_purge_id_before_side_effects(
    storage,
):
    agent_playbook_id = _seed_agent_playbook(storage)

    before_row = storage.conn.execute(
        """SELECT content, trigger, rationale, status, tags
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    before_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)

    with pytest.raises(ValueError, match="purge_id"):
        storage.apply_governance_agent_playbook_rebuild(
            purge_id="request_12345",
            agent_playbook_id=agent_playbook_id,
            remaining_source_windows=[{"user_playbook_id": 99, "source_interaction_ids": [202]}],
            content="updated content",
            trigger="updated trigger",
            rationale="updated rationale",
            blocking_issue=None,
            expanded_terms="updated terms",
            tags=["updated"],
        )

    after_row = storage.conn.execute(
        """SELECT content, trigger, rationale, status, tags
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    after_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)
    assert tuple(before_row) == tuple(after_row)
    assert before_windows == after_windows


def test_fail_purge_operation_rejects_unsafe_purge_id_before_side_effects(storage):
    purge_id = _begin_purge(storage, "purge_fail_unsafe_id")

    with pytest.raises(ValueError, match="purge_id"):
        storage.fail_purge_operation(
            SUBJECT_REF,
            "governance.error",
            "detail.code",
        )

    failed = storage.get_purge_operation(purge_id)
    assert failed.status == "running"
    assert failed.error_code is None
    assert failed.error_detail is None


@pytest.mark.parametrize(
    ("target_name", "phase", "status", "match"),
    [
        pytest.param(cast(Any, "session"), "delete", "running", "target_name", id="target-name"),
        pytest.param("request", cast(Any, "archive"), "running", "phase", id="phase"),
        pytest.param("request", "delete", cast(Any, "done"), "status", id="status"),
    ],
)
def test_record_purge_target_rejects_invalid_enum_values(
    storage, target_name, phase, status, match
):
    purge_id = _begin_purge(storage, "purge_target_invalid_enum")

    with pytest.raises(ValueError, match=match):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name=target_name,
            target_ref="all",
            phase=phase,
            status=status,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        pytest.param("subject_ref", "subref_v1_alice@example.com", id="subject-email"),
        pytest.param("request_ref", "reqref_v1_request_123", id="request-like"),
        pytest.param("request_ref", "reqref_v1_target", id="request-placeholder"),
    ],
)
def test_append_audit_event_rejects_prefix_only_refs(storage, field_name, value):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="export_2",
    ).model_copy(update={field_name: value})

    with pytest.raises(ValueError, match=field_name):
        storage.append_audit_event(event)


@pytest.mark.parametrize(
    "idempotency_key",
    [
        "alice@example.com",
        "request_123",
        "reqref_v1_target",
        "alice",
        "user_123",
        "subject_42",
        "actor.alpha",
    ],
)
def test_governance_persistence_rejects_unsafe_idempotency_keys(storage, idempotency_key):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key=idempotency_key,
    )
    with pytest.raises(ValueError, match="idempotency_key"):
        storage.append_audit_event(event)

    with pytest.raises(ValueError, match="idempotency_key"):
        storage.begin_purge_operation(
            purge_id="purge_unsafe_idem",
            idempotency_key=idempotency_key,
            operation_type="user_erasure",
            scope_type="user",
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
        )


def test_append_audit_event_rejects_user_like_entity_id(storage):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        entity_id="alice",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="export_3",
    )

    with pytest.raises(ValueError, match="entity_id"):
        storage.append_audit_event(event)


@pytest.mark.parametrize(
    "entity_id",
    ["user_123", "subject_42", "actor.alpha"],
)
def test_append_audit_event_rejects_identifier_like_entity_id(storage, entity_id):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        entity_id=entity_id,
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="export_identifier_like_entity",
    )

    with pytest.raises(ValueError, match="entity_id"):
        storage.append_audit_event(event)


@pytest.mark.parametrize(
    "detail",
    [
        {"status": "alice"},
        {"route": "alice"},
        {"status": "user_123"},
        {"route": "subject_42"},
        {"status": "actor.alpha"},
    ],
)
def test_governance_detail_rejects_user_like_status_and_route(storage, detail):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="export_4",
        detail=detail,
    )

    with pytest.raises(ValueError, match="status|route"):
        storage.append_audit_event(event)


@pytest.mark.parametrize(
    "purge_id",
    ["purge_user_123", "purge_subject_42", "purge_actor_alpha"],
)
def test_begin_purge_operation_rejects_identifier_like_purge_suffix(storage, purge_id):
    with pytest.raises(ValueError, match="purge_id"):
        storage.begin_purge_operation(
            purge_id=purge_id,
            idempotency_key="idem_purge_identifier_suffix",
            operation_type="user_erasure",
            scope_type="user",
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
        )


def test_append_audit_event_canonicalizes_detail_keys_before_persistence(storage):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="export_canonical_detail",
        detail={" Deleted_Counts ": {"requests": 1}},
    )

    storage.append_audit_event(event)

    rows = storage.list_audit_events(subject_ref=SUBJECT_REF)
    assert rows[-1].detail == {"deleted_counts": {"requests": 1}}


def test_record_purge_target_canonicalizes_detail_keys_before_persistence(storage):
    purge_id = _begin_purge(storage, "purge_canonical_detail")

    storage.record_purge_target(
        purge_id=purge_id,
        target_name="request",
        target_ref="all",
        phase="delete",
        status="complete",
        detail={" Deleted_Counts ": {"requests": 2}},
        deleted_count=2,
    )

    rows = storage.list_purge_targets(purge_id, phase="delete")
    assert len(rows) == 1
    assert rows[0].detail == {"deleted_counts": {"requests": 2}}


@pytest.mark.parametrize("persistence_path", ["audit_event", "purge_target"])
def test_governance_detail_rejects_duplicate_normalized_keys(storage, persistence_path):
    detail = {"status": "complete", " Status ": "complete"}

    if persistence_path == "audit_event":
        event = AuditEvent(
            org_id="org1",
            operation="EXPORT",
            entity_type="request",
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
            idempotency_key="export_duplicate_detail_key",
            detail=detail,
        )
        with pytest.raises(ValueError, match="duplicate key status"):
            storage.append_audit_event(event)
        return

    purge_id = _begin_purge(storage, "purge_duplicate_detail_key")
    with pytest.raises(ValueError, match="duplicate key status"):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="request",
            target_ref="all",
            phase="delete",
            status="complete",
            detail=detail,
        )


@pytest.mark.parametrize(
    ("target_name", "phase", "target_ref", "match"),
    [
        pytest.param(
            "target_snapshot",
            "prepare_targets",
            "all",
            None,
            id="snapshot-marker-all",
        ),
        pytest.param("request", "delete", "all", None, id="aggregate-delete-all"),
        pytest.param(
            "agent_playbook",
            "hide_for_rebuild",
            "17",
            None,
            id="row-target-hide-internal-id",
        ),
        pytest.param(
            "agent_playbook",
            "rebuild_without_erased_sources",
            "19",
            None,
            id="row-target-rebuild-internal-id",
        ),
        pytest.param(
            "agent_playbook",
            "hide_for_rebuild",
            "all",
            "target_ref",
            id="row-target-hide-rejects-all",
        ),
        pytest.param(
            "agent_playbook",
            "rebuild_without_erased_sources",
            "all",
            "target_ref",
            id="row-target-rebuild-rejects-all",
        ),
        pytest.param(
            "agent_playbook",
            "rebuild_without_erased_sources",
            REQUEST_REF,
            "target_ref",
            id="row-target-rebuild-rejects-minimized-ref",
        ),
    ],
)
def test_record_purge_target_validates_target_ref_by_phase_and_name(
    storage, target_name, phase, target_ref, match
):
    purge_id = _begin_purge(storage, "purge_target_ref_phase_specific")

    if match is None:
        storage.record_purge_target(
            purge_id=purge_id,
            target_name=target_name,
            target_ref=target_ref,
            phase=phase,
            status="running",
        )
        return

    with pytest.raises(ValueError, match=match):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name=target_name,
            target_ref=target_ref,
            phase=phase,
            status="running",
        )


def test_init_governance_tables_upgrades_legacy_purge_target_table(tmp_path):
    db_path = tmp_path / "legacy-governance.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE purge_operations (
            org_id TEXT NOT NULL,
            purge_id TEXT NOT NULL,
            operation_type TEXT NOT NULL,
            scope_type TEXT NOT NULL,
            subject_ref TEXT,
            request_ref TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_code TEXT,
            error_detail TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            completed_at INTEGER,
            PRIMARY KEY (org_id, purge_id)
        );
        CREATE TABLE purge_operation_targets (
            purge_id TEXT NOT NULL,
            target_name TEXT NOT NULL,
            target_ref TEXT NOT NULL DEFAULT '',
            phase TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            detail TEXT,
            deleted_count INTEGER NOT NULL DEFAULT 0,
            error_detail TEXT,
            started_at INTEGER,
            completed_at INTEGER,
            PRIMARY KEY (purge_id, target_name, target_ref, phase)
        );
        """
    )
    conn.commit()

    init_governance_tables(conn)

    target_columns = {
        row[1]: {"pk": row[5], "notnull": row[3]}
        for row in conn.execute("PRAGMA table_info(purge_operation_targets)")
    }
    assert "org_id" in target_columns
    assert target_columns["org_id"]["pk"] == 1
    assert target_columns["org_id"]["notnull"] == 1

    index_names = {
        row[1] for row in conn.execute("PRAGMA index_list(purge_operation_targets)")
    }
    assert "idx_purge_targets_purge_phase" in index_names
    conn.close()

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(org_id="org1", db_path=str(db_path))

    purge_id = storage.begin_purge_operation(
        purge_id="purge_legacy_upgrade",
        idempotency_key="idem_legacy_upgrade",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    ).purge_id
    storage.record_purge_target(
        purge_id=purge_id,
        target_name="target_snapshot",
        target_ref="all",
        phase="prepare_targets",
        status="complete",
    )

    assert storage.purge_targets_prepared(purge_id) is True


def test_init_governance_tables_skips_ambiguous_legacy_purge_target_rows(tmp_path):
    db_path = tmp_path / "legacy-governance-ambiguous.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE purge_operations (
            org_id TEXT NOT NULL,
            purge_id TEXT NOT NULL,
            operation_type TEXT NOT NULL,
            scope_type TEXT NOT NULL,
            subject_ref TEXT,
            request_ref TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_code TEXT,
            error_detail TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            completed_at INTEGER,
            PRIMARY KEY (org_id, purge_id)
        );
        INSERT INTO purge_operations (
            org_id, purge_id, operation_type, scope_type, subject_ref, request_ref,
            idempotency_key, status, created_at, updated_at
        ) VALUES
            ('org1', 'purge_shared', 'user_erasure', 'user', NULL, 'reqref_v1_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'idem_org1', 'pending', 1, 1),
            ('org2', 'purge_shared', 'user_erasure', 'user', NULL, 'reqref_v1_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'idem_org2', 'pending', 1, 1),
            ('org1', 'purge_unique', 'user_erasure', 'user', NULL, 'reqref_v1_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'idem_unique', 'pending', 1, 1);
        CREATE TABLE purge_operation_targets (
            purge_id TEXT NOT NULL,
            target_name TEXT NOT NULL,
            target_ref TEXT NOT NULL DEFAULT '',
            phase TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            detail TEXT,
            deleted_count INTEGER NOT NULL DEFAULT 0,
            error_detail TEXT,
            started_at INTEGER,
            completed_at INTEGER,
            PRIMARY KEY (purge_id, target_name, target_ref, phase)
        );
        INSERT INTO purge_operation_targets (
            purge_id, target_name, target_ref, phase, status
        ) VALUES
            ('purge_shared', 'target_snapshot', 'all', 'prepare_targets', 'complete'),
            ('purge_unique', 'target_snapshot', 'all', 'prepare_targets', 'complete');
        """
    )
    conn.commit()

    init_governance_tables(conn)

    upgraded_rows = conn.execute(
        """
        SELECT org_id, purge_id, target_name, target_ref, phase, status
        FROM purge_operation_targets
        ORDER BY purge_id, org_id
        """
    ).fetchall()
    conn.close()

    assert upgraded_rows == [
        ("org1", "purge_unique", "target_snapshot", "all", "prepare_targets", "complete")
    ]


def test_gc_governance_retention_accepts_config_keyword(storage):
    assert storage.gc_governance_retention(config=object()) == 0
