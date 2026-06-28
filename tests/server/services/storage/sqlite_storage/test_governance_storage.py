from __future__ import annotations

import json
import sqlite3
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
    AgentSuccessEvaluationResult,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.domain.governance import (
    AuditEvent,
    AuditOperation,
    AuditStatus,
)
from reflexio.models.api_schema.retriever_schema import SearchAgentPlaybookRequest
from reflexio.models.config_schema import GovernanceRetentionConfig
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage import (
    _governance as governance_module,
)
from reflexio.server.services.storage.sqlite_storage._governance import (
    init_governance_tables,
)

pytestmark = pytest.mark.integration

SUBJECT_REF = "subref_v1_" + "a" * 32
OTHER_SUBJECT_REF = "subref_v1_" + "c" * 32
REQUEST_REF = "reqref_v1_" + "b" * 32
OTHER_REQUEST_REF = "reqref_v1_" + "d" * 32
ACTOR_REF = "actref_v1_" + "e" * 32
CANONICAL_DELETE_TARGET_NAMES = (
    "request",
    "interaction",
    "profile",
    "user_playbook",
    "agent_success_evaluation_result",
    "profile_purge",
    "user_playbook_purge",
)


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


def _add_complete_delete_target_matrix(storage: SQLiteStorage, purge_id: str) -> None:
    for target_name in CANONICAL_DELETE_TARGET_NAMES:
        storage.record_purge_target(
            purge_id=purge_id,
            target_name=target_name,
            target_ref="all",
            phase="delete",
            status="complete",
        )


def _begin_completeable_purge(storage: SQLiteStorage, purge_id: str) -> str:
    purge_id = _begin_purge(storage, purge_id)
    _add_complete_delete_target_matrix(storage, purge_id)
    return purge_id


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
        (
            "profile_seed",
            user_id,
            "profile-content",
            1,
            "request_seed",
            4102444800,
            created_at,
        ),
    )
    storage.conn.execute(
        """INSERT INTO user_playbooks (
               user_id, playbook_name, created_at, request_id, agent_version,
               content, source_interaction_ids, embedding
           ) VALUES (?, '', ?, ?, '', ?, '[]', '[]')""",
        (user_id, created_at, "request_seed", "playbook-content"),
    )
    storage.conn.commit()


def _user_scoped_row_counts(storage: SQLiteStorage, *, user_id: str) -> dict[str, int]:
    return {
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


def _seed_prepare_counts_user_data(storage: SQLiteStorage, *, user_id: str) -> set[int]:
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
        (
            "profile_seed",
            user_id,
            "profile-content",
            1,
            "request_seed",
            4102444800,
            created_at,
        ),
    )
    storage.conn.execute(
        """INSERT INTO profiles (
               profile_id, user_id, content, last_modified_timestamp,
               generated_from_request_id, profile_time_to_live, expiration_timestamp,
               embedding, source_interaction_ids, merged_into, created_at
           ) VALUES (?, ?, ?, ?, ?, 'infinity', ?, '[]', '[]', ?, ?)""",
        (
            "profile_purge_seed",
            user_id,
            "profile-purge-content",
            2,
            "request_seed",
            4102444800,
            "profile_external_survivor",
            created_at,
        ),
    )
    delete_cursor = storage.conn.execute(
        """INSERT INTO user_playbooks (
               user_id, playbook_name, created_at, request_id, agent_version,
               content, source_interaction_ids, embedding
           ) VALUES (?, '', ?, ?, '', ?, '[]', '[]')""",
        (user_id, created_at, "request_seed", "playbook-delete-content"),
    )
    assert delete_cursor.lastrowid is not None
    delete_playbook_id = int(delete_cursor.lastrowid)
    purge_cursor = storage.conn.execute(
        """INSERT INTO user_playbooks (
               user_id, playbook_name, created_at, request_id, agent_version,
               content, source_interaction_ids, embedding, merged_into
           ) VALUES (?, '', ?, ?, '', ?, '[]', '[]', ?)""",
        (
            user_id,
            created_at,
            "request_seed",
            "playbook-purge-content",
            999999,
        ),
    )
    assert purge_cursor.lastrowid is not None
    purge_playbook_id = int(purge_cursor.lastrowid)
    storage.conn.commit()
    return {delete_playbook_id, purge_playbook_id}


def _seed_eval_result(
    storage: SQLiteStorage,
    *,
    user_id: str,
    session_id: str,
    evaluation_name: str,
    agent_version: str = "agent-v1",
) -> None:
    storage.save_agent_success_evaluation_results(
        [
            AgentSuccessEvaluationResult(
                user_id=user_id,
                session_id=session_id,
                evaluation_name=evaluation_name,
                agent_version=agent_version,
                is_success=True,
            )
        ]
    )


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
        or [
            AgentPlaybookSourceWindow(user_playbook_id=7, source_interaction_ids=[101])
        ],
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


def test_init_governance_tables_backfills_legacy_null_audit_request_ref(tmp_path):
    db_path = tmp_path / "legacy-audit.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE audit_events (
               event_id INTEGER PRIMARY KEY AUTOINCREMENT,
               org_id TEXT NOT NULL,
               actor_type TEXT NOT NULL DEFAULT 'system',
               actor_ref TEXT,
               operation TEXT NOT NULL,
               entity_type TEXT NOT NULL,
               entity_id TEXT,
               subject_ref TEXT,
               request_ref TEXT,
               idempotency_key TEXT,
               status TEXT NOT NULL DEFAULT 'ok',
               detail TEXT,
               created_at INTEGER NOT NULL
           )"""
    )
    conn.execute(
        """INSERT INTO audit_events (
               org_id, actor_type, operation, entity_type, subject_ref,
               request_ref, status, created_at
           ) VALUES (?, 'system', 'EXPORT', 'request', ?, NULL, 'ok', 1)""",
        ("org1", SUBJECT_REF),
    )
    conn.commit()
    conn.close()

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(org_id="org1", db_path=str(db_path))

    events = storage.list_audit_events(subject_ref=SUBJECT_REF)
    assert len(events) == 1
    assert events[0].request_ref == "reqref_v1_legacy_unknown"
    with pytest.raises(sqlite3.IntegrityError, match="request_ref"):
        storage.conn.execute(
            """INSERT INTO audit_events (
                   org_id, actor_type, operation, entity_type, request_ref, status, created_at
               ) VALUES ('org1', 'system', 'EXPORT', 'request', NULL, 'ok', 2)"""
        )


def test_append_audit_event_rejects_numeric_idempotency_key(storage):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="12345",
    )

    with pytest.raises(ValueError, match="idempotency_key"):
        storage.append_audit_event(event)

    assert storage.list_audit_events(subject_ref=SUBJECT_REF) == []


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


def test_list_audit_events_rejects_cross_org_override(storage_factory):
    storage_org1 = storage_factory("org1")
    storage_org2 = storage_factory("org2")

    org1_event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="audit_scope_org1",
        detail={"count": 1},
    )
    org2_event = AuditEvent(
        org_id="org2",
        operation="EXPORT",
        entity_type="request",
        subject_ref=OTHER_SUBJECT_REF,
        request_ref=OTHER_REQUEST_REF,
        idempotency_key="audit_scope_org2",
        detail={"count": 2},
    )

    assert storage_org1.append_audit_event(org1_event) is True
    assert storage_org2.append_audit_event(org2_event) is True

    org1_rows = storage_org1.list_audit_events()

    assert [row.idempotency_key for row in org1_rows] == ["audit_scope_org1"]
    with pytest.raises(ValueError, match="org_id"):
        storage_org1.list_audit_events(org_id="org2")


def test_purge_targets_require_snapshot_marker(storage):
    purge = storage.begin_purge_operation(
        purge_id="purge_snapshot_marker",
        idempotency_key="idem_snapshot_marker",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    storage.record_purge_target(
        purge_id=purge.purge_id,
        target_name="request",
        target_ref="all",
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
    purge_id = _begin_completeable_purge(storage, "purge_atomic_success")
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


def test_complete_purge_operation_with_audit_begins_immediate_transaction_before_reads(
    storage,
):
    purge_id = _begin_completeable_purge(storage, "purge_begin_immediate")
    statements: list[str] = []
    storage.conn.set_trace_callback(statements.append)
    try:
        storage.complete_purge_operation_with_audit(
            purge_id,
            _erase_event(purge_id=purge_id),
        )
    finally:
        storage.conn.set_trace_callback(None)

    begin_index = next(
        i for i, statement in enumerate(statements) if statement == "BEGIN IMMEDIATE"
    )
    first_validation_read_index = next(
        i
        for i, statement in enumerate(statements)
        if "SELECT * FROM purge_operations" in statement
    )
    assert begin_index < first_validation_read_index


def test_complete_purge_operation_with_audit_accepts_planned_success_detail(storage):
    purge_id = _begin_completeable_purge(storage, "purge_success_detail")
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
    ("event_kwargs", "match"),
    [
        pytest.param(
            {"subject_ref": OTHER_SUBJECT_REF}, "subject_ref", id="subject-ref"
        ),
        pytest.param(
            {"request_ref": OTHER_REQUEST_REF}, "request_ref", id="request-ref"
        ),
    ],
)
def test_complete_purge_operation_rejects_audit_refs_that_mismatch_persisted_purge(
    storage, event_kwargs, match
):
    purge_id = _begin_completeable_purge(storage, "purge_row_ref_mismatch")
    event = _erase_event(purge_id=purge_id).model_copy(update=event_kwargs)

    with pytest.raises(ValueError, match=match):
        storage.complete_purge_operation_with_audit(purge_id, event)

    assert storage.get_purge_operation(purge_id).status == "running"
    assert storage.list_audit_events(subject_ref=SUBJECT_REF) == []
    if event.subject_ref is not None:
        assert storage.list_audit_events(subject_ref=event.subject_ref) == []


@pytest.mark.parametrize(
    ("retry_kwargs", "match"),
    [
        pytest.param(
            {"purge_id": "purge_begin_retry_other"}, "purge_id", id="purge-id"
        ),
        pytest.param(
            {"request_ref": OTHER_REQUEST_REF}, "request_ref", id="request-ref"
        ),
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


def test_begin_purge_operation_rejects_numeric_idempotency_key(storage):
    with pytest.raises(ValueError, match="idempotency_key"):
        storage.begin_purge_operation(
            purge_id="purge_numeric_idem",
            idempotency_key="12345",
            operation_type="user_erasure",
            scope_type="user",
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
        )

    with pytest.raises(ValueError, match="not found"):
        storage.get_purge_operation("purge_numeric_idem")


def test_begin_purge_operation_accepts_code_shaped_idempotency_key_with_content(
    storage,
):
    purge = storage.begin_purge_operation(
        purge_id="purge_content_retry",
        idempotency_key="content_purge_retry_1",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )

    assert purge.idempotency_key == "content_purge_retry_1"


@pytest.mark.parametrize("purge_id", ["purge_1", "purge_123"])
def test_begin_purge_operation_rejects_raw_numeric_purge_suffix(storage, purge_id):
    with pytest.raises(ValueError, match="purge_id"):
        storage.begin_purge_operation(
            purge_id=purge_id,
            idempotency_key=f"idem_{purge_id}",
            operation_type="user_erasure",
            scope_type="user",
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
        )

    with pytest.raises(ValueError, match="purge_id"):
        storage.get_purge_operation(purge_id)


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
    purge_id = _begin_completeable_purge(storage, "purge_invalid")

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
    purge_id = _begin_completeable_purge(storage, "purge_seeded")
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
        pytest.param(
            "subject_ref", {"subject_ref": OTHER_SUBJECT_REF}, id="subject-ref"
        ),
        pytest.param(
            "request_ref", {"request_ref": OTHER_REQUEST_REF}, id="request-ref"
        ),
        pytest.param("actor_type", {"actor_type": "jwt"}, id="actor-type"),
        pytest.param("actor_ref", {"actor_ref": ACTOR_REF}, id="actor-ref"),
        pytest.param("entity_id", {"entity_id": "17"}, id="entity-id"),
        pytest.param("detail", {"detail": {"count": 2}}, id="detail"),
    ],
)
def test_complete_purge_operation_rejects_mismatched_existing_erase_row(
    storage, field_name, seed_kwargs
):
    purge_id = _begin_completeable_purge(storage, "purge_seeded_mismatch")
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
            json.dumps(seeded_event.detail)
            if seeded_event.detail is not None
            else None,
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


def test_append_audit_event_rejects_successful_erase_without_idempotency_key(storage):
    with pytest.raises(ValueError, match="Successful ERASE audit rows"):
        storage.append_audit_event(
            AuditEvent(
                org_id="org1",
                operation="ERASE",
                entity_type="request",
                subject_ref=SUBJECT_REF,
                request_ref=REQUEST_REF,
                idempotency_key=None,
                status="ok",
            )
        )


def test_complete_purge_operation_requires_full_delete_target_matrix(storage):
    purge_id = _begin_purge(storage, "purge_snapshot_only")

    with pytest.raises(ValueError, match="delete target matrix"):
        storage.complete_purge_operation_with_audit(
            purge_id,
            _erase_event(purge_id=purge_id),
        )

    assert storage.get_purge_operation(purge_id).status == "running"
    assert storage.list_audit_events(subject_ref=SUBJECT_REF) == []


def test_complete_retry_replaces_failed_completed_at(storage):
    purge_id = _begin_completeable_purge(storage, "purge_retry_completion_time")
    with patch.object(governance_module, "_epoch_now", return_value=111):
        failed = storage.fail_purge_operation(
            purge_id,
            error_code="governance_erase_failed",
            error_detail="RuntimeError",
        )
    assert failed.completed_at == 111

    with patch.object(governance_module, "_epoch_now", return_value=222):
        completed = storage.complete_purge_operation_with_audit(
            purge_id,
            _erase_event(purge_id=purge_id),
        )

    assert completed.status == "complete"
    assert completed.completed_at == 222


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
        for target in storage.list_purge_targets(
            "purge_detail", phase="prepare_targets"
        )
        if target.target_name == "target_snapshot"
    )
    assert snapshot.detail == {
        "owned_user_playbook_ids": [7],
        "affected_agent_playbook_ids": [],
    }


def test_apply_governance_user_data_delete_rejects_playbook_snapshot_drift(storage):
    user_id = "user-snapshot-drift"
    owned_user_playbook_ids = _seed_prepare_counts_user_data(storage, user_id=user_id)
    storage.begin_purge_operation(
        purge_id="purge_snapshot_drift",
        idempotency_key="idem_purge_snapshot_drift",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    storage.prepare_governance_erase_targets(
        purge_id="purge_snapshot_drift",
        user_id=user_id,
    )
    storage.conn.execute(
        """INSERT INTO user_playbooks (
               user_id, playbook_name, created_at, request_id, agent_version,
               content, source_interaction_ids, embedding
           ) VALUES (?, '', ?, ?, '', ?, '[]', '[]')""",
        (
            user_id,
            "2026-01-01T00:00:00.000Z",
            "request_seed",
            "late-playbook-content",
        ),
    )
    storage.conn.commit()

    with pytest.raises(ValueError, match="prepared purge snapshot"):
        storage.apply_governance_user_data_delete("purge_snapshot_drift", user_id)

    remaining_ids = {
        int(row["user_playbook_id"])
        for row in storage.conn.execute(
            "SELECT user_playbook_id FROM user_playbooks WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    }
    assert owned_user_playbook_ids < remaining_ids


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
            AgentPlaybookSourceWindow(
                user_playbook_id=7, source_interaction_ids=[101, 102]
            ),
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
        "previous_lifecycle_status": Status.ARCHIVED.value,
        "remaining_source_windows": [
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
    }


def test_prepare_governance_erase_targets_records_full_delete_matrix_counts(storage):
    purge_id = "purge_prepare_counts"
    user_id = "user_prepare_counts"
    owned_user_playbook_ids = _seed_prepare_counts_user_data(storage, user_id=user_id)
    _seed_eval_result(
        storage,
        user_id=user_id,
        session_id="session_seed",
        evaluation_name="governance_prepare_counts",
    )
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_prepare_counts",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )

    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id=user_id,
        owned_user_playbook_ids=owned_user_playbook_ids,
    )

    delete_targets = {
        target.target_name: target
        for target in storage.list_purge_targets(purge_id, phase="delete")
    }
    assert delete_targets["request"].target_ref == "all"
    assert delete_targets["request"].detail == {"count": 1}
    assert delete_targets["interaction"].target_ref == "all"
    assert delete_targets["interaction"].detail == {"count": 1}
    assert delete_targets["profile"].target_ref == "all"
    assert delete_targets["profile"].detail == {"count": 1}
    assert delete_targets["profile_purge"].target_ref == "all"
    assert delete_targets["profile_purge"].detail == {"count": 1}
    assert delete_targets["user_playbook"].target_ref == "all"
    assert delete_targets["user_playbook"].detail == {"count": 1}
    assert delete_targets["agent_success_evaluation_result"].target_ref == "all"
    assert delete_targets["agent_success_evaluation_result"].detail == {"count": 1}
    assert delete_targets["user_playbook_purge"].target_ref == "all"
    assert delete_targets["user_playbook_purge"].detail == {"count": 1}

    counts = storage.clear_user_data(user_id)
    assert counts == {
        "interactions": 1,
        "user_playbooks": 1,
        "profiles": 1,
        "requests": 1,
        "purged_profiles": 1,
        "purged_user_playbooks": 1,
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
        "previous_lifecycle_status": None,
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
            "previous_lifecycle_status": Status.ARCHIVED.value,
            "remaining_source_windows": [
                {"user_playbook_id": 9, "source_interaction_ids": [201]},
            ],
        },
    )
    storage.record_purge_target(
        purge_id=purge_id,
        target_name="agent_playbook",
        target_ref=str(agent_playbook_id),
        phase="hide_for_rebuild",
        status="complete",
    )
    expected_detail = {
        "original_source_windows": [
            {"user_playbook_id": 7, "source_interaction_ids": [101]},
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
        "previous_lifecycle_status": Status.ARCHIVED.value,
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
    rebuilt_row = storage.conn.execute(
        """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    assert rebuilt_row is not None
    assert tuple(rebuilt_row) == (
        "rebuilt content",
        "rebuilt trigger",
        "rebuilt rationale",
        None,
        "rebuilt terms",
        json.dumps(["rebuilt"]),
        Status.ARCHIVED.value,
    )
    assert storage.get_source_windows_for_agent_playbook(agent_playbook_id) == [
        AgentPlaybookSourceWindow(user_playbook_id=9, source_interaction_ids=[201])
    ]


def test_apply_governance_agent_playbook_rebuild_rejects_ad_hoc_rebuild_without_prepared_target(
    storage,
):
    purge_id = "purge_rebuild_requires_target"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_rebuild_requires_target",
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
    original_row = storage.conn.execute(
        """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    assert original_row is not None
    original_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)

    with pytest.raises(ValueError, match="planned rebuild target does not exist"):
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

    assert (
        storage.conn.execute(
            """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
               FROM agent_playbooks
               WHERE agent_playbook_id = ?""",
            (agent_playbook_id,),
        ).fetchone()
        == original_row
    )
    assert (
        storage.get_source_windows_for_agent_playbook(agent_playbook_id)
        == original_windows
    )
    assert (
        storage.list_purge_targets(purge_id, phase="rebuild_without_erased_sources")
        == []
    )


def test_apply_governance_agent_playbook_rebuild_rejects_rebuild_before_hide_phase_complete(
    storage,
):
    purge_id = "purge_rebuild_requires_hide"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_rebuild_requires_hide",
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
            "previous_lifecycle_status": Status.ARCHIVE_IN_PROGRESS.value,
            "remaining_source_windows": [
                {"user_playbook_id": 9, "source_interaction_ids": [201]},
            ],
        },
    )
    original_row = storage.conn.execute(
        """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    assert original_row is not None
    original_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)

    with pytest.raises(ValueError, match="hide_for_rebuild target must be complete"):
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

    assert (
        storage.conn.execute(
            """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
               FROM agent_playbooks
               WHERE agent_playbook_id = ?""",
            (agent_playbook_id,),
        ).fetchone()
        == original_row
    )
    assert (
        storage.get_source_windows_for_agent_playbook(agent_playbook_id)
        == original_windows
    )
    rebuild_target = next(
        target
        for target in storage.list_purge_targets(
            purge_id, phase="rebuild_without_erased_sources"
        )
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    assert rebuild_target.status == "running"
    assert rebuild_target.detail == {
        "original_source_windows": [
            {"user_playbook_id": 7, "source_interaction_ids": [101]},
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
        "previous_lifecycle_status": Status.ARCHIVE_IN_PROGRESS.value,
        "remaining_source_windows": [
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
    }


def test_apply_governance_agent_playbook_rebuild_succeeds_after_prepare_and_hide(
    storage,
):
    purge_id = "purge_rebuild_prepare_hide"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_rebuild_prepare_hide",
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
    storage.hide_governance_agent_playbooks_for_rebuild(purge_id)

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
    assert rebuild_target.detail == {
        "original_source_windows": [
            {"user_playbook_id": 7, "source_interaction_ids": [101]},
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
        "previous_lifecycle_status": None,
        "remaining_source_windows": [
            {"user_playbook_id": 9, "source_interaction_ids": [201]},
        ],
    }
    assert storage.get_source_windows_for_agent_playbook(agent_playbook_id) == [
        AgentPlaybookSourceWindow(user_playbook_id=9, source_interaction_ids=[201])
    ]


def test_apply_governance_agent_playbook_rebuild_does_not_complete_target_when_search_refresh_fails(
    storage, monkeypatch
):
    purge_id = "purge_rebuild_search_refresh_failure"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_rebuild_search_refresh_failure",
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
            "previous_lifecycle_status": Status.ARCHIVE_IN_PROGRESS.value,
            "remaining_source_windows": [
                {"user_playbook_id": 9, "source_interaction_ids": [201]},
            ],
        },
    )
    storage.record_purge_target(
        purge_id=purge_id,
        target_name="agent_playbook",
        target_ref=str(agent_playbook_id),
        phase="hide_for_rebuild",
        status="complete",
    )
    original_row = storage.conn.execute(
        """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    assert original_row is not None
    original_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)
    original_fts_row = storage.conn.execute(
        "SELECT search_text FROM agent_playbooks_fts WHERE rowid = ?",
        (agent_playbook_id,),
    ).fetchone()
    assert original_fts_row is not None

    def fail_search_refresh(*args, **kwargs):
        raise RuntimeError("search refresh failed")

    monkeypatch.setattr(
        storage,
        "_upsert_agent_playbook_search_rows_locked",
        fail_search_refresh,
    )

    with pytest.raises(RuntimeError, match="search refresh failed"):
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

    assert (
        storage.conn.execute(
            """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
               FROM agent_playbooks
               WHERE agent_playbook_id = ?""",
            (agent_playbook_id,),
        ).fetchone()
        == original_row
    )
    assert (
        storage.get_source_windows_for_agent_playbook(agent_playbook_id)
        == original_windows
    )
    assert (
        storage.conn.execute(
            "SELECT search_text FROM agent_playbooks_fts WHERE rowid = ?",
            (agent_playbook_id,),
        ).fetchone()
        == original_fts_row
    )
    rebuild_target = next(
        target
        for target in storage.list_purge_targets(
            purge_id, phase="rebuild_without_erased_sources"
        )
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    assert rebuild_target.status == "running"


def test_apply_governance_agent_playbook_rebuild_removes_orphaned_aggregate_when_no_sources_remain(
    storage,
):
    purge_id = "purge_rebuild_remove_orphan"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_rebuild_remove_orphan",
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
            ],
            "previous_lifecycle_status": Status.ARCHIVED.value,
            "remaining_source_windows": [],
        },
    )
    storage.record_purge_target(
        purge_id=purge_id,
        target_name="agent_playbook",
        target_ref=str(agent_playbook_id),
        phase="hide_for_rebuild",
        status="complete",
    )

    storage.apply_governance_agent_playbook_rebuild(
        purge_id=purge_id,
        agent_playbook_id=agent_playbook_id,
        remaining_source_windows=[],
        content=None,
        trigger=None,
        rationale=None,
        blocking_issue=None,
        expanded_terms=None,
        tags=None,
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
    assert storage.get_agent_playbook_by_id(agent_playbook_id) is None
    assert (
        storage.get_agent_playbook_by_id(
            agent_playbook_id,
            include_tombstones=True,
        )
        is None
    )
    assert storage.get_source_windows_for_agent_playbook(agent_playbook_id) == []
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM agent_playbooks WHERE agent_playbook_id = ?",
            (agent_playbook_id,),
        ).fetchone()[0]
        == 0
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) FROM agent_playbooks_fts WHERE rowid = ?",
            (agent_playbook_id,),
        ).fetchone()[0]
        == 0
    )
    if storage._has_sqlite_vec:
        assert (
            storage.conn.execute(
                "SELECT COUNT(*) FROM agent_playbooks_vec WHERE rowid = ?",
                (agent_playbook_id,),
            ).fetchone()[0]
            == 0
        )
    assert (
        storage.search_agent_playbooks(
            SearchAgentPlaybookRequest(query="original content", top_k=10)
        )
        == []
    )


def test_apply_governance_agent_playbook_rebuild_restores_previous_lifecycle_status(
    storage,
):
    purge_id = "purge_rebuild_restore_archived"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_rebuild_restore_archived",
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
            "previous_lifecycle_status": Status.SUPERSEDED.value,
            "remaining_source_windows": [
                {"user_playbook_id": 9, "source_interaction_ids": [201]},
            ],
        },
    )
    storage.record_purge_target(
        purge_id=purge_id,
        target_name="agent_playbook",
        target_ref=str(agent_playbook_id),
        phase="hide_for_rebuild",
        status="complete",
    )

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

    rebuilt_status = storage.conn.execute(
        "SELECT status FROM agent_playbooks WHERE agent_playbook_id = ?",
        (agent_playbook_id,),
    ).fetchone()[0]
    assert rebuilt_status == Status.SUPERSEDED.value


def test_apply_governance_agent_playbook_rebuild_rejects_second_call_after_completion(
    storage,
):
    purge_id = "purge_rebuild_second_call"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_rebuild_second_call",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    agent_playbook_id = _seed_agent_playbook(
        storage,
        status=Status.ARCHIVED,
        source_windows=[
            AgentPlaybookSourceWindow(user_playbook_id=7, source_interaction_ids=[101]),
            AgentPlaybookSourceWindow(user_playbook_id=9, source_interaction_ids=[201]),
        ],
    )
    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id="user-rebuild-second-call",
        owned_user_playbook_ids={7},
    )
    storage.hide_governance_agent_playbooks_for_rebuild(purge_id)
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

    before_row = storage.conn.execute(
        """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    assert before_row is not None
    before_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)
    before_hide_target = next(
        target
        for target in storage.list_purge_targets(purge_id, phase="hide_for_rebuild")
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    before_rebuild_target = next(
        target
        for target in storage.list_purge_targets(
            purge_id, phase="rebuild_without_erased_sources"
        )
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )

    with pytest.raises(ValueError, match="already complete"):
        storage.apply_governance_agent_playbook_rebuild(
            purge_id=purge_id,
            agent_playbook_id=agent_playbook_id,
            remaining_source_windows=[
                {"user_playbook_id": 9, "source_interaction_ids": [201]},
            ],
            content="mutated content",
            trigger="mutated trigger",
            rationale="mutated rationale",
            blocking_issue={"issue": "should not persist"},
            expanded_terms="mutated terms",
            tags=["mutated"],
        )

    after_row = storage.conn.execute(
        """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    assert after_row == before_row
    assert (
        storage.get_source_windows_for_agent_playbook(agent_playbook_id)
        == before_windows
    )
    assert (
        next(
            target
            for target in storage.list_purge_targets(purge_id, phase="hide_for_rebuild")
            if target.target_name == "agent_playbook"
            and target.target_ref == str(agent_playbook_id)
        )
        == before_hide_target
    )
    assert (
        next(
            target
            for target in storage.list_purge_targets(
                purge_id, phase="rebuild_without_erased_sources"
            )
            if target.target_name == "agent_playbook"
            and target.target_ref == str(agent_playbook_id)
        )
        == before_rebuild_target
    )


def test_hide_governance_agent_playbooks_for_rebuild_is_idempotent_after_completed_rebuild(
    storage,
):
    purge_id = "purge_hide_after_complete"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_hide_after_complete",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    agent_playbook_id = _seed_agent_playbook(
        storage,
        status=Status.ARCHIVED,
        source_windows=[
            AgentPlaybookSourceWindow(user_playbook_id=7, source_interaction_ids=[101]),
            AgentPlaybookSourceWindow(user_playbook_id=9, source_interaction_ids=[201]),
        ],
    )
    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id="user-hide-after-complete",
        owned_user_playbook_ids={7},
    )
    storage.hide_governance_agent_playbooks_for_rebuild(purge_id)
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

    before_status = storage.conn.execute(
        "SELECT status FROM agent_playbooks WHERE agent_playbook_id = ?",
        (agent_playbook_id,),
    ).fetchone()[0]
    before_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)
    before_hide_target = next(
        target
        for target in storage.list_purge_targets(purge_id, phase="hide_for_rebuild")
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    before_rebuild_target = next(
        target
        for target in storage.list_purge_targets(
            purge_id, phase="rebuild_without_erased_sources"
        )
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )

    hidden_ids = storage.hide_governance_agent_playbooks_for_rebuild(purge_id)

    after_status = storage.conn.execute(
        "SELECT status FROM agent_playbooks WHERE agent_playbook_id = ?",
        (agent_playbook_id,),
    ).fetchone()[0]
    after_hide_target = next(
        target
        for target in storage.list_purge_targets(purge_id, phase="hide_for_rebuild")
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    after_rebuild_target = next(
        target
        for target in storage.list_purge_targets(
            purge_id, phase="rebuild_without_erased_sources"
        )
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )

    assert hidden_ids == []
    assert after_status == before_status == Status.ARCHIVED.value
    assert (
        storage.get_source_windows_for_agent_playbook(agent_playbook_id)
        == before_windows
    )
    assert after_hide_target == before_hide_target
    assert after_rebuild_target == before_rebuild_target


def test_hide_governance_agent_playbooks_for_rebuild_does_not_reopen_complete_target_from_stale_prelock_state(
    storage, monkeypatch
):
    purge_id = "purge_hide_stale_prelock"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_hide_stale_prelock",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    agent_playbook_id = _seed_agent_playbook(
        storage,
        status=Status.ARCHIVED,
        source_windows=[
            AgentPlaybookSourceWindow(user_playbook_id=7, source_interaction_ids=[101]),
            AgentPlaybookSourceWindow(user_playbook_id=9, source_interaction_ids=[201]),
        ],
    )
    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id="user-hide-stale-prelock",
        owned_user_playbook_ids={7},
    )
    storage.hide_governance_agent_playbooks_for_rebuild(purge_id)
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

    before_status = storage.conn.execute(
        "SELECT status FROM agent_playbooks WHERE agent_playbook_id = ?",
        (agent_playbook_id,),
    ).fetchone()[0]
    before_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)
    before_hide_target = next(
        target
        for target in storage.list_purge_targets(purge_id, phase="hide_for_rebuild")
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )
    before_rebuild_target = next(
        target
        for target in storage.list_purge_targets(
            purge_id, phase="rebuild_without_erased_sources"
        )
        if target.target_name == "agent_playbook"
        and target.target_ref == str(agent_playbook_id)
    )

    stale_targets = [
        before_rebuild_target.model_copy(update={"status": "pending"}),
    ]
    original_list_purge_targets = storage.list_purge_targets

    def stale_list_purge_targets(*_args, **_kwargs):
        return stale_targets

    monkeypatch.setattr(storage, "list_purge_targets", stale_list_purge_targets)

    hidden_ids = storage.hide_governance_agent_playbooks_for_rebuild(purge_id)

    assert hidden_ids == []
    assert (
        storage.conn.execute(
            "SELECT status FROM agent_playbooks WHERE agent_playbook_id = ?",
            (agent_playbook_id,),
        ).fetchone()[0]
        == before_status
        == Status.ARCHIVED.value
    )
    assert (
        storage.get_source_windows_for_agent_playbook(agent_playbook_id)
        == before_windows
    )
    assert (
        next(
            target
            for target in original_list_purge_targets(
                purge_id, phase="hide_for_rebuild"
            )
            if target.target_name == "agent_playbook"
            and target.target_ref == str(agent_playbook_id)
        )
        == before_hide_target
    )
    assert (
        next(
            target
            for target in original_list_purge_targets(
                purge_id, phase="rebuild_without_erased_sources"
            )
            if target.target_name == "agent_playbook"
            and target.target_ref == str(agent_playbook_id)
        )
        == before_rebuild_target
    )


def test_prepare_governance_erase_targets_is_idempotent_after_completed_snapshot_and_rebuild(
    storage,
):
    purge_id = "purge_prepare_idempotent_after_rebuild"
    user_id = "user-prepare-idempotent-after-rebuild"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_prepare_idempotent_after_rebuild",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    owned_user_playbook_ids = _seed_prepare_counts_user_data(storage, user_id=user_id)
    _seed_eval_result(
        storage,
        user_id=user_id,
        session_id="session-delete-hide-complete",
        evaluation_name="governance_delete_hide_complete",
    )
    affected_user_playbook_id = min(owned_user_playbook_ids)
    agent_playbook_id = _seed_agent_playbook(
        storage,
        status=Status.ARCHIVED,
        source_windows=[
            AgentPlaybookSourceWindow(
                user_playbook_id=affected_user_playbook_id,
                source_interaction_ids=[101],
            ),
            AgentPlaybookSourceWindow(
                user_playbook_id=999999, source_interaction_ids=[201]
            ),
        ],
    )
    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id=user_id,
        owned_user_playbook_ids=owned_user_playbook_ids,
    )
    storage.hide_governance_agent_playbooks_for_rebuild(purge_id)
    storage.apply_governance_agent_playbook_rebuild(
        purge_id=purge_id,
        agent_playbook_id=agent_playbook_id,
        remaining_source_windows=[
            {"user_playbook_id": 999999, "source_interaction_ids": [201]},
        ],
        content="rebuilt content",
        trigger="rebuilt trigger",
        rationale="rebuilt rationale",
        blocking_issue=None,
        expanded_terms="rebuilt terms",
        tags=["rebuilt"],
    )

    before_targets = [
        (
            target.target_name,
            target.target_ref,
            target.phase,
            target.status,
            target.detail,
            target.deleted_count,
        )
        for target in storage.list_purge_targets(purge_id)
        if target.target_name
        in {*CANONICAL_DELETE_TARGET_NAMES, "agent_playbook", "target_snapshot"}
    ]
    before_playbook_row = storage.conn.execute(
        """SELECT status, content, trigger, rationale, tags
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    before_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)

    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id=user_id,
        owned_user_playbook_ids=owned_user_playbook_ids,
    )

    after_targets = [
        (
            target.target_name,
            target.target_ref,
            target.phase,
            target.status,
            target.detail,
            target.deleted_count,
        )
        for target in storage.list_purge_targets(purge_id)
        if target.target_name
        in {*CANONICAL_DELETE_TARGET_NAMES, "agent_playbook", "target_snapshot"}
    ]
    after_playbook_row = storage.conn.execute(
        """SELECT status, content, trigger, rationale, tags
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    after_windows = storage.get_source_windows_for_agent_playbook(agent_playbook_id)

    assert after_targets == before_targets
    assert after_playbook_row == before_playbook_row
    assert after_windows == before_windows


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
        target_ref="all",
        phase="delete",
        status="pending",
        detail={"count": 1},
    )
    storage_org2.record_purge_target(
        purge_id=purge_id,
        target_name="request",
        target_ref="all",
        phase="delete",
        status="complete",
        detail={"count": 2},
        deleted_count=2,
    )

    org1_targets = storage_org1.list_purge_targets(purge_id)
    org2_targets = storage_org2.list_purge_targets(purge_id)

    assert {
        (target.phase, target.target_ref, target.status) for target in org1_targets
    } == {
        ("delete", "all", "pending"),
        ("prepare_targets", "all", "complete"),
    }
    assert {
        (target.phase, target.target_ref, target.status) for target in org2_targets
    } == {
        ("delete", "all", "complete"),
    }
    assert storage_org1.purge_targets_prepared(purge_id) is True
    assert storage_org2.purge_targets_prepared(purge_id) is False

    storage_org2.record_purge_target(
        purge_id=purge_id,
        target_name="request",
        target_ref="all",
        phase="delete",
        status="running",
        detail={"count": 3},
    )

    org1_request_target = next(
        target
        for target in storage_org1.list_purge_targets(purge_id, phase="delete")
        if target.target_ref == "all"
    )
    org2_delete_targets = storage_org2.list_purge_targets(purge_id, phase="delete")

    assert org1_request_target.status == "pending"
    assert org1_request_target.detail == {"count": 1}
    assert {
        (target.target_ref, target.status, target.deleted_count)
        for target in org2_delete_targets
    } == {
        ("all", "running", 0),
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
            row
            for row in storage.list_purge_targets(purge_id, phase="delete")
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
def test_record_purge_target_rejects_invalid_deleted_count(
    storage, deleted_count, match
):
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


@pytest.mark.parametrize("detail_deleted_count", [0, 2])
def test_record_purge_target_accepts_nonnegative_detail_deleted_count(
    storage, detail_deleted_count
):
    purge_id = _begin_purge(
        storage, f"purge_detail_deleted_count_{detail_deleted_count}"
    )

    storage.record_purge_target(
        purge_id=purge_id,
        target_name="request",
        target_ref="all",
        phase="delete",
        status="complete",
        detail={"deleted_count": detail_deleted_count},
    )

    target = next(
        row
        for row in storage.list_purge_targets(purge_id, phase="delete")
        if row.target_name == "request"
    )
    assert target.detail == {"deleted_count": detail_deleted_count}


def test_record_purge_target_rejects_negative_detail_deleted_count(storage):
    purge_id = _begin_purge(storage, "purge_detail_deleted_count_negative")

    with pytest.raises(ValueError, match="deleted_count"):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="request",
            target_ref="all",
            phase="delete",
            status="complete",
            detail={"deleted_count": -1},
        )


@pytest.mark.parametrize(
    ("detail", "match"),
    [
        pytest.param({"email": "bob@example.com"}, "email", id="audit-detail-email"),
        pytest.param(
            {"request_id": "reqref_123"}, "request_id", id="audit-detail-request-id"
        ),
        pytest.param(
            {"content": "verbatim prompt"}, "content", id="audit-detail-content"
        ),
        pytest.param(
            {"note": "arbitrary string"}, "note", id="audit-detail-neutral-note"
        ),
        pytest.param(
            {"status": "prompt-ready"},
            "prompt/content",
            id="audit-detail-promptish-string",
        ),
        pytest.param(
            {"owned_user_playbook_ids": [7]},
            "owned_user_playbook_ids",
            id="audit-detail-rejects-owned-user-playbook-ids",
        ),
        pytest.param(
            {"source_interaction_ids": [1]},
            "source_interaction_ids",
            id="audit-detail-rejects-source-interaction-ids",
        ),
        pytest.param(
            {
                "original_source_windows": [
                    {"user_playbook_id": 7, "source_interaction_ids": [1]}
                ]
            },
            "original_source_windows",
            id="audit-detail-rejects-original-source-windows",
        ),
        pytest.param(
            {
                "remaining_source_windows": [
                    {"user_playbook_id": 7, "source_interaction_ids": [1]}
                ]
            },
            "remaining_source_windows",
            id="audit-detail-rejects-remaining-source-windows",
        ),
        pytest.param({"count": 2}, None, id="audit-detail-allowed-count"),
        pytest.param(
            {"deleted_count": 1}, None, id="audit-detail-allowed-deleted-count"
        ),
        pytest.param(
            {"deleted_counts": {"requests": 1}},
            None,
            id="audit-detail-allowed-deleted-counts",
        ),
        pytest.param(
            {"agent_playbook_id": 7}, None, id="audit-detail-allowed-agent-playbook-id"
        ),
        pytest.param(
            {"rebuilt_agent_playbook_ids": [7, 8]},
            None,
            id="audit-detail-allowed-rebuilt-agent-playbook-ids",
        ),
        pytest.param({"status": "ok"}, None, id="audit-detail-allowed-status"),
        pytest.param({"route": "delete"}, None, id="audit-detail-allowed-route"),
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


def test_record_purge_target_accepts_target_detail_shapes(storage):
    purge_id = _begin_purge(storage, "purge_target_detail_shapes")

    detail = {
        "owned_user_playbook_ids": [7],
        "source_interaction_ids": [11, 12],
        "original_source_windows": [
            {"user_playbook_id": 7, "source_interaction_ids": [11, 12]}
        ],
        "previous_lifecycle_status": Status.ARCHIVED.value,
        "remaining_source_windows": [
            {"user_playbook_id": 7, "source_interaction_ids": [12]}
        ],
    }

    storage.record_purge_target(
        purge_id=purge_id,
        target_name="agent_playbook",
        target_ref="7",
        phase="rebuild_without_erased_sources",
        status="complete",
        detail=detail,
    )

    targets = storage.list_purge_targets(
        purge_id, phase="rebuild_without_erased_sources"
    )
    stored = next(target for target in targets if target.target_ref == "7")
    assert stored.detail == detail


@pytest.mark.parametrize("count", [0, 2])
def test_append_audit_event_accepts_nonnegative_detail_count(storage, count):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key=f"export_detail_count_{count}",
        detail={"count": count},
    )

    assert storage.append_audit_event(event) is True
    rows = storage.list_audit_events(subject_ref=SUBJECT_REF)
    stored = next(row for row in rows if row.idempotency_key == event.idempotency_key)
    assert stored.detail == {"count": count}


def test_append_audit_event_rejects_negative_detail_count(storage):
    event = AuditEvent(
        org_id="org1",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="export_detail_count_negative",
        detail={"count": -1},
    )

    with pytest.raises(ValueError, match="count"):
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
    "error_code", ["content_purge_failed", "prompt_redaction_route"]
)
def test_fail_purge_operation_accepts_code_shaped_error_code_with_prompt_or_content(
    storage, error_code
):
    purge_id = _begin_purge(storage, f"purge_error_code_{error_code}")

    failed = storage.fail_purge_operation(
        purge_id,
        error_code=error_code,
        error_detail="target_delete_failed",
    )

    assert failed.status == "failed"
    assert failed.error_code == error_code


def test_fail_purge_operation_rejects_prompt_content_prose_error_detail(storage):
    purge_id = _begin_purge(storage, "purge_fail_prompt_content_prose")

    with pytest.raises(ValueError, match="error_detail"):
        storage.fail_purge_operation(
            purge_id,
            error_code="PURGE_TARGET_FAILED",
            error_detail="prompt content leaked from upstream",
        )

    assert storage.get_purge_operation(purge_id).error_detail is None


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
def test_append_audit_event_validates_top_level_governance_fields(
    storage, event, match
):
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


def test_audit_event_requires_request_ref():
    with pytest.raises(ValidationError, match="request_ref"):
        AuditEvent.model_validate(
            {
                "org_id": "org1",
                "operation": "EXPORT",
                "entity_type": "request",
                "subject_ref": SUBJECT_REF,
                "idempotency_key": "missing_request_ref",
            }
        )


@pytest.mark.parametrize(
    ("subject_ref", "request_ref", "match"),
    [
        pytest.param(
            SUBJECT_REF, "request_12345", "request_ref", id="purge-request-ref"
        ),
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
        pytest.param(
            cast(Any, "erase_user"), "user", "operation_type", id="operation-type"
        ),
        pytest.param(
            "user_erasure", cast(Any, "workspace"), "scope_type", id="scope-type"
        ),
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


@pytest.mark.parametrize(
    "detail_key", ["remaining_source_windows", "original_source_windows"]
)
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

    with pytest.raises(ValueError, match=detail_key):
        storage.append_audit_event(event)


@pytest.mark.parametrize(
    "detail_key", ["remaining_source_windows", "original_source_windows"]
)
def test_record_purge_target_rejects_mixed_case_window_keys(storage, detail_key):
    purge_id = _begin_purge(storage, f"purge_{detail_key}")

    with pytest.raises(ValueError, match="user_playbook_id"):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="agent_playbook",
            target_ref="7",
            phase="rebuild_without_erased_sources",
            status="running",
            detail={detail_key: [{"User_Playbook_Id": "alice@example.com"}]},
        )


@pytest.mark.parametrize(
    "detail_key", ["remaining_source_windows", "original_source_windows"]
)
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

    with pytest.raises(ValueError, match=detail_key):
        storage.append_audit_event(event)


@pytest.mark.parametrize(
    "detail_key", ["remaining_source_windows", "original_source_windows"]
)
def test_record_purge_target_requires_window_user_playbook_id(storage, detail_key):
    purge_id = _begin_purge(storage, f"purge_missing_upb_{detail_key}")

    with pytest.raises(ValueError, match="user_playbook_id"):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="agent_playbook",
            target_ref="7",
            phase="rebuild_without_erased_sources",
            status="running",
            detail={detail_key: [{"source_interaction_ids": [1, 2]}]},
        )


@pytest.mark.parametrize(
    "previous_lifecycle_status",
    [None, Status.ARCHIVED.value, Status.SUPERSEDED.value],
)
def test_record_purge_target_accepts_previous_lifecycle_status_for_rebuild_targets(
    storage, previous_lifecycle_status
):
    purge_id = _begin_purge(
        storage, f"purge_prev_lifecycle_{previous_lifecycle_status or 'current'}"
    )

    storage.record_purge_target(
        purge_id=purge_id,
        target_name="agent_playbook",
        target_ref="7",
        phase="rebuild_without_erased_sources",
        status="running",
        detail={
            "original_source_windows": [
                {"user_playbook_id": 7, "source_interaction_ids": [11, 12]}
            ],
            "previous_lifecycle_status": previous_lifecycle_status,
            "remaining_source_windows": [
                {"user_playbook_id": 7, "source_interaction_ids": [12]}
            ],
        },
    )

    stored = next(
        target
        for target in storage.list_purge_targets(
            purge_id, phase="rebuild_without_erased_sources"
        )
        if target.target_ref == "7"
    )
    assert stored.detail is not None
    assert stored.detail["previous_lifecycle_status"] == previous_lifecycle_status


@pytest.mark.parametrize(
    ("detail", "match"),
    [
        pytest.param(
            {
                "original_source_windows": [
                    {"user_playbook_id": 7, "source_interaction_ids": [11, 12]}
                ],
                "previous_lifecycle_status": "approved",
                "remaining_source_windows": [
                    {"user_playbook_id": 7, "source_interaction_ids": [12]}
                ],
            },
            "previous_lifecycle_status",
            id="rejects-non-lifecycle-status",
        ),
        pytest.param(
            {
                "original_source_windows": [
                    {"user_playbook_id": 7, "source_interaction_ids": [11, 12]}
                ],
                "previous_lifecycle_status": {"status": Status.ARCHIVED.value},
                "remaining_source_windows": [
                    {"user_playbook_id": 7, "source_interaction_ids": [12]}
                ],
            },
            "previous_lifecycle_status",
            id="rejects-non-string-status-shape",
        ),
        pytest.param(
            {
                "original_source_windows": [
                    {"user_playbook_id": 7, "source_interaction_ids": [11, 12]}
                ],
                "remaining_source_windows": [
                    {"user_playbook_id": 7, "source_interaction_ids": [12]}
                ],
                "arbitrary_status_copy": Status.ARCHIVED.value,
            },
            "arbitrary_status_copy",
            id="rejects-arbitrary-detail-key",
        ),
    ],
)
def test_record_purge_target_rejects_invalid_previous_lifecycle_status_detail(
    storage, detail, match
):
    purge_id = _begin_purge(storage, "purge_prev_lifecycle_invalid")

    with pytest.raises(ValueError, match=match):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="agent_playbook",
            target_ref="7",
            phase="rebuild_without_erased_sources",
            status="running",
            detail=detail,
        )


@pytest.mark.parametrize(
    ("target_ref", "match"),
    [
        pytest.param("all", None, id="marker-all"),
        pytest.param("17", "target_ref", id="internal-numeric-id"),
        pytest.param(REQUEST_REF, "target_ref", id="minimized-request-ref"),
        pytest.param(SUBJECT_REF, "target_ref", id="minimized-subject-ref"),
        pytest.param("", "target_ref", id="empty-default"),
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


@pytest.mark.parametrize(
    "purge_id", ["alice@example.com", "request_12345", "alice", SUBJECT_REF]
)
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

    remaining = _user_scoped_row_counts(storage, user_id=user_id)
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
    for target_name in CANONICAL_DELETE_TARGET_NAMES:
        storage.record_purge_target(
            purge_id=purge_id,
            target_name=target_name,
            target_ref="all",
            phase="delete",
            status="pending",
            detail={"count": 0},
        )

    def _stub_clear_user_data_for_governance_locked(
        self: SQLiteStorage,
        user_id: str,
        *,
        expected_user_playbook_ids: set[int] | None = None,
    ) -> dict[str, int]:
        del self, user_id, expected_user_playbook_ids
        return {"requests": 1, "surprise_target": 2}

    monkeypatch.setattr(
        SQLiteStorage,
        "_clear_user_data_for_governance_locked",
        _stub_clear_user_data_for_governance_locked,
    )

    with pytest.raises(ValueError, match="target_name"):
        storage.apply_governance_user_data_delete(
            purge_id=purge_id,
            user_id="user-delete-seed",
        )

    delete_targets = storage.list_purge_targets(purge_id, phase="delete")
    assert all(target.target_name != "surprise_target" for target in delete_targets)


def test_apply_governance_user_data_delete_requires_complete_prepared_delete_matrix(
    storage, monkeypatch
):
    purge_id = _begin_purge(storage, "purge_delete_requires_prepared_matrix")
    user_id = "user-delete-seed"
    expected_user_id = user_id
    _seed_user_scoped_rows(storage, user_id=user_id)
    baseline_counts = _user_scoped_row_counts(storage, user_id=user_id)
    storage.record_purge_target(
        purge_id=purge_id,
        target_name="request",
        target_ref="all",
        phase="delete",
        status="pending",
        detail={"count": 1},
    )
    storage.record_purge_target(
        purge_id=purge_id,
        target_name="interaction",
        target_ref="all",
        phase="delete",
        status="complete",
        detail={"count": 0},
        deleted_count=0,
    )

    clear_locked_called = False

    def _stub_clear_user_data_for_governance_locked(
        self: SQLiteStorage, patched_user_id: str
    ) -> dict[str, int]:
        nonlocal clear_locked_called
        del self
        clear_locked_called = True
        assert patched_user_id == expected_user_id
        return {"requests": 1}

    monkeypatch.setattr(
        SQLiteStorage,
        "_clear_user_data_for_governance_locked",
        _stub_clear_user_data_for_governance_locked,
    )

    with pytest.raises(ValueError, match="complete delete target matrix"):
        storage.apply_governance_user_data_delete(
            purge_id=purge_id,
            user_id=user_id,
        )

    assert clear_locked_called is False
    assert _user_scoped_row_counts(storage, user_id=user_id) == baseline_counts
    delete_targets = storage.list_purge_targets(purge_id, phase="delete")
    assert {(target.target_name, target.status) for target in delete_targets} == {
        ("request", "pending"),
        ("interaction", "complete"),
    }


def test_apply_governance_user_data_delete_requires_hide_targets_for_planned_rebuilds(
    storage,
):
    purge_id = "purge_delete_requires_hide"
    user_id = "user-delete-hide-required"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_delete_requires_hide",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    owned_user_playbook_ids = _seed_prepare_counts_user_data(storage, user_id=user_id)
    _seed_eval_result(
        storage,
        user_id=user_id,
        session_id="session-delete-hide-complete",
        evaluation_name="governance_delete_hide_complete",
    )
    affected_user_playbook_id = min(owned_user_playbook_ids)
    _seed_agent_playbook(
        storage,
        status=None,
        source_windows=[
            AgentPlaybookSourceWindow(
                user_playbook_id=affected_user_playbook_id,
                source_interaction_ids=[101],
            )
        ],
    )
    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id=user_id,
        owned_user_playbook_ids=owned_user_playbook_ids,
    )

    with pytest.raises(ValueError, match="hide_for_rebuild"):
        storage.apply_governance_user_data_delete(
            purge_id=purge_id,
            user_id=user_id,
        )

    assert _user_scoped_row_counts(storage, user_id=user_id) == {
        "requests": 1,
        "interactions": 1,
        "profiles": 2,
        "user_playbooks": 2,
    }
    delete_targets = storage.list_purge_targets(purge_id, phase="delete")
    assert {(target.target_name, target.status) for target in delete_targets} == {
        (target_name, "pending") for target_name in CANONICAL_DELETE_TARGET_NAMES
    }


def test_apply_governance_user_data_delete_succeeds_after_hide_targets_complete(
    storage,
):
    purge_id = "purge_delete_after_hide"
    user_id = "user-delete-hide-complete"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_delete_after_hide",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    owned_user_playbook_ids = _seed_prepare_counts_user_data(storage, user_id=user_id)
    _seed_eval_result(
        storage,
        user_id=user_id,
        session_id="session-delete-hide-complete",
        evaluation_name="governance_delete_hide_complete",
    )
    affected_user_playbook_id = min(owned_user_playbook_ids)
    _seed_agent_playbook(
        storage,
        status=None,
        source_windows=[
            AgentPlaybookSourceWindow(
                user_playbook_id=affected_user_playbook_id,
                source_interaction_ids=[101],
            )
        ],
    )
    storage.conn.execute(
        """INSERT INTO lineage_event (
               org_id, entity_type, entity_id, op, prov_relation, source_ids,
               actor, request_id, reason, created_at
           )
           VALUES (?, 'agent_playbook', 'agent_survivor', 'merge', 'wasDerivedFrom',
                   ?, 'test', 'req-unrelated', 'source-user-playbook', 1)""",
        (storage.org_id, json.dumps([str(affected_user_playbook_id)])),
    )
    storage.conn.commit()
    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id=user_id,
        owned_user_playbook_ids=owned_user_playbook_ids,
    )
    storage.hide_governance_agent_playbooks_for_rebuild(purge_id)

    counts = storage.apply_governance_user_data_delete(
        purge_id=purge_id,
        user_id=user_id,
    )

    assert counts == {
        "interactions": 1,
        "user_playbooks": 1,
        "profiles": 1,
        "requests": 1,
        "agent_success_evaluation_results": 1,
        "purged_profiles": 1,
        "purged_user_playbooks": 1,
    }
    assert _user_scoped_row_counts(storage, user_id=user_id) == {
        "requests": 0,
        "interactions": 0,
        "profiles": 0,
        "user_playbooks": 0,
    }
    assert (
        storage.conn.execute(
            """SELECT COUNT(*)
           FROM profiles
           WHERE merged_into IS NOT NULL AND content = '' AND user_id = ''"""
        ).fetchone()[0]
        == 1
    )
    assert (
        storage.conn.execute(
            """SELECT COUNT(*)
           FROM user_playbooks
           WHERE merged_into IS NOT NULL
             AND content = ''
             AND user_id IS NULL
             AND request_id = ''"""
        ).fetchone()[0]
        == 1
    )
    assert (
        storage.conn.execute(
            """SELECT COUNT(*)
               FROM lineage_event
               WHERE org_id = ?
                 AND (
                    request_id = ?
                    OR entity_id IN (?, ?, ?, ?)
                    OR source_ids = ?
                 )""",
            (
                storage.org_id,
                "req-delete-hide-complete",
                "req-delete-hide-complete",
                "101",
                "profile_seed",
                str(affected_user_playbook_id),
                json.dumps([str(affected_user_playbook_id)]),
            ),
        ).fetchone()[0]
        == 0
    )
    delete_targets = storage.list_purge_targets(purge_id, phase="delete")
    assert {target.target_name: target.deleted_count for target in delete_targets} == {
        "request": 1,
        "interaction": 1,
        "profile": 1,
        "user_playbook": 1,
        "agent_success_evaluation_result": 1,
        "profile_purge": 1,
        "user_playbook_purge": 1,
    }
    assert all(target.status == "complete" for target in delete_targets)


def test_apply_governance_user_data_delete_is_failure_atomic(storage, monkeypatch):
    purge_id = "purge_delete_atomic"
    user_id = "user-delete-atomic"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_delete_atomic",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    owned_user_playbook_ids = _seed_prepare_counts_user_data(storage, user_id=user_id)
    affected_user_playbook_id = min(owned_user_playbook_ids)
    _seed_agent_playbook(
        storage,
        status=None,
        source_windows=[
            AgentPlaybookSourceWindow(
                user_playbook_id=affected_user_playbook_id,
                source_interaction_ids=[101],
            )
        ],
    )
    storage.prepare_governance_erase_targets(
        purge_id=purge_id,
        user_id=user_id,
        owned_user_playbook_ids=owned_user_playbook_ids,
    )
    storage.hide_governance_agent_playbooks_for_rebuild(purge_id)

    before_counts = _user_scoped_row_counts(storage, user_id=user_id)
    before_profile_rows = storage.conn.execute(
        """SELECT profile_id, content, user_id
           FROM profiles
           WHERE profile_id IN ('profile_seed', 'profile_purge_seed')
           ORDER BY profile_id ASC"""
    ).fetchall()
    before_playbook_rows = storage.conn.execute(
        """SELECT user_playbook_id, content, user_id
           FROM user_playbooks
           WHERE user_id = ?
           ORDER BY user_playbook_id ASC""",
        (user_id,),
    ).fetchall()
    original_record_purge_target_locked = SQLiteStorage._record_purge_target_locked

    def _raising_record_purge_target_locked(
        self: SQLiteStorage,
        *,
        purge_id: str,
        target_name: str,
        target_ref: str,
        phase: str,
        status: Literal["pending", "running", "failed", "complete"],
        detail: dict[str, object] | None,
        deleted_count: int,
        error_detail: str | None,
    ) -> None:
        if phase == "delete" and status == "complete" and target_name == "request":
            raise RuntimeError("inject target completion failure")
        original_record_purge_target_locked(
            self,
            purge_id=purge_id,
            target_name=target_name,
            target_ref=target_ref,
            phase=phase,
            status=status,
            detail=detail,
            deleted_count=deleted_count,
            error_detail=error_detail,
        )

    monkeypatch.setattr(
        SQLiteStorage,
        "_record_purge_target_locked",
        _raising_record_purge_target_locked,
    )

    with pytest.raises(RuntimeError, match="inject target completion failure"):
        storage.apply_governance_user_data_delete(
            purge_id=purge_id,
            user_id=user_id,
        )

    assert _user_scoped_row_counts(storage, user_id=user_id) == before_counts
    after_profile_rows = storage.conn.execute(
        """SELECT profile_id, content, user_id
           FROM profiles
           WHERE profile_id IN ('profile_seed', 'profile_purge_seed')
           ORDER BY profile_id ASC"""
    ).fetchall()
    after_playbook_rows = storage.conn.execute(
        """SELECT user_playbook_id, content, user_id
           FROM user_playbooks
           WHERE user_id = ?
           ORDER BY user_playbook_id ASC""",
        (user_id,),
    ).fetchall()
    assert after_profile_rows == before_profile_rows
    assert after_playbook_rows == before_playbook_rows
    delete_targets = storage.list_purge_targets(purge_id, phase="delete")
    assert {(target.target_name, target.status) for target in delete_targets} == {
        (target_name, "pending") for target_name in CANONICAL_DELETE_TARGET_NAMES
    }


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
            remaining_source_windows=[
                {"user_playbook_id": 99, "source_interaction_ids": [202]}
            ],
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


def test_apply_governance_agent_playbook_rebuild_rejects_mismatched_remaining_source_windows(
    storage,
):
    purge_id = "purge_rebuild_windows_mismatch"
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key="idem_purge_rebuild_windows_mismatch",
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
            "previous_lifecycle_status": Status.ARCHIVE_IN_PROGRESS.value,
            "remaining_source_windows": [
                {"user_playbook_id": 9, "source_interaction_ids": [201]},
            ],
        },
    )
    storage.record_purge_target(
        purge_id=purge_id,
        target_name="agent_playbook",
        target_ref=str(agent_playbook_id),
        phase="hide_for_rebuild",
        status="complete",
    )
    original_row = storage.conn.execute(
        """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    assert original_row is not None

    with pytest.raises(ValueError, match="remaining_source_windows"):
        storage.apply_governance_agent_playbook_rebuild(
            purge_id=purge_id,
            agent_playbook_id=agent_playbook_id,
            remaining_source_windows=[
                {"user_playbook_id": 9, "source_interaction_ids": [999]},
            ],
            content="rebuilt content",
            trigger="rebuilt trigger",
            rationale="rebuilt rationale",
            blocking_issue=None,
            expanded_terms="rebuilt terms",
            tags=["rebuilt"],
        )

    rebuilt_row = storage.conn.execute(
        """SELECT content, trigger, rationale, blocking_issue, expanded_terms, tags, status
           FROM agent_playbooks
           WHERE agent_playbook_id = ?""",
        (agent_playbook_id,),
    ).fetchone()
    assert rebuilt_row == original_row


def test_get_agent_playbook_by_id_default_excludes_archive_in_progress(storage):
    agent_playbook_id = _seed_agent_playbook(
        storage,
        status=Status.ARCHIVE_IN_PROGRESS,
    )

    assert storage.get_agent_playbook_by_id(agent_playbook_id) is None
    included = storage.get_agent_playbook_by_id(
        agent_playbook_id,
        include_tombstones=True,
    )
    assert included is not None
    assert included.agent_playbook_id == agent_playbook_id
    assert included.status == Status.ARCHIVE_IN_PROGRESS


def test_get_agent_playbooks_default_excludes_archive_in_progress(storage):
    hidden_id = _seed_agent_playbook(
        storage,
        status=Status.ARCHIVE_IN_PROGRESS,
    )
    visible_id = _seed_agent_playbook(
        storage,
        status=None,
    )

    default_ids = {
        playbook.agent_playbook_id for playbook in storage.get_agent_playbooks(limit=10)
    }
    hidden_only_ids = {
        playbook.agent_playbook_id
        for playbook in storage.get_agent_playbooks(
            limit=10,
            status_filter=[Status.ARCHIVE_IN_PROGRESS],
        )
    }

    assert visible_id in default_ids
    assert hidden_id not in default_ids
    assert hidden_only_ids == {hidden_id}


def test_search_agent_playbooks_default_excludes_archive_in_progress_and_explicit_filter_includes_it(
    storage,
):
    hidden_playbook = AgentPlaybook(
        playbook_name="governance-hidden-search",
        agent_version="test-agent",
        content="hidden-search-token",
        trigger="hidden-search-token",
        rationale="hidden-search-rationale",
        status=Status.ARCHIVE_IN_PROGRESS,
    )
    visible_playbook = AgentPlaybook(
        playbook_name="governance-visible-search",
        agent_version="test-agent",
        content="visible-search-token",
        trigger="visible-search-token",
        rationale="visible-search-rationale",
        status=None,
    )
    hidden_id = storage.save_agent_playbooks([hidden_playbook])[0].agent_playbook_id
    visible_id = storage.save_agent_playbooks([visible_playbook])[0].agent_playbook_id

    default_results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(query="hidden-search-token", top_k=10)
    )
    explicit_hidden_results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="hidden-search-token",
            top_k=10,
            status_filter=[Status.ARCHIVE_IN_PROGRESS],
        )
    )
    visible_results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(query="visible-search-token", top_k=10)
    )

    assert hidden_id not in {playbook.agent_playbook_id for playbook in default_results}
    assert [playbook.agent_playbook_id for playbook in explicit_hidden_results] == [
        hidden_id
    ]
    assert [playbook.agent_playbook_id for playbook in visible_results] == [visible_id]


@pytest.mark.parametrize(
    ("target_name", "phase", "status", "match"),
    [
        pytest.param(
            cast(Any, "session"), "delete", "running", "target_name", id="target-name"
        ),
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
def test_governance_persistence_rejects_unsafe_idempotency_keys(
    storage, idempotency_key
):
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
    "detail",
    [
        pytest.param({"status": "archived"}, id="status-archived"),
        pytest.param({"route": "rebuild"}, id="route-rebuild"),
        pytest.param({"route": "custom.route"}, id="route-custom-dot"),
        pytest.param({"route": "alice.team"}, id="route-alice-team"),
    ],
)
@pytest.mark.parametrize("persistence_path", ["audit_event", "purge_target"])
def test_governance_detail_rejects_noncanonical_status_and_route(
    storage, detail, persistence_path
):
    if persistence_path == "audit_event":
        event = AuditEvent(
            org_id="org1",
            operation="EXPORT",
            entity_type="request",
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
            idempotency_key=f"export_noncanonical_{next(iter(detail))}",
            detail=detail,
        )

        with pytest.raises(ValueError, match="status|route"):
            storage.append_audit_event(event)
        return

    purge_id = _begin_purge(storage, f"purge_noncanonical_{next(iter(detail))}")
    with pytest.raises(ValueError, match="status|route"):
        storage.record_purge_target(
            purge_id=purge_id,
            target_name="request",
            target_ref="all",
            phase="delete",
            status="running",
            detail=detail,
        )


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
        pytest.param(
            "target_snapshot",
            "prepare_targets",
            "17",
            "target_ref",
            id="snapshot-rejects-row-ref",
        ),
        pytest.param(
            "target_snapshot",
            "prepare_targets",
            "",
            "target_ref",
            id="snapshot-rejects-empty-ref",
        ),
        pytest.param(
            "target_snapshot",
            "delete",
            "all",
            "target_snapshot",
            id="snapshot-rejects-wrong-phase",
        ),
        pytest.param(
            "request",
            "prepare_targets",
            "all",
            "prepare_targets",
            id="request-rejects-prepare-targets",
        ),
        pytest.param("request", "delete", "all", None, id="aggregate-delete-all"),
        pytest.param(
            "request",
            "hide_for_rebuild",
            "all",
            "hide_for_rebuild",
            id="request-rejects-hide",
        ),
        pytest.param(
            "profile",
            "rebuild_without_erased_sources",
            "all",
            "rebuild_without_erased_sources",
            id="profile-rejects-rebuild",
        ),
        pytest.param(
            "interaction",
            "delete",
            REQUEST_REF,
            "target_ref",
            id="interaction-delete-rejects-non-all-ref",
        ),
        pytest.param(
            "agent_playbook",
            "hide_for_rebuild",
            "17",
            None,
            id="row-target-hide-internal-id",
        ),
        pytest.param(
            "agent_playbook",
            "prepare_targets",
            "17",
            "prepare_targets",
            id="agent-playbook-rejects-prepare-targets",
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
        pytest.param(
            "agent_playbook",
            "rebuild",
            "19",
            "phase",
            id="rebuild-phase-rejected",
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
        (
            "org1",
            "purge_unique",
            "target_snapshot",
            "all",
            "prepare_targets",
            "complete",
        )
    ]


def test_gc_governance_retention_deletes_expired_audit_rows_in_batches(storage):
    old_event = AuditEvent(
        org_id=storage.org_id,
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="audit_old",
        created_at=1,
    )
    newer_event = AuditEvent(
        org_id=storage.org_id,
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="audit_new",
    )
    other_org_event = AuditEvent(
        org_id="org2",
        operation="EXPORT",
        entity_type="request",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
        idempotency_key="audit_other",
        created_at=1,
    )
    storage.append_audit_event(old_event)
    storage.append_audit_event(newer_event)
    other_storage = SQLiteStorage(org_id="org2", db_path=storage.db_path)
    other_storage.append_audit_event(other_org_event)

    deleted = storage.gc_governance_retention(
        config=GovernanceRetentionConfig(
            audit_events_retention_enabled=True,
            audit_events_retention_days=1,
            audit_events_delete_batch_limit=1,
        )
    )

    assert deleted == 1
    assert [event.idempotency_key for event in storage.list_audit_events()] == [
        "audit_new"
    ]
    assert [event.idempotency_key for event in other_storage.list_audit_events()] == [
        "audit_other"
    ]


def test_gc_governance_retention_noops_when_audit_retention_disabled(storage):
    storage.append_audit_event(
        AuditEvent(
            org_id=storage.org_id,
            operation="EXPORT",
            entity_type="request",
            subject_ref=SUBJECT_REF,
            request_ref=REQUEST_REF,
            idempotency_key="audit_disabled",
            created_at=1,
        )
    )

    assert storage.gc_governance_retention(config=GovernanceRetentionConfig()) == 0
    assert len(storage.list_audit_events()) == 1
