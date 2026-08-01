from __future__ import annotations

import threading
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentPlaybook,
    AgentPlaybookSourceWindow,
    AgentSuccessEvaluationResult,
    Interaction,
    Request,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import PlaybookStatus
from reflexio.models.api_schema.domain.governance import UserEraseResult
from reflexio.models.api_schema.retriever_schema import SearchAgentPlaybookRequest
from reflexio.models.config_schema import SearchMode
from reflexio.server.services.governance import service as governance_service_module
from reflexio.server.services.governance.config import governance_subject_ref
from reflexio.server.services.governance.service import GovernanceService
from reflexio.server.services.storage.error import SubjectWriteBarrierError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _now() -> int:
    return int(datetime.now(UTC).timestamp())


def _request(*, request_id: str, user_id: str, session_id: str) -> Request:
    return Request(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        created_at=_now(),
        source="governance-local-e2e",
        agent_version="agent-v1",
    )


def _interaction(
    *,
    user_id: str,
    request_id: str,
    content: str,
    interaction_id: int = 0,
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id=user_id,
        request_id=request_id,
        created_at=_now(),
        content=content,
    )


def _profile(
    *, profile_id: str, user_id: str, content: str, request_id: str
) -> UserProfile:
    return UserProfile(
        profile_id=profile_id,
        user_id=user_id,
        content=content,
        last_modified_timestamp=_now(),
        generated_from_request_id=request_id,
    )


def _user_playbook(
    *,
    user_id: str,
    request_id: str,
    content: str,
    trigger: str,
    rationale: str,
) -> UserPlaybook:
    return UserPlaybook(
        user_id=user_id,
        agent_version="agent-v1",
        request_id=request_id,
        playbook_name="shared-governance-playbook",
        created_at=_now(),
        content=content,
        trigger=trigger,
        rationale=rationale,
        source="governance-local-e2e",
    )


def _agent_playbook(*, content: str, trigger: str, rationale: str) -> AgentPlaybook:
    return AgentPlaybook(
        playbook_name="shared-governance-playbook",
        agent_version="agent-v1",
        created_at=_now(),
        content=content,
        trigger=trigger,
        rationale=rationale,
        playbook_status=PlaybookStatus.APPROVED,
    )


def _eval_result(
    *, user_id: str, session_id: str, agent_version: str
) -> AgentSuccessEvaluationResult:
    return AgentSuccessEvaluationResult(
        user_id=user_id,
        session_id=session_id,
        agent_version=agent_version,
        evaluation_name="governance-local-e2e",
        is_success=True,
    )


@pytest.fixture
def storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[SQLiteStorage, None, None]:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "test-governance-secret")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id="org-local", db_path=str(tmp_path / "governance.db"))


def test_local_governance_e2e_erases_exports_audits_and_preserves_org_agent_playbooks(
    storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(governance_service_module, "_USER_PLAYBOOK_PAGE_SIZE", 1)
    alice_request_id = "req-alice"
    bob_request_id = "req-bob"

    storage.add_request(
        _request(request_id=alice_request_id, user_id="alice", session_id="sess-alice")
    )
    storage.save_agent_success_evaluation_results(
        [
            _eval_result(
                user_id="alice", session_id="sess-alice", agent_version="agent-v1"
            )
        ]
    )
    storage.add_user_interaction(
        "alice",
        _interaction(
            user_id="alice",
            request_id=alice_request_id,
            content="aliceprivateinteractiontoken",
        ),
    )
    storage.add_user_profile(
        "alice",
        [
            _profile(
                profile_id="profile-alice",
                user_id="alice",
                content="aliceprivateprofiletoken",
                request_id=alice_request_id,
            )
        ],
    )
    alice_playbook = _user_playbook(
        user_id="alice",
        request_id=alice_request_id,
        content="aliceuniquesourcetoken",
        trigger="alicetriggerunique",
        rationale="alicerationaleunique",
    )
    storage.save_user_playbooks([alice_playbook])
    alice_orphan_playbook = _user_playbook(
        user_id="alice",
        request_id=alice_request_id,
        content="aliceorphansourcetoken",
        trigger="aliceorphantrigger",
        rationale="aliceorphanrationale",
    )
    storage.save_user_playbooks([alice_orphan_playbook])

    storage.add_request(
        _request(request_id=bob_request_id, user_id="bob", session_id="sess-bob")
    )
    storage.save_agent_success_evaluation_results(
        [_eval_result(user_id="bob", session_id="sess-bob", agent_version="agent-v1")]
    )
    storage.add_user_interaction(
        "bob",
        _interaction(
            user_id="bob",
            request_id=bob_request_id,
            content="bobprivateinteractiontoken",
        ),
    )
    storage.add_user_profile(
        "bob",
        [
            _profile(
                profile_id="profile-bob",
                user_id="bob",
                content="bobprivateprofiletoken",
                request_id=bob_request_id,
            )
        ],
    )
    bob_playbook = _user_playbook(
        user_id="bob",
        request_id=bob_request_id,
        content="bobuniquesourcetoken",
        trigger="bobtriggerunique",
        rationale="bobrationaleunique",
    )
    storage.save_user_playbooks([bob_playbook])

    shared_playbook = storage.save_agent_playbooks(
        [
            _agent_playbook(
                content="aliceuniquesourcetoken\nbobuniquesourcetoken",
                trigger="alicetriggerunique\nbobtriggerunique",
                rationale="alicerationaleunique\nbobrationaleunique",
            )
        ]
    )[0]
    storage.set_source_windows_for_agent_playbook(
        shared_playbook.agent_playbook_id,
        [
            AgentPlaybookSourceWindow(
                user_playbook_id=alice_playbook.user_playbook_id,
                source_interaction_ids=[101],
            ),
            AgentPlaybookSourceWindow(
                user_playbook_id=bob_playbook.user_playbook_id,
                source_interaction_ids=[202],
            ),
        ],
    )
    orphan_playbook = storage.save_agent_playbooks(
        [
            _agent_playbook(
                content="aliceorphansourcetoken",
                trigger="aliceorphantrigger",
                rationale="aliceorphanrationale",
            )
        ]
    )[0]
    storage.set_source_windows_for_agent_playbook(
        orphan_playbook.agent_playbook_id,
        [
            AgentPlaybookSourceWindow(
                user_playbook_id=alice_orphan_playbook.user_playbook_id,
                source_interaction_ids=[303],
            ),
        ],
    )

    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
    )

    exported = service.export_user(user_id="alice", request_id="export-request-1")

    assert exported.subject_ref == governance_subject_ref(
        storage.org_id,
        "alice",
        "test-governance-secret",
    )
    assert exported.export_id.startswith("export_")
    assert [profile["profile_id"] for profile in exported.bundle["profiles"]] == [
        "profile-alice"
    ]
    assert [
        interaction["request_id"] for interaction in exported.bundle["interactions"]
    ] == [alice_request_id]
    assert [request["request_id"] for request in exported.bundle["requests"]] == [
        alice_request_id
    ]
    assert {
        playbook["user_playbook_id"] for playbook in exported.bundle["user_playbooks"]
    } == {
        alice_playbook.user_playbook_id,
        alice_orphan_playbook.user_playbook_id,
    }

    export_events = [
        event
        for event in storage.list_audit_events(subject_ref=exported.subject_ref)
        if event.operation == "EXPORT"
    ]
    assert len(export_events) == 1
    assert export_events[0].detail == {"count": 6}
    export_dump = export_events[0].model_dump_json()
    assert "alice" not in export_dump
    assert alice_request_id not in export_dump

    erased = service.erase_user(user_id="alice", request_id="erase-request-1")

    assert erased.status == "complete"
    assert erased.subject_ref == exported.subject_ref
    assert erased.deleted_counts["interactions"] == 1
    assert erased.deleted_counts["profiles"] == 1
    assert erased.deleted_counts["requests"] == 1
    assert erased.deleted_counts["user_playbooks"] == 2
    assert erased.deleted_counts["agent_success_evaluation_results"] == 1
    assert erased.rebuilt_agent_playbook_ids == []

    assert storage.get_user_interaction("alice") == []
    assert storage.get_user_profile("alice") == []
    assert storage.get_requests_by_session("alice", "sess-alice") == []
    assert storage.get_user_playbooks(user_id="alice", limit=10) == []
    assert (
        storage.get_agent_success_evaluation_result_ids(
            "alice",
            "sess-alice",
            "governance-local-e2e",
            "agent-v1",
        )
        == []
    )

    assert len(storage.get_user_interaction("bob")) == 1
    assert len(storage.get_user_profile("bob")) == 1
    assert len(storage.get_requests_by_session("bob", "sess-bob")) == 1
    assert (
        storage.get_agent_success_evaluation_result_ids(
            "bob",
            "sess-bob",
            "governance-local-e2e",
            "agent-v1",
        )
        != []
    )
    delete_targets = {
        target.target_name: target
        for target in storage.list_purge_targets(erased.purge_id, phase="delete")
    }
    assert delete_targets["agent_success_evaluation_result"].status == "complete"
    assert delete_targets["agent_success_evaluation_result"].deleted_count == 1
    assert len(storage.get_user_playbooks(user_id="bob", limit=10)) == 1

    preserved_playbook = storage.get_agent_playbook_by_id(
        shared_playbook.agent_playbook_id
    )
    assert preserved_playbook is not None
    assert preserved_playbook.content == shared_playbook.content
    assert preserved_playbook.trigger == shared_playbook.trigger
    assert preserved_playbook.rationale == shared_playbook.rationale
    assert storage.get_source_windows_for_agent_playbook(
        shared_playbook.agent_playbook_id
    ) == [
        AgentPlaybookSourceWindow(
            user_playbook_id=bob_playbook.user_playbook_id,
            source_interaction_ids=[202],
        )
    ]
    preserved_orphan_playbook = storage.get_agent_playbook_by_id(
        orphan_playbook.agent_playbook_id
    )
    assert preserved_orphan_playbook is not None
    assert preserved_orphan_playbook.content == orphan_playbook.content
    assert preserved_orphan_playbook.trigger == orphan_playbook.trigger
    assert preserved_orphan_playbook.rationale == orphan_playbook.rationale
    assert orphan_playbook.agent_playbook_id in {
        playbook.agent_playbook_id for playbook in storage.get_agent_playbooks(limit=10)
    }
    assert (
        storage.get_source_windows_for_agent_playbook(orphan_playbook.agent_playbook_id)
        == []
    )
    hard_delete_events = [
        event
        for event in storage.get_lineage_events(
            entity_type="agent_playbook",
            entity_id=str(orphan_playbook.agent_playbook_id),
        )
        if event.op == "hard_delete"
    ]
    assert hard_delete_events == []

    alice_search_results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="aliceuniquesourcetoken",
            top_k=10,
            search_mode=SearchMode.FTS,
        )
    )
    assert [playbook.agent_playbook_id for playbook in alice_search_results] == [
        shared_playbook.agent_playbook_id
    ]
    orphan_search_results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="aliceorphansourcetoken",
            top_k=10,
            search_mode=SearchMode.FTS,
        )
    )
    assert [playbook.agent_playbook_id for playbook in orphan_search_results] == [
        orphan_playbook.agent_playbook_id
    ]
    bob_search_results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="bobuniquesourcetoken",
            top_k=10,
            search_mode=SearchMode.FTS,
        )
    )
    assert [playbook.agent_playbook_id for playbook in bob_search_results] == [
        shared_playbook.agent_playbook_id
    ]

    erase_events = [
        event
        for event in storage.list_audit_events(subject_ref=exported.subject_ref)
        if event.operation == "ERASE" and event.status == "ok"
    ]
    assert len(erase_events) == 1
    assert erase_events[0].idempotency_key == erased.purge_id
    erase_dump = erase_events[0].model_dump_json()
    assert "alice" not in erase_dump
    assert alice_request_id not in erase_dump

    retried = service.erase_user(user_id="alice", request_id="erase-request-1")

    assert retried.purge_id == erased.purge_id
    assert retried.status == "complete"
    assert retried.deleted_counts == erased.deleted_counts
    assert retried.rebuilt_agent_playbook_ids == erased.rebuilt_agent_playbook_ids
    erase_events_after_retry = [
        event
        for event in storage.list_audit_events(subject_ref=exported.subject_ref)
        if event.operation == "ERASE" and event.status == "ok"
    ]
    assert len(erase_events_after_retry) == 1


def test_governance_service_persists_actor_context_in_audit(
    storage: SQLiteStorage,
) -> None:
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
    )

    service.export_user(
        user_id="alice",
        request_id="export-actor",
        actor_context={
            "actor_type": "jwt",
            "actor_ref": "actref_v1_1234567890abcdef1234567890abcdef",
        },
    )

    audit_events = storage.list_audit_events()
    event = audit_events[-1]
    assert event.actor_type == "jwt"
    assert event.actor_ref == "actref_v1_1234567890abcdef1234567890abcdef"


def test_governance_erase_persists_actor_context_in_audit(
    storage: SQLiteStorage,
) -> None:
    storage.add_request(
        _request(request_id="erase-actor-req", user_id="alice", session_id="sess-actor")
    )
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
    )

    service.erase_user(
        user_id="alice",
        request_id="erase-actor",
        actor_context={
            "actor_type": "api_token",
            "actor_ref": "actref_v1_1234567890abcdef1234567890abcdef",
        },
    )

    erase_events = [
        event for event in storage.list_audit_events() if event.operation == "ERASE"
    ]
    assert len(erase_events) == 1
    assert erase_events[0].actor_type == "api_token"
    assert erase_events[0].actor_ref == "actref_v1_1234567890abcdef1234567890abcdef"


def test_completed_erase_retry_reconstructs_response(
    storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "test-governance-secret")
    storage.add_request(
        _request(request_id="retry-req", user_id="alice", session_id="retry-sess")
    )
    storage.add_user_interaction(
        "alice",
        _interaction(user_id="alice", request_id="retry-req", content="retry-token"),
    )
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
    )

    first = service.erase_user(user_id="alice", request_id="erase-retry")
    second = service.erase_user(user_id="alice", request_id="erase-retry")

    assert second.status == "complete"
    assert second.purge_id == first.purge_id
    assert second.deleted_counts == first.deleted_counts
    assert second.rebuilt_agent_playbook_ids == first.rebuilt_agent_playbook_ids


def test_erase_fails_fast_when_service_and_storage_ref_secrets_differ(
    storage: SQLiteStorage,
) -> None:
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="different-service-secret",
    )

    with pytest.raises(RuntimeError, match="ref_secret must match"):
        service.erase_user(user_id="alice", request_id="erase-mismatch")


def test_export_fails_fast_when_service_and_storage_ref_secrets_differ(
    storage: SQLiteStorage,
) -> None:
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="different-service-secret",
    )

    with pytest.raises(RuntimeError, match="ref_secret must match"):
        service.export_user(user_id="alice", request_id="export-mismatch")


@pytest.mark.parametrize(
    ("barrier_sql", "match"),
    [
        (
            "DELETE FROM subject_write_barriers WHERE subject_ref = ?",
            "matching subject barrier",
        ),
        (
            "UPDATE subject_write_barriers SET status = 'failed' WHERE subject_ref = ?",
            "erased subject barrier",
        ),
    ],
)
def test_completed_erase_retry_fails_closed_without_erased_barrier(
    storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
    barrier_sql: str,
    match: str,
) -> None:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "test-governance-secret")
    storage.add_request(
        _request(
            request_id="retry-closed-req",
            user_id="alice",
            session_id="retry-closed-sess",
        )
    )
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
    )

    first = service.erase_user(user_id="alice", request_id="erase-retry-closed")
    storage.conn.execute(barrier_sql, (first.subject_ref,))
    storage.conn.commit()

    with pytest.raises(ValueError, match=match):
        service.erase_user(user_id="alice", request_id="erase-retry-closed")


def test_barrier_acquisition_failure_marks_purge_failed_where_possible(
    storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
    )

    def _raise_begin(*args, **kwargs):
        raise RuntimeError("forced barrier begin failure")

    monkeypatch.setattr(storage, "begin_subject_erasure_barrier", _raise_begin)

    with pytest.raises(RuntimeError, match="forced barrier begin failure"):
        service.erase_user(user_id="alice", request_id="erase-begin-failure")

    failed_purges = [
        row
        for row in storage.conn.execute(
            "SELECT status, error_code, error_detail FROM purge_operations"
        ).fetchall()
        if row["status"] == "failed"
    ]
    assert len(failed_purges) == 1
    assert failed_purges[0]["error_code"] == "governance_erase_failed"
    assert failed_purges[0]["error_detail"] == "RuntimeError"


def test_second_erase_conflict_preserves_original_barrier_and_write_block(
    storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "test-governance-secret")
    subject_ref = governance_subject_ref(
        storage.org_id,
        "alice",
        "test-governance-secret",
    )
    first_purge = storage.begin_purge_operation(
        purge_id="purge_conflict_first",
        idempotency_key="idem_conflict_first",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000061",
    )
    storage.begin_subject_erasure_barrier(subject_ref, first_purge.purge_id)
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
    )

    with pytest.raises(
        ValueError, match="Existing barrier purge_id must match the requested purge_id"
    ):
        service.erase_user(user_id="alice", request_id="erase-conflict-second")

    barrier = storage.get_subject_write_barrier(subject_ref)
    assert barrier is not None
    assert barrier.purge_id == first_purge.purge_id
    assert barrier.status == "erasing"
    with pytest.raises(SubjectWriteBarrierError):
        storage.add_request(
            _request(
                request_id="req-after-conflict",
                user_id="alice",
                session_id="sess-after-conflict",
            )
        )

    failed_purges = list(
        storage.conn.execute(
            "SELECT purge_id, status FROM purge_operations WHERE status = 'failed'"
        ).fetchall()
    )
    assert len(failed_purges) == 1
    assert failed_purges[0]["purge_id"] != first_purge.purge_id
    assert failed_purges[0]["status"] == "failed"


def test_governance_erase_marks_purge_failed_when_workflow_raises(
    storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
    )

    def _raise_prepare(*args, **kwargs) -> None:
        raise RuntimeError("forced prepare failure")

    monkeypatch.setattr(storage, "prepare_governance_erase_targets", _raise_prepare)

    with pytest.raises(RuntimeError, match="forced prepare failure"):
        service.erase_user(user_id="alice", request_id="erase-failure-request")

    failed_purges = [
        row
        for row in storage.conn.execute(
            "SELECT status, error_code, error_detail FROM purge_operations"
        ).fetchall()
        if row["status"] == "failed"
    ]
    assert len(failed_purges) == 1
    assert failed_purges[0]["error_code"] == "governance_erase_failed"
    assert failed_purges[0]["error_detail"] == "RuntimeError"
    failed_barriers = [
        row
        for row in storage.conn.execute(
            "SELECT status, error_code, error_detail FROM subject_write_barriers"
        ).fetchall()
        if row["status"] == "failed"
    ]
    assert len(failed_barriers) == 1
    assert failed_barriers[0]["error_code"] == "governance_erase_failed"
    assert failed_barriers[0]["error_detail"] == "RuntimeError"


def test_subject_erasure_lifecycle_retry_is_idempotent_and_counted(
    storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetrySafeLifecycle:
        calls = 0
        deleted = 0

        def erase_subject(
            self,
            *,
            storage: SQLiteStorage,
            subject_ref: str,
            purge_id: str,
        ) -> None:
            del subject_ref
            self.calls += 1
            target = next(
                target
                for target in storage.list_purge_targets(purge_id, phase="delete")
                if target.target_name == "offline_tuner_reward_label"
            )
            if target.deleted_count == 2:
                return
            self.deleted += 2
            storage.record_purge_target(
                purge_id=purge_id,
                target_name="offline_tuner_reward_label",
                target_ref="all",
                phase="delete",
                status="complete",
                detail={"count": 2},
                deleted_count=2,
            )

    lifecycle = RetrySafeLifecycle()
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
        subject_erasure_lifecycle=lifecycle,
    )
    complete = storage.complete_subject_erasure_barrier_after_empty_check
    completion_attempts = 0

    def fail_after_first_lifecycle(*args, **kwargs):
        nonlocal completion_attempts
        completion_attempts += 1
        if completion_attempts == 1:
            raise RuntimeError("forced post-lifecycle failure")
        return complete(*args, **kwargs)

    monkeypatch.setattr(
        storage,
        "complete_subject_erasure_barrier_after_empty_check",
        fail_after_first_lifecycle,
    )

    with pytest.raises(RuntimeError, match="forced post-lifecycle failure"):
        service.erase_user(user_id="alice", request_id="erase-lifecycle-retry")

    retried = service.erase_user(user_id="alice", request_id="erase-lifecycle-retry")

    assert retried.status == "complete"
    assert retried.deleted_counts["offline_tuner_reward_labels"] == 2
    assert lifecycle.calls == 1
    assert lifecycle.deleted == 2
    snapshot = next(
        target
        for target in storage.list_purge_targets(
            retried.purge_id, phase="prepare_targets"
        )
        if target.target_name == "target_snapshot"
    )
    assert snapshot.detail is not None
    assert snapshot.detail["status"] == "complete"


def test_duplicate_erase_waits_for_lifecycle_winner_beyond_old_deadline(
    storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowSingleUseLifecycle:
        def __init__(self) -> None:
            self.first_call_started = threading.Event()
            self.release_first_call = threading.Event()
            self._lock = threading.Lock()
            self.calls = 0
            self.duplicate_rejections = 0

        def erase_subject(
            self,
            *,
            storage: SQLiteStorage,
            subject_ref: str,
            purge_id: str,
        ) -> None:
            del storage, subject_ref, purge_id
            with self._lock:
                self.calls += 1
                call_number = self.calls
            if call_number == 1:
                self.first_call_started.set()
                assert self.release_first_call.wait(timeout=5)
                return
            self.duplicate_rejections += 1
            raise RuntimeError("provider lifecycle already running")

    lifecycle = SlowSingleUseLifecycle()
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
        subject_erasure_lifecycle=lifecycle,
    )
    monkeypatch.setattr(governance_service_module, "_DUPLICATE_ERASE_POLL_SECONDS", 0.0)
    real_sleep = governance_service_module.time.sleep

    def release_winner_on_duplicate_poll(_seconds: float) -> None:
        lifecycle.release_first_call.set()
        real_sleep(0.001)

    monkeypatch.setattr(
        governance_service_module.time, "sleep", release_winner_on_duplicate_poll
    )
    winner_results: list[UserEraseResult] = []
    winner_errors: list[BaseException] = []

    def run_winner() -> None:
        try:
            winner_results.append(
                service.erase_user(user_id="alice", request_id="erase-slow-duplicate")
            )
        except BaseException as exc:
            winner_errors.append(exc)

    winner = threading.Thread(target=run_winner)
    winner.start()
    assert lifecycle.first_call_started.wait(timeout=5)

    try:
        duplicate = service.erase_user(
            user_id="alice", request_id="erase-slow-duplicate"
        )
    finally:
        lifecycle.release_first_call.set()
        winner.join(timeout=5)

    assert not winner.is_alive()
    assert winner_errors == []
    assert len(winner_results) == 1
    winner_result = winner_results[0]
    assert duplicate.status == "complete"
    assert duplicate.purge_id == winner_result.purge_id
    assert storage.get_purge_operation(duplicate.purge_id).status == "complete"
    barrier = storage.get_subject_write_barrier(duplicate.subject_ref)
    assert barrier is not None
    assert barrier.status == "erased"
    assert lifecycle.calls == 1
    assert lifecycle.duplicate_rejections == 0


def test_pending_duplicate_erase_has_one_durable_execution_owner(
    storage: SQLiteStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_begin = storage.begin_purge_operation
    both_pending = threading.Barrier(2)

    def synchronized_begin(*args, **kwargs):
        purge = original_begin(*args, **kwargs)
        both_pending.wait(timeout=5)
        return purge

    monkeypatch.setattr(storage, "begin_purge_operation", synchronized_begin)

    class CountingLifecycle:
        def __init__(self) -> None:
            self.calls = 0
            self._lock = threading.Lock()

        def erase_subject(self, **_kwargs) -> None:
            with self._lock:
                self.calls += 1

    lifecycle = CountingLifecycle()
    service = GovernanceService(
        storage=storage,
        org_id=storage.org_id,
        ref_secret="test-governance-secret",
        subject_erasure_lifecycle=lifecycle,
    )
    results: list[UserEraseResult] = []
    errors: list[BaseException] = []

    def erase() -> None:
        try:
            results.append(
                service.erase_user(user_id="alice", request_id="erase-pending-race")
            )
        except BaseException as exc:
            errors.append(exc)

    callers = [threading.Thread(target=erase) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=5)

    assert all(not caller.is_alive() for caller in callers)
    assert errors == []
    assert len(results) == 2
    assert results[0].purge_id == results[1].purge_id
    assert all(result.status == "complete" for result in results)
    assert lifecycle.calls == 1
    assert storage.get_purge_operation(results[0].purge_id).status == "complete"


def test_session_export_paginates_by_returned_rows_when_requests_are_missing() -> None:
    class _Storage:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def get_sessions(self, *, user_id: str, top_k: int, offset: int):
            self.calls.append(offset)
            if offset == 0:
                return {
                    "session-a": [
                        *[SimpleNamespace(request=None) for _ in range(999)],
                        SimpleNamespace(request=SimpleNamespace(request_id="req-1")),
                    ]
                }
            return {}

    storage = _Storage()
    service = GovernanceService(storage=storage, org_id="org", ref_secret="secret")

    requests, sessions = service._load_user_requests_and_sessions("user-1")

    assert storage.calls == [0, 1000]
    assert [request.request_id for request in requests] == ["req-1"]
    assert sessions == [{"session_id": "session-a", "request_ids": ["req-1"]}]
