from __future__ import annotations

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
from reflexio.models.api_schema.retriever_schema import SearchAgentPlaybookRequest
from reflexio.models.config_schema import SearchMode
from reflexio.server.services.governance import service as governance_service_module
from reflexio.server.services.governance.service import GovernanceService
from reflexio.server.services.governance.subject_refs import subject_ref
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
def storage(tmp_path: Path) -> Generator[SQLiteStorage, None, None]:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id="org-local", db_path=str(tmp_path / "governance.db"))


def test_local_governance_e2e_erases_exports_audits_and_rebuilds_shared_aggregate(
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

    assert exported.subject_ref == subject_ref("alice", "test-governance-secret")
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
    assert set(erased.rebuilt_agent_playbook_ids) == {
        shared_playbook.agent_playbook_id,
        orphan_playbook.agent_playbook_id,
    }

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

    rebuilt_playbook = storage.get_agent_playbook_by_id(
        shared_playbook.agent_playbook_id
    )
    assert rebuilt_playbook is not None
    assert "aliceuniquesourcetoken" not in rebuilt_playbook.content
    assert rebuilt_playbook.content == "bobuniquesourcetoken"
    assert rebuilt_playbook.trigger == "bobtriggerunique"
    assert rebuilt_playbook.rationale == "bobrationaleunique"
    assert storage.get_source_windows_for_agent_playbook(
        shared_playbook.agent_playbook_id
    ) == [
        AgentPlaybookSourceWindow(
            user_playbook_id=bob_playbook.user_playbook_id,
            source_interaction_ids=[202],
        )
    ]
    assert storage.get_agent_playbook_by_id(orphan_playbook.agent_playbook_id) is None
    assert orphan_playbook.agent_playbook_id not in {
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
    assert len(hard_delete_events) == 1
    assert hard_delete_events[0].request_id == erased.purge_id

    assert (
        storage.search_agent_playbooks(
            SearchAgentPlaybookRequest(
                query="aliceuniquesourcetoken",
                top_k=10,
                search_mode=SearchMode.FTS,
            )
        )
        == []
    )
    assert (
        storage.search_agent_playbooks(
            SearchAgentPlaybookRequest(
                query="aliceorphansourcetoken",
                top_k=10,
                search_mode=SearchMode.FTS,
            )
        )
        == []
    )
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
    assert retried.deleted_counts == {}
    assert retried.rebuilt_agent_playbook_ids == []
    erase_events_after_retry = [
        event
        for event in storage.list_audit_events(subject_ref=exported.subject_ref)
        if event.operation == "ERASE" and event.status == "ok"
    ]
    assert len(erase_events_after_retry) == 1


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
