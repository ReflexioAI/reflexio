from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

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
from reflexio.models.api_schema.domain.governance import AuditEvent
from reflexio.server.services.governance.config import governance_subject_ref
from reflexio.server.services.storage.error import SubjectWriteBarrierError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


def _now() -> int:
    return int(datetime.now(UTC).timestamp())


def _storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SQLiteStorage:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "barrier-secret")
    monkeypatch.setattr(SQLiteStorage, "_get_embedding", lambda *_args: [0.0] * 512)
    return SQLiteStorage(org_id="org-barrier", db_path=str(tmp_path / "barrier.db"))


def _mark_all_completion_targets(storage: SQLiteStorage, purge_id: str) -> None:
    storage.record_purge_target(
        purge_id,
        target_name="target_snapshot",
        phase="prepare_targets",
        status="complete",
        target_ref="all",
        detail={"prepared": True},
    )
    for target_name in (
        "request",
        "interaction",
        "profile",
        "user_playbook",
        "agent_success_evaluation_result",
        "profile_purge",
        "user_playbook_purge",
    ):
        storage.record_purge_target(
            purge_id,
            target_name=target_name,
            phase="delete",
            status="complete",
            target_ref="all",
            detail={"count": 0},
        )


def test_barrier_blocks_request_interaction_and_profile_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    purge = storage.begin_purge_operation(
        purge_id="purge_barrier",
        idempotency_key="idem_barrier",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_11111111111111111111111111111111",
    )

    barrier = storage.begin_subject_erasure_barrier(subject_ref, purge.purge_id)

    assert barrier.status == "erasing"
    with pytest.raises(SubjectWriteBarrierError):
        storage.add_request(
            Request(
                request_id="req-after-barrier",
                user_id="alice",
                session_id="sess-1",
                source="test",
                agent_version="agent-v1",
                created_at=_now(),
            )
        )
    with pytest.raises(SubjectWriteBarrierError):
        storage.add_user_interaction(
            "alice",
            Interaction(
                user_id="alice",
                request_id="req-after-barrier",
                content="blocked interaction",
                created_at=_now(),
            ),
        )
    with pytest.raises(SubjectWriteBarrierError):
        storage.add_user_profile(
            "alice",
            [
                UserProfile(
                    profile_id="profile-after-barrier",
                    user_id="alice",
                    content="blocked profile",
                    generated_from_request_id="req-after-barrier",
                    last_modified_timestamp=_now(),
                )
            ],
        )


def test_barrier_blocks_playbook_eval_and_source_window_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    purge = storage.begin_purge_operation(
        purge_id="purge_barrier",
        idempotency_key="idem_barrier",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_11111111111111111111111111111111",
    )

    user_playbook = UserPlaybook(
        user_id="alice",
        agent_version="agent-v1",
        request_id="req-before-barrier",
        playbook_name="barrier-test",
        created_at=_now(),
        content="initial content",
        trigger="initial trigger",
        rationale="initial rationale",
        source="test",
    )
    storage.save_user_playbooks([user_playbook])
    agent_playbook = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="barrier-test",
                agent_version="agent-v1",
                created_at=_now(),
                content="aggregate content",
                trigger="aggregate trigger",
                rationale="aggregate rationale",
            )
        ]
    )[0]

    storage.begin_subject_erasure_barrier(subject_ref, purge.purge_id)

    with pytest.raises(SubjectWriteBarrierError):
        storage.save_user_playbooks(
            [
                UserPlaybook(
                    user_id="alice",
                    agent_version="agent-v1",
                    request_id="req-after-barrier",
                    playbook_name="barrier-test",
                    created_at=_now(),
                    content="blocked content",
                    trigger="blocked trigger",
                    rationale="blocked rationale",
                    source="test",
                )
            ]
        )
    with pytest.raises(SubjectWriteBarrierError):
        storage.save_agent_success_evaluation_results(
            [
                AgentSuccessEvaluationResult(
                    user_id="alice",
                    session_id="sess-1",
                    agent_version="agent-v1",
                    evaluation_name="barrier-test",
                    is_success=False,
                )
            ]
        )
    with pytest.raises(SubjectWriteBarrierError):
        storage.set_source_windows_for_agent_playbook(
            agent_playbook.agent_playbook_id,
            [
                AgentPlaybookSourceWindow(
                    user_playbook_id=user_playbook.user_playbook_id,
                    source_interaction_ids=[101],
                )
            ],
        )


def test_begin_subject_erasure_barrier_requires_matching_purge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    alice_subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    bob_subject_ref = governance_subject_ref("org-barrier", "bob", "barrier-secret")
    purge = storage.begin_purge_operation(
        purge_id="purge_barrier_match",
        idempotency_key="idem_barrier_match",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=alice_subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000021",
    )

    with pytest.raises(ValueError, match="subject_ref must match"):
        storage.begin_subject_erasure_barrier(bob_subject_ref, purge.purge_id)

    with pytest.raises(ValueError, match="not found"):
        storage.begin_subject_erasure_barrier(
            alice_subject_ref, "purge_barrier_missing"
        )


def test_fail_subject_erasure_barrier_requires_matching_barrier_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    first_purge = storage.begin_purge_operation(
        purge_id="purge_barrier_first",
        idempotency_key="idem_barrier_first",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000051",
    )
    second_purge = storage.begin_purge_operation(
        purge_id="purge_barrier_second",
        idempotency_key="idem_barrier_second",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000052",
    )

    storage.begin_subject_erasure_barrier(subject_ref, first_purge.purge_id)

    with pytest.raises(ValueError, match="matching barrier"):
        storage.fail_subject_erasure_barrier(
            subject_ref,
            second_purge.purge_id,
            error_code="governance_erase_failed",
            error_detail="ValueError",
        )

    barrier = storage.get_subject_write_barrier(subject_ref)
    assert barrier is not None
    assert barrier.purge_id == first_purge.purge_id
    assert barrier.status == "erasing"
    with pytest.raises(SubjectWriteBarrierError):
        storage.add_request(
            Request(
                request_id="req-still-blocked",
                user_id="alice",
                session_id="sess-still-blocked",
                source="test",
                agent_version="agent-v1",
                created_at=_now(),
            )
        )


def test_guarded_completion_requires_empty_subject_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    purge = storage.begin_purge_operation(
        purge_id="purge_guarded_complete",
        idempotency_key="idem_guarded_complete",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_0123456789abcdef0123456789abcdef",
    )
    storage.add_request(
        Request(
            request_id="req-before-barrier",
            user_id="alice",
            session_id="sess-1",
            source="test",
            agent_version="agent-v1",
            created_at=_now(),
        )
    )
    storage.begin_subject_erasure_barrier(subject_ref, purge.purge_id)
    storage.record_purge_target(
        purge.purge_id,
        target_name="target_snapshot",
        phase="prepare_targets",
        status="complete",
        target_ref="all",
        detail={"prepared": True},
    )

    with pytest.raises(ValueError, match="same-subject rows remain"):
        storage.complete_subject_erasure_barrier_after_empty_check(
            purge.purge_id,
            AuditEvent(
                org_id="org-barrier",
                operation="ERASE",
                entity_type="request",
                subject_ref=subject_ref,
                request_ref="reqref_v1_0123456789abcdef0123456789abcdef",
                idempotency_key=purge.purge_id,
                detail={"deleted_counts": {}, "rebuilt_agent_playbook_ids": []},
            ),
        )


def test_guarded_completion_requires_empty_legacy_null_subject_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    purge = storage.begin_purge_operation(
        purge_id="purge_guarded_legacy",
        idempotency_key="idem_guarded_legacy",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000031",
    )
    storage.add_request(
        Request(
            request_id="req-legacy-before-barrier",
            user_id="alice",
            session_id="sess-legacy",
            source="test",
            agent_version="agent-v1",
            created_at=_now(),
        )
    )
    storage.conn.execute(
        """UPDATE requests
           SET governance_subject_ref = NULL
           WHERE request_id = ?""",
        ("req-legacy-before-barrier",),
    )
    storage.conn.commit()

    storage.begin_subject_erasure_barrier(subject_ref, purge.purge_id)
    storage.record_purge_target(
        purge.purge_id,
        target_name="target_snapshot",
        phase="prepare_targets",
        status="complete",
        target_ref="all",
        detail={"prepared": True},
    )

    with pytest.raises(ValueError, match="same-subject rows remain"):
        storage.complete_subject_erasure_barrier_after_empty_check(
            purge.purge_id,
            AuditEvent(
                org_id="org-barrier",
                operation="ERASE",
                entity_type="request",
                subject_ref=subject_ref,
                request_ref="reqref_v1_00000000000000000000000000000031",
                idempotency_key=purge.purge_id,
                detail={"deleted_counts": {}, "rebuilt_agent_playbook_ids": []},
            ),
        )


def test_guarded_completion_requires_existing_erasing_subject_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    purge = storage.begin_purge_operation(
        purge_id="purge_missing_barrier",
        idempotency_key="idem_missing_barrier",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000032",
    )
    _mark_all_completion_targets(storage, purge.purge_id)

    with pytest.raises(ValueError, match="subject erasure barrier is missing"):
        storage.complete_subject_erasure_barrier_after_empty_check(
            purge.purge_id,
            AuditEvent(
                org_id="org-barrier",
                operation="ERASE",
                entity_type="request",
                subject_ref=subject_ref,
                request_ref="reqref_v1_00000000000000000000000000000032",
                idempotency_key=purge.purge_id,
                detail={"deleted_counts": {}, "rebuilt_agent_playbook_ids": []},
            ),
        )


def test_guarded_completion_rejects_failed_subject_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    purge = storage.begin_purge_operation(
        purge_id="purge_failed_barrier",
        idempotency_key="idem_failed_barrier",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000035",
    )
    storage.begin_subject_erasure_barrier(subject_ref, purge.purge_id)
    storage.fail_subject_erasure_barrier(
        subject_ref,
        purge.purge_id,
        error_code="test_failed_barrier",
        error_detail="RuntimeError",
    )
    _mark_all_completion_targets(storage, purge.purge_id)

    with pytest.raises(ValueError, match="subject erasure barrier is missing"):
        storage.complete_subject_erasure_barrier_after_empty_check(
            purge.purge_id,
            AuditEvent(
                org_id="org-barrier",
                operation="ERASE",
                entity_type="request",
                subject_ref=subject_ref,
                request_ref="reqref_v1_00000000000000000000000000000035",
                idempotency_key=purge.purge_id,
                detail={"deleted_counts": {}, "rebuilt_agent_playbook_ids": []},
            ),
        )


def test_barrier_blocks_profile_update_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    profile = UserProfile(
        profile_id="profile-before-barrier",
        user_id="alice",
        content="before",
        generated_from_request_id="req-before-barrier",
        last_modified_timestamp=_now(),
    )
    storage.add_user_profile("alice", [profile])
    purge = storage.begin_purge_operation(
        purge_id="purge_profile_update",
        idempotency_key="idem_profile_update",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000033",
    )
    storage.begin_subject_erasure_barrier(subject_ref, purge.purge_id)

    with pytest.raises(SubjectWriteBarrierError):
        storage.update_user_profile_tags("alice", profile.profile_id, ["blocked"])
    with pytest.raises(SubjectWriteBarrierError):
        storage.archive_profile_by_id("alice", profile.profile_id)
    with pytest.raises(SubjectWriteBarrierError):
        storage.supersede_profiles_by_ids(
            "alice",
            [profile.profile_id],
            request_id="req-supersede-blocked",
        )


def test_barrier_blocks_user_playbook_update_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    playbook = UserPlaybook(
        user_id="alice",
        agent_version="agent-v1",
        request_id="req-before-barrier",
        playbook_name="barrier-update",
        created_at=_now(),
        content="before",
        trigger="trigger",
        rationale="rationale",
        source="test",
    )
    storage.save_user_playbooks([playbook])
    purge = storage.begin_purge_operation(
        purge_id="purge_playbook_update",
        idempotency_key="idem_playbook_update",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000034",
    )
    storage.begin_subject_erasure_barrier(subject_ref, purge.purge_id)

    with pytest.raises(SubjectWriteBarrierError):
        storage.archive_user_playbook_by_id("alice", playbook.user_playbook_id)
    with pytest.raises(SubjectWriteBarrierError):
        storage.update_user_playbook(playbook.user_playbook_id, tags=["blocked"])
    with pytest.raises(SubjectWriteBarrierError):
        storage.supersede_user_playbooks_by_ids(
            [playbook.user_playbook_id],
            request_id="req-supersede-playbook-blocked",
        )


def test_source_window_write_blocks_legacy_null_subject_ref_user_playbook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = _storage(tmp_path, monkeypatch)
    subject_ref = governance_subject_ref("org-barrier", "alice", "barrier-secret")
    purge = storage.begin_purge_operation(
        purge_id="purge_source_window_legacy",
        idempotency_key="idem_source_window_legacy",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=subject_ref,
        request_ref="reqref_v1_00000000000000000000000000000041",
    )

    user_playbook = UserPlaybook(
        user_id="alice",
        agent_version="agent-v1",
        request_id="req-before-barrier",
        playbook_name="legacy-source-window",
        created_at=_now(),
        content="legacy content",
        trigger="legacy trigger",
        rationale="legacy rationale",
        source="test",
    )
    storage.save_user_playbooks([user_playbook])
    storage.conn.execute(
        """UPDATE user_playbooks
           SET governance_subject_ref = NULL
           WHERE user_playbook_id = ?""",
        (user_playbook.user_playbook_id,),
    )
    storage.conn.commit()
    agent_playbook = storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="legacy-source-window",
                agent_version="agent-v1",
                created_at=_now(),
                content="aggregate content",
                trigger="aggregate trigger",
                rationale="aggregate rationale",
            )
        ]
    )[0]

    storage.begin_subject_erasure_barrier(subject_ref, purge.purge_id)

    with pytest.raises(SubjectWriteBarrierError):
        storage.set_source_windows_for_agent_playbook(
            agent_playbook.agent_playbook_id,
            [
                AgentPlaybookSourceWindow(
                    user_playbook_id=user_playbook.user_playbook_id,
                    source_interaction_ids=[404],
                )
            ],
        )
