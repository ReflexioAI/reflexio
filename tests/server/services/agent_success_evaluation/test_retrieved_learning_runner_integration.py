"""End-to-end runner integration for retrieved-learning evaluation.

Real SQLite storage + real prompt manager + real ``LiteLLMClient`` against
the globally mocked ``litellm.completion`` (the mock echoes the learning_refs
listed in each judge prompt). The agent-success service itself is stubbed —
its behavior is covered by its own suite; these tests pin the retrieved
phase's storage effects and the ``GroupEvaluationOutcome`` contract.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    Interaction,
    PlaybookStatus,
    Request,
    RetrievedLearning,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.agent_success_evaluation.runner import (
    run_group_evaluation,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

ORG = "rle_runner_test"
USER = "u1"
SESSION = "sess-1"
OLD_TS = 1_700_000_000


@pytest.fixture
def storage() -> Generator[SQLiteStorage]:
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(org_id=ORG, db_path=f"{tmp_dir}/reflexio.db")


def _request_context(storage: SQLiteStorage) -> SimpleNamespace:
    return SimpleNamespace(
        storage=storage,
        configurator=SimpleNamespace(
            get_config=lambda: Config(storage_config=StorageConfigSQLite())
        ),
        prompt_manager=PromptManager(),
    )


def _seed_learnings(storage: SQLiteStorage) -> tuple[str, int, int]:
    storage.add_user_profile(
        USER,
        [
            UserProfile(
                profile_id="prof-1",
                user_id=USER,
                content="prefers concise answers",
                last_modified_timestamp=1,
                generated_from_request_id="r1",
            )
        ],
    )
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id=USER,
                playbook_name="checklist",
                request_id="r1",
                agent_version="v1",
                content="always produce a checklist",
                trigger="when deploying",
            )
        ]
    )
    upb_id = storage.get_user_playbooks(user_id=USER, limit=10)[0].user_playbook_id
    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="safe deploy",
                agent_version="v1",
                content="verify before deploy",
                trigger="when deploying",
                playbook_status=PlaybookStatus.APPROVED,
            )
        ]
    )
    apb_id = storage.get_agent_playbooks(limit=10)[0].agent_playbook_id
    return "prof-1", upb_id, apb_id


def _seed_session(storage: SQLiteStorage, refs: list[RetrievedLearning]) -> None:
    storage.add_request(
        Request(
            request_id="r1",
            user_id=USER,
            session_id=SESSION,
            created_at=OLD_TS,
            agent_version="v1",
        )
    )
    storage.add_user_interactions_bulk(
        USER,
        [
            Interaction(
                user_id=USER,
                request_id="r1",
                content="give me a deployment checklist",
                role="User",
                created_at=OLD_TS + 1,
            ),
            Interaction(
                user_id=USER,
                request_id="r1",
                content="1. build 2. migrate 3. verify",
                role="Assistant",
                created_at=OLD_TS + 2,
                retrieved_learnings=refs,
            ),
        ],
    )


def _stub_agent_success() -> MagicMock:
    service = MagicMock()
    service.has_run_failures.return_value = False
    service.last_run_save_failed = False
    service.last_run_saved_result_count = 1
    return service


def _run(storage: SQLiteStorage, *, force: bool = False):
    with patch(
        "reflexio.server.services.agent_success_evaluation.runner."
        "AgentSuccessEvaluationService",
        return_value=_stub_agent_success(),
    ):
        return run_group_evaluation(
            org_id=ORG,
            user_id=USER,
            session_id=SESSION,
            agent_version="v1",
            source=None,
            request_context=_request_context(storage),  # type: ignore[arg-type]
            llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
            force_regenerate=force,
        )


def test_publish_to_verdicts_end_to_end(storage: SQLiteStorage) -> None:
    profile_id, upb_id, apb_id = _seed_learnings(storage)
    _seed_session(
        storage,
        refs=[
            RetrievedLearning(kind="profile", learning_id=profile_id),
            RetrievedLearning(kind="user_playbook", learning_id=str(upb_id)),
            RetrievedLearning(kind="agent_playbook", learning_id=str(apb_id)),
            # Attached but nonexistent — must never produce a row.
            RetrievedLearning(kind="user_playbook", learning_id="999999"),
        ],
    )

    outcome = _run(storage)
    assert outcome.agent_success_status == "complete"
    assert outcome.retrieved_learning_status == "complete"
    assert outcome.retrieved_learning_fingerprint

    rows = storage.get_retrieved_learning_evaluation_results(session_id=SESSION)
    assert {(r.kind, r.learning_id) for r in rows} == {
        ("profile", profile_id),
        ("user_playbook", str(upb_id)),
        ("agent_playbook", str(apb_id)),
    }
    for row in rows:
        assert row.is_relevant is True
        assert row.impact == "positive"
        assert row.agent_version == "v1"
        assert row.created_at == OLD_TS


def test_rerun_short_circuits_and_force_is_idempotent(
    storage: SQLiteStorage,
) -> None:
    profile_id, _, _ = _seed_learnings(storage)
    _seed_session(
        storage, refs=[RetrievedLearning(kind="profile", learning_id=profile_id)]
    )
    first = _run(storage)
    assert first.retrieved_learning_status == "complete"
    result_ids = {
        r.result_id
        for r in storage.get_retrieved_learning_evaluation_results(session_id=SESSION)
    }

    # Unforced rerun: agent-success marker short-circuits and the retrieved
    # terminal state matches the unchanged fingerprint — no new rows.
    second = _run(storage)
    assert second.agent_success_status == "skipped"
    assert second.retrieved_learning_status == "complete"
    assert {
        r.result_id
        for r in storage.get_retrieved_learning_evaluation_results(session_id=SESSION)
    } == result_ids

    # Forced rerun replaces the snapshot without duplicating rows.
    third = _run(storage, force=True)
    assert third.retrieved_learning_status == "complete"
    rows = storage.get_retrieved_learning_evaluation_results(session_id=SESSION)
    assert len(rows) == 1


def test_transcript_only_publish_defeats_terminal_shortcut(
    storage: SQLiteStorage,
) -> None:
    profile_id, _, _ = _seed_learnings(storage)
    _seed_session(
        storage, refs=[RetrievedLearning(kind="profile", learning_id=profile_id)]
    )
    first = _run(storage)
    assert first.retrieved_learning_status == "complete"

    # A transcript-only publish (no attachments) changes the fingerprint.
    storage.add_user_interactions_bulk(
        USER,
        [
            Interaction(
                user_id=USER,
                request_id="r1",
                content="thanks!",
                role="User",
                created_at=OLD_TS + 3,
            )
        ],
    )
    second = _run(storage)
    # Fresh evaluation ran (not the terminal shortcut) and re-completed at
    # the new fingerprint.
    assert second.retrieved_learning_status == "complete"
    assert second.retrieved_learning_fingerprint != (
        first.retrieved_learning_fingerprint
    )


def test_session_without_attachments_is_not_applicable(
    storage: SQLiteStorage,
) -> None:
    _seed_session(storage, refs=[])
    outcome = _run(storage)
    assert outcome.agent_success_status == "complete"
    assert outcome.retrieved_learning_status == "not_applicable"
    assert storage.get_retrieved_learning_evaluation_results(session_id=SESSION) == []
    assert storage.get_matching_retrieved_learning_terminal_state(
        USER, SESSION, outcome.retrieved_learning_fingerprint or ""
    )
