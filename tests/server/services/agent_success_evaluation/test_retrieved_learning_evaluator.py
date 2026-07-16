"""Unit tests for the retrieved-learning relevance/impact evaluator."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain import (
    AgentPlaybook,
    LineageContext,
    PlaybookStatus,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.llm.litellm_client import StructuredOutputRepairError
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.agent_success_evaluation.components.retrieved_learning_evaluator import (
    MAX_CANONICAL_CANDIDATES,
    RetrievedLearningEvaluator,
    RetrievedLearningImpactOutput,
    RetrievedLearningImpactVerdict,
    RetrievedLearningRelevanceOutput,
    RetrievedLearningRelevanceVerdict,
    _verdict_coverage_error,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base.retrieved_learning_state import (
    BoundedRetrievedLearningSnapshot,
    SnapshotInteraction,
)

USER = "eval-user"
SESSION = "eval-session"


@pytest.fixture
def storage() -> Generator[SQLiteStorage]:
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(org_id="rle_eval_test", db_path=f"{tmp_dir}/reflexio.db")


def _make_evaluator(
    storage: SQLiteStorage,
    llm_client: MagicMock,
    success_definition: str = "resolve the ticket in one reply",
) -> RetrievedLearningEvaluator:
    request_context = SimpleNamespace(
        storage=storage,
        configurator=SimpleNamespace(
            get_config=lambda: Config(storage_config=StorageConfigSQLite())
        ),
        prompt_manager=PromptManager(),
    )
    return RetrievedLearningEvaluator(
        request_context=request_context,  # type: ignore[arg-type]
        llm_client=llm_client,
        agent_context="test agent",
        success_definition=success_definition,
    )


def _snapshot(
    refs_by_interaction: dict[int, list[tuple[str, str]]],
) -> BoundedRetrievedLearningSnapshot:
    snapshot = BoundedRetrievedLearningSnapshot(
        earliest_request_created_at=1_700_000_000, agent_version="v1"
    )
    for interaction_id, refs in refs_by_interaction.items():
        snapshot.interactions.append(
            SnapshotInteraction(
                interaction_id=interaction_id,
                role="Assistant",
                content="hello",
                created_at=1_700_000_001,
                refs=list(refs),
            )
        )
        snapshot.raw_attachment_count += len(refs)
    return snapshot


def _seed_all_kinds(storage: SQLiteStorage) -> tuple[str, int, int]:
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


def _echoing_llm() -> MagicMock:
    """LLM double that returns exact-coverage verdicts for any chunk."""
    llm = MagicMock()

    def respond(*, messages, model, response_format, **_kwargs):  # noqa: ARG001
        import re

        # Scan only the [Retrieved Learnings] payload — the [Output] example
        # also contains a "learning_ref" placeholder line.
        content = messages[0]["content"]
        payload = content[
            content.find("[Retrieved Learnings]") : content.find("[Output]")
        ]
        refs = list(dict.fromkeys(re.findall(r'"learning_ref": "([^"]+)"', payload)))
        if response_format is RetrievedLearningRelevanceOutput:
            return RetrievedLearningRelevanceOutput(
                verdicts=[
                    RetrievedLearningRelevanceVerdict(
                        learning_ref=ref, is_relevant=True, relevance_reason="fits"
                    )
                    for ref in refs
                ]
            )
        return RetrievedLearningImpactOutput(
            verdicts=[
                RetrievedLearningImpactVerdict(
                    learning_ref=ref, impact="positive", impact_reason="helped"
                )
                for ref in refs
            ]
        )

    llm.generate_chat_response.side_effect = respond
    return llm


def test_resolution_preserves_archived_attached_id(storage: SQLiteStorage) -> None:
    profile_id, upb_id, apb_id = _seed_all_kinds(storage)
    # Archive the user playbook after publication. It is no longer retrievable,
    # but it remains the exact learning that was injected into the response.
    assert storage.archive_user_playbook_by_id(USER, upb_id)
    llm = _echoing_llm()
    evaluator = _make_evaluator(storage, llm)
    snapshot = _snapshot(
        {
            1: [
                ("profile", profile_id),
                ("user_playbook", str(upb_id)),
                ("agent_playbook", str(apb_id)),
                ("user_playbook", "999999"),  # never existed
                ("user_playbook", "not-a-number"),  # malformed id
            ]
        }
    )
    with patch.object(
        storage,
        "get_agent_playbooks_by_ids",
        wraps=storage.get_agent_playbooks_by_ids,
    ) as bulk_agent_lookup:
        run = evaluator.evaluate(USER, SESSION, "evaluated-v2", snapshot)
    assert run.outcome == "evaluated"
    assert run.proposed_status == "complete"
    assert {(r.kind, r.learning_id) for r in run.rows} == {
        ("profile", profile_id),
        ("user_playbook", str(upb_id)),
        ("agent_playbook", str(apb_id)),
    }
    assert run.diagnostics["invalid_ref_count"] == 1
    row = next(r for r in run.rows if r.kind == "profile")
    assert row.interaction_id == 1
    assert row.interaction_created_at == 1_700_000_001
    assert row.is_relevant is True and row.impact == "positive"
    assert row.agent_version == "evaluated-v2"
    assert row.created_at == 1_700_000_000
    bulk_agent_lookup.assert_called_once()


def test_repeated_learning_is_evaluated_once_per_interaction(
    storage: SQLiteStorage,
) -> None:
    profile_id, _, _ = _seed_all_kinds(storage)
    llm = _echoing_llm()
    run = _make_evaluator(storage, llm).evaluate(
        USER,
        SESSION,
        "v1",
        _snapshot(
            {
                10: [("profile", profile_id), ("profile", profile_id)],
                20: [("profile", profile_id)],
            }
        ),
    )

    assert [(row.interaction_id, row.kind, row.learning_id) for row in run.rows] == [
        (10, "profile", profile_id),
        (20, "profile", profile_id),
    ]
    prompt = llm.generate_chat_response.call_args_list[0].kwargs["messages"][0][
        "content"
    ]
    assert '"learning_ref": "10:profile:prof-1"' in prompt
    assert '"target_interaction_id": 10' in prompt
    assert "[interaction_id=20] Assistant: hello" in prompt


def test_unapproved_agent_playbook_is_evaluated_when_attached(
    storage: SQLiteStorage,
) -> None:
    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="pending playbook",
                agent_version="v1",
                content="c",
                trigger="t",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )
    apb_id = storage.get_agent_playbooks(limit=10)[0].agent_playbook_id
    evaluator = _make_evaluator(storage, _echoing_llm())
    run = evaluator.evaluate(
        USER, SESSION, "v1", _snapshot({1: [("agent_playbook", str(apb_id))]})
    )
    assert run.outcome == "evaluated"
    assert [(row.kind, row.learning_id) for row in run.rows] == [
        ("agent_playbook", str(apb_id))
    ]


def test_expired_profile_is_evaluated_when_attached(storage: SQLiteStorage) -> None:
    profile_id, _, _ = _seed_all_kinds(storage)
    storage.conn.execute(
        "UPDATE profiles SET expiration_timestamp = 1 WHERE profile_id = ?",
        (profile_id,),
    )
    storage.conn.commit()
    assert storage.expire_active_profiles(now=2) == 1

    run = _make_evaluator(storage, _echoing_llm()).evaluate(
        USER, SESSION, "v1", _snapshot({1: [("profile", profile_id)]})
    )

    assert [(row.kind, row.learning_id) for row in run.rows] == [
        ("profile", profile_id)
    ]


def test_resolution_uses_original_superseded_ids_and_content(
    storage: SQLiteStorage,
) -> None:
    profile_id, upb_id, apb_id = _seed_all_kinds(storage)
    storage.add_user_profile(
        USER,
        [
            UserProfile(
                profile_id="prof-successor",
                user_id=USER,
                content="successor profile content",
                last_modified_timestamp=2,
                generated_from_request_id="r2",
            )
        ],
    )
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id=USER,
                playbook_name="successor checklist",
                request_id="r2",
                agent_version="v1",
                content="successor user playbook content",
            )
        ]
    )
    successor_upb_id = next(
        playbook.user_playbook_id
        for playbook in storage.get_user_playbooks(user_id=USER, limit=10)
        if playbook.user_playbook_id != upb_id
    )
    storage.save_agent_playbooks(
        [
            AgentPlaybook(
                playbook_name="successor deploy",
                agent_version="v1",
                content="successor agent playbook content",
                playbook_status=PlaybookStatus.PENDING,
            )
        ]
    )
    successor_apb_id = next(
        playbook.agent_playbook_id
        for playbook in storage.get_agent_playbooks(limit=10)
        if playbook.agent_playbook_id != apb_id
    )
    context = LineageContext(op_kind="revise", actor="test", request_id="r2")
    assert storage.supersede_record(
        entity_type="profile",
        incumbent_id=profile_id,
        successor_id="prof-successor",
        context=context,
    )
    assert storage.supersede_record(
        entity_type="user_playbook",
        incumbent_id=str(upb_id),
        successor_id=str(successor_upb_id),
        context=context,
    )
    assert storage.supersede_record(
        entity_type="agent_playbook",
        incumbent_id=str(apb_id),
        successor_id=str(successor_apb_id),
        context=context,
    )

    llm = _echoing_llm()
    run = _make_evaluator(storage, llm).evaluate(
        USER,
        SESSION,
        "v1",
        _snapshot(
            {
                1: [
                    ("profile", profile_id),
                    ("user_playbook", str(upb_id)),
                    ("agent_playbook", str(apb_id)),
                ]
            }
        ),
    )

    assert {(row.kind, row.learning_id) for row in run.rows} == {
        ("profile", profile_id),
        ("user_playbook", str(upb_id)),
        ("agent_playbook", str(apb_id)),
    }
    judge_payloads = "\n".join(
        call.kwargs["messages"][0]["content"]
        for call in llm.generate_chat_response.call_args_list
    )
    assert "prefers concise answers" in judge_payloads
    assert "always produce a checklist" in judge_payloads
    assert "verify before deploy" in judge_payloads
    assert "successor profile content" not in judge_payloads
    assert "successor user playbook content" not in judge_payloads
    assert "successor agent playbook content" not in judge_payloads


def test_empty_candidates_make_zero_llm_calls(storage: SQLiteStorage) -> None:
    llm = _echoing_llm()
    evaluator = _make_evaluator(storage, llm)
    run = evaluator.evaluate(USER, SESSION, "v1", _snapshot({1: []}))
    assert run.outcome == "evaluated"
    assert run.rows == []
    llm.generate_chat_response.assert_not_called()


def test_attachment_limit_fails_with_zero_llm_calls(storage: SQLiteStorage) -> None:
    llm = _echoing_llm()
    evaluator = _make_evaluator(storage, llm)
    snapshot = BoundedRetrievedLearningSnapshot(attachment_limit_exceeded=True)
    run = evaluator.evaluate(USER, SESSION, "v1", snapshot)
    assert run.outcome == "failed"
    assert run.diagnostics["error_type"] == "attachment_limit_exceeded"
    llm.generate_chat_response.assert_not_called()


def test_candidate_limit_fails_with_zero_llm_calls(storage: SQLiteStorage) -> None:
    llm = _echoing_llm()
    evaluator = _make_evaluator(storage, llm)
    refs = [("user_playbook", str(n)) for n in range(1, MAX_CANONICAL_CANDIDATES + 2)]
    run = evaluator.evaluate(USER, SESSION, "v1", _snapshot({1: refs}))
    assert run.outcome == "failed"
    assert run.diagnostics["error_type"] == "candidate_limit_exceeded"
    llm.generate_chat_response.assert_not_called()


def test_one_bounded_repair_then_chunk_failure(storage: SQLiteStorage) -> None:
    """A judge with a persistent coverage error degrades after client repair exhausts."""
    profile_id, _, _ = _seed_all_kinds(storage)
    llm = MagicMock()

    def bad_relevance_good_impact(
        *, messages, model, response_format, structured_output_validator
    ):  # noqa: ARG001
        if response_format is RetrievedLearningRelevanceOutput:
            bad_output = RetrievedLearningRelevanceOutput(
                verdicts=[
                    RetrievedLearningRelevanceVerdict(
                        learning_ref="unknown:ref",
                        is_relevant=True,
                        relevance_reason="x",
                    )
                ]
            )
            errors = structured_output_validator(bad_output)
            raise StructuredOutputRepairError(
                "repair exhausted",
                failure_kind="semantic",
                model=model,
                parsed_output=bad_output,
                validation_errors=tuple(errors),
            )
        return RetrievedLearningImpactOutput(
            verdicts=[
                RetrievedLearningImpactVerdict(
                    learning_ref=f"1:profile:{profile_id}",
                    impact="neutral",
                    impact_reason="none",
                )
            ]
        )

    llm.generate_chat_response.side_effect = bad_relevance_good_impact
    evaluator = _make_evaluator(storage, llm)
    run = evaluator.evaluate(
        USER, SESSION, "v1", _snapshot({1: [("profile", profile_id)]})
    )
    assert run.outcome == "evaluated"
    assert run.proposed_status == "degraded"
    assert run.diagnostics["failed_relevance_chunks"] == 1
    assert run.diagnostics["failed_impact_chunks"] == 0
    assert llm.generate_chat_response.call_count == 2
    row = run.rows[0]
    assert row.is_relevant is None and row.relevance_reason == ""
    assert row.impact == "neutral"


def test_all_judges_failed_reports_failure(storage: SQLiteStorage) -> None:
    profile_id, _, _ = _seed_all_kinds(storage)
    llm = MagicMock()
    llm.generate_chat_response.return_value = None
    evaluator = _make_evaluator(storage, llm)
    run = evaluator.evaluate(
        USER, SESSION, "v1", _snapshot({1: [("profile", profile_id)]})
    )
    assert run.outcome == "failed"
    assert run.diagnostics["error_type"] == "all_judges_failed"


def test_duplicate_refs_across_interactions_are_distinct_occurrences(
    storage: SQLiteStorage,
) -> None:
    profile_id, _, _ = _seed_all_kinds(storage)
    llm = _echoing_llm()
    evaluator = _make_evaluator(storage, llm)
    snapshot = _snapshot({1: [("profile", profile_id)], 2: [("profile", profile_id)]})
    run = evaluator.evaluate(USER, SESSION, "v1", snapshot)
    assert [row.interaction_id for row in run.rows] == [1, 2]
    # Both occurrences still fit in one relevance + one impact chunk.
    assert llm.generate_chat_response.call_count == 2


def test_success_definition_reaches_only_impact_prompt(
    storage: SQLiteStorage,
) -> None:
    """The definition of success anchors impact but not relevance."""
    profile_id, _, _ = _seed_all_kinds(storage)
    marker = "SUCCESS-DEF-MARKER resolve in one reply"
    prompts_by_format: dict[type, str] = {}

    llm = _echoing_llm()
    inner = llm.generate_chat_response.side_effect

    def recording(*, messages, model, response_format, **kwargs):
        prompts_by_format[response_format] = messages[0]["content"]
        return inner(
            messages=messages, model=model, response_format=response_format, **kwargs
        )

    llm.generate_chat_response.side_effect = recording
    evaluator = _make_evaluator(storage, llm, success_definition=marker)
    run = evaluator.evaluate(
        USER, SESSION, "v1", _snapshot({1: [("profile", profile_id)]})
    )
    assert run.outcome == "evaluated"
    assert marker in prompts_by_format[RetrievedLearningImpactOutput]
    assert marker not in prompts_by_format[RetrievedLearningRelevanceOutput]


def test_verdict_coverage_error_names_all_problems() -> None:
    error = _verdict_coverage_error(
        ["a", "a", "z"],
        expected={"a", "b"},
    )
    assert error is not None
    assert "missing refs ['b']" in error
    assert "duplicate refs ['a']" in error
    assert "unknown refs ['z']" in error
    assert _verdict_coverage_error(["a", "b"], expected={"a", "b"}) is None
