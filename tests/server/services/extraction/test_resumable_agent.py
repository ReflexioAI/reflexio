from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.resumable_agent import (
    FINISH_EXTRACTION_TOOL_NAME,
    ResumableExtractionAgent,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookList,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
)


@pytest.fixture
def storage():
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(org_id="org_1", db_path=f"{temp_dir}/reflexio.db")


def _agent_run(run_id: str, extractor_kind: str = "profile") -> AgentRunRecord:
    return AgentRunRecord(
        id=run_id,
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind=extractor_kind,
            extractor_name=f"default_{extractor_kind}_extractor",
            user_id="user_1",
            request_id=f"request_{run_id}",
            agent_version="v1",
            source="api",
            source_interaction_ids=[1, 2],
            window_start_interaction_id=1,
            window_end_interaction_id=2,
            extractor_config_hash="hash_1",
        ),
        status=AgentRunStatus.RUNNING,
        generation_request_snapshot={"request_id": f"request_{run_id}"},
    )


def test_profile_instruction_prompt_is_resumable_by_default():
    """The extraction loop is always-on, so the resumable instruction prompt
    (v1.1.0) is the active version of ``profile_update_instruction_start`` —
    there is no longer a separate resumable prompt-version override."""
    prompt_manager = PromptManager()

    assert prompt_manager.get_active_version("profile_update_instruction_start") == (
        "1.1.0"
    )
    rendered = prompt_manager.render_prompt(
        "profile_update_instruction_start",
        {
            "agent_context_prompt": "agent context",
            "context_prompt": "",
            "extraction_definition_prompt": "user facts",
            "metadata_definition_prompt": None,
        },
    )

    assert "Resumable Extraction Mode" in rendered
    assert "ask_human" in rendered
    assert "finish_extraction" in rendered


def test_resumable_agent_finishes_profile_output(
    monkeypatch,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    make_tc, _make_stop = tool_call_completion
    response = make_tc(
        FINISH_EXTRACTION_TOOL_NAME,
        {
            "profiles": [
                {
                    "content": "User prefers AWS ECS deployments.",
                    "time_to_live": "infinity",
                }
            ]
        },
    )
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage)

    with patch("litellm.completion", side_effect=[response]):
        result = agent.start(
            run=_agent_run("run_profile"),
            messages=[{"role": "user", "content": "extract profiles"}],
            output_schema=StructuredProfilesOutput,
        )

    assert result.finished_reason == "finish_tool"
    assert isinstance(result.output, StructuredProfilesOutput)
    assert result.output.profiles is not None
    assert result.output.profiles[0].content == "User prefers AWS ECS deployments."
    stored = storage.get_agent_run("run_profile")
    assert stored is not None
    assert stored.status == AgentRunStatus.AGENT_COMPLETED
    assert stored.max_steps_remaining == 7
    assert stored.committed_output == {
        "profiles": [
            {
                "content": "User prefers AWS ECS deployments.",
                "time_to_live": "infinity",
                "metadata": None,
                "source_span": None,
                "notes": None,
                "reader_angle": None,
            }
        ]
    }


def test_resumable_agent_finishes_playbook_output(
    monkeypatch,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    make_tc, _make_stop = tool_call_completion
    response = make_tc(
        FINISH_EXTRACTION_TOOL_NAME,
        {
            "playbooks": [
                {
                    "trigger": "Deploying services",
                    "content": "Prefer ECS for this environment.",
                    "rationale": "The team standardizes on AWS.",
                }
            ]
        },
    )
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage)

    with patch("litellm.completion", side_effect=[response]):
        result = agent.start(
            run=_agent_run("run_playbook", extractor_kind="playbook"),
            messages=[{"role": "user", "content": "extract playbooks"}],
            output_schema=StructuredPlaybookList,
        )

    assert result.finished_reason == "finish_tool"
    assert isinstance(result.output, StructuredPlaybookList)
    assert result.output.playbooks[0].content == "Prefer ECS for this environment."
    stored = storage.get_agent_run("run_playbook")
    assert stored is not None
    assert stored.status == AgentRunStatus.AGENT_COMPLETED
    assert stored.committed_output is not None
    assert stored.committed_output["playbooks"][0]["trigger"] == "Deploying services"


def test_resumable_agent_marks_run_failed_on_loop_error(monkeypatch, storage):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    monkeypatch.setattr(
        client,
        "generate_chat_response",
        MagicMock(side_effect=RuntimeError("provider failed")),
    )
    agent = ResumableExtractionAgent(client=client, storage=storage)

    result = agent.start(
        run=_agent_run("run_error"),
        messages=[{"role": "user", "content": "extract profiles"}],
        output_schema=StructuredProfilesOutput,
    )

    assert result.finished_reason == "error"
    stored = storage.get_agent_run("run_error")
    assert stored is not None
    assert stored.status == AgentRunStatus.FAILED
    assert stored.committed_output is None
    assert stored.last_error == "Extraction agent did not finish: error"


def test_resumable_agent_resume_injects_resolved_tool_result(
    monkeypatch,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    run = storage.create_agent_run(_agent_run("run_resume"))
    question = "Which deployment standard should be treated as canonical?"
    scope = human_feedback_scope("org_1")
    now = datetime(2026, 5, 28, tzinfo=UTC)
    pending = storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_resume",
            org_id="org_1",
            user_id="user_1",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text=question,
                answer_format="short text",
            ),
            status=PendingToolCallStatus.PENDING,
            question_text=question,
            answer_format="short text",
            expires_at=now + timedelta(hours=1),
            cache_until=now + timedelta(minutes=5),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(
            run_id=run.id,
            pending_tool_call_id=pending.id,
        )
    )
    storage.update_agent_run_status(run.id, AgentRunStatus.FINALIZED_PENDING_TOOL)
    resolved = storage.resolve_pending_tool_call(
        pending.id,
        result={"answer": "Use AWS ECS as the deployment standard."},
        resolved_at=now,
        valid_for_seconds=3600,
    )
    assert resolved is not None
    claimed = storage.claim_ready_agent_run(
        org_id=run.binding.org_id, worker_id="worker_1"
    )
    assert claimed is not None
    assert claimed.status == AgentRunStatus.RESUMING

    make_tc, _make_stop = tool_call_completion
    response = make_tc(
        FINISH_EXTRACTION_TOOL_NAME,
        {
            "profiles": [
                {
                    "content": "User deployment standard is AWS ECS.",
                    "time_to_live": "infinity",
                }
            ]
        },
    )
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage)

    with patch("litellm.completion", side_effect=[response]):
        result = agent.resume(
            run=claimed,
            messages=[{"role": "user", "content": "resume extraction"}],
            output_schema=StructuredProfilesOutput,
            resolved_tool_calls=[resolved],
        )

    assert result.finished_reason == "finish_tool"
    assert isinstance(result.output, StructuredProfilesOutput)
    assert result.output.profiles is not None
    assert result.output.profiles[0].content == "User deployment standard is AWS ECS."
    assert any(
        message["role"] == "user"
        and "Use AWS ECS as the deployment standard." in str(message["content"])
        for message in result.messages
    )
    stored = storage.get_agent_run(run.id)
    assert stored is not None
    assert stored.status == AgentRunStatus.AGENT_COMPLETED


def test_resumable_agent_resume_uses_persisted_step_budget(
    monkeypatch,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    run = storage.create_agent_run(
        AgentRunRecord(
            id="run_budget",
            binding=_agent_run("run_budget").binding,
            status=AgentRunStatus.RESUMING,
            generation_request_snapshot={"request_id": "request_run_budget"},
            max_steps_remaining=1,
        )
    )
    now = datetime(2026, 5, 28, tzinfo=UTC)
    resolved = PendingToolCallRecord(
        id="ptc_budget",
        org_id="org_1",
        user_id="user_1",
        scope=human_feedback_scope("org_1"),
        scope_hash=build_scope_hash(human_feedback_scope("org_1")),
        tool_name="ask_human",
        dedup_key=build_pending_tool_call_dedup_key(
            tool_name="ask_human",
            question_text="What is the deployment target?",
        ),
        status=PendingToolCallStatus.RESOLVED,
        question_text="What is the deployment target?",
        result={"answer": "AWS ECS"},
        resolved_at=now,
        expires_at=now + timedelta(hours=1),
        cache_until=now + timedelta(minutes=5),
        valid_until=now + timedelta(days=30),
    )
    make_tc, _make_stop = tool_call_completion
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage, max_steps=8)

    with patch(
        "litellm.completion",
        side_effect=[make_tc("unknown_tool", {"value": "ignored"})],
    ) as completion:
        result = agent.resume(
            run=run,
            messages=[{"role": "user", "content": "resume extraction"}],
            output_schema=StructuredProfilesOutput,
            resolved_tool_calls=[resolved],
        )

    assert completion.call_count == 1
    assert result.finished_reason == "max_steps"
    stored = storage.get_agent_run(run.id)
    assert stored is not None
    assert stored.status == AgentRunStatus.FAILED
    assert stored.max_steps_remaining == 0
    assert stored.last_error == "Extraction agent did not finish: max_steps"
