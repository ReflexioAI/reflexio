from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    StructuredOutputRepairError,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.durable_learning.user_lease import user_extraction_lease
from reflexio.server.services.extraction import (
    resumable_agent as resumable_agent_module,
)
from reflexio.server.services.extraction.history_inputs import HistoricalInputs
from reflexio.server.services.extraction.pending_tool_call_dispatch import (
    PendingToolCallToolContext,
    create_ask_human_tool,
    create_attach_pending_info_request_tool,
)
from reflexio.server.services.extraction.resumable_agent import (
    ResumableExtractionAgent,
    create_pending_info_tools_for_extractor_kind,
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
        database = SQLiteStorage(org_id="org_1", db_path=f"{temp_dir}/reflexio.db")
        from reflexio.models.api_schema.service_schemas import Interaction, Request

        database.add_request(
            Request(
                request_id="source_request",
                user_id="user_1",
                session_id="source_session",
                created_at=1000,
            )
        )
        for identifier in (1, 2):
            database._insert_interaction(
                Interaction(
                    interaction_id=identifier,
                    user_id="user_1",
                    request_id="source_request",
                    content=f"Synthetic source {identifier}",
                    created_at=1000 + identifier,
                )
            )
        yield database


def _adopt_continuation(storage, ready):
    """Direct-agent setup under an existing actual owner; no queue discovery."""
    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        owned_agent_run_write,
    )

    adopted = adopt_agent_run_for_execution(storage, ready)
    with owned_agent_run_write(storage, adopted):
        current = storage.update_agent_run_status(
            adopted.id,
            AgentRunStatus.RESUMING,
            expected_claimed_by=adopted.claimed_by,
            expected_claimed_at=adopted.claimed_at,
        )
    assert current is not None and current.status == AgentRunStatus.RESUMING
    assert current.resume_attempts == ready.resume_attempts
    return current


def _source_inputs(storage, run):
    from reflexio.server.services.extraction.resume_worker import (
        _request_interaction_models_from_ids,
    )
    from reflexio.server.services.extraction.source_inputs import capture_source_inputs

    models = _request_interaction_models_from_ids(
        storage,
        run.binding.source_interaction_ids,
        fallback_source=run.binding.source,
        fallback_agent_version=run.binding.agent_version,
    )
    return capture_source_inputs(storage, run, models)


def _agent_run(run_id: str, extractor_kind: str = "profile") -> AgentRunRecord:
    return AgentRunRecord(
        id=run_id,
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind=extractor_kind,
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


def test_profile_instruction_prompt_delivers_structured_response():
    """The extraction agent now delivers its result as a structured response,
    so the active ``profile_update_instruction_start`` version (v1.2.0) frames
    the output as a direct JSON response rather than a ``finish_extraction``
    tool call. ``attach_pending_info_request`` remains a real intermediate tool;
    the profile extractor does not offer ``ask_human``."""
    prompt_manager = PromptManager()

    assert prompt_manager.get_active_version("profile_update_instruction_start") == (
        "1.2.0"
    )
    rendered = prompt_manager.render_prompt(
        "profile_update_instruction_start",
        {
            "agent_context_prompt": "agent context",
            "context_prompt": "",
            "extraction_definition_prompt": "user facts",
            "tagging_definition_prompt": None,
        },
    )

    assert "Resumable Extraction Mode" in rendered
    assert "ask_human" not in rendered
    assert "attach_pending_info_request" in rendered
    # The finish-tool framing is gone — the result is a direct structured response.
    assert "finish_extraction" not in rendered
    assert "final response" in rendered


def test_profile_pending_info_tools_do_not_include_ask_human():
    tools = create_pending_info_tools_for_extractor_kind("profile")

    assert [tool.name for tool in tools] == ["attach_pending_info_request"]


def test_playbook_pending_info_tools_still_include_ask_human():
    tools = create_pending_info_tools_for_extractor_kind("playbook")

    assert [tool.name for tool in tools] == [
        "ask_human",
        "attach_pending_info_request",
    ]


def test_resumable_agent_finishes_profile_output(
    monkeypatch,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _make_tc, make_stop = tool_call_completion
    # The model delivers the result as a structured (no-tool) response.
    response = make_stop(
        json.dumps(
            {
                "profiles": [
                    {
                        "content": "User prefers AWS ECS deployments.",
                        "time_to_live": "infinity",
                    }
                ]
            }
        )
    )
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage)

    with patch("litellm.completion", side_effect=[response]):
        result = agent.start(
            source_inputs=_source_inputs(agent.storage, _agent_run("run_profile")),
            run=_agent_run("run_profile"),
            messages=[{"role": "user", "content": "extract profiles"}],
            history_inputs=HistoricalInputs(),
            output_schema=StructuredProfilesOutput,
        )

    assert result.finished_reason == "structured_output"
    assert isinstance(result.output, StructuredProfilesOutput)
    assert result.output.profiles is not None
    assert result.output.profiles[0].content == "User prefers AWS ECS deployments."
    stored = storage.get_agent_run("run_profile")
    assert stored is not None
    assert stored.status == AgentRunStatus.AGENT_COMPLETED
    assert stored.max_steps_remaining == 7
    assert stored.committed_output["output"] == {
        "profiles": [
            {
                "content": "User prefers AWS ECS deployments.",
                "time_to_live": "infinity",
                "source_span": None,
                "notes": None,
                "reader_angle": None,
            }
        ]
    }
    assert stored.committed_output["model_provenance"] == {
        "model_name": None,
        "provider": None,
    }


def test_resumable_agent_discards_late_output_after_timeout_failure(
    monkeypatch,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _make_tc, make_stop = tool_call_completion
    response = make_stop(
        json.dumps(
            {
                "profiles": [
                    {
                        "content": "User prefers AWS ECS deployments.",
                        "time_to_live": "infinity",
                    }
                ]
            }
        )
    )
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage)

    def fail_then_return(*args, **kwargs):
        storage.update_agent_run_status(
            "run_late",
            AgentRunStatus.FAILED,
            last_error="Extractor timed out after 300 seconds",
        )
        return response

    with patch("litellm.completion", side_effect=fail_then_return):
        result = agent.start(
            source_inputs=_source_inputs(agent.storage, _agent_run("run_late")),
            run=_agent_run("run_late"),
            messages=[{"role": "user", "content": "extract profiles"}],
            history_inputs=HistoricalInputs(),
            output_schema=StructuredProfilesOutput,
        )

    assert result.finished_reason == "late_output_discarded"
    assert result.output is None
    stored = storage.get_agent_run("run_late")
    assert stored is not None
    assert stored.status == AgentRunStatus.FAILED
    assert stored.committed_output is None
    assert stored.last_error == "Extractor timed out after 300 seconds"


def test_resumable_agent_finishes_playbook_output(
    monkeypatch,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _make_tc, make_stop = tool_call_completion
    response = make_stop(
        json.dumps(
            {
                "playbooks": [
                    {
                        "trigger": "Deploying services",
                        "content": "Prefer ECS for this environment.",
                        "rationale": "The team standardizes on AWS.",
                    }
                ]
            }
        )
    )
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage)

    with patch("litellm.completion", side_effect=[response]):
        result = agent.start(
            source_inputs=_source_inputs(
                agent.storage, _agent_run("run_playbook", extractor_kind="playbook")
            ),
            run=_agent_run("run_playbook", extractor_kind="playbook"),
            messages=[{"role": "user", "content": "extract playbooks"}],
            history_inputs=HistoricalInputs(),
            output_schema=StructuredPlaybookList,
        )

    assert result.finished_reason == "structured_output"
    assert isinstance(result.output, StructuredPlaybookList)
    assert result.output.playbooks[0].content == "Prefer ECS for this environment."
    stored = storage.get_agent_run("run_playbook")
    assert stored is not None
    assert stored.status == AgentRunStatus.AGENT_COMPLETED
    assert stored.committed_output is not None
    assert stored.committed_output["output"]["playbooks"][0]["trigger"] == (
        "Deploying services"
    )


def test_resumable_agent_marks_run_failed_on_loop_error(monkeypatch, storage, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    monkeypatch.setattr(
        client,
        "generate_chat_response_with_provenance",
        MagicMock(side_effect=RuntimeError("provider failed")),
    )
    agent = ResumableExtractionAgent(client=client, storage=storage)

    result = agent.start(
        source_inputs=_source_inputs(agent.storage, _agent_run("run_error")),
        run=_agent_run("run_error"),
        messages=[{"role": "user", "content": "extract profiles"}],
        history_inputs=HistoricalInputs(),
        output_schema=StructuredProfilesOutput,
    )

    assert result.finished_reason == "error"
    stored = storage.get_agent_run("run_error")
    assert stored is not None
    assert stored.status == AgentRunStatus.FAILED
    assert stored.committed_output is None
    assert stored.last_error == (
        "Extraction agent did not finish: error (RuntimeError/non_transient)"
    )
    assert any(
        record.levelno == logging.WARNING
        and record.getMessage().startswith(
            "event=extraction_agent_failed release_blocking=true "
        )
        for record in caplog.records
    )


@pytest.mark.parametrize(
    ("error", "expected_label"),
    [
        (
            StructuredOutputRepairError(
                "Structured output repair exhausted",
                failure_kind="semantic",
                model="minimax/MiniMax-M3",
                raw_content="tenant content",
            ),
            "error (StructuredOutputRepairError/semantic)",
        ),
        (
            TimeoutError("tenant content"),
            "error (TimeoutError/transient)",
        ),
    ],
)
def test_resumable_agent_failure_names_cause_without_tenant_text(
    monkeypatch, storage, error, expected_label
):
    """The failure the extractor raises must say WHY, by class and kind only."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    monkeypatch.setattr(
        client,
        "generate_chat_response_with_provenance",
        MagicMock(side_effect=error),
    )
    agent = ResumableExtractionAgent(client=client, storage=storage)

    result = agent.start(
        source_inputs=_source_inputs(
            agent.storage, _agent_run("run_cause", extractor_kind="playbook")
        ),
        run=_agent_run("run_cause", extractor_kind="playbook"),
        messages=[{"role": "user", "content": "extract playbooks"}],
        history_inputs=HistoricalInputs(),
        output_schema=StructuredPlaybookList,
    )

    assert result.finished_reason == "error"
    assert result.failure_label == expected_label
    stored = storage.get_agent_run("run_cause")
    assert stored is not None
    assert stored.last_error == f"Extraction agent did not finish: {expected_label}"
    assert "tenant" not in stored.last_error


@pytest.mark.integration
@pytest.mark.parametrize("asks_question", [False, True])
def test_resumable_agent_uses_auto_tool_choice_with_extra_tools(
    monkeypatch,
    storage,
    tool_call_completion,
    asks_question,
):
    """With async-info tools registered, the loop uses tool_choice='auto'.

    The model is free to call ask_human / attach_pending_info_request OR to
    deliver the final structured response directly — a plain (no-tool) turn
    carrying the schema-conformant JSON is the success terminus.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    make_tc, make_stop = tool_call_completion
    response = make_stop(json.dumps({"profiles": None}))
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage)

    captured: dict[str, object] = {}
    original = client.generate_chat_response_with_provenance

    def _spy(*args, **kwargs):
        captured["tool_choice"] = kwargs.get("tool_choice")
        captured["response_format"] = kwargs.get("response_format")
        return original(*args, **kwargs)

    monkeypatch.setattr(client, "generate_chat_response_with_provenance", _spy)

    extra_ctx = PendingToolCallToolContext(
        storage=storage,
        run_id="run_auto",
        org_id="org_1",
        extractor_kind="profile",
        user_id="user_1",
        config=PendingToolCallConfig(enabled=True),
    )
    responses = (
        [make_tc("ask_human", {"question": "Which deployment standard?"}), response]
        if asks_question
        else [response]
    )
    with patch("litellm.completion", side_effect=responses):
        result = agent.start(
            source_inputs=_source_inputs(agent.storage, _agent_run("run_auto")),
            run=_agent_run("run_auto"),
            messages=[{"role": "user", "content": "extract profiles"}],
            history_inputs=HistoricalInputs(),
            output_schema=StructuredProfilesOutput,
            extra_tools=[
                create_ask_human_tool(),
                create_attach_pending_info_request_tool(),
            ],
            extra_tool_context=extra_ctx,
        )

    assert captured["tool_choice"] == "auto"
    assert captured["response_format"] is StructuredProfilesOutput
    assert result.finished_reason == "structured_output"
    stored = storage.get_agent_run("run_auto")
    assert stored is not None
    assert stored.pending_tool_call_ids == result.pending_tool_call_ids
    assert len(result.pending_tool_call_ids) == int(asks_question)
    if asks_question:
        [pending_id] = result.pending_tool_call_ids
        pending = storage.get_pending_tool_call(pending_id)
        assert pending is not None and pending.status == PendingToolCallStatus.PENDING
        assert pending.question_text == "Which deployment standard?"
        dependencies = storage.list_run_tool_dependencies("run_auto")
        assert [dependency.pending_tool_call_id for dependency in dependencies] == [
            pending_id
        ]


@pytest.mark.integration
@pytest.mark.parametrize("late_answer", [False, True])
def test_resumable_agent_resume_injects_resolved_tool_result(
    monkeypatch,
    storage,
    tool_call_completion,
    late_answer,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    run = storage.create_agent_run(_agent_run("run_resume"))
    question = "Which deployment standard should be treated as canonical?"
    scope = human_feedback_scope("org_1")
    now = datetime.now(UTC)
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
    ready = storage.get_agent_run(run.id)
    assert ready is not None and ready.status == AgentRunStatus.RESUME_READY

    _make_tc, make_stop = tool_call_completion
    response = make_stop(
        json.dumps(
            {
                "profiles": [
                    {
                        "content": "User deployment standard is AWS ECS.",
                        "time_to_live": "infinity",
                    }
                ]
            }
        )
    )
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage)

    from reflexio.server.services.extraction.overflow_answers import answer_witness

    expected_inputs = {resolved.id: build_scope_hash(answer_witness(asdict(resolved)))}

    def complete_after_input_capture(*_args, **_kwargs):
        if late_answer:
            later = storage.create_pending_tool_call(
                replace(pending, id="ptc_later", dedup_key="later-question")
            )
            storage.attach_run_tool_dependency(
                RunToolDependencyRecord(run_id=run.id, pending_tool_call_id=later.id)
            )
            storage.resolve_pending_tool_call(
                later.id,
                result={"answer": "An answer not supplied to this model call."},
                resolved_at=now,
                valid_for_seconds=3600,
            )
        return response

    with (
        user_extraction_lease(storage, "user_1", max_wait_seconds=0),
        patch("litellm.completion", side_effect=complete_after_input_capture),
    ):
        claimed = _adopt_continuation(storage, ready)
        result = agent.resume(
            source_inputs=_source_inputs(agent.storage, claimed),
            run=claimed,
            messages=[{"role": "user", "content": "resume extraction"}],
            history_inputs=HistoricalInputs(),
            output_schema=StructuredProfilesOutput,
            resolved_tool_calls=[resolved],
        )

    assert result.finished_reason == "structured_output"
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
    publication = stored.committed_output["publication"]
    UUID(publication["id"])
    assert publication["dependency_inputs"] == expected_inputs
    assert "An answer not supplied" not in json.dumps(result.messages)


def test_resumable_agent_resume_uses_persisted_step_budget(
    monkeypatch,
    storage,
    tool_call_completion,
    caplog,
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
    now = datetime.now(UTC)
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

    extra_ctx = PendingToolCallToolContext(
        storage=storage,
        run_id=run.id,
        org_id="org_1",
        extractor_kind="profile",
        user_id="user_1",
        config=PendingToolCallConfig(enabled=True),
    )
    # The model keeps calling an intermediate tool (ask_human) instead of
    # delivering the structured finish, consuming the single remaining step so
    # the loop terminates on max_steps.
    with patch(
        "litellm.completion",
        side_effect=[
            make_tc(
                "ask_human",
                {"question": "Need more?", "answer_format": "text", "tags": []},
            )
        ],
    ) as completion:
        result = agent.resume(
            source_inputs=_source_inputs(agent.storage, run),
            run=run,
            messages=[{"role": "user", "content": "resume extraction"}],
            history_inputs=HistoricalInputs(),
            output_schema=StructuredProfilesOutput,
            resolved_tool_calls=[resolved],
            extra_tools=[create_ask_human_tool()],
            extra_tool_context=extra_ctx,
        )

    assert completion.call_count == 1
    assert result.finished_reason == "max_steps"
    stored = storage.get_agent_run(run.id)
    assert stored is not None
    assert stored.status == AgentRunStatus.FAILED
    assert stored.max_steps_remaining == 0
    assert stored.last_error == "Extraction agent did not finish: max_steps"
    # Exhausting the tool budget returns normally and has no ERROR fallback;
    # this terminal warning must carry the release veto itself.
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    assert any(
        record.levelno == logging.WARNING
        and record.getMessage().startswith(
            "event=extraction_agent_failed release_blocking=true "
        )
        for record in caplog.records
    )


def _window_playbook_client(monkeypatch, side_effect):
    """A client whose structured calls follow ``side_effect`` (errors or outputs)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    call = MagicMock(side_effect=side_effect)
    monkeypatch.setattr(client, "generate_chat_response_with_provenance", call)
    return client, call


def _playbook_completion() -> SimpleNamespace:
    return SimpleNamespace(
        value=StructuredPlaybookList.model_validate(
            {
                "playbooks": [
                    {
                        "trigger": "Deploying services",
                        "content": "Prefer ECS for this environment.",
                        "rationale": "The team standardizes on AWS.",
                    }
                ]
            }
        ),
        provenance=None,
    )


def _start_window(agent: ResumableExtractionAgent, run_id: str):
    return agent.start(
        source_inputs=_source_inputs(
            agent.storage, _agent_run(run_id, extractor_kind="playbook")
        ),
        run=_agent_run(run_id, extractor_kind="playbook"),
        messages=[{"role": "user", "content": "extract playbooks"}],
        history_inputs=HistoricalInputs(),
        output_schema=StructuredPlaybookList,
    )


def test_window_retry_after_timeout_makes_a_model_call(monkeypatch, storage):
    """Regression: an errored attempt stores 0 remaining steps; the retry of the
    same window must still call the model instead of failing with max_steps."""
    client, call = _window_playbook_client(
        monkeypatch, [TimeoutError("transient"), _playbook_completion()]
    )
    agent = ResumableExtractionAgent(client=client, storage=storage)

    first = _start_window(agent, "window:w_timeout")
    assert first.finished_reason == "error"
    failed = storage.get_agent_run("window:w_timeout")
    assert failed is not None
    assert failed.status == AgentRunStatus.FAILED
    assert failed.max_steps_remaining == 0

    retry = _start_window(agent, "window:w_timeout")

    assert call.call_count == 2
    assert retry.finished_reason == "structured_output"
    assert isinstance(retry.output, StructuredPlaybookList)
    assert retry.output.playbooks[0].content == "Prefer ECS for this environment."
    stored = storage.get_agent_run("window:w_timeout")
    assert stored is not None
    assert stored.status == AgentRunStatus.AGENT_COMPLETED
    assert stored.max_steps_remaining == 7


@pytest.mark.parametrize("status", [AgentRunStatus.FAILED, AgentRunStatus.RUNNING])
def test_window_retry_of_spent_run_gets_full_step_budget(monkeypatch, storage, status):
    """A seeded FAILED row (or a RUNNING row whose process died after spending
    its budget) with 0 remaining restarts with the full budget, and the RUNNING
    transition persists that budget before any model call."""
    seeded = replace(
        _agent_run("window:w_spent", extractor_kind="playbook"),
        status=status,
        max_steps_remaining=0,
        last_error="Extraction agent did not finish: max_steps",
    )
    storage.create_agent_run(seeded)
    observed: list[AgentRunRecord | None] = []

    def complete(**_kwargs):
        observed.append(storage.get_agent_run("window:w_spent"))
        return _playbook_completion()

    client, call = _window_playbook_client(monkeypatch, complete)
    agent = ResumableExtractionAgent(client=client, storage=storage, max_steps=8)
    loop_budgets: list[int] = []
    real_loop = resumable_agent_module.run_tool_loop

    def spy_loop(**kwargs):
        loop_budgets.append(kwargs["max_steps"])
        return real_loop(**kwargs)

    monkeypatch.setattr(resumable_agent_module, "run_tool_loop", spy_loop)

    result = _start_window(agent, "window:w_spent")

    assert loop_budgets == [8]
    assert call.call_count == 1
    assert result.finished_reason == "structured_output"
    assert observed[0] is not None
    assert observed[0].status == AgentRunStatus.RUNNING
    assert observed[0].max_steps_remaining == 8


def test_window_retry_preserves_admitted_run_identity_and_snapshots(
    monkeypatch, storage
):
    """The fresh budget is the only field reset: the admitted binding, request,
    service and context snapshots, and resume count stay as first written."""
    admitted = replace(
        _agent_run("window:w_keep", extractor_kind="playbook"),
        status=AgentRunStatus.FAILED,
        generation_request_snapshot={"request_id": "admitted", "frozen": True},
        service_config_snapshot={"admitted": "service"},
        agent_context_snapshot="admitted context",
        max_steps_remaining=0,
        resume_attempts=2,
    )
    storage.create_agent_run(admitted)
    original = storage.get_agent_run("window:w_keep")
    assert original is not None
    client, _call = _window_playbook_client(monkeypatch, [_playbook_completion()])
    agent = ResumableExtractionAgent(client=client, storage=storage)

    result = agent.start(
        source_inputs=_source_inputs(
            agent.storage,
            replace(
                _agent_run("window:w_keep", extractor_kind="playbook"),
                generation_request_snapshot={"request_id": "retry"},
                service_config_snapshot={"retry": "service"},
                agent_context_snapshot="retry context",
            ),
        ),
        run=replace(
            _agent_run("window:w_keep", extractor_kind="playbook"),
            generation_request_snapshot={"request_id": "retry"},
            service_config_snapshot={"retry": "service"},
            agent_context_snapshot="retry context",
        ),
        messages=[{"role": "user", "content": "extract playbooks"}],
        history_inputs=HistoricalInputs(),
        output_schema=StructuredPlaybookList,
    )

    assert result.finished_reason == "structured_output"
    assert result.run_id == original.id
    stored = storage.get_agent_run("window:w_keep")
    assert stored is not None
    assert stored.binding == original.binding
    assert stored.generation_request_snapshot == original.generation_request_snapshot
    assert stored.service_config_snapshot == original.service_config_snapshot
    assert stored.agent_context_snapshot == original.agent_context_snapshot
    assert stored.resume_attempts == original.resume_attempts


def _ready_resume(storage, run_id: str) -> tuple[AgentRunRecord, PendingToolCallRecord]:
    """A historical ready run with one actual resolved dependency, unowned."""
    run = storage.create_agent_run(_agent_run(run_id))
    scope = human_feedback_scope("org_1")
    now = datetime.now(UTC)
    question = f"What does {run_id} need?"
    pending = storage.create_pending_tool_call(
        PendingToolCallRecord(
            id=f"ptc_{run_id}",
            org_id="org_1",
            user_id="user_1",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human", question_text=question
            ),
            status=PendingToolCallStatus.PENDING,
            question_text=question,
            expires_at=now + timedelta(hours=1),
            cache_until=now + timedelta(minutes=5),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id=run.id, pending_tool_call_id=pending.id)
    )
    storage.update_agent_run_status(run.id, AgentRunStatus.FINALIZED_PENDING_TOOL)
    resolved = storage.resolve_pending_tool_call(
        pending.id,
        result={"answer": "ECS"},
        resolved_at=now,
        valid_for_seconds=3600,
    )
    assert resolved is not None
    ready = storage.get_agent_run(run.id)
    assert ready is not None and ready.status == AgentRunStatus.RESUME_READY
    return ready, resolved


@pytest.mark.parametrize("outcome", ["agent_completed", "failed"])
@pytest.mark.parametrize("replace_run_revision", [False, True])
def test_a_stale_resume_worker_cannot_finish_a_reclaimed_run(
    monkeypatch,
    storage,
    tool_call_completion,
    caplog,
    outcome: str,
    replace_run_revision,
):
    """Real parent replacement fences both late output and late failure writes."""
    from contextvars import Context

    from reflexio.server.services.durable_learning import worker as admission
    from reflexio.server.services.durable_learning.user_lease import (
        _require_user_work_owner,
        owned_user_work_context,
    )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    ready, resolved = _ready_resume(storage, f"run_stale_{outcome}")
    _make_tc, make_stop = tool_call_completion
    response = make_stop(json.dumps({"profiles": []}))

    replacement_tokens: list[str] = []
    replacement_runs: list[AgentRunRecord] = []

    def reclaim_then_answer(*args, **kwargs):
        old_owner = _require_user_work_owner(storage)
        storage.release_user_extraction("user_1", old_owner.token)
        token = storage.claim_user_extraction("user_1", "worker_2", 300)
        assert token is not None and token != old_owner.token
        replacement_tokens.append(token)

        def replace_revision():
            with owned_user_work_context(storage, "user_1", token, 300):
                current = storage.get_agent_run(ready.id)
                assert current is not None
                if not replace_run_revision:
                    return current
                reclaimed = _adopt_continuation(storage, current)
                assert reclaimed.claimed_by != current.claimed_by
                return reclaimed

        newer = Context().run(replace_revision)
        replacement_runs.append(newer)
        if outcome == "failed":
            raise RuntimeError("provider failed")
        return response

    client = LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))
    agent = ResumableExtractionAgent(client=client, storage=storage)
    assert admission._reserve()
    old_token = None
    try:
        old_token = storage.claim_user_extraction("user_1", "worker_1", 300)
        assert old_token is not None
        with (
            owned_user_work_context(storage, "user_1", old_token, 300),
            patch("litellm.completion", side_effect=reclaim_then_answer),
        ):
            claimed = _adopt_continuation(storage, ready)
            result = agent.resume(
                source_inputs=_source_inputs(agent.storage, claimed),
                run=claimed,
                messages=[{"role": "user", "content": "resume extraction"}],
                history_inputs=HistoricalInputs(),
                output_schema=StructuredProfilesOutput,
                resolved_tool_calls=[resolved],
            )
    finally:
        try:
            if old_token is not None:
                storage.release_user_extraction("user_1", old_token)
        finally:
            admission._release()
            for replacement_token in replacement_tokens:
                try:
                    assert (
                        storage.claim_user_extraction("user_1", "third-owner", 300)
                        is None
                    )
                finally:
                    storage.release_user_extraction("user_1", replacement_token)

    stored = storage.get_agent_run(claimed.id)
    assert stored is not None
    assert len(replacement_tokens) == len(replacement_runs) == 1
    assert stored == replacement_runs[0]
    assert (stored.claimed_by != claimed.claimed_by) is replace_run_revision
    assert stored.status == AgentRunStatus.RESUMING
    assert stored.committed_output is None
    assert stored.last_error is None
    if outcome == "agent_completed":
        assert result.finished_reason == "late_output_discarded"
    else:
        assert any(
            record.getMessage().startswith(
                "event=extraction_agent_failed release_blocking=false "
            )
            for record in caplog.records
        )
    assert not any(
        record.getMessage().startswith(
            "event=extraction_agent_failed release_blocking=true "
        )
        for record in caplog.records
    )


@pytest.mark.parametrize("allowed_orgs", [[], ["org_1"]])
def test_non_neutral_rollout_requires_explicit_org_policy_adoption(
    monkeypatch, tmp_path, allowed_orgs
):
    from reflexio.server.site_var.site_var_manager import SiteVarManager

    (tmp_path / "feature_flags.json").write_text(
        json.dumps(
            {
                "resumable_extraction_agent": {
                    "enabled": False,
                    "enabled_org_ids": allowed_orgs,
                }
            }
        )
    )
    original_init = SiteVarManager.__init__
    monkeypatch.setattr(
        SiteVarManager,
        "__init__",
        lambda self: original_init(self, source_dir=str(tmp_path)),
    )
    context = SimpleNamespace(
        org_id="org_1",
        storage=object(),
        configurator=SimpleNamespace(
            get_config=lambda: SimpleNamespace(
                pending_tool_call_config=PendingToolCallConfig(enabled=True)
            )
        ),
    )
    with pytest.raises(ValueError, match="pending_tool_call_config.enabled"):
        resumable_agent_module.pending_tool_calls_enabled(cast(RequestContext, context))


@pytest.mark.integration
def test_window_output_replay_preserves_publication_identity(monkeypatch, storage):
    client, call = _window_playbook_client(monkeypatch, [_playbook_completion()])
    agent = ResumableExtractionAgent(client=client, storage=storage)
    first = _start_window(agent, "window:w_publication")
    saved = storage.get_agent_run(first.run_id)
    assert saved is not None and saved.committed_output is not None
    publication = saved.committed_output["publication"]
    UUID(publication["id"])
    assert publication["dependency_inputs"] == {}

    replay = _start_window(agent, "window:w_publication")

    assert replay.output == first.output
    assert replay.model_provenance == first.model_provenance
    assert (
        storage.get_agent_run(first.run_id).committed_output == saved.committed_output
    )
    assert replay.execution_run is not None
    assert replay.execution_run.committed_output is not None
    assert replay.execution_run.committed_output["publication"] == publication
    assert call.call_count == 1


@pytest.mark.integration
def test_successive_outputs_of_same_run_have_distinct_publication_identity(
    monkeypatch, storage, tool_call_completion
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    ready, resolved = _ready_resume(storage, "run_publication")
    _, make_stop = tool_call_completion
    agent = ResumableExtractionAgent(
        client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")), storage=storage
    )
    with (
        user_extraction_lease(storage, "user_1", max_wait_seconds=0),
        patch("litellm.completion", side_effect=[make_stop('{"profiles": []}')]),
    ):
        claimed = _adopt_continuation(storage, ready)
        first = agent.resume(
            source_inputs=_source_inputs(agent.storage, claimed),
            run=claimed,
            messages=[{"role": "user", "content": "resume extraction"}],
            history_inputs=HistoricalInputs(),
            output_schema=StructuredProfilesOutput,
            resolved_tool_calls=[resolved],
        )
    first_saved = storage.get_agent_run(first.run_id)
    assert first_saved is not None and first_saved.committed_output is not None
    first_publication = first_saved.committed_output["publication"]
    # Publish the waiting state as a fixture: this test covers the actual
    # agent output writer, not the still-unfinished shared publisher.
    storage.update_agent_run_status(first.run_id, AgentRunStatus.FINALIZED_PENDING_TOOL)
    later = storage.create_pending_tool_call(
        replace(
            resolved,
            id="ptc_next_output",
            dedup_key="next-output",
            result=None,
            resolved_at=None,
            status=PendingToolCallStatus.PENDING,
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id=first.run_id, pending_tool_call_id=later.id)
    )
    answered = storage.resolve_pending_tool_call(
        later.id,
        result={"answer": "Second answer"},
        resolved_at=datetime.now(UTC),
        valid_for_seconds=3600,
    )
    assert answered is not None
    next_ready = storage.get_agent_run(first.run_id)
    assert next_ready is not None and next_ready.status == AgentRunStatus.RESUME_READY
    with (
        user_extraction_lease(storage, "user_1", max_wait_seconds=0),
        patch("litellm.completion", side_effect=[make_stop('{"profiles": []}')]),
    ):
        next_claim = _adopt_continuation(storage, next_ready)
        second = agent.resume(
            source_inputs=_source_inputs(agent.storage, next_claim),
            run=next_claim,
            messages=[{"role": "user", "content": "resume extraction again"}],
            history_inputs=HistoricalInputs(),
            output_schema=StructuredProfilesOutput,
            resolved_tool_calls=[answered],
        )
    second_saved = storage.get_agent_run(second.run_id)
    assert second_saved is not None and second_saved.committed_output is not None
    second_publication = second_saved.committed_output["publication"]
    assert second.output == first.output
    assert second_publication["id"] != first_publication["id"]
    UUID(second_publication["id"])
    assert set(first_publication["dependency_inputs"]) == {resolved.id}
    assert set(second_publication["dependency_inputs"]) == {answered.id}
