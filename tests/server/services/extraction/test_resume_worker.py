from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.extraction.resumable_agent import (
    FINISH_EXTRACTION_TOOL_NAME,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
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


@pytest.fixture
def request_context(storage):
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = "org_1"
    ctx.storage = storage
    ctx.storage_base_dir = None
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="Extract durable user deployment facts.",
        ),
        pending_tool_call_config=PendingToolCallConfig(enabled=True),
    )
    ctx.configurator.get_agent_context.return_value = "Test agent context"
    ctx.prompt_manager = MagicMock()
    ctx.prompt_manager.render_prompt.side_effect = lambda prompt_id, variables: (
        f"{prompt_id}: {variables}"
    )
    return ctx


def _seed_interactions(storage: SQLiteStorage) -> None:
    storage.add_request(
        Request(
            request_id="request_1",
            user_id="user_1",
            created_at=1_000,
            source="api",
            agent_version="v1",
            session_id="request_1",
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=1,
            user_id="user_1",
            request_id="request_1",
            created_at=1_000,
            role="user",
            content="Which deployment target should we use?",
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=2,
            user_id="user_1",
            request_id="request_1",
            created_at=1_001,
            role="assistant",
            content="I need the deployment standard.",
        )
    )


def _seed_ready_run(storage: SQLiteStorage) -> None:
    storage.create_agent_run(
        AgentRunRecord(
            id="run_1",
            binding=AgentBinding(
                org_id="org_1",
                extractor_kind="profile",
                user_id="user_1",
                request_id="request_1",
                agent_version="v1",
                source="api",
                source_interaction_ids=[1, 2],
                window_start_interaction_id=1,
                window_end_interaction_id=2,
                extractor_config_hash="old_hash",
            ),
            status=AgentRunStatus.FINALIZED_PENDING_TOOL,
            generation_request_snapshot={"request_id": "request_1"},
        )
    )
    now = datetime(2026, 5, 28, tzinfo=UTC)
    question = "What is the deployment target?"
    scope = human_feedback_scope("org_1")
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_1",
            org_id="org_1",
            user_id="user_1",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text=question,
            ),
            status=PendingToolCallStatus.PENDING,
            question_text=question,
            args={"question": question},
            expires_at=now + timedelta(hours=1),
            cache_until=now + timedelta(minutes=5),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )
    storage.resolve_pending_tool_call(
        "ptc_1",
        result={"answer": "Use AWS ECS."},
        resolved_at=now,
        valid_for_seconds=3600,
    )


def test_resume_worker_resumes_profile_run_and_consumes_dependency(
    monkeypatch,
    request_context,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _seed_interactions(storage)
    _seed_ready_run(storage)
    make_tc, _make_stop = tool_call_completion
    response = make_tc(
        FINISH_EXTRACTION_TOOL_NAME,
        {
            "profiles": [
                {
                    "content": "User deployment target is AWS ECS.",
                    "time_to_live": "infinity",
                }
            ]
        },
    )
    worker = ExtractionResumeWorker(request_context=request_context)

    with (
        patch("litellm.completion", side_effect=[response]),
        patch(
            "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
            return_value=False,
        ),
    ):
        resumed = worker.drain(max_runs=1)

    assert resumed == 1
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZED
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is not None
    profiles = storage.get_user_profile("user_1")
    assert [profile.content for profile in profiles] == [
        "User deployment target is AWS ECS."
    ]


def test_resume_worker_retries_finalization_without_rerunning_agent(
    monkeypatch,
    request_context,
    storage,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _seed_interactions(storage)
    _seed_ready_run(storage)
    make_tc, _make_stop = tool_call_completion
    response = make_tc(
        FINISH_EXTRACTION_TOOL_NAME,
        {
            "profiles": [
                {
                    "content": "User deployment target is AWS ECS.",
                    "time_to_live": "infinity",
                }
            ]
        },
    )
    worker = ExtractionResumeWorker(request_context=request_context)

    with (
        patch("litellm.completion", side_effect=[response]),
        patch(
            "reflexio.server.services.profile.profile_generation_service."
            "ProfileGenerationService._finalize_extracted_items",
            side_effect=RuntimeError("storage write failed"),
        ),
    ):
        resumed = worker.drain(max_runs=1)

    assert resumed == 1
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZATION_FAILED
    assert run.committed_output is not None
    assert run.finalization_attempts == 1
    assert run.next_resume_at is not None
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is None

    storage.update_agent_run_status(
        "run_1",
        AgentRunStatus.FINALIZATION_FAILED,
        next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
    )
    with (
        patch(
            "litellm.completion",
            side_effect=AssertionError("finalization retry must not call LLM"),
        ),
        patch(
            "reflexio.server.services.profile.profile_generation_service."
            "ProfileGenerationService._finalize_extracted_items",
            return_value=None,
        ) as finalize,
    ):
        resumed = worker.drain(max_runs=1)

    assert resumed == 1
    finalize.assert_called_once()
    retried = storage.get_agent_run("run_1")
    assert retried is not None
    assert retried.status == AgentRunStatus.FINALIZED
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is not None


def test_resume_worker_tagging_schedule_failure_is_best_effort(
    request_context,
):
    run = AgentRunRecord(
        id="run_tagging",
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="profile",
            user_id="user_1",
            request_id="request_1",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.FINALIZED_PENDING_TOOL,
        generation_request_snapshot={"request_id": "request_1"},
    )
    worker = ExtractionResumeWorker(request_context=request_context)

    with patch(
        "reflexio.server.services.extraction.resume_worker.schedule_tagging",
        side_effect=RuntimeError("scheduler unavailable"),
    ):
        worker._schedule_finalized_tagging(run)


def test_resume_worker_fails_run_when_step_budget_exhausted(
    monkeypatch,
    request_context,
    storage,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    _seed_interactions(storage)
    _seed_ready_run(storage)
    storage.update_agent_run_status(
        "run_1",
        AgentRunStatus.RESUME_READY,
        max_steps_remaining=0,
    )
    worker = ExtractionResumeWorker(request_context=request_context)

    with patch(
        "litellm.completion",
        side_effect=AssertionError("exhausted budget must not call LLM"),
    ):
        resumed = worker.drain(max_runs=1)

    assert resumed == 1
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FAILED
    assert run.last_error == "Resumable extraction max-step budget exhausted"
    assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is None
