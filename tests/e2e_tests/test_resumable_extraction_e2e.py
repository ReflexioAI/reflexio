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
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
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

pytestmark = pytest.mark.e2e


def _request_context(storage: SQLiteStorage) -> RequestContext:
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = "e2e_resumable_org"
    ctx.storage = storage
    ctx.storage_base_dir = None
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extractor_name="default_profile_extractor",
            extraction_definition_prompt="Extract durable deployment preferences.",
        ),
        pending_tool_call_config=PendingToolCallConfig(
            enabled=True,
            human_input_enabled=True,
            prior_knowledge_injection_enabled=True,
        ),
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
            content="Please remember our deployment target once confirmed.",
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=2,
            user_id="user_1",
            request_id="request_1",
            created_at=1_001,
            role="assistant",
            content="I will ask for the deployment target and continue.",
        )
    )


def _seed_followup_ready_run(storage: SQLiteStorage) -> None:
    storage.create_agent_run(
        AgentRunRecord(
            id="run_1",
            binding=AgentBinding(
                org_id="e2e_resumable_org",
                extractor_kind="profile",
                extractor_name="default_profile_extractor",
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
            max_steps_remaining=7,
        )
    )
    now = datetime(2026, 5, 28, tzinfo=UTC)
    question = "What deployment target should be treated as canonical?"
    scope = human_feedback_scope("e2e_resumable_org")
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_1",
            org_id="e2e_resumable_org",
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


def test_resumable_extraction_resumes_after_human_answer(
    monkeypatch,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        storage = SQLiteStorage(
            org_id="e2e_resumable_org",
            db_path=f"{temp_dir}/reflexio.db",
        )
        _seed_interactions(storage)
        _seed_followup_ready_run(storage)
        worker = ExtractionResumeWorker(
            request_context=_request_context(storage),
            llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
        )
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

        with (
            patch("litellm.completion", side_effect=[response]),
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=False,
            ),
        ):
            resumed = worker.drain(max_runs=1)

        run = storage.get_agent_run("run_1")
        assert resumed == 1
        assert run is not None
        assert run.status == AgentRunStatus.FINALIZED
        assert run.max_steps_remaining == 6
        assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is not None
        assert [profile.content for profile in storage.get_user_profile("user_1")] == [
            "User deployment target is AWS ECS."
        ]
