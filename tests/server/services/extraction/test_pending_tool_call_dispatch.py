from __future__ import annotations

import logging
import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from reflexio.models.config_schema import PendingToolCallConfig
from reflexio.server.llm.tools import AsyncAccepted
from reflexio.server.services.extraction.pending_tool_call_dispatch import (
    AskHumanArgs,
    AttachPendingInfoRequestArgs,
    NoopPendingToolCallDispatcher,
    PendingToolCallToolContext,
    create_ask_human_tool,
    create_attach_pending_info_request_tool,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
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


def _agent_run(
    run_id: str,
    *,
    user_id: str | None = "user_1",
) -> AgentRunRecord:
    return AgentRunRecord(
        id=run_id,
        binding=AgentBinding(
            org_id="org_1",
            extractor_kind="profile",
            user_id=user_id,
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


def _tool_context(
    storage,
    *,
    run_id: str,
    user_id: str | None = "user_1",
    config: PendingToolCallConfig | None = None,
) -> PendingToolCallToolContext:
    return PendingToolCallToolContext(
        storage=storage,
        run_id=run_id,
        org_id="org_1",
        extractor_kind="profile",
        user_id=user_id,
        config=config or PendingToolCallConfig(),
        dispatcher=NoopPendingToolCallDispatcher(),
    )


def test_ask_human_creates_org_scoped_pending_call_and_dependency(storage):
    storage.create_agent_run(_agent_run("run_1"))
    tool = create_ask_human_tool()

    outcome = tool.handler(
        AskHumanArgs(
            question="What deployment target should this agent optimize for?",
            answer_format="short text",
            tags=["deployment"],
        ),
        _tool_context(storage, run_id="run_1"),
    )

    assert isinstance(outcome, AsyncAccepted)
    pending = storage.get_pending_tool_call(outcome.pending_tool_call_id)
    assert pending is not None
    assert pending.scope == {"org_id": "org_1", "scope_kind": "org"}
    assert pending.user_id == "user_1"
    assert (
        pending.question_text
        == "What deployment target should this agent optimize for?"
    )
    assert pending.answer_format == "short text"
    assert pending.tags == ["deployment"]
    deps = storage.list_run_tool_dependencies("run_1")
    assert [dep.pending_tool_call_id for dep in deps] == [pending.id]
    assert outcome.result["status"] == "request_pending"


def test_ask_human_accepts_comma_separated_tags(storage):
    storage.create_agent_run(_agent_run("run_1"))
    tool = create_ask_human_tool()

    outcome = tool.handler(
        AskHumanArgs(
            question="What deployment target should this agent optimize for?",
            answer_format="short text",
            tags="deployment, org-standard",  # type: ignore[arg-type]
        ),
        _tool_context(storage, run_id="run_1"),
    )

    assert isinstance(outcome, AsyncAccepted)
    pending = storage.get_pending_tool_call(outcome.pending_tool_call_id)
    assert pending is not None
    assert pending.tags == ["deployment", "org-standard"]


def test_same_human_question_attaches_across_users_with_org_scope(storage):
    storage.create_agent_run(_agent_run("run_1", user_id="user_1"))
    storage.create_agent_run(_agent_run("run_2", user_id="user_2"))
    tool = create_ask_human_tool()
    question = "What deployment target should this agent optimize for?"

    first = tool.handler(
        AskHumanArgs(question=question),
        _tool_context(storage, run_id="run_1", user_id="user_1"),
    )
    second = tool.handler(
        AskHumanArgs(question=question),
        _tool_context(storage, run_id="run_2", user_id="user_2"),
    )

    assert isinstance(first, AsyncAccepted)
    assert isinstance(second, AsyncAccepted)
    assert first.pending_tool_call_id == second.pending_tool_call_id
    pending_calls = storage.list_pending_tool_calls()
    assert len(pending_calls) == 1
    assert pending_calls[0].scope == {"org_id": "org_1", "scope_kind": "org"}
    assert storage.list_run_tool_dependencies("run_1")[0].pending_tool_call_id == (
        first.pending_tool_call_id
    )
    assert storage.list_run_tool_dependencies("run_2")[0].pending_tool_call_id == (
        first.pending_tool_call_id
    )


def test_soft_cap_logs_but_still_accepts_human_request(storage, caplog):
    storage.create_agent_run(_agent_run("run_1"))
    storage.create_agent_run(_agent_run("run_2"))
    tool = create_ask_human_tool()
    config = PendingToolCallConfig(max_pending_followups_per_scope=1)

    tool.handler(
        AskHumanArgs(question="What deployment target should this agent optimize for?"),
        _tool_context(storage, run_id="run_1", config=config),
    )

    with caplog.at_level(logging.WARNING):
        outcome = tool.handler(
            AskHumanArgs(question="Which compliance framework applies?"),
            _tool_context(storage, run_id="run_2", config=config),
        )

    assert isinstance(outcome, AsyncAccepted)
    assert "pending_followup_soft_cap_exceeded" in caplog.text
    assert len(storage.list_pending_tool_calls()) == 2


def test_attach_pending_info_request_links_existing_org_scoped_request(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1", user_id="user_1"))
    scope = human_feedback_scope("org_1")
    pending = storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_existing",
            org_id="org_1",
            user_id="other_user",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text="Which compliance framework applies?",
            ),
            status=PendingToolCallStatus.PENDING,
            question_text="Which compliance framework applies?",
            expires_at=now + timedelta(days=30),
            cache_until=now + timedelta(minutes=5),
        )
    )
    tool = create_attach_pending_info_request_tool()

    result = tool.handler(
        AttachPendingInfoRequestArgs(
            pending_tool_call_id=pending.id,
            why_relevant="The current window mentions compliance.",
        ),
        _tool_context(storage, run_id="run_1", user_id="user_1"),
    )

    assert isinstance(result, dict)
    assert result["status"] == "attached_for_followup"
    deps = storage.list_run_tool_dependencies("run_1")
    assert [dep.pending_tool_call_id for dep in deps] == [pending.id]


def test_attach_pending_info_request_returns_resolved_result_without_dependency(
    storage,
):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    storage.create_agent_run(_agent_run("run_1"))
    scope = human_feedback_scope("org_1")
    resolved = storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_resolved",
            org_id="org_1",
            user_id=None,
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text="Which deployment target applies?",
            ),
            status=PendingToolCallStatus.RESOLVED,
            question_text="Which deployment target applies?",
            result={"answer": "AWS ECS"},
            resolved_at=now,
            expires_at=now + timedelta(days=30),
            cache_until=now + timedelta(minutes=5),
            valid_until=now + timedelta(days=30),
        )
    )
    tool = create_attach_pending_info_request_tool()

    result = tool.handler(
        AttachPendingInfoRequestArgs(pending_tool_call_id=resolved.id),
        _tool_context(storage, run_id="run_1"),
    )

    assert isinstance(result, dict)
    assert result["status"] == "already_resolved"
    assert result["result"] == {"answer": "AWS ECS"}
    assert storage.list_run_tool_dependencies("run_1") == []
