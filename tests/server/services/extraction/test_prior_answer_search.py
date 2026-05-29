from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from reflexio.server.services.extraction.prior_answer_search import (
    append_prior_knowledge_context,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
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
        patch.object(SQLiteStorage, "_get_embedding", return_value=[1.0, 0.0]),
    ):
        yield SQLiteStorage(org_id="org_1", db_path=f"{temp_dir}/reflexio.db")


class _ExtractorConfig:
    extraction_definition_prompt = "Extract deployment preferences."


def test_append_prior_knowledge_context_adds_resolved_and_pending_entries(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    scope = human_feedback_scope("org_1")
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_resolved",
            org_id="org_1",
            user_id="user_1",
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
            embedding=[1.0, 0.0],
            resolved_at=now,
            expires_at=now + timedelta(days=30),
            cache_until=now + timedelta(minutes=5),
            valid_until=now + timedelta(days=30),
        )
    )
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_pending",
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
            answer_format="short text",
            embedding=[1.0, 0.0],
            expires_at=now + timedelta(days=30),
            cache_until=now + timedelta(minutes=5),
        )
    )

    messages = append_prior_knowledge_context(
        messages=[{"role": "user", "content": "extract"}],
        storage=storage,
        org_id="org_1",
        extractor_kind="profile",
        extractor_name="default_profile_extractor",
        extractor_config=_ExtractorConfig(),
        source="api",
        agent_version="v1",
    )

    assert len(messages) == 2
    context = messages[-1]["content"]
    assert "Prior Knowledge for org-scoped human feedback" in context
    assert "AWS ECS" in context
    assert "attach_pending_info_request" in context
    assert "ptc_pending" in context


def test_append_prior_knowledge_context_returns_original_messages_without_matches(
    storage,
):
    messages = [{"role": "user", "content": "extract"}]

    result = append_prior_knowledge_context(
        messages=messages,
        storage=storage,
        org_id="org_1",
        extractor_kind="profile",
        extractor_name="default_profile_extractor",
        extractor_config=_ExtractorConfig(),
        source="api",
        agent_version="v1",
    )

    assert result is messages


def test_append_prior_knowledge_context_filters_below_similarity_threshold(storage):
    now = datetime(2026, 5, 28, tzinfo=UTC)
    scope = human_feedback_scope("org_1")
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_low_similarity",
            org_id="org_1",
            user_id="user_1",
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
            embedding=[0.0, 1.0],
            resolved_at=now,
            expires_at=now + timedelta(days=30),
            cache_until=now + timedelta(minutes=5),
            valid_until=now + timedelta(days=30),
        )
    )

    messages = [{"role": "user", "content": "extract"}]
    result = append_prior_knowledge_context(
        messages=messages,
        storage=storage,
        org_id="org_1",
        extractor_kind="profile",
        extractor_name="default_profile_extractor",
        extractor_config=_ExtractorConfig(),
        source="api",
        agent_version="v1",
        similarity_threshold=0.9,
    )

    assert result is messages
