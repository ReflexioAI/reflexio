from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.extraction.prior_answer_search import (
    append_prior_knowledge_context,
    build_knowledge_need_query,
    capture_prior_knowledge_context,
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
    now = datetime.now(UTC)
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
        extractor_config=_ExtractorConfig(),
        source="api",
        agent_version="v1",
    )

    assert len(messages) == 2
    context = messages[-1]["content"]
    assert "Prior Knowledge for org-scoped human feedback" in context
    assert "AWS ECS" in context
    assert "pending_tool_call_id: ptc_resolved" in context
    assert "source_request_id:" not in context
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
        extractor_config=_ExtractorConfig(),
        source="api",
        agent_version="v1",
    )

    assert result is messages


def _capture_record(storage):
    now = datetime.now(UTC)
    scope = human_feedback_scope("org_1")
    record = PendingToolCallRecord(
        id="full-answer",
        org_id="org_1",
        user_id="different-user",
        scope=scope,
        scope_hash=build_scope_hash(scope),
        tool_name="ask_human",
        dedup_key="complete-answer",
        status=PendingToolCallStatus.RESOLVED,
        question_text="When may this deployment proceed?",
        result={"answer": "background " * 200 + "ONLY with customer approval"},
        embedding=[1.0, 0.0],
        resolved_at=now,
        expires_at=now + timedelta(days=1),
        cache_until=now + timedelta(minutes=5),
        valid_until=now + timedelta(days=1),
    )
    storage.create_pending_tool_call(record)
    return record


def _capture(storage, **overrides):
    return capture_prior_knowledge_context(
        **{
            "messages": [{"role": "user", "content": "extract"}],
            "storage": storage,
            "org_id": "org_1",
            "extractor_kind": "profile",
            "extractor_config": _ExtractorConfig(),
            "source": "api",
            "agent_version": "v1",
            **overrides,
        }
    )


def test_capture_keeps_complete_answer_and_freezes_detached_values(storage):
    record = _capture_record(storage)
    with patch.object(
        storage, "checkpoint_extraction_overflow", side_effect=AssertionError("no CAS")
    ):
        captured = _capture(storage)
    assert "ONLY with customer approval" in captured.suffix[0]["content"]
    assert captured.answer_witnesses[record.id]["result"] == record.result
    digest = captured.contract_digest
    captured.suffix[0]["content"] = "tampered"
    captured.answer_witnesses[record.id]["result"].clear()
    assert "ONLY with customer approval" in captured.suffix[0]["content"]
    assert captured.answer_witnesses[record.id]["result"] == record.result
    assert captured.contract_digest == digest

    legacy = append_prior_knowledge_context(
        messages=[{"role": "user", "content": "extract"}],
        storage=storage,
        org_id="org_1",
        extractor_kind="profile",
        extractor_config=_ExtractorConfig(),
        source="api",
        agent_version="v1",
    )
    assert "ONLY with customer approval" not in legacy[-1]["content"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "identity",
        "question",
        "result",
        "resolved_at",
        "valid_until",
        "expired",
        "org",
        "scope",
        "status",
        "superseded",
    ],
)
def test_capture_rejects_answer_race_or_lost_authority(storage, change):
    record = _capture_record(storage)
    match = storage.search_prior_tool_calls(
        org_id="org_1",
        scope_hash=record.scope_hash,
        tool_name="ask_human",
        query_embedding=[1.0, 0.0],
        now=datetime.now(UTC),
        limit=8,
    )[0]
    changes = {
        "identity": {"id": "different-answer"},
        "question": {"question_text": "different"},
        "result": {"result": {"answer": "different"}},
        "resolved_at": {"resolved_at": datetime.now(UTC)},
        "valid_until": {"valid_until": None},
        "expired": {"valid_until": datetime.now(UTC) - timedelta(seconds=1)},
        "org": {"org_id": "other-org"},
        "scope": {"scope": {"org_id": "org_1", "scope_kind": "user"}},
        "status": {"status": PendingToolCallStatus.PENDING},
        "superseded": {"superseded_by": "replacement"},
    }
    current = None if change == "missing" else replace(record, **changes[change])
    if change == "expired":
        assert current is not None
        match = replace(match, valid_until=current.valid_until)
    with (
        patch.object(storage, "search_prior_tool_calls", return_value=[match]),
        patch.object(storage, "get_pending_tool_call", return_value=current),
        pytest.raises(InputBudgetConfigurationError),
    ):
        _capture(storage)


def test_capture_binds_existing_bounded_query_policy_and_reuses_lookup(storage):
    _capture_record(storage)
    messages = [{"role": "user", "content": "original source " * 500}]
    expected = build_knowledge_need_query(
        extractor_kind="profile",
        extractor_config=_ExtractorConfig(),
        source="api",
        agent_version="v1",
        messages=messages,
    )
    with patch.object(storage, "_get_embedding", return_value=[1.0, 0.0]) as embed:
        captured = _capture(storage, messages=messages)
    embed.assert_called_once_with(expected, purpose="query")
    assert len(expected) == 4000
    same = _capture(storage, messages=messages)
    assert same.contract_digest == captured.contract_digest
    assert _capture(storage, messages=[]).contract_digest != captured.contract_digest
    assert _capture(storage, limit=7).contract_digest != same.contract_digest
    excluded = _capture(storage, exclude_pending_tool_call_ids={"full-answer"})
    assert excluded.suffix == []
    assert excluded.answer_witnesses == {}


def test_append_prior_knowledge_context_filters_below_similarity_threshold(storage):
    now = datetime.now(UTC)
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
        extractor_config=_ExtractorConfig(),
        source="api",
        agent_version="v1",
        similarity_threshold=0.9,
    )

    assert result is messages
