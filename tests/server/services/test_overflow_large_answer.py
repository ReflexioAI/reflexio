"""Oversized human answers stay intact and fail closed before provider dispatch."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.models.config_schema import (
    PendingToolCallConfig,
    UserPlaybookExtractorConfig,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.storage.storage_base import (
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_scope_hash,
    human_feedback_scope,
)
from reflexio.test_support.skip_decorators import skip_in_precommit
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import (
    pipeline as imported_pipeline,
)

pytestmark = pytest.mark.integration
answer_pipeline = imported_pipeline


@skip_in_precommit
@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_http_oversized_answer_preserves_qualifier_and_fails_without_dispatch(
    answer_pipeline, monkeypatch, kind
):
    import json

    engine, worker, client = answer_pipeline
    storage = engine.get_storage()
    configurator = engine.request_context.configurator
    storage_factory = type(configurator).create_storage

    def scoped_storage(owner, storage_config):
        if owner.org_id == storage.org_id:
            return storage
        return storage_factory(owner, storage_config)

    # Pending-tool HTTP routes construct their own real RequestContext. Route
    # this fixture's organization to its owned database, leaving other scopes
    # on the ordinary factory and preserving the route/authorization contract.
    monkeypatch.setattr(type(configurator), "create_storage", scoped_storage)
    updates: dict[str, Any] = {
        "pending_tool_call_config": PendingToolCallConfig(enabled=True)
    }
    if kind == "playbook":
        updates.update(
            profile_extractor_config=None,
            user_playbook_extractor_config=UserPlaybookExtractorConfig(
                extraction_definition_prompt="Extract scoped approval practices."
            ),
        )
    configurator.set_config(configurator.get_config().model_copy(update=updates))
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
        lambda _: True,
    )
    if kind == "profile":
        # Profile extraction attaches a real pre-existing question; only playbook
        # extraction offers the native ask_human creation tool.
        scope = human_feedback_scope(storage.org_id)
        now = datetime.now(UTC)
        storage.create_pending_tool_call(
            PendingToolCallRecord(
                id="large-answer-question",
                org_id=storage.org_id,
                scope=scope,
                scope_hash=build_scope_hash(scope),
                tool_name="ask_human",
                dedup_key="large-answer-question",
                status=PendingToolCallStatus.PENDING,
                question_text="What actions are approved?",
                expires_at=now + timedelta(days=1),
                cache_until=now + timedelta(hours=1),
            )
        )

    def respond(params, _hard_timeout):
        assert dispatch.call_count == 1, "Oversized answer reached the provider wire"
        tools = {tool["function"]["name"] for tool in params["tools"]}
        name = "ask_human" if kind == "playbook" else "attach_pending_info_request"
        assert name in tools
        arguments = (
            {"question": "What actions are approved?"}
            if kind == "playbook"
            else {"pending_tool_call_id": "large-answer-question"}
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="ask-scope",
                                type="function",
                                function=SimpleNamespace(
                                    name=name, arguments=json.dumps(arguments)
                                ),
                            )
                        ],
                        refusal=None,
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="minimax/MiniMax-M3",
            _hidden_params={"custom_llm_provider": "minimax"},
        )

    dispatch = Mock(side_effect=respond)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    token_bill, learning_bill = Mock(), Mock()
    monkeypatch.setattr(
        "reflexio.server.metering_events.record_extraction_tokens", token_bill
    )
    monkeypatch.setattr(
        "reflexio.server.metering_events.record_learnings_generated_records_strict",
        learning_bill,
    )
    request_id = f"large-answer-{kind}"
    request = payload(request_id, force=True)
    request["agent_version"] = "v1"
    for interaction in request["interaction_data_list"]:
        interaction["content"] = "Prepare only; deployment requires separate approval."
    assert client.post("/api/publish_interaction", json=request).json()["success"]
    worker.drain_org("stream-test", 1, 300)
    assert dispatch.call_count == 1
    with storage._stream_sql() as db:
        pending = db.query("SELECT id,status FROM _pending_tool_calls LIMIT 2")
        assert len(pending) == 1 and pending[0]["status"] == "pending"
        parent_id = db.query("SELECT id FROM _agent_runs LIMIT 2")[0]["id"]
    before = storage.get_agent_run(parent_id)
    assert before is not None
    attempts = deepcopy(before.generation_request_snapshot["overflow"]["attempts"])
    assert len(attempts) == 1
    tail = "Preparation only. Deployment is NOT approved; obtain separate approval."
    answer = {"answer": "Background context. " * 20000 + tail}
    response = client.post(
        f"/api/pending_tool_calls/{pending[0]['id']}/resolve",
        json={"result": answer, "valid_for_seconds": 3600},
    )
    assert response.status_code == 200
    assert response.json()["result"] == answer
    assert (
        client.get(f"/api/pending_tool_calls/{pending[0]['id']}").json()["result"]
        == answer
    )
    with storage._stream_sql() as db:
        db.query("UPDATE learning_work SET due_at=0")
        db.query("UPDATE extraction_cursors SET retry_at=0")
    worker.drain_org("stream-test", 1, 300)
    assert client.get(
        "/api/learning_status", params={"request_id": request_id}
    ).json() == {"status": "failed", "reason": "input_budget_exceeded"}
    failed = storage.get_agent_run(parent_id)
    assert failed is not None
    manifest = failed.generation_request_snapshot["overflow"]
    assert manifest["attempts"] == attempts
    witness = manifest["checkpoint"]["answer_witnesses"][pending[0]["id"]]
    assert witness["result"] == answer and witness["result"]["answer"].endswith(tail)
    assert storage.count_all_profiles() == 0
    with storage._stream_sql() as db:
        assert db.query("SELECT COUNT(*) AS n FROM user_playbooks")[0]["n"] == 0
        window = db.query(
            "SELECT completed,outcome,effects FROM extraction_windows WHERE window_id=?",
            (parent_id.removeprefix("window:"),),
        )[0]
        assert (
            not window["completed"]
            and window["outcome"] is None
            and window["effects"] is None
        )
    worker.drain_org("stream-test", 5, 300)
    assert dispatch.call_count == 1
    current = storage.get_agent_run(parent_id)
    assert current is not None
    assert current.generation_request_snapshot["overflow"]["attempts"] == attempts
    token_bill.assert_not_called()
    learning_bill.assert_not_called()
