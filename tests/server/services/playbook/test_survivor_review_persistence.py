"""Inactive Native review reaches real isolated SQLite successor persistence."""

import json

import litellm
import pytest

from reflexio.models.api_schema.domain import Status
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import provider_env_vars
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
)
from tests.server.services.playbook.test_playbook_review_service_integration import (
    _request,
    _service,
    _storage,
)
from tests.server.services.playbook.test_survivor_reviewer import (
    _ACTION,
    _PARAMETER,
    _SUPPLY,
    _assignment_response,
    _source_and_candidate,
)

pytestmark = pytest.mark.integration


def test_repaired_task_assignment_persists_successor_with_truthful_rationale_and_lineage(
    tmp_path, monkeypatch
):
    for name in provider_env_vars():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "unit-test-placeholder")
    storage = _storage(tmp_path)
    sources, original = _source_and_candidate()
    assert original.user_id is not None and original.notes is not None
    original.created_at = 200
    original.source = "task_input"
    original.reader_angle = "assigned-input-use"
    sources[0].interactions[0].embedding = [0.0] * 512
    storage.add_request(sources[0].request)
    storage.add_user_interactions_bulk(
        original.user_id,
        sources[0].interactions,
        embeddings_prepared=True,
    )
    storage.create_agent_run(
        AgentRunRecord(
            id="synthetic-finalized-generation",
            binding=AgentBinding(
                org_id=storage.org_id,
                extractor_kind="playbook",
                user_id=original.user_id,
                request_id=original.request_id,
                agent_version=original.agent_version,
                source="task_input",
                source_interaction_ids=[1],
                window_start_interaction_id=1,
                window_end_interaction_id=1,
            ),
            status=AgentRunStatus.FINALIZED,
            generation_request_snapshot={
                "request_id": original.request_id,
                "source_interaction_ids": [1],
            },
        )
    )
    storage.save_user_playbooks([original])
    responses = [_assignment_response("assigned_input"), _assignment_response()]
    dispatches = []

    def completion(**params):
        dispatches.append(params)
        return litellm.ModelResponse(
            model="MiniMax-M3",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(responses.pop(0)),
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180},
        )

    monkeypatch.setattr(litellm, "completion", completion)
    service = _service(storage)
    # This is an explicit test injection, not a public per-org version setting.
    service.request_context.prompt_manager = PromptManager(
        version_override={"playbook_candidate_review": "1.51.0"}
    )
    service.client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_retries=0, fallback_models=[])
    )
    response = service.run(_request(report_only=False), run_id="native-review-persist")
    assert response.success, response.msg
    assert response.selected_count == response.edited_count == 1
    assert response.rejected_count == response.skipped_count == 0
    assert len(dispatches) == 2 and responses == []
    feedback = dispatches[1]["messages"][-1]["content"]
    assert "assigned_input asserts User-supplied input" in feedback
    assert "task_assignment" in feedback
    result = response.results[0]
    assert result.applied and result.reason_code == "speculative"
    successor_id = result.successor_user_playbook_id
    assert successor_id is not None and successor_id != original.user_playbook_id
    current = storage.get_user_playbooks(status_filter=[None])
    assert [row.user_playbook_id for row in current] == [successor_id]
    successor = current[0]
    assert successor.content == _ACTION
    assert (
        successor.trigger == "When preparing an export job with an assigned delimiter"
    )
    assert successor.rationale == (
        f"The recorded task has an assigned value for this input: {_PARAMETER}."
    )
    assert successor.notes and successor.notes.startswith(original.notes + "\n")
    assert "revised (speculative)" in successor.notes
    assert successor.source_interaction_ids == [1] and successor.source_span == _SUPPLY
    assert (successor.user_id, successor.agent_version, successor.request_id) == (
        original.user_id,
        original.agent_version,
        original.request_id,
    )
    assert successor.source == original.source
    assert successor.rationale is not None
    assert "semicolon" not in successor.content + successor.rationale
    superseded = storage.get_user_playbooks(status_filter=[Status.SUPERSEDED])
    assert len(superseded) == 1
    assert superseded[0].user_playbook_id == original.user_playbook_id
    assert superseded[0].superseded_by == successor_id
    assert superseded[0].content == original.content
    assert (
        storage.get_user_playbooks(status_filter=[Status.PENDING, Status.ARCHIVED])
        == []
    )
    events = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(successor_id)
    )
    revise = [event for event in events if event.op == "revise"]
    assert len(revise) == 1
    assert revise[0].actor == "playbook_review"
    assert revise[0].org_id == storage.org_id
    assert revise[0].request_id == response.run_id == "native-review-persist"
