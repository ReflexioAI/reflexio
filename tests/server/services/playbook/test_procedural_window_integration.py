"""Real HTTP/window/SQLite publication with deterministic provider wire replies."""

import json
from unittest.mock import Mock

import litellm
import pytest

from reflexio.models.config_schema import UserPlaybookExtractorConfig
from reflexio.server.llm.input_budget import ExtractionInputIntegrityError
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.server.services.playbook.procedural import native
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


def setup(pipeline, monkeypatch, *, retire=False, change_before_commit=None):
    engine, _, http = pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-no-provider")
    monkeypatch.setenv("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", '{"gpt-4o-mini":128000}')
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_text_generation.resolve_model_name",
        lambda **_: "gpt-4o-mini",
    )
    configured = engine.request_context.configurator.get_config().model_copy(deep=True)
    configured.profile_extractor_config = None
    configured.user_playbook_extractor_config = UserPlaybookExtractorConfig(
        extraction_definition_prompt="Extract reusable deployment procedures."
    )
    engine.request_context.configurator.set_config(configured)
    storage = engine.get_storage()
    monkeypatch.setattr(storage, "_get_embedding", Mock(return_value=[0.0] * 512))
    existing = []
    if retire:
        from reflexio.models.api_schema.domain.entities import UserPlaybook

        row = UserPlaybook(
            user_id="u",
            agent_version="v",
            request_id="old",
            content="Use the obsolete deploy command.",
            trigger="When deploying.",
        )
        storage.save_user_playbooks([row], skip_embedding=True)
        existing = [row]
    monkeypatch.setattr(
        PlaybookConsolidator, "retrieve_existing_playbooks", lambda *_a, **_k: existing
    )
    request = durable.payload("procedural", count=5, force=True)
    request["agent_version"] = "v"
    for interaction in request["interaction_data_list"]:
        interaction["content"] = (
            "Require explicit approval before deploying. The obsolete deploy command is no longer valid."
        )
    assert http.post("/api/publish_interaction", json=request).json()["success"]
    user, token = storage.claim_extraction("procedural-test", 600)
    window = storage.prepare_extraction(user, token)
    assert window.kind == "playbook"
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", max_tokens=16384, fallback_models=[])
    )
    calls = []

    def wire(params, _timeout):
        calls.append(params)
        names = {tool["function"]["name"] for tool in params["tools"]}
        body = json.loads(params["messages"][1]["content"])
        if "record_observation" in names:
            evidence = body["sources"][0]["segments"][0]["excerpts"][0]["id"]
            name, arguments = (
                "record_observation",
                {
                    "topic": "Deployment",
                    "observation": "The deployment policy changed.",
                    "evidence": evidence,
                },
            )
        elif "create_procedure" in names:
            evidence = body["source_observations"][0]["evidence"]
            if retire:
                name, arguments = (
                    "retire_procedure",
                    {"group": "deployment", "target": "P1", "evidence": evidence},
                )
            else:
                name, arguments = (
                    "create_procedure",
                    {
                        "group": "deployment",
                        "name": "deployment-approval",
                        "trigger": "When deploying.",
                        "entry": "Require explicit approval before deploying.",
                        "evidence": evidence,
                    },
                )
        elif "review_result" in names:
            if change_before_commit == "history":
                storage._execute(
                    "UPDATE user_playbooks SET content=? WHERE user_playbook_id=?",
                    ("Changed by another owner", existing[0].user_playbook_id),
                )
            elif change_before_commit == "source":
                storage._execute(
                    "UPDATE interactions SET content=? WHERE interaction_id=?",
                    (
                        "Changed original source",
                        body["source"]["sources"][0]["interaction_id"],
                    ),
                )
            name, arguments = (
                "review_result",
                {
                    "unit": "U0",
                    "decision": "accept",
                    "result_support": "pass",
                    "scope_conditions": "pass",
                    "useful_nonduplicate": "pass",
                    "preservation_and_correction": "pass",
                    "resource_consistency": "pass",
                    "reason": "The source directly supports the complete change.",
                },
            )
        else:
            raise AssertionError(f"Retired model stage invoked: {names}")
        return litellm.ModelResponse(
            model=params["model"],
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{len(calls)}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    return engine, storage, window, token, client, calls, existing


@pytest.mark.parametrize("retire", [False, True])
@pytest.mark.parametrize("fail_commit", [False, True])
def test_procedural_native_window_publishes_exact_plan_and_replays_without_models(
    pipeline, monkeypatch, retire, fail_commit
):
    engine, storage, window, token, client, calls, existing = setup(
        pipeline, monkeypatch, retire=retire
    )
    executor = WindowExecutor(engine.request_context, client)
    monkeypatch.setattr(
        executor,
        "_prepare_overflow_context",
        Mock(side_effect=AssertionError("Retired protected-record stage must not run")),
    )
    original_complete = storage.complete_extraction
    if fail_commit:
        monkeypatch.setattr(
            storage,
            "complete_extraction",
            Mock(side_effect=RuntimeError("synthetic interrupted commit")),
        )
        with pytest.raises(RuntimeError, match="interrupted commit"):
            executor.execute(window, token)
        assert len(calls) == 3
        monkeypatch.setattr(storage, "complete_extraction", original_complete)
        # Refresh actual saved outcome, as the real worker does on continuation.
        window = storage.prepare_extraction(window.user_id, token)
    executor.execute(window, token)
    assert len(calls) == 3
    run = storage.get_agent_run(f"window:{window.window_id}")
    checkpoint = run.generation_request_snapshot["overflow"]["checkpoint"]
    assert (
        run.generation_request_snapshot["overflow"]["preparation_version"]
        == native.PREPARATION_VERSION
    )
    assert native.KEY in checkpoint
    assert len(checkpoint[native.KEY]["reader_steps"]) == 1
    assert len(run.generation_request_snapshot["overflow"]["attempts"]) == 3
    rows = storage.get_user_playbooks(user_id="u", agent_version="v")
    if retire:
        assert rows == []
        with storage._stream_sql(read_only=True) as db:
            retired = db.query(
                "SELECT status, superseded_by FROM user_playbooks WHERE user_playbook_id=? LIMIT 1",
                (existing[0].user_playbook_id,),
            )
        assert len(retired) == 1
        assert (
            retired[0]["status"] is not None or retired[0]["superseded_by"] is not None
        )
    else:
        assert len(rows) == 1
        assert "Require explicit approval" in rows[0].content
        assert rows[0].source_interaction_ids
        assert rows[0].expanded_terms is None
    assert storage.extraction_status("u", "procedural")["status"] == "done"


@pytest.mark.parametrize("change", ["history", "source"])
def test_changed_original_evidence_refuses_native_publication(
    pipeline, monkeypatch, change
):
    engine, storage, window, token, client, calls, existing = setup(
        pipeline, monkeypatch, retire=True, change_before_commit=change
    )
    from reflexio.server.llm.input_budget import ComputationEvidenceChangedError
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        WindowInputsDeletedError,
    )

    with pytest.raises(
        (
            ComputationEvidenceChangedError,
            WindowInputsDeletedError,
            ExtractionInputIntegrityError,
        )
    ):
        WindowExecutor(engine.request_context, client).execute(window, token)
    assert len(calls) == 3
    with storage._stream_sql(read_only=True) as db:
        rows = db.query(
            "SELECT status, superseded_by FROM user_playbooks WHERE user_playbook_id=? LIMIT 1",
            (existing[0].user_playbook_id,),
        )
    assert rows[0]["status"] is None and rows[0]["superseded_by"] is None
    assert storage.extraction_status("u", "procedural")["status"] != "done"


def test_embedding_retry_restores_all_three_completed_model_stages(
    pipeline, monkeypatch
):
    engine, storage, window, token, client, calls, _ = setup(pipeline, monkeypatch)
    from reflexio.server.llm.providers.embedding_service_provider import (
        EmbeddingUnavailableError,
    )

    original = storage.precompute_user_playbook_embeddings
    monkeypatch.setattr(
        storage,
        "precompute_user_playbook_embeddings",
        Mock(side_effect=EmbeddingUnavailableError("synthetic unavailable index")),
    )
    with pytest.raises(EmbeddingUnavailableError, match="unavailable index"):
        WindowExecutor(engine.request_context, client).execute(window, token)
    assert len(calls) == 3
    assert storage.get_user_playbooks(user_id="u", agent_version="v") == []
    monkeypatch.setattr(storage, "precompute_user_playbook_embeddings", original)
    WindowExecutor(engine.request_context, client).execute(window, token)
    assert len(calls) == 3
    assert len(storage.get_user_playbooks(user_id="u", agent_version="v")) == 1
