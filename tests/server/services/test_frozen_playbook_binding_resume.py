"""A historical parent keeps its evidence and quota across schema retirement."""

import copy
import json

import litellm
import pytest

from reflexio.models.config_schema import UserPlaybookExtractorConfig
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning import window_executor
from reflexio.server.services.extraction import resumable_agent
from reflexio.server.services.extraction.overflow_presentation import (
    OverflowPresentation,
    present_extraction_chunk,
)
from reflexio.server.services.extraction.protected_record import ProtectedRecord
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookList,
)
from reflexio.server.services.storage.storage_base import build_scope_hash
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


def test_retired_playbook_name_readmits_original_parent_without_refunding_calls(
    pipeline, monkeypatch
):
    engine, _, http = pipeline
    configurator = engine.request_context.configurator
    configurator.set_config(
        configurator.get_config().model_copy(
            update={
                "profile_extractor_config": None,
                "user_playbook_extractor_config": UserPlaybookExtractorConfig(
                    extraction_definition_prompt="Extract deployment instructions"
                ),
            }
        )
    )
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", '{"gpt-4o-mini":128000}')
    assert http.post(
        "/api/publish_interaction", json=durable.payload("aged-playbook", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("aged-parent-test", 600)
    window = storage.prepare_extraction(user, token)
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", max_tokens=8192, fallback_models=[])
    )
    executor = window_executor.WindowExecutor(engine.request_context, client)
    service, request, config = executor.service(window)
    coordinator = executor._coordinator(window, token, service, request, config)
    assert coordinator is not None
    parent = storage.get_agent_run(f"window:{window.window_id}")
    snapshot = copy.deepcopy(parent.generation_request_snapshot)
    snapshot["extractor_config"]["extractor_name"] = "historical-playbook-label"
    original_hash = build_scope_hash(snapshot["extractor_config"])
    journal = snapshot["overflow"]
    journal["attempts"] = [
        {"id": f"paid-{i}", "usage": None, "model_provenance": None} for i in range(5)
    ]
    journal["revision"] = 23
    policy = copy.deepcopy(window.policy)
    policy["extractor"] = copy.deepcopy(snapshot["extractor_config"])
    with storage.conn:
        storage.conn.execute(
            "UPDATE _agent_runs SET extractor_config_hash=?,generation_request_snapshot=? WHERE id=?",
            (original_hash, json.dumps(snapshot), parent.id),
        )
        storage.conn.execute(
            "UPDATE extraction_windows SET policy=? WHERE window_id=?",
            (json.dumps(policy), window.window_id),
        )
    resumed = storage.prepare_extraction(user, token)
    service, request, config = executor.service(resumed)
    compatible = window_executor.matches_frozen_extractor_binding
    monkeypatch.setattr(
        window_executor,
        "matches_frozen_extractor_binding",
        lambda prior, expected, _: prior.binding == expected.binding,
    )
    with pytest.raises(InputBudgetConfigurationError, match="parent binding changed"):
        executor._coordinator(resumed, token, service, request, config)
    monkeypatch.setattr(window_executor, "matches_frozen_extractor_binding", compatible)
    admitted = executor._coordinator(resumed, token, service, request, config)
    assert admitted is not None
    assert admitted.saved_run is not None
    assert admitted.saved_run.id == parent.id
    assert admitted.saved_run.binding.extractor_config_hash == original_hash
    assert admitted.saved_run.generation_request_snapshot == snapshot
    assert storage.get_agent_run(parent.id).generation_request_snapshot == snapshot
    assert admitted.checkpoint == journal["checkpoint"]

    # The generic agent rebuilds its run with today's normalized config. This
    # reaches the child guard too, unlike a package consumer reusing the parent.
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-no-provider")
    for module in ("tools", "_litellm_text_generation"):
        monkeypatch.setattr(
            f"reflexio.server.llm.{module}.resolve_model_name",
            lambda **_: "gpt-4o-mini",
        )
    calls = []

    def wire(params, _timeout):
        calls.append(copy.deepcopy(params))
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": '{"playbooks":[]}'},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    prepared = admitted.require_prepared()
    chunk = prepared.chunks[0]
    messages = [
        {
            "role": "user",
            "content": OverflowPresentation(prepared, chunk, ProtectedRecord()).text,
        }
    ]

    def run_child():
        with (
            admitted.provider_scope(),
            present_extraction_chunk(prepared, chunk, ProtectedRecord()),
        ):
            return resumable_agent.run_resumable_extraction_agent(
                request_context=engine.request_context,
                client=client,
                extractor_kind="playbook",
                user_id=resumed.user_id,
                request_id=request.request_id,
                agent_version=request.agent_version,
                source=request.source,
                request_interaction_data_models=service.service_config.window_interactions,
                extractor_config=config,
                service_config=service.service_config,
                agent_context=admitted.agent_context,
                messages=messages,
                output_schema=StructuredPlaybookList,
                log_label="historical-playbook-child",
            )

    child_compatible = resumable_agent.matches_frozen_extractor_binding
    monkeypatch.setattr(
        resumable_agent,
        "matches_frozen_extractor_binding",
        lambda prior, expected, _: prior.binding == expected.binding,
    )
    with pytest.raises(RuntimeError, match="child parent binding changed"):
        run_child()
    assert calls == []
    monkeypatch.setattr(
        resumable_agent, "matches_frozen_extractor_binding", child_compatible
    )
    result = run_child()
    assert result.output == StructuredPlaybookList(playbooks=[])
    assert run_child().output == result.output
    assert len(calls) == 1
    saved = storage.get_agent_run(parent.id)
    assert saved.binding.extractor_config_hash == original_hash
    assert (
        saved.generation_request_snapshot["extractor_config"]
        == snapshot["extractor_config"]
    )
    attempts = saved.generation_request_snapshot["overflow"]["attempts"]
    assert attempts[:5] == journal["attempts"]
    assert len(attempts) == 6
