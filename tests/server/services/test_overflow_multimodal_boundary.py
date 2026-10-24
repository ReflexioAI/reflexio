"""Native multimodal extraction remains outside text-only overflow admission."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.lib._base import create_generation_litellm_client
from reflexio.models.config_schema import UserPlaybookExtractorConfig
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
)
from reflexio.server.services.extraction.input_preparation import (
    prepare_extraction_input,
)
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from tests.server.services.test_durable_window_pipeline import (
    payload,
)
from tests.server.services.test_durable_window_pipeline import (
    pipeline as pipeline,
)

pytestmark = pytest.mark.integration

CAPTION = "Prepare a draft only; obtain separate approval before deploying."
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII="


def _publish_mixed(pipeline, monkeypatch, kind, image_kind):
    engine, _, http = pipeline
    if kind == "playbook":
        config = engine.request_context.configurator
        config.set_config(
            config.get_config().model_copy(
                update={
                    "profile_extractor_config": None,
                    "user_playbook_extractor_config": UserPlaybookExtractorConfig(
                        extraction_definition_prompt="Extract scoped approval practices."
                    ),
                }
            )
        )
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    data = payload("mixed-image", force=True)
    data["agent_version"] = "v1"
    for interaction in data["interaction_data_list"]:
        interaction["content"] = CAPTION
    field, value = (
        ("interacted_image_url", "https://example.com/approval.png")
        if image_kind == "url"
        else ("image_encoding", PNG)
    )
    data["interaction_data_list"][-1][field] = value
    assert http.post("/api/publish_interaction", json=data).json()["success"]
    return value if image_kind == "url" else f"data:image/png;base64,{PNG}"


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("image_kind", ["url", "base64"])
def test_http_mixed_image_keeps_native_prompt_and_persistence_without_overflow(
    pipeline, monkeypatch, kind, image_kind
):
    engine, worker, _ = pipeline
    expected_image = _publish_mixed(pipeline, monkeypatch, kind, image_kind)
    storage = engine.get_storage()
    monkeypatch.setattr(
        storage, "_get_embedding", lambda *_args, **_kwargs: [0.1] * 512
    )
    monkeypatch.setattr(storage, "_expand_document", lambda *_args, **_kwargs: None)
    image_blocks = []

    def respond(params, hard_timeout):
        prompt = json.dumps(params["messages"], ensure_ascii=False)
        name = getattr(params.get("response_format"), "__name__", "") or json.dumps(
            params.get("response_format"), default=str
        )
        assert "Select original evidence needed" not in prompt
        for message in params["messages"]:
            if isinstance(message.get("content"), list):
                image_blocks.extend(
                    block["image_url"]["url"]
                    for block in message["content"]
                    if block.get("type") == "image_url"
                )
        if "unique_ids" in prompt or "ProfileDeduplicationOutput" in name:
            output = {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
        elif "reason_code" in prompt or "Review" in name:
            output = {
                "decisions": [
                    {
                        "id": "C1",
                        "decision": "accept",
                        "reason_code": "grounded_useful",
                        "evidence_ids": ["C1-E1"],
                    }
                ]
            }
            if "supporting_evidence" in prompt:
                output["supporting_evidence"] = [
                    {
                        "id": "C1",
                        "excerpts": [{"evidence_id": "C1-E1", "text": CAPTION}],
                    }
                ]
        elif kind == "profile":
            output = {
                "profiles": [
                    {
                        "content": "The user requires separate deployment approval.",
                        "time_to_live": "infinity",
                        "source_span": CAPTION,
                    }
                ]
            }
        else:
            assert "T10" in prompt
            output = {
                "playbooks": [
                    {
                        "content": "Prepare a draft and ask separately before deploying.",
                        "trigger": "When preparing a release",
                        "rationale": "The user restricts approval to preparation.",
                        "evidence_kind": "correction",
                        "evidence_refs": ["T10"],
                    }
                ]
            }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(output), tool_calls=None, refusal=None
                    ),
                    finish_reason="stop",
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
    worker.drain_org("stream-test", 1, 300)
    assert dispatch.call_count > 0
    assert expected_image in image_blocks
    assert storage.extraction_status("u", "mixed-image")["status"] == "done"
    with storage._stream_sql() as db:
        runs = db.query("SELECT id FROM _agent_runs WHERE user_id=? LIMIT 10", ("u",))
        if kind == "playbook":
            assert (
                len(
                    db.query(
                        "SELECT user_playbook_id FROM user_playbooks WHERE user_id=? LIMIT 2",
                        ("u",),
                    )
                )
                == 1
            )
        else:
            assert storage.count_all_profiles() == 1
    assert runs
    assert all(
        "overflow" not in storage.get_agent_run(row["id"]).generation_request_snapshot
        for row in runs
    )


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("enabled", [False, True])
def test_existing_mixed_image_manifest_holds_without_reset_or_publication(
    pipeline, monkeypatch, kind, enabled
):
    engine, _, _ = pipeline
    _publish_mixed(pipeline, monkeypatch, kind, "url")
    storage = engine.get_storage()
    user, token = storage.claim_extraction("mixed-image-test", 300)
    window = storage.prepare_extraction(user, token)
    assert window is not None
    executor = WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    service, request, config = executor.service(window)
    groups = service.service_config.window_interactions
    prepared = prepare_extraction_input(groups, user_id=user)
    run = build_extractor_agent_run_record(
        org_id=storage.org_id,
        extractor_kind=kind,
        user_id=user,
        generation_request_id=request.request_id,
        agent_version=getattr(request, "agent_version", None),
        source=request.source,
        request_interaction_data_models=groups,
        extractor_config=config,
        service_config=service.service_config,
        agent_context=engine.request_context.configurator.get_agent_context(),
    )
    manifest = OverflowCoordinator.initial_manifest(prepared.source_digest)
    manifest["attempts"] = [
        {
            "id": "spent-attempt",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cache_read_input_tokens": 0,
                "cache_write_input_tokens": 0,
            },
            "model_provenance": None,
        }
    ]
    manifest["checkpoint"]["completed_steps"] = {
        "frozen-step": {
            "output": {"profiles": []},
            "model_provenance": {"model_name": None, "provider": None},
        }
    }
    run = replace(run, id=f"window:{window.window_id}")
    run.generation_request_snapshot["overflow"] = manifest
    storage.create_agent_run(run)
    before = copy.deepcopy(storage.get_agent_run(run.id).generation_request_snapshot)
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", str(enabled).lower())
    dispatch = Mock(
        side_effect=AssertionError("Existing image manifest must not call a provider")
    )
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    with pytest.raises(InputBudgetConfigurationError, match="image evidence"):
        executor.execute(window, token)
    dispatch.assert_not_called()
    assert storage.get_agent_run(run.id).generation_request_snapshot == before
    assert storage.count_all_profiles() == 0
    with storage._stream_sql() as db:
        assert (
            db.query(
                "SELECT user_playbook_id FROM user_playbooks WHERE user_id=? LIMIT 1",
                (user,),
            )
            == []
        )
    assert storage.pending_extraction_effects() == []
