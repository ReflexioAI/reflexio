"""Mechanical journal/native-cache composition; no consumer or publication claim."""

import copy
import hashlib
import json
from dataclasses import replace
from unittest.mock import Mock

import litellm
import pytest

from reflexio.server.llm.input_budget import extraction_provider_guard
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning.window_executor import (
    WindowExecutor,
    load_window_inputs,
)
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
)
from reflexio.server.services.extraction.compression import plan_extraction_window
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.overflow_protocol import package_stage_caps
from reflexio.server.services.extraction.package_compression import (
    compress_source_package,
)
from reflexio.server.services.input_compression import CompressionBudget
from tests.server.services import test_durable_window_pipeline as durable

package_pipeline = durable.pipeline
pytestmark = pytest.mark.integration


def prepared_package(package_pipeline, monkeypatch):
    engine, _, http = package_pipeline
    payload = durable.payload("package-compression", force=True)
    payload["interaction_data_list"][0]["tools_used"] = [
        {"tool_name": "diagnostic", "tool_data": {"output": "routine trace " * 2500}}
    ]
    assert http.post("/api/publish_interaction", json=payload).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("package-compression-test", 600)
    window = storage.prepare_extraction(user, token)
    client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", Mock(return_value=100)
    )
    service, request, config = WindowExecutor(engine.request_context, client).service(
        window
    )
    plan = plan_extraction_window(
        service.service_config.window_interactions,
        org_id=storage.org_id,
        project_id=window.project_id,
        user_id=user,
        budget=CompressionBudget(output_tokens=1024, section_tokens=1200),
        client=client,
        context_limits={client.config.model: 512000},
    )
    parent = build_extractor_agent_run_record(
        org_id=storage.org_id,
        extractor_kind=window.kind,
        user_id=user,
        generation_request_id=request.request_id,
        agent_version=None,
        source=request.source,
        request_interaction_data_models=service.service_config.window_interactions,
        extractor_config=config,
        service_config=service.service_config,
        agent_context="",
    )
    # Reserve a synthetically declared follow-on envelope solely for this cache
    # test. This does not qualify or admit the native profile consumer contract.
    consumer = {
        "kind": "profile",
        "gate_enabled": False,
        "ladders": {
            "gate": [],
            "agent": [client.config.model],
            "dedup": [client.config.model],
            "intent": [],
        },
    }
    descriptor = {
        "name": "source-package",
        "version": 1,
        "plan_digest": plan.witness,
        "source_cache_digest": plan.source_cache_digest,
        "contract_digest": hashlib.sha256(b"compression-only-test-v1").hexdigest(),
        "scope": {
            "org_id": storage.org_id,
            "project_id": window.project_id,
            "user_id": user,
            "window_id": window.window_id,
            "request_id": request.request_id,
            "agent_version": None,
            "kind": window.kind,
        },
        "section_digests": list(plan.native.compression.section_digests),
        "consumer": consumer,
        "stage_caps": package_stage_caps(consumer),
    }
    manifest = OverflowCoordinator.initial_package_manifest(
        plan.source_digest, descriptor
    )
    parent = replace(parent, id=f"window:{window.window_id}")
    parent.generation_request_snapshot["overflow"] = manifest
    if parent.service_config_snapshot is not None:
        parent.service_config_snapshot.pop("window_interactions", None)
    with storage.commit_scope():
        storage.fence_user_extraction(user, token)
        storage.validate_extraction_source_digest(window, plan.source_digest)
        storage.create_agent_run(parent)
    return client, plan, storage, window, token, manifest


def coordinator(storage, window, token, manifest):
    return OverflowCoordinator(
        storage,
        window,
        token,
        manifest,
        source_digest=manifest["source_digest"],
        context_limits={"minimax/MiniMax-M3": 512000},
        turn_seconds=600,
    )


def rebuild_plan(storage, window, client):
    return plan_extraction_window(
        load_window_inputs(storage, window),
        org_id=storage.org_id,
        project_id=window.project_id,
        user_id=window.user_id,
        budget=CompressionBudget(output_tokens=1024, section_tokens=1200),
        client=client,
        context_limits={client.config.model: 512000},
    )


def install_transport(client, monkeypatch):
    dispatched = []

    def transport(params, _hard_timeout):
        dispatched.append(params)
        body = json.loads(params["messages"][1]["content"])
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "points": [
                                    {"evidence_refs": body["allowed_evidence_refs"][:1]}
                                ]
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    return dispatched


def test_native_selections_replay_from_real_durable_parent_without_new_dispatch(
    package_pipeline, monkeypatch
):
    client, plan, storage, window, token, manifest = prepared_package(
        package_pipeline, monkeypatch
    )
    dispatched = install_transport(client, monkeypatch)
    first = compress_source_package(plan, coordinator(storage, window, token, manifest))
    saved = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert first.result.calls == first.result.provider_attempts == len(dispatched) > 0
    assert all(attempt["stage"] == "compress" for attempt in saved["attempts"])
    assert all(
        attempt["usage"]["prompt_tokens"] == 100 for attempt in saved["attempts"]
    )
    assert len(saved["checkpoint"]["completed_steps"]) == first.result.calls
    for key, receipt in saved["checkpoint"]["completed_steps"].items():
        assert receipt["node_key"] == key and receipt["attempt_ids"]
        assert receipt["request_contract_digest"] == plan.native.request_contract_digest
        assert receipt["input_refs_digest"] != key
    frozen = copy.deepcopy(saved)
    forbidden = Mock(side_effect=AssertionError("Completed selections must be reused"))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", forbidden)
    fresh_plan = rebuild_plan(storage, window, client)
    assert fresh_plan.witness == plan.witness
    replay = compress_source_package(
        fresh_plan, coordinator(storage, window, token, saved)
    )
    assert replay.result.provider_attempts == 0
    assert replay.result.packet == first.result.packet
    assert replay.presentation == first.presentation
    forbidden.assert_not_called()
    assert (
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ]
        == frozen
    )
    assert storage.count_all_profiles() == 0
    assert extraction_provider_guard.get() is None


@pytest.mark.parametrize("field", ["input_refs_digest", "request_contract_digest"])
def test_saved_node_witness_tamper_refuses_before_another_native_call(
    package_pipeline, monkeypatch, field
):
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError

    client, plan, storage, window, token, manifest = prepared_package(
        package_pipeline, monkeypatch
    )
    install_transport(client, monkeypatch)
    compress_source_package(plan, coordinator(storage, window, token, manifest))
    parent = storage.get_agent_run(f"window:{window.window_id}")
    snapshot = copy.deepcopy(parent.generation_request_snapshot)
    saved = snapshot["overflow"]
    next(iter(saved["checkpoint"]["completed_steps"].values()))[field] = "f" * 64
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), parent.id),
        )
    forbidden = Mock(side_effect=AssertionError("Tampered proof must not dispatch"))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", forbidden)
    with pytest.raises(
        InputBudgetConfigurationError, match="Saved compression node changed"
    ):
        compress_source_package(plan, coordinator(storage, window, token, saved))
    forbidden.assert_not_called()
    assert storage.get_agent_run(parent.id).generation_request_snapshot == snapshot
    assert extraction_provider_guard.get() is None


def test_partial_node_save_survives_restart_without_repaying_first_selection(
    package_pipeline, monkeypatch
):
    client, plan, storage, window, token, manifest = prepared_package(
        package_pipeline, monkeypatch
    )
    dispatched = install_transport(client, monkeypatch)
    persist = storage.checkpoint_extraction_overflow

    def crash_after_receipt(*args, **kwargs):
        saved = persist(*args, **kwargs)
        if kwargs.get("checkpoint", {}).get("completed_steps"):
            raise RuntimeError("Simulated exit after durable selection commit")
        return saved

    monkeypatch.setattr(storage, "checkpoint_extraction_overflow", crash_after_receipt)
    with pytest.raises(RuntimeError, match="Simulated exit"):
        compress_source_package(plan, coordinator(storage, window, token, manifest))
    saved = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert (
        len(saved["attempts"])
        == len(saved["checkpoint"]["completed_steps"])
        == len(dispatched)
        == 1
    )
    monkeypatch.setattr(storage, "checkpoint_extraction_overflow", persist)
    fresh_plan = rebuild_plan(storage, window, client)
    assert fresh_plan.witness == plan.witness
    replay = compress_source_package(
        fresh_plan, coordinator(storage, window, token, saved)
    )
    assert replay.result.cache_hits >= 1
    final = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert final["attempts"][0] == saved["attempts"][0]
    assert replay.result.provider_attempts is not None
    assert (
        len(final["attempts"]) == len(dispatched) == replay.result.provider_attempts + 1
    )
    assert all(
        final["checkpoint"]["completed_steps"][key] == receipt
        for key, receipt in saved["checkpoint"]["completed_steps"].items()
    )
    assert storage.count_all_profiles() == 0


def test_replaced_original_source_refuses_before_native_dispatch(
    package_pipeline, monkeypatch
):
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        WindowInputsDeletedError,
    )

    client, plan, storage, window, token, manifest = prepared_package(
        package_pipeline, monkeypatch
    )
    with storage._stream_sql() as db:
        db.query(
            "UPDATE interactions SET content=? WHERE interaction_id=?",
            ("Replaced original source", window.manifest[0]["interaction_id"]),
        )
    forbidden = Mock(side_effect=AssertionError("Changed original must not dispatch"))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", forbidden)
    with pytest.raises(WindowInputsDeletedError, match="content was replaced"):
        compress_source_package(plan, coordinator(storage, window, token, manifest))
    forbidden.assert_not_called()
    assert (
        storage.get_agent_run(f"window:{window.window_id}").generation_request_snapshot[
            "overflow"
        ]["attempts"]
        == []
    )
