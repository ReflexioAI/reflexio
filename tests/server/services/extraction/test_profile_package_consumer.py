"""HTTP originals, native consumer requests and real durable SQLite journals."""

import copy
import json
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import Mock

import litellm
import pytest
from pydantic import Field, create_model

from reflexio.models.config_schema import (
    PendingToolCallConfig,
    PendingToolCallToolOverrideConfig,
)
from reflexio.server.llm._litellm_types import ProviderRequestGuardError
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
)
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowCoordinator,
    OverflowSuspendedError,
)
from reflexio.server.services.extraction.prior_answer_search import (
    capture_prior_knowledge_context,
)
from reflexio.server.services.extraction.profile_package_consumer import (
    consume_profile_package,
)
from reflexio.server.services.extraction.profile_package_preparation import (
    prepare_profile_package,
    profile_package_original_seed,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_scope_hash,
    human_feedback_scope,
)
from tests.server.services import test_durable_window_pipeline as durable

package_pipeline = durable.pipeline
pytestmark = pytest.mark.integration


def prepare_consumer(
    package_pipeline,
    monkeypatch,
    *,
    gate_enabled=True,
    pending_config=None,
    long_tools=False,
):
    engine, _, http = package_pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-no-provider")
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_text_generation.resolve_model_name",
        lambda **_: "gpt-4o-mini",
    )
    monkeypatch.setattr(
        "reflexio.server.llm.tools.resolve_model_name", lambda **_: "gpt-4o-mini"
    )
    payload = durable.payload("profile-package")
    if long_tools:
        payload["interaction_data_list"][0]["tools_used"] = [
            {
                "tool_name": "diagnostic",
                "tool_data": {"output": "routine trace " * 40000},
            }
        ]
    assert http.post("/api/publish_interaction", json=payload).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("profile-package-test", 600)
    window = storage.prepare_extraction(user, token)
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", max_tokens=8192, fallback_models=[])
    )
    service, request, config = WindowExecutor(engine.request_context, client).service(
        window
    )
    groups = service.service_config.window_interactions
    prior = None
    if pending_config is not None:
        monkeypatch.setattr(storage, "_get_embedding", Mock(return_value=[1.0, 0.0]))
        prior = capture_prior_knowledge_context(
            messages=profile_package_original_seed(
                prompt_manager=engine.request_context.prompt_manager,
                groups=groups,
                existing_profiles=[],
                agent_context="",
                extractor_config=config,
            ),
            storage=storage,
            org_id=storage.org_id,
            extractor_kind="profile",
            extractor_config=config,
            source=request.source,
            agent_version=None,
            similarity_threshold=pending_config.for_tool(
                "attach_pending_info_request"
            ).similarity_threshold,
        )
    preparation = prepare_profile_package(
        groups,
        org_id=storage.org_id,
        project_id=window.project_id,
        user_id=user,
        source=request.source,
        client=client,
        context_limits={"gpt-4o-mini": 128000},
        prompt_manager=engine.request_context.prompt_manager,
        extractor_config=config,
        existing_profiles=[],
        agent_context="",
        gate_enabled=gate_enabled,
        gate_model="gpt-4o-mini",
        dedup_model="gpt-4o-mini",
        prior_context=prior,
        pending_tools_active=pending_config is not None,
        pending_tool_config=pending_config,
    )
    plan = preparation.compression
    snapshot = preparation.snapshot
    descriptor = {
        "name": "source-package",
        "version": 1,
        "plan_digest": plan.witness,
        "source_cache_digest": plan.source_cache_digest,
        "contract_digest": preparation.contract_digest,
        "scope": {
            "org_id": storage.org_id,
            "project_id": window.project_id,
            "user_id": user,
            "window_id": window.window_id,
            "request_id": request.request_id,
            "agent_version": None,
            "kind": "profile",
        },
        "section_digests": list(plan.native.compression.section_digests),
        "consumer": snapshot["consumer"],
        "stage_caps": snapshot["stage_caps"],
    }
    manifest = OverflowCoordinator.initial_package_manifest(
        plan.source_digest, descriptor
    )
    if prior is not None:
        manifest["checkpoint"]["answer_witnesses"] = prior.answer_witnesses
    parent = build_extractor_agent_run_record(
        org_id=storage.org_id,
        extractor_kind="profile",
        user_id=user,
        generation_request_id=request.request_id,
        agent_version=None,
        source=request.source,
        request_interaction_data_models=groups,
        extractor_config=config,
        service_config=service.service_config,
        agent_context="",
    )
    parent = replace(parent, id=f"window:{window.window_id}")
    parent.generation_request_snapshot.update(
        overflow=manifest, profile_package_preparation=snapshot
    )
    assert parent.service_config_snapshot is not None
    parent.service_config_snapshot.pop("window_interactions", None)
    with storage.commit_scope():
        storage.fence_user_extraction(user, token)
        storage.validate_extraction_source_digest(window, plan.source_digest)
        storage.create_agent_run(parent)
    return client, preparation, storage, window, token


def coordinator(preparation, storage, window, token):
    parent = storage.get_agent_run(f"window:{window.window_id}")
    current = OverflowCoordinator(
        storage,
        window,
        token,
        parent.generation_request_snapshot["overflow"],
        source_digest=preparation.compression.source_digest,
        context_limits={"gpt-4o-mini": 128000},
        turn_seconds=600,
    )
    current.saved_run = parent
    return current


def install_transport(
    client, monkeypatch, *, gate=True, profiles=True, bad_quote=False
):
    calls = []

    def wire(params, _timeout):
        calls.append(copy.deepcopy(params))
        content = str(gate).lower()
        body = None
        if len(params["messages"]) > 1:
            with suppress(ValueError, TypeError):
                body = json.loads(params["messages"][1]["content"])
        if isinstance(body, dict) and "allowed_evidence_refs" in body:
            content = json.dumps(
                {"points": [{"evidence_refs": body["allowed_evidence_refs"][:1]}]}
            )
        elif params.get("response_format") is not None:
            content = json.dumps(
                {
                    "profiles": [
                        {
                            "content": "Prefers concise answers",
                            "time_to_live": "infinity",
                            "source_span": "absent quotation"
                            if bad_quote
                            else "I prefer concise answers with details 0",
                        }
                    ]
                    if profiles
                    else []
                }
            )
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    return calls


@pytest.mark.parametrize(
    ("gate_enabled", "gate", "profiles"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_actual_native_consumer_replays_without_dispatch_or_parent_completion(
    package_pipeline, monkeypatch, gate_enabled, gate, profiles
):
    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline, monkeypatch, gate_enabled=gate_enabled
    )
    calls = install_transport(client, monkeypatch, gate=gate, profiles=profiles)
    first = consume_profile_package(
        preparation=preparation,
        coordinator=coordinator(preparation, storage, window, token),
        client=client,
    )
    assert first.gate_decision == gate
    assert first.package.result.passthrough
    assert first.package.node_keys == ()
    assert (first.agent is not None) == gate
    if first.agent is not None:
        assert isinstance(first.agent.output, StructuredProfilesOutput)
        assert bool(first.agent.output.profiles) == profiles
    parent = storage.get_agent_run(f"window:{window.window_id}")
    frozen = copy.deepcopy(parent.generation_request_snapshot)
    stages = [attempt["stage"] for attempt in frozen["overflow"]["attempts"]]
    assert stages == (["gate"] if gate_enabled else []) + (
        ["profile_agent"] if gate else []
    )
    assert all(
        attempt["usage"]["prompt_tokens"] == 100
        for attempt in frozen["overflow"]["attempts"]
    )
    refuse = Mock(side_effect=AssertionError("replay must not dispatch"))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", refuse)
    replay = consume_profile_package(
        preparation=preparation,
        coordinator=coordinator(preparation, storage, window, token),
        client=client,
        cache_only=True,
    )
    assert replay.package.presentation == first.package.presentation
    assert replay.gate_decision == first.gate_decision
    assert storage.get_agent_run(parent.id).generation_request_snapshot == frozen
    assert storage.get_agent_run(parent.id).status == AgentRunStatus.RUNNING
    assert storage.get_agent_run(parent.id).committed_output is None
    assert storage.count_all_profiles() == 0
    assert storage.pending_extraction_effects() == []
    assert len(storage.get_user_interaction("u")) == 10
    assert len(calls) == len(stages)
    refuse.assert_not_called()


def test_actual_native_consumer_stage_cap_survives_restart(
    package_pipeline, monkeypatch
):
    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline, monkeypatch
    )
    calls = install_transport(client, monkeypatch)
    first = consume_profile_package(
        preparation=preparation,
        coordinator=coordinator(preparation, storage, window, token),
        client=client,
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    frozen = copy.deepcopy(parent.generation_request_snapshot)
    current = coordinator(preparation, storage, window, token)
    messages = preparation.gate_messages(first.package.presentation.text)
    gate_key = next(
        key
        for key, receipt in frozen["overflow"]["checkpoint"]["completed_steps"].items()
        if isinstance(receipt["output"], dict) and "decision" in receipt["output"]
    )
    with (
        current.profile_request(preparation, "gate", messages, receipt_key=gate_key),
        pytest.raises(ProviderRequestGuardError, match="reservation failed"),
    ):
        client.generate_chat_response_with_provenance(
            messages=messages, model="gpt-4o-mini"
        )
    assert storage.get_agent_run(parent.id).generation_request_snapshot == frozen
    assert len(calls) == 2


def test_consumer_requires_exact_frozen_preparation_before_any_dispatch(
    package_pipeline, monkeypatch
):
    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline, monkeypatch
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    parent.generation_request_snapshot["profile_package_preparation"][
        "agent_context"
    ] = "changed"
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(parent.generation_request_snapshot), parent.id),
        )
    wire = Mock(side_effect=AssertionError("no dispatch"))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    with pytest.raises(InputBudgetConfigurationError, match="binding changed"):
        consume_profile_package(
            preparation=preparation,
            coordinator=coordinator(preparation, storage, window, token),
            client=client,
        )
    wire.assert_not_called()
    assert (
        storage.get_agent_run(parent.id).generation_request_snapshot["overflow"][
            "attempts"
        ]
        == []
    )


def test_missing_cache_only_processing_receipt_cannot_start_paid_work(
    package_pipeline, monkeypatch
):
    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline, monkeypatch
    )
    wire = Mock(side_effect=AssertionError("no dispatch"))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    frozen = copy.deepcopy(parent.generation_request_snapshot)
    with pytest.raises(InputBudgetConfigurationError, match="processing receipt"):
        consume_profile_package(
            preparation=preparation,
            coordinator=coordinator(preparation, storage, window, token),
            client=client,
            cache_only=True,
        )
    assert storage.get_agent_run(parent.id).generation_request_snapshot == frozen
    wire.assert_not_called()


def legacy_warning_config(config, *, future_policy=False):
    """Construct the old serialized shape independently of the current schema."""
    legacy_override = create_model(
        "LegacyPendingToolOverride",
        __base__=PendingToolCallToolOverrideConfig,
        max_pending_followups_per_scope=(int, 73),
    )
    fields: dict[str, Any] = {
        "max_pending_followups_per_scope": (int, 37),
        "tool_overrides": (dict[str, Any], Field(default_factory=dict)),
    }
    if future_policy:
        fields["unknown_execution_policy"] = (int, 11)
    legacy_config = create_model(
        "LegacyPendingToolConfig", __base__=PendingToolCallConfig, **fields
    )
    values = config.model_dump(
        exclude={"tool_overrides", "max_pending_followups_per_scope"}
    )
    return legacy_config(
        **values,
        tool_overrides={
            name: legacy_override(
                **override.model_dump(exclude={"max_pending_followups_per_scope"})
            )
            for name, override in config.tool_overrides.items()
        },
    )


@pytest.mark.parametrize("historical_warning", [False, True])
def test_actual_pending_tool_wait_resumes_full_answer_and_remaining_turns(
    package_pipeline, monkeypatch, historical_warning
):
    engine, _, _ = package_pipeline
    storage = engine.get_storage()
    now = datetime.now(UTC)
    scope = human_feedback_scope(storage.org_id)
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="package-human",
            org_id=storage.org_id,
            user_id="other-user",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key="package-human-question",
            status=PendingToolCallStatus.PENDING,
            question_text="Which approval is needed?",
            embedding=[1.0, 0.0],
            expires_at=now + timedelta(days=1),
            cache_until=now + timedelta(hours=1),
        )
    )
    tools_config = PendingToolCallConfig(
        hmac_secrets=["fixture-signing-secret"],
        tool_overrides={
            "ask_human": PendingToolCallToolOverrideConfig(pending_ttl_seconds=90)
        },
    )
    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline,
        monkeypatch,
        gate_enabled=False,
        pending_config=legacy_warning_config(tools_config)
        if historical_warning
        else tools_config,
    )
    frozen_preparation = copy.deepcopy(preparation.snapshot)
    calls = []
    qualifier = "Background " * 200 + "ONLY with separate customer approval"

    def wire(params, _timeout):
        calls.append(copy.deepcopy(params))
        if len(calls) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-human",
                        "type": "function",
                        "function": {
                            "name": "attach_pending_info_request",
                            "arguments": json.dumps(
                                {
                                    "pending_tool_call_id": "package-human",
                                    "why_relevant": "approval conditions",
                                }
                            ),
                        },
                    }
                ],
            }
        else:
            assert qualifier in str(params["messages"])
            message = {"role": "assistant", "content": '{"profiles":[]}'}
        return litellm.ModelResponse(
            choices=[
                {
                    "message": message,
                    "finish_reason": "tool_calls" if len(calls) == 1 else "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)

    def consume():
        return consume_profile_package(
            preparation=preparation,
            coordinator=coordinator(preparation, storage, window, token),
            client=client,
            pending_tool_config=tools_config,
        )

    with pytest.raises(OverflowSuspendedError):
        consume()
    parent = storage.get_agent_run(f"window:{window.window_id}")
    active = parent.generation_request_snapshot["overflow"]["checkpoint"][
        "active_child"
    ]
    assert active["max_steps_remaining"] == 7
    assert len(active["attempt_ids"]) == 1
    assert active["messages"][-1]["role"] == "tool"
    assert "package-human" in active["messages"][-1]["content"]
    with pytest.raises(OverflowSuspendedError):
        consume()
    assert len(calls) == 1
    storage.resolve_pending_tool_call(
        "package-human", result={"answer": qualifier}, valid_for_seconds=3600
    )
    result = consume()
    assert result.agent is not None
    assert isinstance(result.agent.output, StructuredProfilesOutput)
    assert result.agent.output.profiles == []
    final = storage.get_agent_run(parent.id)
    journal = final.generation_request_snapshot["overflow"]
    assert [a["stage"] for a in journal["attempts"]] == [
        "profile_agent",
        "profile_agent",
    ]
    assert "waiting_child" not in journal["checkpoint"]
    assert "active_child" not in journal["checkpoint"]
    assert journal["checkpoint"]["answer_witnesses"]["package-human"]["result"] == {
        "answer": qualifier
    }
    child = journal["checkpoint"]["package_child"]["key"]
    assert journal["checkpoint"]["completed_steps"][child]["output"]["attempt_ids"] == [
        a["id"] for a in journal["attempts"]
    ]
    assert final.status == AgentRunStatus.RUNNING and final.committed_output is None
    assert storage.count_all_profiles() == 0
    assert final.generation_request_snapshot["profile_package_preparation"] == (
        frozen_preparation
    )


@pytest.mark.parametrize(
    "change", ["secrets", "ttl", "tool_override", "unknown_policy"]
)
def test_warning_retirement_keeps_real_frozen_pending_policy_strict(
    package_pipeline, monkeypatch, change
):
    config = PendingToolCallConfig(
        hmac_secrets=["fixture-signing-secret"],
        tool_overrides={
            "ask_human": PendingToolCallToolOverrideConfig(pending_ttl_seconds=90)
        },
    )
    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline,
        monkeypatch,
        gate_enabled=False,
        pending_config=legacy_warning_config(
            config, future_policy=change == "unknown_policy"
        ),
    )
    saved = copy.deepcopy(storage.get_agent_run(f"window:{window.window_id}"))
    if change == "secrets":
        config.hmac_secrets = ["different-signing-secret"]
    elif change == "ttl":
        config.pending_ttl_seconds += 1
    elif change == "tool_override":
        config.tool_overrides["ask_human"].pending_ttl_seconds = 91
    calls = install_transport(client, monkeypatch)
    with pytest.raises(
        InputBudgetConfigurationError, match="pending tool configuration"
    ):
        consume_profile_package(
            preparation=preparation,
            coordinator=coordinator(preparation, storage, window, token),
            client=client,
            pending_tool_config=config,
        )
    assert calls == []
    assert storage.get_agent_run(saved.id) == saved


def test_compressed_package_consumption_reuses_all_actual_selection_receipts(
    package_pipeline, monkeypatch
):
    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline, monkeypatch, gate_enabled=False, long_tools=True
    )
    calls = install_transport(client, monkeypatch, profiles=False)
    first = consume_profile_package(
        preparation=preparation,
        coordinator=coordinator(preparation, storage, window, token),
        client=client,
    )
    assert not first.package.result.passthrough
    assert first.package.node_keys
    parent = storage.get_agent_run(f"window:{window.window_id}")
    frozen = copy.deepcopy(parent.generation_request_snapshot)
    journal = frozen["overflow"]
    assert len(calls) == len(journal["attempts"])
    assert journal["attempts"][-1]["stage"] == "profile_agent"
    assert all(a["stage"] == "compress" for a in journal["attempts"][:-1])
    witness = journal["checkpoint"]["package_processing"]
    assert witness["node_keys"] == list(first.package.node_keys)
    assert witness["covered_source_ids"] == list(
        first.package.result.covered_source_ids
    )
    for key in first.package.node_keys:
        receipt = journal["checkpoint"]["completed_steps"][key]
        assert receipt["kind"] == "source_selection" and receipt["attempt_ids"]
    refuse = Mock(side_effect=AssertionError("cached package must not dispatch"))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", refuse)
    replay = consume_profile_package(
        preparation=preparation,
        coordinator=coordinator(preparation, storage, window, token),
        client=client,
        cache_only=True,
    )
    assert replay.package.presentation == first.package.presentation
    assert storage.get_agent_run(parent.id).generation_request_snapshot == frozen
    refuse.assert_not_called()


@pytest.mark.parametrize("stage", ["gate", "profile_agent"])
@pytest.mark.parametrize(
    "damage", ["missing", "empty", "duplicate", "unknown", "wrong_stage", "wrong_step"]
)
def test_cached_consumer_result_requires_exact_actual_attempts(
    package_pipeline, monkeypatch, stage, damage
):
    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline, monkeypatch
    )
    install_transport(client, monkeypatch)
    consume_profile_package(
        preparation=preparation,
        coordinator=coordinator(preparation, storage, window, token),
        client=client,
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    snapshot = parent.generation_request_snapshot
    journal = snapshot["overflow"]
    checkpoint = journal["checkpoint"]
    key = (
        checkpoint["package_child"]["key"]
        if stage == "profile_agent"
        else next(
            key
            for key, receipt in checkpoint["completed_steps"].items()
            if isinstance(receipt["output"], dict) and "decision" in receipt["output"]
        )
    )
    output = checkpoint["completed_steps"][key]["output"]
    identities = output["attempt_ids"]
    if damage == "missing":
        del output["attempt_ids"]
    elif damage == "empty":
        output["attempt_ids"] = []
    elif damage == "duplicate":
        output["attempt_ids"] = [identities[0], identities[0]]
    elif damage == "unknown":
        output["attempt_ids"] = ["never-paid"]
    elif damage == "wrong_stage":
        other = next(a["id"] for a in journal["attempts"] if a["stage"] != stage)
        output["attempt_ids"] = [other]
        checkpoint["profile_attempt_bindings"][other] = key
    else:
        checkpoint["profile_attempt_bindings"][identities[0]] = "f" * 64
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), parent.id),
        )
    wire = Mock(side_effect=AssertionError("damaged receipt must not dispatch"))
    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    with pytest.raises(InputBudgetConfigurationError):
        consume_profile_package(
            preparation=preparation,
            coordinator=coordinator(preparation, storage, window, token),
            client=client,
            cache_only=True,
        )
    assert storage.get_agent_run(parent.id).generation_request_snapshot == snapshot
    assert storage.count_all_profiles() == 0
    wire.assert_not_called()


def test_unknown_crash_attempt_stays_paid_without_inventing_success_receipt(
    package_pipeline, monkeypatch
):
    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline, monkeypatch, gate_enabled=False
    )

    def lost_response(_params, _timeout):
        parent = storage.get_agent_run(f"window:{window.window_id}")
        journal = parent.generation_request_snapshot["overflow"]
        identity = journal["attempts"][-1]["id"]
        assert (
            journal["checkpoint"]["profile_attempt_bindings"][identity]
            == journal["checkpoint"]["package_child"]["key"]
        )
        raise OverflowSuspendedError("Simulated lost response after reservation")

    monkeypatch.setattr(client, "_completion_with_hard_timeout", lost_response)
    with pytest.raises(OverflowSuspendedError):
        consume_profile_package(
            preparation=preparation,
            coordinator=coordinator(preparation, storage, window, token),
            client=client,
        )
    before = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert len(before["attempts"]) == 1 and before["attempts"][0]["usage"] is None
    calls = install_transport(client, monkeypatch)
    consume_profile_package(
        preparation=preparation,
        coordinator=coordinator(preparation, storage, window, token),
        client=client,
    )
    after = storage.get_agent_run(
        f"window:{window.window_id}"
    ).generation_request_snapshot["overflow"]
    assert len(calls) == 1 and len(after["attempts"]) == 2
    assert after["attempts"][0] == before["attempts"][0]
    child = after["checkpoint"]["package_child"]["key"]
    assert after["checkpoint"]["completed_steps"][child]["output"]["attempt_ids"] == [
        after["attempts"][1]["id"]
    ]


def test_paid_attempt_logical_binding_cannot_be_reassigned(
    package_pipeline, monkeypatch
):
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        OverflowCheckpointConflictError,
    )

    client, preparation, storage, window, token = prepare_consumer(
        package_pipeline, monkeypatch
    )
    install_transport(client, monkeypatch)
    consume_profile_package(
        preparation=preparation,
        coordinator=coordinator(preparation, storage, window, token),
        client=client,
    )
    current = coordinator(preparation, storage, window, token)
    checkpoint = current.checkpoint
    identity = next(iter(checkpoint["profile_attempt_bindings"]))
    checkpoint["profile_attempt_bindings"][identity] = "f" * 64
    parent = storage.get_agent_run(f"window:{window.window_id}")
    frozen = copy.deepcopy(parent.generation_request_snapshot)
    with pytest.raises(OverflowCheckpointConflictError, match="immutable"):
        current.save_checkpoint(checkpoint)
    assert storage.get_agent_run(parent.id).generation_request_snapshot == frozen
