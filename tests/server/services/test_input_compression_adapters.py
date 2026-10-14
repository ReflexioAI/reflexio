"""Scoped window preparation and real model-wrapper dispatch contracts."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.common import ToolUsed
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.llm import ProviderRequestGuardError
from reflexio.server.llm.token_accounting import RunTokenCapture, run_token_capture
from reflexio.server.services.extraction.compression import compress_extraction_window
from reflexio.server.services.input_compression import (
    CompressionBudget,
    EvidenceSource,
)
from reflexio.server.services.input_compression_llm import compress_with_client


def client(monkeypatch, callback=None) -> Any:
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", Mock(return_value=100)
    )
    fake = SimpleNamespace(
        config=SimpleNamespace(model="minimax/MiniMax-M3", max_tokens=8192)
    )
    fake._build_completion_params = lambda **kwargs: (
        {
            "model": fake.config.model,
            "max_tokens": kwargs["max_tokens"],
        },
        None,
        False,
        0,
        [],
    )

    def generate(messages, **kwargs):
        schema = kwargs["response_format"].model_json_schema()
        assert set(schema["properties"]) == {"points"}
        assert set(schema["$defs"]["_ReferenceGroup"]["properties"]) == {
            "evidence_refs",
        }
        params = {
            "model": fake.config.model,
            "messages": messages,
            "max_tokens": kwargs["max_tokens"],
            "extra_body": kwargs["extra_body"],
        }
        kwargs["provider_request_guard"](params)
        if callback:
            callback(params, kwargs)
        packet = json.loads(messages[1]["content"])["input_packets"][0]
        selected = [
            s["source_id"]
            for s in packet["evidence"]
            if "approval" in s.get("text", "")
        ]
        value = kwargs["response_format"].model_validate(
            {"points": [{"evidence_refs": selected[:1]}] if selected else []}
        )
        assert not kwargs["structured_output_validator"](value)
        capture = run_token_capture.get()
        assert capture is not None
        capture.observe(prompt_tokens=100, completion_tokens=20)
        return value

    fake.generate_chat_response = Mock(side_effect=generate)
    return fake


def group(user_id="alice"):
    return RequestInteractionDataModel(
        session_id="session",
        request=Request(
            request_id="request",
            user_id=user_id,
            session_id="session",
            source="coding_agent",
        ),
        interactions=[
            Interaction(
                interaction_id=1,
                request_id="request",
                user_id=user_id,
                role="User",
                content="Staging requires approval; production needs separate approval. "
                + "log " * 3000,
            )
        ],
    )


def test_large_rendered_packet_never_raises_configured_provider_output_cap(monkeypatch):
    from reflexio.server.services.input_compression import packet_tokens

    dispatched_caps = []
    fake = client(
        monkeypatch,
        callback=lambda params, _kwargs: dispatched_caps.append(params["max_tokens"]),
    )
    result = compress_with_client(
        [
            EvidenceSource(
                source_id="source",
                request_id="request",
                interaction_id=1,
                field="content",
                speaker="User",
                source_order=1,
                text="approval " + "log " * 3000,
            )
        ],
        scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
        policy="extraction",
        budget=CompressionBudget(output_tokens=32768),
        client=fake,
        context_limits={fake.config.model: 512000},
        measure_output=lambda packet: packet_tokens(packet) * 16,
    )
    assert result.provider_attempts and dispatched_caps
    assert set(dispatched_caps) == {8192}
    assert 8192 < result.output_tokens <= 32768


def test_window_adapter_keeps_original_and_parent_billing_capture(monkeypatch):
    fake = client(monkeypatch)
    original = group()
    witness = original.model_dump_json()
    parent = RunTokenCapture()
    parent.observe(prompt_tokens=7, completion_tokens=3)
    token = run_token_capture.set(parent)
    try:
        result = compress_extraction_window(
            [original],
            org_id="org",
            project_id="project",
            user_id="alice",
            budget=CompressionBudget(output_tokens=1024),
            client=fake,
            context_limits={fake.config.model: 512000},
        )
        assert run_token_capture.get() is parent
        assert result.provider_attempts == result.calls > 0
        assert result.provider_input_tokens == result.calls * 100
        assert result.provider_output_tokens == result.calls * 20
        assert result.provider_input_tokens is not None
        assert parent.totals.prompt_tokens == 7 + result.provider_input_tokens
        assert "production needs separate approval" in result.packet.render()
        from reflexio.server.services.extraction.input_preparation import (
            collect_extraction_sources,
            resolve_evidence_span,
        )

        approved = collect_extraction_sources(
            [original],
            user_id="alice",
            include_timestamps=True,
            include_request_source=True,
        )
        assert result.source_digest == approved.source_digest
        for evidence in result.packet.evidence:
            assert (
                resolve_evidence_span(evidence.source_id, approved.spans).text
                == evidence.text
            )
            assert evidence.created_at == original.interactions[0].created_at
            assert evidence.request_created_at == original.request.created_at
            assert evidence.session_id == original.session_id
        assert original.model_dump_json() == witness
    finally:
        run_token_capture.reset(token)


def test_window_witness_includes_empty_interactions_without_provider(monkeypatch):
    from reflexio.server.services.extraction.input_preparation import (
        collect_extraction_sources,
    )

    fake = client(monkeypatch)
    original = group()
    original.interactions[0].content = "Separate production approval."
    kwargs: dict[str, Any] = {
        "org_id": "org",
        "project_id": "project",
        "user_id": "alice",
        "budget": CompressionBudget(output_tokens=4096),
        "client": fake,
        "context_limits": {fake.config.model: 512000},
    }
    before = compress_extraction_window([original], **kwargs)
    original.interactions.append(
        original.interactions[0].model_copy(update={"interaction_id": 2, "content": ""})
    )
    after = compress_extraction_window([original], **kwargs)
    assert before.packet == after.packet
    assert before.source_digest != after.source_digest
    assert (
        after.source_digest
        == collect_extraction_sources(
            [original],
            user_id="alice",
            include_timestamps=True,
            include_request_source=True,
        ).source_digest
    )
    fake.generate_chat_response.assert_not_called()


def test_window_adapter_rejects_mixed_owner_before_provider(monkeypatch):
    fake = client(monkeypatch)
    with pytest.raises(ValueError, match="owner mismatch"):
        compress_extraction_window(
            [group("bob")],
            org_id="org",
            project_id="project",
            user_id="alice",
            budget=CompressionBudget(output_tokens=1024),
            client=fake,
            context_limits={},
        )
    fake.generate_chat_response.assert_not_called()


def test_short_conversation_survives_tool_only_selection_and_binds_cache(monkeypatch):
    from reflexio.server.services.input_compression import packet_tokens

    fake = client(monkeypatch)
    original = group()
    original.interactions[0].content = "Production requires separate approval."
    original.interactions[0].tools_used = [
        ToolUsed(tool_name="diagnostic", tool_data={"output": "routine log " * 3000})
    ]
    original.interactions.append(
        original.interactions[0].model_copy(
            update={
                "interaction_id": 2,
                "role": "Assistant",
                "content": "Deployment is still unverified.",
                "tools_used": [],
            }
        )
    )
    cache = {}
    kwargs: dict[str, Any] = {
        "org_id": "org",
        "project_id": "project",
        "user_id": "alice",
        "budget": CompressionBudget(output_tokens=2048),
        "client": fake,
        "context_limits": {fake.config.model: 512000},
        "load": cache.get,
        "save": cache.__setitem__,
    }
    result = compress_extraction_window([original], **kwargs)
    # The fake selects no tool evidence: original conversation is independent
    # of model ranking and still counts against the real final output budget.
    assert result.calls > 0
    assert [source.text for source in result.packet.evidence] == [
        interaction.content for interaction in original.interactions
    ]
    assert packet_tokens(result.packet) == result.output_tokens <= 2048
    warm = compress_extraction_window([original], **kwargs)
    assert warm.calls == 0 and warm.packet == result.packet
    original.interactions[0].content = "Production approval was revoked."
    changed = compress_extraction_window([original], **kwargs)
    assert changed.calls > 0
    assert changed.packet.evidence[0].text == "Production approval was revoked."


def test_every_repair_is_guarded_and_counts_toward_provider_attempt_limit(monkeypatch):
    before = Mock()

    def repair(params, kwargs):
        kwargs["provider_request_guard"](params)
        with pytest.raises(ProviderRequestGuardError, match="attempt budget"):
            kwargs["provider_request_guard"](params)

    fake = client(monkeypatch, callback=repair)
    result = compress_with_client(
        [
            EvidenceSource(
                source_id="source",
                request_id="request",
                interaction_id=1,
                field="content",
                speaker="User",
                source_order=0,
                text="approval " + "log " * 1500,
            )
        ],
        scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
        policy="extraction",
        budget=CompressionBudget(output_tokens=1024, max_calls=2),
        client=fake,
        context_limits={fake.config.model: 512000},
        before_call=before,
    )
    assert result.provider_attempts == before.call_count == 2
    assert result.provider_input_tokens is None  # First repair has no observation.
    assert result.provider_output_tokens is None
    assert run_token_capture.get() is None


def test_model_and_thinking_changes_rejected_before_dispatch(monkeypatch):
    def mutate(params, kwargs):
        for altered in (
            {**params, "model": "different-model"},
            {**params, "extra_body": {"thinking": {"type": "enabled"}}},
        ):
            with pytest.raises(ProviderRequestGuardError, match="changed"):
                kwargs["provider_request_guard"](altered)

    fake = client(monkeypatch, callback=mutate)
    compress_extraction_window(
        [group()],
        org_id="org",
        project_id="project",
        user_id="alice",
        budget=CompressionBudget(output_tokens=1024),
        client=fake,
        context_limits={fake.config.model: 512000},
    )


def test_cache_identity_tracks_actual_seed_sampling_and_endpoint(monkeypatch):
    fake = client(monkeypatch)
    state = {"top_p": 0.9, "seed": 42, "api_base": "https://endpoint.example/v1"}
    fake._build_completion_params = lambda **kwargs: (
        {
            "model": fake.config.model,
            "max_tokens": kwargs["max_tokens"],
            **state,
        },
        None,
        False,
        0,
        [],
    )
    # This test validates cache lookup identity; no provider requests are needed.
    seen = []

    def load(key):
        seen.append(key)
        return {"overview": "", "points": [], "unresolved": [], "omissions": []}

    kwargs = {
        "org_id": "org",
        "project_id": "project",
        "user_id": "alice",
        "budget": CompressionBudget(output_tokens=1024),
        "client": fake,
        "context_limits": {},
        "load": load,
    }
    original = group()
    compress_extraction_window([original], **kwargs)
    first = tuple(seen)
    seen.clear()
    compress_extraction_window([original], **kwargs)
    assert tuple(seen) == first
    for name, value in (
        ("top_p", 0.5),
        ("seed", 17),
        ("api_base", "https://other.example/v1"),
    ):
        state[name] = value
        seen.clear()
        compress_extraction_window([original], **kwargs)
        assert tuple(seen) != first
    fake.generate_chat_response.assert_not_called()


def test_real_litellm_repair_dispatch_preserves_guards_and_parent_accounting(
    monkeypatch,
):
    import litellm

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    live_client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    requests = []

    def transport(params, _hard_timeout):
        requests.append(params)
        body = next(
            json.JSONDecoder().raw_decode(message["content"])[0]
            for message in params["messages"]
            if isinstance(message.get("content"), str)
            and message["content"].startswith("{")
            and "input_packets" in json.JSONDecoder().raw_decode(message["content"])[0]
        )
        ref = body["input_packets"][0]["evidence"][0]["source_id"]
        choice = {
            "points": [
                {
                    "evidence_refs": ["unseen" if len(requests) == 1 else ref],
                }
            ],
        }
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": json.dumps(choice)},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(live_client, "_completion_with_hard_timeout", transport)
    result = compress_extraction_window(
        [group()],
        org_id="org",
        project_id="project",
        user_id="alice",
        budget=CompressionBudget(output_tokens=1024),
        client=live_client,
        context_limits={live_client.config.model: 512000},
    )
    assert len(requests) == result.provider_attempts == 2
    assert result.provider_input_tokens == 200
    assert result.provider_output_tokens == 40
    assert result.packet.overview == "" and result.packet.unresolved == []
    assert all(point.detail == "Original evidence." for point in result.packet.points)
    assert all(request["model"] == live_client.config.model for request in requests)


@pytest.mark.parametrize("repair_fits", [True, False])
def test_capacity_feedback_repairs_or_keeps_explicit_whole_point_partial_result(
    monkeypatch, repair_fits
):
    import litellm

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    live_client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    requests = []

    def transport(params, _hard_timeout):
        requests.append(params)
        body = next(
            json.loads(message["content"])
            for message in params["messages"]
            if isinstance(message.get("content"), str)
            and message["content"].startswith('{"allowed_evidence_refs"')
        )
        refs = body["allowed_evidence_refs"]
        if repair_fits and len(requests) == 2:
            refs = refs[:1]
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {"points": [{"evidence_refs": [ref]} for ref in refs]}
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(live_client, "_completion_with_hard_timeout", transport)
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", Mock(return_value=100)
    )
    sources = [
        EvidenceSource(
            source_id=f"source-{index}",
            request_id="request",
            interaction_id=index,
            field="content",
            speaker="User",
            source_order=index,
            text=f"Constraint {index} approval " * 300,
        )
        for index in (1, 2)
    ]
    result = compress_with_client(
        sources,
        scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
        policy="offline_tuning",
        budget=CompressionBudget(output_tokens=1024),
        client=live_client,
        context_limits={live_client.config.model: 512000},
    )
    assert result.sections == 1
    assert len(requests) == result.provider_attempts == 2
    assert result.provider_reference_recoveries == 0
    assert result.packet.points and result.output_tokens <= 1024
    assert bool(result.packet.omission_counts.capacity_points) is not repair_fits
    assert result.provider_input_tokens == 200 and result.provider_output_tokens == 40


def test_capacity_repair_preserves_all_constraints_without_neighboring_routine_logs(
    monkeypatch,
):
    import litellm

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    live_client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    requests = []

    def transport(params, _hard_timeout):
        requests.append(params)
        body = next(
            json.loads(message["content"])
            for message in params["messages"]
            if isinstance(message.get("content"), str)
            and message["content"].startswith('{"allowed_evidence_refs"')
        )
        records = body["input_packets"][0]["evidence"]
        groups = []
        for index in range(0, len(records), 2):
            refs = [records[index]["source_id"]]
            if len(requests) == 1:
                refs.append(records[index + 1]["source_id"])
            groups.append({"evidence_refs": refs})
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"points": groups}),
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(live_client, "_completion_with_hard_timeout", transport)
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", Mock(return_value=100)
    )
    sources = []
    for index in range(1, 11):
        for field, text in (
            (
                "content",
                f"Target service_{index} needs owner_{index} approval, never another owner's approval.",
            ),
            ("tools_used[0]", "routine " * 400),
        ):
            sources.append(
                EvidenceSource(
                    source_id=f"{index}:{field}",
                    request_id=f"request-{index}",
                    interaction_id=index,
                    field=field,
                    speaker="User" if field == "content" else "Assistant",
                    source_order=index,
                    text=text,
                )
            )
    result = compress_with_client(
        sources,
        scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
        policy="offline_tuning",
        budget=CompressionBudget(output_tokens=4096),
        client=live_client,
        context_limits={live_client.config.model: 512000},
    )
    assert result.sections == result.calls == 1
    assert result.provider_attempts == len(requests) == 2
    assert result.packet.omission_counts.capacity_points == 0
    assert len(result.packet.evidence) == 10
    assert [source.text for source in result.packet.evidence] == [
        source.text for source in sources if source.field == "content"
    ]
    assert result.output_tokens <= 4096


def test_bounded_repair_discards_only_whole_unverifiable_reference_groups(monkeypatch):
    import litellm

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    live_client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    requests = []

    def transport(params, _hard_timeout):
        requests.append(params)
        body = next(
            json.loads(message["content"])
            for message in params["messages"]
            if isinstance(message.get("content"), str)
            and message["content"].startswith('{"allowed_evidence_refs"')
        )
        ref = body["allowed_evidence_refs"][0]
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "points": [
                                    {"evidence_refs": [ref]},
                                    # The valid part of a rejected group must NOT be salvaged:
                                    # it may omit an unseen qualification from that same group.
                                    {"evidence_refs": [ref, "e999999"]},
                                ]
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(live_client, "_completion_with_hard_timeout", transport)
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", Mock(return_value=100)
    )
    original = group()
    cache = {}
    result = compress_extraction_window(
        [original],
        org_id="org",
        project_id="project",
        user_id="alice",
        budget=CompressionBudget(output_tokens=2048),
        client=live_client,
        context_limits={live_client.config.model: 512000},
        load=cache.get,
        save=cache.__setitem__,
    )
    assert len(requests) == result.provider_attempts == 2
    assert result.provider_reference_recoveries == 1
    assert result.packet.omission_counts.unseen_reference_groups == 1
    assert len(result.packet.points) == len(result.packet.evidence) == 1
    assert result.packet.evidence[0].text in original.interactions[0].content
    assert any(
        "discarded 1 entire groups" in notice for notice in result.packet.omissions
    )
    assert result.provider_input_tokens == 200 and result.provider_output_tokens == 40
    replay = compress_extraction_window(
        [original],
        org_id="org",
        project_id="project",
        user_id="alice",
        budget=CompressionBudget(output_tokens=2048),
        client=live_client,
        context_limits={live_client.config.model: 512000},
        load=cache.get,
        save=cache.__setitem__,
    )
    assert replay.packet == result.packet
    assert replay.packet.omission_counts.unseen_reference_groups == 1
    assert replay.provider_attempts == replay.provider_reference_recoveries == 0
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("failure_kind", "error_model", "integrity_fault"),
    [
        ("parse", "minimax/MiniMax-M3", False),
        ("refusal", "minimax/MiniMax-M3", False),
        ("semantic", "another-model", False),
        ("semantic", "minimax/MiniMax-M3", True),
    ],
)
def test_reference_recovery_never_accepts_parse_refusal_route_or_integrity_failure(
    monkeypatch, failure_kind, error_model, integrity_fault
):
    from reflexio.server.llm import StructuredOutputRepairError
    from reflexio.server.services.input_compression import (
        CompressionError,
        packet_tokens,
    )

    fake = client(monkeypatch)
    dispatched = False

    def fail(messages, **_kwargs):
        nonlocal dispatched
        dispatched = True
        ref = json.loads(messages[1]["content"])["allowed_evidence_refs"][0]
        raise StructuredOutputRepairError(
            "Rejected output",
            failure_kind=failure_kind,
            model=error_model,
            raw_content=json.dumps(
                {
                    "points": [
                        {"evidence_refs": [ref]},
                        {"evidence_refs": [ref, "e999999"]},
                    ]
                }
            ),
        )

    def measure(packet):
        if integrity_fault and dispatched and packet.evidence:
            raise CompressionError("Original source changed")
        return packet_tokens(packet)

    fake.generate_chat_response = Mock(side_effect=fail)
    parent = run_token_capture.get()
    with pytest.raises(StructuredOutputRepairError):
        compress_with_client(
            [
                EvidenceSource(
                    source_id="source",
                    request_id="request",
                    interaction_id=1,
                    field="content",
                    speaker="User",
                    source_order=0,
                    text="approval " + "log " * 1500,
                )
            ],
            scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
            policy="extraction",
            budget=CompressionBudget(output_tokens=1024),
            client=fake,
            context_limits={fake.config.model: 512000},
            measure_output=measure,
        )
    assert dispatched and run_token_capture.get() is parent


def test_compression_usage_excludes_sibling_usage_but_parent_bills_both(monkeypatch):
    parent = RunTokenCapture()

    def sibling(_params, _kwargs):
        # A sibling context copied the parent before this operation began.
        parent.observe(prompt_tokens=700, completion_tokens=30)

    fake = client(monkeypatch, callback=sibling)
    token = run_token_capture.set(parent)
    try:
        result = compress_extraction_window(
            [group()],
            org_id="org",
            project_id="project",
            user_id="alice",
            budget=CompressionBudget(output_tokens=1024),
            client=fake,
            context_limits={fake.config.model: 512000},
        )
        assert result.provider_input_tokens == result.calls * 100
        assert result.provider_output_tokens == result.calls * 20
        assert parent.totals.prompt_tokens == result.calls * 800
        assert parent.totals.completion_tokens == result.calls * 50
        assert run_token_capture.get() is parent
    finally:
        run_token_capture.reset(token)


def test_partial_repair_usage_stays_unknown_without_changing_parent_billing(
    monkeypatch,
):
    def partial_response(params, kwargs):
        kwargs["provider_request_guard"](params)
        capture = run_token_capture.get()
        assert capture is not None
        capture.observe(prompt_tokens=100, completion_tokens=None)

    fake = client(monkeypatch, callback=partial_response)
    parent = RunTokenCapture()
    token = run_token_capture.set(parent)
    try:
        result = compress_extraction_window(
            [group()],
            org_id="org",
            project_id="project",
            user_id="alice",
            budget=CompressionBudget(output_tokens=1024),
            client=fake,
            context_limits={fake.config.model: 512000},
        )
        assert result.provider_attempts == 2 * result.calls
        assert result.provider_input_tokens == result.calls * 200
        assert result.provider_output_tokens is None
        assert parent.totals.prompt_tokens == result.calls * 200
        assert parent.totals.completion_tokens == result.calls * 20
    finally:
        run_token_capture.reset(token)


def test_passthrough_rechecks_authority_and_reports_known_zero_usage(monkeypatch):
    fake = client(monkeypatch)
    before = Mock()
    result = compress_extraction_window(
        [group()],
        org_id="org",
        project_id="project",
        user_id="alice",
        budget=CompressionBudget(output_tokens=16384),
        client=fake,
        context_limits={},
        before_call=before,
    )
    assert result.passthrough and result.provider_attempts == 0
    assert result.provider_input_tokens == result.provider_output_tokens == 0
    assert before.call_count == 2
    fake.generate_chat_response.assert_not_called()


def test_cached_result_cannot_escape_a_revoked_lease(monkeypatch):
    fake = client(monkeypatch)
    before = Mock(side_effect=[None, RuntimeError("Lease was revoked")])
    with pytest.raises(RuntimeError, match="revoked"):
        compress_extraction_window(
            [group()],
            org_id="org",
            project_id="project",
            user_id="alice",
            budget=CompressionBudget(output_tokens=1024),
            client=fake,
            context_limits={},
            before_call=before,
            load=lambda _key: {
                "overview": "",
                "points": [],
                "unresolved": [],
                "omissions": [],
            },
        )
    fake.generate_chat_response.assert_not_called()
