"""Scoped window preparation and real model-wrapper dispatch contracts."""

import json
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.common import ToolUsed
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.llm import ProviderRequestGuardError
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.token_accounting import RunTokenCapture, run_token_capture
from reflexio.server.services.extraction.compression import compress_extraction_window
from reflexio.server.services.input_compression import (
    CAPACITY_OMISSION,
    CompressionBudget,
    EvidenceSource,
)
from reflexio.server.services.input_compression_llm import (
    compress_with_client,
    plan_with_client,
)


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
            "messages": kwargs["messages"],
            **({"extra_body": kwargs["extra_body"]} if "extra_body" in kwargs else {}),
        },
        None,
        False,
        0,
        [],
    )

    fake.get_completion_request_contract = lambda **kwargs: (
        LiteLLMClient._completion_request_contract_from_params(
            fake._build_completion_params(messages=[], **kwargs)[0]
        )
    )
    fake.get_completion_model_ladder = lambda **_kwargs: (fake.config.model,)

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


@pytest.mark.unit
def test_every_repair_is_guarded_and_counts_toward_provider_attempt_limit(
    monkeypatch, caplog
):
    before = Mock()

    def repair(params, kwargs):
        kwargs["provider_request_guard"](params)
        with pytest.raises(
            ProviderRequestGuardError, match="attempt budget"
        ) as failure:
            kwargs["provider_request_guard"](params)
        assert type(failure.value) is ProviderRequestGuardError

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
                text="approval PRIVATE_SOURCE_SENTINEL " + "log " * 1500,
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
    assert caplog.messages == [
        "event=compression_provider_attempt_budget_exhausted "
        "attempts_used=2 max_calls=2 sections_planned=1"
    ]
    assert caplog.records[0].args == (2, 2, 1)


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
            "messages": kwargs["messages"],
            **({"extra_body": kwargs["extra_body"]} if "extra_body" in kwargs else {}),
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
        "context_limits": {
            "minimax/MiniMax-M3@https://endpoint.example/v1": 512000,
            "minimax/MiniMax-M3@https://other.example/v1": 512000,
        },
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
        # Parse failures keep the client's bounded repair turn.
        content = (
            "{"
            if len(requests) == 1
            else json.dumps({"points": [{"evidence_refs": [ref]}]})
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


def _live_client(monkeypatch, reply):
    """A real LiteLLMClient whose transport answers ``reply(body, requests)``."""
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
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(reply(body, requests)),
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
    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
    return live_client, requests, sink


def _failures(sink) -> int:
    return len([c for c in sink.record.call_args_list if c.args[0] == "llm.failures"])


def _salvage_events(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING"
        and r.getMessage().startswith("event=compression_selection_salvaged")
    ]


def test_capacity_violation_keeps_explicit_whole_point_partial_result_without_repair(
    monkeypatch, caplog
):
    live_client, requests, sink = _live_client(
        monkeypatch,
        lambda body, _requests: {
            "points": [
                {"evidence_refs": [ref]} for ref in body["allowed_evidence_refs"]
            ]
        },
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
    assert len(requests) == result.provider_attempts == 1
    assert result.provider_reference_recoveries == 0
    assert result.packet.points and result.output_tokens <= 1024
    assert result.packet.omission_counts.capacity_points > 0
    assert CAPACITY_OMISSION in result.packet.omissions
    assert result.provider_input_tokens == 100 and result.provider_output_tokens == 20
    assert _salvage_events(caplog) == [
        "event=compression_selection_salvaged reason=capacity fallback=none "
        f"kept_points={len(result.packet.points)} discarded_groups=0"
    ]
    assert not _failures(sink)


def test_capacity_violation_never_severs_a_group_without_repair(monkeypatch):
    def reply(body, _requests):
        records = body["input_packets"][0]["evidence"]
        return {
            "points": [
                {
                    "evidence_refs": [
                        records[index]["source_id"],
                        records[index + 1]["source_id"],
                    ]
                }
                for index in range(0, len(records), 2)
            ]
        }

    live_client, requests, _sink = _live_client(monkeypatch, reply)
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
    assert result.provider_attempts == len(requests) == 1
    assert result.packet.omission_counts.capacity_points > 0
    assert result.packet.points
    # Whole points only: every kept constraint keeps its paired passage.
    assert all(len(point.evidence_refs) == 2 for point in result.packet.points)
    assert len(result.packet.evidence) == 2 * len(result.packet.points)
    originals = {source.source_id: source.text for source in sources}
    assert all(
        item.text in originals[item.source_id.rsplit(":", 2)[0]]
        for item in result.packet.evidence
    )
    assert result.output_tokens <= 4096


def test_selection_with_only_unseen_refs_falls_back_to_recent_original_evidence(
    monkeypatch, caplog
):
    live_client, requests, sink = _live_client(
        monkeypatch,
        lambda _body, _requests: {"points": [{"evidence_refs": ["e999999"]}]},
    )
    sources = [
        EvidenceSource(
            source_id=f"source-{index}",
            request_id="request",
            interaction_id=index,
            field="content",
            speaker="User",
            source_order=index,
            text=f"Turn {index} approval " * 150,
        )
        for index in range(1, 9)
    ]
    cache = {}
    kwargs = {
        "scope": {"org_id": "org", "project_id": "project", "user_id": "alice"},
        "policy": "extraction",
        "budget": CompressionBudget(output_tokens=1024),
        "client": live_client,
        "context_limits": {live_client.config.model: 512000},
        "load": cache.get,
        "save": cache.__setitem__,
    }
    result = compress_with_client(sources, **kwargs)
    assert len(requests) == result.provider_attempts == 1
    assert result.provider_reference_recoveries == 1
    assert result.output_tokens <= 1024
    kept = list(
        dict.fromkeys(
            item.source_id.rsplit(":", 2)[0] for item in result.packet.evidence
        )
    )
    # The newest passages that fit, in chronological order, copied verbatim.
    assert kept and kept == [f"source-{i}" for i in range(9 - len(kept), 9)]
    assert len(kept) < len(sources)
    originals = {source.source_id: source.text for source in sources}
    assert all(
        item.text in originals[item.source_id.rsplit(":", 2)[0]]
        for item in result.packet.evidence
    )
    assert all(
        point.category == "selected_evidence" and point.detail == "Original evidence."
        for point in result.packet.points
    )
    assert result.packet.omission_counts.unseen_reference_groups == 1
    assert result.packet.omission_counts.capacity_points > 0
    assert any(
        re.fullmatch(
            r"Host selected the (\d+) most recent of \1 visible passages because "
            r"the provider selection was unusable; source selection is incomplete\.",
            note,
        )
        for note in result.packet.omissions
    )
    assert any("discarded 1 entire groups" in n for n in result.packet.omissions)
    assert _salvage_events(caplog) == [
        "event=compression_selection_salvaged reason=unseen_refs "
        f"fallback=recent_evidence kept_points={len(result.packet.points)} "
        "discarded_groups=1"
    ]
    # Evidence survived, so this is not a terminal failure.
    assert not _failures(sink)
    replay = compress_with_client(sources, **kwargs)
    assert replay.packet == result.packet and replay.provider_attempts == 0


def test_recent_evidence_fallback_leads_with_each_required_speaker(monkeypatch):
    live_client, requests, _sink = _live_client(
        monkeypatch, lambda _body, _requests: {"points": []}
    )
    sources = [
        EvidenceSource(
            source_id="assistant-early",
            request_id="request",
            interaction_id=1,
            field="content",
            speaker="Assistant",
            source_order=1,
            text="Assistant approval note " * 150,
        ),
        *(
            EvidenceSource(
                source_id=f"user-{index}",
                request_id="request",
                interaction_id=index,
                field="content",
                speaker="User",
                source_order=index,
                text=f"User turn {index} approval " * 150,
            )
            for index in range(2, 9)
        ),
    ]
    result = compress_with_client(
        sources,
        scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
        policy="extraction",
        budget=CompressionBudget(output_tokens=1024),
        client=live_client,
        context_limits={live_client.config.model: 512000},
        required_speakers=("Assistant",),
    )
    assert len(requests) == 1
    speakers = {item.speaker for item in result.packet.evidence}
    assert "Assistant" in speakers and "User" in speakers
    assert result.packet.evidence[-1].source_id.startswith("user-8")


def test_unusable_selection_with_no_fitting_fallback_is_terminal(monkeypatch):
    from reflexio.server.services.input_compression import (
        CompressionError,
        packet_tokens,
    )

    live_client, requests, sink = _live_client(
        monkeypatch, lambda _body, _requests: {"points": [{"evidence_refs": ["e9999"]}]}
    )

    def measure(packet):
        if requests and packet.evidence:
            raise CompressionError("Original source changed")
        return packet_tokens(packet)

    with pytest.raises(CompressionError, match="Original source changed"):
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
            client=live_client,
            context_limits={live_client.config.model: 512000},
            measure_output=measure,
        )
    assert len(requests) == 1
    assert _failures(sink) == 1


def test_valid_selection_is_accepted_unchanged_without_salvage(monkeypatch, caplog):
    live_client, requests, sink = _live_client(
        monkeypatch,
        lambda body, _requests: {
            "points": [{"evidence_refs": [body["allowed_evidence_refs"][-1]]}]
        },
    )
    result = compress_extraction_window(
        [group()],
        org_id="org",
        project_id="project",
        user_id="alice",
        budget=CompressionBudget(output_tokens=2048),
        client=live_client,
        context_limits={live_client.config.model: 512000},
    )
    assert len(requests) == result.provider_attempts == result.calls
    assert result.provider_reference_recoveries == 0
    assert result.packet.points
    assert not any(
        note.startswith(("Host discarded", "Host selected", CAPACITY_OMISSION))
        for note in result.packet.omissions
    )
    assert result.packet.omission_counts.unseen_reference_groups == 0
    assert not _salvage_events(caplog)
    assert not _failures(sink)


def test_salvage_discards_only_whole_unverifiable_reference_groups_without_repair(
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
    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
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
    assert len(requests) == result.provider_attempts == 1
    assert result.provider_reference_recoveries == 1
    assert result.packet.omission_counts.unseen_reference_groups == 1
    assert len(result.packet.points) == len(result.packet.evidence) == 1
    assert result.packet.evidence[0].text in original.interactions[0].content
    assert any(
        "discarded 1 entire groups" in notice for notice in result.packet.omissions
    )
    assert result.provider_input_tokens == 100 and result.provider_output_tokens == 20
    # A salvaged selection is not a terminal LLM failure, so it must not page.
    assert not [c for c in sink.record.call_args_list if c.args[0] == "llm.failures"]
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
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("failure_kind", "error_model", "integrity_fault"),
    [
        ("parse", "minimax/MiniMax-M3", False),
        ("refusal", "minimax/MiniMax-M3", False),
        ("semantic", "another-model", False),
        ("semantic", "minimax/MiniMax-M3", True),
    ],
)
def test_exhausted_client_repair_is_never_salvaged_and_counts_once(
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
    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
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
    # The client defers repair failures to this caller, which counts the
    # unrecovered one exactly once.
    assert fake.generate_chat_response.call_args.kwargs["caller_reports_repair_failure"]
    failures = [c for c in sink.record.call_args_list if c.args[0] == "llm.failures"]
    assert len(failures) == 1


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


@pytest.mark.unit
@pytest.mark.parametrize("model", ["minimax/MiniMax-M3", "gpt-4o-mini", "gpt-5.4-mini"])
def test_native_preparation_prices_actual_requests_without_dispatch_or_usage(
    monkeypatch,
    model,
):
    import reflexio.server.services.input_compression_llm as native

    actual_client = LiteLLMClient(LiteLLMConfig(model=model, max_tokens=8192))
    transport = Mock(side_effect=AssertionError("No provider in planning"))
    monkeypatch.setattr(actual_client, "_completion_with_hard_timeout", transport)
    checks = []
    check_budget = native.check_request_input_budget

    def check(params, **kwargs):
        checks.append(params)
        return check_budget(params, **kwargs)

    monkeypatch.setattr(native, "check_request_input_budget", check)
    parent = RunTokenCapture()
    token = run_token_capture.set(parent)
    try:
        prepared = plan_with_client(
            [
                EvidenceSource(
                    source_id="source",
                    request_id="request",
                    interaction_id=1,
                    field="content",
                    speaker="User",
                    source_order=1,
                    text="approval required for staging. " + "diagnostic " * 1700,
                )
            ],
            scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
            policy="extraction",
            budget=CompressionBudget(output_tokens=1024, section_tokens=1200),
            client=actual_client,
            context_limits={actual_client.config.model: 512000},
        )
        assert prepared.compression.sections > 1
        assert len(checks) == prepared.compression.sections
        assert all(
            p.get("max_completion_tokens", p.get("max_tokens")) == 8192 for p in checks
        )
        if model == "minimax/MiniMax-M3":
            assert all(
                p["extra_body"]["thinking"] == {"type": "disabled"} for p in checks
            )
        assert all(p["messages"] for p in checks)
        assert run_token_capture.get() is parent and parent.completions == 0
        assert "approval" not in repr(prepared)
    finally:
        run_token_capture.reset(token)
    transport.assert_not_called()


@pytest.mark.unit
def test_native_preparation_overflow_refuses_before_provider_and_cache(monkeypatch):
    from reflexio.server.llm.input_budget import InputBudgetExceededError

    actual_client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192)
    )
    transport = Mock(side_effect=AssertionError("Over-budget input cannot dispatch"))
    monkeypatch.setattr(actual_client, "_completion_with_hard_timeout", transport)
    with pytest.raises(InputBudgetExceededError):
        plan_with_client(
            [
                EvidenceSource(
                    source_id="source",
                    request_id="request",
                    interaction_id=1,
                    field="content",
                    speaker="User",
                    source_order=1,
                    text="diagnostic " * 1700,
                )
            ],
            scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
            policy="extraction",
            budget=CompressionBudget(output_tokens=1024),
            client=actual_client,
            context_limits={actual_client.config.model: 8192},
        )
    transport.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("drift", ["model", "seed", "policy", "prompt"])
def test_native_plan_rejects_contract_drift_before_cached_selection(monkeypatch, drift):
    fake = client(monkeypatch)
    state = {}
    original_builder = fake._build_completion_params

    def builder(**kwargs):
        result = original_builder(**kwargs)
        return ({**result[0], **state}, *result[1:])

    fake._build_completion_params = builder
    prepared = plan_with_client(
        [
            EvidenceSource(
                source_id="source",
                request_id="request",
                interaction_id=1,
                field="content",
                speaker="User",
                source_order=1,
                text="approval " + "diagnostic " * 1700,
            )
        ],
        scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
        policy="extraction",
        budget=CompressionBudget(output_tokens=1024),
        client=fake,
        context_limits={fake.config.model: 512000},
    )
    if drift == "model":
        fake.config.model = "different"
    elif drift == "seed":
        state["seed"] = 99
    elif drift == "policy":
        state["frequency_penalty"] = 0.5
    else:
        # Message augmentation is excluded from the policy descriptor but
        # included in the native leaf witness.
        def augment(**kwargs):
            result = builder(**kwargs)
            return (
                {
                    **result[0],
                    "messages": [
                        *result[0]["messages"],
                        {"role": "system", "content": "changed"},
                    ],
                },
                *result[1:],
            )

        fake._build_completion_params = augment
    before = Mock()
    load = Mock()
    with pytest.raises(ProviderRequestGuardError, match="changed"):
        prepared.execute(before_call=before, load=load)
    before.assert_not_called()
    load.assert_not_called()
    fake.generate_chat_response.assert_not_called()


@pytest.mark.unit
def test_actual_native_repair_overflow_stops_before_second_transport(monkeypatch):
    import litellm

    from reflexio.server.llm.input_budget import InputBudgetExceededError

    actual_client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    requests = []

    def count(*, messages=None, **_kwargs):
        # Initial requests fit. A real parse correction adds its own messages.
        return 40000 if messages and len(messages) > 2 else 100

    monkeypatch.setattr("reflexio.server.llm.input_budget.litellm.token_counter", count)

    def transport(params, _timeout):
        requests.append(params)
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": "not valid JSON"},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(actual_client, "_completion_with_hard_timeout", transport)
    prepared = plan_with_client(
        [
            EvidenceSource(
                source_id="source",
                request_id="request",
                interaction_id=1,
                field="content",
                speaker="User",
                source_order=1,
                text="diagnostic " * 1700,
            )
        ],
        scope={"org_id": "org", "project_id": "project", "user_id": "alice"},
        policy="extraction",
        budget=CompressionBudget(output_tokens=1024),
        client=actual_client,
        context_limits={actual_client.config.model: 512000},
    )
    before = Mock()
    with pytest.raises(InputBudgetExceededError):
        prepared.execute(before_call=before)
    assert len(requests) == before.call_count == 1


@pytest.mark.unit
def test_native_prepared_cache_replay_preserves_keys_usage_and_parent_capture(
    monkeypatch,
):
    fake = client(monkeypatch)
    contexts = {fake.config.model: 512000}
    options = {
        "scope": {"org_id": "org", "project_id": "project", "user_id": "alice"},
        "policy": "extraction",
        "budget": CompressionBudget(output_tokens=1024),
        "client": fake,
        "context_limits": contexts,
    }
    sources = [
        EvidenceSource(
            source_id="source",
            request_id="request",
            interaction_id=1,
            field="content",
            speaker="User",
            source_order=1,
            text="approval " + "diagnostic " * 1700,
        )
    ]
    prepared = plan_with_client(sources, **options)
    contexts.clear()  # A caller's later edit cannot change admitted capacity.
    cache = {}
    before = Mock()
    parent = RunTokenCapture()
    token = run_token_capture.set(parent)
    try:
        result = prepared.execute(before_call=before, save=cache.__setitem__)
        assert result.provider_attempts == before.call_count > 0
        assert parent.completions == result.provider_attempts
        paid_prompt_tokens = parent.totals.prompt_tokens
        fake.generate_chat_response.reset_mock()
        before.reset_mock()
        replay = prepared.execute(before_call=before, load=cache.get)
        assert replay.packet == result.packet
        assert replay.cache_hits == result.calls and replay.calls == 0
        assert replay.provider_attempts == 0
        assert replay.provider_input_tokens == replay.provider_output_tokens == 0
        assert parent.completions == result.provider_attempts
        assert parent.totals.prompt_tokens == paid_prompt_tokens
        assert run_token_capture.get() is parent
        before.assert_not_called()
        fake.generate_chat_response.assert_not_called()
    finally:
        run_token_capture.reset(token)


@pytest.mark.unit
def test_native_selection_contract_and_node_witnesses_survive_warm_replay(monkeypatch):
    fake = client(monkeypatch)
    original = EvidenceSource(
        source_id="source",
        request_id="request",
        interaction_id=1,
        field="content",
        speaker="User",
        source_order=1,
        text="approval " + "log " * 1700,
    )
    options: dict[str, Any] = {
        "scope": {"org_id": "org", "project_id": "project", "user_id": "alice"},
        "policy": "extraction",
        "budget": CompressionBudget(output_tokens=1024),
        "client": fake,
        "context_limits": {fake.config.model: 512000},
    }
    prepared = plan_with_client([original], **options)
    assert len(prepared.request_contract_digest) == 64
    assert prepared.request_contract_digest != prepared.witness
    nodes, cache = [], {}
    cold = prepared.execute(inspect_node=nodes.append, save=cache.__setitem__)
    original_nodes = tuple(nodes)
    assert len(nodes) == cold.calls > 0
    assert all(len(node.input_refs_digest) == 64 for node in nodes)
    nodes.clear()
    warm = prepared.execute(inspect_node=nodes.append, load=cache.get)
    assert warm.packet == cold.packet and warm.provider_attempts == 0
    assert tuple(nodes) == original_nodes
    fake.config.model = "other-model"
    changed_nodes = Mock()
    with pytest.raises(ProviderRequestGuardError, match="changed"):
        prepared.execute(inspect_node=changed_nodes)
    changed_nodes.assert_not_called()


def test_salvage_judges_usability_with_its_own_omission_notice(caplog):
    """A group that fits only without the host notice must not end empty."""
    from reflexio.server.services.input_compression import (
        CompressionCall,
        CompressionChoice,
        CompressionPoint,
        _pack_choice,
        _recent_evidence_choice,
    )
    from reflexio.server.services.input_compression_llm import (
        _ReferenceSelection,
        _salvage_selection,
    )

    def source(source_id, text):
        return EvidenceSource(
            source_id=source_id,
            request_id="r",
            interaction_id=1,
            field="content",
            speaker="User",
            source_order=1,
            text=text,
        )

    available = {"e1": source("s1", "approval " * 150), "e2": source("s2", "short")}

    def measure(packet):
        return len(packet.render())

    exact = CompressionChoice(
        points=[CompressionPoint(category="selected_evidence", evidence_refs=["e1"])]
    )
    budget = measure(_pack_choice(exact, available, 10_000, measure, lambda _p: True))

    def pack(choice):
        return _pack_choice(choice, available, budget, measure, lambda _p: True)

    call = CompressionCall(
        "key",
        [],
        budget,
        pack,
        tuple(available),
        lambda notes, counts: _recent_evidence_choice(
            available, frozenset(), pack, notes, counts
        ),
    )
    choice, discarded = _salvage_selection(
        call,
        _ReferenceSelection.model_validate(
            {"points": [{"evidence_refs": ["e1"]}, {"evidence_refs": ["bad"]}]}
        ),
    )
    packet = pack(choice)
    assert discarded == 1
    assert [item.text for item in packet.evidence] == ["short"]
    assert choice.omission_counts.unseen_reference_groups == 1
    assert _salvage_events(caplog) == [
        "event=compression_selection_salvaged reason=unseen_refs "
        "fallback=recent_evidence kept_points=1 discarded_groups=1"
    ]


def test_fallback_keeps_an_older_required_speaker_passage_that_fits(caplog):
    """The newest required-speaker passage may not fit; an older one must lead."""
    from reflexio.server.services.input_compression import (
        RECENT_EVIDENCE_OMISSION,
        CompressionCall,
        CompressionChoice,
        CompressionPoint,
        OmissionCounts,
        _pack_choice,
        _recent_evidence_choice,
        _require_speakers,
    )
    from reflexio.server.services.input_compression_llm import (
        _ReferenceSelection,
        _salvage_selection,
    )

    def source(source_id, speaker, order, text):
        return EvidenceSource(
            source_id=source_id,
            request_id="r",
            interaction_id=order,
            field="content",
            speaker=speaker,
            source_order=order,
            text=text,
        )

    available = {
        "e1": source("agent-old", "Assistant", 1, "agent note " * 40),
        "e2": source("user-a", "User", 2, "user turn a " * 40),
        "e3": source("user-b", "User", 3, "user turn b " * 40),
        "e4": source("agent-new", "Assistant", 4, "agent dump " * 4000),
    }
    required = frozenset({"Assistant"})

    def measure(packet):
        return len(packet.render())

    notice = RECENT_EVIDENCE_OMISSION.format(offered=4, visible=4)
    two_users = CompressionChoice(
        points=[
            CompressionPoint(category="selected_evidence", evidence_refs=[ref])
            for ref in ("e3", "e2")
        ],
        omissions=[notice],
    )
    budget = measure(
        _pack_choice(two_users, available, 1_000_000, measure, lambda _p: True)
    )

    def pack_points(choice):
        return _pack_choice(choice, available, budget, measure, lambda _p: True)

    def pack(choice):
        packet = pack_points(choice)
        _require_speakers(packet, available, required)
        return packet

    call = CompressionCall(
        "key",
        [],
        budget,
        pack,
        tuple(available),
        lambda notes, counts: _recent_evidence_choice(
            available, required, pack_points, notes, counts
        ),
    )
    # Newest-first alone would admit the two User turns and leave no room.
    assert len(available["e4"].text) > budget
    choice, _ = _salvage_selection(call, _ReferenceSelection(points=[]))
    packet = pack(choice)
    assert "agent-old" in {item.source_id for item in packet.evidence}
    assert choice.omission_counts == OmissionCounts()
    assert _salvage_events(caplog)[0].startswith(
        "event=compression_selection_salvaged reason=rejected fallback=recent_evidence"
    )
