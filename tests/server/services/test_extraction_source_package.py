"""Original-source integrity and provider-free extraction request planning."""

from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.server.services.extraction.compression import plan_extraction_window
from reflexio.server.services.extraction.input_preparation import (
    collect_extraction_sources,
    resolve_evidence_span,
    resolve_visible_quote,
)
from reflexio.server.services.extraction.source_package import (
    evidence_from_source_span,
    render_source_package,
    source_package_presentation,
)
from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionError,
    CompressionPoint,
    EvidencePacket,
    packet_tokens,
)
from tests.server.services.test_input_compression_adapters import client, group

pytestmark = pytest.mark.unit


def original_sources():
    original = group()
    original.interactions[
        0
    ].content = "Only staging is approved. Production needs review."
    return collect_extraction_sources(
        [original],
        user_id="alice",
        include_timestamps=True,
        include_request_source=True,
    )


def packet_for(sources, ranges):
    evidence = [
        evidence_from_source_span(resolve_evidence_span(ref, sources.spans))
        for ref in ranges
    ]
    return EvidencePacket(
        evidence=evidence,
        points=[
            CompressionPoint(
                category="constraint",
                detail="Original evidence.",
                evidence_refs=[s.source_id for s in evidence],
            )
        ],
    )


def test_presentation_freezes_exact_selected_originals_and_attribution():
    sources = original_sources()
    packet = packet_for(sources, ["i1:content:0:12", "i1:content:12:25"])
    presentation = source_package_presentation(packet, sources=sources)
    assert presentation.text == render_source_package(
        packet, source_digest=sources.source_digest
    )
    assert presentation.is_present_in(
        [{"content": [{"type": "text", "text": presentation.text}]}]
    )
    assert "does not establish human authorship" in presentation.text
    assert (
        "".join(
            s.text
            for s in resolve_visible_quote(
                "Only staging is approved.", presentation.visible_spans
            )
        )
        == "Only staging is approved."
    )
    with pytest.raises(ValueError):
        resolve_visible_quote(sources.spans[0].text, presentation.visible_spans)
    frozen = presentation.text
    packet.evidence.clear()
    packet.points.clear()
    assert presentation.text == frozen
    assert len(presentation.visible_spans) == 2
    assert "Only staging" not in repr(presentation)


@pytest.mark.parametrize(
    "change",
    [
        {"speaker": "Assistant"},
        {"request_id": "other"},
        {"text": "Approval!"},
        {"source_order": 99},
        {"request_source": "human"},
        {"start": 1},
    ],
)
def test_presentation_rejects_forged_original_metadata(change):
    sources = original_sources()
    packet = packet_for(sources, ["i1:content:0:12"])
    packet.evidence[0] = packet.evidence[0].model_copy(update=change)
    with pytest.raises(ValueError, match="original evidence changed"):
        source_package_presentation(packet, sources=sources)


@pytest.mark.parametrize("failure", ["duplicate", "order", "unseen"])
def test_presentation_rejects_ambiguous_or_unseen_selection(failure):
    sources = original_sources()
    packet = packet_for(sources, ["i1:content:0:12", "i1:content:24:35"])
    if failure == "duplicate":
        packet.evidence.append(packet.evidence[0])
    elif failure == "order":
        packet.evidence.reverse()
    else:
        packet.points[0].evidence_refs.append("i1:content:35:40")
    with pytest.raises(ValueError):
        source_package_presentation(packet, sources=sources)


def test_presentation_does_not_join_a_gap_in_selected_originals():
    sources = original_sources()
    presentation = source_package_presentation(
        packet_for(sources, ["i1:content:0:12", "i1:content:24:35"]), sources=sources
    )
    with pytest.raises(ValueError):
        resolve_visible_quote(sources.spans[0].text[:35], presentation.visible_spans)


def test_window_plan_is_provider_free_and_binds_original_empty_interactions(
    monkeypatch,
):
    fake = client(monkeypatch)
    original = group()
    original.interactions[0].content = "Production needs separate approval."
    kwargs: dict[str, Any] = {
        "org_id": "org",
        "project_id": "project",
        "user_id": "alice",
        "budget": CompressionBudget(output_tokens=4096),
        "client": fake,
        "context_limits": {fake.config.model: 512000},
    }
    first = plan_extraction_window([original], **kwargs)
    original.interactions.append(
        original.interactions[0].model_copy(update={"interaction_id": 2, "content": ""})
    )
    second = plan_extraction_window([original], **kwargs)
    assert first.source_digest != second.source_digest
    assert first.source_cache_digest == second.source_cache_digest
    assert first.witness != second.witness
    fake.generate_chat_response.assert_not_called()
    guard, load, save = Mock(), Mock(), Mock()
    result = second.execute(before_call=guard, load=load, save=save)
    assert result.packet.evidence[0].text == "Production needs separate approval."
    assert guard.call_count == 2
    load.assert_not_called()
    save.assert_not_called()


@pytest.mark.parametrize("renderer", ["measure", "estimate"])
def test_custom_window_costing_requires_a_new_contract(monkeypatch, renderer):
    fake = client(monkeypatch)
    extra: dict[str, Any] = {f"{renderer}_output": packet_tokens}
    with pytest.raises(CompressionError, match="own output contract"):
        # Distinct function identity is required even for an equivalent measure.
        if renderer == "measure":
            extra["measure_output"] = lambda packet: packet_tokens(packet)
        plan_extraction_window(
            [group()],
            org_id="org",
            project_id="project",
            user_id="alice",
            budget=CompressionBudget(output_tokens=4096),
            client=fake,
            context_limits={},
            **extra,
        )
    fake.generate_chat_response.assert_not_called()


def test_complete_consumer_cost_includes_instructions_and_preserved_content(
    monkeypatch,
):
    fake = client(monkeypatch)
    original = group()
    original.interactions[0].content = "Production needs separate approval."
    measured = []

    def measure(packet):
        measured.append(packet.model_copy(deep=True))
        return 900 + packet_tokens(packet)

    plan = plan_extraction_window(
        [original],
        org_id="org",
        project_id="project",
        user_id="alice",
        budget=CompressionBudget(output_tokens=2048),
        client=fake,
        context_limits={fake.config.model: 512000},
        measure_output=measure,
        estimate_output=measure,
        output_contract="complete-consumer-input-v1",
    )
    result = plan.execute()
    assert result.output_tokens == measure(result.packet) <= 2048
    assert result.output_tokens > packet_tokens(result.packet)
    assert result.packet.evidence[0].text == original.interactions[0].content
    assert measured
    fake.generate_chat_response.assert_not_called()


@pytest.mark.parametrize("image", ["interacted_image_url", "image_encoding"])
def test_text_plan_refuses_image_evidence_before_native_work(monkeypatch, image):
    fake = client(monkeypatch)
    original = group()
    original.interactions[0] = original.interactions[0].model_copy(
        update={image: "image"}
    )
    with pytest.raises(CompressionError, match="image evidence"):
        plan_extraction_window(
            [original],
            org_id="org",
            project_id="project",
            user_id="alice",
            budget=CompressionBudget(output_tokens=2048),
            client=fake,
            context_limits={},
        )
    fake.generate_chat_response.assert_not_called()


def test_complete_native_profile_requests_supply_window_cost(monkeypatch):
    from reflexio.server.llm.input_budget import measure_request_input_tokens
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.profile.profile_generation_service_utils import (
        build_profile_gate_prompt,
        construct_profile_extraction_messages_from_sessions,
    )

    actual = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192))
    transport = Mock(side_effect=AssertionError("Planning must not dispatch"))
    monkeypatch.setattr(actual, "_completion_with_hard_timeout", transport)
    manager = PromptManager()
    original = group()
    original.interactions[0].content = "Production requires separate approval."
    sources = collect_extraction_sources(
        [original],
        user_id="alice",
        include_timestamps=True,
        include_request_source=True,
    )
    counts = []

    def measure(packet):
        text = render_source_package(packet, source_digest=sources.source_digest)
        gate = build_profile_gate_prompt(
            prompt_manager=manager,
            agent_context_prompt="Agent",
            extraction_definition_prompt="Stable preferences",
            condition_prompt="Corrections",
            source_text=text,
        )
        messages = construct_profile_extraction_messages_from_sessions(
            prompt_manager=manager,
            request_interaction_data_models=[original],
            existing_profiles=[],
            agent_context_prompt="Agent",
            context_prompt="Context",
            extraction_definition_prompt="Stable preferences",
            source_text=text,
        )
        requests = [[{"role": "user", "content": gate}], messages]
        current = [
            measure_request_input_tokens(
                actual._build_completion_params(messages=m, max_tokens=8192)[0]
            )
            for m in requests
        ]
        counts.append(current)
        return max(current)

    plan = plan_extraction_window(
        [original],
        org_id="org",
        project_id="project",
        user_id="alice",
        budget=CompressionBudget(output_tokens=32768),
        client=actual,
        context_limits={actual.config.model: 512000},
        measure_output=measure,
        output_contract="test-complete-native-profile-prompts-v1",
    )
    result = plan.execute()
    assert result.output_tokens == measure(result.packet) <= 32768
    assert result.calls == 0 and len(counts[-1]) == 2
    # These are the real initial templates; runtime must additionally freeze and
    # price its tools/schema/prior answers and guard each actual continuation.
    transport.assert_not_called()
