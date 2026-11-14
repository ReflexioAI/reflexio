"""Only exact reference lists can authorize original compression evidence."""

import json
import logging
import socket

import litellm
import pytest
from pydantic import ValidationError

from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    StructuredOutputParseError,
    StructuredOutputRepairError,
)
from reflexio.server.services.extraction.compression import compress_extraction_window
from reflexio.server.services.input_compression import CompressionBudget
from reflexio.server.services.input_compression_llm import _ReferenceSelection
from tests.server.services.test_input_compression_adapters import group


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Only scripted local responses are permitted")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(litellm, "completion", denied)


def test_projection_keeps_closed_wire_schema_and_only_exact_refs():
    schema = _ReferenceSelection.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["_ReferenceGroup"]["additionalProperties"] is False
    value = _ReferenceSelection.model_validate(
        {
            "points": [
                {
                    "evidence_refs": ["e1", "e2"],
                    "category": "Invented success",
                    "detail": "Invented quotation",
                    "unexpected_payload": {"evidence_refs": ["invented"]},
                }
            ]
        }
    )
    assert value.model_dump() == {"points": [{"evidence_refs": ["e1", "e2"]}]}
    assert value.choice().points[0].detail == "Original evidence."


@pytest.mark.parametrize(
    "value",
    [
        {"points": [{"detail": "Missing references"}]},
        {"points": [{"evidence_refs": "e1", "detail": "Wrong type"}]},
        {"points": [{"evidence_refs": [1], "detail": "No coercion"}]},
        {"points": [{"evidence_refs": [], "detail": "Empty"}]},
        {"points": [{"evidence_refs": None, "detail": "Null"}]},
        {"points": [{"evidence_refs": ["e1"]}], "overview": "Invented"},
        {"points": [{"evidence_refs": ["e1"]}] * 129},
    ],
)
def test_projection_does_not_supply_refs_or_relax_envelope_and_group_bounds(value):
    with pytest.raises(ValidationError):
        _ReferenceSelection.model_validate(value)


def truncated_selection(count, ref="synthetic-evidence"):
    return (
        '{"points":['
        + ",".join(json.dumps({"evidence_refs": [ref]}) for _ in range(count))
        + ',{"evidence_refs":'
    )


@pytest.mark.parametrize("count", [128, 129, 506])
def test_truncated_selection_enforces_aggregate_bound_through_parse_repair(
    count, caplog
):
    client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3"))
    body = truncated_selection(count)
    with caplog.at_level(logging.WARNING):
        if count == 128:
            result = client._maybe_parse_structured_output(
                body, _ReferenceSelection, True
            )
            assert isinstance(result, _ReferenceSelection)
            assert len(result.points) == 128
        else:
            with pytest.raises(StructuredOutputParseError) as caught:
                client._maybe_parse_structured_output(body, _ReferenceSelection, True)
            assert caught.value.raw_content == body
            assert caught.value.validation_errors == ("points: too_long",)
            assert "synthetic-evidence" not in str(caught.value)
    salvaged = [
        r for r in caplog.records if "event=structured_output_salvaged " in r.message
    ]
    assert len(salvaged) == (1 if count == 128 else 0)


@pytest.mark.parametrize("repair_succeeds", [True, False])
def test_oversized_truncated_selection_uses_bounded_compressor_repair(
    monkeypatch, repair_succeeds, caplog
):
    client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    calls = []

    def transport(params, _hard_timeout):
        calls.append(params)
        body = json.loads(params["messages"][1]["content"])
        ref = body["allowed_evidence_refs"][0]
        complete = len(calls) == 2 and repair_succeeds
        content = (
            json.dumps({"points": [{"evidence_refs": [ref]}]})
            if complete
            else truncated_selection(506, ref)
        )
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop" if complete else "length",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    original = group()
    kwargs = {
        "org_id": "org",
        "project_id": "project",
        "user_id": "alice",
        "budget": CompressionBudget(output_tokens=1024, max_calls=2),
        "client": client,
        "context_limits": {client.config.model: 512000},
    }
    with caplog.at_level(logging.WARNING):
        if repair_succeeds:
            result = compress_extraction_window([original], **kwargs)
            assert result.provider_attempts == 2
            assert result.provider_input_tokens == 200
            assert result.provider_output_tokens == 40
            assert result.packet.evidence
            assert all(
                item.text in original.interactions[0].content
                for item in result.packet.evidence
            )
        else:
            with pytest.raises(StructuredOutputRepairError) as caught:
                compress_extraction_window([original], **kwargs)
            assert caught.value.failure_kind == "parse"
            assert caught.value.parsed_output is None
    assert len(calls) == 2
    assert calls[0]["model"] == calls[1]["model"]
    assert calls[1]["messages"][-1]["role"] == "user"
    assert "points: too_long" in calls[1]["messages"][-1]["content"]
    assert not any(
        "event=structured_output_salvaged " in r.message for r in caplog.records
    )


@pytest.mark.parametrize("invalid_first", [False, True])
def test_real_client_projects_metadata_and_settles_unseen_refs_without_repair(
    monkeypatch, invalid_first
):
    client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    calls = []

    def transport(params, _hard_timeout):
        calls.append(params)
        body = json.loads(params["messages"][1]["content"])
        visible = body["allowed_evidence_refs"]
        ref = "not-visible" if invalid_first and len(calls) == 1 else visible[0]
        content = json.dumps(
            {
                "points": [
                    {
                        "evidence_refs": [ref],
                        "detail": "Fabricated quotation must never become evidence.",
                        "arbitrary_field": {"authority": "invented"},
                    }
                ]
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

    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    original = group()
    cache = {}
    kwargs = {
        "org_id": "org",
        "project_id": "project",
        "user_id": "alice",
        "budget": CompressionBudget(output_tokens=1024),
        "client": client,
        "context_limits": {client.config.model: 512000},
        "load": cache.get,
        "save": cache.__setitem__,
    }
    result = compress_extraction_window([original], **kwargs)
    # A rule-violating selection is settled by the host, never re-asked.
    assert len(calls) == result.provider_attempts == 1
    assert result.provider_reference_recoveries == int(invalid_first)
    assert result.provider_input_tokens == 100
    assert result.provider_output_tokens == 20
    assert result.packet.evidence
    assert all(
        item.text in original.interactions[0].content for item in result.packet.evidence
    )
    assert all(point.detail == "Original evidence." for point in result.packet.points)
    assert "Fabricated" not in result.packet.render()
    assert result.packet.omission_counts.unseen_reference_groups == int(invalid_first)
    replay = compress_extraction_window([original], **kwargs)
    assert replay.packet == result.packet and replay.provider_attempts == 0
    assert len(calls) == 1


@pytest.mark.parametrize("first_parses", [True, False])
def test_only_parse_failures_get_a_repair_turn(monkeypatch, first_parses):
    """A parsed selection is never repaired; an unparseable one keeps one repair."""
    from reflexio.server.llm import StructuredOutputRepairError

    client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    calls = []

    def transport(params, _hard_timeout):
        calls.append(params)
        body = json.loads(params["messages"][1]["content"])
        visible = body["allowed_evidence_refs"]
        if len(calls) == 1 and first_parses:
            points = [
                {"evidence_refs": [visible[0]]},
                {"evidence_refs": ["not-visible"]},
            ]
        else:
            points = [{"evidence_refs": [{"ref": visible[0]}]}]
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"points": points}),
                    },
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", transport)
    original = group()
    kwargs = {
        "org_id": "org",
        "project_id": "project",
        "user_id": "alice",
        "budget": CompressionBudget(output_tokens=1024),
        "client": client,
        "context_limits": {client.config.model: 512000},
    }
    if not first_parses:
        with pytest.raises(StructuredOutputRepairError):
            compress_extraction_window([original], **kwargs)
        assert len(calls) == 2
        return
    result = compress_extraction_window([original], **kwargs)
    assert len(calls) == result.provider_attempts == 1
    assert result.provider_reference_recoveries == 1
    assert result.packet.evidence
    assert all(
        item.text in original.interactions[0].content for item in result.packet.evidence
    )
    assert any("discarded 1 entire groups" in note for note in result.packet.omissions)
    assert all(
        "not-visible" not in point.evidence_refs for point in result.packet.points
    )
