"""Only exact reference lists can authorize original compression evidence."""

import json
import socket

import litellm
import pytest
from pydantic import ValidationError

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
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


@pytest.mark.parametrize("invalid_first", [False, True])
def test_real_client_projects_metadata_but_still_repairs_unseen_refs_and_replays(
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
    attempts = 2 if invalid_first else 1
    assert len(calls) == result.provider_attempts == attempts
    assert result.provider_input_tokens == attempts * 100
    assert result.provider_output_tokens == attempts * 20
    assert result.packet.evidence
    assert all(
        item.text in original.interactions[0].content for item in result.packet.evidence
    )
    assert all(point.detail == "Original evidence." for point in result.packet.points)
    assert "Fabricated" not in result.packet.render()
    replay = compress_extraction_window([original], **kwargs)
    assert replay.packet == result.packet and replay.provider_attempts == 0
    assert len(calls) == attempts


@pytest.mark.parametrize(
    ("prior_finish", "final_finish", "prior_parses", "recover"),
    [
        ("stop", "stop", True, True),
        ("length", "stop", True, False),
        ("stop", "length", True, False),
        (None, "stop", True, False),
        ("stop", None, True, False),
        ("stop", "stop", False, False),
    ],
)
def test_complete_prior_selection_survives_malformed_final_repair_only_with_stop_metadata(
    monkeypatch, prior_finish, final_finish, prior_parses, recover
):
    from reflexio.server.llm import StructuredOutputRepairError

    client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_tokens=8192, fallback_models=[])
    )
    calls = []

    def transport(params, _hard_timeout):
        calls.append(params)
        body = json.loads(params["messages"][1]["content"])
        visible = body["allowed_evidence_refs"]
        if len(calls) == 1 and prior_parses:
            points = [
                {"evidence_refs": [visible[0]]},
                {"evidence_refs": ["not-visible"]},
            ]
        else:
            points = [{"evidence_refs": [{"ref": visible[0]}]}]
        finish = prior_finish if len(calls) == 1 else final_finish
        response = litellm.ModelResponse(
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"points": points}),
                    },
                    "finish_reason": finish,
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )
        # The SDK constructor defaults None to stop; exercise actual unknown metadata.
        response.choices[0].finish_reason = finish
        return response

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
    if not recover:
        with pytest.raises(StructuredOutputRepairError):
            compress_extraction_window([original], **kwargs)
        assert len(calls) == 2
        return
    result = compress_extraction_window([original], **kwargs)
    assert len(calls) == result.provider_attempts == 2
    assert result.provider_reference_recoveries == 1
    assert result.packet.evidence
    assert all(
        item.text in original.interactions[0].content for item in result.packet.evidence
    )
    assert any(
        "earlier complete schema-valid selection" in note
        for note in result.packet.omissions
    )
    assert all(
        "not-visible" not in point.evidence_refs for point in result.packet.points
    )
    replay = compress_extraction_window([original], **kwargs)
    assert replay.packet == result.packet and replay.provider_attempts == 0
    assert len(calls) == 2
