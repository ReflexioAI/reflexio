"""Provider-free transfer coverage for the selected complete V8 workflow."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm._litellm_types import (
    CompletionResult,
    LiteLLMConfig,
    ModelProvenance,
    ProviderRequestGuardError,
    ToolCallingChatResponse,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.playbook.procedural.models import (
    ContractError,
    Input,
    Observation,
    Source,
)
from reflexio.server.services.playbook.procedural.transport import StageClient
from reflexio.server.services.playbook.procedural.workflow import run_pipeline

WITNESS = "EXCERPT_1_1_1"


def _call(
    name: str, arguments: dict[str, Any] | str, identity: str = "return-1"
) -> dict[str, Any]:
    return {
        "id": identity,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments
            if isinstance(arguments, str)
            else json.dumps(arguments),
        },
    }


def _response(*calls: dict[str, Any], finish: str = "tool_calls") -> CompletionResult:
    return CompletionResult(
        value=ToolCallingChatResponse(
            content=None, tool_calls=list(calls), finish_reason=finish
        ),
        provenance=ModelProvenance(model_name="observed-model", provider="minimax"),
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


def _finish(status: str = "complete", identity: str = "finish") -> dict[str, Any]:
    return _call(
        "finish_batch", {"status": status, "reason": "Nothing useful."}, identity
    )


def _source() -> CompletionResult:
    return _response(
        _call(
            "record_observation",
            {
                "topic": "Release order",
                "evidence": WITNESS,
                "observation": "Reader's untrusted proposed answer.",
            },
        )
    )


def _create(
    name: str = "deploy-service", group: str = "A", identity: str = "create"
) -> dict[str, Any]:
    return _call(
        "create_procedure",
        {
            "group": group,
            "name": name,
            "trigger": "Preparing a release",
            "entry": "Use staging before production.",
            "evidence": WITNESS,
        },
        identity,
    )


def _verdict(
    unit: str, *, state: str = "pass", decision: str = "accept"
) -> dict[str, Any]:
    return _call(
        "review_result",
        {
            "unit": unit,
            "decision": decision,
            "result_support": "pass",
            "scope_conditions": state,
            "useful_nonduplicate": "pass",
            "preservation_and_correction": "pass",
            "resource_consistency": "pass",
            "reason": "Original source supports the complete result.",
        },
        unit,
    )


def _client(responses: list[CompletionResult]) -> tuple[StageClient, MagicMock]:
    native = MagicMock(spec=LiteLLMClient)
    native.config = LiteLLMConfig(
        model="minimax/MiniMax-M3", max_tokens=None, timeout=240
    )
    native.get_completion_model_ladder.side_effect = lambda **kwargs: (kwargs["model"],)
    pending = iter(responses)

    def complete(**kwargs: Any) -> CompletionResult:
        kwargs["provider_request_guard"]({}, 240, (kwargs["model"],), False)
        return next(pending)

    native.generate_chat_response_with_provenance.side_effect = complete
    return StageClient(native, "minimax/MiniMax-M3"), native


def _input() -> Input:
    return Input(
        owner="owner",
        sources=[
            Source(
                interaction_id=2,
                request_id="request",
                owner="owner",
                role="User",
                request_source="app",
                text="Use staging before production.",
            )
        ],
        existing=[],
        agent_context="Release assistant",
        definition="Learn recurring release procedures",
        tool_context="",
    )


def _ask_source(client: StageClient) -> list[Observation]:
    return client.ask(
        "source",
        "Read source",
        {},
        {"record_observation": Observation},
        lambda batch: [value for _, value in batch],
        lambda _: None,
        evidence_choices=[WITNESS],
    )


def test_complete_selected_pipeline_uses_three_calls_and_exact_full_result() -> None:
    client, native = _client(
        [_source(), _response(_create()), _response(_verdict("U0"))]
    )
    receipt = run_pipeline(_input(), {2: ["Use staging before production."]}, client)
    assert receipt["outcome"] == "accept"
    # Generated independently by the selected frozen V8 implementation for this
    # scripted three-stage fixture, not by the maintained port under test.
    assert (
        receipt["proposed_sha256"]
        == receipt["reviewed_sha256"]
        == ("c825e672d77246ab81d8eb1141b6e5e8decd34e8edbe7718bf1486b2d055820f")
    )
    assert receipt["accepted"] == ["W0"] and receipt["accepted_groups"] == ["A"]
    assert (
        receipt["rendered"]["W0"]["content"]
        == '<!-- reflexio-skill-bundle-v1 {"name":"deploy-service","pages":[{"path":"SKILL.md","chars":30}]} -->\n\n## SKILL.md\n\nUse staging before production.'
    )
    assert receipt["rendered"]["W0"]["source_interaction_ids"] == [2]
    assert receipt["rendered"]["W0"]["source_span"] == "Use staging before production."
    assert native.generate_chat_response_with_provenance.call_count == 3
    assert client.calls == client.physical_calls == 3
    requests = [
        call.kwargs
        for call in native.generate_chat_response_with_provenance.call_args_list
    ]
    assert [request["model"] for request in requests] == [
        "minimax/MiniMax-M3",
        "minimax/MiniMax-M3.1-Flash",
        "minimax/MiniMax-M3",
    ]
    assert (
        "reasoning_effort" not in requests[0]
        and requests[1]["reasoning_effort"] == "low"
        and "reasoning_effort" not in requests[2]
    )
    compiler_body = json.loads(requests[1]["messages"][1]["content"])
    assert compiler_body["source_observations"] == [
        {
            "topic": "Source-linked learning",
            "evidence": WITNESS,
            "observation": "Read the original cited source before authoring a rule.",
        }
    ]
    assert (
        "Reader's untrusted proposed answer."
        not in requests[1]["messages"][1]["content"]
    )
    review_body = json.loads(requests[2]["messages"][1]["content"])
    assert (
        review_body["result_review_units"][0]["resulting_record"]
        == receipt["rendered"]["W0"]
    )
    assert [turn.tool_name for turn in client.trace.turns] == [
        "procedural_source",
        "procedural_maintain",
        "procedural_review",
    ]
    assert [turn.total_tokens for turn in client.trace.turns] == [15, 15, 15]


def test_valid_source_noop_skips_compiler_and_reviewer() -> None:
    client, native = _client([_response(_finish())])
    receipt = run_pipeline(_input(), {2: ["Use staging before production."]}, client)
    assert receipt["outcome"] == "empty" and receipt["rendered"] == {}
    assert receipt["produced"] == {"changes": []} and receipt["group_review"] is None
    assert native.generate_chat_response_with_provenance.call_count == 1


def test_rejected_group_keeps_independent_group_and_original_output_alias() -> None:
    client, native = _client(
        [
            _source(),
            _response(
                _create("first", "coupled", "c0"),
                _create("second", "coupled", "c1"),
                _create("third", "independent", "c2"),
            ),
            _response(_verdict("U0"), _verdict("U1", state="fail"), _verdict("U2")),
        ]
    )
    receipt = run_pipeline(_input(), {2: ["Use staging before production."]}, client)
    assert receipt["accepted"] == ["W2"] and receipt["accepted_groups"] == [
        "independent"
    ]
    assert list(receipt["rendered"]) == ["W2"]
    assert receipt["accepted_changes"]["changes"][0]["document"]["name"] == "third"
    assert native.generate_chat_response_with_provenance.call_count == 3


def test_unknown_review_is_terminal_rejection_without_semantic_reroll() -> None:
    client, native = _client(
        [_source(), _response(_create()), _response(_verdict("U0", state="unknown"))]
    )
    receipt = run_pipeline(_input(), {2: ["Use staging before production."]}, client)
    assert receipt["outcome"] == "reject" and receipt["accepted"] == []
    assert native.generate_chat_response_with_provenance.call_count == 3


def test_missing_review_unit_gets_one_structural_repair_then_fails() -> None:
    client, native = _client(
        [_source(), _response(_create()), _response(_finish()), _response(_finish())]
    )
    with pytest.raises(ContractError, match="exactly once"):
        run_pipeline(_input(), {2: ["Use staging before production."]}, client)
    assert native.generate_chat_response_with_provenance.call_count == 4


def test_malformed_json_repairs_complete_batch_and_preserves_bad_delivery() -> None:
    bad = _call("record_observation", "{invalid json")
    client, native = _client([_response(bad), _source()])
    assert len(_ask_source(client)) == 1
    calls = native.generate_chat_response_with_provenance.call_args_list
    assert len(calls) == 2
    assert calls[1].kwargs["messages"][2]["tool_calls"] == [bad]
    assert calls[1].kwargs["messages"][3]["tool_call_id"] == bad["id"]
    assert [turn.result for turn in client.trace.turns] == [
        {"structural_repair": False},
        {"structural_repair": True},
    ]


@pytest.mark.parametrize("finish", ["length", "stop"])
def test_truncated_or_plain_stop_is_terminal_not_repair(finish: str) -> None:
    client, native = _client([_response(_finish(), finish=finish)])
    with pytest.raises(ContractError, match="complete terminal"):
        _ask_source(client)
    assert native.generate_chat_response_with_provenance.call_count == 1


def test_explicit_incomplete_overrides_bad_prior_declaration_without_reroll() -> None:
    client, native = _client(
        [_response(_call("foreign", "{bad", "prior"), _finish("incomplete"))]
    )
    with pytest.raises(ContractError, match="method failure"):
        _ask_source(client)
    assert native.generate_chat_response_with_provenance.call_count == 1


@pytest.mark.parametrize("bad_evidence", ["EXCERPT_missing", WITNESS + " " + WITNESS])
def test_foreign_or_joined_evidence_cannot_bypass_offered_enum(
    bad_evidence: str,
) -> None:
    wrong = _call(
        "record_observation",
        {"topic": "Release", "evidence": bad_evidence, "observation": "Claim"},
    )
    client, native = _client([_response(wrong), _response(wrong)])
    with pytest.raises(ContractError, match="one offered witness"):
        _ask_source(client)
    schema = native.generate_chat_response_with_provenance.call_args_list[0].kwargs[
        "tools"
    ][0]["function"]["parameters"]
    assert schema["properties"]["evidence"]["enum"] == [WITNESS]
    assert native.generate_chat_response_with_provenance.call_count == 2


def test_six_call_operation_bound_does_not_allow_a_seventh_stage() -> None:
    client, native = _client([_response(_finish()) for _ in range(6)])
    for _ in range(6):
        assert _ask_source(client) == []
    with pytest.raises(ContractError, match="allowance exhausted"):
        _ask_source(client)
    assert client.calls == client.physical_calls == 6
    assert native.generate_chat_response_with_provenance.call_count == 6


def test_native_fallback_dispatches_share_same_six_physical_call_bound() -> None:
    client, native = _client([])

    def excess_fallbacks(**kwargs: Any) -> CompletionResult:
        for _ in range(7):
            kwargs["provider_request_guard"]({}, 240, (kwargs["model"],), False)
        raise AssertionError("Seventh dispatch must not be admitted")

    native.generate_chat_response_with_provenance.side_effect = excess_fallbacks
    with pytest.raises(ProviderRequestGuardError, match="dispatch allowance exhausted"):
        _ask_source(client)
    assert client.physical_calls == 6 and client.calls == 1


def test_foreign_or_duplicate_provider_call_ids_fail_without_repair() -> None:
    first = _call(
        "record_observation",
        {"topic": "Release", "evidence": WITNESS, "observation": "Claim"},
        "same",
    )
    client, native = _client([_response(first, _finish(identity="same"))])
    with pytest.raises(ContractError, match="unique nonempty"):
        _ask_source(client)
    assert native.generate_chat_response_with_provenance.call_count == 1


def test_explicit_native_output_and_timeout_policy_is_respected() -> None:
    _, native = _client([_response(_finish())])
    native.config = LiteLLMConfig(
        model="minimax/MiniMax-M3", max_tokens=1024, timeout=60
    )
    client = StageClient(native, "minimax/MiniMax-M3")
    assert _ask_source(client) == []
    assert all(
        profile["max_tokens"] == 1024 and profile["timeout"] == 60
        for profile in client.profiles.values()
    )
    request = native.generate_chat_response_with_provenance.call_args.kwargs
    assert request["max_tokens"] == 1024 and request["timeout"] == 60


def test_non_minimax_gateway_keeps_its_configured_model_in_every_stage() -> None:
    _, native = _client([])
    native.config = LiteLLMConfig(
        model="openai/native-functional", max_tokens=1024, timeout=90
    )
    client = StageClient(native, "openai/native-functional")
    assert {profile["model"] for profile in client.profiles.values()} == {
        "openai/native-functional"
    }
    assert all(
        "reasoning_effort" not in profile for profile in client.profiles.values()
    )


def test_existing_body_edit_noop_hashes_and_keeps_actual_bytes_not_read_view() -> None:
    from reflexio.server.services.playbook.procedural import codec
    from reflexio.server.services.playbook.procedural.models import Document, Page
    from reflexio.server.services.playbook.procedural.workflow import (
        digest,
        incumbent_state,
    )

    original = codec.encode(
        Document(
            name="deploy-service",
            pages=[
                Page(path="SKILL.md", content="Deploy after approval."),
                Page(path="assets/check.txt", content="Retain this resource."),
            ],
        )
    )
    raw = original.replace(
        "Deploy after approval.", "Deploy only after explicit approval."
    )
    incumbent = {
        "user_playbook_id": 31,
        "user_id": "owner",
        "content": raw,
        "trigger": "Deploying a release",
        "source_interaction_ids": [1],
        "source_span": "Explicit approval is required.",
    }
    data = _input().model_copy(update={"existing": [incumbent]})
    client, native = _client([_response(_finish())])
    receipt = run_pipeline(data, {2: ["Use staging before production."]}, client)
    assert receipt["outcome"] == "empty"
    assert receipt["plan"]["keep_existing"] == [31]
    assert receipt["old_state"]["31"] == digest(incumbent_state(incumbent))
    source = json.loads(
        native.generate_chat_response_with_provenance.call_args.kwargs["messages"][1][
            "content"
        ]
    )
    assert source["existing"]["P1"]["content"] == raw
    assert source["existing"]["P1"]["document"]["pages"] == [
        {"path": "SKILL.md", "content": "Deploy only after explicit approval."},
        {"path": "assets/check.txt", "content": "Retain this resource."},
    ]
    assert data.existing[0]["content"] == raw
