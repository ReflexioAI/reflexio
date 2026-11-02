"""Repair salvage metadata pairs a parsed candidate with its own stop reason."""

import json

import litellm
import pytest
from pydantic import BaseModel

from reflexio.server.llm.completion_io import completion_io_scope
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    StructuredOutputRepairError,
)


class ScoredAnswer(BaseModel):
    answer: str
    score: int


def response(body, finish_reason):
    result = litellm.ModelResponse(
        model="gpt-4o",
        choices=[
            {
                "message": {"role": "assistant", "content": json.dumps(body)},
                "finish_reason": finish_reason,
            }
        ],
    )
    # LiteLLM construction defaults a missing stop reason to "stop". Retain
    # an explicitly unknown observation to test that our error adds no inference.
    result.choices[0].finish_reason = finish_reason
    return result


def exhausted(replies):
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=1024, fallback_models=[])
    )
    calls = []

    def serve(_params, _timeout, _dispatch):
        calls.append(len(calls))
        return replies[len(calls) - 1]

    with (
        completion_io_scope(serve),
        pytest.raises(StructuredOutputRepairError) as caught,
    ):
        client.generate_chat_response(
            [{"role": "user", "content": "Synthetic answer"}],
            response_format=ScoredAnswer,
            structured_output_validator=lambda _: ["synthetic semantic failure"],
        )
    assert len(calls) == len(replies)
    return caught.value


@pytest.mark.parametrize("first_finish", ["stop", "length", None])
@pytest.mark.parametrize("last_finish", ["stop", "length", None])
def test_final_parse_failure_preserves_prior_parsed_candidate_stop_reason(
    first_finish, last_finish
):
    original = {"answer": "prior", "score": 2}
    invalid = {"wrong_field": "not a scored answer"}
    error = exhausted(
        [response(original, first_finish), response(invalid, last_finish)]
    )
    assert error.failure_kind == "parse"
    assert error.parsed_output == ScoredAnswer.model_validate(original)
    assert error.parsed_output_finish_reason == first_finish
    assert error.finish_reason == last_finish
    assert isinstance(error.raw_content, str)
    assert json.loads(error.raw_content) == invalid


@pytest.mark.parametrize("finish_reason", ["stop", "length", None])
def test_latest_schema_valid_output_owns_its_stop_metadata(finish_reason):
    original = {"answer": "prior", "score": 2}
    latest = {"answer": "latest", "score": 1}
    error = exhausted([response(original, "length"), response(latest, finish_reason)])
    assert error.failure_kind == "semantic"
    assert error.parsed_output == ScoredAnswer.model_validate(latest)
    assert error.parsed_output_finish_reason == finish_reason
    assert error.finish_reason == finish_reason
    assert isinstance(error.raw_content, str)
    assert json.loads(error.raw_content) == latest


def test_no_parsed_candidate_has_no_candidate_stop_metadata():
    error = exhausted([response({"wrong": 1}, "stop"), response({"wrong": 2}, "stop")])
    assert error.failure_kind == "parse"
    assert error.parsed_output is None
    assert error.parsed_output_finish_reason is None
    assert error.finish_reason == "stop"


def test_refusal_carries_actual_final_stop_without_candidate():
    error = exhausted([response({"answer": "refused", "score": 0}, "content_filter")])
    assert error.failure_kind == "refusal"
    assert error.finish_reason == "content_filter"
    assert error.parsed_output is None
    assert error.parsed_output_finish_reason is None


def test_legacy_error_constructor_does_not_invent_stop_metadata():
    error = StructuredOutputRepairError(
        "synthetic", failure_kind="parse", model="gpt-4o"
    )
    assert error.finish_reason is None
    assert error.parsed_output_finish_reason is None
