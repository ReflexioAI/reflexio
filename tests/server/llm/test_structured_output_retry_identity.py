"""What a retry's request must and must not carry over from the attempt before it.

WHY THIS FILE EXISTS

``_make_request`` re-enters the provider three different ways, and each one has
a DIFFERENT correct answer for the request's identity:

===============================  ==========================  ==================
crossing                          prompt                      Idempotency-Key
===============================  ==========================  ==================
plain rung, corrective retry      original + correction turn  RE-KEYED
plain rung, truncation re-issue   unchanged                   unchanged
validated rung, repair turn       original + raw echo + fix   RE-KEYED
===============================  ==========================  ==================

Two properties fall out of that table, and neither was guarded:

1. Where the prompt CHANGED, the key must change with it. A provider honouring
   ``Idempotency-Key`` is entitled to replay the first answer for a repeated
   key, which turns the correction into a silent no-op -- the whole defect
   ``structured_output_repair_idempotency_key`` exists to prevent. The
   validated rung's repair, which is a strictly larger change to the request
   than the plain rung's corrective retry, was reusing attempt 1's key.

   These tests assert the key is DIFFERENT. They never re-derive the expected
   value by calling the helper, because both sides of such an assertion move
   together: mutate ``structured_output_repair_idempotency_key`` to
   ``return base_key`` and a re-derived assertion still passes while the one
   property the helper exists for is gone.

2. Where the prompt did NOT change, the request must be re-issued BYTE for
   BYTE -- including the key, which a qualified caller pins while budgeting a
   bounded number of identical re-issues. That is only observable against a
   transport that mutates the params it is handed, the way litellm's
   ``validate_environment`` does (see ``test_transport_state_detachment``). A
   polite fake re-issues cleanly no matter how badly the code reuses state, so
   it is structurally incapable of failing here.
"""

import logging
import re
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from reflexio.server.llm._litellm_text_generation import (
    MAX_LOGGED_VALIDATION_ERRORS,
    _log_safe_validation_errors,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    StructuredOutputParseError,
    StructuredOutputRepairError,
    structured_output_repair_idempotency_key,
)

_MODEL = "primary-model"
_CALLER_KEY = "a" * 64
_CALLER_HEADERS = {"Idempotency-Key": _CALLER_KEY, "X-Reflexio-Caller": "unit-test"}

# A body that parses but fails the schema (two missing fields), and one that
# stops mid-string so the parser raises the deliberate "appears truncated"
# diagnostic whose ``validation_errors`` is empty.
_SCHEMA_FAIL_BODY = '{"wrong_field": 1}'
_TRUNCATED_BODY = '{"answer": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
_VALID_BODY = '{"answer": "ok", "score": 42}'


class SampleResponse(BaseModel):
    answer: str
    score: int


def _response(content: str, *, finish_reason: str = "stop") -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.message.refusal = None
    choice.message.tool_calls = None
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    response.model = _MODEL
    response._hidden_params = {"custom_llm_provider": "openai"}
    response.usage = None
    return response


def _scribbling_transport(snapshots: list[dict[str, Any]], bodies: list[str]) -> Any:
    """Serve ``bodies`` in order, scribbling on the params exactly as litellm does.

    ``OpenAIGPTConfig.validate_environment`` writes into the ``extra_headers``
    dict it RECEIVED and hands the same object onward; the same in-place shape
    is reproduced for ``messages``. The snapshot is taken BEFORE the scribble,
    so ``snapshots[n]`` is what the caller's code actually chose to send on
    crossing ``n`` rather than what the previous crossing left behind.
    """

    def serve(**params: Any) -> MagicMock:
        snapshots.append(deepcopy(params))
        headers = params.get("extra_headers")
        if isinstance(headers, dict):
            headers["Authorization"] = "Bearer sk-not-a-real-key"
            headers.setdefault("Content-Type", "application/json")
        messages = params.get("messages")
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                first["_transport_scribble"] = True
        return _response(bodies[len(snapshots) - 1])

    return serve


def _sent_keys(snapshots: list[dict[str, Any]]) -> list[str]:
    return [snapshot["extra_headers"]["Idempotency-Key"] for snapshot in snapshots]


# ---------------------------------------------------------------------------
# The helper's one property, asserted directly rather than re-derived.
# ---------------------------------------------------------------------------


def test_the_repair_key_is_not_the_key_it_repairs() -> None:
    """The single reason the helper exists.

    ``return base_key`` is the mutation this guards: it is what every
    re-derived assertion elsewhere would still accept.
    """
    assert structured_output_repair_idempotency_key(_CALLER_KEY) != _CALLER_KEY


def test_the_repair_key_is_64_hex_characters() -> None:
    """Provider clients require the width; a shorter key is rejected on the wire.

    The base key here is deliberately NOT hex, so an identity implementation
    fails this on shape as well as on distinctness.
    """
    derived = structured_output_repair_idempotency_key("caller-supplied-request-id")
    assert re.fullmatch(r"[0-9a-f]{64}", derived)


def test_the_repair_key_is_deterministic() -> None:
    """A replayed request must stay idempotent with ITSELF."""
    assert structured_output_repair_idempotency_key(
        _CALLER_KEY
    ) == structured_output_repair_idempotency_key(_CALLER_KEY)


def test_distinct_base_keys_derive_distinct_repair_keys() -> None:
    """Two callers' corrections must not collide onto one provider-side entry."""
    assert structured_output_repair_idempotency_key(
        "a" * 64
    ) != structured_output_repair_idempotency_key("b" * 64)


def test_the_repair_key_does_not_collide_with_its_own_derivation() -> None:
    """Deriving twice must not land back on either earlier value."""
    once = structured_output_repair_idempotency_key(_CALLER_KEY)
    twice = structured_output_repair_idempotency_key(once)
    assert len({_CALLER_KEY, once, twice}) == 3


# ---------------------------------------------------------------------------
# Crossings whose PROMPT changed must be re-keyed.
# ---------------------------------------------------------------------------


def test_the_validated_rungs_repair_re_keys_the_idempotency_header() -> None:
    """The repair turn is a different request, so it must be a different key.

    ``_repair_messages`` appends the model's own raw output AND a correction
    instruction, making this a strictly larger change than the plain rung's
    corrective retry -- which was already re-keyed. Reusing attempt 1's key
    entitles a provider honouring the header to replay the very answer being
    repaired.
    """
    snapshots: list[dict[str, Any]] = []
    client = LiteLLMClient(LiteLLMConfig(model=_MODEL))

    def validator(parsed: BaseModel) -> list[str]:
        assert isinstance(parsed, SampleResponse)
        return [] if parsed.score == 42 else ["score must be 42"]

    with patch(
        "litellm.completion",
        side_effect=_scribbling_transport(
            snapshots, ['{"answer": "bad", "score": 1}', _VALID_BODY]
        ),
    ):
        result = client.generate_chat_response(
            [{"role": "user", "content": "hi"}],
            response_format=SampleResponse,
            structured_output_validator=validator,
            extra_headers=dict(_CALLER_HEADERS),
        )

    assert isinstance(result, SampleResponse)
    assert len(snapshots) == 2
    # The repair really is a repair -- otherwise this measures the wrong crossing.
    assert [m["role"] for m in snapshots[1]["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    first_key, repair_key = _sent_keys(snapshots)
    assert first_key == _CALLER_KEY
    assert repair_key != first_key
    # The caller's other headers ride along untouched, and the provider's
    # injected ones never do.
    assert snapshots[1]["extra_headers"]["X-Reflexio-Caller"] == "unit-test"
    assert "Authorization" not in snapshots[1]["extra_headers"]


def test_the_plain_rungs_corrective_retry_re_keys_the_idempotency_header() -> None:
    """The sibling crossing, asserted the same way rather than by re-derivation."""
    snapshots: list[dict[str, Any]] = []
    client = LiteLLMClient(LiteLLMConfig(model=_MODEL))

    with patch(
        "litellm.completion",
        side_effect=_scribbling_transport(snapshots, [_SCHEMA_FAIL_BODY, _VALID_BODY]),
    ):
        result = client.generate_chat_response(
            [{"role": "user", "content": "hi"}],
            response_format=SampleResponse,
            extra_headers=dict(_CALLER_HEADERS),
        )

    assert isinstance(result, SampleResponse)
    assert len(snapshots) == 2
    # A corrective turn was appended -- the branch under test, not the re-issue.
    assert len(snapshots[1]["messages"]) > len(snapshots[0]["messages"])
    first_key, corrective_key = _sent_keys(snapshots)
    assert first_key == _CALLER_KEY
    assert corrective_key != first_key
    assert "Authorization" not in snapshots[1]["extra_headers"]


# ---------------------------------------------------------------------------
# The crossing whose prompt did NOT change must be byte-identical.
# ---------------------------------------------------------------------------


def test_a_truncation_diagnostic_reissues_the_callers_exact_request() -> None:
    """Crossing 2 must be what the CALLER asked for, not what crossing 1 left.

    ``_detach_request_state`` runs inside ``_prepare_turn``, so it protects the
    caller's objects and every later rung -- but WITHIN one rung the params
    dict has already crossed to the transport, and litellm's in-process route
    (``litellm.completion(**params)``) lets ``validate_environment`` write into
    ``params["extra_headers"]``. Re-calling with that same object sends the
    provider's own headers as caller intent, which a ``provider_request_guard``
    pinning the exact request correctly refuses.

    The re-issue must also keep attempt 1's key: nothing about the request
    changed, and the qualified caller this branch serves pins ``extra_headers``
    to the exact key it issued while budgeting a bounded number of IDENTICAL
    re-issues. A derived key here would fail that pin.
    """
    snapshots: list[dict[str, Any]] = []
    client = LiteLLMClient(LiteLLMConfig(model=_MODEL))

    with patch(
        "litellm.completion",
        side_effect=_scribbling_transport(snapshots, [_TRUNCATED_BODY, _VALID_BODY]),
    ):
        result = client.generate_chat_response(
            [{"role": "user", "content": "hi"}],
            response_format=SampleResponse,
            extra_headers=dict(_CALLER_HEADERS),
        )

    assert isinstance(result, SampleResponse)
    assert len(snapshots) == 2
    # The fake really did scribble -- otherwise the equality below is vacuous.
    assert snapshots[0] is not snapshots[1]
    assert snapshots[1] == snapshots[0]
    assert snapshots[1]["extra_headers"] == _CALLER_HEADERS
    assert snapshots[1]["messages"] == [{"role": "user", "content": "hi"}]


def test_the_scribbling_transport_would_expose_a_reused_params_dict() -> None:
    """The fake used above is not polite: it mutates what it is handed.

    Without this, ``snapshots[1] == snapshots[0]`` could pass because the fake
    never wrote anything, on the exact branch that defect lives in.
    """
    snapshots: list[dict[str, Any]] = []
    serve = _scribbling_transport(snapshots, [_VALID_BODY, _VALID_BODY])
    params = {
        "extra_headers": {"Idempotency-Key": _CALLER_KEY},
        "messages": [{"role": "user", "content": "hi"}],
    }

    serve(**params)

    assert sorted(params["extra_headers"]) == [
        "Authorization",
        "Content-Type",
        "Idempotency-Key",
    ]
    assert params["messages"][0]["_transport_scribble"] is True


# ---------------------------------------------------------------------------
# Raw pydantic ``loc`` strings must not reach a log sink unclamped.
# ---------------------------------------------------------------------------


def test_a_log_safe_rendering_passes_ordinary_schema_paths_through() -> None:
    """Clamping must not cost the diagnostic its whole value."""
    rendered = _log_safe_validation_errors(
        ("items.0.name: missing", "<root>: json_invalid at line 3 column 5")
    )
    assert rendered == "items.0.name: missing,<root>: json_invalid at line 3 column 5"


@pytest.mark.parametrize(
    "hostile",
    [
        "evil\nWARNING event=fabricated: extra_forbidden",  # log-line injection
        "k" * 200 + ": extra_forbidden",  # unbounded length
        'say "hi": extra_forbidden',  # quotes
        "ключ: extra_forbidden",  # non-ASCII
        "a\x00b: extra_forbidden",  # control character
    ],
)
def test_a_log_safe_rendering_redacts_a_model_chosen_key(hostile: str) -> None:
    """``loc`` is built from ``str(part)`` with no clamp; an ``extra_forbidden``
    entry therefore carries whatever key the MODEL invented."""
    assert _log_safe_validation_errors((hostile,)) == "<redacted>"


def test_a_log_safe_rendering_bounds_the_error_count() -> None:
    """A pathological body can produce an unbounded number of entries."""
    errors = tuple(f"field{index}: missing" for index in range(20))

    rendered = _log_safe_validation_errors(errors)

    assert rendered.count(",") == MAX_LOGGED_VALIDATION_ERRORS
    assert rendered.endswith(f"...({20 - MAX_LOGGED_VALIDATION_ERRORS} more)")
    assert "field19" not in rendered


def test_the_validated_rung_does_not_log_a_model_chosen_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End to end: the parse-failure log line the validated rung emits.

    ``_run_rung_validated`` joined ``exc.validation_errors`` straight into a
    ``logger.warning``, and that line bridges to external telemetry. The
    injected substring below is what an ``extra_forbidden`` entry would put
    there verbatim.
    """
    hostile = "evil\nWARNING event=FABRICATED_BY_THE_MODEL: extra_forbidden"
    client = LiteLLMClient(LiteLLMConfig(model=_MODEL))
    calls: list[int] = []

    def failing_parse(content: Any, response_format: Any, parse: bool) -> BaseModel:
        calls.append(1)
        raise StructuredOutputParseError(
            "Structured output parse failed",
            raw_content="{}",
            validation_errors=(hostile,),
        )

    with (
        patch("litellm.completion", side_effect=lambda **_: _response(_VALID_BODY)),
        patch.object(client, "_maybe_parse_structured_output", failing_parse),
        caplog.at_level(logging.WARNING),
        pytest.raises(StructuredOutputRepairError),
    ):
        client.generate_chat_response(
            [{"role": "user", "content": "hi"}],
            response_format=SampleResponse,
            structured_output_validator=lambda _parsed: [],
        )

    assert calls, "the parse seam was never reached"
    validation_records = [
        record.getMessage()
        for record in caplog.records
        if "llm_structured_schema_validation_failed" in record.getMessage()
    ]
    # Both log sites -- the first attempt's and the repair attempt's -- are
    # reached, so this covers the pair rather than whichever one runs first.
    assert len(validation_records) == 2
    assert all(
        "FABRICATED_BY_THE_MODEL" not in message for message in validation_records
    )
    assert all("<redacted>" in message for message in validation_records)
