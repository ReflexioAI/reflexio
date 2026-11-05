"""Provider rate limiting: detect BOTH shapes, back off, bound the budget.

WHAT THESE TESTS ARE FOR, in priority order:

1. **The HTTP-200 shape is detected at all.** MiniMax documents its rate-limit
   TIERS and not its error SEMANTICS -- no published 429 contract, no
   ``Retry-After``. Third-party integration reports describe the limit arriving
   as an application-level ``base_resp.status_code`` INSIDE an HTTP 200 body,
   which a detector keyed on HTTP 429 would never see. Nothing here has been
   measured against a live rate-limited MiniMax response (provoking one needs
   far more than a normal sweep's request rate), so the detector covers both
   shapes and both are tested.

   The 200 case is the one that is easy to get wrong in a way that still looks
   right: ``StructuredOutputParseError``'s message is deliberately content-free
   and carries the body on ``raw_content``. Measured by mutation: removing the
   ``raw_content`` arm still leaves the end-to-end case green, because the
   chained pydantic ``ValidationError`` happens to render the body in
   ``input_value=``. It breaks only the shapes with no ``ValidationError``
   beneath them -- an empty completion, a truncated one -- which is why the
   unit test below constructs that exact chain rather than relying on the
   end-to-end one to cover it.

2. **The budget is a budget, not a count.** Deployments that share one
   provider key meet the same rolling window at the same instant, so N deployments
   each retrying "three times" is 3N requests at one quota in one window. The
   budget is seconds, spent across the whole attempt, with FULL jitter.

3. **A rate-limited tick does not spend the playbook's day.** That is the
   progress-record half, and it is the one with money on it: the day record has
   to come back RETRY-ELIGIBLE rather than done.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import litellm
import pytest

from reflexio.server.llm import LiteLLMClientError
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.provider_circuit import ProviderCircuitOpenError
from reflexio.server.services.offline_tuner.open_world.analyst import (
    _RATE_LIMIT_BACKOFF_CAP_SECONDS,
    _RATE_LIMIT_RETRY_BUDGET_SECONDS,
    ChatMessage,
    OpenWorldProviderRateLimitedError,
    compose_litellm_open_world_model_client,
    rate_limit_evidence,
)
from reflexio.server.services.offline_tuner.open_world.models import DiscoveryMemo

_MODEL = "gpt-4o-2024-08-06"
_PROVIDER = "openai"
_IDEMPOTENCY_KEY = "a" * 64

#: A MiniMax rate limit as third-party reports describe it: HTTP 200, no
#: ``choices``, the refusal carried in the application envelope.
_BASE_RESP_BODY = (
    '{"base_resp": {"status_code": 1002, "status_msg": '
    '"rate limit, please retry later"}}'
)

_MEMO = DiscoveryMemo(
    decision="abstain",
    hypothesis="The frozen sessions do not support a bounded hypothesis.",
    observations=("The rate-limit test does not exercise citation verification.",),
    citations=(),
    relevance_to_success="Not established.",
    alternative_explanations=("The evidence is insufficient.",),
    counterevidence=(),
    scope="None.",
    uncertainties=("Historical evidence is observational.",),
)


class RateLimitError(Exception):
    """A stand-in for the provider SDK class, matched by NAME.

    ``_RATE_LIMIT_TRANSPORT_ERROR_NAMES`` classifies by class name rather than
    by import, the same way ``_TRANSIENT_LLM_ERROR_NAMES`` does upstream, so
    that neither the heavy ``litellm`` nor the ``openai`` exception hierarchy
    has to be imported to recognise one.
    """


def _provider_response(content: object) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.message.refusal = None
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.model = _MODEL
    response._hidden_params = {"custom_llm_provider": _PROVIDER}
    response.usage = None
    return response


def _composed(monkeypatch: pytest.MonkeyPatch, serve: Any) -> Any:
    """Compose the production transport with sleeps observed, not served."""
    composed = compose_litellm_open_world_model_client(
        LiteLLMClient(LiteLLMConfig(model=_MODEL))
    )
    slept: list[float] = []
    monkeypatch.setattr(litellm, "completion", serve)
    composed._sleep = slept.append
    # Full jitter draws uniform(0, window); pinning the draw to the TOP of the
    # window makes the budget arithmetic deterministic without replacing the
    # jitter itself -- `test_the_backoff_is_fully_jittered` asserts the real
    # call is `uniform(0, window)`.
    composed._jitter = lambda _low, high: high
    return composed, slept


def _complete(composed: Any) -> Any:
    return composed.complete(
        messages=(ChatMessage("system", "policy"), ChatMessage("user", "data")),
        response_format=DiscoveryMemo,
        model_revision=_MODEL,
        idempotency_key=_IDEMPOTENCY_KEY,
    )


# ---------------------------------------------------------------------------
# 1. Detection -- both shapes
# ---------------------------------------------------------------------------


def test_an_http_429_is_detected_as_the_transport_shape() -> None:
    assert rate_limit_evidence(RateLimitError("boom")) == "transport"
    assert rate_limit_evidence(Exception("provider returned status 429")) == "transport"


def test_the_upstream_error_type_is_read_when_the_ladder_wraps_it() -> None:
    """The wrap keeps the CHAIN, so the chain is what must be walked.

    ``_litellm_text_generation`` re-wraps a final-rung failure as
    ``LiteLLMClientError(f"API call failed: {exc}") from exc`` and does NOT
    forward ``upstream_error_type``. A detector that inspected only the object
    it was handed would miss every laddered rate limit.
    """
    inner = LiteLLMClientError(
        "transport gave up", upstream_error_type="RateLimitError"
    )
    outer = LiteLLMClientError("API call failed: transport gave up")
    outer.__cause__ = inner
    assert rate_limit_evidence(outer) == "transport"
    assert rate_limit_evidence(ProviderCircuitOpenError("refused")) == "circuit_open"


def test_the_http_200_base_resp_shape_is_detected_from_raw_content() -> None:
    """THE shape a 429-only detector would never see.

    The chain built here is the one with NO pydantic ``ValidationError`` in it
    -- an empty or truncated completion -- where ``raw_content`` is the body's
    only carrier. That is deliberate: with a ``ValidationError`` present, its
    ``input_value=`` fragment leaks the body into ``str()`` and a detector
    reading only messages passes anyway, so a test built on that chain would
    not notice the arm being removed. Measured by mutation, not assumed.
    """
    parse_error = Exception("Structured output parse failed (content omitted)")
    parse_error.raw_content = _BASE_RESP_BODY  # type: ignore[attr-defined]
    wrapped = LiteLLMClientError("API call failed: parse failed")
    wrapped.__cause__ = parse_error
    assert rate_limit_evidence(wrapped) == "base_resp"


def test_a_bare_status_code_without_the_envelope_is_not_a_rate_limit() -> None:
    """1002 is only a rate limit inside a ``base_resp`` envelope.

    Without the envelope requirement any message carrying the substring "1002"
    -- a job id, a token count, a byte length -- would be classified as a rate
    limit and retried, spending budget on a failure that will recur identically.
    """
    assert rate_limit_evidence(Exception('{"status_code": 1002}')) is None
    assert rate_limit_evidence(Exception("processed 1002 evidence spans")) is None


def test_an_ordinary_failure_is_not_a_rate_limit() -> None:
    assert (
        rate_limit_evidence(LiteLLMClientError("API call failed: bad schema")) is None
    )
    assert rate_limit_evidence(ValueError("provider served a different model")) is None


def test_a_four_digit_number_containing_429_is_not_a_status_code() -> None:
    assert rate_limit_evidence(Exception("job 4290 failed")) is None
    assert rate_limit_evidence(Exception("completed in 1429 ms")) is None


def test_the_walk_terminates_on_a_self_referential_chain() -> None:
    """A cycle in ``__context__`` must not hang the attempt."""
    first = Exception("one")
    second = Exception("two")
    first.__context__ = second
    second.__context__ = first
    assert rate_limit_evidence(first) is None


# ---------------------------------------------------------------------------
# 2. Backoff and the budget
# ---------------------------------------------------------------------------


def test_a_rate_limited_crossing_is_retried_and_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect this task fixes: one 429 used to cost the whole day."""
    crossings: list[int] = []

    def serve(**_params: object) -> MagicMock:
        crossings.append(1)
        if len(crossings) == 1:
            raise RateLimitError("429 Too Many Requests")
        return _provider_response(_MEMO.model_dump_json())

    composed, slept = _composed(monkeypatch, serve)
    result = _complete(composed)

    assert len(crossings) == 2
    assert len(slept) == 1
    assert result.value == _MEMO
    assert composed.rate_limit_budget_remaining < _RATE_LIMIT_RETRY_BUDGET_SECONDS


def test_the_retry_reuses_the_idempotency_key_and_the_qualified_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rate-limited crossing never reached the model, so nothing is re-asked.

    Both halves are load-bearing. The MESSAGES must be byte-identical, because
    ``guard_exact_request`` admits a second crossing only as a re-issue or as
    the client's own bounded correction -- and this is neither a correction nor
    a new question. The KEY must be unchanged, because this is still one
    logical attempt and a provider honouring the header should be free to
    collapse the retries rather than bill for both.

    Without the per-crossing reset of ``provider_attempts`` inside the retry
    loop, our OWN guard would refuse the second crossing as over-budget.
    """
    calls: list[dict[str, object]] = []

    def serve(**params: object) -> MagicMock:
        calls.append(dict(params))
        if len(calls) == 1:
            raise RateLimitError("429 Too Many Requests")
        return _provider_response(_MEMO.model_dump_json())

    composed, _slept = _composed(monkeypatch, serve)
    _complete(composed)

    assert len(calls) == 2
    assert calls[0]["messages"] == calls[1]["messages"]
    assert calls[0]["extra_headers"] == {"Idempotency-Key": _IDEMPOTENCY_KEY}
    assert calls[1]["extra_headers"] == {"Idempotency-Key": _IDEMPOTENCY_KEY}


def test_the_base_resp_shape_is_retried_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTTP-200 shape, driven through the real client rather than mocked.

    This is the assertion the unit detector test cannot make: it proves the
    body survives litellm's own structured-output path in a form the detector
    can still see. If ``raw_content`` were ever dropped on the way through,
    this fails and the unit test above does not.
    """
    bodies: list[str] = []

    def serve(**_params: object) -> MagicMock:
        bodies.append("x")
        if len(bodies) <= 2:
            # Two crossings: the shared client re-issues once on a parse
            # failure before the ladder gives up, so ONE rate-limited attempt
            # is two calls here.
            return _provider_response(_BASE_RESP_BODY)
        return _provider_response(_MEMO.model_dump_json())

    composed, slept = _composed(monkeypatch, serve)
    result = _complete(composed)

    assert result.value == _MEMO
    assert len(slept) == 1


def test_the_budget_bounds_the_total_sleep_and_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permanent limit spends the budget ONCE and reports it, not forever.

    The assertion that matters is the SUM, not the call count: a per-request
    count would let the same total be spent again by the next call in the same
    attempt, which against a shared key is the amplification this is sized to
    prevent.
    """

    def serve(**_params: object) -> MagicMock:
        raise RateLimitError("429 Too Many Requests")

    composed, slept = _composed(monkeypatch, serve)
    with pytest.raises(OpenWorldProviderRateLimitedError) as raised:
        _complete(composed)

    assert sum(slept) == pytest.approx(_RATE_LIMIT_RETRY_BUDGET_SECONDS)
    assert composed.rate_limit_budget_remaining == pytest.approx(0.0)
    # The cause is the LADDER's wrapper, not the provider class -- litellm
    # re-wraps before this code ever sees it. What must survive is the
    # EVIDENCE, so that is what is asserted; asserting the raw class here would
    # be asserting something this layer never receives.
    cause = raised.value.__cause__
    assert isinstance(cause, LiteLLMClientError)
    # A permanent limit opens the provider circuit after repeated 429s, so the
    # last crossing is the circuit's refusal: still rate-limit evidence, and
    # still inside the one shared budget.
    assert rate_limit_evidence(cause) == "circuit_open"
    assert max(slept) <= _RATE_LIMIT_BACKOFF_CAP_SECONDS


def test_the_budget_is_shared_across_every_crossing_of_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One composed client is one attempt, and one budget covers all its calls.

    This is what makes it a budget rather than a per-request count. A pipeline
    rate-limited at discovery AND again at held-out spends ONE budget between
    them; resetting per call would silently double the aggregate every attempt
    aims at the shared quota.
    """

    def serve(**_params: object) -> MagicMock:
        raise RateLimitError("429 Too Many Requests")

    composed, slept = _composed(monkeypatch, serve)
    with pytest.raises(OpenWorldProviderRateLimitedError):
        _complete(composed)
    first_call_sleep = sum(slept)

    with pytest.raises(OpenWorldProviderRateLimitedError):
        _complete(composed)

    assert first_call_sleep == pytest.approx(_RATE_LIMIT_RETRY_BUDGET_SECONDS)
    assert sum(slept) == pytest.approx(_RATE_LIMIT_RETRY_BUDGET_SECONDS)


def test_the_backoff_is_fully_jittered_and_exponential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FULL jitter -- ``uniform(0, window)`` -- not ``window`` and not a half.

    Every deployment sharing one provider key meets the rolling window at the
    same instant. An unjittered backoff has them all retry at the same instant
    too, reproducing the burst that caused the limit. The low bound of the draw
    is therefore asserted to be ZERO, which is the whole difference between
    full and "equal" jitter.
    """
    draws: list[tuple[float, float]] = []

    def serve(**_params: object) -> MagicMock:
        raise RateLimitError("429 Too Many Requests")

    composed = compose_litellm_open_world_model_client(
        LiteLLMClient(LiteLLMConfig(model=_MODEL))
    )
    monkeypatch.setattr(litellm, "completion", serve)
    composed._sleep = lambda _seconds: None

    def jitter(low: float, high: float) -> float:
        draws.append((low, high))
        return high

    composed._jitter = jitter
    with pytest.raises(OpenWorldProviderRateLimitedError):
        _complete(composed)

    assert draws, "no backoff was ever computed"
    assert all(low == 0.0 for low, _high in draws)
    # Exponential until the cap, then flat at it.
    assert [high for _low, high in draws[:6]] == [
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        _RATE_LIMIT_BACKOFF_CAP_SECONDS,
    ]


def test_a_non_rate_limit_failure_is_reraised_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop must not swallow the failure it was not built for."""

    def serve(**_params: object) -> MagicMock:
        return _provider_response("not json at all")

    composed, slept = _composed(monkeypatch, serve)
    with pytest.raises(LiteLLMClientError):
        _complete(composed)
    assert slept == []
    assert composed.rate_limit_budget_remaining == _RATE_LIMIT_RETRY_BUDGET_SECONDS


def test_the_transport_records_the_shape_and_never_the_body(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The detector reads the provider body; nothing may REPEAT it.

    ``diagnostics.py`` keeps a hard content-free guarantee on everything this
    subsystem records, and the application-shape detector is the one place that
    touches a raw body at all. It may decide on it; it may not carry it
    anywhere -- so what survives is the arm that fired, and nothing else.

    The transport RECORDS rather than logs. ``open_world/`` may not import
    ``offline_tuner.config``, whose ``TUNER_OUTCOME_LOG_LEVEL`` every tuner
    line must go through, so the host's composition root does the emitting.
    """
    bodies: list[str] = []

    def serve(**_params: object) -> MagicMock:
        bodies.append("x")
        if len(bodies) <= 2:
            return _provider_response(_BASE_RESP_BODY)
        return _provider_response(_MEMO.model_dump_json())

    composed, _slept = _composed(monkeypatch, serve)
    with caplog.at_level(logging.DEBUG):
        _complete(composed)

    assert composed.absorbed_rate_limits == ("base_resp",)
    # The body reached the detector and stopped there: not in the record, and
    # not in any line this call produced.
    everything_logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "rate limit, please retry later" not in everything_logged
    assert "status_code" not in everything_logged
