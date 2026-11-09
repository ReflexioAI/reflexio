"""One verdict for a failed LLM call: what the caller should do about it.

:func:`classify_llm_failure` reads an exception and returns a closed, frozen
:data:`LLMFailureVerdict`:

- :class:`OutageWait` -- the provider is down for everyone on the route the
  call failed on (its breaker is open, or the breaker refused the call). Durable
  work waits ``delay_seconds`` and is not charged (the charging invariant in
  ``provider_circuit``).
- :class:`Transient` -- the provider could not serve this call (timeout,
  connection error, rate limit, overload, a retry-later HTTP status, a breaker
  refusal, or a saturated local provider cap), but nothing says it is down for
  everyone. ``timed_out`` says a timeout was among the evidence: the client
  counts a timeout as outage evidence only when the call was allowed long
  enough. ``cap_saturated`` says the only evidence is a saturated local
  provider cap (``ProviderCapSaturatedError``): no request reached the
  provider, so it is not a transport failure.
- :class:`RepairExhausted` -- the structured-output repair ladder ran out
  (``StructuredOutputRepairError``); ``kind`` is its ``failure_kind``.
- :class:`Truncated` -- the reply was cut off by its output limit
  (``finish_reason == "length"`` on a parse failure), or an unchanged request
  already cut off that way was refused (``OutputTruncatedRequestError``).
- :class:`Terminal` -- anything else: the failure is the item's own.

Precedence, highest first: ``OutageWait``, ``Transient``, ``RepairExhausted``,
``Truncated``, ``Terminal``. Availability wins because a failure the provider
caused says nothing about the item; the repair ladder's final judgement wins
over a cut-off it already tried to repair.

Classification reads exception types, attributes and status codes, never
message text, and makes one walk over the exception chain.

**Chain rule.** The walk follows the links Python's traceback printer shows:
from each exception, its ``__cause__`` when set (``raise ... from exc``), else
its ``__context__`` (the exception being handled when it was raised) unless
``__suppress_context__`` is true (``raise ... from None`` severs the chain on
purpose). The walk stops at a repeated exception (cycle-safe) and after
``MAX_CHAIN_LINKS`` links.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from reflexio.server.llm import provider_circuit as _circuit
from reflexio.server.llm._litellm_types import (
    StructuredOutputParseError,
    StructuredOutputRepairError,
)
from reflexio.server.llm._provider_concurrency import ProviderCapSaturatedError
from reflexio.server.llm.input_budget import OutputTruncatedRequestError

MAX_CHAIN_LINKS = 16

# Expected transient upstream failures, by exception TYPE NAME so the heavy
# ``litellm``/``openai`` hierarchies are not imported here. When one reaches a
# caller the client's fallback ladder is already exhausted, but the caller owns
# fatality, and the client logs these at WARNING so a flaky provider cannot
# flood ERROR-level alerts for handled failures.
TRANSIENT_ERROR_NAMES: frozenset[str] = frozenset(
    {
        "Timeout",
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
    }
)

# Upstream HTTP statuses that mean "retry later", for provider errors LiteLLM
# does not map to one of the classes above. LiteLLM 1.103-1.105 raises a
# MiniMax 529 overload as a generic ``APIError(status_code=529)`` and a 502 as
# ``BadGatewayError``; neither name is listed. 500 is deliberately absent: a
# genuine upstream 500 already arrives as ``InternalServerError``, while LiteLLM
# itself raises ``APIError(status_code=500)`` for a malformed 200 body (e.g.
# MiniMax ``choices: null``) -- a response-shape defect, not a measured outage.
TRANSIENT_UPSTREAM_STATUS_CODES: frozenset[int] = frozenset(
    {408, 429, 502, 503, 504, 529}
)

_TIMEOUT_ERROR_NAMES: frozenset[str] = frozenset({"Timeout", "APITimeoutError"})


@dataclass(frozen=True)
class OutageWait:
    """The route is down for everyone: wait ``delay_seconds``, uncharged."""

    delay_seconds: float


@dataclass(frozen=True)
class Transient:
    """The provider could not serve this call; its breaker does not say down."""

    timed_out: bool
    cap_saturated: bool = False


@dataclass(frozen=True)
class RepairExhausted:
    """The structured-output repair ladder ran out; ``kind`` is why."""

    kind: str


@dataclass(frozen=True)
class Truncated:
    """The reply was cut off by its output limit."""


@dataclass(frozen=True)
class Terminal:
    """The failure is the item's own."""


LLMFailureVerdict = OutageWait | Transient | RepairExhausted | Truncated | Terminal


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and the exceptions it was raised from (module docstring)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while (
        current is not None and id(current) not in seen and len(seen) < MAX_CHAIN_LINKS
    ):
        seen.add(id(current))
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            current = None
        else:
            current = current.__context__


def _is_transport_failure(link: BaseException) -> bool:
    # Subprocess-isolated calls cross a pickle boundary, so the wrapper keeps
    # ``upstream_error_type`` / ``upstream_status_code`` instead of the
    # provider's exception. A breaker refusal sent nothing: the provider
    # recently failed this way.
    if isinstance(link, TimeoutError | _circuit.ProviderCircuitOpenError):
        return True
    if type(link).__name__ in TRANSIENT_ERROR_NAMES:
        return True
    if getattr(link, "upstream_error_type", None) in TRANSIENT_ERROR_NAMES:
        return True
    status = getattr(link, "upstream_status_code", None)
    if status is None:
        status = getattr(link, "status_code", None)
    return isinstance(status, int) and status in TRANSIENT_UPSTREAM_STATUS_CODES


def _is_timeout(link: BaseException) -> bool:
    return (
        isinstance(link, TimeoutError)
        or type(link).__name__ in _TIMEOUT_ERROR_NAMES
        or getattr(link, "upstream_error_type", None) in _TIMEOUT_ERROR_NAMES
    )


def _is_truncated(link: BaseException) -> bool:
    return isinstance(link, OutputTruncatedRequestError) or (
        isinstance(link, StructuredOutputParseError) and link.finish_reason == "length"
    )


def classify_llm_failure(
    exc: BaseException, *, consult_breaker: bool = True
) -> LLMFailureVerdict:
    """Classify a failed LLM call (rules and chain walk: module docstring).

    Args:
        exc (BaseException): The failure, as the caller caught it.
        consult_breaker (bool): Whether an availability failure may become
            :class:`OutageWait`. The client passes ``False`` when deciding
            whether a failure is evidence for the breaker, so the breaker never
            feeds on its own state; every other caller keeps the default.

    Returns:
        LLMFailureVerdict: :class:`OutageWait` only when the failure carries
            its route (``circuit_route``) and that route's breaker is open or
            refused the call; another route's outage never excuses it.
    """
    transport = saturated = timed_out = refused = truncated = False
    route: str | None = None
    repair_kind: str | None = None
    for link in _chain(exc):
        transport |= _is_transport_failure(link)
        saturated |= isinstance(link, ProviderCapSaturatedError)
        timed_out |= _is_timeout(link)
        refused |= isinstance(link, _circuit.ProviderCircuitOpenError)
        truncated |= _is_truncated(link)
        link_route = getattr(link, "circuit_route", None)
        if route is None and isinstance(link_route, str):
            route = link_route
        if repair_kind is None and isinstance(link, StructuredOutputRepairError):
            repair_kind = link.failure_kind
    if transport or saturated:
        if consult_breaker and route is not None:
            delay = _circuit.provider_circuit.outage_retry_delay(route)
            # A refusal sent nothing, so it is never charged, even if the
            # breaker closed since.
            if delay is None and refused:
                delay = 0.0
            if delay is not None:
                return OutageWait(delay)
        return Transient(timed_out=timed_out, cap_saturated=not transport)
    if repair_kind is not None:
        return RepairExhausted(repair_kind)
    if truncated:
        return Truncated()
    return Terminal()
