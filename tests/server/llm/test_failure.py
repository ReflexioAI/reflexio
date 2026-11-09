"""``classify_llm_failure``: every exception shape under every chain shape.

The verdict is the one place a failed LLM call is sorted, so the table below is
the contract: each shape on its own, wrapped explicitly (``raise ... from``),
wrapped implicitly (raised while handling), severed (``raise ... from None``),
at the depth cap, past it, and inside a cycle.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from litellm.exceptions import ContextWindowExceededError

from reflexio.server.llm._litellm_types import (
    LiteLLMClientError,
    LLMHardTimeoutError,
    StructuredOutputParseError,
    StructuredOutputRepairError,
)
from reflexio.server.llm._provider_concurrency import ProviderCapSaturatedError
from reflexio.server.llm.compute_admission import ComputeInterruptedError
from reflexio.server.llm.failure import (
    MAX_CHAIN_LINKS,
    LLMFailureVerdict,
    OutageWait,
    RepairExhausted,
    Terminal,
    Transient,
    Truncated,
    classify_llm_failure,
    exception_chain,
)
from reflexio.server.llm.input_budget import (
    OutputTruncatedRequestError,
    ProviderInputOverflowError,
)
from reflexio.server.llm.provider_circuit import (
    OPEN_AFTER_FAILURES,
    ProviderCircuitOpenError,
    circuit_route,
    provider_circuit,
)
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowSuspendedError,
)
from reflexio.server.services.storage.error import ReadinessUnavailableError

ROUTE = circuit_route({"model": "test/verdict"})


def _named(name: str, **attrs: object) -> Exception:
    """An exception whose class carries a LiteLLM/OpenAI type name."""
    error = type(name, (Exception,), {})(name)
    for key, value in attrs.items():
        setattr(error, key, value)
    return error


def _status(code: int) -> Exception:
    return _named("APIError", status_code=code)


def _parse(finish_reason: str | None, *, empty: bool = False) -> Exception:
    return StructuredOutputParseError(
        "bad", finish_reason=finish_reason, empty_response=empty
    )


def _repair(kind: str) -> Exception:
    return StructuredOutputRepairError("exhausted", failure_kind=kind, model="m")  # type: ignore[arg-type]


SLOW, DOWN = Transient(timed_out=True), Transient(timed_out=False)


def _iso(**attrs: object) -> Exception:
    """A subprocess-isolated failure: the wrapper keeps the upstream identity."""
    return LiteLLMClientError("isolated", **attrs)  # type: ignore[arg-type]


#: (id, factory, verdict on its own with no route). Every HTTP status is
#: covered by ``test_transient_upstream_status.py``.
SHAPES: list[tuple[str, Callable[[], BaseException], LLMFailureVerdict]] = [
    ("timeout", lambda: TimeoutError("slow"), SLOW),
    ("hard-timeout", lambda: LLMHardTimeoutError("killed"), SLOW),
    ("litellm-timeout", lambda: _named("Timeout"), SLOW),
    ("openai-timeout", lambda: _named("APITimeoutError"), SLOW),
    ("connection", lambda: _named("APIConnectionError"), DOWN),
    ("rate-limit", lambda: _named("RateLimitError"), DOWN),
    ("internal", lambda: _named("InternalServerError"), DOWN),
    ("unavailable", lambda: _named("ServiceUnavailableError"), DOWN),
    ("status-529", lambda: _status(529), DOWN),
    ("status-500", lambda: _status(500), Terminal()),
    ("status-str", lambda: _named("APIError", status_code="529"), Terminal()),
    ("isolated-timeout", lambda: _iso(upstream_error_type="APITimeoutError"), SLOW),
    ("isolated-rate-limit", lambda: _iso(upstream_error_type="RateLimitError"), DOWN),
    ("isolated-529", lambda: _iso(upstream_status_code=529), DOWN),
    ("isolated-400", lambda: _iso(upstream_status_code=400), Terminal()),
    (
        "upstream-status-wins",
        lambda: _named("APIError", upstream_status_code=400, status_code=529),
        Terminal(),
    ),
    ("refusal", lambda: ProviderCircuitOpenError("refused"), DOWN),
    ("cap-saturated", lambda: ProviderCapSaturatedError("no permit"), OutageWait(0.0)),
    ("repair-parse", lambda: _repair("parse"), RepairExhausted("parse")),
    ("repair-semantic", lambda: _repair("semantic"), RepairExhausted("semantic")),
    ("repair-refusal", lambda: _repair("refusal"), RepairExhausted("refusal")),
    ("cut-off-body", lambda: _parse("length"), Truncated()),
    ("reasoning-exhausted", lambda: _parse("length", empty=True), Truncated()),
    ("parse-stop", lambda: _parse("stop"), Terminal()),
    ("parse-empty", lambda: _parse(None, empty=True), Terminal()),
    ("output-truncated", lambda: OutputTruncatedRequestError("same"), Truncated()),
    ("context-window", lambda: _named("ContextWindowExceededError"), Terminal()),
    ("client-error", lambda: LiteLLMClientError("API call failed"), Terminal()),
    ("value-error", lambda: ValueError("bad output"), Terminal()),
]


def _explicit(inner: BaseException) -> BaseException:
    try:
        raise RuntimeError("wrapper") from inner
    except RuntimeError as wrapper:
        return wrapper


def _implicit_context(inner: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException:
            raise RuntimeError("raised while handling")  # noqa: B904
    except RuntimeError as wrapper:
        return wrapper


def _severed(inner: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException:
            raise RuntimeError("severed") from None
    except RuntimeError as wrapper:
        return wrapper


def _buried(inner: BaseException, links: int) -> BaseException:
    """``inner`` as the ``links``-th link of an explicit chain."""
    current = inner
    for index in range(links - 1):
        outer = RuntimeError(f"wrapper {index}")
        outer.__cause__ = current
        current = outer
    return current


def _cycle(inner: BaseException) -> BaseException:
    """``entry`` -> ``inner`` -> ``entry`` -> ...: the walk sees each once."""
    entry = RuntimeError("entry")
    entry.__cause__ = inner
    inner.__cause__ = entry
    return entry


#: (id, chain builder, whether ``inner``'s verdict survives the chain).
CHAINS: list[tuple[str, Callable[[BaseException], BaseException], bool]] = [
    ("bare", lambda inner: inner, True),
    ("explicit-cause", _explicit, True),
    ("implicit-context", _implicit_context, True),
    ("explicit-twice", lambda inner: _explicit(_explicit(inner)), True),
    ("from-none", _severed, False),
    ("explicit-over-from-none", lambda inner: _explicit(_severed(inner)), False),
    ("at-depth-cap", lambda inner: _buried(inner, MAX_CHAIN_LINKS), True),
    ("past-depth-cap", lambda inner: _buried(inner, MAX_CHAIN_LINKS + 1), False),
    ("cycle", _cycle, True),
]


@pytest.mark.parametrize(("chain_id", "chain", "survives"), CHAINS)
@pytest.mark.parametrize(("shape_id", "factory", "verdict"), SHAPES)
def test_verdict_by_shape_and_chain(
    shape_id, factory, verdict, chain_id, chain, survives
):
    expected = verdict if survives else Terminal()
    assert classify_llm_failure(chain(factory())) == expected


def test_a_self_cycle_terminates():
    loop = RuntimeError("loop")
    loop.__cause__ = loop
    assert classify_llm_failure(loop) == Terminal()


def test_a_cause_hides_the_context():
    """``raise W from X`` while handling Y: the printer shows X, not Y."""
    try:
        try:
            raise TimeoutError("handled")
        except TimeoutError:
            raise RuntimeError("wrapper") from ValueError("the real cause")
    except RuntimeError as wrapper:
        assert classify_llm_failure(wrapper) == Terminal()


@pytest.mark.parametrize(
    ("outer", "inner", "expected"),
    [
        # Availability outranks everything else in the chain.
        (_repair("parse"), TimeoutError("repair turn"), Transient(timed_out=True)),
        (_parse("length"), _named("RateLimitError"), Transient(timed_out=False)),
        # The repair ladder's final judgement outranks a cut-off it repaired.
        (_repair("semantic"), _parse("length"), RepairExhausted("semantic")),
        (_parse("stop"), OutputTruncatedRequestError("same"), Truncated()),
        # A saturated local cap waits uncharged even beside a transport failure.
        (
            ProviderCapSaturatedError("no permit"),
            TimeoutError("earlier"),
            OutageWait(0.0),
        ),
        # The first repair error in the chain names the kind.
        (_repair("refusal"), _repair("parse"), RepairExhausted("refusal")),
    ],
)
def test_precedence(outer, inner, expected):
    outer.__cause__ = inner
    assert classify_llm_failure(outer) == expected


def _routed(error: BaseException, route: str = ROUTE) -> BaseException:
    error.circuit_route = route  # type: ignore[attr-defined]
    return error


def _open(route: str = ROUTE) -> None:
    for _ in range(OPEN_AFTER_FAILURES):
        provider_circuit.admit(route, model="test/verdict").failed()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TimeoutError("slow"),
        lambda: _named("RateLimitError"),
        lambda: ProviderCapSaturatedError("no permit"),
        lambda: ProviderCircuitOpenError("refused"),
    ],
)
def test_an_open_route_turns_availability_into_an_outage_wait(factory):
    _open()
    verdict = classify_llm_failure(_explicit(_routed(LiteLLMClientError("x"))))
    assert verdict == Terminal()  # a route alone is not an availability failure
    failure = _routed(LiteLLMClientError("API call failed"))
    failure.__cause__ = factory()
    verdict = classify_llm_failure(_explicit(failure))
    assert isinstance(verdict, OutageWait) and verdict.delay_seconds > 0


def test_the_route_is_found_on_any_link():
    _open()
    inner = _routed(TimeoutError("slow"))
    assert isinstance(classify_llm_failure(_explicit(inner)), OutageWait)


def test_a_route_beyond_from_none_is_not_found():
    _open()
    assert classify_llm_failure(_severed(_routed(TimeoutError("slow")))) == Terminal()


def test_a_closed_route_charges_the_item():
    verdict = classify_llm_failure(_explicit(_routed(TimeoutError("slow"))))
    assert verdict == Transient(timed_out=True)


def test_another_routes_outage_never_excuses_the_item():
    other = circuit_route({"model": "test/verdict", "api_base": "https://o.invalid"})
    _open(other)
    verdict = classify_llm_failure(_routed(TimeoutError("slow")))
    assert verdict == Transient(timed_out=True)


def test_a_refusal_waits_uncharged_after_the_breaker_closed():
    refused = _routed(ProviderCircuitOpenError("refused"))
    assert classify_llm_failure(_explicit(refused)) == OutageWait(0.0)


def test_an_unrouted_failure_never_waits():
    _open()
    assert classify_llm_failure(TimeoutError("slow")) == Transient(timed_out=True)


def test_a_saturated_cap_always_waits_uncharged():
    """Decision 3A: our own cap was full, so nothing was sent and no attempt is
    spent, whether or not the route's breaker is open or consulted."""
    saturated = _explicit(_routed(ProviderCapSaturatedError("no permit")))
    assert classify_llm_failure(saturated) == OutageWait(0.0)
    assert classify_llm_failure(saturated, consult_breaker=False) == OutageWait(0.0)
    assert classify_llm_failure(ProviderCapSaturatedError("unrouted")) == OutageWait(
        0.0
    )
    _open()
    waited = classify_llm_failure(saturated)
    assert isinstance(waited, OutageWait) and waited.delay_seconds > 0
    assert classify_llm_failure(saturated, consult_breaker=False) == OutageWait(0.0)


def test_exception_chain_is_the_walk_the_verdict_uses():
    inner = TimeoutError("slow")
    outer = _explicit(inner)
    assert list(exception_chain(outer)) == [outer, inner]
    assert list(exception_chain(_severed(inner)))[1:] == []
    assert len(list(exception_chain(_buried(inner, MAX_CHAIN_LINKS + 5)))) == (
        MAX_CHAIN_LINKS
    )


def test_breaker_evidence_never_reads_the_breaker():
    _open()
    refused = _routed(ProviderCircuitOpenError("refused"))
    assert classify_llm_failure(refused, consult_breaker=False) == Transient(False)
    slow = _routed(TimeoutError("slow"))
    assert classify_llm_failure(slow, consult_breaker=False) == Transient(True)


def _raised_from_none(
    make: Callable[[], BaseException], context: BaseException
) -> BaseException:
    try:
        try:
            raise context
        except BaseException:
            raise make() from None
    except BaseException as raised:
        return raised


@pytest.mark.parametrize(
    ("site", "make", "context"),
    [
        (
            "client: provider input overflow",
            lambda: ProviderInputOverflowError({"model": "m"}),
            ContextWindowExceededError("too long", model="m", llm_provider="openai"),
        ),
        (
            "compute admission: interrupted computation",
            lambda: ComputeInterruptedError("interrupted"),
            ReadinessUnavailableError("not ready"),
        ),
        (
            "window executor: overflow suspended",
            lambda: OverflowSuspendedError("frozen smaller plan"),
            ProviderInputOverflowError({"model": "m"}),
        ),
    ],
)
def test_from_none_sites_on_moved_paths_classify_as_before(site, make, context):
    """The old walks followed ``__context__`` past ``from None``; this one stops.

    Each ``raise ... from None`` that can reach a moved caller severs a context
    that is itself Terminal, so following it (the old walk) and stopping (the
    traceback rule) give the same verdict.
    """
    _open()
    raised = _raised_from_none(make, _routed(context))
    assert raised.__suppress_context__ and raised.__context__ is not None
    assert classify_llm_failure(raised.__context__) == Terminal()
    assert classify_llm_failure(raised) == Terminal()


def _wrapped(inner: BaseException, *, route: str | None = None) -> BaseException:
    """The walker's shape: a client error raised from the rung's failure."""
    try:
        raise LiteLLMClientError("API call failed") from inner
    except LiteLLMClientError as failure:
        if route is not None:
            failure.circuit_route = route
        return failure


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (_wrapped(TimeoutError("slow"), route=ROUTE), "transient"),
        (_wrapped(ProviderCircuitOpenError("refused"), route=ROUTE), "transient"),
        # Decision 3A: a saturated local cap is an expected wait, labelled
        # transient like any other availability failure (was non_transient).
        (_wrapped(ProviderCapSaturatedError("no permit"), route=ROUTE), "transient"),
        (ProviderCapSaturatedError("no permit"), "transient"),
        (_explicit(_repair("parse")), "non_transient"),
        (_severed(TimeoutError("severed")), "non_transient"),
    ],
)
def test_tool_loop_label_reads_the_verdict_not_the_breaker(error, detail):
    from reflexio.server.llm.tools import describe_tool_loop_error

    _open()
    assert describe_tool_loop_error(error) == (type(error).__name__, detail)


def test_tool_loop_label_keeps_the_repair_kind_over_availability():
    repair = _repair("semantic")
    repair.__cause__ = TimeoutError("repair turn")
    from reflexio.server.llm.tools import describe_tool_loop_error

    assert describe_tool_loop_error(repair) == (
        "StructuredOutputRepairError",
        "semantic",
    )
