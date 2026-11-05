"""The provider breaker tells "the provider is down" from "this item fails".

Durable work charges a failure only when the provider is serving other calls,
so the breaker's state is what decides whether an outage can park work.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, Mock

import pytest

from reflexio.server.llm import provider_circuit as circuit_module
from reflexio.server.llm._litellm_types import LiteLLMClientError
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.provider_circuit import (
    FIRST_OPEN_SECONDS,
    MAX_OPEN_SECONDS,
    OPEN_AFTER_FAILURES,
    RECOVERED_SIGNAL,
    UNAVAILABLE_SIGNAL,
    ProviderCircuitBreaker,
    ProviderCircuitOpenError,
    circuit_route,
    provider_circuit,
    provider_wait_seconds,
)

MODEL = "test/model"
ROUTE = circuit_route({"model": MODEL})


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def breaker(clock: _Clock) -> ProviderCircuitBreaker:
    return ProviderCircuitBreaker(clock=clock, jitter=lambda: 0.0)


@pytest.fixture
def anomalies(monkeypatch: pytest.MonkeyPatch) -> Mock:
    captured = Mock()
    monkeypatch.setattr(circuit_module, "capture_anomaly", captured)
    return captured


def _fail(breaker: ProviderCircuitBreaker, times: int = 1) -> None:
    for _ in range(times):
        breaker.admit(ROUTE, model=MODEL).failed()


def test_opens_only_after_consecutive_failures(breaker):
    _fail(breaker, OPEN_AFTER_FAILURES - 1)
    breaker.admit(ROUTE, model=MODEL).succeeded()  # a success resets the streak
    _fail(breaker, OPEN_AFTER_FAILURES - 1)
    assert breaker.outage_retry_delay(ROUTE) is None
    breaker.admit(ROUTE, model=MODEL).close()  # no verdict leaves the streak as it was
    _fail(breaker)
    assert breaker.outage_retry_delay(ROUTE) == FIRST_OPEN_SECONDS
    with pytest.raises(ProviderCircuitOpenError):
        breaker.admit(ROUTE, model=MODEL)


def test_one_probe_after_the_open_period_and_backoff_doubles_to_a_cap(breaker, clock):
    _fail(breaker, OPEN_AFTER_FAILURES)
    expected = FIRST_OPEN_SECONDS
    for _ in range(8):
        clock.now += expected - 1
        with pytest.raises(ProviderCircuitOpenError):
            breaker.admit(ROUTE, model=MODEL)
        clock.now += 1
        probe = breaker.admit(ROUTE, model=MODEL)
        with pytest.raises(ProviderCircuitOpenError):
            breaker.admit(ROUTE, model=MODEL)  # only one probe at a time
        probe.failed()
        expected = min(expected * 2, MAX_OPEN_SECONDS)
        assert breaker.outage_retry_delay(ROUTE) == expected
    assert expected == MAX_OPEN_SECONDS


def test_a_successful_probe_closes_and_resets_the_backoff(breaker, clock):
    _fail(breaker, OPEN_AFTER_FAILURES)
    clock.now += FIRST_OPEN_SECONDS
    breaker.admit(ROUTE, model=MODEL).failed()
    clock.now += 2 * FIRST_OPEN_SECONDS
    breaker.admit(ROUTE, model=MODEL).succeeded()
    assert breaker.outage_retry_delay(ROUTE) is None
    _fail(breaker, OPEN_AFTER_FAILURES)
    assert breaker.outage_retry_delay(ROUTE) == FIRST_OPEN_SECONDS


def test_a_probe_without_a_verdict_frees_the_next_probe(breaker, clock):
    _fail(breaker, OPEN_AFTER_FAILURES)
    clock.now += FIRST_OPEN_SECONDS
    breaker.admit(ROUTE, model=MODEL).close()
    breaker.admit(ROUTE, model=MODEL).succeeded()
    assert breaker.outage_retry_delay(ROUTE) is None


def test_a_probe_that_never_reports_back_goes_stale(breaker, clock):
    _fail(breaker, OPEN_AFTER_FAILURES)
    clock.now += FIRST_OPEN_SECONDS
    breaker.admit(ROUTE, model=MODEL)  # caller vanished
    with pytest.raises(ProviderCircuitOpenError):
        breaker.admit(ROUTE, model=MODEL)
    clock.now += circuit_module._PROBE_STALE_SECONDS
    breaker.admit(ROUTE, model=MODEL).succeeded()
    assert breaker.outage_retry_delay(ROUTE) is None


def test_a_sustained_outage_is_reported_once_and_so_is_its_recovery(
    breaker, clock, anomalies
):
    _fail(breaker, OPEN_AFTER_FAILURES)
    while clock.now < 1000 + 3 * circuit_module.ANNOUNCE_AFTER_SECONDS:
        clock.now += 60
        with contextlib.suppress(ProviderCircuitOpenError):
            breaker.admit(ROUTE, model=MODEL).failed()
    assert [c.args[0] for c in anomalies.call_args_list] == [UNAVAILABLE_SIGNAL]
    assert anomalies.call_args.kwargs["signal"] == UNAVAILABLE_SIGNAL
    assert anomalies.call_args.kwargs["release_blocking"] == "false"
    clock.now += MAX_OPEN_SECONDS
    breaker.admit(ROUTE, model=MODEL).succeeded()
    assert [c.args[0] for c in anomalies.call_args_list] == [
        UNAVAILABLE_SIGNAL,
        RECOVERED_SIGNAL,
    ]


def test_a_short_outage_reports_nothing(breaker, clock, anomalies):
    _fail(breaker, OPEN_AFTER_FAILURES)
    clock.now += FIRST_OPEN_SECONDS
    breaker.admit(ROUTE, model=MODEL).succeeded()
    anomalies.assert_not_called()


def test_waiting_items_spread_over_part_of_the_open_period(clock):
    breaker = ProviderCircuitBreaker(clock=clock, jitter=lambda: 1.0)
    _fail(breaker, OPEN_AFTER_FAILURES)
    clock.now += 20
    assert breaker.outage_retry_delay(ROUTE) == (FIRST_OPEN_SECONDS - 20) + (
        FIRST_OPEN_SECONDS / 2
    )


def _routed(error: BaseException, route: str = ROUTE) -> LiteLLMClientError:
    """The shape a durable worker receives: the client's wrapper, routed."""
    try:
        raise LiteLLMClientError(f"API call failed: {error}") from error
    except LiteLLMClientError as failure:
        failure.circuit_route = route
        return failure


@pytest.mark.parametrize(
    ("error", "outage", "expected"),
    [
        (_routed(TimeoutError("slow")), True, "wait"),
        (_routed(TimeoutError("slow")), False, None),  # serves others: charged
        (ValueError("bad output"), True, None),  # not a provider failure
        (TimeoutError("no route"), True, None),  # cannot name its breaker
        (_routed(ProviderCircuitOpenError("refused")), False, 0.0),  # never sent
    ],
)
def test_provider_wait_seconds(error, outage, expected):
    if outage:
        for _ in range(OPEN_AFTER_FAILURES):
            provider_circuit.admit(ROUTE, model=MODEL).failed()
    # An unrelated error alone is never a provider wait.
    assert provider_wait_seconds(RuntimeError("wrapped")) is None
    try:
        raise RuntimeError("extractor failed") from error
    except RuntimeError as wrapped:
        wait = provider_wait_seconds(wrapped)
    if expected == "wait":
        assert wait is not None and wait > 0
    else:
        assert wait == expected


def test_routes_of_one_model_do_not_share_an_outage():
    other = circuit_route({"model": MODEL, "api_base": "https://other.invalid/v1"})
    based = circuit_route({"model": MODEL, "base_url": "https://based.invalid/v1"})
    keyed = circuit_route({"model": MODEL, "api_key": "secret-a"})
    assert len({ROUTE, other, based, keyed}) == 4
    assert "secret-a" not in keyed
    for _ in range(OPEN_AFTER_FAILURES):
        provider_circuit.admit(ROUTE, model=MODEL).failed()
    # Another endpoint or account of the same model is still admitted ...
    provider_circuit.admit(other, model=MODEL).succeeded()
    provider_circuit.admit(keyed, model=MODEL).succeeded()
    # ... and a failure there is charged although this route is down.
    assert provider_wait_seconds(_routed(TimeoutError("slow"), other)) is None
    assert provider_wait_seconds(_routed(TimeoutError("slow"))) is not None


def _response() -> MagicMock:
    choice = MagicMock()
    choice.message.content = "ok"
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    resp.usage.prompt_tokens_details = None
    resp.usage.cache_creation_input_tokens = None
    resp.usage.cache_read_input_tokens = None
    return resp


def test_client_opens_on_timeouts_then_refuses_without_calling_the_provider(
    monkeypatch,
):
    sink = Mock()
    monkeypatch.setattr("reflexio.server.operational_metrics._sink", sink)
    client = LiteLLMClient(LiteLLMConfig(model=MODEL, timeout=120, fallback_models=[]))
    sent: list[str] = []

    def timing_out(params, _hard_timeout):
        sent.append(params["model"])
        raise TimeoutError("provider stalled")

    monkeypatch.setattr(client, "_completion_with_hard_timeout", timing_out)
    for _ in range(OPEN_AFTER_FAILURES + 3):
        with pytest.raises(Exception) as raised:
            client.generate_chat_response([{"role": "user", "content": "hi"}])
    assert len(sent) == OPEN_AFTER_FAILURES
    assert isinstance(raised.value, ProviderCircuitOpenError)
    route = raised.value.circuit_route
    assert route is not None
    names = [call.args[0] for call in sink.record.call_args_list]
    # Provider outages are not failures a person must act on.
    assert names.count("llm.failures") == 0
    outcomes = [
        call.kwargs["attributes"]["outcome"]
        for call in sink.record.call_args_list
        if call.args[0] == "llm.requests"
    ]
    assert outcomes == ["provider_unavailable"] * (OPEN_AFTER_FAILURES + 3)

    # The provider answers again: once the probe gets through, calls flow.
    monkeypatch.setattr(
        client, "_completion_with_hard_timeout", lambda *_args: _response()
    )
    breaker = circuit_module.provider_circuit
    with breaker._lock:
        breaker._circuits[route].open_until = 0.0
    assert client.generate_chat_response([{"role": "user", "content": "hi"}]) == "ok"
    assert breaker.outage_retry_delay(route) is None


@pytest.mark.parametrize(
    ("timeout", "error", "opens"),
    [
        (25, TimeoutError("caller budget spent"), False),
        (120, TimeoutError("provider stalled"), True),
        # Unavailability that is not a timeout counts under any budget.
        (25, ConnectionError("refused"), True),
    ],
)
def test_short_budget_timeouts_are_not_outage_evidence(
    monkeypatch, timeout, error, opens
):
    import litellm

    client = LiteLLMClient(
        LiteLLMConfig(model=MODEL, timeout=timeout, fallback_models=[])
    )
    if isinstance(error, ConnectionError):
        error = litellm.exceptions.APIConnectionError(
            message="refused", llm_provider="test", model=MODEL
        )

    def failing(_params, _hard_timeout):
        raise error

    monkeypatch.setattr(client, "_completion_with_hard_timeout", failing)
    for _ in range(OPEN_AFTER_FAILURES):
        with pytest.raises(LiteLLMClientError) as raised:
            client.generate_chat_response([{"role": "user", "content": "hi"}])
    route = raised.value.circuit_route
    assert route is not None
    assert (provider_circuit.outage_retry_delay(route) is not None) is opens


def test_sustained_provider_cap_saturation_opens_the_breaker(monkeypatch):
    from contextlib import contextmanager

    from reflexio.server.llm import _litellm_text_generation as ttg
    from reflexio.server.llm._provider_concurrency import ProviderCapSaturatedError

    @contextmanager
    def saturated(_model, *, max_wait):
        raise ProviderCapSaturatedError("no permit")
        yield

    monkeypatch.setattr(ttg, "provider_slot", saturated)
    client = LiteLLMClient(LiteLLMConfig(model=MODEL, timeout=120, fallback_models=[]))
    for _ in range(OPEN_AFTER_FAILURES):
        with pytest.raises(LiteLLMClientError) as raised:
            client.generate_chat_response([{"role": "user", "content": "hi"}])
    # The last rung wraps the saturation; durable work finds its route on the cause.
    assert isinstance(raised.value.__cause__, ProviderCapSaturatedError)
    assert provider_wait_seconds(raised.value) is not None


def test_a_search_deadline_expiring_is_not_outage_evidence(monkeypatch):
    import time

    from reflexio.server import search_runtime

    client = LiteLLMClient(LiteLLMConfig(model=MODEL, timeout=120, fallback_models=[]))

    def timing_out(_params, _hard_timeout):
        raise TimeoutError("search budget spent")

    monkeypatch.setattr(client, "_completion_with_hard_timeout", timing_out)
    # The configured timeout is long, but a search scope leaves five seconds.
    scope = search_runtime.SearchScope(deadline=time.monotonic() + 5)
    token = search_runtime._scope.set(scope)
    try:
        for _ in range(OPEN_AFTER_FAILURES):
            with pytest.raises(LiteLLMClientError) as raised:
                client.generate_chat_response([{"role": "user", "content": "hi"}])
    finally:
        search_runtime._scope.reset(token)
    route = raised.value.circuit_route
    assert route is not None
    assert provider_circuit.outage_retry_delay(route) is None
