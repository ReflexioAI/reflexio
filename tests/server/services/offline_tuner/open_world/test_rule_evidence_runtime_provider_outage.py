"""A provider outage leaves an open-world attempt retry-eligible, not day-spending.

Production (org 133, jobs 4132-4136): compaction ran through a plain client,
the provider's breaker refused it, and the bare ``ProviderCircuitOpenError``
reached the runner's failure boundary, which labelled every attempt
``terminal_failure_infrastructure_failure`` and spent the playbook's day. The
boundary now classifies with ``classify_llm_failure``, the rule durable
extraction uses, whichever stage the failure came from.
"""

from __future__ import annotations

from reflexio.server.llm._litellm_types import LiteLLMClientError
from reflexio.server.llm.provider_circuit import (
    OPEN_AFTER_FAILURES,
    ProviderCircuitOpenError,
    provider_circuit,
)
from tests.server.services.offline_tuner.open_world.test_rule_evidence_native import (
    source as source,
)
from tests.server.services.offline_tuner.open_world.test_runner import (
    runtime as runtime,
)

_ROUTE = "minimax/MiniMax-M3||-"


def _open_breaker() -> None:
    for _ in range(OPEN_AFTER_FAILURES):
        provider_circuit.admit(_ROUTE, model="minimax/MiniMax-M3").failed()


def _fails_with(exc: BaseException):
    def load(before_call):
        before_call()
        raise exc

    return load


def test_a_breaker_refusal_during_preparation_is_retry_eligible(runtime) -> None:
    runner, _transport, storage, kwargs = runtime
    _open_breaker()
    refused = ProviderCircuitOpenError("Provider circuit open; request not sent")
    refused.circuit_route = _ROUTE

    result = runner.run(**{**kwargs, "evidence": _fails_with(refused)})

    assert result.rate_limited is True
    # The persisted vocabulary is unchanged; only the sweep label differs.
    assert result.reason == "infrastructure_failure"
    assert storage.job.status == "failed"


def test_a_provider_failure_while_the_breaker_is_open_is_retry_eligible(
    runtime,
) -> None:
    runner, _transport, _storage, kwargs = runtime
    _open_breaker()
    failure = LiteLLMClientError("API call failed: rate limited")
    failure.circuit_route = _ROUTE
    failure.__cause__ = TimeoutError("provider timed out")

    result = runner.run(**{**kwargs, "evidence": _fails_with(failure)})

    assert result.rate_limited is True


def test_a_provider_failure_while_the_breaker_is_closed_spends_the_day(
    runtime,
) -> None:
    """Other calls succeed on that route, so this attempt owns its failure."""
    runner, _transport, _storage, kwargs = runtime
    failure = LiteLLMClientError("API call failed: rate limited")
    failure.circuit_route = _ROUTE
    failure.__cause__ = TimeoutError("provider timed out")

    result = runner.run(**{**kwargs, "evidence": _fails_with(failure)})

    assert result.rate_limited is False
    assert result.reason == "infrastructure_failure"


def test_an_ordinary_failure_with_the_breaker_open_spends_the_day(runtime) -> None:
    runner, _transport, _storage, kwargs = runtime
    _open_breaker()

    result = runner.run(**{**kwargs, "evidence": _fails_with(ValueError("bug"))})

    assert result.rate_limited is False
