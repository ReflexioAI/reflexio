"""A provider outage leaves an open-world attempt retry-eligible, not day-spending.

Production (org 133, jobs 4132-4136): compaction ran through a plain client,
the provider's breaker refused it, and the bare ``ProviderCircuitOpenError``
reached the runner's failure boundary, which labelled every attempt
``terminal_failure_infrastructure_failure`` and spent the playbook's day. The
boundary now classifies with ``classify_llm_failure``, the rule durable
extraction uses, whichever stage the failure came from.

The original policy spent the day for a transient failure when its route's
breaker remained closed. Production timeouts and HTTP 529s showed that this
also strands recoverable attempts. They now use the same bounded sweep retry
path; the failed job and its consumed/reserved invocations remain intact.
"""

from __future__ import annotations

from queue import Empty

import pytest

from reflexio.server.llm._litellm_types import (
    LiteLLMClientError,
    LLMHardTimeoutError,
    ProviderRequestGuardError,
    StructuredOutputRepairError,
)
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


def test_a_provider_failure_while_the_breaker_is_closed_is_retry_eligible(
    runtime,
) -> None:
    """One transient call may recover even without a route-wide outage."""
    runner, _transport, _storage, kwargs = runtime
    failure = LiteLLMClientError("API call failed: rate limited")
    failure.circuit_route = _ROUTE
    failure.__cause__ = TimeoutError("provider timed out")

    result = runner.run(**{**kwargs, "evidence": _fails_with(failure)})

    assert result.rate_limited is True
    assert result.reason == "infrastructure_failure"


def _wrapped_transient(kind):
    if kind == "hard_timeout":
        inner = LLMHardTimeoutError("LLM request exceeded hard timeout of 120s")
        inner.__cause__ = Empty()
    else:
        inner = LiteLLMClientError(
            "isolated provider failure",
            upstream_error_type="APIError",
            upstream_status_code=529,
        )
    outer = LiteLLMClientError("fallback ladder exhausted")
    outer.__cause__ = inner
    outer.circuit_route = _ROUTE
    return outer


@pytest.mark.parametrize("kind", ["hard_timeout", "http_529"])
@pytest.mark.parametrize("stage", ["preparation", "discovery", "candidate"])
def test_transients_leave_failed_job_and_invocations_for_bounded_sweep_retry(
    runtime,
    kind,
    stage,
    monkeypatch,
):
    runner, transport, storage, kwargs = runtime
    error = _wrapped_transient(kind)
    completed = []
    if stage == "preparation":
        kwargs = {**kwargs, "evidence": _fails_with(error)}
    else:
        complete = transport.complete

        def fail_at_selected_call(**request):
            # Candidate failure follows a paid successful discovery. Preserve
            # that exact row as well as the failed call's prepared reservation.
            if stage == "candidate" and not transport.calls:
                return complete(**request)
            completed.extend(
                row for row in storage.rows.values() if row.state == "completed"
            )
            transport.calls.append(request)
            raise error

        monkeypatch.setattr(transport, "complete", fail_at_selected_call)

    result = runner.run(**kwargs)

    assert result.reason == "infrastructure_failure"
    assert result.rate_limited is True
    assert result.output_rejected is False
    assert storage.job.status == storage.job.stage == "failed"
    assert storage.job.terminal_outcome == "infrastructure_failure"
    rows = list(storage.rows.values())
    assert (
        len(transport.calls)
        == {"preparation": 0, "discovery": 1, "candidate": 2}[stage]
    )
    assert [row.state for row in rows] == {
        "preparation": [],
        "discovery": ["prepared"],
        "candidate": ["completed", "prepared"],
    }[stage]
    assert all(row.job_id == storage.job.job_id for row in rows)
    assert all(storage.rows[row.analyst_input_identity] == row for row in completed)


@pytest.mark.parametrize("stage", ["preparation", "invocation"])
@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid evidence"),
        ProviderRequestGuardError("unqualified request"),
        LiteLLMClientError("invalid request", upstream_status_code=400),
        StructuredOutputRepairError("exhausted", failure_kind="parse", model="test"),
    ],
)
def test_permanent_failures_do_not_gain_transport_retries(
    runtime,
    stage,
    error,
    monkeypatch,
):
    runner, transport, storage, kwargs = runtime
    if stage == "preparation":
        kwargs = {**kwargs, "evidence": _fails_with(error)}
    else:

        def fail(**request):
            transport.calls.append(request)
            raise error

        monkeypatch.setattr(transport, "complete", fail)

    result = runner.run(**kwargs)

    assert result.rate_limited is False
    assert result.reason == "infrastructure_failure"
    assert storage.job.status == "failed"
    assert len(transport.calls) == (stage == "invocation")


def test_an_ordinary_failure_with_the_breaker_open_spends_the_day(runtime) -> None:
    runner, _transport, _storage, kwargs = runtime
    _open_breaker()

    result = runner.run(**{**kwargs, "evidence": _fails_with(ValueError("bug"))})

    assert result.rate_limited is False
