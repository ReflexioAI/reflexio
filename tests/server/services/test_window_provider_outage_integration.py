"""A provider outage of any length never parks a window; a window's own failure still does.

Invariant: a window failure is charged against ``MAX_EXTRACTION_WINDOW_ATTEMPTS``
only when the provider was serving other calls (every breaker closed). While a
breaker is open, a provider-availability failure waits for the provider
without spending an attempt, so the published interactions are processed once
it recovers.

Before this, every failure was charged, so a provider outage longer than about
18 minutes parked every active cursor (production, 2026-10-31: 937 windows
parked by a usage-limit outage and held until an operator re-armed them).
"""

from __future__ import annotations

import pytest

from reflexio.server.llm._litellm_types import LiteLLMClientError
from reflexio.server.llm.provider_circuit import (
    OPEN_AFTER_FAILURES,
    circuit_route,
    provider_circuit,
)
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.storage_base._extraction_stream import (
    MAX_EXTRACTION_WINDOW_ATTEMPTS,
    PROVIDER_WAIT_PREFIX,
    RETRY_EXHAUSTED_PREFIX,
)
from tests.server.services.test_durable_window_pipeline import (
    payload,
)
from tests.server.services.test_durable_window_pipeline import (
    pipeline as pipeline,
)
from tests.server.services.test_window_retry_ceiling_integration import (
    _cursor,
    _drain_once,
)

ROUTE = circuit_route({"model": "test/model"})


def _provider_times_out(*_args: object) -> bool:
    # The shape the client raises: its wrapper, naming the route that failed.
    failure = LiteLLMClientError("API call failed: provider request timed out")
    failure.circuit_route = ROUTE
    raise failure from TimeoutError("provider request timed out")


def _open_breaker() -> None:
    for _ in range(OPEN_AFTER_FAILURES):
        provider_circuit.admit(ROUTE, model="test/model").failed()
    assert provider_circuit.outage_retry_delay(ROUTE) is not None


def test_outage_waits_without_charging_and_completes_after_recovery(
    pipeline, monkeypatch
):
    engine, worker, client = pipeline
    assert client.post("/api/publish_interaction", json=payload("outage")).json()[
        "success"
    ]
    storage = engine.get_storage()
    calls: list[int] = []

    def gate(*args: object) -> bool:
        calls.append(1)
        return _provider_times_out(*args)

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    _open_breaker()

    # Three times the ceiling: an outage this long parked the window before.
    for _ in range(3 * MAX_EXTRACTION_WINDOW_ATTEMPTS):
        _drain_once(worker, storage)

    assert len(calls) == 3 * MAX_EXTRACTION_WINDOW_ATTEMPTS
    cursor = _cursor(storage)
    assert cursor["attempts"] == 0
    assert cursor["retry_at"] != float("inf")
    assert cursor["last_error"].startswith(f"{PROVIDER_WAIT_PREFIX}LiteLLMClientError@")
    assert storage.extraction_status("u", "outage")["status"] == "pending"

    # The provider recovers: the same window runs and completes.
    provider_circuit.reset()
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", lambda *_: False
    )
    _drain_once(worker, storage)
    assert storage.extraction_status("u", "outage")["status"] == "done"
    assert _cursor(storage)["attempts"] == 0


def test_failure_while_the_provider_serves_others_is_charged_and_parks(
    pipeline, monkeypatch
):
    engine, worker, client = pipeline
    assert client.post("/api/publish_interaction", json=payload("poison")).json()[
        "success"
    ]
    storage = engine.get_storage()
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", _provider_times_out
    )

    # Every breaker closed: the provider answers everyone else, so a window
    # that keeps timing out is failing on its own and must stop.
    for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 3):
        _drain_once(worker, storage)

    cursor = _cursor(storage)
    assert cursor["attempts"] == MAX_EXTRACTION_WINDOW_ATTEMPTS
    assert cursor["retry_at"] == float("inf")
    assert cursor["last_error"].startswith(
        f"{RETRY_EXHAUSTED_PREFIX}LiteLLMClientError@"
    )


@pytest.mark.parametrize("outage_first", [True, False])
def test_only_failures_outside_an_outage_count(pipeline, monkeypatch, outage_first):
    engine, worker, client = pipeline
    assert client.post("/api/publish_interaction", json=payload("mixed")).json()[
        "success"
    ]
    storage = engine.get_storage()
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", _provider_times_out
    )

    def outage(turns: int) -> None:
        _open_breaker()
        for _ in range(turns):
            _drain_once(worker, storage)
        provider_circuit.reset()

    def healthy(turns: int) -> None:
        for _ in range(turns):
            _drain_once(worker, storage)

    if outage_first:
        outage(20)
        healthy(3)
    else:
        healthy(3)
        outage(20)
    assert _cursor(storage)["attempts"] == 3


def test_an_outage_inside_the_extraction_agent_turn_waits_uncharged(
    pipeline, monkeypatch
):
    """The provider fails inside the agent's own model turn, not before it.

    The tool loop turns a provider failure into a result rather than raising;
    the window must still see the provider failure (and its route) beneath
    the extractor's own error, through the real client and the real breaker.
    """
    import litellm

    from reflexio.server.llm.litellm_client import LiteLLMClient

    engine, worker, client = pipeline
    assert client.post("/api/publish_interaction", json=payload("agent")).json()[
        "success"
    ]
    storage = engine.get_storage()
    # Run the real extractor: its agent turn is where the provider fails.
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", lambda *_: True
    )
    sent: list[str] = []

    def rate_limited(self, params, _hard_timeout):
        sent.append(params["model"])
        raise litellm.exceptions.RateLimitError(
            message="usage limit reached",
            llm_provider="test",
            model=str(params["model"]),
        )

    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", rate_limited)

    for _ in range(3 * MAX_EXTRACTION_WINDOW_ATTEMPTS):
        _drain_once(worker, storage)

    cursor = _cursor(storage)
    # At most the failures that opened the breaker were charged; none after it.
    assert cursor["attempts"] <= OPEN_AFTER_FAILURES
    assert cursor["retry_at"] != float("inf")
    assert cursor["last_error"].startswith(PROVIDER_WAIT_PREFIX)
    assert storage.extraction_status("u", "agent")["status"] == "pending"
    # While open, the breaker sent nothing more than its probes.
    assert len(sent) < 3 * MAX_EXTRACTION_WINDOW_ATTEMPTS
