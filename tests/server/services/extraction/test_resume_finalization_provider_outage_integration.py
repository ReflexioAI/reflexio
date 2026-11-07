"""A provider outage during resumed publication waits; it never spends finalization attempts.

Invariant (``provider_circuit`` module docstring): a failure is charged only
when the provider was serving other calls. Publishing a resumed run's retained
output prepares a write plan, and preparation calls the model; a call that
fails closed (the playbook candidate reviewer, for one) fails the publication.
While the route's breaker is open, such a failure waits for the provider
without spending ``max_finalization_attempts``; once it recovers, the same
retained output is published. A failure while every breaker is closed is the
run's own and still exhausts the allowance.

The tests raise the routed provider failure from the profile deduplication
step, the model call in this fixture's preparation; ``admitted`` covers a call
that had already opened the preparation's compute entry before it failed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.server.llm._litellm_types import LiteLLMClientError
from reflexio.server.llm.compute_admission import admit_external_compute
from reflexio.server.llm.provider_circuit import (
    OPEN_AFTER_FAILURES,
    circuit_route,
    provider_circuit,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.extraction.test_resume_worker import (
    _commit_resumed_output,
    _seed_interactions,
    _seed_ready_run,
)
from tests.server.services.extraction.test_resume_worker import (
    explicit_test_billing_exemption as explicit_test_billing_exemption,
)
from tests.server.services.extraction.test_resume_worker import (
    no_background_tagging as no_background_tagging,
)
from tests.server.services.extraction.test_resume_worker import (
    request_context as request_context,
)
from tests.server.services.extraction.test_resume_worker import (
    storage as storage,
)

ROUTE = circuit_route({"model": "test/model"})
_DEDUPLICATE = (
    "reflexio.server.services.profile.components.consolidator."
    "ProfileConsolidator.deduplicate"
)


def _provider_rate_limited(*_args: object) -> None:
    # The shape the client raises: its wrapper, naming the route that failed.
    failure = LiteLLMClientError("API call failed: usage limit reached")
    failure.circuit_route = ROUTE
    raise failure from TimeoutError("usage limit reached")


def _admitted_then_rate_limited(*args: object) -> None:
    # Admission at dispatch opens the preparation's compute entry first.
    admit_external_compute()
    _provider_rate_limited(*args)


def _open_breaker() -> None:
    for _ in range(OPEN_AFTER_FAILURES):
        provider_circuit.admit(ROUTE, model="test/model").failed()
    assert provider_circuit.outage_retry_delay(ROUTE) is not None


def _active_preparation_entry(storage) -> object:
    with storage._stream_sql(read_only=True) as db:
        rows = db.query(
            "SELECT active_attempt FROM _agent_output_preparations WHERE run_id=?",
            ("run_1",),
        )
    return rows[0]["active_attempt"] if rows else None


def _make_due(storage) -> None:
    run = storage.get_agent_run("run_1")
    if run is None or run.status == AgentRunStatus.FAILED:
        return  # A failed run must stay failed.
    storage.update_agent_run_status(
        "run_1",
        AgentRunStatus.FINALIZATION_FAILED,
        next_resume_at=datetime(2000, 1, 1, tzinfo=UTC),
    )


def _resume_commits_once(monkeypatch) -> list[int]:
    """The agent turn succeeds once; every later turn is publication only."""
    calls: list[int] = []

    def resume(active_worker, run, _calls):
        calls.append(1)
        return _commit_resumed_output(
            active_worker,
            run,
            {
                "profiles": [
                    {
                        "content": "Deploy to the container cluster.",
                        "time_to_live": "infinity",
                    }
                ]
            },
        )

    monkeypatch.setattr(ExtractionResumeWorker, "_resume_run", resume)
    return calls


@pytest.mark.parametrize("admitted", [False, True])
def test_outage_during_finalization_waits_uncharged_and_publishes_after_recovery(
    storage, request_context, monkeypatch, admitted
):
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_interactions(storage)
    _seed_ready_run(storage)
    worker = DurableLearningWorker(lambda _org: request_context)
    resumed = _resume_commits_once(monkeypatch)
    monkeypatch.setattr(
        _DEDUPLICATE,
        _admitted_then_rate_limited if admitted else _provider_rate_limited,
    )
    _open_breaker()
    limit = request_context.configurator.get_config().pending_tool_call_config

    # Three times the allowance: an outage this long failed the run before.
    for _ in range(3 * limit.max_finalization_attempts):
        before = datetime.now(UTC)
        worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300)
        run = storage.get_agent_run("run_1")
        assert run is not None
        assert run.status == AgentRunStatus.FINALIZATION_FAILED
        assert run.finalization_attempts == 0
        assert run.last_error == "provider_unavailable"
        # It waits for the breaker's next probe rather than retrying at once.
        assert run.next_resume_at is not None and run.next_resume_at > before
        # The compute entry is settled, so the next turn is not "interrupted".
        assert _active_preparation_entry(storage) is None
        _make_due(storage)

    assert resumed == [1]
    assert storage.get_user_profile("user_1") == []

    # The provider recovers: the same retained output is published.
    provider_circuit.reset()
    monkeypatch.setattr(
        _DEDUPLICATE, lambda _self, profiles, _user_id, _request_id: (profiles, [], [])
    )
    worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300)
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZED
    assert run.finalization_attempts == 0
    assert resumed == [1]
    assert [profile.content for profile in storage.get_user_profile("user_1")] == [
        "Deploy to the container cluster."
    ]


def test_finalization_failure_while_the_provider_serves_others_is_charged(
    storage, request_context, monkeypatch
):
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_interactions(storage)
    _seed_ready_run(storage)
    worker = DurableLearningWorker(lambda _org: request_context)
    _resume_commits_once(monkeypatch)
    # Every breaker closed: the provider answers everyone else, so a run whose
    # publication keeps failing is failing on its own and must stop.
    monkeypatch.setattr(_DEDUPLICATE, _provider_rate_limited)
    limit = request_context.configurator.get_config().pending_tool_call_config

    for _ in range(limit.max_finalization_attempts + 3):
        worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300)
        run = storage.get_agent_run("run_1")
        assert run is not None
        if run.status == AgentRunStatus.FAILED:
            break
        _make_due(storage)

    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FAILED
    assert run.finalization_attempts == limit.max_finalization_attempts


@pytest.mark.parametrize("outage_first", [True, False])
def test_only_finalization_failures_outside_an_outage_count(
    storage, request_context, monkeypatch, outage_first
):
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker

    _seed_interactions(storage)
    _seed_ready_run(storage)
    worker = DurableLearningWorker(lambda _org: request_context)
    _resume_commits_once(monkeypatch)
    monkeypatch.setattr(_DEDUPLICATE, _provider_rate_limited)

    def drain(turns: int) -> None:
        for _ in range(turns):
            worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300)
            _make_due(storage)

    def outage(turns: int) -> None:
        _open_breaker()
        drain(turns)
        provider_circuit.reset()

    if outage_first:
        outage(6)
        drain(1)
    else:
        drain(1)
        outage(6)
    run = storage.get_agent_run("run_1")
    assert run is not None
    assert run.status == AgentRunStatus.FINALIZATION_FAILED
    assert run.finalization_attempts == 1
