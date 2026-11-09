"""A provider outage never spends a window's overflow attempt journal; its own failures do.

Invariant (``provider_circuit`` module docstring): a failure is charged only
when the provider was serving other calls. An overflow window also records
every reserved dispatch in its durable journal (128 attempts, plus package
stage caps and the reconciliation allowance). While the failing route's
breaker is open, each probe that fails returns its attempt, so an outage of any
length never exhausts the journal and holds the window. Before this, every
failed probe stayed charged: a long enough outage held the window for operator
recovery although nothing was wrong with it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import litellm
import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.provider_circuit import (
    MAX_OPEN_SECONDS,
    circuit_route,
    provider_circuit,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    PROVIDER_WAIT_PREFIX,
)
from reflexio.test_support.skip_decorators import skip_in_precommit
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline

# The journal's capacity: one more failed probe than this held the window.
_JOURNAL_ATTEMPTS = 128
_PROFILE = {
    "content": "The user prefers concise answers.",
    "time_to_live": "infinity",
    "source_span": "I prefer concise answers with details 0",
}


class _Provider:
    """A provider that is down until told otherwise; records the routes it was sent."""

    def __init__(self) -> None:
        self.down = True
        self.routes: list[str] = []

    def __call__(self, params, _hard_timeout):
        self.routes.append(circuit_route(params))
        if self.down:
            raise litellm.exceptions.RateLimitError(
                message="usage limit reached",
                llm_provider="test",
                model=str(params["model"]),
            )
        schema = getattr(params.get("response_format"), "__name__", "") or json.dumps(
            params.get("response_format"), default=str
        )
        if "ProfileDeduplicationOutput" in schema or any(
            "unique_ids" in str(message.get("content", ""))
            for message in params["messages"]
        ):
            content = {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
        else:
            content = {"profiles": [_PROFILE]}
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(content), tool_calls=None, refusal=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model=str(params["model"]),
        )


@pytest.fixture
def overflow(pipeline, monkeypatch):
    """A real overflow window behind a provider that is down, on a test clock."""
    engine, worker, client = pipeline
    storage = engine.get_storage()
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    clock = [1000.0]
    monkeypatch.setattr(provider_circuit, "_clock", lambda: clock[0])
    provider = _Provider()
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", provider)
    assert client.post(
        "/api/publish_interaction", json=payload("outage", force=True)
    ).json()["success"]

    def drain(*, probe: bool) -> None:
        with storage._stream_sql() as db:
            # Skip backoff, never a park: a held cursor must stay held.
            db.query("UPDATE extraction_cursors SET retry_at=0 WHERE retry_at<1e300")
            db.query("UPDATE learning_work SET due_at=0 WHERE user_id='u'")
        if probe:
            # Past any open period: the next call is the breaker's probe.
            clock[0] += MAX_OPEN_SECONDS + 1
        worker.drain_org(storage.org_id, 1, 300)

    return storage, provider, drain


def _state(storage) -> tuple[dict, list[dict]]:
    with storage._stream_sql(read_only=True) as db:
        cursor = db.query(
            "SELECT window_id,attempts,retry_at,last_error FROM extraction_cursors "
            "WHERE user_id='u' AND kind='profile'"
        )[0]
    run = storage.get_agent_run(f"window:{cursor['window_id']}")
    assert run is not None
    return cursor, run.generation_request_snapshot["overflow"]["attempts"]


def _breaker_open(provider: _Provider) -> bool:
    return bool(provider.routes) and (
        provider_circuit.outage_retry_delay(provider.routes[-1]) is not None
    )


@skip_in_precommit
def test_outage_probes_spend_no_overflow_attempts_and_the_window_completes(overflow):
    storage, provider, drain = overflow
    # The failures that open the breaker happen while it is closed: the
    # provider might still be serving others, so those are charged.
    while not _breaker_open(provider):
        drain(probe=False)
    _cursor, opened = _state(storage)
    assert 0 < len(opened) < len(provider.routes)

    for _ in range(_JOURNAL_ATTEMPTS + 1):
        sent = len(provider.routes)
        drain(probe=True)
        assert len(provider.routes) == sent + 1  # exactly the breaker's probe

    cursor, attempts = _state(storage)
    assert attempts == opened
    assert cursor["retry_at"] != float("inf")
    assert cursor["last_error"].startswith(PROVIDER_WAIT_PREFIX)
    assert storage.extraction_status("u", "outage")["status"] == "pending"

    # The provider recovers: the same window runs and completes.
    provider.down = False
    provider_circuit.reset()
    drain(probe=False)
    assert storage.extraction_status("u", "outage")["status"] == "done"
    assert storage.count_all_profiles() > 0


def test_failures_while_the_provider_serves_others_stay_charged(overflow):
    storage, provider, drain = overflow
    for _ in range(4):
        drain(probe=False)
        # Every breaker closed: the provider answers everyone else.
        provider_circuit.reset()
    _cursor, attempts = _state(storage)
    assert len(provider.routes) == 4
    assert len(attempts) == 4
    assert all(attempt["usage"] is None for attempt in attempts)


def test_outage_probes_spend_no_package_stage_attempts(pipeline, monkeypatch):
    """A package stage holds one attempt per model rung; a probe used to spend it.

    The gate stage of a single-model ladder has exactly one attempt. Before
    the fix the first failed probe spent it, and the next turn held the window
    with ``Package stage attempt budget exhausted``.
    """
    from reflexio.server.llm.failure import OutageWait, classify_llm_failure
    from reflexio.server.llm.provider_circuit import OPEN_AFTER_FAILURES
    from reflexio.server.services.durable_learning.profile_package_executor import (
        ProfilePackageWindowExecutor,
    )
    from tests.server.services.test_profile_package_executor import setup

    engine, storage, window, token, client, calls = setup(pipeline, monkeypatch)
    clock = [1000.0]
    monkeypatch.setattr(provider_circuit, "_clock", lambda: clock[0])
    answer = client._completion_with_hard_timeout
    down = [True]

    def wire(params, timeout):
        if down[0] and not params.get("response_format"):  # the gate request
            route = circuit_route(params)
            if provider_circuit.outage_retry_delay(route) is None:
                # Everyone else's calls on this route are failing too.
                for _ in range(OPEN_AFTER_FAILURES):
                    provider_circuit.admit(route, model=str(params["model"])).failed()
            raise litellm.exceptions.RateLimitError(
                message="usage limit reached",
                llm_provider="test",
                model=str(params["model"]),
            )
        return answer(params, timeout)

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    executor = ProfilePackageWindowExecutor(engine.request_context, client)

    def gate_attempts() -> int:
        parent = storage.get_agent_run(f"window:{window.window_id}")
        journal = parent.generation_request_snapshot["overflow"]
        assert journal["version"] == 2
        assert journal["input_protocol"]["stage_caps"]["gate"] == 1
        return sum(item["stage"] == "gate" for item in journal["attempts"])

    for _ in range(4):
        clock[0] += MAX_OPEN_SECONDS + 1  # the next call is the breaker's probe
        with pytest.raises(Exception) as failure:
            executor.execute(storage.prepare_extraction("u", token), token)
        assert isinstance(classify_llm_failure(failure.value), OutageWait)
        assert gate_attempts() == 0

    down[0] = False
    provider_circuit.reset()
    executor.execute(storage.prepare_extraction("u", token), token)
    assert gate_attempts() == 1
    assert len(storage.get_user_profile("u")) == 1
    assert storage.extraction_status("u", "package-parent")["status"] == "done"


def test_primary_quota_failover_leaves_only_answered_attempts(overflow, monkeypatch):
    """An exhausted primary key fails over to the backup key within one request.

    The exhausted primary opens its own route's breaker at once, so its
    dispatch is an outage dispatch and returns its attempt; the backup's
    answered dispatch is the one the journal keeps. Without the release, every
    failover left an unanswered attempt charged in the journal.
    """
    from reflexio.server.llm._litellm_types import LiteLLMClientError

    storage, provider, drain = overflow
    monkeypatch.setenv("MINIMAX_API_KEY", "primary-key")
    monkeypatch.setenv("MINIMAX_BACKUP_API_KEY", "backup-key")
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    answer = provider.__call__
    keys: list[str | None] = []

    def plan_exhausted_primary(_client, params, hard_timeout):
        keys.append(params.get("api_key"))
        if params.get("api_key") == "backup-key":
            provider.down = False
            return answer(params, hard_timeout)
        provider.routes.append(circuit_route(params))
        raise LiteLLMClientError(
            "RateLimitError: Token Plan usage limit reached (2056)",
            upstream_error_type="RateLimitError",
            upstream_status_code=429,
        )

    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", plan_exhausted_primary
    )
    drain(probe=False)

    assert "backup-key" in keys and keys.count("backup-key") < len(keys)
    assert storage.extraction_status("u", "outage")["status"] == "done"
    with storage._stream_sql(read_only=True) as db:
        window_id = db.query(
            "SELECT window_id FROM extraction_windows WHERE user_id='u' AND kind='profile'"
        )[0]["window_id"]
    run = storage.get_agent_run(f"window:{window_id}")
    assert run is not None
    attempts = run.generation_request_snapshot["overflow"]["attempts"]
    assert attempts
    assert all(attempt["usage"] is not None for attempt in attempts)
