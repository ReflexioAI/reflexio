"""A persisted local-cli attempt cannot resume under a changed launch policy.

The native launch policy is part of the composed transport's persisted
identity, so a process recomposed with a different executable or account
directory fails the durable attempt-context binding instead of resuming the
stored invocations under a different CLI.
"""

# Complete provider initialization before importing the third-party library.
import reflexio.provider_bootstrap  # noqa: F401

# isort: split

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.offline_tuner.open_world.analyst import (
    _litellm_transport_identity,
)
from reflexio.server.services.offline_tuner.open_world.evidence_documents import (
    rehydrate_frozen_evidence,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    AttemptTerminalFailure,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_authority import (
    qualify_rule_evidence_method,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_runtime import (
    RuleEvidenceAnalysisStore,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldInvocationConflictError,
)
from reflexio.test_support.open_world_attempt_contract import (
    GroundedTransport,
    _attempt,
)
from tests.server.services.offline_tuner.open_world.test_served_model_admission import (
    _native,
)
from tests.server.services.storage.test_open_world_port_contract import (
    port_harness as port_harness,
)


class _ComposedTransport(GroundedTransport):
    """Report the composed provider, as the production transport does."""

    def complete(self, **kwargs: Any) -> Any:
        return dataclasses.replace(
            super().complete(**kwargs),
            observed_provider=self.transport_identity.provider,
        )


def _transport(monkeypatch, model: str, *, called_at: int) -> GroundedTransport:
    """A recorded transport carrying the identity a real composition attests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test")
    client = LiteLLMClient(
        LiteLLMConfig(model=model, max_tokens=512, fallback_models=[])
    )
    transport = _ComposedTransport(called_at=called_at)
    transport.transport_identity = SimpleNamespace(
        provider=_litellm_transport_identity(client).provider,
        model_revision=model,
    )
    return transport


def _store(attempt, transport: GroundedTransport) -> RuleEvidenceAnalysisStore:
    """Recompose the method and store, as a restarted process does."""
    method = qualify_rule_evidence_method(transport)
    method = type(method)(
        method.diagnostic, method.proposal, method.qualification_receipt
    )
    store = RuleEvidenceAnalysisStore(
        attempt.storage, method, transport, lambda: attempt.now
    )
    store.reconcile_authority()
    return store


def test_resume_under_the_same_launch_policy_reuses_the_ledger(
    port_harness, monkeypatch
):
    model = _native(monkeypatch, "claude-code")
    original = _transport(monkeypatch, model, called_at=port_harness.now())
    attempt = _attempt(port_harness, transport=original)
    first = attempt.run()
    assert first.decision.decision == "accept"

    restarted = _transport(monkeypatch, model, called_at=attempt.now)
    second = attempt.run(_store(attempt, restarted))

    assert second.decision == first.decision
    assert restarted.calls == [], "no provider call is repeated"


def test_resume_under_a_changed_launch_policy_is_refused(port_harness, monkeypatch):
    model = _native(monkeypatch, "claude-code")
    original = _transport(monkeypatch, model, called_at=port_harness.now())
    attempt = _attempt(port_harness, transport=original)
    assert attempt.run().decision.decision == "accept"
    rows = port_harness.attempt_rows(attempt.job_id)

    # The restarted process launches a different CLI executable.
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/synthetic/other/claude")
    moved = _transport(monkeypatch, model, called_at=attempt.now)
    assert moved.transport_identity.provider != original.transport_identity.provider
    store = _store(attempt, moved)

    with pytest.raises(OpenWorldInvocationConflictError, match="conflicts"):
        attempt.run(store)
    outcome = attempt.runner(store).run(
        frozen=attempt.frozen,
        incumbent=attempt.incumbent,
        evidence=rehydrate_frozen_evidence(attempt.frozen, attempt.snapshot),
    )

    assert outcome == AttemptTerminalFailure(reason="infrastructure_failure")
    assert moved.calls == [], "no provider call under the changed policy"
    assert port_harness.attempt_rows(attempt.job_id) == rows
    jobs = port_harness.optimization_jobs(attempt.incumbent.user_playbook_id)
    assert [(job.stage, job.terminal_outcome) for job in jobs] == [
        ("failed", "infrastructure_failure")
    ]
