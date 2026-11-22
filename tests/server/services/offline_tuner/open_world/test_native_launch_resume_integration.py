"""A persisted local-cli attempt resumes only under the same launch identity.

The native launch identity (resolved executable, account directories, target)
is part of the composed transport's persisted identity, so a process
recomposed with a different executable or account fails the durable
attempt-context binding instead of resuming the stored invocations under a
different CLI -- while a restart from another directory or shell that still
launches the same binary resumes normally.
"""

# Complete provider initialization before importing the third-party library.
import reflexio.provider_bootstrap  # noqa: F401

# isort: split

import dataclasses
import os
from types import SimpleNamespace
from typing import Any

import pytest

from reflexio.server.llm._native_model_ref import CLI_TIMEOUT_ENV
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
    _claude_on_path,
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


def _launch(monkeypatch, tmp_path) -> str:
    """Own every launch input: target, HOME, cwd and a `claude` on PATH."""
    model = _native(monkeypatch, "claude-code")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    _claude_on_path(monkeypatch, tmp_path / "bin")
    return model


@pytest.mark.parametrize("change", [None, "cwd", "path-same-binary", "timeout"])
def test_resume_after_an_inert_environment_change_reuses_the_ledger(
    port_harness, monkeypatch, tmp_path, change
):
    model = _launch(monkeypatch, tmp_path)
    original = _transport(monkeypatch, model, called_at=port_harness.now())
    attempt = _attempt(port_harness, transport=original)
    first = attempt.run()
    assert first.decision.decision == "accept"

    # A restart from another directory or shell that launches the same binary.
    if change == "cwd":
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.chdir(tmp_path / "elsewhere")
    elif change == "path-same-binary":
        (tmp_path / "empty").mkdir()
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "empty"), str(tmp_path / "bin")])
        )
    elif change == "timeout":
        monkeypatch.setenv(CLI_TIMEOUT_ENV, "30")
    restarted = _transport(monkeypatch, model, called_at=attempt.now)
    second = attempt.run(_store(attempt, restarted))

    assert second.decision == first.decision
    assert restarted.calls == [], "no provider call is repeated"


@pytest.mark.parametrize(
    "change",
    [
        "cli-path-override",
        "other-binary-on-path",
        "claude-config-dir",
        "home",
        "target",
    ],
)
def test_resume_under_a_changed_launch_identity_is_refused(
    port_harness, monkeypatch, tmp_path, change
):
    model = _launch(monkeypatch, tmp_path)
    original = _transport(monkeypatch, model, called_at=port_harness.now())
    attempt = _attempt(port_harness, transport=original)
    assert attempt.run().decision.decision == "accept"
    rows = port_harness.attempt_rows(attempt.job_id)

    # The restarted process would launch a different CLI or account.
    if change == "cli-path-override":
        monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/synthetic/other/claude")
    elif change == "other-binary-on-path":
        _claude_on_path(monkeypatch, tmp_path / "other-bin")
    elif change == "claude-config-dir":
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "account-b"))
    elif change == "home":
        monkeypatch.setenv("HOME", str(tmp_path / "other-home"))
    else:
        model = _native(monkeypatch, "claude-code", "claude-x")
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
    assert moved.calls == [], "no provider call under the changed identity"
    assert port_harness.attempt_rows(attempt.job_id) == rows
    jobs = port_harness.optimization_jobs(attempt.incumbent.user_playbook_id)
    assert [(job.stage, job.terminal_outcome) for job in jobs] == [
        ("failed", "infrastructure_failure")
    ]
