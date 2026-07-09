"""Unit tests for the in-process embedder warm-before-ready gate (D5/D8).

These prove the Phase-2 dormancy invariant: nothing here activates unless a
deployment has explicitly flipped ``REFLEXIO_EMBEDDING_PROVIDER=inprocess`` with
a ``local/*`` default embedding model.
"""

from __future__ import annotations

import logging
import time

import pytest

from reflexio.server.llm.providers import embedder_warmup
from reflexio.server.llm.providers.embedder_warmup import (
    inprocess_local_gate_active,
    is_embedder_ready,
    maybe_start_embedder_warmup,
    reset_warmup_state_for_test,
    run_startup_config_guards,
)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Clear the process-global readiness signal before every test."""
    reset_warmup_state_for_test()


# --------------------------------------------------------------------------- #
# Gate: dormant unless inprocess + local/*                                    #
# --------------------------------------------------------------------------- #


def test_gate_inactive_when_provider_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No REFLEXIO_EMBEDDING_PROVIDER → gate inactive (current prod state)."""
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    assert inprocess_local_gate_active() is False


@pytest.mark.parametrize("mode", ["cloud", "off", "local_service", "internal_service"])
def test_gate_inactive_for_non_inprocess_modes(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Any explicit non-inprocess provider keeps the gate dormant."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", mode)
    if mode in {"local_service", "internal_service"}:
        # These modes require a service URL to resolve; the gate must short out
        # on the provider check before touching URL resolution.
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://127.0.0.1:8072")
    assert inprocess_local_gate_active() is False


def test_gate_active_when_inprocess_and_local_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """inprocess + the default local embedding model → gate active."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    assert inprocess_local_gate_active() is True


def test_gate_inactive_when_inprocess_but_model_not_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """inprocess but the resolved default is a cloud model → gate inactive.

    Proves the gate is a genuine AND of provider AND model, not provider alone.
    """
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    monkeypatch.setattr(
        embedder_warmup,
        "_provider",
        lambda: "inprocess",
    )

    from reflexio.server.llm import model_defaults

    monkeypatch.setattr(
        model_defaults,
        "resolve_model_name",
        lambda *_a, **_k: "text-embedding-3-small",
    )
    assert inprocess_local_gate_active() is False


# --------------------------------------------------------------------------- #
# D5 warm-before-ready thread                                                 #
# --------------------------------------------------------------------------- #


def test_warmup_not_started_when_gate_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate inactive → no thread spawned, readiness stays clear."""
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    assert maybe_start_embedder_warmup() is False
    assert is_embedder_ready() is False


def test_warmup_starts_and_marks_ready_under_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate active → a daemon thread loads a (fake) embedder and flips ready.

    Uses a fast fake so the real ~550MB model never loads. Startup itself must
    not block: ``maybe_start_embedder_warmup`` returns immediately while the
    load happens on the background thread.
    """
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")

    load_calls: list[int] = []

    from reflexio.server.llm.providers import embedder_warmup

    # Patch the loader seam so the test controls warm success regardless of which
    # local embedder (nomic vs minilm/LocalEmbedder) the default model resolves to.
    monkeypatch.setattr(
        embedder_warmup,
        "_resolve_local_embedder_loader",
        lambda: lambda: load_calls.append(1),
    )

    assert is_embedder_ready() is False
    started = maybe_start_embedder_warmup()
    assert started is True

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not is_embedder_ready():
        time.sleep(0.01)

    assert is_embedder_ready() is True
    assert load_calls == [1]


def test_warmup_failure_leaves_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """A load failure (all retries) must NOT mark ready — /health stays 503 (fail-safe)."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")

    from reflexio.server.llm.providers import embedder_warmup

    # Fast, deterministic retry for the test.
    monkeypatch.setattr(embedder_warmup, "_WARM_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(embedder_warmup, "_WARM_RETRY_BACKOFF_S", 0.0)

    attempts: list[int] = []

    def _boom() -> None:
        attempts.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(
        embedder_warmup, "_resolve_local_embedder_loader", lambda: _boom
    )

    assert maybe_start_embedder_warmup() is True
    time.sleep(0.2)
    assert is_embedder_ready() is False
    # Bounded retry actually retried before giving up.
    assert attempts == [1, 1]


def test_warm_target_follows_resolved_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The warmed embedder matches the resolved local model (nomic vs minilm),
    not a hardcoded class — so readiness reflects the model that will serve."""
    from reflexio.server.llm import model_defaults
    from reflexio.server.llm.providers import (
        embedder_warmup,
        local_embedding_provider,
        nomic_embedding_provider,
    )

    warmed: list[str] = []
    monkeypatch.setattr(
        nomic_embedding_provider.NomicEmbedder,
        "get",
        staticmethod(
            lambda: type("N", (), {"_load": lambda _self: warmed.append("nomic")})()
        ),
    )
    monkeypatch.setattr(
        local_embedding_provider.LocalEmbedder,
        "get",
        staticmethod(
            lambda: type("L", (), {"_load": lambda _self: warmed.append("minilm")})()
        ),
    )

    monkeypatch.setattr(
        model_defaults,
        "resolve_model_name",
        lambda _role: "local/nomic-embed-text-v1.5",
    )
    embedder_warmup._resolve_local_embedder_loader()()  # type: ignore[misc]
    assert warmed == ["nomic"]

    warmed.clear()
    monkeypatch.setattr(
        model_defaults, "resolve_model_name", lambda _role: "local/minilm-l6-v2"
    )
    embedder_warmup._resolve_local_embedder_loader()()  # type: ignore[misc]
    assert warmed == ["minilm"]

    # A non-local resolved model yields no loader (gate would be inactive too).
    monkeypatch.setattr(
        model_defaults, "resolve_model_name", lambda _role: "text-embedding-3-small"
    )
    assert embedder_warmup._resolve_local_embedder_loader() is None


# --------------------------------------------------------------------------- #
# D8(a): workers-multiply-model guard                                         #
# --------------------------------------------------------------------------- #


def test_workers_guard_warns_on_multiworker_inprocess(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    monkeypatch.setenv("REFLEXIO_SERVER_WORKERS", "2")
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_workers_multiply_model()
    assert any("model copy PER worker" in r.message for r in caplog.records)


def test_workers_guard_silent_on_single_worker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    monkeypatch.setenv("REFLEXIO_SERVER_WORKERS", "1")
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_workers_multiply_model()
    assert caplog.records == []


def test_workers_guard_warns_when_undetectable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    monkeypatch.delenv("REFLEXIO_SERVER_WORKERS", raising=False)
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_workers_multiply_model()
    assert any("could not be verified" in r.message for r in caplog.records)


def test_workers_guard_silent_when_not_inprocess(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Even with many workers, the guard is silent off the in-process path."""
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("REFLEXIO_SERVER_WORKERS", "8")
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_workers_multiply_model()
    assert caplog.records == []


# --------------------------------------------------------------------------- #
# D8(b): half-pair daemon-disable / provider guard                            #
# --------------------------------------------------------------------------- #


def test_half_pair_guard_warns_when_provider_set_alone(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    monkeypatch.delenv("REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON", raising=False)
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_daemon_disable_half_pair()
    assert any("Half-configured" in r.message for r in caplog.records)


def test_half_pair_guard_warns_when_disable_set_alone(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON", "1")
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_daemon_disable_half_pair()
    assert any("Half-configured" in r.message for r in caplog.records)


def test_half_pair_guard_silent_when_both_set(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    monkeypatch.setenv("REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON", "1")
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_daemon_disable_half_pair()
    assert caplog.records == []


def test_half_pair_guard_silent_when_neither_set(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON", raising=False)
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_daemon_disable_half_pair()
    assert caplog.records == []


# --------------------------------------------------------------------------- #
# D8(c): inprocess-overrides-service-host guard                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("REFLEXIO_EMBEDDING_SERVICE_URL", "http://embedding.internal:8089"),
        ("REFLEXIO_EMBEDDING_DAEMON_HOST", "embedding.internal"),
    ],
)
def test_inprocess_override_guard_warns_when_service_endpoint_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env_var: str,
    value: str,
) -> None:
    """inprocess + a configured service endpoint → one warning naming the var."""
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_DAEMON_HOST", raising=False)
    monkeypatch.setenv(env_var, value)
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_inprocess_overrides_service_host()
    warnings = [r for r in caplog.records if "takes precedence" in r.message]
    assert len(warnings) == 1
    assert env_var in warnings[0].getMessage()


def test_inprocess_override_guard_silent_for_intended_step_b_config(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The intended flip (inprocess + disable-daemon, NO service endpoints) is silent.

    This is the exact Step B config the in-process rollout ships, so the guard
    must not false-positive on it.
    """
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "inprocess")
    monkeypatch.setenv("REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON", "true")
    monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
    monkeypatch.delenv("REFLEXIO_EMBEDDING_DAEMON_HOST", raising=False)
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_inprocess_overrides_service_host()
    assert caplog.records == []


def test_inprocess_override_guard_silent_when_not_inprocess(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A service host without PROVIDER=inprocess is the normal service topology.

    That routes to the service (no in-process load), so the guard stays scoped to
    the in-process path and says nothing.
    """
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setenv("REFLEXIO_EMBEDDING_DAEMON_HOST", "embedding.internal")
    with caplog.at_level(logging.WARNING):
        embedder_warmup._guard_inprocess_overrides_service_host()
    assert caplog.records == []


def test_run_startup_config_guards_is_silent_when_dormant(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The whole guard pass emits nothing in the default (pre-flip) state."""
    monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON", raising=False)
    monkeypatch.setenv("REFLEXIO_SERVER_WORKERS", "1")
    with caplog.at_level(logging.WARNING):
        run_startup_config_guards()
    assert caplog.records == []
