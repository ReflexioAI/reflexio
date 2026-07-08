"""Warm-before-ready readiness gate for the in-process local embedder.

Phase 2 preparation for the embedding-stability redesign. Everything here is
**dormant** until a future config flip sets ``REFLEXIO_EMBEDDING_PROVIDER=inprocess``
with a ``local/*`` default embedding model. Pre-flip (the current daemon-mode
prod state, or any cloud/off/OSS-dev deployment) none of this changes ``/health``
or startup behaviour.

Three concerns live here:

- **D5 warm-before-ready** — a process-level ``threading.Event`` set once the
  in-process embedder has loaded, plus a non-blocking startup thread that loads
  it. ``/health`` reports 503 while the gate is active and the embedder is not
  yet warm, so the load balancer does not route embedding traffic at a worker
  that would pay the ~2-8s cold-load on its first request.
- **D8 config guards** — loud startup warnings for foot-guns of the in-process
  topology: multiple uvicorn workers (each loads its own model copy) and a
  half-configured daemon-disable / provider pair.

The gate is intentionally cheap to evaluate (no network probe, no provider
auto-detection) because ``/health`` is the ALB + container health target and is
polled frequently.
"""

from __future__ import annotations

import logging
import os
import threading

_LOGGER = logging.getLogger(__name__)

_ENV_PROVIDER = "REFLEXIO_EMBEDDING_PROVIDER"
_ENV_DISABLE_DAEMON = "REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON"
# Recorded by ``reflexio.server.__main__`` before ``uvicorn.run`` so the guard
# can read the configured worker count from inside a worker process, where
# uvicorn exposes no worker-count env of its own.
_ENV_WORKERS = "REFLEXIO_SERVER_WORKERS"
_ENV_WEB_CONCURRENCY = "WEB_CONCURRENCY"

_INPROCESS = "inprocess"

# Process-level readiness signal: set once the in-process embedder is loaded.
_ready = threading.Event()


def mark_embedder_ready() -> None:
    """Mark the in-process embedder as loaded/warm for this process."""
    _ready.set()


def is_embedder_ready() -> bool:
    """Return True once the in-process embedder has been warmed in this process."""
    return _ready.is_set()


def reset_warmup_state_for_test() -> None:
    """Clear the readiness signal. Test-only hygiene helper."""
    _ready.clear()


def _provider() -> str:
    return os.environ.get(_ENV_PROVIDER, "").strip().lower()


def inprocess_local_gate_active() -> bool:
    """Return True iff this deployment serves embeddings via the in-process local path.

    The gate is active only when BOTH hold:

    - ``REFLEXIO_EMBEDDING_PROVIDER == "inprocess"`` (explicit config flip), and
    - the resolved default embedding model is a ``local/*`` model.

    Any other configuration (cloud, off, daemon-mode ``local_service`` /
    ``internal_service``, or no explicit provider at all) returns False, keeping
    the warm-before-ready behaviour dormant. Cheap by construction: neither
    branch performs a network probe or provider auto-detection.

    Returns:
        bool: True when warm-before-ready ``/health`` gating should apply.
    """
    from reflexio.server.llm.providers.embedding_service_provider import (
        embedding_provider_mode,
    )

    if embedding_provider_mode() != _INPROCESS:
        return False

    from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name

    try:
        model = resolve_model_name(ModelRole.EMBEDDING)
    except Exception:  # noqa: BLE001
        # No embedding-capable provider resolvable — cannot be the in-process
        # local path, so the gate stays inactive rather than wedging /health.
        _LOGGER.debug("Embedding model resolution failed; gate inactive", exc_info=True)
        return False
    return model.startswith("local/")


def _warm_embedder() -> None:
    """Load the in-process embedder, then flip the readiness signal.

    Fire-and-forget; runs in a daemon thread so a slow model load never blocks
    startup. On failure the readiness signal stays clear and ``/health`` keeps
    reporting not-ready, which is the intended fail-safe.
    """
    from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder

    try:
        NomicEmbedder.get()._load()
        mark_embedder_ready()
        _LOGGER.info("In-process embedder warm; /health now reports ready.")
    except Exception:  # noqa: BLE001
        _LOGGER.exception("In-process embedder warmup failed; /health stays not-ready.")


def maybe_start_embedder_warmup() -> bool:
    """Spawn the non-blocking warm-before-ready thread when the gate is active.

    Returns:
        bool: True when a warmup thread was started (gate active), else False.
    """
    if not inprocess_local_gate_active():
        return False
    threading.Thread(target=_warm_embedder, daemon=True, name="embedder-warmup").start()
    return True


def _detected_worker_count() -> int | None:
    """Best-effort read of the configured uvicorn worker count.

    uvicorn exposes no worker-count env inside a worker process, so
    ``reflexio.server.__main__`` records it in ``REFLEXIO_SERVER_WORKERS``. A
    ``WEB_CONCURRENCY`` fallback covers gunicorn-style entrypoints. Returns
    ``None`` when neither is set (custom entrypoint) — the guard then warns that
    it could not verify the count rather than refusing.

    Returns:
        int | None: The detected worker count, or None if undetectable.
    """
    for name in (_ENV_WORKERS, _ENV_WEB_CONCURRENCY):
        raw = os.environ.get(name)
        if raw:
            try:
                return int(raw)
            except ValueError:
                continue
    return None


def _guard_workers_multiply_model() -> None:
    """D8(a): warn when in-process mode runs under multiple workers.

    In-process embedding loads one model copy per worker process, so a
    memory-bounded host that is fine with 1 worker can OOM at N. Warn-only: the
    worker count is not always reliably detectable from inside a worker.
    """
    if _provider() != _INPROCESS:
        return
    count = _detected_worker_count()
    if count is not None and count > 1:
        _LOGGER.warning(
            "%s=%s with %d uvicorn workers: the in-process embedder loads one "
            "model copy PER worker process, multiplying memory on a "
            "memory-bounded task. Prefer 1 worker for in-process embedding, or "
            "run the shared embedding daemon.",
            _ENV_PROVIDER,
            _INPROCESS,
            count,
        )
    elif count is None:
        _LOGGER.warning(
            "%s=%s but the uvicorn worker count could not be verified (neither "
            "%s nor %s is set). If this deployment runs multiple workers, each "
            "loads its own in-process model copy.",
            _ENV_PROVIDER,
            _INPROCESS,
            _ENV_WORKERS,
            _ENV_WEB_CONCURRENCY,
        )


def _guard_daemon_disable_half_pair() -> None:
    """D8(b): warn when the daemon-disable flag and provider disagree.

    ``REFLEXIO_DISABLE_LOCAL_EMBEDDING_DAEMON`` (truthy) and
    ``REFLEXIO_EMBEDDING_PROVIDER=inprocess`` describe the same intent from two
    angles and are meant to be flipped together. If exactly one is set the
    topology is half-configured (e.g. the daemon is disabled but requests still
    route to daemon mode, or vice-versa).
    """
    from reflexio.server.env_utils import env_truthy

    daemon_disabled = env_truthy(os.environ.get(_ENV_DISABLE_DAEMON, ""))
    inprocess = _provider() == _INPROCESS
    if daemon_disabled != inprocess:
        _LOGGER.warning(
            "Half-configured in-process embedding: %s=%s and %s=%s disagree. "
            "Set them together (disable the daemon AND select the in-process "
            "provider) or neither.",
            _ENV_DISABLE_DAEMON,
            daemon_disabled,
            _ENV_PROVIDER,
            _provider() or "<unset>",
        )


def run_startup_config_guards() -> None:
    """Run the D8 config guards once at server startup (idempotent, warn-only)."""
    _guard_workers_multiply_model()
    _guard_daemon_disable_half_pair()


__all__ = [
    "inprocess_local_gate_active",
    "is_embedder_ready",
    "mark_embedder_ready",
    "maybe_start_embedder_warmup",
    "reset_warmup_state_for_test",
    "run_startup_config_guards",
]
