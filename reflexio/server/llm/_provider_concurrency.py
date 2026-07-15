"""Per-provider, per-instance concurrency cap for remote LLM calls.

A bounded, fail-open semaphore keyed by LLM provider. Acquired in the PARENT
process around a provider call so N tasks don't fan into a provider-429 storm.
On saturation it fails OPEN (proceeds without a permit) after a bounded wait,
never blocking unboundedly — that would re-open the hung-provider stall class
(Sentry PYTHON-FASTAPI-62) through the limiter. Part of Scalability Workstream C
(design §5). 429 recovery stays with the fallback ladder (Decision A).
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import litellm

from reflexio.server.llm.llm_utils import positive_int_env

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONCURRENCY = 8
REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY = positive_int_env(
    "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", _DEFAULT_MAX_CONCURRENCY, logger
)
# Bounded wait before failing open. Long enough to queue a burst, short enough
# to never park a request thread near the ~hard-timeout ceiling. Module constant
# (not an env var) to keep C3 to a single knob.
_ACQUIRE_TIMEOUT_SECONDS = 30.0

_semaphores: dict[str, threading.BoundedSemaphore] = {}
_registry_lock = threading.Lock()


def _provider_key(model: str) -> str | None:
    """Resolve the litellm provider for ``model``; None if unresolvable."""
    try:
        return litellm.get_llm_provider(model)[1]
    except Exception:  # noqa: BLE001 — unknown model must not be capped or raise
        return None


def _get_semaphore(provider: str) -> threading.BoundedSemaphore:
    with _registry_lock:
        sem = _semaphores.get(provider)
        if sem is None:
            sem = threading.BoundedSemaphore(REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY)
            _semaphores[provider] = sem
        return sem


@contextmanager
def provider_slot(model: str) -> Iterator[None]:
    """Cap concurrent in-flight calls to ``model``'s provider (fail-open)."""
    provider = _provider_key(model)
    if provider is None:
        yield  # unknown provider → do not cap
        return
    sem = _get_semaphore(provider)
    acquired = sem.acquire(timeout=_ACQUIRE_TIMEOUT_SECONDS)
    if not acquired:
        logger.warning(
            "event=llm_provider_cap_saturated provider=%s cap=%d "
            "timeout=%.1fs — proceeding without permit (fail-open)",
            provider,
            REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY,
            _ACQUIRE_TIMEOUT_SECONDS,
        )
        yield
        return
    try:
        yield
    finally:
        sem.release()
