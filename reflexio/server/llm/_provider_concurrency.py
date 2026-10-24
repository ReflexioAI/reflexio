"""Per-provider admission for remote LLM calls in one process.

Every resolved-provider call holds its provider's bounded semaphore until the
call and any retiring isolated worker finish. Saturation fails within the caller's remaining deadline and
raises ProviderCapSaturatedError; the generation fallback walk may advance to
another independently admitted provider. Capacity is never bypassed.
"""

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import litellm

from reflexio.removed_env import reject_removed_env
from reflexio.server.env_utils import env_str
from reflexio.server.llm.llm_utils import positive_int_env
from reflexio.server.operational_metrics import record_health

if TYPE_CHECKING:
    from reflexio.server.llm._litellm_types import ModelProvenance

logger = logging.getLogger(__name__)

reject_removed_env(os.environ, canonical="REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY")

_DEFAULT_MAX_CONCURRENCY = 8
REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY = positive_int_env(
    "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", _DEFAULT_MAX_CONCURRENCY, logger
)
# Admission wait is also clamped by the caller's remaining deadline.
_ACQUIRE_TIMEOUT_SECONDS = 30.0


class ProviderCapSaturatedError(Exception):
    """The provider has no available permit within the admission budget.

    Generation may advance to an independently admitted fallback; no provider
    request was dispatched on the saturated rung.
    """

    def __init__(
        self,
        message: str,
        *,
        first_parsed_provenance: "ModelProvenance | None" = None,
    ) -> None:
        super().__init__(message)
        self.first_parsed_provenance = first_parsed_provenance


def _parse_per_provider_cap() -> dict[str, int]:
    # Format: "zai=2,openai=16"
    raw = env_str("REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY_OVERRIDES", "")
    out: dict[str, int] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        provider, _, value = pair.partition("=")
        try:
            n = int(value.strip())
        except ValueError:
            continue
        if provider.strip() and n > 0:
            out[provider.strip()] = n
    return out


_per_provider_cap: dict[str, int] = _parse_per_provider_cap()


def _max_concurrency_for_provider(provider: str) -> int:
    return _per_provider_cap.get(provider, REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY)


_semaphores: dict[str, threading.BoundedSemaphore] = {}
_registry_lock = threading.Lock()


class _ProviderLease:
    """The caller and a retiring isolated worker share one admission permit."""

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self._references = 1
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._caller_finished: float | None = None
        self._outcome = "success"

    def retain(self) -> Callable[[], None]:
        with self._lock:
            if self._references == 0:
                raise RuntimeError("Cannot retain a released provider permit")
            self._references += 1
        return self.release

    def release(self) -> None:
        retirement = None
        with self._lock:
            self._references -= 1
            if self._references == 0:
                self._release()
                if self._caller_finished is not None:
                    retirement = time.monotonic() - self._caller_finished
        if retirement is not None:
            _record_duration(retirement, phase="retirement", outcome=self._outcome)

    def finish(self, outcome: str) -> None:
        """Finish the caller while retained workers still own their permits."""
        with self._lock:
            self._caller_finished = time.monotonic()
            elapsed = self._caller_finished - self._started
            self._outcome = outcome
        self.release()
        _record_duration(elapsed, phase="execution", outcome=outcome)


def _record_duration(elapsed: float, *, phase: str, outcome: str) -> None:
    record_health(
        "llm.provider.duration",
        elapsed,
        kind="distribution",
        unit="second",
        phase=phase,
        outcome=outcome,
    )


_current_provider_lease: ContextVar[_ProviderLease | None] = ContextVar(
    "current_provider_lease", default=None
)


def retain_current_provider_slot() -> Callable[[], None]:
    """Transfer worker ownership before leaving its caller's admission scope."""
    lease = _current_provider_lease.get()
    return lease.retain() if lease is not None else lambda: None


def _registered_custom_provider(model: str) -> str | None:
    """Resolve a REGISTERED custom provider from ``model``'s prefix.

    ``litellm.get_llm_provider`` raises for a custom provider even after the
    provider has registered itself into ``litellm.custom_provider_map`` --
    verified against a registered handler, not assumed. So the cap silently
    skipped every custom provider, and both of ours are CLI bridges that shell
    out to a local binary: ``claude-code`` and ``openclaw``. The providers least
    able to tolerate concurrency were the only ones the gate could not see
    (claude-smart#162).

    Only a provider present in ``custom_provider_map`` is accepted, so an
    arbitrary ``foo/bar`` model name still resolves to None and stays uncapped.

    Args:
        model (str): Model name, possibly ``"<provider>/<model>"``.

    Returns:
        str | None: The provider key when it names a registered custom
            provider, else None.
    """
    prefix, separator, _ = model.partition("/")
    if not separator or not prefix:
        return None
    registered = {
        entry.get("provider")
        for entry in (getattr(litellm, "custom_provider_map", None) or [])
    }
    return prefix if prefix in registered else None


def _provider_key(model: str) -> str | None:
    """Resolve the litellm provider for ``model``; None if unresolvable."""
    try:
        return litellm.get_llm_provider(model)[1]
    except Exception:  # noqa: BLE001 — unknown model must not be capped or raise
        return _registered_custom_provider(model)


def _get_semaphore(provider: str) -> threading.BoundedSemaphore:
    with _registry_lock:
        sem = _semaphores.get(provider)
        if sem is None:
            sem = threading.BoundedSemaphore(_max_concurrency_for_provider(provider))
            _semaphores[provider] = sem
        return sem


@contextmanager
def provider_slot(model: str, *, max_wait: float | None = None) -> Iterator[None]:
    """Cap concurrent in-flight calls to ``model``'s provider.

    Raises ``ProviderCapSaturatedError`` when admission expires. A caller may
    choose another provider through its existing fallback policy, but must not
    dispatch without a permit.

    Args:
        model (str): The model whose provider is capped.
        max_wait (float | None): Cap on the wait for a permit, in seconds, on top
            of the usual bound -- the caller's remaining deadline (the LLM
            ladder walk's budget).
    """
    from reflexio.server import search_runtime

    search_runtime.checkpoint()
    provider = _provider_key(model)
    if provider is None:
        token = _current_provider_lease.set(None)
        try:
            yield  # unknown provider → do not cap or retain an outer permit
        finally:
            _current_provider_lease.reset(token)
        return
    sem = _get_semaphore(provider)
    started = time.monotonic()
    acquired = False
    admission_outcome = "aborted"
    try:
        with search_runtime.phase("llm.provider_queue"):
            wait = search_runtime.remaining(_ACQUIRE_TIMEOUT_SECONDS)
            if max_wait is not None:
                wait = min(wait, max_wait)
            acquired = sem.acquire(timeout=wait)
            admission_outcome = "admitted" if acquired else "timeout"
    finally:
        _record_duration(
            time.monotonic() - started, phase="admission", outcome=admission_outcome
        )
    if not acquired:
        search_runtime.checkpoint()
        cap = _max_concurrency_for_provider(provider)
        logger.warning(
            "event=llm_provider_cap_saturated provider=%s cap=%d "
            "timeout=%.1fs — request not admitted",
            provider,
            cap,
            wait,
        )
        raise ProviderCapSaturatedError(
            f"provider {provider} concurrency cap {cap} saturated"
        )
    lease = _ProviderLease(sem.release)
    token = _current_provider_lease.set(lease)
    outcome = "success"
    try:
        search_runtime.checkpoint()
        yield
    except Exception:
        outcome = "failure"
        raise
    except BaseException:
        outcome = "aborted"
        raise
    finally:
        _current_provider_lease.reset(token)
        lease.finish(outcome)
