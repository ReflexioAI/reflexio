"""Per-provider, per-instance concurrency cap for remote LLM calls.

A bounded semaphore keyed by LLM provider. Acquired in the PARENT process
around a provider call so N tasks don't fan into a provider-429 storm.
By default, on saturation it fails OPEN (proceeds without a permit) after a
bounded wait, never blocking unboundedly — that would re-open the hung-provider
stall class through the limiter. Providers listed in
``REFLEXIO_LLM_FAIL_CLOSED_PROVIDERS`` instead fail CLOSED — they raise
``ProviderCapSaturatedError`` on saturation to protect a fixed-quota
subscription (e.g. the Z.ai GLM coding plan used as a fallback), which the
fallback walk treats as an advance-worthy rung failure. Part of Scalability
Workstream C (design §5). 429 recovery stays with the fallback ladder (Decision A).
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import litellm

from reflexio.server.env_utils import env_str
from reflexio.server.llm.llm_utils import positive_int_env

if TYPE_CHECKING:
    from reflexio.server.llm._litellm_types import ModelProvenance

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONCURRENCY = 8
REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY = positive_int_env(
    "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", _DEFAULT_MAX_CONCURRENCY, logger
)
# Bounded wait before failing open. Long enough to queue a burst, short enough
# to never park a request thread near the ~hard-timeout ceiling. Module constant
# (not an env var) to keep C3 to a single knob.
_ACQUIRE_TIMEOUT_SECONDS = 30.0


class ProviderCapSaturatedError(Exception):
    """Raised when a fail-closed provider's concurrency cap is saturated.

    Unlike the fail-open default, providers in ``REFLEXIO_LLM_FAIL_CLOSED_PROVIDERS``
    (e.g. a fixed-quota Z.ai coding-plan fallback) raise this instead of
    proceeding without a permit, so a mass-fallback event degrades rather than
    exhausting the subscription. The fallback walker treats it as an
    advance-worthy rung failure.
    """

    def __init__(
        self,
        message: str,
        *,
        first_parsed_provenance: "ModelProvenance | None" = None,
    ) -> None:
        super().__init__(message)
        self.first_parsed_provenance = first_parsed_provenance


def _parse_fail_closed() -> frozenset[str]:
    raw = env_str("REFLEXIO_LLM_FAIL_CLOSED_PROVIDERS", "")
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


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


_fail_closed_providers: frozenset[str] = _parse_fail_closed()
_per_provider_cap: dict[str, int] = _parse_per_provider_cap()


def _max_concurrency_for_provider(provider: str) -> int:
    return _per_provider_cap.get(provider, REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY)


_semaphores: dict[str, threading.BoundedSemaphore] = {}
_registry_lock = threading.Lock()


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

    Fails OPEN on saturation by default; fails CLOSED (raises
    ``ProviderCapSaturatedError``) for providers in
    ``REFLEXIO_LLM_FAIL_CLOSED_PROVIDERS``.

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
        yield  # unknown provider → do not cap
        return
    sem = _get_semaphore(provider)
    with search_runtime.phase("llm.provider_queue"):
        wait = search_runtime.remaining(_ACQUIRE_TIMEOUT_SECONDS)
        if max_wait is not None:
            wait = min(wait, max_wait)
        acquired = sem.acquire(timeout=wait)
    if not acquired:
        search_runtime.checkpoint()
        cap = _max_concurrency_for_provider(provider)
        if provider in _fail_closed_providers:
            logger.warning(
                "event=llm_provider_cap_saturated provider=%s cap=%d "
                "timeout=%.1fs — FAIL-CLOSED (raising to protect quota)",
                provider,
                cap,
                _ACQUIRE_TIMEOUT_SECONDS,
            )
            raise ProviderCapSaturatedError(
                f"provider {provider} concurrency cap {cap} saturated"
            )
        logger.warning(
            "event=llm_provider_cap_saturated provider=%s cap=%d "
            "timeout=%.1fs — proceeding without permit (fail-open)",
            provider,
            cap,
            _ACQUIRE_TIMEOUT_SECONDS,
        )
        yield
        return
    try:
        search_runtime.checkpoint()
        yield
    finally:
        sem.release()
