"""Reflexio instance cache with explicit invalidation and version-based auto-eviction."""

import logging
import threading
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

from cachetools import TTLCache

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.llm_utils import positive_int_env
from reflexio.server.services.configurator.config_storage import (
    CONFIG_VERSION_PROBE_FAILED as _PROBE_FAILED,
)
from reflexio.server.services.configurator.config_storage import ConfigVersionProbe
from reflexio.server.tracing import profile_step

logger = logging.getLogger(__name__)

# Cache configuration.
# The default must exceed the combined working set of the background
# schedulers that page through the whole fleet (two independent 100-org
# pages) plus request-path headroom — at 100, scheduler sweeps on a
# 100+-org fleet would continuously evict warm request-path entries.
REFLEXIO_CACHE_MAX_SIZE = positive_int_env("REFLEXIO_CACHE_MAX_SIZE", 512, logger)
REFLEXIO_CACHE_TTL_SECONDS = 3600  # 1 hour safety net

# Type alias for cache key: (org_id, storage_base_dir)
CacheKey = tuple[str, str | None]


# Sentinel returned by ``_probe_version_safe`` when the underlying probe
# raised an exception. Distinguishing "probe failed transiently" from
# "backend can't probe" matters: the former should NOT promote the entry
# to permanent unprobeable state, otherwise a single transient error
# (e.g. brief Postgres reconnect, file lock contention) would silently
# disable version-based auto-eviction for the lifetime of the entry.
# Type alias for version probe results plus the shared failure sentinel.
# ``None`` means "backend doesn't support probing"
# (permanent), the sentinel means "probe raised this time" (transient).
_ProbeResult = ConfigVersionProbe


@dataclass
class _CacheEntry:
    """Per-org cache entry pairing a Reflexio instance with the config version stamped at load time.

    The ``cached_version`` is captured before the instance loads config,
    so a concurrent write cannot stamp an older config with a newer revision.
    On each cache hit we re-probe and evict the entry if the value changed — this catches
    out-of-band config mutations (file edits, sibling-replica writes,
    direct DB updates) that don't go through ``invalidate_reflexio_cache``.

    Attributes:
        reflexio (Reflexio): The cached Reflexio instance.
        cached_version (tuple[str, Any] | None): The version stamp
            captured at load time. ``None`` means "no probe available";
            entries with ``None`` are never auto-evicted (they fall
            through to the TTL safety net). Probe failures during
            construction prevent the instance from being cached, so a later
            successful construction can restore version-based eviction.
    """

    reflexio: Reflexio
    cached_version: tuple[str, Any] | None


# Module-level cache and lock
_reflexio_cache: TTLCache = TTLCache(
    maxsize=REFLEXIO_CACHE_MAX_SIZE, ttl=REFLEXIO_CACHE_TTL_SECONDS
)
_reflexio_cache_lock = threading.Lock()
_REFLEXIO_CONSTRUCTION_LOCK_STRIPES = 64
_reflexio_construction_locks = tuple(
    threading.Lock() for _ in range(_REFLEXIO_CONSTRUCTION_LOCK_STRIPES)
)


def _get_construction_lock(cache_key: CacheKey) -> threading.Lock:
    """Return a bounded striped lock for cold Reflexio construction."""
    return _reflexio_construction_locks[
        hash(cache_key) % _REFLEXIO_CONSTRUCTION_LOCK_STRIPES
    ]


@contextmanager
def _construction_guard(cache_key: CacheKey) -> Iterator[None]:
    from reflexio.server import search_runtime

    lock = _get_construction_lock(cache_key)
    if search_runtime.current() is None:
        with lock:
            yield
        return

    acquired = False
    try:
        with profile_step("reflexio.cache.construction_wait"):
            # Recheck disconnects while the constructor retains ownership;
            # the caller's existing deadline also bounds every lock attempt.
            while not acquired:
                acquired = lock.acquire(timeout=search_runtime.remaining(0.1))
        search_runtime.checkpoint()
        yield
    finally:
        if acquired:
            lock.release()


def _probe_version_safe(reflexio: Reflexio) -> _ProbeResult:
    """Probe the current config version, distinguishing failure from "no probe".

    A failing probe must never break a cache hit, but it also must not
    be conflated with a backend that legitimately can't probe (which
    returns ``None``). Conflating the two would let a single transient
    failure permanently disable auto-eviction for the entry.

    Args:
        reflexio (Reflexio): The cached Reflexio instance to probe.

    Returns:
        ``tuple[str, Any]`` on success, ``None`` when the backend
        intentionally doesn't expose a version (permanent), or the
        ``_PROBE_FAILED`` sentinel when the call raised (transient).
    """
    from reflexio.server.search_runtime import config_version

    try:
        return config_version(
            "reflexio", reflexio, None, reflexio.current_config_version
        )
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        logger.warning(
            "Failed to probe config version for org %s: %s — keeping entry warm",
            reflexio.org_id,
            exc,
        )
        return _PROBE_FAILED


def prefetch_config_version(org_id: str) -> Future[Any] | None:
    """Check a warm search instance while independent request reads run.

    This only warms the request memo: eviction and construction remain owned
    by get_reflexio after billing passes. Non-search and cold requests do nothing.
    """
    from reflexio.server import search_runtime
    from reflexio.server.services.unified_search_service import submit_search_work

    if search_runtime.current() is None:
        return None
    with _reflexio_cache_lock:
        entry = _reflexio_cache.get((org_id, None))
    if entry is None or entry.cached_version is None:
        return None
    return submit_search_work(lambda: _probe_version_safe(entry.reflexio))


def _entry_is_fresh(entry: _CacheEntry) -> bool:
    """Probe a cached entry, retaining its stamp across transient failures."""
    if entry.cached_version is None:
        return True
    with profile_step("reflexio.cache.version_probe", cache_state="hit") as span:
        current_version = _probe_version_safe(entry.reflexio)
        span.set_data("probe_failed", current_version is _PROBE_FAILED)
        span.set_data(
            "probe_supported",
            current_version is not None and current_version is not _PROBE_FAILED,
        )
    return current_version is _PROBE_FAILED or current_version == entry.cached_version


def get_reflexio(org_id: str, storage_base_dir: str | None = None) -> Reflexio:
    """Get or create a cached Reflexio instance.

    On cache hit, the entry's stamped config version is re-probed. If
    the persisted version has changed (e.g. config file mtime bumped,
    sibling replica wrote new DB version), the entry is evicted and a
    fresh instance is constructed.

    Args:
        org_id (str): Organization ID
        storage_base_dir (Optional[str]): Base directory for storage (self-host mode)

    Returns:
        Reflexio: Cached or newly created instance
    """
    cache_key: CacheKey = (org_id, storage_base_dir)

    # Cache lookup — held briefly to extract the entry, then released
    # before doing any I/O (file stat, DB query) for the version probe.
    with profile_step("reflexio.cache.lookup") as span:
        with _reflexio_cache_lock:
            entry: _CacheEntry | None = _reflexio_cache.get(cache_key)
        span.set_data("cache_hit", entry is not None)
        span.set_data(
            "has_cached_version",
            entry is not None and entry.cached_version is not None,
        )

    if entry is not None:
        if _entry_is_fresh(entry):
            return entry.reflexio
        # Stale entry. Evict only the entry we inspected: another thread may have
        # already replaced the entry while we were probing.
        # Do NOT close() the evicted instance here: callers receive
        # instances after the cache lock is released, so another thread
        # may still be using the one being evicted. Closing its client
        # pools would cause spurious mid-request failures; correct
        # cleanup would need refcounting, for no demonstrated leak.
        with profile_step("reflexio.cache.evict_stale") as span:
            with _reflexio_cache_lock:
                existing = _reflexio_cache.get(cache_key)
                evicted = existing is entry
                if evicted:
                    del _reflexio_cache[cache_key]
            span.set_data("evicted", evicted)

    # Cache miss (or just-evicted stale entry). Only one request per cache
    # key may construct at a time; construction can initialize storage and
    # run migrations, so parallel cold starts for the same org are not safe.
    with _construction_guard(cache_key):
        with _reflexio_cache_lock:
            entry = _reflexio_cache.get(cache_key)
        if entry is not None:
            # A constructor we waited behind may have loaded before a sibling
            # replica wrote. Validate its captured stamp just like a warm hit.
            if _entry_is_fresh(entry):
                return entry.reflexio
            with _reflexio_cache_lock:
                _reflexio_cache.pop(cache_key, None)

        logger.info(
            "Constructing Reflexio instance for org %s storage_base_dir=%s",
            org_id,
            storage_base_dir,
        )
        with profile_step("reflexio.cache.construct"):
            reflexio = Reflexio(org_id=org_id, storage_base_dir=storage_base_dir)
        with profile_step("reflexio.cache.version_probe", cache_state="miss") as span:
            new_version = reflexio.loaded_config_version()
            span.set_data("probe_failed", new_version is _PROBE_FAILED)
            span.set_data(
                "probe_supported",
                new_version is not None and new_version is not _PROBE_FAILED,
            )

        # Construction-time probe failure: serve this request from the
        # newly-built instance but DON'T cache it. Caching with
        # ``cached_version=None`` would conflate the entry with a
        # legitimately-unprobeable backend and permanently disable
        # auto-eviction. Skipping the cache means the next request pays a
        # construction cost, but version-based eviction is preserved as
        # soon as the backend recovers.
        if new_version is _PROBE_FAILED:
            logger.warning(
                "Constructed uncached Reflexio instance for org %s because config version probe failed",
                org_id,
            )
            return reflexio

        new_entry = _CacheEntry(
            reflexio=reflexio,
            cached_version=new_version,
        )

        with profile_step("reflexio.cache.store") as span:
            with _reflexio_cache_lock:
                _reflexio_cache[cache_key] = new_entry
            span.set_data("stored", True)
            logger.info(
                "Stored Reflexio instance for org %s storage_base_dir=%s",
                org_id,
                storage_base_dir,
            )
            return reflexio


def get_cached_request_context(
    org_id: str, storage_base_dir: str | None = None
) -> RequestContext:
    """Get the RequestContext owned by the cached Reflexio instance.

    For background schedulers that fan out per-org work: delegates to
    :func:`get_reflexio` so they share the same cache entry, per-hit
    config-version eviction, striped construction locks, and every
    existing ``invalidate_reflexio_cache`` call site as the request
    path — instead of rebuilding a ``RequestContext`` (config decrypt,
    storage client pools, LiteLLM client) on every tick.

    Named distinctly from the FastAPI ``get_request_context`` dependency
    in ``api_endpoints/request_context.py`` to avoid collisions.

    Args:
        org_id (str): Organization ID
        storage_base_dir (Optional[str]): Base directory for storage (self-host mode)

    Returns:
        RequestContext: The context of the cached (or newly constructed) instance.
    """
    return get_reflexio(org_id, storage_base_dir).request_context


def invalidate_reflexio_cache(org_id: str, storage_base_dir: str | None = None) -> bool:
    """Invalidate cached Reflexio for specific org.

    Wait for in-flight construction, then evict so the next request reloads.

    Args:
        org_id (str): Organization ID to invalidate
        storage_base_dir (Optional[str]): Base directory for storage

    Returns:
        bool: True if entry was removed, False if not found
    """
    cache_key: CacheKey = (org_id, storage_base_dir)
    with _get_construction_lock(cache_key), _reflexio_cache_lock:
        if cache_key in _reflexio_cache:
            del _reflexio_cache[cache_key]
            return True
        return False


def clear_reflexio_cache() -> None:
    """Clear the cache after fencing all in-flight constructors."""
    with ExitStack() as locks:
        for construction_lock in _reflexio_construction_locks:
            locks.enter_context(construction_lock)
        with _reflexio_cache_lock:
            _reflexio_cache.clear()


def get_cache_stats() -> dict:
    """Get cache statistics for monitoring.

    Returns:
        dict: Cache statistics including current size, max size, and TTL
    """
    with _reflexio_cache_lock:
        return {
            "current_size": len(_reflexio_cache),
            "max_size": REFLEXIO_CACHE_MAX_SIZE,
            "ttl_seconds": REFLEXIO_CACHE_TTL_SECONDS,
        }
