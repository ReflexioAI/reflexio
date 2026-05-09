"""Reflexio instance cache with explicit invalidation and version-based auto-eviction."""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Final

from cachetools import TTLCache

from reflexio.lib.reflexio_lib import Reflexio

logger = logging.getLogger(__name__)

# Cache configuration
REFLEXIO_CACHE_MAX_SIZE = 100
REFLEXIO_CACHE_TTL_SECONDS = 3600  # 1 hour safety net

# Type alias for cache key: (org_id, storage_base_dir)
CacheKey = tuple[str, str | None]


# Sentinel returned by ``_probe_version_safe`` when the underlying probe
# raised an exception. Distinguishing "probe failed transiently" from
# "backend can't probe" matters: the former should NOT promote the entry
# to permanent unprobeable state, otherwise a single transient error
# (e.g. brief Postgres reconnect, file lock contention) would silently
# disable version-based auto-eviction for the lifetime of the entry.
_PROBE_FAILED: Final = object()

# Type alias for ``current_config_version()`` return values plus our
# private sentinel. ``None`` means "backend doesn't support probing"
# (permanent), the sentinel means "probe raised this time" (transient).
_ProbeResult = tuple[str, Any] | None | object


@dataclass
class _CacheEntry:
    """Per-org cache entry pairing a Reflexio instance with the config version stamped at load time.

    The ``cached_version`` is whatever ``Reflexio.current_config_version()``
    returned when the instance was constructed. On each cache hit we
    re-probe and evict the entry if the value changed — this catches
    out-of-band config mutations (file edits, sibling-replica writes,
    direct DB updates) that don't go through ``invalidate_reflexio_cache``.

    Attributes:
        reflexio (Reflexio): The cached Reflexio instance.
        cached_version (tuple[str, Any] | None): The version stamp
            captured at load time. ``None`` means "no probe available";
            entries with ``None`` are never auto-evicted (they fall
            through to the TTL safety net). Probe failures during
            construction never produce ``None`` here — they're recorded
            as the same value the next probe attempt would compare
            against, so a later successful probe can still evict the
            stale entry.
    """

    reflexio: Reflexio
    cached_version: tuple[str, Any] | None


# Module-level cache and lock
_reflexio_cache: TTLCache = TTLCache(
    maxsize=REFLEXIO_CACHE_MAX_SIZE, ttl=REFLEXIO_CACHE_TTL_SECONDS
)
_reflexio_cache_lock = threading.Lock()


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
    try:
        return reflexio.current_config_version()
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        logger.warning(
            "Failed to probe config version for org %s: %s — keeping entry warm",
            reflexio.org_id,
            exc,
        )
        return _PROBE_FAILED


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
    with _reflexio_cache_lock:
        entry: _CacheEntry | None = _reflexio_cache.get(cache_key)

    if entry is not None:
        cached_version = entry.cached_version
        # Skip probing when we have nothing to compare against — a
        # ``None`` stamp means the backend can't probe cheaply, so we
        # rely on TTL + explicit invalidation instead.
        if cached_version is None:
            return entry.reflexio
        current_version = _probe_version_safe(entry.reflexio)
        # Transient probe failure: keep the cached instance for this
        # request but DON'T mutate the stamp — the next request will
        # try again, so a brief outage doesn't permanently disable
        # eviction the way collapsing failures into ``None`` would.
        if current_version is _PROBE_FAILED:
            return entry.reflexio
        if current_version == cached_version:
            return entry.reflexio
        # Stale entry. Evict only if the cached version still matches
        # the one we just compared against — another thread may have
        # already replaced the entry while we were probing.
        with _reflexio_cache_lock:
            existing = _reflexio_cache.get(cache_key)
            if existing is not None and existing.cached_version == cached_version:
                del _reflexio_cache[cache_key]

    # Cache miss (or just-evicted stale entry) - create a new instance
    # outside the lock to avoid blocking concurrent requests for other orgs.
    reflexio = Reflexio(org_id=org_id, storage_base_dir=storage_base_dir)
    new_version = _probe_version_safe(reflexio)

    # Construction-time probe failure: serve this request from the
    # newly-built instance but DON'T cache it. Caching with
    # ``cached_version=None`` would conflate the entry with a
    # legitimately-unprobeable backend and permanently disable
    # auto-eviction. Skipping the cache means the next request pays a
    # construction cost, but version-based eviction is preserved as
    # soon as the backend recovers.
    if new_version is _PROBE_FAILED:
        return reflexio

    new_entry = _CacheEntry(
        reflexio=reflexio,
        cached_version=new_version,  # type: ignore[arg-type]
    )

    with _reflexio_cache_lock:
        # Double-check in case another thread populated while we were constructing.
        existing = _reflexio_cache.get(cache_key)
        if existing is None:
            _reflexio_cache[cache_key] = new_entry
            return reflexio
        return existing.reflexio


def invalidate_reflexio_cache(org_id: str, storage_base_dir: str | None = None) -> bool:
    """Invalidate cached Reflexio for specific org.

    Call this after set_config to ensure next request gets fresh instance.

    Args:
        org_id (str): Organization ID to invalidate
        storage_base_dir (Optional[str]): Base directory for storage

    Returns:
        bool: True if entry was removed, False if not found
    """
    cache_key: CacheKey = (org_id, storage_base_dir)
    with _reflexio_cache_lock:
        if cache_key in _reflexio_cache:
            del _reflexio_cache[cache_key]
            return True
        return False


def clear_reflexio_cache() -> None:
    """Clear entire cache (for testing/admin)."""
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
