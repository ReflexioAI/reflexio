"""Reflexio instance cache with explicit invalidation and version-based auto-eviction."""

import logging
import threading
from dataclasses import dataclass
from typing import Any

from cachetools import TTLCache

from reflexio.lib.reflexio_lib import Reflexio

logger = logging.getLogger(__name__)

# Cache configuration
REFLEXIO_CACHE_MAX_SIZE = 100
REFLEXIO_CACHE_TTL_SECONDS = 3600  # 1 hour safety net

# Type alias for cache key: (org_id, storage_base_dir)
CacheKey = tuple[str, str | None]


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
            through to the TTL safety net).
    """

    reflexio: Reflexio
    cached_version: tuple[str, Any] | None


# Module-level cache and lock
_reflexio_cache: TTLCache = TTLCache(
    maxsize=REFLEXIO_CACHE_MAX_SIZE, ttl=REFLEXIO_CACHE_TTL_SECONDS
)
_reflexio_cache_lock = threading.Lock()


def _probe_version_safe(reflexio: Reflexio) -> tuple[str, Any] | None:
    """Probe the current config version, swallowing errors.

    A failing probe must never break a cache hit — falling back to
    ``None`` means "treat as fresh" and the request still succeeds.

    Args:
        reflexio (Reflexio): The cached Reflexio instance to probe.

    Returns:
        tuple[str, Any] | None: The probe result, or ``None`` on error.
    """
    try:
        return reflexio.current_config_version()
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        logger.warning(
            "Failed to probe config version for org %s: %s — assuming fresh",
            reflexio.org_id,
            exc,
        )
        return None


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
        if current_version is None or current_version == cached_version:
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
    new_entry = _CacheEntry(reflexio=reflexio, cached_version=new_version)

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
