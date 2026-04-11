"""Reflexio instance cache with explicit invalidation."""

import threading

from cachetools import TTLCache

from reflexio.lib.reflexio_lib import Reflexio

# Cache configuration
REFLEXIO_CACHE_MAX_SIZE = 100
REFLEXIO_CACHE_TTL_SECONDS = 3600  # 1 hour safety net

# Type alias for cache key: (org_id, storage_base_dir)
CacheKey = tuple[str, str | None]

# Module-level cache and lock
_reflexio_cache: TTLCache = TTLCache(
    maxsize=REFLEXIO_CACHE_MAX_SIZE, ttl=REFLEXIO_CACHE_TTL_SECONDS
)
_reflexio_cache_lock = threading.Lock()


def get_reflexio(org_id: str, storage_base_dir: str | None = None) -> Reflexio:
    """Get or create cached Reflexio instance.

    Args:
        org_id (str): Organization ID
        storage_base_dir (Optional[str]): Base directory for storage (self-host mode)

    Returns:
        Reflexio: Cached or newly created instance
    """
    cache_key: CacheKey = (org_id, storage_base_dir)

    with _reflexio_cache_lock:
        if cache_key in _reflexio_cache:
            return _reflexio_cache[cache_key]

    # Cache miss - create new instance (outside lock to avoid blocking)
    reflexio = Reflexio(org_id=org_id, storage_base_dir=storage_base_dir)

    with _reflexio_cache_lock:
        # Double-check in case another thread created it
        if cache_key not in _reflexio_cache:
            _reflexio_cache[cache_key] = reflexio
        return _reflexio_cache[cache_key]


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
