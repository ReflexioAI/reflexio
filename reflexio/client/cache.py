"""In-memory cache module for Reflexio client."""

import hashlib
import json
import threading
from datetime import datetime, timedelta
from typing import Any


class InMemoryCache:
    """
    Thread-safe in-memory cache with time-based expiration.

    Stores function results in memory to avoid redundant API calls.
    Cache entries expire after a specified TTL (default: 10 minutes).
    """

    def __init__(self, ttl_seconds: int = 600):
        """
        Initialize the in-memory cache.

        Args:
            ttl_seconds (int): Time-to-live for cache entries in seconds. Default is 600 (10 minutes).
        """
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def _generate_cache_key(self, method_name: str, **kwargs) -> str:
        """
        Generate a unique cache key from method name and parameters.

        Handles datetime objects, None values, and nested dictionaries.

        Args:
            method_name (str): Name of the cached method
            **kwargs: Method parameters to include in the cache key

        Returns:
            str: Unique cache key hash
        """
        # Create a normalized representation of the parameters
        normalized_params = {}
        for key, value in sorted(kwargs.items()):
            if isinstance(value, datetime):
                # Convert datetime to ISO format string
                normalized_params[key] = value.isoformat()
            elif value is None:
                normalized_params[key] = None
            elif hasattr(value, "model_dump"):
                # Handle Pydantic models
                normalized_params[key] = value.model_dump()
            elif isinstance(value, dict):
                normalized_params[key] = value
            else:
                normalized_params[key] = value

        # Create a stable JSON representation
        key_data = {"method": method_name, "params": normalized_params}
        key_string = json.dumps(key_data, sort_keys=True, default=str)

        # Generate hash for the cache key
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(self, method_name: str, **kwargs) -> Any | None:
        """
        Retrieve a cached value if it exists and is not expired.

        Args:
            method_name (str): Name of the cached method
            **kwargs: Method parameters used to generate cache key

        Returns:
            Optional[Any]: Cached value if found and valid, None otherwise
        """
        cache_key = self._generate_cache_key(method_name, **kwargs)

        with self._lock:
            if cache_key not in self._cache:
                return None

            entry = self._cache[cache_key]

            # Check if entry has expired
            if datetime.now() > entry["expires_at"]:
                # Remove expired entry
                del self._cache[cache_key]
                return None

            return entry["data"]

    def set(self, method_name: str, value: Any, **kwargs) -> None:
        """
        Store a value in the cache with automatic expiration.

        Args:
            method_name (str): Name of the cached method
            value (Any): Value to cache
            **kwargs: Method parameters used to generate cache key
        """
        cache_key = self._generate_cache_key(method_name, **kwargs)

        with self._lock:
            self._cache[cache_key] = {
                "data": value,
                "timestamp": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=self._ttl_seconds),
            }

            # Cleanup expired entries periodically (every 100 sets)
            if len(self._cache) % 100 == 0:
                self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        """
        Remove all expired entries from the cache.

        Note: This method assumes the lock is already acquired by the caller.
        """
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items() if now > entry["expires_at"]
        ]
        for key in expired_keys:
            del self._cache[key]
