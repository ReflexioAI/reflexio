"""Shared slowapi rate limiter for the OSS FastAPI app (Tier3 A2).

The ``limiter`` singleton and ``configure_rate_limiter`` live here so the domain
route modules can decorate handlers with ``@limiter.limit(...)`` without importing
from ``reflexio.server.api`` (which would be circular — ``api`` aggregates the
route modules). ``reflexio.server.api`` re-exports ``limiter`` and
``configure_rate_limiter`` so the existing enterprise imports keep resolving.
"""

from collections.abc import Callable
from typing import Any

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from reflexio.server.tracing import profile_step


def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key based on IP address.

    Args:
        request (Request): The incoming request

    Returns:
        str: Rate limit key (IP address)
    """
    return get_remote_address(request)


def _storage_backend_name(limiter_obj: Limiter) -> str:
    storage = getattr(limiter_obj, "_storage", None)
    if storage is None:
        return "unknown"
    return storage.__class__.__name__


def _trace_external_rate_limit_backend(limiter_obj: Limiter) -> None:
    """Trace rate-limit storage hits when the backend is an external service."""
    backend = _storage_backend_name(limiter_obj)
    if backend == "MemoryStorage":
        return

    strategy = getattr(limiter_obj, "limiter", None)
    if strategy is None or getattr(strategy, "_reflexio_traced", False):
        return

    original_hit = strategy.hit

    def traced_hit(item: Any, *identifiers: str, cost: int = 1) -> bool:
        with profile_step(
            "rate_limit.backend_hit",
            storage_backend=backend,
            strategy=strategy.__class__.__name__,
            cost=cost,
        ):
            return original_hit(item, *identifiers, cost=cost)

    strategy.hit = traced_hit
    strategy._reflexio_traced = True


# Initialize rate limiter
limiter = Limiter(key_func=get_rate_limit_key)
_trace_external_rate_limit_backend(limiter)


def configure_rate_limiter(key_func: Callable[..., str]) -> None:
    """
    Replace the rate limiter's key function.

    This is the supported way to override the default IP-based key function
    (e.g. with an org-scoped or token-scoped variant in the enterprise layer).

    Args:
        key_func: A callable that accepts a Request and returns a string key.
    """
    limiter._key_func = key_func  # type: ignore[reportAttributeAccessIssue]
