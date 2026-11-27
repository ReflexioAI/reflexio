"""Shared slowapi rate limiter for the OSS FastAPI app (Tier3 A2).

The ``limiter`` singleton and ``configure_rate_limiter`` live here so the domain
route modules can decorate handlers with ``@limiter.limit(...)`` without importing
from ``reflexio.server.api`` (which would be circular — ``api`` aggregates the
route modules). ``reflexio.server.api`` re-exports ``limiter`` and
``configure_rate_limiter`` so the existing enterprise imports keep resolving.
"""

import asyncio
import inspect
from collections.abc import Callable
from copy import copy
from functools import wraps
from pathlib import Path
from threading import Lock
from typing import Any, cast

from anyio.lowlevel import checkpoint_if_cancelled
from fastapi import Request
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import _find_route_handler, _should_exempt
from slowapi.util import get_remote_address
from slowapi.wrappers import Limit, LimitGroup
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from reflexio.server.rate_limit_policy import RateLimitPolicy
from reflexio.server.tracing import profile_step

_configured_key_func: Callable[..., str] | None = None


def get_rate_limit_key(request: Request) -> str:
    """Dispatch to the configured key function, defaulting to the client IP.

    Args:
        request (Request): The incoming request

    Returns:
        str: Configured rate limit key, or the client IP before configuration
    """
    if _configured_key_func is not None:
        return _configured_key_func(request)
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


def rate_limit_exceeded_handler(
    _request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Keep the error body and advise one rejecting-rule window, without I/O."""
    return JSONResponse(
        {"error": f"Rate limit exceeded: {exc.detail}"},
        status_code=429,
        headers={"Retry-After": str(cast(Limit, exc.limit).limit.get_expiry())},
    )


class _DynamicRuleRegistry(dict[str, list[LimitGroup]]):
    """Registration owns groups; each evaluation gets its own request binding."""

    def __getitem__(self, name: str) -> list[LimitGroup]:
        # Upstream registers with setdefault(...).append(...) and reads with
        # indexing before with_request mutates a group's request attribute.
        return [copy(group) for group in super().__getitem__(name)]


_quota_checks: set[asyncio.Task] = set()


def _finished_quota_check(task: asyncio.Task) -> None:
    _quota_checks.discard(task)
    if not task.cancelled():
        task.exception()  # A cancelled caller may no longer receive the error.


async def _run_quota_check(check: Callable, *args: Any) -> None:
    """Keep the existing worker token until its physical quota call finishes."""
    lock = Lock()
    started = False
    abandoned = False

    def owned_check() -> None:
        nonlocal started
        with lock:
            if abandoned:
                return
            started = True
        check(*args)

    # Shield the task owning AnyIO's existing token, not only its cancel scope:
    # raw Task.cancel() otherwise returns that token while its thread continues.
    task = asyncio.create_task(run_in_threadpool(owned_check))
    _quota_checks.add(task)
    task.add_done_callback(_finished_quota_check)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with lock:
            if not started:
                abandoned = True
                task.cancel()  # Drop queued work; a racing thread sees abandoned.
        raise


class _SDKLimiter(Limiter):
    """Keep upstream counters/registration; own configuration and async admission."""

    def __init__(self, key_func: Callable[..., str]):
        self._policy = RateLimitPolicy.from_environment().dependency_values()
        try:
            super().__init__(
                key_func=key_func,
                config_filename=str(Path(__file__).with_name("_rate_limit_empty.env")),
            )
        except Exception:
            # Storage URIs and parser errors may contain credentials.
            raise ValueError(
                "Invalid rate-limit backend or policy configuration"
            ) from None

        self._dynamic_route_limits = _DynamicRuleRegistry(self._dynamic_route_limits)

    def get_app_config(self, key: str, default_value: Any = None) -> Any:  # noqa: ARG002
        if key not in self._policy:
            raise RuntimeError("SlowAPI configuration contract changed; audit required")
        return self._policy[key]

    def _admission(self, decorate: Callable) -> Callable:
        def decorator(endpoint: Callable) -> Callable:
            # Keep upstream validation and live registration maps, but omit its
            # response wrapper: the SDK does not read success quota metadata.
            decorate(endpoint)
            signature = inspect.signature(endpoint)

            def check(args: tuple, kwargs: dict) -> None:
                if not self.enabled or not self._auto_check:
                    return
                arguments = signature.bind_partial(*args, **kwargs).arguments
                request = arguments.get("request", arguments.get("websocket"))
                if not isinstance(request, Request):
                    raise TypeError("A rate-limited endpoint requires a Request")
                if not getattr(request.state, "_rate_limiting_complete", False):
                    self._check_request_limit(request, endpoint, False)
                    request.state._rate_limiting_complete = True

            if inspect.iscoroutinefunction(endpoint):

                @wraps(endpoint)
                async def admitted(*args: Any, **kwargs: Any) -> Any:
                    if self.enabled and self._auto_check:
                        await _run_quota_check(check, args, kwargs)
                        await checkpoint_if_cancelled()
                    return await endpoint(*args, **kwargs)

                return admitted

            @wraps(endpoint)
            def admitted_sync(*args: Any, **kwargs: Any) -> Any:
                check(args, kwargs)
                return endpoint(*args, **kwargs)

            return admitted_sync

        return decorator

    def limit(self, *args: Any, **kwargs: Any) -> Callable:
        return self._admission(super().limit(*args, **kwargs))

    def shared_limit(self, *args: Any, **kwargs: Any) -> Callable:
        return self._admission(super().shared_limit(*args, **kwargs))


class RateLimitMiddleware:
    """Opt-in ASGI integration for applications migrating vendor middleware."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        app = scope["app"]
        limiter = app.state.limiter
        handler = _find_route_handler(app.routes, scope)
        if not limiter.enabled or _should_exempt(limiter, handler):
            return await self.app(scope, receive, send)
        request = Request(scope, receive=receive)
        if limiter._auto_check and not getattr(
            request.state, "_rate_limiting_complete", False
        ):
            try:
                await _run_quota_check(
                    limiter._check_request_limit, request, handler, True
                )
                await checkpoint_if_cancelled()
            except RateLimitExceeded as exc:
                response = rate_limit_exceeded_handler(request, exc)
                return await response(scope, receive, send)
            except Exception as exc:
                formatter = app.exception_handlers.get(type(exc))
                if formatter is None:
                    raise
                if inspect.iscoroutinefunction(formatter):
                    response = await formatter(request, exc)
                else:
                    response = await run_in_threadpool(formatter, request, exc)
                return await response(scope, receive, send)
        return await self.app(scope, receive, send)


# Initialize rate limiter after canonical server environment initialization.
limiter = _SDKLimiter(key_func=get_rate_limit_key)
_trace_external_rate_limit_backend(limiter)


def configure_rate_limiter(key_func: Callable[..., str]) -> None:
    """
    Replace the key function used by existing and future default route limits.

    This is the supported way to override the default IP-based key function
    (e.g. with an org-scoped or token-scoped variant in the enterprise layer).

    Args:
        key_func: A callable that accepts a Request and returns a string key.
            Passing ``get_rate_limit_key`` restores the default IP key.
    """
    global _configured_key_func
    # Slowapi captures a callable when each decorator runs. Keep its stable
    # dispatcher so routes imported before configuration use the override too.
    _configured_key_func = None if key_func is get_rate_limit_key else key_func
