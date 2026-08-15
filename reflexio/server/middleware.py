"""HTTP middleware for the OSS FastAPI app (Tier3 A2 extraction from api.py).

Holds the five middleware classes ``create_app`` wires in (in that order) plus
the request-limit / CORS-origin helpers and their tunable constants. No
middleware reads mutable module-global app state — every read is of an
import-time constant or an env var — so this extraction is behavior-preserving.
"""

import asyncio
import logging
import os

from anyio.to_thread import current_default_thread_limiter
from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from reflexio.server.correlation import correlation_id_var, generate_correlation_id

logger = logging.getLogger(__name__)

# Bot protection configuration
REQUEST_TIMEOUT_SECONDS = 60
SYNC_REQUEST_TIMEOUT_SECONDS = (
    600  # Longer timeout for synchronous long-running processing.
)
SYNC_REQUEST_PATHS = frozenset(
    {"/api/review_user_playbooks", "/api/run_playbook_aggregation"}
)
SUSPICIOUS_USER_AGENTS = ["bot", "crawler", "spider", "scraper", "curl", "wget"]
ALLOWED_EMPTY_UA_PATHS = ["/health", "/"]  # Paths that allow empty user agents
DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024


def _resolve_cors_origins() -> list[str]:
    """Resolve browser origins allowed to make credentialed CORS requests."""
    configured_origins = os.getenv("REFLEXIO_ALLOWED_ORIGINS", "").strip()
    if configured_origins:
        origins = [
            origin.strip().rstrip("/")
            for origin in configured_origins.split(",")
            if origin.strip()
        ]
        return origins or ["http://localhost:8080"]

    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    if frontend_url:
        return [frontend_url.rstrip("/")]

    return ["http://localhost:8080"]


def _max_body_bytes_from_env() -> int:
    raw_value = os.getenv("REFLEXIO_MAX_BODY_BYTES", str(DEFAULT_MAX_BODY_BYTES))
    try:
        max_bytes = int(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid REFLEXIO_MAX_BODY_BYTES=%r; using %s",
            raw_value,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    if max_bytes <= 0:
        logger.warning(
            "Ignoring non-positive REFLEXIO_MAX_BODY_BYTES=%r; using %s",
            raw_value,
            DEFAULT_MAX_BODY_BYTES,
        )
        return DEFAULT_MAX_BODY_BYTES
    return max_bytes


class BotProtectionMiddleware(BaseHTTPMiddleware):
    """Middleware to detect and block suspicious bot-like requests."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and block suspicious patterns.

        Args:
            request (Request): The incoming request
            call_next (RequestResponseEndpoint): Next middleware/handler in chain

        Returns:
            Response: The response from the next handler or a 403 JSON response
        """
        from starlette.responses import JSONResponse

        user_agent = request.headers.get("user-agent", "").lower()
        path = request.url.path

        # Allow health check and root without user agent
        if path not in ALLOWED_EMPTY_UA_PATHS:
            # Block requests with no user agent
            if not user_agent:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Forbidden: Missing user agent"},
                )

            # Block requests with suspicious user agents
            for suspicious in SUSPICIOUS_USER_AGENTS:
                if suspicious in user_agent:
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={"detail": "Forbidden: Suspicious user agent"},
                    )

        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeout."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request with timeout enforcement.

        Args:
            request (Request): The incoming request
            call_next (RequestResponseEndpoint): Next middleware/handler in chain

        Returns:
            Response: The response from the next handler or a 504 JSON response
        """
        from starlette.responses import JSONResponse

        # Use longer timeout for synchronous processing requests
        timeout = REQUEST_TIMEOUT_SECONDS
        if (
            request.url.path in SYNC_REQUEST_PATHS
            or request.query_params.get("wait_for_response", "").lower() == "true"
        ):
            timeout = SYNC_REQUEST_TIMEOUT_SECONDS

        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except TimeoutError:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": "Request timeout"},
            )


class _RequestBodyTooLargeError(Exception):
    """Raised when the streamed request body exceeds the configured limit."""


class BodySizeLimitMiddleware:
    """Reject requests whose declared or streamed body size exceeds the limit."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        from starlette.responses import JSONResponse

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_body_bytes = _max_body_bytes_from_env()
        content_length = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                content_length = value.decode("latin-1")
                break

        if content_length is not None:
            try:
                body_bytes = int(content_length)
            except ValueError:
                body_bytes = 0
            if body_bytes > max_body_bytes:
                await JSONResponse(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    content={"detail": "Request body too large"},
                )(scope, receive, send)
                return

        consumed_bytes = 0

        async def limited_receive() -> Message:
            nonlocal consumed_bytes
            message = await receive()
            if message["type"] == "http.request":
                consumed_bytes += len(message.get("body", b""))
                if consumed_bytes > max_body_bytes:
                    raise _RequestBodyTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLargeError:
            await JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content={"detail": "Request body too large"},
            )(scope, receive, send)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach conservative browser security headers to every response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if (
            request.url.scheme == "https"
            or request.headers.get("x-forwarded-proto", "").lower() == "https"
        ):
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique correlation ID to each request.

    The ID is stored in a ContextVar so it propagates to log records
    (via CorrelationIdFilter) and to ThreadPoolExecutor workers when
    ``contextvars.copy_context()`` is used.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        cid = generate_correlation_id()
        correlation_id_var.set(cid)
        try:
            stats = current_default_thread_limiter().statistics()
            request.state.tp_borrowed = stats.borrowed_tokens
            request.state.tp_total = stats.total_tokens
            request.state.tp_waiting = stats.tasks_waiting
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to snapshot threadpool limiter stats: %s", exc)
            request.state.tp_borrowed = None
            request.state.tp_total = None
            request.state.tp_waiting = None
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response
