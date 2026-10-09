"""Access policy for the standalone OSS app, independent of enterprise auth."""

import hmac
import ipaddress
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from reflexio.server.env_utils import env_str


def _loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


class LocalAccessMiddleware:
    """Keep no-key access local; a configured key protects local access too.

    Only the default standalone app installs this policy. Hosts composing
    ``create_app`` themselves remain responsible for supplying authentication.
    Forwarded headers are not used to grant access.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.api_key = env_str("REFLEXIO_API_KEY", "")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["method"] == "OPTIONS" or (
            scope["method"] in {"GET", "HEAD"}
            and scope["path"]
            in {
                "/",
                "/health",
                "/healthz",
                "/healthz/eval",
                "/meta/version",
                "/docs",
                "/redoc",
                "/openapi.json",
            }
        ):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if self.api_key:
            scheme, _, credential = headers.get("authorization", "").partition(" ")
            permitted = scheme.lower() == "bearer" and hmac.compare_digest(
                credential.encode(), self.api_key.encode()
            )
        else:
            client = scope.get("client")
            try:
                host = urlsplit("//" + headers.get("host", "")).hostname
                origin = headers.get("origin")
                parsed_origin = urlsplit(origin) if origin else None
                permitted = bool(
                    client
                    and _loopback(client[0])
                    and _loopback(host)
                    and not any(
                        name in headers
                        for name in ("forwarded", "x-forwarded-for", "x-real-ip")
                    )
                    and (
                        parsed_origin is None
                        or (
                            parsed_origin.scheme in {"http", "https"}
                            and _loopback(parsed_origin.hostname)
                        )
                    )
                )
            except ValueError:
                permitted = False
        if not permitted:
            await JSONResponse(
                {"detail": "Authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
            return
        await self.app(scope, receive, send)
