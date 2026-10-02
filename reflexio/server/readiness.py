"""Deployment-neutral, cached readiness interface for application hosts."""

from collections.abc import Callable

from fastapi import Request

from reflexio.models.api_schema.readiness import ReadinessCheck as ReadinessCheck
from reflexio.models.api_schema.readiness import ReadinessResponse as ReadinessResponse

ReadinessProvider = Callable[[], ReadinessResponse]


def default_readiness(request: Request) -> ReadinessResponse:
    """Read the host's snapshot; standalone OSS retains its existing health."""
    provider: ReadinessProvider | None = getattr(
        request.app.state, "readiness_provider", None
    )
    return provider() if provider is not None else ReadinessResponse()
