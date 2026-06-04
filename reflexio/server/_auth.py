"""Auth dependency primitives shared between :mod:`reflexio.server.api` and
endpoint helpers.

Lives in its own module to avoid an import cycle: api endpoints reference
:func:`default_get_org_id` (e.g. via FastAPI ``Depends``) and are themselves
imported by :mod:`reflexio.server.api`. Putting the dependency here keeps the
import graph acyclic.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

DEFAULT_ORG_ID = "self-host-org"


def default_get_org_id() -> str:
    """Return the default organization ID for local hosting.

    Enterprise deployments override this via
    ``app.dependency_overrides[default_get_org_id] = <bearer_auth_resolver>``
    inside :func:`reflexio.server.api.create_app`.

    Returns:
        str: The default org identifier used for self-hosted deployments.
    """
    return DEFAULT_ORG_ID


def default_get_caller_type() -> str:
    """Return the default caller type for local / self-hosted deployments.

    Every call is treated as ``"internal"`` (no billing discrimination) in the
    OSS build.  Enterprise deployments override this via
    ``app.dependency_overrides[default_get_caller_type] = <classifier>``
    inside :func:`reflexio.server.api.create_app`, exactly like
    :func:`default_get_org_id`.

    Returns:
        str: The literal ``"internal"`` — equals ``BillingCallerType.INTERNAL.value``
            (kept as a plain string here so OSS stays free of reflexio_ext imports).
    """
    return "internal"  # == BillingCallerType.INTERNAL.value; literal keeps OSS clean


@cache
def default_billing_gate(line: str) -> Callable[..., None]:  # noqa: ARG001
    """Return a stable no-op FastAPI dependency for the billing gate.

    OSS / self-hosted deployments run without enforcement — this factory
    always returns a dependency that does nothing.  Enterprise deployments
    supply a real gate factory via the ``get_billing_gate`` parameter of
    :func:`reflexio.server.api.create_app`, exactly mirroring how
    ``get_caller_type`` overrides :func:`default_get_caller_type`.

    ``lru_cache`` ensures the **same callable object** is returned for the
    same ``line`` value.  FastAPI uses callable identity for
    ``dependency_overrides``, so caching is essential: without it, each
    ``Depends(default_billing_gate("application"))`` would create a fresh
    closure and overrides would silently never fire.

    OSS code must never import from ``reflexio_ext`` — the gate wiring is
    purely additive and backwards-compatible.

    Args:
        line (str): Billing line (e.g. ``"application"``, ``"learnings_generated"``).
            Accepted but unused in the default no-op implementation.

    Returns:
        Callable[..., None]: A FastAPI dependency that always passes through.
    """

    def _noop() -> None:
        """No-op billing gate for OSS / self-hosted deployments."""

    return _noop
