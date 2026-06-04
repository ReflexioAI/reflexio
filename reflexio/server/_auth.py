"""Auth dependency primitives shared between :mod:`reflexio.server.api` and
endpoint helpers.

Lives in its own module to avoid an import cycle: api endpoints reference
:func:`default_get_org_id` (e.g. via FastAPI ``Depends``) and are themselves
imported by :mod:`reflexio.server.api`. Putting the dependency here keeps the
import graph acyclic.
"""

from __future__ import annotations

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
