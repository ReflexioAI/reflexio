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
