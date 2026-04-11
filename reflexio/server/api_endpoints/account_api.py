"""Account/diagnostics endpoints.

Exposes two read-only endpoints that power CLI observability:

- ``whoami``: masked summary of the caller's resolved storage routing.
  Safe to call without special permission — never returns raw secrets.
- ``my_config``: raw storage credentials for the caller's org.
  Token-gated and intended for operators pulling their own config down
  to a fresh machine via ``reflexio config pull``.

Both call through the cached ``Reflexio`` instance so per-org config
lookups honour the same configurator that data reads/writes use.
"""

from __future__ import annotations

import logging
import os

from reflexio.lib._storage_labels import describe_storage
from reflexio.models.api_schema.service_schemas import (
    MyConfigResponse,
    WhoamiResponse,
)
from reflexio.server.cache.reflexio_cache import get_reflexio

logger = logging.getLogger(__name__)

# Generic, user-facing message used whenever loading a caller's storage
# configuration fails. Exact exception text (connection strings, file
# paths, SQL errors) must never leak to the HTTP response body.
_GENERIC_STORAGE_LOAD_FAILURE = "Failed to load storage configuration"


def whoami(org_id: str) -> WhoamiResponse:
    """Return the caller's masked storage routing summary.

    Args:
        org_id: Organization identifier, resolved by the FastAPI
            auth dependency from the Bearer token (or ``DEFAULT_ORG_ID``
            in self-host mode).

    Returns:
        WhoamiResponse: Always includes ``org_id``; storage fields are
            ``None`` / ``storage_configured=False`` when the org has no
            storage configured yet.
    """
    try:
        reflexio = get_reflexio(org_id=org_id)
    except Exception:
        logger.exception("whoami: failed to load reflexio for %s", org_id)
        return WhoamiResponse(
            success=False,
            org_id=org_id,
            storage_configured=False,
            message=_GENERIC_STORAGE_LOAD_FAILURE,
        )

    configurator = reflexio.request_context.configurator
    storage_config = configurator.get_current_storage_configuration()
    storage_type, storage_label = describe_storage(storage_config)
    return WhoamiResponse(
        success=True,
        org_id=org_id,
        storage_type=storage_type,
        storage_label=storage_label,
        storage_configured=storage_config is not None
        and configurator.is_storage_configured(),
    )


# Guard the my_config endpoint on OS/self-host so it isn't exposed by
# default on unauthenticated localhost deployments. The FastAPI layer
# (``reflexio.server.api.create_app``) now owns the enterprise opt-in:
# when a custom ``get_org_id`` is provided together with ``require_auth``,
# it sets ``app.state.my_config_enabled = True``. This helper keeps the
# OS-only ``REFLEXIO_ALLOW_MY_CONFIG`` env var as a fallback for self-hosts.
_ALLOW_MY_CONFIG_ENV_VAR = "REFLEXIO_ALLOW_MY_CONFIG"


def my_config_allowed() -> bool:
    """Return whether ``GET /api/my_config`` is enabled via the env var.

    Returns True when ``REFLEXIO_ALLOW_MY_CONFIG=true`` is set
    (OS self-host opt-in). App-state driven enablement lives in the
    FastAPI endpoint wrapper in ``reflexio.server.api`` and does not
    flow through this helper.
    """
    return os.environ.get(_ALLOW_MY_CONFIG_ENV_VAR, "").lower() in {"1", "true", "yes"}


def my_config(org_id: str) -> MyConfigResponse:
    """Return the caller's raw storage configuration.

    This is the "download my creds" endpoint. On OS/self-host it is
    guarded by ``REFLEXIO_ALLOW_MY_CONFIG``; on enterprise the FastAPI
    route wraps it in Bearer-token auth and sets ``app.state.my_config_enabled``.

    Args:
        org_id: Organization identifier.

    Returns:
        MyConfigResponse: Serialised ``StorageConfig`` as a dict, or an
            empty response with ``success=False`` when nothing is
            configured.
    """
    try:
        reflexio = get_reflexio(org_id=org_id)
    except Exception:
        logger.exception("my_config: failed to load reflexio for %s", org_id)
        return MyConfigResponse(
            success=False,
            message=_GENERIC_STORAGE_LOAD_FAILURE,
        )

    storage_config = (
        reflexio.request_context.configurator.get_current_storage_configuration()
    )
    if storage_config is None:
        return MyConfigResponse(
            success=False,
            storage_config=None,
            message="No storage configured for this org",
        )
    storage_type, _ = describe_storage(storage_config)
    # Credential-export requests are security-sensitive — log at warning
    # so they survive production log-level filtering and show up in any
    # audit review.
    logger.warning(
        "my_config credential export requested for org=%s type=%s",
        org_id,
        storage_type,
    )
    return MyConfigResponse(
        success=True,
        storage_config=storage_config.model_dump(),
        storage_type=storage_type,
    )
