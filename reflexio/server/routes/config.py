"""Config route handlers (extracted from api.py, Tier3 A2)."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    status,
)

from reflexio.models.api_schema.retriever_schema import (
    SetConfigResponse,
)
from reflexio.models.api_schema.service_schemas import (
    MyConfigResponse,
)
from reflexio.models.config_schema import (
    Config,
)
from reflexio.server.api_endpoints import (
    account_api,
)
from reflexio.server.auth import (
    default_get_org_id,
)
from reflexio.server.cache import reflexio_cache
from reflexio.server.rate_limit import limiter
from reflexio.server.services.configurator.config_storage import (
    ConfigWriteConflictError,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/api/my_config",
    response_model=MyConfigResponse,
    response_model_exclude_none=True,
)
def my_config_endpoint(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> MyConfigResponse:
    """Return raw storage credentials for the caller's org.

    Enablement is controlled by two independent opt-ins so the endpoint
    is closed by default on unauthenticated self-host deployments:

    - ``request.app.state.my_config_enabled`` — set to True by
      :func:`create_app` when the host wires in a Bearer-auth
      ``get_org_id`` dependency, so enterprise callers are always
      authenticated before they reach this route.
    - ``REFLEXIO_ALLOW_MY_CONFIG=true`` — OS self-host escape hatch.

    If neither is set we return a closed response instead of a 404 so
    the CLI can display an actionable hint.
    """
    app_state_enabled = bool(getattr(request.app.state, "my_config_enabled", False))
    if not (app_state_enabled or account_api.my_config_allowed()):
        return MyConfigResponse(
            success=False,
            message=(
                "GET /api/my_config is disabled. Set "
                "REFLEXIO_ALLOW_MY_CONFIG=true to enable."
            ),
        )
    return account_api.my_config(org_id=org_id)


@router.post("/api/set_config")
@limiter.limit("10/minute")
def set_config(
    request: Request,
    config: dict[str, Any],
    org_id: str = Depends(default_get_org_id),
) -> SetConfigResponse:
    """Set configuration for the organization.

    Args:
        config (dict[str, Any]): The configuration payload to set
        org_id (str): Organization ID

    Returns:
        dict: Response containing success status and message
    """
    from pydantic import ValidationError

    # Create Reflexio instance to access the configurator through request_context
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    configurator = reflexio.request_context.configurator
    try:
        normalized_config = configurator.normalize_config_payload(config)
        Config.model_validate(normalized_config)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.errors(),
        ) from exc

    # Set the config using Reflexio's set_config method
    try:
        response = reflexio.set_config(normalized_config)
    except ConfigWriteConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "config_write_conflict",
                "message": str(exc),
            },
        ) from exc

    # Invalidate cache on successful config change to ensure fresh instance next request
    if response.success:
        reflexio_cache.invalidate_reflexio_cache(org_id=org_id)

    return response


@router.post("/api/update_config")
@limiter.limit("10/minute")
def update_config(
    request: Request,
    partial: dict[str, Any],
    org_id: str = Depends(default_get_org_id),
) -> SetConfigResponse:
    """Apply a partial update to the org's config (PATCH semantics).

    Performs a **top-level shallow merge** of *partial* over the existing
    config and delegates normalization to the active configurator before the
    shared ``Config`` validation. This lets deployment-specific configurators
    consume their overlay fields while the default configurator still rejects
    unknown fields.

    .. warning::
       Nested objects (e.g. ``storage_config``, ``profile_extractor_config``,
       ``user_playbook_extractor_config``) are **replaced wholesale**.
       Deep merging is intentionally not supported -- the discriminator on
       ``storage_config`` would be lost on partial updates, and merging nested
       dicts has ambiguous semantics.

       To update a field inside an extractor config you must resend that
       extractor object fully populated (including ``extractor_name``,
       ``extraction_definition_prompt``, etc.). For one-off mutations prefer
       ``GET /api/get_config`` followed by ``POST /api/set_config`` with the
       modified full config.

    Unlike :func:`set_config`, callers do not need to re-send the full
    config (including required fields like ``storage_config``) just to
    flip a single top-level boolean. The merge happens server-side
    atomically within the request, eliminating the read-modify-write
    race a client would otherwise hit.

    Args:
        partial: Top-level fields to overlay on the existing config.
        org_id: Organization ID resolved by the auth layer.

    Returns:
        SetConfigResponse: Success status and message from
        :meth:`Reflexio.set_config`.
    """
    from pydantic import ValidationError

    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    configurator = reflexio.request_context.configurator
    existing_config = configurator.get_config()
    existing = existing_config.model_dump(mode="python")
    # Convert ValidationError into 422 so callers passing a partial that
    # would replace a nested extractor object with an incomplete dict (e.g.
    # {"user_playbook_extractor_config": {"aggregation_config": {...}}})
    # get a clean client-error response instead of a 500.
    try:
        merged_config = configurator.prepare_config_patch(partial)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "error": "Invalid partial config (top-level shallow merge)",
                "hint": (
                    "Nested objects (e.g. user_playbook_extractor_config) "
                    "are replaced wholesale, not deep-merged. To mutate a "
                    "single nested field, fetch the full config via "
                    "/api/get_config, edit, and POST it back via "
                    "/api/set_config."
                ),
                "validation_errors": exc.errors(),
            },
        ) from exc
    partial_uses_only_shared_fields = (
        partial.keys() <= type(existing_config).model_fields.keys()
    )
    if (
        partial_uses_only_shared_fields
        and merged_config.model_dump(mode="python") == existing
    ):
        logger.info("Skipping no-op config update for org %s", org_id)
        return SetConfigResponse(success=True, msg="Configuration unchanged")

    try:
        response = reflexio.set_config(merged_config)
    except ConfigWriteConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "config_write_conflict",
                "message": str(exc),
            },
        ) from exc
    if response.success:
        reflexio_cache.invalidate_reflexio_cache(org_id=org_id)
    return response


@router.get("/api/get_config")
def get_config(
    org_id: str = Depends(default_get_org_id),
) -> dict[str, Any]:
    """Get configuration for the organization.

    Args:
        org_id (str): Organization ID

    Returns:
        Config: The current configuration
    """
    # Create Reflexio instance to access the configurator through request_context
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)

    return reflexio.request_context.configurator.get_config_for_response()
