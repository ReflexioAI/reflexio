"""Braintrust connector route handlers (extracted from api.py, Tier3 A2)."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    Depends,
    Request,
)

from reflexio.models.api_schema.braintrust_schema import (
    BraintrustStatusResponse,
    ConnectBraintrustRequest,
    ConnectBraintrustResponse,
    SelectProjectsRequest,
    SelectProjectsResponse,
    SyncBraintrustResponse,
)
from reflexio.server.auth import (
    default_get_org_id,
)
from reflexio.server.cache import reflexio_cache
from reflexio.server.rate_limit import limiter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/api/braintrust/connect",
    response_model=ConnectBraintrustResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def braintrust_connect(
    request: Request,
    payload: ConnectBraintrustRequest,
    org_id: str = Depends(default_get_org_id),
) -> ConnectBraintrustResponse:
    """Step 1: validate the Braintrust API key and list workspaces/projects.

    Persists nothing — call `/api/braintrust/select_projects` to commit.

    Args:
        request (Request): The HTTP request object for rate limiting.
        payload (ConnectBraintrustRequest): Customer's Braintrust API key.
        org_id (str): Resolved by auth dependency.

    Returns:
        ConnectBraintrustResponse: Workspaces tree on success; `success=False`
            with a message when the key is rejected.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    return reflexio.braintrust_connect(payload)


@router.post(
    "/api/braintrust/select_projects",
    response_model=SelectProjectsResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def braintrust_select_projects(
    request: Request,
    payload: SelectProjectsRequest,
    org_id: str = Depends(default_get_org_id),
) -> SelectProjectsResponse:
    """Step 2: commit the Braintrust connection with selected projects.

    The API key is encrypted at rest. Subsequent syncs use the persisted
    connection until the customer calls DELETE /api/braintrust/connection.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    return reflexio.braintrust_select_projects(payload)


@router.get(
    "/api/braintrust/status",
    response_model=BraintrustStatusResponse,
    response_model_exclude_none=True,
)
def braintrust_status(
    org_id: str = Depends(default_get_org_id),
) -> BraintrustStatusResponse:
    """Return Braintrust connection state. Never echoes the API key."""
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    return reflexio.braintrust_status()


@router.delete("/api/braintrust/connection")
@limiter.limit("10/minute")
def braintrust_disconnect(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> dict:
    """Delete the persisted Braintrust connection for the org.

    Args:
        org_id (str): Resolved by auth dependency.

    Returns:
        dict: ``{"success": True}`` on completion.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    reflexio.braintrust_disconnect()
    return {"success": True}


@router.post(
    "/api/braintrust/sync",
    response_model=SyncBraintrustResponse,
    response_model_exclude_none=True,
)
@limiter.limit("10/minute")
def braintrust_sync(
    request: Request,
    org_id: str = Depends(default_get_org_id),
) -> SyncBraintrustResponse:
    """Trigger a one-shot sync of Braintrust scorer outputs.

    Scheduled (cron) sync is a follow-up; for now the endpoint exists so
    operators can drive a manual import.
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    return reflexio.braintrust_sync()
