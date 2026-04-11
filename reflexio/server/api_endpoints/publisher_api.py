"""
Create, edit, delete user interaction and user profile
"""

import logging

from reflexio.models.api_schema.retriever_schema import (
    UpdateAgentPlaybookRequest,
    UpdateAgentPlaybookResponse,
    UpdatePlaybookStatusRequest,
    UpdatePlaybookStatusResponse,
    UpdateUserPlaybookRequest,
    UpdateUserPlaybookResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AddAgentPlaybookRequest,
    AddAgentPlaybookResponse,
    AddUserPlaybookRequest,
    AddUserPlaybookResponse,
    AddUserProfileRequest,
    AddUserProfileResponse,
    BulkDeleteResponse,
    DeleteAgentPlaybookRequest,
    DeleteAgentPlaybookResponse,
    DeleteAgentPlaybooksByIdsRequest,
    DeleteProfilesByIdsRequest,
    DeleteRequestRequest,
    DeleteRequestResponse,
    DeleteRequestsByIdsRequest,
    DeleteSessionRequest,
    DeleteSessionResponse,
    DeleteUserInteractionRequest,
    DeleteUserInteractionResponse,
    DeleteUserPlaybookRequest,
    DeleteUserPlaybookResponse,
    DeleteUserPlaybooksByIdsRequest,
    DeleteUserProfileRequest,
    DeleteUserProfileResponse,
    PublishUserInteractionRequest,
    PublishUserInteractionResponse,
    RunPlaybookAggregationRequest,
    RunPlaybookAggregationResponse,
)
from reflexio.server.api_endpoints.precondition_checks import (
    validate_delete_user_profile_request,
    validate_publish_user_interaction_request,
)
from reflexio.server.cache.reflexio_cache import get_reflexio

logger = logging.getLogger(__name__)

# ==============================
# Create user interaction and profile
# ==============================


def add_user_interaction(
    org_id: str,
    request: PublishUserInteractionRequest,
) -> PublishUserInteractionResponse:
    """Add user interaction

    Args:
        org_id (str): Organization ID
        request (PublishUserInteractionRequest): The request containing interaction data

    Returns:
        PublishUserInteractionResponse: Response containing success status and message
    """
    is_valid, message = validate_publish_user_interaction_request(request)
    if not is_valid:
        return PublishUserInteractionResponse(success=False, message=message)

    reflexio = get_reflexio(org_id=org_id)
    return reflexio.publish_interaction(request=request)


def add_user_playbook(
    org_id: str,
    request: AddUserPlaybookRequest,
) -> AddUserPlaybookResponse:
    """Add user playbook directly to storage.

    Args:
        org_id (str): Organization ID
        request (AddUserPlaybookRequest): The request containing user playbooks

    Returns:
        AddUserPlaybookResponse: Response containing success status, message, and added count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.add_user_playbook(request=request)


def add_agent_playbook(
    org_id: str,
    request: AddAgentPlaybookRequest,
) -> AddAgentPlaybookResponse:
    """Add agent playbook directly to storage.

    Args:
        org_id (str): Organization ID
        request (AddAgentPlaybookRequest): The request containing agent playbooks

    Returns:
        AddAgentPlaybookResponse: Response containing success status, message, and added count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.add_agent_playbook(request=request)


def add_user_profile(
    org_id: str,
    request: AddUserProfileRequest,
) -> AddUserProfileResponse:
    """Add user profile directly to storage, bypassing inference.

    Mirror of :func:`add_user_playbook` for the profile resource —
    useful for seeding a known fact about the user (testing, migration,
    manual fact injection) without producing an interaction first.

    Args:
        org_id (str): Organization ID
        request (AddUserProfileRequest): The request containing user profiles

    Returns:
        AddUserProfileResponse: Response containing success status, message, and added count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.add_user_profile(request=request)


def delete_user_profile(
    org_id: str, request: DeleteUserProfileRequest
) -> DeleteUserProfileResponse:
    """Delete user profile

    Args:
        org_id (str): Organization ID
        request (DeleteUserProfileRequest): The delete request

    Returns:
        DeleteUserProfileResponse: Response containing success status and message
    """
    is_valid, message = validate_delete_user_profile_request(request)
    if not is_valid:
        return DeleteUserProfileResponse(success=False, message=message)

    reflexio = get_reflexio(org_id=org_id)
    try:
        return reflexio.delete_profile(request)
    except Exception as e:
        logger.error("Failed to delete user profile: %s", e)
        return DeleteUserProfileResponse(success=False, message=str(e))


def delete_user_interaction(
    org_id: str, request: DeleteUserInteractionRequest
) -> DeleteUserInteractionResponse:
    """Delete user interaction

    Args:
        org_id (str): Organization ID
        request (DeleteUserInteractionRequest): The delete request

    Returns:
        DeleteUserInteractionResponse: Response containing success status and message
    """
    reflexio = get_reflexio(org_id=org_id)
    try:
        return reflexio.delete_interaction(request)
    except Exception as e:
        logger.error("Failed to delete user interaction: %s", e)
        return DeleteUserInteractionResponse(success=False, message=str(e))


def delete_request(org_id: str, request: DeleteRequestRequest) -> DeleteRequestResponse:
    """Delete request and all its associated interactions

    Args:
        org_id (str): Organization ID
        request (DeleteRequestRequest): The delete request

    Returns:
        DeleteRequestResponse: Response containing success status and message
    """
    reflexio = get_reflexio(org_id=org_id)
    try:
        return reflexio.delete_request(request)
    except Exception as e:
        logger.error("Failed to delete request: %s", e)
        return DeleteRequestResponse(success=False, message=str(e))


def delete_session(org_id: str, request: DeleteSessionRequest) -> DeleteSessionResponse:
    """Delete all requests and interactions in a session

    Args:
        org_id (str): Organization ID
        request (DeleteSessionRequest): The delete request

    Returns:
        DeleteSessionResponse: Response containing success status, message, and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    try:
        return reflexio.delete_session(request)
    except Exception as e:
        logger.error("Failed to delete session: %s", e)
        return DeleteSessionResponse(success=False, message=str(e))


def delete_agent_playbook(
    org_id: str, request: DeleteAgentPlaybookRequest
) -> DeleteAgentPlaybookResponse:
    """Delete agent playbook by ID.

    Args:
        org_id (str): Organization ID
        request (DeleteAgentPlaybookRequest): The delete request

    Returns:
        DeleteAgentPlaybookResponse: Response containing success status and message
    """
    reflexio = get_reflexio(org_id=org_id)
    try:
        return reflexio.delete_agent_playbook(request)
    except Exception as e:
        logger.error("Failed to delete agent playbook: %s", e)
        return DeleteAgentPlaybookResponse(success=False, message=str(e))


def delete_user_playbook(
    org_id: str, request: DeleteUserPlaybookRequest
) -> DeleteUserPlaybookResponse:
    """Delete user playbook by ID.

    Args:
        org_id (str): Organization ID
        request (DeleteUserPlaybookRequest): The delete request

    Returns:
        DeleteUserPlaybookResponse: Response containing success status and message
    """
    reflexio = get_reflexio(org_id=org_id)
    try:
        return reflexio.delete_user_playbook(request)
    except Exception as e:
        logger.error("Failed to delete user playbook: %s", e)
        return DeleteUserPlaybookResponse(success=False, message=str(e))


# ==============================
# Bulk delete operations
# ==============================


def delete_all_interactions_bulk(org_id: str) -> BulkDeleteResponse:
    """Delete all requests and their associated interactions.

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.delete_all_interactions_bulk()


def delete_all_profiles_bulk(org_id: str) -> BulkDeleteResponse:
    """Delete all profiles.

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.delete_all_profiles_bulk()


def delete_all_playbooks_bulk(org_id: str) -> BulkDeleteResponse:
    """Delete all playbooks (both user and agent).

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.delete_all_playbooks_bulk()


def delete_all_user_playbooks_bulk(org_id: str) -> BulkDeleteResponse:
    """Delete all user playbooks (user only, not agent).

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.delete_all_user_playbooks_bulk()


def delete_all_agent_playbooks_bulk(org_id: str) -> BulkDeleteResponse:
    """Delete all agent playbooks (agent only, not user).

    Args:
        org_id (str): Organization ID

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.delete_all_agent_playbooks_bulk()


def delete_requests_by_ids(
    org_id: str, request: DeleteRequestsByIdsRequest
) -> BulkDeleteResponse:
    """Delete requests by their IDs.

    Args:
        org_id (str): Organization ID
        request (DeleteRequestsByIdsRequest): The delete request

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.delete_requests_by_ids(request)


def delete_profiles_by_ids(
    org_id: str, request: DeleteProfilesByIdsRequest
) -> BulkDeleteResponse:
    """Delete profiles by their IDs.

    Args:
        org_id (str): Organization ID
        request (DeleteProfilesByIdsRequest): The delete request

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.delete_profiles_by_ids(request)


def delete_agent_playbooks_by_ids_bulk(
    org_id: str, request: DeleteAgentPlaybooksByIdsRequest
) -> BulkDeleteResponse:
    """Delete agent playbooks by their IDs.

    Args:
        org_id (str): Organization ID
        request (DeleteAgentPlaybooksByIdsRequest): The delete request

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.delete_agent_playbooks_by_ids_bulk(request)


def delete_user_playbooks_by_ids_bulk(
    org_id: str, request: DeleteUserPlaybooksByIdsRequest
) -> BulkDeleteResponse:
    """Delete user playbooks by their IDs.

    Args:
        org_id (str): Organization ID
        request (DeleteUserPlaybooksByIdsRequest): The delete request

    Returns:
        BulkDeleteResponse: Response containing success status and deleted count
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.delete_user_playbooks_by_ids_bulk(request)


# ==============================
# Run playbook aggregation
# ==============================


def run_playbook_aggregation(
    org_id: str, request: RunPlaybookAggregationRequest
) -> RunPlaybookAggregationResponse:
    """Run playbook aggregation for a given agent version and playbook name.

    Args:
        org_id (str): Organization ID
        request (RunPlaybookAggregationRequest): The run playbook aggregation request

    Returns:
        RunPlaybookAggregationResponse: Response containing success status and message
    """
    reflexio = get_reflexio(org_id=org_id)
    try:
        result = reflexio.run_playbook_aggregation(
            request.agent_version, request.playbook_name
        )
    except Exception as e:
        logger.error("Failed to run playbook aggregation: %s", e)
        return RunPlaybookAggregationResponse(success=False, message=str(e))

    if result.get("skipped"):
        return RunPlaybookAggregationResponse(
            success=True, message=f"Skipped: {result['skipped']}"
        )

    message = (
        f"Processed {result['user_playbooks_processed']} user playbooks, "
        f"found {result['clusters_found']} clusters, "
        f"generated {result['playbooks_generated']} agent playbooks"
    )
    return RunPlaybookAggregationResponse(success=True, message=message)


# ==============================
# Update playbook status
# ==============================


def update_agent_playbook_status(
    org_id: str, request: UpdatePlaybookStatusRequest
) -> UpdatePlaybookStatusResponse:
    """Update the status of a specific playbook.

    Args:
        org_id (str): Organization ID
        request (UpdatePlaybookStatusRequest): The update request

    Returns:
        UpdatePlaybookStatusResponse: Response containing success status and message
    """
    reflexio = get_reflexio(org_id=org_id)
    try:
        return reflexio.update_agent_playbook_status(request)
    except Exception as e:
        logger.error("Failed to update playbook status: %s", e)
        return UpdatePlaybookStatusResponse(success=False, msg=str(e))


def update_agent_playbook(
    org_id: str, request: UpdateAgentPlaybookRequest
) -> UpdateAgentPlaybookResponse:
    """Update editable fields of an agent playbook.

    Args:
        org_id (str): Organization ID
        request (UpdateAgentPlaybookRequest): The update request

    Returns:
        UpdateAgentPlaybookResponse: Response containing success status and message
    """
    reflexio = get_reflexio(org_id=org_id)
    try:
        return reflexio.update_agent_playbook(request)
    except Exception as e:
        logger.error("Failed to update agent playbook: %s", e)
        return UpdateAgentPlaybookResponse(success=False, msg=str(e))


def update_user_playbook(
    org_id: str, request: UpdateUserPlaybookRequest
) -> UpdateUserPlaybookResponse:
    """Update editable fields of a user playbook.

    Args:
        org_id (str): Organization ID
        request (UpdateUserPlaybookRequest): The update request

    Returns:
        UpdateUserPlaybookResponse: Response containing success status and message
    """
    reflexio = get_reflexio(org_id=org_id)
    try:
        return reflexio.update_user_playbook(request)
    except Exception as e:
        logger.error("Failed to update user playbook: %s", e)
        return UpdateUserPlaybookResponse(success=False, msg=str(e))
