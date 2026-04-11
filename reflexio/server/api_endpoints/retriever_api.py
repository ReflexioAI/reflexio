"""
Search user profiles and interactions
"""

from reflexio.models.api_schema.retriever_schema import (
    GetInteractionsRequest,
    GetInteractionsResponse,
    GetRequestsRequest,
    GetRequestsResponse,
    GetUserProfilesRequest,
    GetUserProfilesResponse,
    SearchAgentPlaybookRequest,
    SearchAgentPlaybookResponse,
    SearchInteractionRequest,
    SearchInteractionResponse,
    SearchUserPlaybookRequest,
    SearchUserPlaybookResponse,
    SearchUserProfileRequest,
    SearchUserProfileResponse,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from reflexio.models.api_schema.service_schemas import (
    PlaybookAggregationChangeLogResponse,
    ProfileChangeLogResponse,
)
from reflexio.server.cache.reflexio_cache import get_reflexio

# ==============================
# Search profiles and interactions
# ==============================


def search_user_profiles(
    org_id: str,
    request: SearchUserProfileRequest,
) -> SearchUserProfileResponse:
    """Search user profiles and returns response by
    - user_id
    - generated_from_request_id
    - query
    - start_time
    - end_time
    - top_k

    Args:
        org_id (str): Organization ID
        request (SearchUserProfileRequest): The search request

    Returns:
        SearchUserProfileResponse: Response containing matching user profiles
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.search_profiles(request)


def search_interactions(
    org_id: str,
    request: SearchInteractionRequest,
) -> SearchInteractionResponse:
    """Search interactions and returns response by
    - user_id
    - request_id
    - query
    - start_time
    - end_time
    - top_k

    Args:
        org_id (str): Organization ID
        request (SearchInteractionRequest): The search request

    Returns:
        SearchInteractionResponse: Response containing matching interactions
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.search_interactions(request)


# ==============================
# Get user profiles and interactions
# ==============================


def get_user_profiles(
    org_id: str,
    request: GetUserProfilesRequest,
) -> GetUserProfilesResponse:
    """Get user profiles and returns response by
    - user_id
    - start_time
    - end_time
    - top_k
    - status_filter

    Args:
        org_id (str): Organization ID
        request (GetUserProfilesRequest): The get request

    Returns:
        GetUserProfilesResponse: Response containing user profiles
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.get_profiles(request, status_filter=request.status_filter)


def get_user_interactions(
    org_id: str,
    request: GetInteractionsRequest,
) -> GetInteractionsResponse:
    """Get user interactions and returns response by
    - user_id
    - start_time
    - end_time
    - top_k

    Args:
        org_id (str): Organization ID
        request (GetInteractionsRequest): The get request

    Returns:
        GetInteractionsResponse: Response containing user interactions
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.get_interactions(request)


def get_profile_change_logs(
    org_id: str,
) -> ProfileChangeLogResponse:
    """Get profile change logs for an organization.

    Args:
        org_id (str): Organization ID to get change logs for

    Returns:
        ProfileChangeLogResponse: Response containing list of profile change logs
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.get_profile_change_logs()


def get_playbook_aggregation_change_logs(
    org_id: str,
    playbook_name: str,
    agent_version: str,
) -> PlaybookAggregationChangeLogResponse:
    """Get playbook aggregation change logs.

    Args:
        org_id (str): Organization ID
        playbook_name (str): Playbook name to filter by
        agent_version (str): Agent version to filter by

    Returns:
        PlaybookAggregationChangeLogResponse: Response containing list of change logs
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.get_playbook_aggregation_change_logs(
        playbook_name=playbook_name, agent_version=agent_version
    )


def get_requests(
    org_id: str,
    request: GetRequestsRequest,
) -> GetRequestsResponse:
    """Get requests with their associated interactions, grouped by session.

    Args:
        org_id (str): Organization ID
        request (GetRequestsRequest): The get request

    Returns:
        GetRequestsResponse: Response containing requests grouped by session with their interactions
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.get_requests(request)


# ==============================
# Search user playbooks and agent playbooks
# ==============================


def search_user_playbooks(
    org_id: str,
    request: SearchUserPlaybookRequest,
) -> SearchUserPlaybookResponse:
    """Search user playbooks with advanced filtering.

    Supports filtering by:
    - query (semantic/text search)
    - user_id (via request_id linkage to requests table)
    - agent_version
    - playbook_name
    - start_time, end_time (datetime range on created_at)
    - status_filter
    - top_k, threshold

    Args:
        org_id (str): Organization ID
        request (SearchUserPlaybookRequest): The search request

    Returns:
        SearchUserPlaybookResponse: Response containing matching user playbooks
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.search_user_playbooks(request)


def search_agent_playbooks(
    org_id: str,
    request: SearchAgentPlaybookRequest,
) -> SearchAgentPlaybookResponse:
    """Search agent playbooks with advanced filtering.

    Supports filtering by:
    - query (semantic/text search)
    - agent_version
    - playbook_name
    - start_time, end_time (datetime range on created_at)
    - status_filter
    - playbook_status_filter
    - top_k, threshold

    Args:
        org_id (str): Organization ID
        request (SearchAgentPlaybookRequest): The search request

    Returns:
        SearchAgentPlaybookResponse: Response containing matching agent playbooks
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.search_agent_playbooks(request)


# ==============================
# Unified search
# ==============================


def unified_search(
    org_id: str,
    request: UnifiedSearchRequest,
) -> UnifiedSearchResponse:
    """Search across all entity types (profiles, agent playbooks, user playbooks) in parallel.

    Query reformulation is controlled per-request via the enable_reformulation param.

    Args:
        org_id (str): Organization ID
        request (UnifiedSearchRequest): The unified search request

    Returns:
        UnifiedSearchResponse: Combined search results from all entity types
    """
    reflexio = get_reflexio(org_id=org_id)
    return reflexio.unified_search(request, org_id=org_id)
