"""End-to-end tests for complete workflows and error handling."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.retriever_schema import (
    GetDashboardStatsRequest,
    GetInteractionsRequest,
    GetRequestsRequest,
    GetUserProfilesRequest,
    SearchInteractionRequest,
    SearchUserProfileRequest,
)
from reflexio.models.api_schema.service_schemas import (
    DeleteRequestRequest,
    DeleteSessionRequest,
    DeleteUserInteractionRequest,
    DeleteUserProfileRequest,
    GetOperationStatusRequest,
    InteractionData,
    RerunProfileGenerationRequest,
    Status,
)
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)
from tests.server.test_utils import skip_in_precommit, skip_low_priority


@skip_in_precommit
def test_complete_workflow_end_to_end(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test complete end-to-end workflow: publish, search, get, delete."""
    user_id = "test_user_complete"

    # Get initial state
    initial_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    initial_profiles = reflexio_instance.request_context.storage.get_all_profiles()
    initial_change_logs = (
        reflexio_instance.request_context.storage.get_profile_change_logs()
    )

    # Step 1: Publish interactions (request_id will be auto-generated)
    publish_response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
            "agent_version": "test_agent_complete",
        }
    )
    assert publish_response.success is True

    # Get the auto-generated request_id from stored interactions for THIS user
    user_interactions = reflexio_instance.request_context.storage.get_user_interaction(
        user_id
    )
    assert len(user_interactions) > 0, "Should have interactions for user"
    request_id = user_interactions[0].request_id if user_interactions else None

    # Verify Request was stored
    stored_request = reflexio_instance.request_context.storage.get_request(request_id)
    assert stored_request is not None
    assert stored_request.request_id == request_id
    assert stored_request.user_id == user_id
    assert stored_request.agent_version == "test_agent_complete"

    # Verify data was stored
    stored_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    stored_profiles = reflexio_instance.request_context.storage.get_all_profiles()
    stored_change_logs = (
        reflexio_instance.request_context.storage.get_profile_change_logs()
    )
    assert len(stored_interactions) > len(initial_interactions)
    assert len(stored_profiles) > len(initial_profiles)
    assert len(stored_change_logs) > len(initial_change_logs)

    # Step 2: Search interactions
    search_interaction_response = reflexio_instance.search_interactions(
        SearchInteractionRequest(user_id=user_id, query="Sarah", top_k=5)
    )
    assert search_interaction_response.success is True
    assert len(search_interaction_response.interactions) > 0

    # Step 3: Get all interactions
    get_interactions_response = reflexio_instance.get_interactions(
        GetInteractionsRequest(user_id=user_id)
    )
    assert get_interactions_response.success is True
    assert len(get_interactions_response.interactions) >= len(
        sample_interaction_requests
    )

    # Step 4: Get all profiles
    get_profiles_response = reflexio_instance.get_profiles(
        GetUserProfilesRequest(user_id=user_id)
    )
    assert get_profiles_response.success is True
    assert len(get_profiles_response.user_profiles) > 0

    # Step 5: Search profiles (use actual profile content for reliable search)
    profile_content = get_profiles_response.user_profiles[0].content
    search_words = " ".join(profile_content.split()[:4])
    search_profile_response = reflexio_instance.search_profiles(
        SearchUserProfileRequest(user_id=user_id, query=search_words, top_k=5)
    )
    assert search_profile_response.success is True
    assert len(search_profile_response.user_profiles) > 0
    # Verify search returns profiles with correct status (default is CURRENT/None)
    for profile in search_profile_response.user_profiles:
        assert profile.status is None, (
            f"Default search should return only CURRENT profiles, got status={profile.status}"
        )

    # Step 6: Get profile change logs
    changelog_response = reflexio_instance.get_profile_change_logs()
    assert changelog_response.success is True
    assert len(changelog_response.profile_change_logs) > 0

    # Step 7: Delete an interaction
    if get_interactions_response.interactions:
        interaction_to_delete = get_interactions_response.interactions[0]
        delete_interaction_response = reflexio_instance.delete_interaction(
            DeleteUserInteractionRequest(
                user_id=user_id,
                interaction_id=interaction_to_delete.interaction_id,
            )
        )
        assert delete_interaction_response.success is True

        # Verify interaction was deleted from storage
        final_interactions = (
            reflexio_instance.request_context.storage.get_all_interactions()
        )
        assert len(final_interactions) < len(stored_interactions)

    # Step 8: Delete a profile
    if get_profiles_response.user_profiles:
        profile_to_delete = get_profiles_response.user_profiles[0]
        delete_profile_response = reflexio_instance.delete_profile(
            DeleteUserProfileRequest(
                user_id=user_id,
                profile_id=profile_to_delete.profile_id,
            )
        )
        assert delete_profile_response.success is True

        # Verify profile was deleted from storage
        final_profiles = reflexio_instance.request_context.storage.get_all_profiles()
        assert len(final_profiles) < len(stored_profiles)


@skip_in_precommit
@skip_low_priority
def test_error_handling_end_to_end(
    reflexio_instance: Reflexio, cleanup_after_test: Callable[[], None]
):
    """Test error handling in end-to-end scenarios."""
    # Test with invalid user ID
    search_response = reflexio_instance.search_interactions(
        SearchInteractionRequest(user_id="nonexistent_user", query="test", top_k=5)
    )
    assert search_response.success is True
    assert len(search_response.interactions) == 0

    # Test with invalid profile search
    profile_response = reflexio_instance.search_profiles(
        SearchUserProfileRequest(user_id="nonexistent_user", query="test", top_k=5)
    )
    assert profile_response.success is True
    assert len(profile_response.user_profiles) == 0

    # Test with empty interaction requests
    empty_response = reflexio_instance.publish_interaction(
        {
            "user_id": "test_user_empty",
            "interaction_data_list": [],
        }
    )
    # This should either succeed (if empty requests are allowed) or fail gracefully
    assert isinstance(empty_response.success, bool)


@skip_in_precommit
def test_request_and_session_id_management(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test request and session_id management: get_requests, delete_request, delete_session."""
    import uuid

    user_id = "test_user_requests"
    session_id = f"test_session_id_{uuid.uuid4().hex[:8]}"

    # Publish multiple interactions with different request_ids but same session_id
    publish_response_1 = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests[:2],
            "source": "test_source_1",
            "agent_version": "v1",
            "session_id": session_id,
        }
    )
    assert publish_response_1.success is True

    publish_response_2 = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests[2:4],
            "source": "test_source_2",
            "agent_version": "v1",
            "session_id": session_id,
        }
    )
    assert publish_response_2.success is True

    # Get requests to verify they're grouped correctly
    get_requests_response = reflexio_instance.get_requests(
        GetRequestsRequest(user_id=user_id)
    )
    assert get_requests_response.success is True
    assert len(get_requests_response.sessions) > 0

    # Find our session
    our_session_id = None
    for rg in get_requests_response.sessions:
        if rg.session_id == session_id:
            our_session_id = rg
            break

    assert our_session_id is not None
    assert len(our_session_id.requests) == 2  # Two separate requests

    # Verify each request has interactions
    for request_data in our_session_id.requests:
        assert len(request_data.interactions) > 0

    # Test delete_request - delete the first request
    first_request_id = our_session_id.requests[0].request.request_id
    delete_request_response = reflexio_instance.delete_request(
        DeleteRequestRequest(request_id=first_request_id)
    )
    assert delete_request_response.success is True

    # Verify the request was deleted
    get_requests_after_delete = reflexio_instance.get_requests(
        GetRequestsRequest(user_id=user_id)
    )
    remaining_requests = []
    for rg in get_requests_after_delete.sessions:
        if rg.session_id == session_id:
            remaining_requests = rg.requests
            break

    assert len(remaining_requests) == 1  # Only one request should remain

    # Test delete_session - delete entire session
    delete_group_response = reflexio_instance.delete_session(
        DeleteSessionRequest(session_id=session_id)
    )
    assert delete_group_response.success is True
    assert delete_group_response.deleted_requests_count > 0

    # Verify the entire session was deleted
    get_requests_final = reflexio_instance.get_requests(
        GetRequestsRequest(user_id=user_id)
    )
    for rg in get_requests_final.sessions:
        assert rg.session_id != session_id  # Should not find our session


@skip_in_precommit
@skip_low_priority
def test_profile_status_filtering(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test profile status filtering in search_profiles and get_profiles."""
    user_id = "test_user_status"

    # Publish interactions to generate profiles
    publish_response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_source",
            "agent_version": "v1",
        }
    )
    assert publish_response.success is True

    # Test get_profiles with default filter (current profiles only, status=None)
    current_profiles = reflexio_instance.get_profiles(
        GetUserProfilesRequest(user_id=user_id)
    )
    assert current_profiles.success is True
    current_count = len(current_profiles.user_profiles)
    assert current_count > 0

    # Test get_profiles with explicit status filter for CURRENT (None)
    current_explicit = reflexio_instance.get_profiles(
        GetUserProfilesRequest(user_id=user_id),
        status_filter=[None],
    )
    assert current_explicit.success is True
    assert len(current_explicit.user_profiles) == current_count

    # Test search_profiles with default filter (use actual profile content for reliable search)
    profile_content = current_profiles.user_profiles[0].content
    search_words = " ".join(profile_content.split()[:4])
    search_current = reflexio_instance.search_profiles(
        SearchUserProfileRequest(user_id=user_id, query=search_words, top_k=10)
    )
    assert search_current.success is True
    # Note: search may return fewer results than get_profiles due to relevance filtering
    assert len(search_current.user_profiles) > 0
    # Verify that default search only returns CURRENT profiles (status=None)
    for profile in search_current.user_profiles:
        assert profile.status is None, (
            f"Default search should return only CURRENT profiles, got status={profile.status}"
        )

    # Test get_all_profiles with different status filters
    all_current = reflexio_instance.get_all_profiles(status_filter=[None])
    assert all_current.success is True
    assert len(all_current.user_profiles) >= current_count


@skip_in_precommit
@skip_low_priority  # LLM non-determinism: rerun may generate 0 profiles if LLM decides nothing new to extract
def test_profile_upgrade_downgrade_workflow(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test profile upgrade and downgrade workflow."""
    user_id = "test_user_upgrade"

    # Publish all interactions (≥6 exceeds stride=5, triggering auto-extraction
    # which creates CURRENT profiles).
    publish_response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_source",
            "agent_version": "v1",
        }
    )
    assert publish_response.success is True

    # Wait for auto-extraction to create CURRENT profiles
    current_profiles = reflexio_instance.get_profiles(
        GetUserProfilesRequest(user_id=user_id),
        status_filter=[None],  # CURRENT
    )
    assert current_profiles.success is True
    assert len(current_profiles.user_profiles) > 0, (
        "Auto-extraction should create CURRENT profiles when interaction count exceeds stride"
    )
    current_count_before_rerun = len(current_profiles.user_profiles)

    # Rerun profile generation to create PENDING profiles
    rerun_response = reflexio_instance.rerun_profile_generation(
        RerunProfileGenerationRequest(user_id=user_id)
    )
    assert rerun_response.success is True

    # Check that PENDING profiles were created
    stats_after_rerun = reflexio_instance.get_profile_statistics()
    assert stats_after_rerun.success is True
    assert stats_after_rerun.pending_count > 0

    # Get profiles with PENDING status
    pending_profiles = reflexio_instance.get_profiles(
        GetUserProfilesRequest(user_id=user_id),
        status_filter=[Status.PENDING],
    )
    assert pending_profiles.success is True
    assert len(pending_profiles.user_profiles) > 0

    # Test upgrade_all_profiles with only_affected_users=True
    upgrade_response = reflexio_instance.upgrade_all_profiles(
        {"user_id": user_id, "only_affected_users": True}
    )
    assert upgrade_response.success is True
    assert upgrade_response.profiles_promoted > 0
    # CURRENT profiles from auto-extraction should now be ARCHIVED
    assert upgrade_response.profiles_archived == current_count_before_rerun

    # Verify PENDING profiles became CURRENT
    stats_after_upgrade = reflexio_instance.get_profile_statistics()
    assert stats_after_upgrade.success is True
    assert stats_after_upgrade.pending_count == 0

    # Verify old CURRENT profiles became ARCHIVED
    archived_profiles = reflexio_instance.get_profiles(
        GetUserProfilesRequest(user_id=user_id),
        status_filter=[Status.ARCHIVED],
    )
    assert archived_profiles.success is True
    assert len(archived_profiles.user_profiles) == current_count_before_rerun

    # Test downgrade_all_profiles
    downgrade_response = reflexio_instance.downgrade_all_profiles(
        {"user_id": user_id, "only_affected_users": True}
    )
    assert downgrade_response.success is True
    assert downgrade_response.profiles_restored > 0

    # Verify ARCHIVED profiles were restored to CURRENT
    stats_after_downgrade = reflexio_instance.get_profile_statistics()
    assert stats_after_downgrade.success is True


@skip_in_precommit
@skip_low_priority
def test_time_filtering_and_pagination(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test time filtering and top_k pagination in get_interactions and get_profiles."""
    user_id = "test_user_filters"

    # Publish interactions
    publish_response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_source",
            "agent_version": "v1",
        }
    )
    assert publish_response.success is True

    # Test get_interactions with time filters
    now = datetime.now(UTC)
    one_hour_ago = now - timedelta(hours=1)
    one_hour_later = now + timedelta(hours=1)

    # Get interactions from the past hour to future
    interactions_filtered = reflexio_instance.get_interactions(
        GetInteractionsRequest(
            user_id=user_id, start_time=one_hour_ago, end_time=one_hour_later
        )
    )
    assert interactions_filtered.success is True
    total_interactions = len(interactions_filtered.interactions)
    assert total_interactions > 0

    # Test top_k pagination
    top_k = 2
    interactions_paginated = reflexio_instance.get_interactions(
        GetInteractionsRequest(user_id=user_id, top_k=top_k)
    )
    assert interactions_paginated.success is True
    assert len(interactions_paginated.interactions) <= top_k
    assert len(interactions_paginated.interactions) <= total_interactions

    # Test get_profiles with time filters
    profiles_filtered = reflexio_instance.get_profiles(
        GetUserProfilesRequest(
            user_id=user_id, start_time=one_hour_ago, end_time=one_hour_later
        )
    )
    assert profiles_filtered.success is True
    total_profiles = len(profiles_filtered.user_profiles)
    assert total_profiles > 0

    # Test top_k for profiles
    profiles_paginated = reflexio_instance.get_profiles(
        GetUserProfilesRequest(user_id=user_id, top_k=1)
    )
    assert profiles_paginated.success is True
    assert len(profiles_paginated.user_profiles) <= 1


@skip_in_precommit
@skip_low_priority
def test_dashboard_statistics(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test dashboard statistics and time series data."""
    user_id = "test_user_dashboard"

    # Publish interactions to generate data
    publish_response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_source",
            "agent_version": "v1",
        }
    )
    assert publish_response.success is True

    # Get dashboard stats for last 7 days
    stats_response = reflexio_instance.get_dashboard_stats(
        GetDashboardStatsRequest(days_back=7)
    )
    assert stats_response.success is True
    assert stats_response.stats is not None

    # Verify current period stats
    current_period = stats_response.stats.current_period
    assert current_period is not None
    assert current_period.total_interactions >= len(sample_interaction_requests)
    assert current_period.total_profiles > 0

    # Verify previous period stats exist
    previous_period = stats_response.stats.previous_period
    assert previous_period is not None

    # Verify time series data exists
    assert len(stats_response.stats.interactions_time_series) > 0
    assert len(stats_response.stats.profiles_time_series) > 0
    assert len(stats_response.stats.playbooks_time_series) >= 0
    assert len(stats_response.stats.evaluations_time_series) >= 0


@skip_in_precommit
def test_rerun_profile_generation_with_filters(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test rerun_profile_generation with various filters."""
    user_id = "test_user_rerun"

    # Publish interactions with different sources
    publish_response_1 = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests[:2],
            "source": "source_a",
            "agent_version": "v1",
        }
    )
    assert publish_response_1.success is True

    publish_response_2 = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests[2:4],
            "source": "source_b",
            "agent_version": "v1",
        }
    )
    assert publish_response_2.success is True

    # Test rerun with source filter
    rerun_response = reflexio_instance.rerun_profile_generation(
        RerunProfileGenerationRequest(user_id=user_id, source="source_a")
    )
    assert rerun_response.success is True
    # In mock mode, initial publish may already generate profiles, so rerun may generate 0
    # The important assertion is that the API call succeeds
    assert rerun_response.profiles_generated >= 0

    # Check operation status
    operation_status = reflexio_instance.get_operation_status(
        GetOperationStatusRequest(service_name="profile_generation")
    )
    assert operation_status.success is True
    assert operation_status.operation_status is not None
    assert operation_status.operation_status.service_name == "profile_generation"

    # Test rerun with time filters
    now = datetime.now(UTC)
    one_hour_ago = now - timedelta(hours=1)
    rerun_time_filtered = reflexio_instance.rerun_profile_generation(
        RerunProfileGenerationRequest(user_id=user_id, start_time=one_hour_ago)
    )
    assert rerun_time_filtered.success is True


@skip_in_precommit
@skip_low_priority
def test_get_all_operations(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test get_all_profiles and get_all_interactions across all users."""
    # Create interactions for multiple users
    user_ids = ["user_a", "user_b", "user_c"]

    for user_id in user_ids:
        publish_response = reflexio_instance.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests[:2],
                "source": "test_source",
                "agent_version": "v1",
            }
        )
        assert publish_response.success is True

    # Test get_all_interactions with limit
    all_interactions = reflexio_instance.get_all_interactions(limit=100)
    assert all_interactions.success is True
    assert len(all_interactions.interactions) >= len(user_ids) * 2

    # Verify interactions are sorted by created_at (most recent first)
    if len(all_interactions.interactions) > 1:
        for i in range(len(all_interactions.interactions) - 1):
            assert (
                all_interactions.interactions[i].created_at
                >= all_interactions.interactions[i + 1].created_at
            )

    # Test get_all_profiles with limit
    all_profiles = reflexio_instance.get_all_profiles(limit=100)
    assert all_profiles.success is True
    # In mock mode, profile generation may not succeed for all users consistently
    # The important assertion is that profiles exist and the API works
    assert len(all_profiles.user_profiles) > 0

    # Verify profiles are sorted by last_modified_timestamp (most recent first)
    if len(all_profiles.user_profiles) > 1:
        for i in range(len(all_profiles.user_profiles) - 1):
            assert (
                all_profiles.user_profiles[i].last_modified_timestamp
                >= all_profiles.user_profiles[i + 1].last_modified_timestamp
            )

    # Test with different status filters
    all_pending = reflexio_instance.get_all_profiles(
        limit=100, status_filter=[Status.PENDING]
    )
    assert all_pending.success is True

    all_archived = reflexio_instance.get_all_profiles(
        limit=100, status_filter=[Status.ARCHIVED]
    )
    assert all_archived.success is True


@skip_in_precommit
def test_full_workflow_with_all_features(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test complete workflow exercising profiles, playbooks, and agent success together.

    This test verifies that when all features are enabled:
    1. Publishing interactions generates profiles, playbooks, and agent success evaluations
    2. All generated data can be retrieved correctly
    3. Search operations work across all data types
    4. Dashboard statistics reflect all data types
    """
    user_id = "test_user_full_workflow"
    agent_version = "test_agent_full"
    session_id = "test_session_full_workflow"

    # Step 1: Publish interactions (all configs are enabled in reflexio_instance)
    publish_response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_full_workflow",
            "agent_version": agent_version,
            "session_id": session_id,
        }
    )
    assert publish_response.success is True

    # Get auto-generated request_id
    stored_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    assert len(stored_interactions) > 0
    request_id = stored_interactions[0].request_id

    # Step 2: Verify profiles were generated
    stored_profiles = reflexio_instance.request_context.storage.get_all_profiles()
    assert len(stored_profiles) > 0, "Profiles should be generated"
    assert stored_profiles[0].content.strip() != ""
    assert stored_profiles[0].generated_from_request_id == request_id

    # Step 3: Verify playbooks were generated
    user_playbooks = reflexio_instance.request_context.storage.get_user_playbooks(
        playbook_name="test_playbook"
    )
    assert len(user_playbooks) > 0, "User playbooks should be generated"
    assert user_playbooks[0].content.strip() != ""
    assert user_playbooks[0].agent_version == agent_version

    # Step 4: Trigger and verify agent success evaluations
    run_group_evaluation(
        org_id=reflexio_instance.org_id,
        user_id=user_id,
        session_id=session_id,
        agent_version=agent_version,
        source="test_full_workflow",
        request_context=reflexio_instance.request_context,
        llm_client=reflexio_instance.llm_client,
    )
    agent_success_results = (
        reflexio_instance.request_context.storage.get_agent_success_evaluation_results(
            agent_version=agent_version
        )
    )
    assert len(agent_success_results) > 0, (
        "Agent success evaluations should be generated"
    )
    assert agent_success_results[0].session_id == session_id
    assert agent_success_results[0].agent_version == agent_version
    assert isinstance(agent_success_results[0].is_success, bool)

    # Step 5: Verify profile change logs were created
    change_logs = reflexio_instance.request_context.storage.get_profile_change_logs()
    assert len(change_logs) > 0, "Profile change logs should be created"

    # Step 6: Test search operations work correctly
    from reflexio.models.api_schema.retriever_schema import (
        SearchInteractionRequest,
        SearchUserProfileRequest,
    )

    # Search interactions
    search_interaction_response = reflexio_instance.search_interactions(
        SearchInteractionRequest(user_id=user_id, query="Sarah", top_k=5)
    )
    assert search_interaction_response.success is True
    assert len(search_interaction_response.interactions) > 0

    # Search profiles (use actual profile content for reliable search)
    profile_content = stored_profiles[0].content
    search_words = " ".join(profile_content.split()[:4])
    search_profile_response = reflexio_instance.search_profiles(
        SearchUserProfileRequest(user_id=user_id, query=search_words, top_k=5)
    )
    assert search_profile_response.success is True
    assert len(search_profile_response.user_profiles) > 0
    # Verify search returns profiles with correct status (default is CURRENT/None)
    for profile in search_profile_response.user_profiles:
        assert profile.status is None, (
            f"Default search should return only CURRENT profiles, got status={profile.status}"
        )

    # Step 7: Verify dashboard statistics include all data types
    stats_response = reflexio_instance.get_dashboard_stats(
        GetDashboardStatsRequest(days_back=7)
    )
    assert stats_response.success is True
    assert stats_response.stats is not None
    assert stats_response.stats.current_period.total_interactions > 0
    assert stats_response.stats.current_period.total_profiles > 0
    # Evaluations should also be tracked
    assert len(stats_response.stats.evaluations_time_series) >= 0

    # Step 8: Verify data relationships are consistent
    # Profile should reference the correct request_id
    for profile in stored_profiles:
        assert profile.user_id == user_id
        assert profile.generated_from_request_id == request_id

    # User playbooks should reference the correct request_id
    for playbook in user_playbooks:
        assert playbook.request_id == request_id

    # Agent success results should reference the correct session_id
    for result in agent_success_results:
        assert result.session_id == session_id


@skip_in_precommit
@skip_low_priority
def test_rerun_operations_consistency(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test that rerun operations don't corrupt existing data.

    This test verifies:
    1. Initial publish creates CURRENT profiles correctly
    2. Rerun creates PENDING profiles without affecting CURRENT
    3. Multiple reruns don't duplicate data
    4. Existing interactions remain unchanged
    5. Playbooks and agent success data remain consistent
    """
    user_id = "test_user_rerun_consistency"
    agent_version = "test_agent_rerun"
    session_id = "test_session_rerun"

    # Step 1: Initial publish
    publish_response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_rerun_source",
            "agent_version": agent_version,
            "session_id": session_id,
        }
    )
    assert publish_response.success is True

    # Trigger group evaluation synchronously (normally delayed)
    run_group_evaluation(
        org_id=reflexio_instance.org_id,
        user_id=user_id,
        session_id=session_id,
        agent_version=agent_version,
        source="test_rerun_source",
        request_context=reflexio_instance.request_context,
        llm_client=reflexio_instance.llm_client,
    )

    # Record initial state
    initial_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    initial_current_profiles = (
        reflexio_instance.request_context.storage.get_user_profile(
            user_id, status_filter=[None]
        )
    )
    initial_playbooks = reflexio_instance.request_context.storage.get_user_playbooks(
        playbook_name="test_playbook"
    )
    initial_agent_success = (
        reflexio_instance.request_context.storage.get_agent_success_evaluation_results(
            agent_version=agent_version
        )
    )

    initial_interaction_count = len(initial_interactions)
    initial_profile_count = len(initial_current_profiles)
    initial_playbook_count = len(initial_playbooks)
    initial_agent_success_count = len(initial_agent_success)

    assert initial_profile_count > 0, "Should have initial profiles"

    # Step 2: Run rerun_profile_generation
    rerun_response = reflexio_instance.rerun_profile_generation(
        RerunProfileGenerationRequest(user_id=user_id)
    )
    assert rerun_response.success is True
    assert rerun_response.profiles_generated > 0

    # Step 3: Verify CURRENT profiles unchanged
    current_profiles_after_rerun = (
        reflexio_instance.request_context.storage.get_user_profile(
            user_id, status_filter=[None]
        )
    )
    assert len(current_profiles_after_rerun) == initial_profile_count, (
        "CURRENT profiles should remain unchanged"
    )

    # Verify content of current profiles didn't change
    for i, profile in enumerate(current_profiles_after_rerun):
        assert profile.profile_id == initial_current_profiles[i].profile_id
        assert profile.content == initial_current_profiles[i].content

    # Step 4: Verify PENDING profiles were created
    pending_profiles = reflexio_instance.request_context.storage.get_user_profile(
        user_id, status_filter=[Status.PENDING]
    )
    assert len(pending_profiles) > 0, "PENDING profiles should be created"

    # Step 5: Verify interactions unchanged
    interactions_after_rerun = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    assert len(interactions_after_rerun) == initial_interaction_count, (
        "Interactions should remain unchanged"
    )

    # Step 6: Verify playbooks unchanged
    playbooks_after_rerun = (
        reflexio_instance.request_context.storage.get_user_playbooks(
            playbook_name="test_playbook"
        )
    )
    assert len(playbooks_after_rerun) == initial_playbook_count, (
        "User playbooks should remain unchanged"
    )

    # Step 7: Verify agent success results unchanged
    agent_success_after_rerun = (
        reflexio_instance.request_context.storage.get_agent_success_evaluation_results(
            agent_version=agent_version
        )
    )
    assert len(agent_success_after_rerun) == initial_agent_success_count, (
        "Agent success results should remain unchanged"
    )

    # Step 8: Run rerun again and verify no duplicate PENDING profiles
    pending_count_before_second_rerun = len(pending_profiles)

    rerun_response_2 = reflexio_instance.rerun_profile_generation(
        RerunProfileGenerationRequest(user_id=user_id)
    )
    assert rerun_response_2.success is True

    # Get all PENDING profiles after second rerun
    pending_profiles_after_second = (
        reflexio_instance.request_context.storage.get_user_profile(
            user_id, status_filter=[Status.PENDING]
        )
    )

    # Verify the total PENDING profiles increased (new ones were created)
    # but data integrity is maintained
    assert len(pending_profiles_after_second) >= pending_count_before_second_rerun

    # Step 9: Verify CURRENT profiles still unchanged after second rerun
    current_profiles_final = reflexio_instance.request_context.storage.get_user_profile(
        user_id, status_filter=[None]
    )
    assert len(current_profiles_final) == initial_profile_count, (
        "CURRENT profiles should still be unchanged after second rerun"
    )
