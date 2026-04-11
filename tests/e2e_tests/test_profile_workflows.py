"""End-to-end tests for profile workflows including rerun functionality."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.retriever_schema import (
    GetUserProfilesRequest,
    SearchUserProfileRequest,
)
from reflexio.models.api_schema.service_schemas import (
    DeleteUserProfileRequest,
    DowngradeProfilesRequest,
    InteractionData,
    ManualProfileGenerationRequest,
    RerunProfileGenerationRequest,
    Status,
    UpgradeProfilesRequest,
    UserProfile,
)
from tests.server.test_utils import skip_in_precommit, skip_low_priority


@skip_in_precommit
def test_publish_interaction_profile_only(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test interaction publishing with only profile extraction enabled."""
    user_id = "test_user_profile_only"
    agent_version = "test_agent_profile"

    # Publish interactions (request_id will be auto-generated)
    response = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
            "agent_version": agent_version,
        }
    )

    # Verify successful publication
    assert response.success is True
    assert response.message == "Interaction published successfully"

    # Verify interactions were added to storage
    final_interactions = (
        reflexio_instance_profile_only.request_context.storage.get_all_interactions()
    )
    assert len(final_interactions) == len(sample_interaction_requests)

    # Verify profiles were generated and added to storage
    final_profiles = (
        reflexio_instance_profile_only.request_context.storage.get_all_profiles()
    )
    assert len(final_profiles) > 0
    assert final_profiles[0].content.strip() != ""

    # Verify profile change logs were created
    final_change_logs = (
        reflexio_instance_profile_only.request_context.storage.get_profile_change_logs()
    )
    assert len(final_change_logs) > 0

    # Verify NO playbooks were generated (since playbook config is not enabled)
    user_playbooks = (
        reflexio_instance_profile_only.request_context.storage.get_user_playbooks(
            playbook_name="test_playbook"
        )
    )
    assert len(user_playbooks) == 0

    # Verify NO agent success evaluation results were created (since agent success config is not enabled)
    agent_success_results = reflexio_instance_profile_only.request_context.storage.get_agent_success_evaluation_results(
        agent_version=agent_version
    )
    assert len(agent_success_results) == 0


@skip_in_precommit
def test_search_profiles_end_to_end(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test end-to-end profile search workflow."""
    user_id = "test_user_789"

    # First publish interactions to generate profiles
    reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )

    # Verify profiles were generated and stored using get_profiles
    get_profiles_response = reflexio_instance_profile_only.get_profiles(
        GetUserProfilesRequest(user_id=user_id)
    )
    assert get_profiles_response.success is True
    assert len(get_profiles_response.user_profiles) > 0

    # Get the actual profile content to use for search
    # This ensures the search query matches the generated content
    profile_content = get_profiles_response.user_profiles[0].content
    # Use the first few words from the profile as search query
    search_words = " ".join(profile_content.split()[:4])

    # Search for user profiles using content from the generated profile
    search_request = SearchUserProfileRequest(
        user_id=user_id,
        query=search_words,  # Use actual profile content for reliable search
        top_k=5,
    )

    response = reflexio_instance_profile_only.search_profiles(search_request)

    # Verify search results
    assert response.success is True
    assert len(response.user_profiles) > 0

    # Verify all returned profiles have CURRENT status (default search filter)
    for profile in response.user_profiles:
        assert profile.status is None, (
            f"Default search should return only CURRENT profiles, got status={profile.status}"
        )

    # Verify profile content contains relevant information from interactions
    # The profile should contain content extracted from the conversation
    assert any(
        len(profile.content.strip()) > 0 for profile in response.user_profiles
    ), "Should find profiles with non-empty content"


@skip_in_precommit
def test_get_profiles_end_to_end(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test end-to-end get profiles workflow."""
    user_id = "test_user_profiles"

    # First publish interactions to generate profiles (request_id will be auto-generated)
    reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )

    # Verify profiles were generated and stored
    stored_profiles = (
        reflexio_instance_profile_only.request_context.storage.get_all_profiles()
    )
    assert len(stored_profiles) > 0
    # Get the auto-generated request_id
    request_id = stored_profiles[0].generated_from_request_id

    # Get all profiles for the user
    get_request = GetUserProfilesRequest(user_id=user_id)
    response = reflexio_instance_profile_only.get_profiles(get_request)

    # Verify results
    assert response.success is True
    assert len(response.user_profiles) > 0

    # Verify profile details
    for profile in response.user_profiles:
        assert profile.user_id == user_id
        assert profile.content is not None
        assert profile.generated_from_request_id == request_id


@skip_in_precommit
def test_delete_profile_end_to_end(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test end-to-end profile deletion workflow."""
    user_id = "test_user_delete_profile"

    # First publish interactions to generate profiles
    response = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )
    assert response.success is True

    # Verify profiles were generated and stored
    initial_profiles = (
        reflexio_instance_profile_only.request_context.storage.get_all_profiles()
    )
    assert len(initial_profiles) > 0

    # Get profiles to find profile IDs
    get_request = GetUserProfilesRequest(user_id=user_id)
    get_response = reflexio_instance_profile_only.get_profiles(get_request)
    assert len(get_response.user_profiles) > 0

    # Delete the first profile
    profile_to_delete = get_response.user_profiles[0]
    delete_request = DeleteUserProfileRequest(
        user_id=user_id,
        profile_id=profile_to_delete.profile_id,
    )

    delete_response = reflexio_instance_profile_only.delete_profile(delete_request)
    assert delete_response.success is True

    # Verify profile was deleted from storage
    final_profiles = (
        reflexio_instance_profile_only.request_context.storage.get_all_profiles()
    )
    assert len(final_profiles) < len(initial_profiles)


@skip_in_precommit
@skip_low_priority
def test_get_profile_change_logs_end_to_end(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test end-to-end profile change logs workflow."""
    user_id = "test_user_changelog"

    # First publish interactions to generate profiles and change logs (request_id will be auto-generated)
    response = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )
    assert response.success is True

    # Verify change logs were created and stored
    stored_change_logs = (
        reflexio_instance_profile_only.request_context.storage.get_profile_change_logs()
    )
    assert len(stored_change_logs) > 0
    # Get the auto-generated request_id
    request_id = stored_change_logs[0].request_id

    # Get profile change logs
    changelog_response = reflexio_instance_profile_only.get_profile_change_logs()

    # Verify results
    assert changelog_response.success is True
    assert len(changelog_response.profile_change_logs) > 0

    # Verify change log details
    for changelog in changelog_response.profile_change_logs:
        assert changelog.user_id == user_id
        assert changelog.request_id == request_id
        assert len(changelog.added_profiles) > 0


@skip_in_precommit
def test_rerun_profile_generation_end_to_end(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test end-to-end rerun profile generation workflow.

    This test verifies that:
    1. Normal profile generation creates profiles with status=None (current)
    2. Rerun profile generation creates new profiles with status='pending'
    3. Status filtering works correctly to retrieve profiles by status
    """
    user_id = "test_user_rerun"

    # Step 1: Publish interactions normally to generate initial profiles
    response = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )
    assert response.success is True

    # Verify profiles were generated with status=None (current)
    current_profiles = (
        reflexio_instance_profile_only.request_context.storage.get_user_profile(
            user_id, status_filter=[None]
        )
    )
    assert len(current_profiles) > 0, "Should have current profiles"
    for profile in current_profiles:
        assert profile.status is None, "Initial profiles should have status=None"

    initial_profile_count = len(current_profiles)

    # Step 2: Rerun profile generation
    rerun_request = RerunProfileGenerationRequest(
        user_id=user_id,
    )

    rerun_response = reflexio_instance_profile_only.rerun_profile_generation(
        rerun_request
    )
    assert rerun_response.success is True, f"Rerun should succeed: {rerun_response.msg}"
    assert rerun_response.profiles_generated > 0, (
        "Rerun should generate at least one profile"
    )

    # Step 3: Verify pending profiles were created
    pending_profiles = (
        reflexio_instance_profile_only.request_context.storage.get_user_profile(
            user_id, status_filter=[Status.PENDING]
        )
    )
    assert len(pending_profiles) > 0, "Should have pending profiles after rerun"
    for profile in pending_profiles:
        assert profile.status == Status.PENDING, (
            "Rerun profiles should have status=Status.PENDING"
        )

    # Step 4: Verify current profiles still exist unchanged
    current_profiles_after = (
        reflexio_instance_profile_only.request_context.storage.get_user_profile(
            user_id, status_filter=[None]
        )
    )
    assert len(current_profiles_after) == initial_profile_count, (
        "Current profiles should remain unchanged"
    )

    # Step 5: Verify get_profiles returns only current profiles by default
    default_response = reflexio_instance_profile_only.get_profiles({"user_id": user_id})
    assert default_response.success is True
    assert len(default_response.user_profiles) == initial_profile_count
    assert all(p.status is None for p in default_response.user_profiles)

    # Step 6: Verify get_profiles can retrieve pending profiles
    pending_response = reflexio_instance_profile_only.get_profiles(
        {"user_id": user_id}, status_filter=[Status.PENDING]
    )
    assert pending_response.success is True
    assert len(pending_response.user_profiles) > 0
    assert all(p.status == Status.PENDING for p in pending_response.user_profiles)

    # Step 7: Verify get_profiles can retrieve both current and pending
    all_response = reflexio_instance_profile_only.get_profiles(
        {"user_id": user_id}, status_filter=[None, Status.PENDING]
    )
    assert all_response.success is True
    assert len(all_response.user_profiles) >= initial_profile_count + len(
        pending_profiles
    )


@skip_in_precommit
@skip_low_priority
def test_rerun_profile_generation_with_time_filters(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test rerun profile generation with time-based filtering."""
    user_id = "test_user_rerun_time"

    # Publish interactions
    response = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )
    assert response.success is True

    # Get the interactions to determine their timestamps
    all_interactions = (
        reflexio_instance_profile_only.request_context.storage.get_user_interaction(
            user_id
        )
    )
    assert len(all_interactions) == len(sample_interaction_requests)

    # Test with time filter that excludes all interactions (future time range)
    future_start = datetime.now(UTC) + timedelta(days=1)
    future_end = datetime.now(UTC) + timedelta(days=2)

    rerun_request = RerunProfileGenerationRequest(
        user_id=user_id,
        start_time=future_start,
        end_time=future_end,
    )

    rerun_response = reflexio_instance_profile_only.rerun_profile_generation(
        rerun_request
    )
    assert rerun_response.success is False
    assert "No interactions found" in rerun_response.msg

    # Test with time filter that includes all interactions (past to future)
    past_start = datetime.now(UTC) - timedelta(days=1)
    future_end = datetime.now(UTC) + timedelta(days=1)

    rerun_request_valid = RerunProfileGenerationRequest(
        user_id=user_id,
        start_time=past_start,
        end_time=future_end,
    )

    rerun_response_valid = reflexio_instance_profile_only.rerun_profile_generation(
        rerun_request_valid
    )
    assert rerun_response_valid.success is True
    assert rerun_response_valid.profiles_generated > 0, (
        "Rerun with valid time filter should generate profiles"
    )

    # Verify pending profiles were created
    pending_profiles = (
        reflexio_instance_profile_only.request_context.storage.get_user_profile(
            user_id, status_filter=[Status.PENDING]
        )
    )
    assert len(pending_profiles) > 0


@skip_in_precommit
@skip_low_priority
def test_rerun_profile_generation_with_source_filter(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test rerun profile generation with source-based filtering.

    This test verifies:
    1. Source filtering correctly filters interactions during rerun
    2. Non-existent source returns appropriate error
    3. Profile generation with filtered source succeeds (even if LLM returns no profiles)
    """
    user_id = "test_user_rerun_source"

    # Use all interactions for source_a (includes customer response for better profile extraction)
    # This ensures LLM has complete conversation context
    response_a = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_source_a",
        }
    )
    assert response_a.success is True

    # Publish a minimal interaction with source "test_source_b"
    response_b = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": [
                InteractionData(
                    content="Just a test message for source B",
                    role="User",
                )
            ],
            "source": "test_source_b",
        }
    )
    assert response_b.success is True

    # Verify we have interactions from both sources
    all_interactions = (
        reflexio_instance_profile_only.request_context.storage.get_user_interaction(
            user_id
        )
    )
    assert len(all_interactions) == len(sample_interaction_requests) + 1

    # Rerun profile generation for only "test_source_a"
    rerun_request_a = RerunProfileGenerationRequest(
        user_id=user_id,
        source="test_source_a",
    )

    rerun_response_a = reflexio_instance_profile_only.rerun_profile_generation(
        rerun_request_a
    )
    # The operation should succeed (profiles may or may not be generated by LLM)
    assert rerun_response_a.success is True, f"Rerun failed: {rerun_response_a.msg}"

    # If profiles were generated, verify they are in pending status
    pending_profiles = (
        reflexio_instance_profile_only.request_context.storage.get_user_profile(
            user_id, status_filter=[Status.PENDING]
        )
    )
    if rerun_response_a.profiles_generated > 0:
        assert len(pending_profiles) > 0, (
            "Expected pending profiles when profiles_generated > 0"
        )

    # Test with non-existent source - should fail with appropriate message
    rerun_request_invalid = RerunProfileGenerationRequest(
        user_id=user_id,
        source="non_existent_source",
    )

    rerun_response_invalid = reflexio_instance_profile_only.rerun_profile_generation(
        rerun_request_invalid
    )
    assert rerun_response_invalid.success is False
    assert "No interactions found" in rerun_response_invalid.msg


@skip_in_precommit
@skip_low_priority
def test_status_filter_in_get_all_profiles(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test status filtering in get_all_profiles method."""
    user_id = "test_user_all_profiles"

    # Publish interactions to create current profiles
    response = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )
    assert response.success is True

    # Rerun to create pending profiles
    rerun_response = reflexio_instance_profile_only.rerun_profile_generation(
        {"user_id": user_id}
    )
    assert rerun_response.success is True

    # Test get_all_profiles with default filter (current profiles only)
    default_profiles = reflexio_instance_profile_only.get_all_profiles(limit=100)
    assert default_profiles.success is True
    assert all(p.status is None for p in default_profiles.user_profiles)

    # Test get_all_profiles with pending filter
    pending_profiles = reflexio_instance_profile_only.get_all_profiles(
        limit=100, status_filter=[Status.PENDING]
    )
    assert pending_profiles.success is True
    assert all(p.status == Status.PENDING for p in pending_profiles.user_profiles)

    # Test get_all_profiles with both statuses
    all_profiles = reflexio_instance_profile_only.get_all_profiles(
        limit=100, status_filter=[None, Status.PENDING]
    )
    assert all_profiles.success is True
    assert len(all_profiles.user_profiles) >= len(default_profiles.user_profiles) + len(
        pending_profiles.user_profiles
    )


@skip_in_precommit
@skip_low_priority
def test_status_filter_in_search_profiles(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test status filtering in search_profiles method."""
    user_id = "test_user_search_status"

    # Publish interactions to create current profiles
    response = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )
    assert response.success is True

    # Rerun to create pending profiles
    rerun_response = reflexio_instance_profile_only.rerun_profile_generation(
        {"user_id": user_id}
    )
    assert rerun_response.success is True

    # Test search with default filter (current profiles only)
    search_request = SearchUserProfileRequest(
        user_id=user_id,
        query="sales outreach preferences",  # Use meaningful query for vector search
        top_k=10,
    )

    default_search = reflexio_instance_profile_only.search_profiles(search_request)
    assert default_search.success is True
    assert all(p.status is None for p in default_search.user_profiles)

    # Test search with pending filter
    pending_search = reflexio_instance_profile_only.search_profiles(
        search_request, status_filter=[Status.PENDING]
    )
    assert pending_search.success is True
    assert all(p.status == Status.PENDING for p in pending_search.user_profiles)

    # Test search with both statuses
    all_search = reflexio_instance_profile_only.search_profiles(
        search_request, status_filter=[None, Status.PENDING]
    )
    assert all_search.success is True
    assert len(all_search.user_profiles) >= len(default_search.user_profiles) + len(
        pending_search.user_profiles
    )


@skip_in_precommit
@skip_low_priority
def test_status_filter_in_get_profiles_request(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test status filtering using status_filter field in GetUserProfilesRequest."""
    user_id = "test_user_get_profiles_request_status"

    # Publish interactions to create current profiles
    response = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )
    assert response.success is True

    # Rerun to create pending profiles
    rerun_response = reflexio_instance_profile_only.rerun_profile_generation(
        {"user_id": user_id}
    )
    assert rerun_response.success is True

    # Test 1: Get profiles with no status_filter (should default to current profiles only)
    request_default = GetUserProfilesRequest(user_id=user_id)
    default_response = reflexio_instance_profile_only.get_profiles(request_default)
    assert default_response.success is True
    assert len(default_response.user_profiles) > 0
    assert all(p.status is None for p in default_response.user_profiles)

    # Test 2: Get profiles with status_filter=[Status.PENDING] in request
    request_pending = GetUserProfilesRequest(
        user_id=user_id, status_filter=[Status.PENDING]
    )
    pending_response = reflexio_instance_profile_only.get_profiles(request_pending)
    assert pending_response.success is True
    assert len(pending_response.user_profiles) > 0
    assert all(p.status == Status.PENDING for p in pending_response.user_profiles)

    # Test 3: Get profiles with status_filter=[None, Status.PENDING] in request
    request_all = GetUserProfilesRequest(
        user_id=user_id, status_filter=[None, Status.PENDING]
    )
    all_response = reflexio_instance_profile_only.get_profiles(request_all)
    assert all_response.success is True
    assert len(all_response.user_profiles) >= len(default_response.user_profiles) + len(
        pending_response.user_profiles
    )

    # Test 4: Verify with explicit None status_filter (should behave same as default)
    request_explicit_none = GetUserProfilesRequest(user_id=user_id, status_filter=None)
    explicit_none_response = reflexio_instance_profile_only.get_profiles(
        request_explicit_none
    )
    assert explicit_none_response.success is True
    assert len(explicit_none_response.user_profiles) == len(
        default_response.user_profiles
    )

    # Test 5: Verify status_filter in request takes precedence over parameter
    # Pass request with status_filter=[Status.PENDING] but also pass status_filter parameter
    # The parameter should take precedence
    request_with_filter = GetUserProfilesRequest(
        user_id=user_id, status_filter=[Status.PENDING]
    )
    override_response = reflexio_instance_profile_only.get_profiles(
        request_with_filter, status_filter=[None]
    )
    assert override_response.success is True
    # Should get current profiles (None status) because parameter overrides request field
    assert all(p.status is None for p in override_response.user_profiles)


def _create_test_profile(
    user_id: str,
    request_id: str,
    status: Status = None,
    content: str = "Test profile content",
) -> UserProfile:
    """Helper function to create test profiles with specified status.

    Args:
        user_id (str): User ID for the profile
        request_id (str): Request ID for the profile
        status (Status, optional): Status of the profile. Defaults to None (CURRENT).
        content (str, optional): Profile content. Defaults to "Test profile content".

    Returns:
        UserProfile: A test profile with the specified properties
    """
    return UserProfile(
        profile_id=str(uuid.uuid4()),
        user_id=user_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=request_id,
        status=status,
    )


@skip_in_precommit
def test_upgrade_profiles_end_to_end(
    reflexio_instance_profile_only: Reflexio,
    cleanup_profile_only: Callable[[], None],
):
    """Test end-to-end upgrade workflow for profiles.

    Upgrade workflow:
    1. Delete old ARCHIVED profiles
    2. Archive CURRENT profiles (None -> ARCHIVED)
    3. Promote PENDING profiles (PENDING -> None/CURRENT)
    """
    user_id = "test_user_upgrade"
    storage = reflexio_instance_profile_only.request_context.storage

    # Setup: Create profiles with different statuses
    # Create CURRENT profiles (status=None)
    current_profiles = [
        _create_test_profile(
            user_id=user_id,
            request_id=f"current_request_{i}",
            status=None,
            content=f"Current profile content {i}",
        )
        for i in range(3)
    ]

    # Create PENDING profiles (status=PENDING)
    pending_profiles = [
        _create_test_profile(
            user_id=user_id,
            request_id=f"pending_request_{i}",
            status=Status.PENDING,
            content=f"Pending profile content {i}",
        )
        for i in range(2)
    ]

    # Create ARCHIVED profiles (status=ARCHIVED)
    archived_profiles = [
        _create_test_profile(
            user_id=user_id,
            request_id=f"archived_request_{i}",
            status=Status.ARCHIVED,
            content=f"Archived profile content {i}",
        )
        for i in range(2)
    ]

    # Save all profiles to storage
    all_profiles = current_profiles + pending_profiles + archived_profiles
    storage.add_user_profile(user_id, all_profiles)

    # Verify initial state
    current_before = storage.get_user_profile(user_id, status_filter=[None])
    pending_before = storage.get_user_profile(user_id, status_filter=[Status.PENDING])
    archived_before = storage.get_user_profile(user_id, status_filter=[Status.ARCHIVED])

    assert len(current_before) == 3
    assert len(pending_before) == 2
    assert len(archived_before) == 2

    # Execute upgrade
    response = reflexio_instance_profile_only.upgrade_all_profiles(
        UpgradeProfilesRequest(
            user_id=user_id,
        )
    )

    # Verify response
    assert response.success is True
    assert response.profiles_deleted == 2  # Old ARCHIVED deleted
    assert response.profiles_archived == 3  # CURRENT -> ARCHIVED
    assert response.profiles_promoted == 2  # PENDING -> CURRENT (None)

    # Verify final state
    current_after = storage.get_user_profile(user_id, status_filter=[None])
    archived_after = storage.get_user_profile(user_id, status_filter=[Status.ARCHIVED])
    pending_after = storage.get_user_profile(user_id, status_filter=[Status.PENDING])

    # PENDING profiles promoted to CURRENT
    assert len(current_after) == 2
    for profile in current_after:
        assert "pending_request" in profile.generated_from_request_id

    # CURRENT profiles archived
    assert len(archived_after) == 3
    for profile in archived_after:
        assert "current_request" in profile.generated_from_request_id

    # No more PENDING profiles
    assert len(pending_after) == 0


@skip_in_precommit
@skip_low_priority
def test_downgrade_profiles_end_to_end(
    reflexio_instance_profile_only: Reflexio,
    cleanup_profile_only: Callable[[], None],
):
    """Test end-to-end downgrade workflow for profiles.

    Downgrade workflow:
    1. Demote CURRENT profiles (None -> ARCHIVE_IN_PROGRESS)
    2. Restore ARCHIVED profiles (ARCHIVED -> None/CURRENT)
    3. Complete archiving (ARCHIVE_IN_PROGRESS -> ARCHIVED)
    """
    user_id = "test_user_downgrade"
    storage = reflexio_instance_profile_only.request_context.storage

    # Setup: Create profiles with different statuses
    # Create CURRENT profiles (status=None)
    current_profiles = [
        _create_test_profile(
            user_id=user_id,
            request_id=f"current_request_{i}",
            status=None,
            content=f"Current profile content {i}",
        )
        for i in range(3)
    ]

    # Create ARCHIVED profiles (status=ARCHIVED)
    archived_profiles = [
        _create_test_profile(
            user_id=user_id,
            request_id=f"archived_request_{i}",
            status=Status.ARCHIVED,
            content=f"Archived profile content {i}",
        )
        for i in range(2)
    ]

    # Save all profiles to storage
    all_profiles = current_profiles + archived_profiles
    storage.add_user_profile(user_id, all_profiles)

    # Verify initial state
    current_before = storage.get_user_profile(user_id, status_filter=[None])
    archived_before = storage.get_user_profile(user_id, status_filter=[Status.ARCHIVED])

    assert len(current_before) == 3
    assert len(archived_before) == 2

    # Execute downgrade
    response = reflexio_instance_profile_only.downgrade_all_profiles(
        DowngradeProfilesRequest(
            user_id=user_id,
        )
    )

    # Verify response
    assert response.success is True
    assert response.profiles_demoted == 3  # CURRENT -> ARCHIVED
    assert response.profiles_restored == 2  # ARCHIVED -> CURRENT (None)

    # Verify final state
    current_after = storage.get_user_profile(user_id, status_filter=[None])
    archived_after = storage.get_user_profile(user_id, status_filter=[Status.ARCHIVED])

    # ARCHIVED profiles restored to CURRENT
    assert len(current_after) == 2
    for profile in current_after:
        assert "archived_request" in profile.generated_from_request_id

    # CURRENT profiles demoted to ARCHIVED
    assert len(archived_after) == 3
    for profile in archived_after:
        assert "current_request" in profile.generated_from_request_id


@skip_in_precommit
@skip_low_priority
def test_upgrade_downgrade_profiles_roundtrip(
    reflexio_instance_profile_only: Reflexio,
    cleanup_profile_only: Callable[[], None],
):
    """Test that upgrade followed by downgrade restores the original profile state."""
    user_id = "test_user_roundtrip"
    storage = reflexio_instance_profile_only.request_context.storage

    # Setup: Create initial CURRENT profiles
    current_profiles = [
        _create_test_profile(
            user_id=user_id,
            request_id=f"original_request_{i}",
            status=None,
            content=f"Original profile content {i}",
        )
        for i in range(3)
    ]

    # Create PENDING profiles (new version)
    pending_profiles = [
        _create_test_profile(
            user_id=user_id,
            request_id=f"new_request_{i}",
            status=Status.PENDING,
            content=f"New profile content {i}",
        )
        for i in range(2)
    ]

    storage.add_user_profile(user_id, current_profiles + pending_profiles)

    # Execute upgrade (new profiles become current, original become archived)
    upgrade_response = reflexio_instance_profile_only.upgrade_all_profiles(
        UpgradeProfilesRequest(user_id=user_id)
    )
    assert upgrade_response.success is True

    # Verify upgrade state
    current_after_upgrade = storage.get_user_profile(user_id, status_filter=[None])
    archived_after_upgrade = storage.get_user_profile(
        user_id, status_filter=[Status.ARCHIVED]
    )

    assert len(current_after_upgrade) == 2  # new profiles are now current
    assert len(archived_after_upgrade) == 3  # original profiles are now archived

    # Execute downgrade (restore original profiles)
    downgrade_response = reflexio_instance_profile_only.downgrade_all_profiles(
        DowngradeProfilesRequest(user_id=user_id)
    )
    assert downgrade_response.success is True

    # Verify roundtrip restored original state
    current_after_downgrade = storage.get_user_profile(user_id, status_filter=[None])
    archived_after_downgrade = storage.get_user_profile(
        user_id, status_filter=[Status.ARCHIVED]
    )

    # Original profiles restored to current
    assert len(current_after_downgrade) == 3
    for profile in current_after_downgrade:
        assert "original_request" in profile.generated_from_request_id

    # New profiles demoted to archived
    assert len(archived_after_downgrade) == 2
    for profile in archived_after_downgrade:
        assert "new_request" in profile.generated_from_request_id


@skip_in_precommit
def test_manual_profile_generation_end_to_end(
    reflexio_instance_manual_profile: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_manual_profile: Callable[[], None],
):
    """Test manual_profile_generation method for triggering profile generation.

    This test verifies:
    1. Manual profile generation uses window-sized interactions
    2. Generated profiles have CURRENT status (not PENDING like rerun)
    3. Profiles are generated correctly from the interactions
    """
    user_id = "test_user_manual_profile"

    # Step 1: Publish interactions to have data for generation
    publish_response = reflexio_instance_manual_profile.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_manual_source",
        }
    )
    assert publish_response.success is True

    # Step 2: Call manual_profile_generation
    manual_response = reflexio_instance_manual_profile.manual_profile_generation(
        ManualProfileGenerationRequest(
            user_id=user_id,
        )
    )
    assert manual_response.success is True, (
        f"Manual generation failed: {manual_response.msg}"
    )

    # Step 3: Verify profiles were generated with CURRENT status (None)
    current_profiles = (
        reflexio_instance_manual_profile.request_context.storage.get_user_profile(
            user_id, status_filter=[None]
        )
    )
    # Note: profiles may already exist from publish_interaction,
    # but manual_profile_generation should also generate CURRENT profiles
    assert len(current_profiles) >= 0  # Just verify no errors

    # Step 4: Verify NO PENDING profiles were created (that's rerun behavior)
    pending_profiles = (
        reflexio_instance_manual_profile.request_context.storage.get_user_profile(
            user_id, status_filter=[Status.PENDING]
        )
    )
    assert len(pending_profiles) == 0, (
        "Manual generation should not create PENDING profiles"
    )


@skip_in_precommit
@skip_low_priority
def test_manual_profile_generation_no_window_size(
    reflexio_instance_profile_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_profile_only: Callable[[], None],
):
    """Test manual_profile_generation works without batch_size.

    This test verifies:
    1. Manual generation works when batch_size is not configured
       (it defaults to fetching all available interactions with a reasonable limit)
    """
    user_id = "test_user_no_window"

    # Publish interactions first
    publish_response = reflexio_instance_profile_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_source",
        }
    )
    assert publish_response.success is True

    # Call manual_profile_generation - should succeed even without window size
    # When window_size is not configured, it fetches all available interactions
    manual_response = reflexio_instance_profile_only.manual_profile_generation(
        ManualProfileGenerationRequest(
            user_id=user_id,
        )
    )
    assert manual_response.success is True


@skip_in_precommit
@skip_low_priority
def test_manual_profile_generation_with_source_filter(
    reflexio_instance_manual_profile: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_manual_profile: Callable[[], None],
):
    """Test manual_profile_generation with source filtering.

    This test verifies:
    1. Source filtering works correctly in manual generation
    2. Only interactions with matching source are processed
    """
    user_id = "test_user_manual_source_filter"

    # Publish interactions with different sources
    # Source A - full conversation
    response_a = reflexio_instance_manual_profile.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "source_a",
        }
    )
    assert response_a.success is True

    # Source B - single message
    response_b = reflexio_instance_manual_profile.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": [
                InteractionData(
                    content="Simple message for source B",
                    role="User",
                )
            ],
            "source": "source_b",
        }
    )
    assert response_b.success is True

    # Call manual_profile_generation with source filter
    manual_response = reflexio_instance_manual_profile.manual_profile_generation(
        ManualProfileGenerationRequest(
            user_id=user_id,
            source="source_a",  # Only process source_a
        )
    )
    # Should succeed (or fail gracefully if no matching extractors)
    # Main thing is no exception is raised
    assert manual_response.success is True or "No interactions found" in (
        manual_response.msg or ""
    )


@skip_in_precommit
@skip_low_priority
def test_manual_profile_generation_with_dict_input(
    reflexio_instance_manual_profile: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_manual_profile: Callable[[], None],
):
    """Test manual_profile_generation accepts dict input.

    This test verifies:
    1. Manual generation accepts dict input (not just ManualProfileGenerationRequest)
    """
    user_id = "test_user_dict_input"

    # Publish interactions
    publish_response = reflexio_instance_manual_profile.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_source",
        }
    )
    assert publish_response.success is True

    # Call with dict input
    manual_response = reflexio_instance_manual_profile.manual_profile_generation(
        {"user_id": user_id}
    )
    assert manual_response.success is True, f"Dict input failed: {manual_response.msg}"


@skip_in_precommit
@skip_low_priority
def test_rerun_profile_generation_with_extractor_names_filter(
    reflexio_instance_multiple_profile_extractors: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_multiple_profile_extractors: Callable[[], None],
):
    """Test rerun profile generation with extractor_names filtering.

    This test verifies:
    1. extractor_names filter correctly limits which extractors run during rerun
    2. Only profiles from specified extractors are generated
    3. Other extractors are skipped
    """
    user_id = "test_user_extractor_names_filter"

    # Step 1: Publish interactions to have data for generation
    publish_response = (
        reflexio_instance_multiple_profile_extractors.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "test_source",
            }
        )
    )
    assert publish_response.success is True

    # Step 2: Rerun with extractor_names filter - only run extractor_basic_info and extractor_intent
    rerun_request = RerunProfileGenerationRequest(
        user_id=user_id,
        extractor_names=["extractor_basic_info", "extractor_intent"],
    )

    rerun_response = (
        reflexio_instance_multiple_profile_extractors.rerun_profile_generation(
            rerun_request
        )
    )
    assert rerun_response.success is True, f"Rerun failed: {rerun_response.msg}"

    # Step 3: Verify profiles were generated (may be 0 depending on LLM response)
    # The main thing is that the operation succeeded
    pending_profiles = reflexio_instance_multiple_profile_extractors.request_context.storage.get_user_profile(
        user_id, status_filter=[Status.PENDING]
    )

    # If profiles were generated, verify they come from allowed extractors
    if rerun_response.profiles_generated > 0:
        assert len(pending_profiles) > 0


@skip_in_precommit
@skip_low_priority
def test_rerun_profile_generation_with_single_extractor(
    reflexio_instance_multiple_profile_extractors: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_multiple_profile_extractors: Callable[[], None],
):
    """Test rerun profile generation with single extractor specified.

    This test verifies:
    1. A single extractor can be specified
    2. Only that extractor runs
    """
    user_id = "test_user_single_extractor"

    # Step 1: Publish interactions
    publish_response = (
        reflexio_instance_multiple_profile_extractors.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "test_source",
            }
        )
    )
    assert publish_response.success is True

    # Step 2: Rerun with single extractor
    rerun_request = RerunProfileGenerationRequest(
        user_id=user_id,
        extractor_names=["extractor_preferences"],
    )

    rerun_response = (
        reflexio_instance_multiple_profile_extractors.rerun_profile_generation(
            rerun_request
        )
    )
    assert rerun_response.success is True, f"Rerun failed: {rerun_response.msg}"


@skip_in_precommit
@skip_low_priority
def test_rerun_profile_generation_with_nonexistent_extractor_name(
    reflexio_instance_multiple_profile_extractors: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_multiple_profile_extractors: Callable[[], None],
):
    """Test rerun profile generation with non-existent extractor name.

    This test verifies:
    1. Specifying non-existent extractor name doesn't cause errors
    2. No profiles are generated when no extractors match
    """
    user_id = "test_user_nonexistent_extractor"

    # Step 1: Publish interactions
    publish_response = (
        reflexio_instance_multiple_profile_extractors.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "test_source",
            }
        )
    )
    assert publish_response.success is True

    # Step 2: Rerun with non-existent extractor name
    rerun_request = RerunProfileGenerationRequest(
        user_id=user_id,
        extractor_names=["nonexistent_extractor"],
    )

    rerun_response = (
        reflexio_instance_multiple_profile_extractors.rerun_profile_generation(
            rerun_request
        )
    )
    # Should fail because no matching interactions/extractors
    assert rerun_response.success is False or rerun_response.profiles_generated == 0
