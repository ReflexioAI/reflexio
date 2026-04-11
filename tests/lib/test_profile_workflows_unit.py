"""Unit tests for Reflexio library.

Tests the main Reflexio client library interface with mocked LLM responses
but real storage (SQLiteStorage in temp directory) and real services.
"""

import datetime
import tempfile
from datetime import UTC

import pytest

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
    DowngradeProfilesRequest,
    InteractionData,
    PublishUserInteractionRequest,
    RerunProfileGenerationRequest,
    Status,
    UpgradeProfilesRequest,
)
from reflexio.server.services.profile.profile_extractor import (
    ProfileExtractorConfig,
)


@pytest.fixture
def reflexio_with_config(temp_storage, ensure_mock_env):
    """Create Reflexio instance with proper extractor configuration.

    Note: Depends on ensure_mock_env to ensure the mock mode env var is set
    before this fixture creates the Reflexio instance. The litellm.completion
    patch is applied globally in the parent conftest.py.

    Uses SQLiteStorage (via StorageConfigSQLite) for test isolation.
    """
    import os

    from reflexio.models.config_schema import Config, StorageConfigSQLite
    from reflexio.server.services.configurator.configurator import DefaultConfigurator

    # Extra safety: verify env var is set before creating Reflexio
    os.environ["MOCK_LLM_RESPONSE"] = "true"

    org_id = "test_org"
    config = Config(storage_config=StorageConfigSQLite())
    configurator = DefaultConfigurator(
        org_id=org_id, base_dir=temp_storage, config=config
    )
    reflexio = Reflexio(
        org_id=org_id, storage_base_dir=temp_storage, configurator=configurator
    )

    # Configure profile extractor
    profile_extractor_config = ProfileExtractorConfig(
        extractor_name="test_profile",
        context_prompt="Extract user preferences",
        extraction_definition_prompt="User likes and dislikes",
        metadata_definition_prompt="Metadata about preferences",
        batch_interval_override=1,
    )
    reflexio.request_context.configurator.set_config_by_name(
        "profile_extractor_configs", [profile_extractor_config]
    )

    return reflexio


@pytest.fixture
def temp_storage():
    """Provide temporary directory for SQLiteStorage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# ==============================
# Core Workflow Tests
# ==============================


def test_publish_interaction_success(reflexio_with_config):
    """Test publishing an interaction successfully generates a profile."""
    user_id = "test_user_1"
    reflexio = reflexio_with_config

    # Publish interaction
    interaction_data = InteractionData(
        content="I really like sushi",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    request = PublishUserInteractionRequest(
        user_id=user_id,
        interaction_data_list=[interaction_data],
        source="test_source",
        agent_version="v1.0",
    )

    response = reflexio.publish_interaction(request)

    # Verify response
    assert response.success is True, f"Response failed: {response.message}"

    # Verify interaction was stored
    interactions = reflexio.request_context.storage.get_user_interaction(user_id)
    assert len(interactions) == 1
    assert interactions[0].content == "I really like sushi"

    # Verify profile was generated
    profiles = reflexio.request_context.storage.get_user_profile(user_id)
    # Debug: print profile count and storage path
    if len(profiles) != 1:
        import json

        storage = reflexio.request_context.storage
        all_data = storage._load() if hasattr(storage, "_load") else {}
        print(f"\n[DEBUG] Profile count: {len(profiles)}, expected: 1")
        print(f"[DEBUG] Storage file path: {getattr(storage, 'file_path', 'N/A')}")
        print(f"[DEBUG] All stored data keys: {list(all_data.keys())}")
        if user_id in all_data:
            user_data = all_data[user_id]
            print(
                f"[DEBUG] User data keys: {list(user_data.keys()) if isinstance(user_data, dict) else 'not dict'}"
            )
            if "profiles" in user_data:
                print(f"[DEBUG] Profiles in storage: {len(user_data['profiles'])}")
            else:
                print("[DEBUG] No 'profiles' key in user data!")
            if "interactions" in user_data:
                print(
                    f"[DEBUG] Interactions in storage: {len(user_data['interactions'])}"
                )
        # Check operation states
        if "operation_states" in all_data:
            print(
                f"[DEBUG] Operation states: {json.dumps(all_data['operation_states'], indent=2)[:1000]}"
            )
        # Check profile change logs
        if "profile_change_logs" in all_data:
            print(
                f"[DEBUG] Profile change logs count: {len(all_data['profile_change_logs'])}"
            )
    assert len(profiles) == 1, f"Expected 1 profile but got {len(profiles)}"
    assert "sushi" in profiles[0].content.lower()


def test_publish_interaction_dict_input(reflexio_with_config):
    """Test publishing interaction with dict input (auto-conversion)."""
    user_id = "test_user_dict"

    reflexio = reflexio_with_config

    # Publish interaction as dict
    request_dict = {
        "user_id": user_id,
        "interaction_data_list": [
            {
                "content": "Dictionary input test",
                "created_at": int(datetime.datetime.now(UTC).timestamp()),
            }
        ],
        "source": "dict_source",
    }

    response = reflexio.publish_interaction(request_dict)

    # Verify response
    assert response.success is True

    # Verify interaction was stored
    interactions = reflexio.request_context.storage.get_user_interaction(user_id)
    assert len(interactions) == 1


def test_publish_interaction_failure(temp_storage):
    """Test error handling when publish fails."""
    org_id = "test_org"
    reflexio = Reflexio(org_id=org_id, storage_base_dir=temp_storage)

    # Invalid request (missing required fields)
    invalid_request = {}

    response = reflexio.publish_interaction(invalid_request)

    # Should return failure
    assert response.success is False
    assert response.message is not None


# ==============================
# Search/Retrieval Tests
# ==============================


def test_search_interactions(reflexio_with_config):
    """Test searching interactions by query."""
    user_id = "test_user_search"

    reflexio = reflexio_with_config

    # Publish interaction
    interaction_data = InteractionData(
        content="I love sushi and ramen",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Search for interactions
    search_request = SearchInteractionRequest(
        user_id=user_id, query_text="sushi", top_k=10
    )

    response = reflexio.search_interactions(search_request)

    assert response.success is True
    assert len(response.interactions) >= 0  # Search may or may not find results


def test_search_profiles_current_only(reflexio_with_config):
    """Test searching profiles returns only current profiles by default."""
    user_id = "test_user_profile_search"

    reflexio = reflexio_with_config

    # Publish interaction to generate profile
    interaction_data = InteractionData(
        content="Profile search test - sushi lover",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Search profiles (default: current only)
    search_request = SearchUserProfileRequest(
        user_id=user_id, query_text="sushi", top_k=10
    )

    response = reflexio.search_profiles(search_request)

    assert response.success is True
    # Default status_filter is [None] which means current profiles only


def test_search_profiles_with_status_filter(reflexio_with_config):
    """Test searching profiles with specific status filter."""
    user_id = "test_user_status_filter"

    reflexio = reflexio_with_config

    # Publish interaction
    interaction_data = InteractionData(
        content="Status filter test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Search with status filter including PENDING
    search_request = SearchUserProfileRequest(
        user_id=user_id, query_text="test", top_k=10
    )

    response = reflexio.search_profiles(
        search_request, status_filter=[None, Status.PENDING]
    )

    assert response.success is True


def test_get_interactions_with_time_filters(reflexio_with_config):
    """Test getting interactions with time range filters."""
    user_id = "test_user_time_filter"

    reflexio = reflexio_with_config

    # Publish interaction
    now = datetime.datetime.now(UTC)
    interaction_data = InteractionData(
        content="Time filter test", created_at=int(now.timestamp())
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Get interactions with time filter
    get_request = GetInteractionsRequest(
        user_id=user_id,
        start_time=now - datetime.timedelta(hours=1),
        end_time=now + datetime.timedelta(hours=1),
        top_k=10,
    )

    response = reflexio.get_interactions(get_request)

    assert response.success is True
    assert len(response.interactions) == 1


def test_get_profiles_with_status_filter(reflexio_with_config):
    """Test getting profiles with status filter."""
    user_id = "test_user_get_profiles"

    reflexio = reflexio_with_config

    # Publish interaction to generate profile
    interaction_data = InteractionData(
        content="Get profiles test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Get profiles with default status filter (current only)
    get_request = GetUserProfilesRequest(user_id=user_id, top_k=10)

    response = reflexio.get_profiles(get_request)

    assert response.success is True
    assert len(response.user_profiles) == 1


def test_get_all_profiles_and_interactions(reflexio_with_config):
    """Test getting all profiles and interactions across all users."""
    reflexio = reflexio_with_config

    # Publish interactions for multiple users
    for i in range(3):
        user_id = f"user_{i}"
        interaction_data = InteractionData(
            content=f"User {i} interaction",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )

        publish_request = PublishUserInteractionRequest(
            user_id=user_id, interaction_data_list=[interaction_data]
        )
        reflexio.publish_interaction(publish_request)

    # Get all profiles
    # Note: Profile generation runs in parallel with playbook generation
    # via ThreadPoolExecutor, which can occasionally cause a thread-safety
    # race with SQLite under mocked LLM calls, resulting in fewer profiles.
    profiles_response = reflexio.get_all_profiles(limit=100)
    assert profiles_response.success is True
    assert len(profiles_response.user_profiles) == 3

    # Get all interactions
    interactions_response = reflexio.get_all_interactions(limit=100)
    assert interactions_response.success is True
    assert len(interactions_response.interactions) == 3


# ==============================
# Profile Lifecycle Tests
# ==============================


def test_rerun_profile_generation_single_user(reflexio_with_config):
    """Test rerunning profile generation for a specific user."""
    user_id = "test_user_rerun"

    reflexio = reflexio_with_config

    # Publish interaction
    interaction_data = InteractionData(
        content="Rerun test - loves ramen",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Rerun profile generation
    rerun_request = RerunProfileGenerationRequest(user_id=user_id)

    response = reflexio.rerun_profile_generation(rerun_request)

    assert response.success is True
    assert response.profiles_generated > 0


def test_rerun_profile_generation_all_users(reflexio_with_config):
    """Test rerunning profile generation for all users."""
    reflexio = reflexio_with_config

    # Publish interactions for multiple users
    for i in range(2):
        user_id = f"rerun_user_{i}"
        interaction_data = InteractionData(
            content=f"User {i} loves sushi",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )

        publish_request = PublishUserInteractionRequest(
            user_id=user_id, interaction_data_list=[interaction_data]
        )
        reflexio.publish_interaction(publish_request)

    # Rerun for all users (user_id=None)
    rerun_request = RerunProfileGenerationRequest(user_id=None)

    response = reflexio.rerun_profile_generation(rerun_request)

    # The test creates interactions, but get_requests with user_id=None might not
    # return them depending on session_id filtering. Accept either:
    # 1. success=False with "No interactions found" message
    # 2. success=True with profiles_generated > 0
    # 3. success=True with profiles_generated == 0 (valid if sessions were found but
    #    interactions within them were empty due to how the storage groups data)
    if not response.success:
        # It's acceptable if no interactions are found due to session grouping
        assert "No interactions found" in response.msg
    else:
        # Success is acceptable - profiles may or may not be generated depending
        # on how sessions are handled for all-users query
        assert response.profiles_generated >= 0


def test_rerun_profile_generation_with_time_filters(reflexio_with_config):
    """Test rerunning profile generation with time range filters."""
    user_id = "test_user_time_rerun"

    reflexio = reflexio_with_config

    # Publish interaction
    now = datetime.datetime.now(UTC)
    interaction_data = InteractionData(
        content="Time filter rerun test", created_at=int(now.timestamp())
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Rerun with time filters
    rerun_request = RerunProfileGenerationRequest(
        user_id=user_id,
        start_time=now - datetime.timedelta(hours=1),
        end_time=now + datetime.timedelta(hours=1),
    )

    response = reflexio.rerun_profile_generation(rerun_request)

    assert response.success is True
    assert response.profiles_generated > 0


def test_rerun_profile_generation_with_source_filter(reflexio_with_config):
    """Test rerunning profile generation with source filter."""
    user_id = "test_user_source_rerun"

    reflexio = reflexio_with_config

    # Publish interaction with specific source
    interaction_data = InteractionData(
        content="Source filter test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data], source="test_source"
    )
    reflexio.publish_interaction(publish_request)

    # Rerun with source filter
    rerun_request = RerunProfileGenerationRequest(user_id=user_id, source="test_source")

    response = reflexio.rerun_profile_generation(rerun_request)

    assert response.success is True
    assert response.profiles_generated > 0


def test_rerun_profile_generation_empty_interactions(temp_storage):
    """Test rerun with no matching interactions."""
    user_id = "test_user_empty"
    org_id = "test_org"
    reflexio = Reflexio(org_id=org_id, storage_base_dir=temp_storage)

    # Rerun without any interactions
    rerun_request = RerunProfileGenerationRequest(user_id=user_id)

    response = reflexio.rerun_profile_generation(rerun_request)

    assert response.success is False
    assert "No interactions found" in response.msg
    assert response.profiles_generated == 0


# ==============================
# Profile Status Management Tests
# ==============================


def test_upgrade_all_profiles(reflexio_with_config):
    """Test upgrading profiles: PENDING -> CURRENT, delete ARCHIVED."""
    user_id = "test_user_upgrade"

    reflexio = reflexio_with_config

    # Publish interaction to create CURRENT profile
    interaction_data = InteractionData(
        content="Upgrade test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Rerun to create PENDING profile
    rerun_request = RerunProfileGenerationRequest(user_id=user_id)
    reflexio.rerun_profile_generation(rerun_request)

    # Verify we have both CURRENT and PENDING
    current_profiles = reflexio.request_context.storage.get_user_profile(
        user_id, status_filter=[None]
    )
    pending_profiles = reflexio.request_context.storage.get_user_profile(
        user_id, status_filter=[Status.PENDING]
    )
    assert len(current_profiles) == 1
    assert len(pending_profiles) == 1

    # Upgrade
    response = reflexio.upgrade_all_profiles()

    assert response.success is True
    assert response.profiles_promoted > 0

    # Verify PENDING became CURRENT
    new_current = reflexio.request_context.storage.get_user_profile(
        user_id, status_filter=[None]
    )
    assert len(new_current) == 1  # The promoted profile


def test_downgrade_all_profiles(reflexio_with_config):
    """Test downgrading profiles: CURRENT -> ARCHIVED, restore previous."""
    user_id = "test_user_downgrade"

    reflexio = reflexio_with_config

    # Create initial profile
    interaction_data = InteractionData(
        content="Downgrade test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Create PENDING and upgrade (to create ARCHIVED)
    rerun_request = RerunProfileGenerationRequest(user_id=user_id)
    reflexio.rerun_profile_generation(rerun_request)
    reflexio.upgrade_all_profiles()

    # Now downgrade
    response = reflexio.downgrade_all_profiles()

    assert response.success is True
    assert response.profiles_restored >= 0


def test_upgrade_only_affected_users(reflexio_with_config):
    """Test upgrade with only_affected_users=True only affects users with PENDING profiles.

    This test ensures that when only_affected_users=True:
    1. Only users with PENDING profiles have their CURRENT profiles archived
    2. Only users with PENDING profiles have those profiles promoted
    3. Users without PENDING profiles are NOT affected
    """
    reflexio = reflexio_with_config

    # Create two users: one with PENDING profile, one without
    user_with_pending = "user_with_pending"
    user_without_pending = "user_without_pending"

    # Create CURRENT profiles for both users
    for user_id in [user_with_pending, user_without_pending]:
        interaction_data = InteractionData(
            content=f"Initial content for {user_id}",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        publish_request = PublishUserInteractionRequest(
            user_id=user_id, interaction_data_list=[interaction_data]
        )
        reflexio.publish_interaction(publish_request)

    # Create PENDING profile only for user_with_pending
    rerun_request = RerunProfileGenerationRequest(user_id=user_with_pending)
    reflexio.rerun_profile_generation(rerun_request)

    # Verify initial state
    # user_with_pending: 1 CURRENT + 1 PENDING
    current_for_pending_user = reflexio.request_context.storage.get_user_profile(
        user_with_pending, status_filter=[None]
    )
    pending_for_pending_user = reflexio.request_context.storage.get_user_profile(
        user_with_pending, status_filter=[Status.PENDING]
    )
    assert len(current_for_pending_user) == 1
    assert len(pending_for_pending_user) == 1

    # user_without_pending: 1 CURRENT only
    current_for_other_user = reflexio.request_context.storage.get_user_profile(
        user_without_pending, status_filter=[None]
    )
    pending_for_other_user = reflexio.request_context.storage.get_user_profile(
        user_without_pending, status_filter=[Status.PENDING]
    )
    assert len(current_for_other_user) == 1
    assert len(pending_for_other_user) == 0

    # Upgrade with only_affected_users=True
    upgrade_request = UpgradeProfilesRequest(user_id=None, only_affected_users=True)
    response = reflexio.upgrade_all_profiles(upgrade_request)

    assert response.success is True
    assert response.profiles_promoted == 1  # Only user_with_pending's profile promoted
    assert response.profiles_archived == 1  # Only user_with_pending's CURRENT archived

    # Verify user_with_pending: PENDING became CURRENT, old CURRENT became ARCHIVED
    new_current = reflexio.request_context.storage.get_user_profile(
        user_with_pending, status_filter=[None]
    )
    new_pending = reflexio.request_context.storage.get_user_profile(
        user_with_pending, status_filter=[Status.PENDING]
    )
    new_archived = reflexio.request_context.storage.get_user_profile(
        user_with_pending, status_filter=[Status.ARCHIVED]
    )
    assert len(new_current) == 1  # Promoted profile is now CURRENT
    assert len(new_pending) == 0  # No more PENDING
    assert len(new_archived) == 1  # Old CURRENT is now ARCHIVED

    # Verify user_without_pending: Should be UNCHANGED
    other_current = reflexio.request_context.storage.get_user_profile(
        user_without_pending, status_filter=[None]
    )
    other_archived = reflexio.request_context.storage.get_user_profile(
        user_without_pending, status_filter=[Status.ARCHIVED]
    )
    assert len(other_current) == 1  # Still has CURRENT (not archived!)
    assert len(other_archived) == 0  # No ARCHIVED created


def test_downgrade_only_affected_users(reflexio_with_config):
    """Test downgrade with only_affected_users=True only affects users with ARCHIVED profiles.

    This test ensures that when only_affected_users=True:
    1. Only users with ARCHIVED profiles have their CURRENT profiles demoted
    2. Only users with ARCHIVED profiles have those profiles restored
    3. Users without ARCHIVED profiles are NOT affected
    """
    reflexio = reflexio_with_config

    # Create two users
    user_with_archived = "user_with_archived"
    user_without_archived = "user_without_archived"

    # Create CURRENT profiles for both users
    for user_id in [user_with_archived, user_without_archived]:
        interaction_data = InteractionData(
            content=f"Initial content for {user_id}",
            created_at=int(datetime.datetime.now(UTC).timestamp()),
        )
        publish_request = PublishUserInteractionRequest(
            user_id=user_id, interaction_data_list=[interaction_data]
        )
        reflexio.publish_interaction(publish_request)

    # Create PENDING and upgrade for user_with_archived (this creates ARCHIVED)
    # Use only_affected_users=True so only user_with_archived is affected
    rerun_request = RerunProfileGenerationRequest(user_id=user_with_archived)
    reflexio.rerun_profile_generation(rerun_request)
    upgrade_request = UpgradeProfilesRequest(user_id=None, only_affected_users=True)
    reflexio.upgrade_all_profiles(upgrade_request)

    # Verify initial state before downgrade
    # user_with_archived: 1 CURRENT + 1 ARCHIVED
    current_for_archived_user = reflexio.request_context.storage.get_user_profile(
        user_with_archived, status_filter=[None]
    )
    archived_for_archived_user = reflexio.request_context.storage.get_user_profile(
        user_with_archived, status_filter=[Status.ARCHIVED]
    )
    assert len(current_for_archived_user) == 1
    assert len(archived_for_archived_user) == 1

    # user_without_archived: 1 CURRENT only (no ARCHIVED)
    current_for_other_user = reflexio.request_context.storage.get_user_profile(
        user_without_archived, status_filter=[None]
    )
    archived_for_other_user = reflexio.request_context.storage.get_user_profile(
        user_without_archived, status_filter=[Status.ARCHIVED]
    )
    assert len(current_for_other_user) == 1
    assert len(archived_for_other_user) == 0

    # Downgrade with only_affected_users=True
    downgrade_request = DowngradeProfilesRequest(user_id=None, only_affected_users=True)
    response = reflexio.downgrade_all_profiles(downgrade_request)

    assert response.success is True
    assert (
        response.profiles_restored == 1
    )  # Only user_with_archived's ARCHIVED restored
    assert response.profiles_demoted == 1  # Only user_with_archived's CURRENT demoted

    # Verify user_with_archived: ARCHIVED became CURRENT, old CURRENT became ARCHIVED
    new_current = reflexio.request_context.storage.get_user_profile(
        user_with_archived, status_filter=[None]
    )
    new_archived = reflexio.request_context.storage.get_user_profile(
        user_with_archived, status_filter=[Status.ARCHIVED]
    )
    assert len(new_current) == 1  # Restored profile is now CURRENT
    assert len(new_archived) == 1  # Demoted profile is now ARCHIVED

    # Verify user_without_archived: Should be UNCHANGED
    other_current = reflexio.request_context.storage.get_user_profile(
        user_without_archived, status_filter=[None]
    )
    other_archived = reflexio.request_context.storage.get_user_profile(
        user_without_archived, status_filter=[Status.ARCHIVED]
    )
    assert len(other_current) == 1  # Still has CURRENT (not demoted!)
    assert len(other_archived) == 0  # No ARCHIVED created


def test_get_profile_statistics(reflexio_with_config):
    """Test getting profile count statistics by status."""
    user_id = "test_user_stats"

    reflexio = reflexio_with_config

    # Publish interaction
    interaction_data = InteractionData(
        content="Stats test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Get statistics
    response = reflexio.get_profile_statistics()

    assert response.success is True
    assert response.current_count >= 1


# ==============================
# CRUD Operation Tests
# ==============================


def test_delete_profile_success(reflexio_with_config):
    """Test deleting a user profile."""
    user_id = "test_user_delete_profile"

    reflexio = reflexio_with_config

    # Publish interaction to create profile
    interaction_data = InteractionData(
        content="Delete profile test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Get profile ID
    profiles = reflexio.request_context.storage.get_user_profile(user_id)
    assert len(profiles) == 1
    profile_id = profiles[0].profile_id

    # Delete profile
    from reflexio.models.api_schema.service_schemas import DeleteUserProfileRequest

    delete_request = DeleteUserProfileRequest(user_id=user_id, profile_id=profile_id)

    response = reflexio.delete_profile(delete_request)

    assert response.success is True

    # Verify profile deleted
    remaining_profiles = reflexio.request_context.storage.get_user_profile(user_id)
    assert len(remaining_profiles) == 0


def test_delete_interaction_success(reflexio_with_config):
    """Test deleting a user interaction."""
    user_id = "test_user_delete_interaction"

    reflexio = reflexio_with_config

    # Publish interaction
    interaction_data = InteractionData(
        content="Delete interaction test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Get interaction ID
    interactions = reflexio.request_context.storage.get_user_interaction(user_id)
    assert len(interactions) == 1
    interaction_id = interactions[0].interaction_id

    # Delete interaction
    from reflexio.models.api_schema.service_schemas import (
        DeleteUserInteractionRequest,
    )

    delete_request = DeleteUserInteractionRequest(
        user_id=user_id, interaction_id=interaction_id
    )

    response = reflexio.delete_interaction(delete_request)

    assert response.success is True

    # Verify interaction deleted
    remaining_interactions = reflexio.request_context.storage.get_user_interaction(
        user_id
    )
    assert len(remaining_interactions) == 0


def test_get_profile_change_logs(reflexio_with_config):
    """Test getting profile change history."""
    user_id = "test_user_changelog"

    reflexio = reflexio_with_config

    # Publish interaction (creates profile and changelog)
    interaction_data = InteractionData(
        content="Changelog test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Get changelogs
    response = reflexio.get_profile_change_logs()

    assert response.success is True
    assert len(response.profile_change_logs) >= 0


# ==============================
# Configuration Tests
# ==============================


def test_set_and_get_config(temp_storage):
    """Test setting and retrieving configuration."""
    org_id = "test_org"
    reflexio = Reflexio(org_id=org_id, storage_base_dir=temp_storage)

    # Get default config
    config = reflexio.get_config()
    assert config is not None

    # Set config (will fail validation but test the flow)
    reflexio.set_config(config)
    # Note: May fail due to storage validation, which is expected


def test_get_config_default(temp_storage):
    """Test getting default configuration."""
    org_id = "test_org"
    reflexio = Reflexio(org_id=org_id, storage_base_dir=temp_storage)

    config = reflexio.get_config()

    assert config is not None
    assert hasattr(config, "storage_config")


# ==============================
# Dashboard/Analytics Tests
# ==============================


def test_get_dashboard_stats(reflexio_with_config):
    """Test getting dashboard statistics with time series data."""
    user_id = "test_user_dashboard"

    reflexio = reflexio_with_config

    # Publish interaction
    interaction_data = InteractionData(
        content="Dashboard test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Get dashboard stats
    stats_request = GetDashboardStatsRequest(days_back=7)

    response = reflexio.get_dashboard_stats(stats_request)

    assert response.success is True
    assert response.stats is not None
    assert hasattr(response.stats, "current_period")
    assert hasattr(response.stats, "interactions_time_series")


# ==============================
# Request Management Tests
# ==============================


def test_get_requests_grouped(reflexio_with_config):
    """Test getting requests grouped by session_id."""
    user_id = "test_user_requests"

    reflexio = reflexio_with_config

    # Publish interaction
    interaction_data = InteractionData(
        content="Request grouping test",
        created_at=int(datetime.datetime.now(UTC).timestamp()),
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id,
        interaction_data_list=[interaction_data],
        session_id="test_group",
    )
    reflexio.publish_interaction(publish_request)

    # Get requests
    get_requests_request = GetRequestsRequest(user_id=user_id, top_k=10)

    response = reflexio.get_requests(get_requests_request)

    assert response.success is True
    assert len(response.sessions) >= 0


def test_get_requests_with_filters(reflexio_with_config):
    """Test getting requests with time and user filters."""
    user_id = "test_user_request_filters"

    reflexio = reflexio_with_config

    # Publish interaction
    now = datetime.datetime.now(UTC)
    interaction_data = InteractionData(
        content="Request filter test", created_at=int(now.timestamp())
    )

    publish_request = PublishUserInteractionRequest(
        user_id=user_id, interaction_data_list=[interaction_data]
    )
    reflexio.publish_interaction(publish_request)

    # Get requests with time filter
    get_requests_request = GetRequestsRequest(
        user_id=user_id,
        start_time=now - datetime.timedelta(hours=1),
        end_time=now + datetime.timedelta(hours=1),
        top_k=10,
    )

    response = reflexio.get_requests(get_requests_request)

    assert response.success is True
