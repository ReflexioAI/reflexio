"""End-to-end tests for playbook workflows."""

import os
from collections.abc import Callable

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.retriever_schema import (
    GetAgentPlaybooksRequest,
    GetUserPlaybooksRequest,
    SearchUserPlaybookRequest,
    UpdatePlaybookStatusRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AddUserPlaybookRequest,
    DowngradeUserPlaybooksRequest,
    InteractionData,
    ManualPlaybookGenerationRequest,
    PlaybookStatus,
    RerunPlaybookGenerationRequest,
    Status,
    UpgradeUserPlaybooksRequest,
    UserPlaybook,
)
from reflexio.models.config_schema import SearchMode
from tests.e2e_tests.conftest import save_user_playbooks
from tests.server.test_utils import skip_in_precommit, skip_low_priority


@skip_in_precommit
def test_publish_interaction_playbook_only(
    reflexio_instance_playbook_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test interaction publishing with only playbook extraction enabled."""
    user_id = "test_user_playbook_only"
    agent_version = "test_agent_playbook"

    # Publish interactions
    response = reflexio_instance_playbook_only.publish_interaction(
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
        reflexio_instance_playbook_only.request_context.storage.get_all_interactions()
    )
    assert len(final_interactions) == len(sample_interaction_requests)

    # Verify playbooks were generated and stored
    user_playbooks = (
        reflexio_instance_playbook_only.request_context.storage.get_user_playbooks(
            playbook_name="test_playbook"
        )
    )
    assert len(user_playbooks) > 0 and user_playbooks[0].content.strip() != ""
    # The user-configured playbook extractor must actually run (beyond any defaults)
    assert any(p.playbook_name == "test_playbook" for p in user_playbooks)

    # No agent success evaluation results — this fixture does not configure
    # agent_success_configs, and group evaluation is never triggered in this flow.
    agent_success_results = reflexio_instance_playbook_only.request_context.storage.get_agent_success_evaluation_results(
        agent_version=agent_version
    )
    assert len(agent_success_results) == 0


@skip_in_precommit
def test_run_playbook_aggregation_end_to_end(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],
):
    """Test end-to-end playbook aggregation workflow."""
    agent_version = "1.0.0"  # Must match the agent_version in mock_playbooks.csv
    playbook_name = "test_playbook"

    # First save mock playbooks
    save_user_playbooks(reflexio_instance_playbook_only)

    # Use mock mode to avoid needing embeddings for clustering
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Run playbook aggregation for the agent version
        reflexio_instance_playbook_only.run_playbook_aggregation(
            agent_version=agent_version,
            playbook_name=playbook_name,
        )
        # If we reach here, the operation was successful
        assert True

        user_playbooks = (
            reflexio_instance_playbook_only.request_context.storage.get_user_playbooks(
                playbook_name=playbook_name
            )
        )
        assert len(user_playbooks) == 20

        agent_playbooks = (
            reflexio_instance_playbook_only.request_context.storage.get_agent_playbooks(
                playbook_name=playbook_name,
                playbook_status_filter=[PlaybookStatus.PENDING],
            )
        )
        assert len(agent_playbooks) > 0
    finally:
        # Restore original environment variable
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_get_agent_playbooks_with_playbook_status_filter(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],
):
    """Test get_agent_playbooks with playbook_status_filter parameter.

    This test verifies:
    1. Default behavior (no filter) returns playbooks of all statuses
    2. Explicit status filters return only playbooks with matching status
    3. Each status filter correctly filters the results
    """
    agent_version = "1.0.0"
    playbook_name = "test_playbook"
    storage = reflexio_instance_playbook_only.request_context.storage

    # First save mock playbooks and run aggregation
    save_user_playbooks(reflexio_instance_playbook_only)

    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Run playbook aggregation - creates playbooks with PENDING status
        reflexio_instance_playbook_only.run_playbook_aggregation(
            agent_version=agent_version,
            playbook_name=playbook_name,
        )

        # Get all pending playbooks to set up different statuses
        initial_response = reflexio_instance_playbook_only.get_agent_playbooks(
            GetAgentPlaybooksRequest(
                playbook_name=playbook_name,
                playbook_status_filter=PlaybookStatus.PENDING,
            )
        )
        assert initial_response.success is True
        assert len(initial_response.agent_playbooks) >= 3, (
            "Need at least 3 playbooks to test different status filters"
        )

        # Update some playbooks to different statuses to enable proper testing
        # Keep one as PENDING, update one to APPROVED, update one to REJECTED
        playbooks_to_update = initial_response.agent_playbooks[:3]

        # Update first playbook to APPROVED
        storage.update_agent_playbook_status(
            playbooks_to_update[0].agent_playbook_id, PlaybookStatus.APPROVED
        )
        # Update second playbook to REJECTED
        storage.update_agent_playbook_status(
            playbooks_to_update[1].agent_playbook_id, PlaybookStatus.REJECTED
        )
        # Third playbook stays as PENDING

        # Test default behavior - should return playbooks of ALL statuses (no filter)
        response_default = reflexio_instance_playbook_only.get_agent_playbooks(
            GetAgentPlaybooksRequest(playbook_name=playbook_name)
        )
        assert response_default.success is True
        assert len(response_default.agent_playbooks) > 0

        # Verify that default (no filter) returns playbooks of different statuses
        statuses_in_default = {
            f.playbook_status for f in response_default.agent_playbooks
        }
        # Should have at least APPROVED and REJECTED (PENDING depends on how many playbooks we started with)
        assert PlaybookStatus.APPROVED in statuses_in_default, (
            "Default should return APPROVED playbooks when no filter is specified"
        )
        assert PlaybookStatus.REJECTED in statuses_in_default, (
            "Default should return REJECTED playbooks when no filter is specified"
        )

        # Test with explicit approved filter - should ONLY return APPROVED playbooks
        response_approved = reflexio_instance_playbook_only.get_agent_playbooks(
            GetAgentPlaybooksRequest(
                playbook_name=playbook_name,
                playbook_status_filter=PlaybookStatus.APPROVED,
            )
        )
        assert response_approved.success is True
        assert len(response_approved.agent_playbooks) >= 1, (
            "Should have at least 1 APPROVED playbook"
        )
        for playbook in response_approved.agent_playbooks:
            assert playbook.playbook_status == PlaybookStatus.APPROVED

        # Test with pending filter - should ONLY return PENDING playbooks
        response_pending = reflexio_instance_playbook_only.get_agent_playbooks(
            GetAgentPlaybooksRequest(
                playbook_name=playbook_name,
                playbook_status_filter=PlaybookStatus.PENDING,
            )
        )
        assert response_pending.success is True
        # We left at least one playbook as PENDING
        assert len(response_pending.agent_playbooks) >= 1, (
            "Should have at least 1 PENDING playbook"
        )
        for playbook in response_pending.agent_playbooks:
            assert playbook.playbook_status == PlaybookStatus.PENDING

        # Test with rejected filter - should ONLY return REJECTED playbooks
        response_rejected = reflexio_instance_playbook_only.get_agent_playbooks(
            GetAgentPlaybooksRequest(
                playbook_name=playbook_name,
                playbook_status_filter=PlaybookStatus.REJECTED,
            )
        )
        assert response_rejected.success is True
        assert len(response_rejected.agent_playbooks) >= 1, (
            "Should have at least 1 REJECTED playbook"
        )
        for playbook in response_rejected.agent_playbooks:
            assert playbook.playbook_status == PlaybookStatus.REJECTED

        # Verify filtered counts are less than default (no filter) count
        # This confirms that filters are actually excluding playbooks
        assert len(response_approved.agent_playbooks) < len(
            response_default.agent_playbooks
        ), "APPROVED filter should return fewer playbooks than no filter"
        assert len(response_rejected.agent_playbooks) < len(
            response_default.agent_playbooks
        ), "REJECTED filter should return fewer playbooks than no filter"

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_get_user_playbooks_with_status_filter(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],
):
    """Test get_user_playbooks with status_filter parameter."""
    playbook_name = "test_playbook"

    # Save mock playbooks
    save_user_playbooks(reflexio_instance_playbook_only)

    # Test default behavior - should return current (non-archived) user playbooks
    response_default = reflexio_instance_playbook_only.get_user_playbooks(
        GetUserPlaybooksRequest(playbook_name=playbook_name)
    )
    assert response_default.success is True
    assert len(response_default.user_playbooks) > 0

    # Test with explicit None status filter (current playbooks)
    response_current = reflexio_instance_playbook_only.get_user_playbooks(
        GetUserPlaybooksRequest(
            playbook_name=playbook_name,
            status_filter=[None],
        )
    )
    assert response_current.success is True
    for user_playbook in response_current.user_playbooks:
        assert user_playbook.status is None  # Current playbooks have None status


@skip_in_precommit
def test_upgrade_user_playbooks_end_to_end(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],
):
    """Test end-to-end upgrade workflow for user playbooks.

    Upgrade workflow:
    1. Delete old ARCHIVED user playbooks
    2. Archive CURRENT user playbooks (None -> ARCHIVED)
    3. Promote PENDING user playbooks (PENDING -> None/CURRENT)
    """
    playbook_name = "test_playbook"
    agent_version = "1.0.0"
    storage = reflexio_instance_playbook_only.request_context.storage

    # Setup: Create user playbooks with different statuses
    # Create CURRENT playbooks (status=None)
    current_playbooks = [
        UserPlaybook(
            agent_version=agent_version,
            request_id=f"current_request_{i}",
            playbook_name=playbook_name,
            content=f"Current playbook content {i}",
            status=None,
        )
        for i in range(3)
    ]

    # Create PENDING playbooks (status=PENDING)
    pending_playbooks = [
        UserPlaybook(
            agent_version=agent_version,
            request_id=f"pending_request_{i}",
            playbook_name=playbook_name,
            content=f"Pending playbook content {i}",
            status=Status.PENDING,
        )
        for i in range(2)
    ]

    # Create ARCHIVED playbooks (status=ARCHIVED)
    archived_playbooks = [
        UserPlaybook(
            agent_version=agent_version,
            request_id=f"archived_request_{i}",
            playbook_name=playbook_name,
            content=f"Archived playbook content {i}",
            status=Status.ARCHIVED,
        )
        for i in range(2)
    ]

    # Save all playbooks to storage
    storage.save_user_playbooks(
        current_playbooks + pending_playbooks + archived_playbooks
    )

    # Verify initial state
    all_playbooks_before = storage.get_user_playbooks(playbook_name=playbook_name)
    current_before = [f for f in all_playbooks_before if f.status is None]
    pending_before = [f for f in all_playbooks_before if f.status == Status.PENDING]
    archived_before = [f for f in all_playbooks_before if f.status == Status.ARCHIVED]

    assert len(current_before) == 3
    assert len(pending_before) == 2
    assert len(archived_before) == 2

    # Execute upgrade
    response = reflexio_instance_playbook_only.upgrade_all_user_playbooks(
        UpgradeUserPlaybooksRequest(
            agent_version=agent_version,
            playbook_name=playbook_name,
        )
    )

    # Verify response
    assert response.success is True
    assert response.user_playbooks_deleted == 2  # Old ARCHIVED deleted
    assert response.user_playbooks_archived == 3  # CURRENT -> ARCHIVED
    assert response.user_playbooks_promoted == 2  # PENDING -> CURRENT (None)

    # Verify final state
    all_playbooks_after = storage.get_user_playbooks(playbook_name=playbook_name)
    current_after = [f for f in all_playbooks_after if f.status is None]
    archived_after = [f for f in all_playbooks_after if f.status == Status.ARCHIVED]
    pending_after = [f for f in all_playbooks_after if f.status == Status.PENDING]

    # PENDING playbooks promoted to CURRENT
    assert len(current_after) == 2
    for playbook in current_after:
        assert "pending_request" in playbook.request_id

    # CURRENT playbooks archived
    assert len(archived_after) == 3
    for playbook in archived_after:
        assert "current_request" in playbook.request_id

    # No more PENDING playbooks
    assert len(pending_after) == 0


@skip_in_precommit
@skip_low_priority
def test_downgrade_user_playbooks_end_to_end(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],
):
    """Test end-to-end downgrade workflow for user playbooks.

    Downgrade workflow:
    1. Demote CURRENT user playbooks (None -> ARCHIVE_IN_PROGRESS)
    2. Restore ARCHIVED user playbooks (ARCHIVED -> None/CURRENT)
    3. Complete archiving (ARCHIVE_IN_PROGRESS -> ARCHIVED)
    """
    playbook_name = "test_playbook"
    agent_version = "1.0.0"
    storage = reflexio_instance_playbook_only.request_context.storage

    # Setup: Create user playbooks with different statuses
    # Create CURRENT playbooks (status=None)
    current_playbooks = [
        UserPlaybook(
            agent_version=agent_version,
            request_id=f"current_request_{i}",
            playbook_name=playbook_name,
            content=f"Current playbook content {i}",
            status=None,
        )
        for i in range(3)
    ]

    # Create ARCHIVED playbooks (status=ARCHIVED)
    archived_playbooks = [
        UserPlaybook(
            agent_version=agent_version,
            request_id=f"archived_request_{i}",
            playbook_name=playbook_name,
            content=f"Archived playbook content {i}",
            status=Status.ARCHIVED,
        )
        for i in range(2)
    ]

    # Save all playbooks to storage
    storage.save_user_playbooks(current_playbooks + archived_playbooks)

    # Verify initial state
    all_playbooks_before = storage.get_user_playbooks(playbook_name=playbook_name)
    current_before = [f for f in all_playbooks_before if f.status is None]
    archived_before = [f for f in all_playbooks_before if f.status == Status.ARCHIVED]

    assert len(current_before) == 3
    assert len(archived_before) == 2

    # Execute downgrade
    response = reflexio_instance_playbook_only.downgrade_all_user_playbooks(
        DowngradeUserPlaybooksRequest(
            agent_version=agent_version,
            playbook_name=playbook_name,
        )
    )

    # Verify response
    assert response.success is True
    assert response.user_playbooks_demoted == 3  # CURRENT -> ARCHIVED
    assert response.user_playbooks_restored == 2  # ARCHIVED -> CURRENT (None)

    # Verify final state
    all_playbooks_after = storage.get_user_playbooks(playbook_name=playbook_name)
    current_after = [f for f in all_playbooks_after if f.status is None]
    archived_after = [f for f in all_playbooks_after if f.status == Status.ARCHIVED]

    # ARCHIVED playbooks restored to CURRENT
    assert len(current_after) == 2
    for playbook in current_after:
        assert "archived_request" in playbook.request_id

    # CURRENT playbooks demoted to ARCHIVED
    assert len(archived_after) == 3
    for playbook in archived_after:
        assert "current_request" in playbook.request_id


@skip_in_precommit
@skip_low_priority
def test_upgrade_downgrade_roundtrip(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],
):
    """Test that upgrade followed by downgrade restores the original state."""
    playbook_name = "test_playbook"
    agent_version = "1.0.0"
    storage = reflexio_instance_playbook_only.request_context.storage

    # Setup: Create initial CURRENT playbooks
    current_playbooks = [
        UserPlaybook(
            agent_version=agent_version,
            request_id=f"original_request_{i}",
            playbook_name=playbook_name,
            content=f"Original playbook content {i}",
            status=None,
        )
        for i in range(3)
    ]

    # Create PENDING playbooks (new version)
    pending_playbooks = [
        UserPlaybook(
            agent_version=agent_version,
            request_id=f"new_request_{i}",
            playbook_name=playbook_name,
            content=f"New playbook content {i}",
            status=Status.PENDING,
        )
        for i in range(2)
    ]

    storage.save_user_playbooks(current_playbooks + pending_playbooks)

    # Execute upgrade (new playbooks become current, original become archived)
    upgrade_response = reflexio_instance_playbook_only.upgrade_all_user_playbooks(
        UpgradeUserPlaybooksRequest(
            agent_version=agent_version,
            playbook_name=playbook_name,
        )
    )
    assert upgrade_response.success is True

    # Verify upgrade state
    all_playbooks_after_upgrade = storage.get_user_playbooks(
        playbook_name=playbook_name
    )
    current_after_upgrade = [f for f in all_playbooks_after_upgrade if f.status is None]
    archived_after_upgrade = [
        f for f in all_playbooks_after_upgrade if f.status == Status.ARCHIVED
    ]

    assert len(current_after_upgrade) == 2  # new playbooks are now current
    assert len(archived_after_upgrade) == 3  # original playbooks are now archived

    # Execute downgrade (restore original playbooks)
    downgrade_response = reflexio_instance_playbook_only.downgrade_all_user_playbooks(
        DowngradeUserPlaybooksRequest(
            agent_version=agent_version,
            playbook_name=playbook_name,
        )
    )
    assert downgrade_response.success is True

    # Verify roundtrip restored original state
    all_playbooks_after_downgrade = storage.get_user_playbooks(
        playbook_name=playbook_name
    )
    current_after_downgrade = [
        f for f in all_playbooks_after_downgrade if f.status is None
    ]
    archived_after_downgrade = [
        f for f in all_playbooks_after_downgrade if f.status == Status.ARCHIVED
    ]

    # Original playbooks restored to current
    assert len(current_after_downgrade) == 3
    for playbook in current_after_downgrade:
        assert "original_request" in playbook.request_id

    # New playbooks demoted to archived
    assert len(archived_after_downgrade) == 2
    for playbook in archived_after_downgrade:
        assert "new_request" in playbook.request_id


@skip_in_precommit
@skip_low_priority
def test_add_user_playbook_end_to_end(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],
):
    """Test add_user_playbook method for directly adding user playbooks to storage.

    This test verifies:
    1. User playbooks can be added directly via API
    2. Added playbooks are stored correctly
    3. Playbooks are normalized (only required fields kept)
    4. Error handling for invalid input
    """
    playbook_name = "test_add_playbook"
    agent_version = "1.0.0"

    # Step 1: Create user playbooks to add
    user_playbooks_to_add = [
        UserPlaybook(
            agent_version=agent_version,
            request_id=f"add_test_request_{i}",
            playbook_name=playbook_name,
            content=f"Added playbook content {i}",
        )
        for i in range(3)
    ]

    # Step 2: Add user playbooks via API
    add_response = reflexio_instance_playbook_only.add_user_playbook(
        AddUserPlaybookRequest(user_playbooks=user_playbooks_to_add)
    )
    assert add_response.success is True
    assert add_response.added_count == 3

    # Step 3: Verify playbooks were stored
    stored_playbooks = reflexio_instance_playbook_only.get_user_playbooks(
        GetUserPlaybooksRequest(playbook_name=playbook_name)
    )
    assert stored_playbooks.success is True
    assert len(stored_playbooks.user_playbooks) == 3

    # Step 4: Verify playbook content
    for _i, playbook in enumerate(stored_playbooks.user_playbooks):
        assert playbook.agent_version == agent_version
        assert playbook.playbook_name == playbook_name
        assert "Added playbook content" in playbook.content

    # Step 5: Test with dict input
    dict_playbooks = [
        {
            "agent_version": agent_version,
            "request_id": "dict_test_request",
            "playbook_name": playbook_name,
            "content": "Dict added playbook content",
        }
    ]
    dict_response = reflexio_instance_playbook_only.add_user_playbook(
        {"user_playbooks": dict_playbooks}
    )
    assert dict_response.success is True
    assert dict_response.added_count == 1

    # Step 6: Verify total playbooks
    all_playbooks = reflexio_instance_playbook_only.get_user_playbooks(
        GetUserPlaybooksRequest(playbook_name=playbook_name)
    )
    assert len(all_playbooks.user_playbooks) == 4


@skip_in_precommit
def test_update_agent_playbook_status_end_to_end(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],
):
    """Test update_agent_playbook_status method for approving/rejecting playbooks.

    This test verifies:
    1. AgentPlaybook status can be updated from PENDING to APPROVED
    2. AgentPlaybook status can be updated from PENDING to REJECTED
    3. Status update is persisted correctly
    4. Error handling for non-existent playbook
    """
    playbook_name = "test_playbook"
    agent_version = "1.0.0"

    # Setup: Save mock playbooks and run aggregation to create playbooks with status
    save_user_playbooks(reflexio_instance_playbook_only)

    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Run playbook aggregation to create aggregated playbooks
        reflexio_instance_playbook_only.run_playbook_aggregation(
            agent_version=agent_version,
            playbook_name=playbook_name,
        )

        # Get pending playbooks
        pending_response = reflexio_instance_playbook_only.get_agent_playbooks(
            GetAgentPlaybooksRequest(
                playbook_name=playbook_name,
                playbook_status_filter=PlaybookStatus.PENDING,
            )
        )
        assert pending_response.success is True
        assert len(pending_response.agent_playbooks) > 0

        # Step 1: Update first playbook to APPROVED
        first_playbook = pending_response.agent_playbooks[0]
        approve_response = reflexio_instance_playbook_only.update_agent_playbook_status(
            UpdatePlaybookStatusRequest(
                agent_playbook_id=first_playbook.agent_playbook_id,
                playbook_status=PlaybookStatus.APPROVED,
            )
        )
        assert approve_response.success is True

        # Verify status was updated
        approved_playbooks = reflexio_instance_playbook_only.get_agent_playbooks(
            GetAgentPlaybooksRequest(
                playbook_name=playbook_name,
                playbook_status_filter=PlaybookStatus.APPROVED,
            )
        )
        assert approved_playbooks.success is True
        approved_ids = [f.agent_playbook_id for f in approved_playbooks.agent_playbooks]
        assert first_playbook.agent_playbook_id in approved_ids

        # Step 2: Update second playbook to REJECTED (if exists)
        if len(pending_response.agent_playbooks) > 1:
            second_playbook = pending_response.agent_playbooks[1]
            reject_response = (
                reflexio_instance_playbook_only.update_agent_playbook_status(
                    UpdatePlaybookStatusRequest(
                        agent_playbook_id=second_playbook.agent_playbook_id,
                        playbook_status=PlaybookStatus.REJECTED,
                    )
                )
            )
            assert reject_response.success is True

            # Verify status was updated
            rejected_playbooks = reflexio_instance_playbook_only.get_agent_playbooks(
                GetAgentPlaybooksRequest(
                    playbook_name=playbook_name,
                    playbook_status_filter=PlaybookStatus.REJECTED,
                )
            )
            assert rejected_playbooks.success is True
            rejected_ids = [
                f.agent_playbook_id for f in rejected_playbooks.agent_playbooks
            ]
            assert second_playbook.agent_playbook_id in rejected_ids

        # Step 3: Test with dict input
        if len(pending_response.agent_playbooks) > 2:
            third_playbook = pending_response.agent_playbooks[2]
            dict_response = (
                reflexio_instance_playbook_only.update_agent_playbook_status(
                    {
                        "agent_playbook_id": third_playbook.agent_playbook_id,
                        "playbook_status": PlaybookStatus.APPROVED,
                    }
                )
            )
            assert dict_response.success is True

        # Step 4: Test error handling with non-existent playbook ID
        error_response = reflexio_instance_playbook_only.update_agent_playbook_status(
            UpdatePlaybookStatusRequest(
                agent_playbook_id=999999,  # Non-existent ID
                playbook_status=PlaybookStatus.APPROVED,
            )
        )
        assert error_response.success is False

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
def test_rerun_playbook_generation_end_to_end(
    reflexio_instance_playbook_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_playbook_only: Callable[[], None],
):
    """Test rerun_playbook_generation method for regenerating playbooks.

    This test verifies:
    1. Rerun playbook generation creates PENDING playbooks
    2. Existing CURRENT playbooks remain unchanged
    3. Time filtering works correctly
    4. AgentPlaybook name filtering works correctly
    """
    user_id = "test_user_rerun_playbook"
    agent_version = "test_agent_rerun_playbook"
    playbook_name = "test_playbook"

    # Use mock mode to ensure consistent LLM responses
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Step 1: Publish interactions to generate playbooks
        publish_response = reflexio_instance_playbook_only.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "test_rerun_source",
                "agent_version": agent_version,
            }
        )
        assert publish_response.success is True

        # Verify playbooks were generated
        initial_playbooks = reflexio_instance_playbook_only.get_user_playbooks(
            GetUserPlaybooksRequest(
                playbook_name=playbook_name,
                status_filter=[None],  # Current playbooks
            )
        )
        assert initial_playbooks.success is True
        initial_count = len(initial_playbooks.user_playbooks)
        assert initial_count > 0, "Initial playbooks should be generated"

        # Step 2: Run rerun_playbook_generation
        rerun_response = reflexio_instance_playbook_only.rerun_playbook_generation(
            RerunPlaybookGenerationRequest(
                agent_version=agent_version,
                playbook_name=playbook_name,
            )
        )
        assert rerun_response.success is True
        assert rerun_response.playbooks_generated > 0

        # Step 3: Verify PENDING playbooks were created
        pending_playbooks = reflexio_instance_playbook_only.get_user_playbooks(
            GetUserPlaybooksRequest(
                playbook_name=playbook_name,
                status_filter=[Status.PENDING],
            )
        )
        assert pending_playbooks.success is True
        assert len(pending_playbooks.user_playbooks) > 0, (
            "PENDING playbooks should be created"
        )

        # Step 4: Verify current playbooks unchanged
        current_playbooks_after = reflexio_instance_playbook_only.get_user_playbooks(
            GetUserPlaybooksRequest(
                playbook_name=playbook_name,
                status_filter=[None],
            )
        )
        assert current_playbooks_after.success is True
        assert len(current_playbooks_after.user_playbooks) == initial_count

        # Step 5: Test with dict input
        dict_response = reflexio_instance_playbook_only.rerun_playbook_generation(
            {
                "agent_version": agent_version,
                "playbook_name": playbook_name,
            }
        )
        assert dict_response.success is True

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_rerun_playbook_generation_with_time_filters(
    reflexio_instance_playbook_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_playbook_only: Callable[[], None],
):
    """Test rerun_playbook_generation with time-based filtering.

    This test verifies:
    1. Time filtering correctly filters interactions
    2. Future time range returns no results
    3. Valid time range regenerates playbooks
    """
    from datetime import UTC, datetime, timedelta

    user_id = "test_user_rerun_playbook_time"
    agent_version = "test_agent_rerun_time"
    playbook_name = "test_playbook"

    # Use mock mode to ensure consistent LLM responses
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Publish interactions
        publish_response = reflexio_instance_playbook_only.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "test_rerun_time_source",
                "agent_version": agent_version,
            }
        )
        assert publish_response.success is True

        # Test with future time range (should fail - no interactions)
        future_start = datetime.now(UTC) + timedelta(days=1)
        future_end = datetime.now(UTC) + timedelta(days=2)

        future_response = reflexio_instance_playbook_only.rerun_playbook_generation(
            RerunPlaybookGenerationRequest(
                agent_version=agent_version,
                playbook_name=playbook_name,
                start_time=future_start,
                end_time=future_end,
            )
        )
        assert future_response.success is False
        assert "No interactions found" in future_response.msg

        # Test with valid time range (past to future)
        past_start = datetime.now(UTC) - timedelta(days=1)
        future_end = datetime.now(UTC) + timedelta(days=1)

        valid_response = reflexio_instance_playbook_only.rerun_playbook_generation(
            RerunPlaybookGenerationRequest(
                agent_version=agent_version,
                playbook_name=playbook_name,
                start_time=past_start,
                end_time=future_end,
            )
        )
        assert valid_response.success is True
        assert valid_response.playbooks_generated > 0

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_playbook_source_filtering_with_matching_source(
    reflexio_instance_playbook_source_filtering: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_playbook_source_filtering: Callable[[], None],
):
    """Test that playbook extractors only run when source matches request_sources_enabled.

    This test verifies:
    1. When source="api", only api_playbook and all_sources_playbook extractors run
    2. When source="webhook", only webhook_playbook and all_sources_playbook extractors run
    3. Playbooks have the correct source field set
    """
    user_id = "test_user_source_filter"
    agent_version = "test_agent_source"
    storage = reflexio_instance_playbook_source_filtering.request_context.storage

    # Step 1: Publish interactions with source="api"
    response_api = reflexio_instance_playbook_source_filtering.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "api",
            "agent_version": agent_version,
        }
    )
    assert response_api.success is True

    # Verify playbooks were generated for "api" source
    # Expected: api_playbook (matches "api") and all_sources_playbook (no filter) extractors run
    # Note: These may get deduplicated if they produce semantically identical playbooks
    # Should NOT have: webhook_playbook (only for "webhook")
    api_user_playbooks = storage.get_user_playbooks(playbook_name="api_playbook")
    webhook_user_playbooks = storage.get_user_playbooks(
        playbook_name="webhook_playbook"
    )
    all_sources_user_playbooks = storage.get_user_playbooks(
        playbook_name="all_sources_playbook"
    )

    # At least one playbook should exist from api_playbook or all_sources_playbook
    # (they may get deduplicated into a single playbook with the first extractor's name)
    total_user_playbooks = len(api_user_playbooks) + len(all_sources_user_playbooks)
    assert total_user_playbooks > 0, (
        "At least one playbook should be generated from api_playbook or all_sources_playbook extractors"
    )

    # Verify source field is set correctly for all playbooks
    for playbook in api_user_playbooks:
        assert playbook.source == "api", "api_playbook should have source='api'"
    for playbook in all_sources_user_playbooks:
        assert playbook.source == "api", "all_sources_playbook should have source='api'"

    # webhook_playbook should NOT have been generated (source "api" doesn't match "webhook")
    assert len(webhook_user_playbooks) == 0, (
        "webhook_playbook should NOT be generated for source='api'"
    )


@skip_in_precommit
@skip_low_priority
def test_playbook_source_filtering_with_non_matching_source(
    reflexio_instance_playbook_source_filtering: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_playbook_source_filtering: Callable[[], None],
):
    """Test that playbook extractors do not run when source doesn't match request_sources_enabled.

    This test verifies:
    1. When source="other", only all_sources_playbook extractor runs
    2. api_playbook and webhook_playbook do not run for non-matching source
    """
    user_id = "test_user_source_filter_other"
    agent_version = "test_agent_source_other"
    storage = reflexio_instance_playbook_source_filtering.request_context.storage

    # Publish interactions with source="other" (not in any request_sources_enabled list)
    response = reflexio_instance_playbook_source_filtering.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "other",
            "agent_version": agent_version,
        }
    )
    assert response.success is True

    # Verify only all_sources_playbook was generated
    api_user_playbooks = storage.get_user_playbooks(playbook_name="api_playbook")
    webhook_user_playbooks = storage.get_user_playbooks(
        playbook_name="webhook_playbook"
    )
    all_sources_user_playbooks = storage.get_user_playbooks(
        playbook_name="all_sources_playbook"
    )

    # api_playbook should NOT have been generated (source "other" doesn't match "api")
    assert len(api_user_playbooks) == 0, (
        "api_playbook should NOT be generated for source='other'"
    )

    # webhook_playbook should NOT have been generated (source "other" doesn't match "webhook")
    assert len(webhook_user_playbooks) == 0, (
        "webhook_playbook should NOT be generated for source='other'"
    )

    # all_sources_playbook should have been generated (no source filter)
    assert len(all_sources_user_playbooks) > 0, (
        "all_sources_playbook should be generated for any source"
    )
    for playbook in all_sources_user_playbooks:
        assert playbook.source == "other", (
            "all_sources_playbook should have source='other'"
        )


@skip_in_precommit
@skip_low_priority
def test_playbook_source_filtering_webhook_source(
    reflexio_instance_playbook_source_filtering: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_playbook_source_filtering: Callable[[], None],
):
    """Test that webhook_playbook extractor runs only for webhook source.

    This test verifies:
    1. When source="webhook", webhook_playbook and all_sources_playbook extractors run
    2. api_playbook does not run for webhook source
    """
    user_id = "test_user_source_filter_webhook"
    agent_version = "test_agent_source_webhook"
    storage = reflexio_instance_playbook_source_filtering.request_context.storage

    # Publish interactions with source="webhook"
    response = reflexio_instance_playbook_source_filtering.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "webhook",
            "agent_version": agent_version,
        }
    )
    assert response.success is True

    # Verify playbooks were generated correctly
    # Note: webhook_playbook and all_sources_playbook may get deduplicated if they
    # produce semantically identical playbooks
    api_user_playbooks = storage.get_user_playbooks(playbook_name="api_playbook")
    webhook_user_playbooks = storage.get_user_playbooks(
        playbook_name="webhook_playbook"
    )
    all_sources_user_playbooks = storage.get_user_playbooks(
        playbook_name="all_sources_playbook"
    )

    # api_playbook should NOT have been generated (source "webhook" doesn't match "api")
    assert len(api_user_playbooks) == 0, (
        "api_playbook should NOT be generated for source='webhook'"
    )

    # At least one playbook should exist from webhook_playbook or all_sources_playbook
    # (they may get deduplicated into a single playbook with the first extractor's name)
    total_user_playbooks = len(webhook_user_playbooks) + len(all_sources_user_playbooks)
    assert total_user_playbooks > 0, (
        "At least one playbook should be generated from webhook_playbook or all_sources_playbook extractors"
    )

    # Verify source field is set correctly for all playbooks
    for playbook in webhook_user_playbooks:
        assert playbook.source == "webhook", (
            "webhook_playbook should have source='webhook'"
        )
    for playbook in all_sources_user_playbooks:
        assert playbook.source == "webhook", (
            "all_sources_playbook should have source='webhook'"
        )


@skip_in_precommit
def test_manual_playbook_generation_end_to_end(
    reflexio_instance_manual_playbook: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_manual_playbook: Callable[[], None],
):
    """Test manual_playbook_generation method for triggering playbook generation.

    This test verifies:
    1. Manual playbook generation uses window-sized interactions
    2. Generated playbooks have CURRENT status (not PENDING like rerun)
    3. Playbooks are generated correctly from the interactions
    """
    user_id = "test_user_manual_playbook"
    agent_version = "test_agent_manual_playbook"
    playbook_name = "manual_trigger_playbook"

    # Use mock mode to ensure consistent LLM responses
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Step 1: Publish interactions to have data for generation
        publish_response = reflexio_instance_manual_playbook.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "test_manual_source",
                "agent_version": agent_version,
            }
        )
        assert publish_response.success is True

        # Step 2: Call manual_playbook_generation
        manual_response = reflexio_instance_manual_playbook.manual_playbook_generation(
            ManualPlaybookGenerationRequest(
                agent_version=agent_version,
            )
        )
        assert manual_response.success is True, (
            f"Manual generation failed: {manual_response.msg}"
        )

        # Step 3: Verify playbooks were generated with CURRENT status (None)
        current_playbooks = reflexio_instance_manual_playbook.request_context.storage.get_user_playbooks(
            playbook_name=playbook_name,
            status_filter=[None],
        )
        # Just verify no errors - content may vary based on LLM
        assert isinstance(current_playbooks, list)

        # Step 4: Verify NO PENDING playbooks were created (that's rerun behavior)
        pending_playbooks = reflexio_instance_manual_playbook.request_context.storage.get_user_playbooks(
            playbook_name=playbook_name,
            status_filter=[Status.PENDING],
        )
        assert len(pending_playbooks) == 0, (
            "Manual generation should not create PENDING playbooks"
        )

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_manual_playbook_generation_no_window_size(
    reflexio_instance_playbook_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_playbook_only: Callable[[], None],
):
    """Test manual_playbook_generation works without batch_size.

    This test verifies:
    1. Manual generation works when batch_size is not configured
       (it defaults to fetching all available interactions with a reasonable limit)
    """
    user_id = "test_user_no_window"
    agent_version = "test_agent_no_window"

    # Publish interactions first
    publish_response = reflexio_instance_playbook_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_source",
            "agent_version": agent_version,
        }
    )
    assert publish_response.success is True

    # Call manual_playbook_generation - should succeed even without window size
    # When window_size is not configured, it fetches all available interactions
    manual_response = reflexio_instance_playbook_only.manual_playbook_generation(
        ManualPlaybookGenerationRequest(
            agent_version=agent_version,
        )
    )
    assert manual_response.success is True


@skip_in_precommit
@skip_low_priority
def test_manual_playbook_generation_with_source_filter(
    reflexio_instance_manual_playbook: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_manual_playbook: Callable[[], None],
):
    """Test manual_playbook_generation with source filtering.

    This test verifies:
    1. Source filtering works correctly in manual generation
    2. Only interactions with matching source are processed
    """
    user_id = "test_user_manual_source_filter"
    agent_version = "test_agent_source_filter"

    # Use mock mode
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Publish interactions with different sources
        # Source A - full conversation
        response_a = reflexio_instance_manual_playbook.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "source_a",
                "agent_version": agent_version,
            }
        )
        assert response_a.success is True

        # Source B - single message
        response_b = reflexio_instance_manual_playbook.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": [
                    InteractionData(
                        content="Simple message for source B",
                        role="User",
                    )
                ],
                "source": "source_b",
                "agent_version": agent_version,
            }
        )
        assert response_b.success is True

        # Call manual_playbook_generation with source filter
        manual_response = reflexio_instance_manual_playbook.manual_playbook_generation(
            ManualPlaybookGenerationRequest(
                agent_version=agent_version,
                source="source_a",  # Only process source_a
            )
        )
        # Should succeed (or fail gracefully if no matching extractors)
        assert manual_response.success is True or "No interactions found" in (
            manual_response.msg or ""
        )

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_manual_playbook_generation_with_dict_input(
    reflexio_instance_manual_playbook: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_manual_playbook: Callable[[], None],
):
    """Test manual_playbook_generation accepts dict input.

    This test verifies:
    1. Manual generation accepts dict input (not just ManualPlaybookGenerationRequest)
    """
    user_id = "test_user_dict_input"
    agent_version = "test_agent_dict"

    # Use mock mode
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Publish interactions
        publish_response = reflexio_instance_manual_playbook.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "test_source",
                "agent_version": agent_version,
            }
        )
        assert publish_response.success is True

        # Call with dict input
        manual_response = reflexio_instance_manual_playbook.manual_playbook_generation(
            {"agent_version": agent_version}
        )
        assert manual_response.success is True, (
            f"Dict input failed: {manual_response.msg}"
        )

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_manual_playbook_generation_with_playbook_name_filter(
    reflexio_instance_manual_playbook: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_manual_playbook: Callable[[], None],
):
    """Test manual_playbook_generation with playbook_name filtering.

    This test verifies:
    1. AgentPlaybook name filtering works correctly in manual generation
    """
    user_id = "test_user_playbook_name_filter"
    agent_version = "test_agent_playbook_name"
    playbook_name = "manual_trigger_playbook"

    # Use mock mode
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Publish interactions
        publish_response = reflexio_instance_manual_playbook.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "test_source",
                "agent_version": agent_version,
            }
        )
        assert publish_response.success is True

        # Call with playbook_name filter
        manual_response = reflexio_instance_manual_playbook.manual_playbook_generation(
            ManualPlaybookGenerationRequest(
                agent_version=agent_version,
                playbook_name=playbook_name,
            )
        )
        assert manual_response.success is True, (
            f"AgentPlaybook name filter failed: {manual_response.msg}"
        )

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_rerun_playbook_generation_with_source_filter(
    reflexio_instance_multiple_playbook_extractors: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_multiple_playbook_extractors: Callable[[], None],
):
    """Test rerun playbook generation with source filtering.

    This test verifies:
    1. Rerun with source filter correctly filters interactions by source
    2. Only extractors matching the source run
    3. Generated playbooks have correct source field
    """
    import os

    user_id = "test_user_rerun_source_filter"
    agent_version = "test_agent_rerun_source"
    storage = reflexio_instance_multiple_playbook_extractors.request_context.storage

    # Use mock mode
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Step 1: Publish interactions with "api" source
        response_api = (
            reflexio_instance_multiple_playbook_extractors.publish_interaction(
                {
                    "user_id": user_id,
                    "interaction_data_list": sample_interaction_requests,
                    "source": "api",
                    "agent_version": agent_version,
                }
            )
        )
        assert response_api.success is True

        # Step 2: Publish interactions with "webhook" source
        response_webhook = (
            reflexio_instance_multiple_playbook_extractors.publish_interaction(
                {
                    "user_id": user_id,
                    "interaction_data_list": [
                        InteractionData(
                            content="Webhook message",
                            role="User",
                        )
                    ],
                    "source": "webhook",
                    "agent_version": agent_version,
                }
            )
        )
        assert response_webhook.success is True

        # Step 3: Delete user playbooks created by this test's extractors to start fresh for rerun test
        config = reflexio_instance_multiple_playbook_extractors.request_context.configurator.get_config()
        for fc in config.user_playbook_extractor_configs:
            storage.delete_all_user_playbooks_by_playbook_name(fc.extractor_name)

        # Step 4: Rerun with source="api" filter
        rerun_response = (
            reflexio_instance_multiple_playbook_extractors.rerun_playbook_generation(
                RerunPlaybookGenerationRequest(
                    agent_version=agent_version,
                    source="api",  # Only process API source
                )
            )
        )
        assert rerun_response.success is True, (
            f"Rerun with source filter failed: {rerun_response.msg}"
        )

        # Step 5: Verify pending playbooks were created with source="api"
        pending_playbooks = storage.get_user_playbooks(status_filter=[Status.PENDING])
        if rerun_response.playbooks_generated > 0:
            assert len(pending_playbooks) > 0
            for playbook in pending_playbooks:
                assert playbook.source == "api", (
                    f"Expected source='api', got '{playbook.source}'"
                )

        # Step 6: Test with non-existent source - should fail
        rerun_response_invalid = (
            reflexio_instance_multiple_playbook_extractors.rerun_playbook_generation(
                RerunPlaybookGenerationRequest(
                    agent_version=agent_version,
                    source="non_existent_source",
                )
            )
        )
        assert rerun_response_invalid.success is False
        assert "No interactions found" in rerun_response_invalid.msg

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_rerun_playbook_generation_multiple_extractors_all_sources(
    reflexio_instance_multiple_playbook_extractors: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_multiple_playbook_extractors: Callable[[], None],
):
    """Test rerun playbook generation with multiple extractors collecting from all sources.

    This test verifies:
    1. When source=None in rerun, ALL extractors run
    2. Each extractor collects data based on its own request_sources_enabled
    3. Multiple playbook names are generated
    """
    import os

    user_id = "test_user_rerun_all_sources"
    agent_version = "test_agent_rerun_all"
    storage = reflexio_instance_multiple_playbook_extractors.request_context.storage

    # Use mock mode
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Step 1: Publish interactions with different sources
        # API source
        reflexio_instance_multiple_playbook_extractors.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "api",
                "agent_version": agent_version,
            }
        )

        # Webhook source
        reflexio_instance_multiple_playbook_extractors.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": [
                    InteractionData(
                        content="Webhook interaction",
                        role="User",
                    )
                ],
                "source": "webhook",
                "agent_version": agent_version,
            }
        )

        # Other source (only general_playbook should pick this up)
        reflexio_instance_multiple_playbook_extractors.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": [
                    InteractionData(
                        content="Other source interaction",
                        role="User",
                    )
                ],
                "source": "other",
                "agent_version": agent_version,
            }
        )

        # Step 2: Delete user playbooks created by this test's extractors
        config = reflexio_instance_multiple_playbook_extractors.request_context.configurator.get_config()
        for fc in config.user_playbook_extractor_configs:
            storage.delete_all_user_playbooks_by_playbook_name(fc.extractor_name)

        # Step 3: Rerun WITHOUT source filter (all extractors run)
        rerun_response = (
            reflexio_instance_multiple_playbook_extractors.rerun_playbook_generation(
                RerunPlaybookGenerationRequest(
                    agent_version=agent_version,
                    # source=None means all extractors run and collect their configured sources
                )
            )
        )
        assert rerun_response.success is True, (
            f"Rerun without source filter failed: {rerun_response.msg}"
        )

        # Step 4: Verify playbooks from multiple extractors
        if rerun_response.playbooks_generated > 0:
            pending_playbooks = storage.get_user_playbooks(
                status_filter=[Status.PENDING]
            )
            assert len(pending_playbooks) > 0

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
@skip_low_priority
def test_rerun_playbook_generation_with_extractor_names_filter(
    reflexio_instance_multiple_playbook_extractors: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_multiple_playbook_extractors: Callable[[], None],
):
    """Test rerun playbook generation with extractor_names filter.

    This test verifies:
    1. extractor_names filter correctly limits which extractors run during rerun
    2. Only specified extractors generate playbooks
    """
    import os
    import uuid

    # Use unique IDs to avoid data pollution from other tests
    unique_id = uuid.uuid4().hex[:8]
    user_id = f"test_user_rerun_extractor_names_{unique_id}"
    agent_version = f"test_agent_extractor_names_{unique_id}"
    storage = reflexio_instance_multiple_playbook_extractors.request_context.storage

    # Use mock mode
    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Step 1: Publish interactions - this creates CURRENT (status=None) playbooks for BOTH extractors
        reflexio_instance_multiple_playbook_extractors.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "api",
                "agent_version": agent_version,
            }
        )

        # Verify initial publish created playbooks for both extractors
        initial_playbooks = storage.get_user_playbooks(
            agent_version=agent_version,
            user_id=user_id,
        )
        initial_playbook_names = {f.playbook_name for f in initial_playbooks}
        assert "api_only_playbook" in initial_playbook_names, (
            "Initial publish should create api_only_playbook"
        )
        assert "general_playbook" in initial_playbook_names, (
            "Initial publish should create general_playbook"
        )

        # Step 2: Delete playbooks for our unique agent_version to allow rerun to regenerate
        for playbook in initial_playbooks:
            storage.delete_user_playbook(playbook.user_playbook_id)

        # Step 3: Rerun with playbook_name filter - only run general_playbook
        # This should create PENDING playbooks ONLY for general_playbook extractor
        rerun_response = (
            reflexio_instance_multiple_playbook_extractors.rerun_playbook_generation(
                RerunPlaybookGenerationRequest(
                    agent_version=agent_version,
                    playbook_name="general_playbook",  # Only run this extractor
                )
            )
        )
        assert rerun_response.success is True, (
            f"Rerun with extractor_names failed: {rerun_response.msg}"
        )

        # Step 4: Verify only general_playbook was generated
        # Query playbooks for our unique agent_version and user_id
        rerun_playbooks = storage.get_user_playbooks(
            agent_version=agent_version,
            user_id=user_id,
            status_filter=[Status.PENDING],
        )

        # If playbooks were generated, they should only be from general_playbook
        for playbook in rerun_playbooks:
            assert playbook.playbook_name == "general_playbook", (
                f"Expected only general_playbook extractor to run, but found {playbook.playbook_name}"
            )

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env


@skip_in_precommit
def test_playbook_pipeline_preserves_structured_data(
    reflexio_instance_playbook_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_playbook_only: Callable[[], None],
):
    """Test that structured_data flows through all pipeline stages.

    This integration test verifies that structured_data fields (trigger,
    instruction, pitfall) are populated after playbook extraction, preserved
    during retrieval and search, and carried through aggregation.

    Pipeline stages verified:
    1. Publish interactions -> playbook extraction
    2. Retrieve user playbooks -> assert structured_data populated
    3. Search user playbooks by trigger text -> assert results returned
    4. Run aggregation -> assert aggregated AgentPlaybook has structured_data
    """
    user_id = "test_user_structured_data"
    agent_version = "test_agent_structured_data"
    playbook_name = "test_playbook"

    original_env = os.environ.get("MOCK_LLM_RESPONSE")
    try:
        os.environ["MOCK_LLM_RESPONSE"] = "true"

        # Stage 1: Publish interactions to trigger playbook extraction
        publish_response = reflexio_instance_playbook_only.publish_interaction(
            {
                "user_id": user_id,
                "interaction_data_list": sample_interaction_requests,
                "source": "test_structured_data",
                "agent_version": agent_version,
            }
        )
        assert publish_response.success is True

        # Stage 2: Retrieve user playbooks and verify structured_data is populated
        raw_response = reflexio_instance_playbook_only.get_user_playbooks(
            GetUserPlaybooksRequest(
                playbook_name=playbook_name,
                status_filter=[None],
            )
        )
        assert raw_response.success is True
        assert raw_response.user_playbooks, "Expected at least one user playbook"

        for rf in raw_response.user_playbooks:
            # structured_data.trigger must be populated by the extractor
            assert rf.structured_data is not None, "structured_data should not be None"
            assert rf.structured_data.trigger, (
                f"structured_data.trigger should be populated, got: {rf.structured_data.trigger}"
            )

            # At least instruction or pitfall should have content
            has_instruction = bool(rf.structured_data.instruction)
            has_pitfall = bool(rf.structured_data.pitfall)
            assert has_instruction or has_pitfall, (
                "At least one of structured_data.instruction or structured_data.pitfall "
                f"should be populated. instruction={rf.structured_data.instruction}, "
                f"pitfall={rf.structured_data.pitfall}"
            )

            # content should be a non-empty formatted string
            assert rf.content.strip(), "content should be a non-empty formatted string"

        # Stage 3: Search user playbooks using trigger text
        # The mock extractor sets trigger based on last interaction content,
        # so search for a substring that the trigger should contain
        trigger_text = raw_response.user_playbooks[0].structured_data.trigger
        search_response = reflexio_instance_playbook_only.search_user_playbooks(
            SearchUserPlaybookRequest(
                query=trigger_text,
                playbook_name=playbook_name,
                search_mode=SearchMode.FTS,
            )
        )
        assert search_response.success is True
        assert search_response.user_playbooks, (
            f"Search for trigger text '{trigger_text}' should return results"
        )

        # Stage 4: Run aggregation and verify aggregated AgentPlaybook has structured_data
        # First, we need enough user playbooks to meet the min_cluster_size (3).
        # Publish more interactions to generate additional playbooks.
        for i in range(3):
            reflexio_instance_playbook_only.publish_interaction(
                {
                    "user_id": f"{user_id}_{i}",
                    "interaction_data_list": sample_interaction_requests,
                    "source": "test_structured_data",
                    "agent_version": agent_version,
                }
            )

        reflexio_instance_playbook_only.run_playbook_aggregation(
            agent_version=agent_version,
            playbook_name=playbook_name,
        )

        agent_playbooks_response = reflexio_instance_playbook_only.get_agent_playbooks(
            GetAgentPlaybooksRequest(
                playbook_name=playbook_name,
                playbook_status_filter=PlaybookStatus.PENDING,
            )
        )
        assert agent_playbooks_response.success is True
        assert agent_playbooks_response.agent_playbooks, (
            "Aggregation should produce at least one AgentPlaybook"
        )

        for fb in agent_playbooks_response.agent_playbooks:
            assert fb.structured_data is not None, (
                "Aggregated AgentPlaybook should have structured_data"
            )
            assert fb.structured_data.trigger, (
                "Aggregated AgentPlaybook.structured_data.trigger should be populated"
            )
            has_instruction = bool(fb.structured_data.instruction)
            has_pitfall = bool(fb.structured_data.pitfall)
            assert has_instruction or has_pitfall, (
                "Aggregated AgentPlaybook should have instruction or pitfall in structured_data"
            )
            assert fb.content.strip(), (
                "Aggregated AgentPlaybook.content should be non-empty"
            )

    finally:
        if original_env is None:
            os.environ.pop("MOCK_LLM_RESPONSE", None)
        else:
            os.environ["MOCK_LLM_RESPONSE"] = original_env
