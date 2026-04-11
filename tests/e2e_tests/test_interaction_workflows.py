"""End-to-end tests for interaction workflows."""

from collections.abc import Callable

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.retriever_schema import (
    GetInteractionsRequest,
    SearchInteractionRequest,
)
from reflexio.models.api_schema.service_schemas import (
    DeleteUserInteractionRequest,
    InteractionData,
)
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)
from tests.server.test_utils import skip_in_precommit, skip_low_priority


@skip_in_precommit
def test_publish_interaction_end_to_end(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test end-to-end interaction publishing workflow."""
    user_id = "test_user_123"
    agent_version = "test_agent_v1"
    session_id = "test_session_interaction_e2e"

    # Publish interactions (request_id will be auto-generated)
    response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
            "agent_version": agent_version,
            "session_id": session_id,
        }
    )

    # Verify successful publication
    assert response.success is True
    assert response.message == "Interaction published successfully"

    # Get the auto-generated request_id from stored interactions
    final_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    assert len(final_interactions) == len(sample_interaction_requests)
    request_id = final_interactions[0].request_id

    # Verify Request was stored
    stored_request = reflexio_instance.request_context.storage.get_request(request_id)
    assert stored_request is not None
    assert stored_request.request_id == request_id
    assert stored_request.user_id == user_id
    assert stored_request.agent_version == agent_version

    # Verify interactions were added to storage
    final_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    assert len(final_interactions) == len(sample_interaction_requests)

    # Verify profiles were generated and added to storage
    final_profiles = reflexio_instance.request_context.storage.get_all_profiles()
    assert len(final_profiles) > 0
    assert final_profiles[0].content.strip() != ""
    assert (
        final_profiles[0].custom_features is not None
        and final_profiles[0].custom_features.get("metadata") is not None
    )

    # Verify profile change logs were created
    final_change_logs = (
        reflexio_instance.request_context.storage.get_profile_change_logs()
    )
    assert len(final_change_logs) > 0

    # Verify playbooks were generated and stored
    user_playbooks = reflexio_instance.request_context.storage.get_user_playbooks(
        playbook_name="test_playbook"
    )
    assert len(user_playbooks) > 0 and user_playbooks[0].content.strip() != ""

    # Trigger and verify agent success evaluation results
    run_group_evaluation(
        org_id=reflexio_instance.org_id,
        user_id=user_id,
        session_id=session_id,
        agent_version=agent_version,
        source="test_conversation",
        request_context=reflexio_instance.request_context,
        llm_client=reflexio_instance.llm_client,
    )
    agent_success_results = (
        reflexio_instance.request_context.storage.get_agent_success_evaluation_results(
            agent_version=agent_version
        )
    )
    assert len(agent_success_results) > 0
    assert agent_success_results[0].session_id == session_id
    assert agent_success_results[0].agent_version == agent_version
    assert isinstance(agent_success_results[0].is_success, bool)


@skip_in_precommit
def test_search_interactions_end_to_end(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test end-to-end interaction search workflow."""
    user_id = "test_user_456"

    # First publish some interactions
    reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )

    # Verify interactions were stored
    stored_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    assert len(stored_interactions) == len(sample_interaction_requests)

    # Search for interactions
    search_request = SearchInteractionRequest(
        user_id=user_id,
        query="software solution",
        top_k=5,
    )

    response = reflexio_instance.search_interactions(search_request)

    # Verify search results
    assert response.success is True
    assert len(response.interactions) > 0

    # Verify interaction content contains search terms
    found_content = False
    for interaction in response.interactions:
        if "software" in interaction.content.lower():
            found_content = True
            break
    assert found_content, "Should find interactions containing search terms"


@skip_in_precommit
def test_get_interactions_end_to_end(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test end-to-end get interactions workflow."""
    user_id = "test_user_get"

    # First publish interactions
    reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )

    # Verify interactions were stored
    stored_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    assert len(stored_interactions) == len(sample_interaction_requests)
    request_id = stored_interactions[0].request_id if stored_interactions else None

    # Get all interactions for the user
    get_request = GetInteractionsRequest(user_id=user_id)
    response = reflexio_instance.get_interactions(get_request)

    # Verify results
    assert response.success is True
    assert len(response.interactions) >= len(sample_interaction_requests)

    # Verify interaction details
    for interaction in response.interactions:
        assert interaction.user_id == user_id
        assert interaction.request_id == request_id
        assert interaction.content is not None


@skip_in_precommit
def test_delete_interaction_end_to_end(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test end-to-end interaction deletion workflow."""
    user_id = "test_user_delete"

    # First publish interactions
    response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_conversation",
        }
    )
    assert response.success is True

    # Verify interactions were stored
    initial_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    assert len(initial_interactions) == len(sample_interaction_requests)

    # Get interactions to find interaction IDs
    get_request = GetInteractionsRequest(user_id=user_id)
    get_response = reflexio_instance.get_interactions(get_request)
    assert len(get_response.interactions) > 0

    # Delete the first interaction
    interaction_to_delete = get_response.interactions[0]
    delete_request = DeleteUserInteractionRequest(
        user_id=user_id,
        interaction_id=interaction_to_delete.interaction_id,
    )

    delete_response = reflexio_instance.delete_interaction(delete_request)
    assert delete_response.success is True

    # Verify interaction was deleted from storage
    final_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    assert len(final_interactions) < len(initial_interactions)


@skip_in_precommit
@skip_low_priority
def test_dict_input_handling_end_to_end(
    reflexio_instance: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_after_test: Callable[[], None],
):
    """Test that the library handles both dict and object inputs correctly."""
    user_id = "test_user_dict"

    # Get initial state
    initial_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    reflexio_instance.request_context.storage.get_all_profiles()

    # Convert interaction requests to dictionaries
    interaction_dicts = [
        {
            "content": interaction.content,
            "role": interaction.role,
            "user_action": interaction.user_action,
            "user_action_description": interaction.user_action_description,
            "interacted_image_url": interaction.interacted_image_url,
        }
        for interaction in sample_interaction_requests
    ]

    # Publish using dict inputs
    response = reflexio_instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": interaction_dicts,
            "source": "test_conversation",
        }
    )
    assert response.success is True

    # Verify data was stored
    stored_interactions = (
        reflexio_instance.request_context.storage.get_all_interactions()
    )
    stored_profiles = reflexio_instance.request_context.storage.get_user_profile(
        user_id
    )
    assert len(stored_interactions) > len(initial_interactions)
    assert len(stored_profiles) > 0

    # Search using dict input
    search_dict = {
        "user_id": user_id,
        "query": "Sarah",
        "top_k": 5,
    }
    search_response = reflexio_instance.search_interactions(search_dict)
    assert search_response.success is True
    assert len(search_response.interactions) > 0

    # Search profiles using dict input (use actual profile content for reliable search)
    profile_content = stored_profiles[0].content
    search_words = " ".join(profile_content.split()[:4])
    profile_search_dict = {
        "user_id": user_id,
        "query": search_words,  # Use actual profile content for search
        "top_k": 5,
    }
    profile_response = reflexio_instance.search_profiles(profile_search_dict)
    assert profile_response.success is True
    assert len(profile_response.user_profiles) > 0
    # Verify all returned profiles have CURRENT status (default search filter)
    for profile in profile_response.user_profiles:
        assert profile.status is None, (
            f"Default search should return only CURRENT profiles, got status={profile.status}"
        )
