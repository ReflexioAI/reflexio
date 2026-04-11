"""End-to-end tests for agent success evaluation workflows."""

from collections.abc import Callable
from unittest.mock import patch

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.retriever_schema import (
    GetAgentSuccessEvaluationResultsRequest,
    GetDashboardStatsRequest,
)
from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    RegularVsShadow,
    UserActionType,
)
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)
from tests.server.test_utils import skip_in_precommit, skip_low_priority


def _trigger_group_evaluation(
    instance: Reflexio,
    user_id: str,
    session_id: str,
    agent_version: str,
    source: str | None = None,
) -> None:
    """Trigger group evaluation synchronously for e2e tests.

    In production, evaluation is scheduled via the delayed group evaluator.
    This helper calls the runner directly so tests don't have to wait.
    The delay guard is bypassed since we're invoking the runner directly.
    """
    with patch(
        "reflexio.server.services.agent_success_evaluation.group_evaluation_runner._EFFECTIVE_DELAY_SECONDS",
        0,
    ):
        run_group_evaluation(
            org_id=instance.org_id,
            user_id=user_id,
            session_id=session_id,
            agent_version=agent_version,
            source=source,
            request_context=instance.request_context,
            llm_client=instance.llm_client,
        )


@skip_in_precommit
def test_publish_interaction_agent_success_only(
    reflexio_instance_agent_success_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_agent_success_only: None,  # noqa: ARG001
):
    """Test interaction publishing with only agent success evaluation enabled."""
    user_id = "test_user_agent_success_only"
    agent_version = "test_agent_success"
    session_id = "test_group_agent_success_only"

    # Publish interactions (request_id will be auto-generated)
    response = reflexio_instance_agent_success_only.publish_interaction(
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

    # Verify interactions were added to storage
    final_interactions = reflexio_instance_agent_success_only.request_context.storage.get_all_interactions()
    assert len(final_interactions) == len(sample_interaction_requests)

    # Trigger group evaluation synchronously (normally delayed)
    _trigger_group_evaluation(
        reflexio_instance_agent_success_only,
        user_id,
        session_id,
        agent_version,
        source="test_conversation",
    )

    # Verify agent success evaluation results were created
    agent_success_results = reflexio_instance_agent_success_only.request_context.storage.get_agent_success_evaluation_results(
        agent_version=agent_version
    )
    assert len(agent_success_results) > 0
    assert agent_success_results[0].session_id == session_id
    assert agent_success_results[0].agent_version == agent_version
    assert isinstance(agent_success_results[0].is_success, bool)
    # Verify new evaluation metric fields
    assert isinstance(agent_success_results[0].number_of_correction_per_session, int)
    assert isinstance(agent_success_results[0].is_escalated, bool)
    if agent_success_results[0].is_success:
        assert agent_success_results[0].user_turns_to_resolution is None or isinstance(
            agent_success_results[0].user_turns_to_resolution, int
        )
    else:
        assert agent_success_results[0].user_turns_to_resolution is None

    # Note: profiles and playbooks may still be generated because Config defaults
    # always populate profile_extractor_configs and user_playbook_extractor_configs.
    # This test focuses on verifying agent success evaluation works correctly.


@skip_in_precommit
def test_get_agent_success_evaluations_end_to_end(
    reflexio_instance_agent_success_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_agent_success_only: Callable[[], None],
):
    """Test end-to-end workflow for getting agent success evaluation results.

    This test verifies:
    1. Publishing interactions creates agent success evaluations
    2. get_agent_success_evaluation_results retrieves results correctly
    3. Filtering by agent_version works
    4. Limit parameter works correctly
    5. Results contain expected fields
    """
    user_id = "test_user_get_evaluations"
    agent_version = "test_agent_v1"
    session_id = "test_group_get_evaluations"

    # Step 1: Publish interactions to generate evaluations
    publish_response = reflexio_instance_agent_success_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_evaluations_source",
            "agent_version": agent_version,
            "session_id": session_id,
        }
    )
    assert publish_response.success is True

    # Trigger group evaluation synchronously (normally delayed)
    _trigger_group_evaluation(
        reflexio_instance_agent_success_only,
        user_id,
        session_id,
        agent_version,
        source="test_evaluations_source",
    )

    # Step 2: Get agent success evaluations via API
    get_response = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            GetAgentSuccessEvaluationResultsRequest(agent_version=agent_version)
        )
    )
    assert get_response.success is True
    assert len(get_response.agent_success_evaluation_results) > 0

    # Step 3: Verify result fields
    result = get_response.agent_success_evaluation_results[0]
    assert result.session_id == session_id
    assert result.agent_version == agent_version
    assert isinstance(result.is_success, bool)
    assert isinstance(result.failure_type, str)
    assert isinstance(result.failure_reason, str)
    assert result.created_at > 0

    # Step 4: Test filtering by agent_version (non-existent version)
    empty_response = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            GetAgentSuccessEvaluationResultsRequest(
                agent_version="non_existent_version"
            )
        )
    )
    assert empty_response.success is True
    assert len(empty_response.agent_success_evaluation_results) == 0

    # Step 5: Test limit parameter
    limited_response = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            GetAgentSuccessEvaluationResultsRequest(
                agent_version=agent_version, limit=1
            )
        )
    )
    assert limited_response.success is True
    assert len(limited_response.agent_success_evaluation_results) <= 1

    # Step 6: Test with dict input
    dict_response = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            {"agent_version": agent_version, "limit": 10}
        )
    )
    assert dict_response.success is True
    assert len(dict_response.agent_success_evaluation_results) > 0


@skip_in_precommit
@skip_low_priority
def test_agent_success_evaluation_statistics(
    reflexio_instance_agent_success_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_agent_success_only: Callable[[], None],
):
    """Test agent success evaluation statistics in dashboard.

    This test verifies:
    1. Dashboard stats include evaluation time series data
    2. Evaluation counts are reflected in statistics
    3. Time series data is properly formatted
    """
    user_id = "test_user_stats"
    agent_version = "test_agent_stats"
    session_id = "test_group_stats"

    # Step 1: Publish interactions to generate evaluations
    publish_response = reflexio_instance_agent_success_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_stats_source",
            "agent_version": agent_version,
            "session_id": session_id,
        }
    )
    assert publish_response.success is True

    # Trigger group evaluation synchronously (normally delayed)
    _trigger_group_evaluation(
        reflexio_instance_agent_success_only,
        user_id,
        session_id,
        agent_version,
        source="test_stats_source",
    )

    # Verify evaluations were created
    evaluations = reflexio_instance_agent_success_only.request_context.storage.get_agent_success_evaluation_results(
        agent_version=agent_version
    )
    assert len(evaluations) > 0

    # Step 2: Get dashboard statistics
    stats_response = reflexio_instance_agent_success_only.get_dashboard_stats(
        GetDashboardStatsRequest(days_back=7)
    )
    assert stats_response.success is True
    assert stats_response.stats is not None

    # Step 3: Verify evaluations time series exists
    assert stats_response.stats.evaluations_time_series is not None
    assert len(stats_response.stats.evaluations_time_series) >= 0

    # Step 4: Verify time series data format (if data exists)
    if len(stats_response.stats.evaluations_time_series) > 0:
        ts_point = stats_response.stats.evaluations_time_series[0]
        assert hasattr(ts_point, "timestamp")
        assert hasattr(ts_point, "value")

    # Step 5: Verify current period stats
    current_period = stats_response.stats.current_period
    assert current_period is not None
    # Interactions should be reflected
    assert current_period.total_interactions >= len(sample_interaction_requests)


@skip_in_precommit
@skip_low_priority
def test_multiple_agent_versions_evaluation(
    reflexio_instance_agent_success_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_agent_success_only: Callable[[], None],
):
    """Test agent success evaluations across multiple agent versions.

    This test verifies:
    1. Different agent versions create separate evaluation records
    2. Filtering by agent_version returns only matching results
    3. Results without version filter return all versions
    4. Each version's evaluations are independent
    """
    user_id = "test_user_multi_version"
    agent_version_v1 = "test_agent_v1.0"
    agent_version_v2 = "test_agent_v2.0"
    agent_version_v3 = "test_agent_v3.0"
    session_id_v1 = "test_session_v1"
    session_id_v2 = "test_session_v2"
    session_id_v3 = "test_session_v3"

    # Step 1: Publish interactions with different agent versions
    # Version 1
    publish_v1 = reflexio_instance_agent_success_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests[:2],
            "source": "source_v1",
            "agent_version": agent_version_v1,
            "session_id": session_id_v1,
        }
    )
    assert publish_v1.success is True

    # Version 2
    publish_v2 = reflexio_instance_agent_success_only.publish_interaction(
        {
            "user_id": f"{user_id}_v2",
            "interaction_data_list": sample_interaction_requests[1:3],
            "source": "source_v2",
            "agent_version": agent_version_v2,
            "session_id": session_id_v2,
        }
    )
    assert publish_v2.success is True

    # Version 3
    publish_v3 = reflexio_instance_agent_success_only.publish_interaction(
        {
            "user_id": f"{user_id}_v3",
            "interaction_data_list": sample_interaction_requests,
            "source": "source_v3",
            "agent_version": agent_version_v3,
            "session_id": session_id_v3,
        }
    )
    assert publish_v3.success is True

    # Trigger group evaluations synchronously (normally delayed)
    _trigger_group_evaluation(
        reflexio_instance_agent_success_only,
        user_id,
        session_id_v1,
        agent_version_v1,
        source="source_v1",
    )
    _trigger_group_evaluation(
        reflexio_instance_agent_success_only,
        f"{user_id}_v2",
        session_id_v2,
        agent_version_v2,
        source="source_v2",
    )
    _trigger_group_evaluation(
        reflexio_instance_agent_success_only,
        f"{user_id}_v3",
        session_id_v3,
        agent_version_v3,
        source="source_v3",
    )

    # Step 2: Get evaluations for each version separately
    results_v1 = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            GetAgentSuccessEvaluationResultsRequest(agent_version=agent_version_v1)
        )
    )
    assert results_v1.success is True
    assert len(results_v1.agent_success_evaluation_results) > 0
    for result in results_v1.agent_success_evaluation_results:
        assert result.agent_version == agent_version_v1

    results_v2 = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            GetAgentSuccessEvaluationResultsRequest(agent_version=agent_version_v2)
        )
    )
    assert results_v2.success is True
    assert len(results_v2.agent_success_evaluation_results) > 0
    for result in results_v2.agent_success_evaluation_results:
        assert result.agent_version == agent_version_v2

    results_v3 = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            GetAgentSuccessEvaluationResultsRequest(agent_version=agent_version_v3)
        )
    )
    assert results_v3.success is True
    assert len(results_v3.agent_success_evaluation_results) > 0
    for result in results_v3.agent_success_evaluation_results:
        assert result.agent_version == agent_version_v3

    # Step 3: Get all evaluations without version filter
    all_results = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            GetAgentSuccessEvaluationResultsRequest(limit=100)
        )
    )
    assert all_results.success is True

    # Total should include all versions
    total_expected = (
        len(results_v1.agent_success_evaluation_results)
        + len(results_v2.agent_success_evaluation_results)
        + len(results_v3.agent_success_evaluation_results)
    )
    assert len(all_results.agent_success_evaluation_results) >= total_expected

    # Step 4: Verify version isolation - v1 results don't appear in v2 query
    v1_session_ids = {r.session_id for r in results_v1.agent_success_evaluation_results}
    v2_session_ids = {r.session_id for r in results_v2.agent_success_evaluation_results}
    v3_session_ids = {r.session_id for r in results_v3.agent_success_evaluation_results}

    # Session IDs should be unique across versions (different publishes)
    assert len(v1_session_ids & v2_session_ids) == 0, (
        "V1 and V2 should have different session_ids"
    )
    assert len(v2_session_ids & v3_session_ids) == 0, (
        "V2 and V3 should have different session_ids"
    )
    assert len(v1_session_ids & v3_session_ids) == 0, (
        "V1 and V3 should have different session_ids"
    )

    # Step 5: Verify each result has proper structure
    for results in [results_v1, results_v2, results_v3]:
        for result in results.agent_success_evaluation_results:
            assert result.result_id >= 0
            assert result.session_id != ""
            assert result.agent_version != ""
            assert isinstance(result.is_success, bool)
            assert isinstance(result.failure_type, str)
            assert isinstance(result.failure_reason, str)
            assert result.created_at > 0


@skip_in_precommit
def test_evaluate_regular_vs_shadow_content(
    reflexio_instance_agent_success_only: Reflexio,
    cleanup_agent_success_only: Callable[[], None],
):
    """Test agent success evaluation with regular vs shadow content comparison.

    This test verifies:
    1. Publishing interactions with shadow_content triggers comparison evaluation
    2. The regular_vs_shadow field is populated in evaluation results
    3. The comparison result is a valid RegularVsShadow enum value
    4. Both success evaluation and comparison work in a single LLM call
    """
    user_id = "test_user_shadow_comparison"
    agent_version = "test_agent_shadow"
    session_id = "test_session_shadow"

    # Create interactions with shadow_content
    # Regular: Sales rep gives a brief response
    # Shadow: Sales rep gives a more detailed, helpful response
    interactions_with_shadow = [
        InteractionData(
            content="Hi, I'm looking for information about your software pricing.",
            role="Customer",
            user_action=UserActionType.NONE,
            user_action_description="",
            interacted_image_url="",
        ),
        InteractionData(
            content="Our pricing starts at $99/month.",
            shadow_content="Our pricing starts at $99/month for the basic plan, which includes 5 users and 10GB storage. We also have a Pro plan at $199/month with unlimited users and 100GB storage. Would you like me to explain the differences in more detail?",
            role="Sales Rep",
            user_action=UserActionType.NONE,
            user_action_description="",
            interacted_image_url="",
        ),
        InteractionData(
            content="That sounds interesting, can you tell me more?",
            role="Customer",
            user_action=UserActionType.NONE,
            user_action_description="",
            interacted_image_url="",
        ),
    ]

    # Step 1: Publish interactions with shadow content
    publish_response = reflexio_instance_agent_success_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": interactions_with_shadow,
            "source": "test_shadow_comparison",
            "agent_version": agent_version,
            "session_id": session_id,
        }
    )
    assert publish_response.success is True

    # Step 2: Verify interactions were stored
    stored_interactions = reflexio_instance_agent_success_only.request_context.storage.get_all_interactions()
    assert len(stored_interactions) == len(interactions_with_shadow)

    # Verify shadow_content was stored
    shadow_interactions = [i for i in stored_interactions if i.shadow_content]
    assert len(shadow_interactions) > 0, "Shadow content should be stored"

    # Trigger group evaluation synchronously (normally delayed)
    _trigger_group_evaluation(
        reflexio_instance_agent_success_only,
        user_id,
        session_id,
        agent_version,
        source="test_shadow_comparison",
    )

    # Step 3: Get agent success evaluations
    get_response = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            GetAgentSuccessEvaluationResultsRequest(agent_version=agent_version)
        )
    )
    assert get_response.success is True
    assert len(get_response.agent_success_evaluation_results) > 0

    # Step 4: Verify regular_vs_shadow field is populated
    result = get_response.agent_success_evaluation_results[0]
    assert result.agent_version == agent_version
    assert isinstance(result.is_success, bool)

    # The regular_vs_shadow should be populated when shadow content exists
    assert result.regular_vs_shadow is not None, (
        "regular_vs_shadow should be populated when shadow content exists"
    )

    # Step 5: Verify the value is a valid RegularVsShadow enum
    valid_values = [e.value for e in RegularVsShadow]
    assert result.regular_vs_shadow in valid_values, (
        f"regular_vs_shadow should be one of {valid_values}, got {result.regular_vs_shadow}"
    )

    # Step 6: Log the comparison result for debugging
    print(f"Evaluation result: is_success={result.is_success}")
    print(f"Regular vs Shadow comparison: {result.regular_vs_shadow}")


@skip_in_precommit
@skip_low_priority
def test_evaluate_without_shadow_content(
    reflexio_instance_agent_success_only: Reflexio,
    sample_interaction_requests: list[InteractionData],
    cleanup_agent_success_only: Callable[[], None],
):
    """Test that regular_vs_shadow is None when no shadow content exists.

    This test verifies:
    1. Interactions without shadow_content don't trigger comparison
    2. The regular_vs_shadow field remains None
    3. Regular success evaluation still works correctly
    """
    user_id = "test_user_no_shadow"
    agent_version = "test_agent_no_shadow"
    session_id = "test_session_no_shadow"

    # Step 1: Publish interactions WITHOUT shadow content
    publish_response = reflexio_instance_agent_success_only.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": sample_interaction_requests,
            "source": "test_no_shadow",
            "agent_version": agent_version,
            "session_id": session_id,
        }
    )
    assert publish_response.success is True

    # Trigger group evaluation synchronously (normally delayed)
    _trigger_group_evaluation(
        reflexio_instance_agent_success_only,
        user_id,
        session_id,
        agent_version,
        source="test_no_shadow",
    )

    # Step 2: Get agent success evaluations
    get_response = (
        reflexio_instance_agent_success_only.get_agent_success_evaluation_results(
            GetAgentSuccessEvaluationResultsRequest(agent_version=agent_version)
        )
    )
    assert get_response.success is True
    assert len(get_response.agent_success_evaluation_results) > 0

    # Step 3: Verify regular_vs_shadow is None (no shadow content)
    result = get_response.agent_success_evaluation_results[0]
    assert result.agent_version == agent_version
    assert isinstance(result.is_success, bool)
    assert result.regular_vs_shadow is None, (
        "regular_vs_shadow should be None when no shadow content exists"
    )
