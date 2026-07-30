"""OSS route + app-shape golden — Phase-0 safety harness, set-based (Task 0.1 / A2).

This is the behavioral backstop for the create_app re-architecture: it pins the
*observable shape* of the OSS app built by ``create_app()`` so that the A2
route-module split (75 handlers moved into ``reflexio/server/routes/<domain>.py``
sub-routers, aggregated back into ``core_router``) is proven behavior-preserving.

**Ordered → set (A2 adjustment).** The 75 core handlers were interleaved by
domain in the old monolithic ``api.py``; clean per-domain modules necessarily
regroup the global route list (all playbooks together, etc.). An ordered
comparison would flag that pure reordering as a regression even though routing
is unaffected: **all route paths are distinct literal paths with no two
templates collapsing to the same shape** (proved by
``test_no_overlapping_path_templates``), so FastAPI's declaration-order
first-match is never exercised. The route assertion therefore compares the
inventory as a SET of ``(path, sorted methods, response_model name, status_code,
tags, route.name, recursive dep qualnames)`` tuples — order-insensitive but
otherwise byte-identical.

Order that DOES matter is still asserted strictly:

* ``app.user_middleware`` class names in order (Starlette applies them
  outermost-first, so order is load-bearing) — NOT reordered by the split;
* the exception-handler key set;
* the ``dependency_overrides`` key set (auth / caller-type / billing-gate seams).

Plus structural invariants: no duplicate ``(path, method)`` and no two distinct
path templates that collapse to the same routing shape.

The golden is hardcoded from the current output (not a syrupy snapshot) so a
diff is reviewable in the PR without a ``--snapshot-update`` step. Regenerate by
running ``create_app()`` and dumping the same tuples if an *intended* API change
lands.
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from reflexio.server.api import create_app

# ── Golden: create_app() (full, data-plane mounted) ─────────────────────
FULL_MIDDLEWARE = [
    "BaseHTTPMiddleware",
    "CorrelationIdMiddleware",
    "BotProtectionMiddleware",
    "TimeoutMiddleware",
    "SecurityHeadersMiddleware",
    "BodySizeLimitMiddleware",
    "CORSMiddleware",
]
FULL_EXC = [
    "<class 'fastapi.exceptions.RequestValidationError'>",
    "<class 'fastapi.exceptions.WebSocketRequestValidationError'>",
    "<class 'slowapi.errors.RateLimitExceeded'>",
    "<class 'starlette.exceptions.HTTPException'>",
]
FULL_OVERRIDES: list[str] = []
FULL_ROUTES = [
    (
        "/meta/version",
        ("GET",),
        "dict",
        None,
        (),
        "get_version_info",
        (),
    ),
    (
        "/",
        ("GET",),
        "dict",
        None,
        (),
        "root",
        (),
    ),
    (
        "/health",
        ("GET",),
        # response_model is None: the handler may return a JSONResponse (503,
        # not-ready) when the in-process warm gate is active, so the route
        # cannot declare a Pydantic response model. The 200 body is unchanged.
        None,
        None,
        (),
        "health_check",
        (),
    ),
    (
        "/api/whoami",
        ("GET",),
        "WhoamiResponse",
        None,
        (),
        "whoami_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/my_config",
        ("GET",),
        "MyConfigResponse",
        None,
        (),
        "my_config_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/publish_interaction",
        ("POST",),
        "PublishUserInteractionResponse",
        None,
        (),
        "publish_user_interaction",
        ("default_get_org_id", "default_billing_gate.<locals>._noop"),
    ),
    (
        "/api/session_outcome",
        ("POST",),
        "SetSessionOutcomeResponse",
        None,
        (),
        "set_session_outcome",
        ("default_get_org_id",),
    ),
    (
        "/api/get_session_outcomes",
        ("POST",),
        "GetSessionOutcomesResponse",
        None,
        (),
        "get_session_outcomes",
        ("default_get_org_id",),
    ),
    (
        "/api/review_user_playbooks",
        ("POST",),
        "ReviewUserPlaybooksResponse",
        None,
        (),
        "review_user_playbooks_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/add_user_playbook",
        ("POST",),
        "AddUserPlaybookResponse",
        None,
        (),
        "add_user_playbook_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/add_agent_playbook",
        ("POST",),
        "AddAgentPlaybookResponse",
        None,
        (),
        "add_agent_playbook_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/add_user_profile",
        ("POST",),
        "AddUserProfileResponse",
        None,
        (),
        "add_user_profile_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/search_profiles",
        ("POST",),
        "SearchProfilesViewResponse",
        None,
        (),
        "search_user_profiles",
        (
            "default_get_org_id",
            "default_get_caller_type",
            "default_billing_gate.<locals>._noop",
        ),
    ),
    (
        "/api/rerank_user_profiles",
        ("POST",),
        "SearchProfilesViewResponse",
        None,
        (),
        "rerank_user_profiles",
        ("default_get_org_id",),
    ),
    (
        "/api/storage_stats",
        ("GET",),
        "StorageStatsResponse",
        None,
        (),
        "storage_stats",
        ("default_get_org_id",),
    ),
    (
        "/api/search_interactions",
        ("POST",),
        "SearchInteractionsViewResponse",
        None,
        (),
        "search_interactions",
        ("default_get_org_id",),
    ),
    (
        "/api/search_user_playbooks",
        ("POST",),
        "SearchUserPlaybooksViewResponse",
        None,
        (),
        "search_user_playbooks_endpoint",
        (
            "default_get_org_id",
            "default_get_caller_type",
            "default_billing_gate.<locals>._noop",
        ),
    ),
    (
        "/api/search_agent_playbooks",
        ("POST",),
        "SearchAgentPlaybooksViewResponse",
        None,
        (),
        "search_agent_playbooks_endpoint",
        (
            "default_get_org_id",
            "default_get_caller_type",
            "default_billing_gate.<locals>._noop",
        ),
    ),
    (
        "/api/search",
        ("POST",),
        "UnifiedSearchViewResponse",
        None,
        (),
        "unified_search_endpoint",
        (
            "default_get_org_id",
            "default_get_caller_type",
            "default_billing_gate.<locals>._noop",
            "_stamp_search_dependencies_done",
        ),
    ),
    (
        "/api/retrieval_experiments",
        ("GET",),
        "RetrievalExperimentListResponse",
        None,
        (),
        "list_retrieval_experiments",
        ("default_get_org_id",),
    ),
    (
        "/api/retrieval_experiments",
        ("POST",),
        "RetrievalExperimentListResponse",
        None,
        (),
        "start_retrieval_experiment",
        ("default_get_org_id",),
    ),
    (
        "/api/retrieval_experiments/stop",
        ("POST",),
        "RetrievalExperimentListResponse",
        None,
        (),
        "stop_retrieval_experiment",
        ("default_get_org_id",),
    ),
    (
        "/api/retrieval_experiments/{experiment_id}/results",
        ("GET",),
        "RetrievalExperimentResultsResponse",
        None,
        (),
        "get_retrieval_experiment_results",
        ("default_get_org_id",),
    ),
    (
        "/api/profile_change_log",
        ("GET",),
        "ProfileChangeLogViewResponse",
        None,
        (),
        "get_profile_change_log",
        ("default_get_org_id",),
    ),
    (
        "/api/playbook_aggregation_change_logs",
        ("GET",),
        "PlaybookAggregationChangeLogResponse",
        None,
        (),
        "get_playbook_aggregation_change_logs",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_profile",
        ("DELETE",),
        "DeleteUserProfileResponse",
        None,
        (),
        "delete_profile",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_interaction",
        ("DELETE",),
        "DeleteUserInteractionResponse",
        None,
        (),
        "delete_interaction",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_request",
        ("DELETE",),
        "DeleteRequestResponse",
        None,
        (),
        "delete_request",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_session",
        ("DELETE",),
        "DeleteSessionResponse",
        None,
        (),
        "delete_session",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_agent_playbook",
        ("DELETE",),
        "DeleteAgentPlaybookResponse",
        None,
        (),
        "delete_agent_playbook",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_user_playbook",
        ("DELETE",),
        "DeleteUserPlaybookResponse",
        None,
        (),
        "delete_user_playbook",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_requests_by_ids",
        ("DELETE",),
        "BulkDeleteResponse",
        None,
        (),
        "delete_requests_by_ids",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_profiles_by_ids",
        ("DELETE",),
        "BulkDeleteResponse",
        None,
        (),
        "delete_profiles_by_ids",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_agent_playbooks_by_ids",
        ("DELETE",),
        "BulkDeleteResponse",
        None,
        (),
        "delete_agent_playbooks_by_ids",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_user_playbooks_by_ids",
        ("DELETE",),
        "BulkDeleteResponse",
        None,
        (),
        "delete_user_playbooks_by_ids",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_all_interactions",
        ("DELETE",),
        "BulkDeleteResponse",
        None,
        (),
        "delete_all_interactions",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_all_profiles",
        ("DELETE",),
        "BulkDeleteResponse",
        None,
        (),
        "delete_all_profiles",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_all_playbooks",
        ("DELETE",),
        "BulkDeleteResponse",
        None,
        (),
        "delete_all_playbooks",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_all_user_playbooks",
        ("DELETE",),
        "BulkDeleteResponse",
        None,
        (),
        "delete_all_user_playbooks",
        ("default_get_org_id",),
    ),
    (
        "/api/delete_all_agent_playbooks",
        ("DELETE",),
        "BulkDeleteResponse",
        None,
        (),
        "delete_all_agent_playbooks",
        ("default_get_org_id",),
    ),
    (
        "/api/clear_user_data",
        ("POST",),
        "ClearUserDataResponse",
        None,
        (),
        "clear_user_data",
        ("default_get_org_id",),
    ),
    (
        "/api/get_interactions",
        ("POST",),
        "GetInteractionsViewResponse",
        None,
        (),
        "get_interactions",
        ("default_get_org_id",),
    ),
    (
        "/api/get_all_interactions",
        ("GET",),
        "GetInteractionsViewResponse",
        None,
        (),
        "get_all_interactions",
        ("default_get_org_id",),
    ),
    (
        "/api/learning_status",
        ("GET",),
        "LearningStatusResponse",
        None,
        (),
        "get_learning_status",
        ("default_get_org_id",),
    ),
    (
        "/api/get_requests",
        ("POST",),
        "GetRequestsViewResponse",
        None,
        (),
        "get_requests_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/get_learning_provenance",
        ("POST",),
        "LearningProvenanceViewResponse",
        None,
        (),
        "get_learning_provenance",
        ("default_get_org_id",),
    ),
    (
        "/api/get_profiles",
        ("POST",),
        "GetProfilesViewResponse",
        None,
        (),
        "get_profiles",
        ("default_get_org_id",),
    ),
    (
        "/api/get_all_profiles",
        ("GET",),
        "GetProfilesViewResponse",
        None,
        (),
        "get_all_profiles",
        ("default_get_org_id",),
    ),
    (
        "/api/get_profile_statistics",
        ("GET",),
        "GetProfileStatisticsResponse",
        None,
        (),
        "get_profile_statistics",
        ("default_get_org_id",),
    ),
    (
        "/api/run_playbook_aggregation",
        ("POST",),
        "RunPlaybookAggregationResponse",
        None,
        (),
        "run_playbook_aggregation",
        ("default_get_org_id",),
    ),
    (
        "/api/set_config",
        ("POST",),
        "SetConfigResponse",
        None,
        (),
        "set_config",
        ("default_get_org_id",),
    ),
    (
        "/api/update_config",
        ("POST",),
        "SetConfigResponse",
        None,
        (),
        "update_config",
        ("default_get_org_id",),
    ),
    (
        "/api/admin/cache/invalidate",
        ("POST",),
        "AdminInvalidateCacheResponse",
        None,
        (),
        "admin_invalidate_cache",
        ("default_get_org_id",),
    ),
    (
        "/api/get_config",
        ("GET",),
        "dict",
        None,
        (),
        "get_config",
        ("default_get_org_id",),
    ),
    (
        "/api/get_user_playbooks",
        ("POST",),
        "GetUserPlaybooksViewResponse",
        None,
        (),
        "get_user_playbooks",
        ("default_get_org_id",),
    ),
    (
        "/api/get_agent_playbooks",
        ("POST",),
        "GetAgentPlaybooksViewResponse",
        None,
        (),
        "get_agent_playbooks",
        (
            "default_get_org_id",
            "default_get_caller_type",
            "default_billing_gate.<locals>._noop",
        ),
    ),
    (
        "/api/get_agent_success_evaluation_results",
        ("POST",),
        "GetEvaluationResultsViewResponse",
        None,
        (),
        "get_agent_success_evaluation_results",
        ("default_get_org_id",),
    ),
    (
        "/api/get_retrieved_learning_evaluation_results",
        ("POST",),
        "GetRetrievedLearningEvaluationResultsResponse",
        None,
        (),
        "get_retrieved_learning_evaluation_results",
        ("default_get_org_id",),
    ),
    (
        "/api/update_agent_playbook_status",
        ("PUT",),
        "UpdatePlaybookStatusResponse",
        None,
        (),
        "update_agent_playbook_status_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/update_agent_playbook",
        ("PUT",),
        "UpdateAgentPlaybookResponse",
        None,
        (),
        "update_agent_playbook_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/update_user_playbook",
        ("PUT",),
        "UpdateUserPlaybookResponse",
        None,
        (),
        "update_user_playbook_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/update_user_profile",
        ("PUT",),
        "UpdateUserProfileResponse",
        None,
        (),
        "update_user_profile_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/get_dashboard_stats",
        ("POST",),
        "GetDashboardStatsResponse",
        None,
        (),
        "get_dashboard_stats",
        ("default_get_org_id",),
    ),
    (
        "/api/get_playbook_application_stats",
        ("POST",),
        "GetPlaybookApplicationStatsResponse",
        None,
        (),
        "get_playbook_application_stats",
        ("default_get_org_id",),
    ),
    (
        "/api/braintrust/connect",
        ("POST",),
        "ConnectBraintrustResponse",
        None,
        (),
        "braintrust_connect",
        ("default_get_org_id",),
    ),
    (
        "/api/braintrust/select_projects",
        ("POST",),
        "SelectProjectsResponse",
        None,
        (),
        "braintrust_select_projects",
        ("default_get_org_id",),
    ),
    (
        "/api/braintrust/status",
        ("GET",),
        "BraintrustStatusResponse",
        None,
        (),
        "braintrust_status",
        ("default_get_org_id",),
    ),
    (
        "/api/braintrust/connection",
        ("DELETE",),
        "dict",
        None,
        (),
        "braintrust_disconnect",
        ("default_get_org_id",),
    ),
    (
        "/api/braintrust/sync",
        ("POST",),
        "SyncBraintrustResponse",
        None,
        (),
        "braintrust_sync",
        ("default_get_org_id",),
    ),
    (
        "/api/get_evaluation_overview",
        ("POST",),
        "GetEvaluationOverviewResponse",
        None,
        (),
        "get_evaluation_overview",
        ("default_get_org_id",),
    ),
    (
        "/api/evaluations/regenerate",
        ("POST",),
        "RegenerateStartResponse",
        None,
        (),
        "start_regenerate",
        ("default_get_org_id",),
    ),
    (
        "/api/evaluations/regenerate/{job_id}",
        ("GET",),
        "RegenerateStatusResponse",
        None,
        (),
        "get_regenerate_status",
        ("default_get_org_id",),
    ),
    (
        "/api/evaluations/regenerate/{job_id}",
        ("DELETE",),
        "dict",
        None,
        (),
        "cancel_regenerate",
        ("default_get_org_id",),
    ),
    (
        "/api/evaluations/grade_on_demand",
        ("POST",),
        "GradeOnDemandResponse",
        None,
        (),
        "grade_on_demand",
        ("default_get_org_id",),
    ),
    (
        "/api/evaluations/shadow_comparisons/recent",
        ("GET",),
        "GetRecentShadowComparisonsResponse",
        None,
        (),
        "get_recent_shadow_comparisons",
        ("default_get_org_id",),
    ),
    (
        "/api/rerun_profile_generation",
        ("POST",),
        "RerunProfileGenerationResponse",
        None,
        (),
        "rerun_profile_generation_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/manual_profile_generation",
        ("POST",),
        "ManualProfileGenerationResponse",
        None,
        (),
        "manual_profile_generation_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/rerun_playbook_generation",
        ("POST",),
        "RerunPlaybookGenerationResponse",
        None,
        (),
        "rerun_playbook_generation_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/manual_playbook_generation",
        ("POST",),
        "ManualPlaybookGenerationResponse",
        None,
        (),
        "manual_playbook_generation_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/upgrade_all_profiles",
        ("POST",),
        "UpgradeProfilesResponse",
        None,
        (),
        "upgrade_all_profiles_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/downgrade_all_profiles",
        ("POST",),
        "DowngradeProfilesResponse",
        None,
        (),
        "downgrade_all_profiles_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/upgrade_all_user_playbooks",
        ("POST",),
        "UpgradeUserPlaybooksResponse",
        None,
        (),
        "upgrade_all_user_playbooks_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/downgrade_all_user_playbooks",
        ("POST",),
        "DowngradeUserPlaybooksResponse",
        None,
        (),
        "downgrade_all_user_playbooks_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/get_operation_status",
        ("GET",),
        "GetOperationStatusResponse",
        None,
        (),
        "get_operation_status_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/api/cancel_operation",
        ("POST",),
        "CancelOperationResponse",
        None,
        (),
        "cancel_operation_endpoint",
        ("default_get_org_id",),
    ),
    (
        "/stall_state",
        ("GET",),
        "StallStateResponse",
        None,
        ("stall_state",),
        "read_stall_state",
        ("get_request_context", "default_get_org_id"),
    ),
    (
        "/stall_state/notified",
        ("POST",),
        "MarkNotifiedResponse",
        None,
        ("stall_state",),
        "post_notified",
        ("get_request_context", "default_get_org_id"),
    ),
    (
        "/api/pending_tool_calls",
        ("GET",),
        "PendingToolCallListResponse",
        None,
        ("pending_tool_calls",),
        "list_pending_tool_calls",
        ("get_request_context", "default_get_org_id"),
    ),
    (
        "/api/pending_tool_calls/{pending_tool_call_id}",
        ("GET",),
        "PendingToolCallResponse",
        None,
        ("pending_tool_calls",),
        "get_pending_tool_call",
        ("get_request_context", "default_get_org_id"),
    ),
    (
        "/api/pending_tool_calls/{pending_tool_call_id}/resolve",
        ("POST",),
        "PendingToolCallResponse",
        None,
        ("pending_tool_calls",),
        "resolve_pending_tool_call",
        ("get_request_context", "default_get_org_id"),
    ),
    (
        "/api/pending_tool_calls/{pending_tool_call_id}/answer",
        ("PATCH",),
        "PendingToolCallResponse",
        None,
        ("pending_tool_calls",),
        "update_pending_tool_call_answer",
        ("get_request_context", "default_get_org_id"),
    ),
    (
        "/api/pending_tool_calls/{pending_tool_call_id}/not_applicable",
        ("POST",),
        "PendingToolCallResponse",
        None,
        ("pending_tool_calls",),
        "mark_pending_tool_call_not_applicable",
        ("get_request_context", "default_get_org_id"),
    ),
    (
        "/api/pending_tool_calls/{pending_tool_call_id}/cancel",
        ("POST",),
        "PendingToolCallResponse",
        None,
        ("pending_tool_calls",),
        "cancel_pending_tool_call",
        ("get_request_context", "default_get_org_id"),
    ),
    (
        "/healthz",
        ("GET",),
        "dict",
        None,
        (),
        "healthz",
        (),
    ),
    (
        "/healthz/eval",
        ("GET",),
        "dict",
        None,
        (),
        "healthz_eval",
        (),
    ),
]

# ── Golden: create_app(mount_data_plane=False) (control-plane scaffolding only) ──
NODP_MIDDLEWARE = list(FULL_MIDDLEWARE)
NODP_EXC = list(FULL_EXC)
NODP_OVERRIDES: list[str] = []
NODP_ROUTES = [
    (
        "/meta/version",
        ("GET",),
        "dict",
        None,
        (),
        "get_version_info",
        (),
    ),
    (
        "/healthz",
        ("GET",),
        "dict",
        None,
        (),
        "healthz",
        (),
    ),
    (
        "/healthz/eval",
        ("GET",),
        "dict",
        None,
        (),
        "healthz_eval",
        (),
    ),
]

_PARAM = re.compile(r"\{[^}]+\}")


def _dep_qualnames(dependant) -> list[str]:
    """Return ordered, recursive dependency call qualnames for a route dependant.

    Args:
        dependant: A FastAPI ``Dependant`` (``route.dependant`` or a sub-dep).

    Returns:
        list[str]: Ordered ``__qualname__``s of every dependency callable.
    """
    names: list[str] = []
    for dep in dependant.dependencies:
        call = dep.call
        if call is not None:
            names.append(
                getattr(call, "__qualname__", getattr(call, "__name__", repr(call)))
            )
        names.extend(_dep_qualnames(dep))
    return names


def _route_inventory(app: FastAPI) -> list[tuple]:
    """Snapshot every ``APIRoute`` on ``app`` as a tuple inventory."""
    return [
        (
            route.path,
            tuple(sorted(route.methods)),
            route.response_model.__name__ if route.response_model else None,
            route.status_code,
            tuple(route.tags),
            route.name,
            tuple(_dep_qualnames(route.dependant)),
        )
        for route in app.routes
        if isinstance(route, APIRoute)
    ]


def _middleware_names(app: FastAPI) -> list[str]:
    return [
        getattr(m.cls, "__name__", type(m.cls).__name__) for m in app.user_middleware
    ]


@pytest.mark.parametrize(
    ("build", "routes", "middleware", "exc", "overrides"),
    [
        pytest.param(
            lambda: create_app(),
            FULL_ROUTES,
            FULL_MIDDLEWARE,
            FULL_EXC,
            FULL_OVERRIDES,
            id="full",
        ),
        pytest.param(
            lambda: create_app(mount_data_plane=False),
            NODP_ROUTES,
            NODP_MIDDLEWARE,
            NODP_EXC,
            NODP_OVERRIDES,
            id="mount_data_plane_false",
        ),
    ],
)
def test_oss_app_shape_matches_golden(
    build, routes, middleware, exc, overrides
) -> None:
    """The full app shape is pinned: routes as a SET (order-insensitive — see
    module docstring), middleware/handlers/overrides strictly."""
    app = build()
    assert set(_route_inventory(app)) == {tuple(r) for r in routes}
    assert _middleware_names(app) == middleware
    assert sorted(str(k) for k in app.exception_handlers) == exc
    assert sorted(str(k) for k in app.dependency_overrides) == overrides


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: create_app(), id="full"),
        pytest.param(
            lambda: create_app(mount_data_plane=False), id="mount_data_plane_false"
        ),
    ],
)
def test_no_duplicate_path_method(build) -> None:
    """No ``(path, method)`` pair is registered twice — a duplicate would make
    the second registration unreachable."""
    app = build()
    seen: set[tuple[str, str]] = set()
    dups: list[tuple[str, str]] = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            for method in route.methods:
                key = (route.path, method)
                if key in seen:
                    dups.append(key)
                seen.add(key)
    assert not dups, f"duplicate (path, method) registrations: {dups}"


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: create_app(), id="full"),
        pytest.param(
            lambda: create_app(mount_data_plane=False), id="mount_data_plane_false"
        ),
    ],
)
def test_no_overlapping_path_templates(build) -> None:
    """No two *distinct* raw path templates collapse to the same routing shape
    for the same method (e.g. ``/x/{a}`` vs ``/x/{b}``) — such a pair is an
    ambiguous match FastAPI would resolve by declaration order. Proving this
    absent is what licenses the set-based (order-insensitive) route golden."""
    app = build()
    by_shape: dict[tuple[str, str], set[str]] = {}
    for route in app.routes:
        if isinstance(route, APIRoute):
            normalized = _PARAM.sub("{}", route.path)
            for method in route.methods:
                by_shape.setdefault((normalized, method), set()).add(route.path)
    overlaps = {k: v for k, v in by_shape.items() if len(v) > 1}
    assert not overlaps, f"overlapping path templates: {overlaps}"
