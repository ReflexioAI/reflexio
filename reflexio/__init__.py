"""Public exports loaded on demand so inference stays independent of app code."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("reflexio-ai")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

debug = False
log = None

if TYPE_CHECKING:
    from reflexio.models.api_schema.eval_overview_schema import (
        GradeOnDemandRequest,
        GradeOnDemandResponse,
        RegenerateRequest,
        RegenerateStartResponse,
        RegenerateStatusResponse,
    )
    from reflexio.models.api_schema.retriever_schema import (
        ConversationTurn,
        GetAgentPlaybooksViewResponse,
        GetEvaluationResultsViewResponse,
        GetInteractionsViewResponse,
        GetProfilesViewResponse,
        GetRequestsViewResponse,
        GetUserPlaybooksViewResponse,
        ProfileChangeLogViewResponse,
        RequestDataView,
        SearchAgentPlaybooksViewResponse,
        SearchInteractionRequest,
        SearchInteractionResponse,
        SearchInteractionsViewResponse,
        SearchProfilesViewResponse,
        SearchUserPlaybooksViewResponse,
        SearchUserProfileRequest,
        SearchUserProfileResponse,
        SessionView,
        UnifiedSearchViewResponse,
    )
    from reflexio.models.api_schema.service_schemas import (
        AddUserPlaybookRequest,
        AddUserPlaybookResponse,
        AgentPlaybook,
        BlockingIssue,
        BlockingIssueKind,
        BulkDeleteResponse,
        Citation,
        DeleteAgentPlaybooksByIdsRequest,
        DeleteProfilesByIdsRequest,
        DeleteRequestsByIdsRequest,
        DeleteUserInteractionRequest,
        DeleteUserInteractionResponse,
        DeleteUserPlaybooksByIdsRequest,
        DeleteUserProfileRequest,
        DeleteUserProfileResponse,
        Interaction,
        InteractionData,
        PlaybookStatus,
        ProfileTimeToLive,
        PublishUserInteractionRequest,
        PublishUserInteractionResponse,
        RerunPlaybookGenerationRequest,
        RerunPlaybookGenerationResponse,
        RerunProfileGenerationRequest,
        RerunProfileGenerationResponse,
        Status,
        ToolUsed,
        UserActionType,
        UserPlaybook,
        UserProfile,
    )
    from reflexio.models.api_schema.ui.entities import (
        AgentPlaybookView,
        EvaluationResultView,
        InteractionView,
        ProfileChangeLogView,
        ProfileView,
        UserPlaybookView,
    )
    from reflexio.models.config_schema import (
        AgentSuccessConfig,
        Config,
        PlaybookAggregatorConfig,
        PlaybookConfig,
        ProfileExtractorConfig,
        StorageConfig,
        StorageConfigPostgres,
        StorageConfigSQLite,
        StorageConfigSupabase,
        StorageConfigTest,
        ToolUseConfig,
        UserPlaybookExtractorConfig,
    )

    from .client import ReflexioClient, SessionScopedClient

_EXPORTS = {
    name: module
    for module, names in {
        "reflexio.models.api_schema.eval_overview_schema": (
            "GradeOnDemandRequest",
            "GradeOnDemandResponse",
            "RegenerateRequest",
            "RegenerateStartResponse",
            "RegenerateStatusResponse",
        ),
        "reflexio.models.api_schema.retriever_schema": (
            "ConversationTurn",
            "GetAgentPlaybooksViewResponse",
            "GetEvaluationResultsViewResponse",
            "GetInteractionsViewResponse",
            "GetProfilesViewResponse",
            "GetRequestsViewResponse",
            "GetUserPlaybooksViewResponse",
            "ProfileChangeLogViewResponse",
            "RequestDataView",
            "SearchAgentPlaybooksViewResponse",
            "SearchInteractionRequest",
            "SearchInteractionResponse",
            "SearchInteractionsViewResponse",
            "SearchProfilesViewResponse",
            "SearchUserPlaybooksViewResponse",
            "SearchUserProfileRequest",
            "SearchUserProfileResponse",
            "SessionView",
            "UnifiedSearchViewResponse",
        ),
        "reflexio.models.api_schema.service_schemas": (
            "AddUserPlaybookRequest",
            "AddUserPlaybookResponse",
            "AgentPlaybook",
            "BlockingIssue",
            "BlockingIssueKind",
            "BulkDeleteResponse",
            "Citation",
            "DeleteAgentPlaybooksByIdsRequest",
            "DeleteProfilesByIdsRequest",
            "DeleteRequestsByIdsRequest",
            "DeleteUserInteractionRequest",
            "DeleteUserInteractionResponse",
            "DeleteUserPlaybooksByIdsRequest",
            "DeleteUserProfileRequest",
            "DeleteUserProfileResponse",
            "Interaction",
            "InteractionData",
            "PlaybookStatus",
            "ProfileTimeToLive",
            "PublishUserInteractionRequest",
            "PublishUserInteractionResponse",
            "RerunPlaybookGenerationRequest",
            "RerunPlaybookGenerationResponse",
            "RerunProfileGenerationRequest",
            "RerunProfileGenerationResponse",
            "Status",
            "ToolUsed",
            "UserActionType",
            "UserPlaybook",
            "UserProfile",
        ),
        "reflexio.models.api_schema.ui.entities": (
            "AgentPlaybookView",
            "EvaluationResultView",
            "InteractionView",
            "ProfileChangeLogView",
            "ProfileView",
            "UserPlaybookView",
        ),
        "reflexio.models.config_schema": (
            "AgentSuccessConfig",
            "Config",
            "PlaybookAggregatorConfig",
            "PlaybookConfig",
            "ProfileExtractorConfig",
            "StorageConfig",
            "StorageConfigPostgres",
            "StorageConfigSQLite",
            "StorageConfigSupabase",
            "StorageConfigTest",
            "ToolUseConfig",
            "UserPlaybookExtractorConfig",
        ),
        "reflexio.client": ("ReflexioClient", "SessionScopedClient"),
    }.items()
    for name in names
}

__all__ = [
    "__version__",
    "ReflexioClient",
    "SessionScopedClient",
    # Data models (internal, with embeddings)
    "UserActionType",
    "ProfileTimeToLive",
    "InteractionData",
    "Interaction",
    "UserProfile",
    "AgentPlaybook",
    "UserPlaybook",
    "BlockingIssue",
    "BlockingIssueKind",
    "ToolUsed",
    "Citation",
    "Status",
    "PlaybookStatus",
    # View models (user-facing, without embeddings)
    "InteractionView",
    "ProfileView",
    "UserPlaybookView",
    "AgentPlaybookView",
    "EvaluationResultView",
    "ProfileChangeLogView",
    # Request types
    "PublishUserInteractionRequest",
    "DeleteUserProfileRequest",
    "DeleteUserInteractionRequest",
    "AddUserPlaybookRequest",
    "RerunProfileGenerationRequest",
    "RerunPlaybookGenerationRequest",
    "ConversationTurn",
    "SearchInteractionRequest",
    "SearchUserProfileRequest",
    "DeleteRequestsByIdsRequest",
    "DeleteProfilesByIdsRequest",
    "DeleteAgentPlaybooksByIdsRequest",
    "DeleteUserPlaybooksByIdsRequest",
    "GradeOnDemandRequest",
    "RegenerateRequest",
    # Response types (internal)
    "PublishUserInteractionResponse",
    "DeleteUserProfileResponse",
    "DeleteUserInteractionResponse",
    "AddUserPlaybookResponse",
    "RerunProfileGenerationResponse",
    "RerunPlaybookGenerationResponse",
    "SearchInteractionResponse",
    "SearchUserProfileResponse",
    "BulkDeleteResponse",
    "GradeOnDemandResponse",
    "RegenerateStartResponse",
    "RegenerateStatusResponse",
    # View response types (user-facing)
    "GetInteractionsViewResponse",
    "GetProfilesViewResponse",
    "SearchInteractionsViewResponse",
    "SearchProfilesViewResponse",
    "GetEvaluationResultsViewResponse",
    "ProfileChangeLogViewResponse",
    "RequestDataView",
    "SessionView",
    "GetRequestsViewResponse",
    "UnifiedSearchViewResponse",
    "GetUserPlaybooksViewResponse",
    "GetAgentPlaybooksViewResponse",
    "SearchUserPlaybooksViewResponse",
    "SearchAgentPlaybooksViewResponse",
    # Config types
    "StorageConfigTest",
    "StorageConfigSQLite",
    "StorageConfigSupabase",
    "StorageConfigPostgres",
    "StorageConfig",
    "ProfileExtractorConfig",
    "PlaybookAggregatorConfig",
    "PlaybookConfig",
    "UserPlaybookExtractorConfig",
    "AgentSuccessConfig",
    "ToolUseConfig",
    "Config",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
