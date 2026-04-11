import logging
from dataclasses import dataclass

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.config_schema import AgentSuccessConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    AgentSuccessEvaluationRequest,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluator import (
    AgentSuccessEvaluator,
)
from reflexio.server.services.base_generation_service import BaseGenerationService

logger = logging.getLogger(__name__)


@dataclass
class AgentSuccessGenerationServiceConfig:
    """Runtime configuration for agent success evaluation service shared across all extractors.

    Attributes:
        session_id: The session being evaluated
        agent_version: The agent version
        request_interaction_data_models: The interactions to evaluate
        source: Source of the interactions
    """

    session_id: str
    agent_version: str
    request_interaction_data_models: list[RequestInteractionDataModel]
    source: str | None = None


class AgentSuccessEvaluationService(
    BaseGenerationService[
        AgentSuccessConfig,
        AgentSuccessEvaluator,
        AgentSuccessGenerationServiceConfig,
        AgentSuccessEvaluationRequest,
    ]
):
    """
    Service for evaluating agent success across multiple evaluation criteria.
    Runs multiple AgentSuccessEvaluator instances sequentially.
    """

    def __init__(
        self, llm_client: LiteLLMClient, request_context: RequestContext
    ) -> None:
        """Initialize service and reset per-run outcome flags."""
        super().__init__(llm_client=llm_client, request_context=request_context)
        self.last_run_result_count = 0
        self.last_run_saved_result_count = 0
        self.last_run_save_failed = False

    def run(self, request: AgentSuccessEvaluationRequest) -> None:
        """Run evaluation and reset run outcome flags for this invocation."""
        self.last_run_result_count = 0
        self.last_run_saved_result_count = 0
        self.last_run_save_failed = False
        super().run(request)

    def _load_generation_service_config(
        self, request: AgentSuccessEvaluationRequest
    ) -> AgentSuccessGenerationServiceConfig:
        """
        Extract request parameters from AgentSuccessEvaluationRequest.

        Args:
            request: AgentSuccessEvaluationRequest containing evaluation parameters

        Returns:
            AgentSuccessGenerationServiceConfig object
        """
        return AgentSuccessGenerationServiceConfig(
            session_id=request.session_id,
            agent_version=request.agent_version,
            request_interaction_data_models=request.request_interaction_data_models,
            source=request.source,
        )

    def _load_extractor_configs(self) -> list[AgentSuccessConfig]:
        """
        Load agent success configs from configurator.

        Returns:
            list[AgentSuccessConfig]: List of agent success configuration objects from YAML
        """
        return self.configurator.get_config().agent_success_configs  # type: ignore[reportReturnType]

    def _create_extractor(
        self,
        extractor_config: AgentSuccessConfig,
        service_config: AgentSuccessGenerationServiceConfig,
    ) -> AgentSuccessEvaluator:
        """
        Create an AgentSuccessEvaluator instance from configuration.

        Args:
            extractor_config: AgentSuccessConfig configuration object from YAML
            service_config: AgentSuccessGenerationServiceConfig containing runtime parameters

        Returns:
            AgentSuccessEvaluator instance
        """
        return AgentSuccessEvaluator(
            request_context=self.request_context,
            llm_client=self.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=self.configurator.get_agent_context(),
        )

    def _process_results(self, results: list) -> None:
        """
        Process and save agent success evaluation results.

        Args:
            results: List of AgentSuccessEvaluationResult results from extractors
        """
        # Flatten results (each extractor returns list[AgentSuccessEvaluationResult])
        all_results = []
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
        self.last_run_result_count = len(all_results)

        logger.info(
            "Successfully completed %d %s evaluations for session: %s",
            len(all_results),
            self._get_service_name(),
            self.service_config.session_id,  # type: ignore[reportOptionalMemberAccess]
        )

        # Save results
        if all_results:
            try:
                self.storage.save_agent_success_evaluation_results(all_results)  # type: ignore[reportOptionalMemberAccess]
                self.last_run_saved_result_count = len(all_results)
                logger.info(
                    "Saved %d agent success evaluation results for session: %s",
                    len(all_results),
                    self.service_config.session_id,  # type: ignore[reportOptionalMemberAccess]
                )
            except Exception as e:
                self.last_run_save_failed = True
                logger.error(
                    "Failed to save %s results for session: %s due to %s, exception type: %s",
                    self._get_service_name(),
                    self.service_config.session_id,  # type: ignore[reportOptionalMemberAccess]
                    str(e),
                    type(e).__name__,
                )

    def has_run_failures(self) -> bool:
        """Return True if extractor execution or result persistence failed."""
        extractor_failed_count = self._last_extractor_run_stats.get("failed", 0)
        return extractor_failed_count > 0 or self.last_run_save_failed

    def _get_service_name(self) -> str:
        """
        Get the name of the service for logging.

        Returns:
            Service name string
        """
        return "agent_success_evaluation"

    def _get_base_service_name(self) -> str:
        """
        Get the base service name for OperationStateManager keys.

        Returns:
            str: "agent_success_evaluation"
        """
        return "agent_success_evaluation"

    def _should_track_in_progress(self) -> bool:
        """
        Agent success evaluation does NOT track in-progress state.

        Agent success evaluation is tied to specific requests and doesn't have
        the sliding window duplication issue that profile/feedback have.

        Returns:
            bool: False - agent success evaluation does not track in-progress state
        """
        return False

    def _get_lock_scope_id(self, request: AgentSuccessEvaluationRequest) -> str | None:
        """
        Not used since _should_track_in_progress returns False.

        Args:
            request: The AgentSuccessEvaluationRequest

        Raises:
            NotImplementedError: This method is not used for this service
        """
        raise NotImplementedError(
            "AgentSuccessEvaluationService does not track in-progress state"
        )
