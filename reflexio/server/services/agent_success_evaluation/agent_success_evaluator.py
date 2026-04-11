import logging
import random
from typing import TYPE_CHECKING

# Roles considered as agent/system-side (not user turns) when counting user turns.
_AGENT_ROLES = {"agent", "assistant", "system", "tool", "internal"}

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    AgentSuccessEvaluationResult,
    RegularVsShadow,
)
from reflexio.models.config_schema import AgentSuccessConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_constants import (
    AgentSuccessEvaluationOutput,
    AgentSuccessEvaluationWithComparisonOutput,
)
from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_utils import (
    construct_agent_success_evaluation_messages_from_sessions,
    construct_agent_success_evaluation_with_comparison_messages,
    format_interactions_for_request,
    has_shadow_content,
)
from reflexio.server.services.extractor_interaction_utils import (
    filter_interactions_by_source,
    get_effective_source_filter,
)
from reflexio.server.services.service_utils import (
    extract_interactions_from_request_interaction_data_models,
    log_llm_messages,
    log_model_response,
)
from reflexio.server.site_var.site_var_manager import SiteVarManager

if TYPE_CHECKING:
    from reflexio.server.services.agent_success_evaluation.agent_success_evaluation_service import (
        AgentSuccessGenerationServiceConfig,
    )

logger = logging.getLogger(__name__)

"""
Extract playbooks from user interactions for developers to improve the agent on next iteration.
Identify missing features, tools, etc.
"""


class AgentSuccessEvaluator:
    """
    Evaluate agent success based on user interactions.

    This class analyzes agent-user interactions to determine if the agent
    successfully completed its task and identifies areas for improvement.
    """

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        extractor_config: AgentSuccessConfig,
        service_config: AgentSuccessGenerationServiceConfig,
        agent_context: str,
    ):
        """
        Initialize the agent success evaluator.

        Args:
            request_context: Request context with storage and prompt manager
            llm_client: Unified LLM client supporting both OpenAI and Claude
            extractor_config: Agent success evaluation configuration from YAML
            service_config: Runtime service configuration with request data
            agent_context: Context about the agent
        """
        self.request_context: RequestContext = request_context
        self.client: LiteLLMClient = llm_client
        self.config: AgentSuccessConfig = extractor_config
        self.service_config: AgentSuccessGenerationServiceConfig = service_config
        self.agent_context: str = agent_context

        # Get LLM config overrides from configuration
        config = self.request_context.configurator.get_config()
        llm_config = config.llm_config if config else None

        # Resolve model name: config override → site var → auto-detect
        model_setting = SiteVarManager().get_site_var("llm_model_setting")
        site_var = model_setting if isinstance(model_setting, dict) else {}
        api_key_config = self.request_context.configurator.get_config().api_key_config

        self.default_evaluate_model_name = resolve_model_name(
            ModelRole.EVALUATION,
            site_var_value=site_var.get("default_evaluate_model_name"),
            config_override=llm_config.generation_model_name if llm_config else None,
            api_key_config=api_key_config,
        )

    # ===============================
    # public methods
    # ===============================

    def run(self) -> list[AgentSuccessEvaluationResult]:
        """
        Evaluate agent success at the session level.

        Treats all request_interaction_data_models as a single conversation.
        Applies source filtering based on extractor config.
        Applies sampling rate once per group.

        Returns:
            List of AgentSuccessEvaluationResult objects (single result for the group)
        """
        # Get interactions from service config (required)
        request_interaction_data_models = (
            self.service_config.request_interaction_data_models
        )

        # Filter by source based on extractor config
        should_skip, source_filter = get_effective_source_filter(
            self.config,
            self.service_config.source,
        )
        if should_skip:
            return []

        request_interaction_data_models = filter_interactions_by_source(
            request_interaction_data_models,
            source_filter,
        )
        if not request_interaction_data_models:
            # No matching interactions after source filter
            return []

        # Check sampling rate once per group
        if self.config.sampling_rate < 1.0:
            random_value = random.random()  # noqa: S311
            if random_value >= self.config.sampling_rate:
                logger.info(
                    "Skipping evaluation for session %s due to sampling rate. "
                    "sampling_rate=%s, random_value=%.3f",
                    self.service_config.session_id,
                    self.config.sampling_rate,
                    random_value,
                )
                return []

        result = self._evaluate_group(request_interaction_data_models)
        return [result] if result else []

    def _evaluate_group(
        self, request_interaction_data_models: list[RequestInteractionDataModel]
    ) -> AgentSuccessEvaluationResult | None:
        """
        Evaluate agent success for the entire session.

        If interactions contain shadow_content, uses a combined prompt that:
        1. Evaluates the regular version for success
        2. Compares regular vs shadow to determine which is better

        Args:
            request_interaction_data_models: All request interaction data models in the group

        Returns:
            Optional[AgentSuccessEvaluationResult]: Evaluation result or None if evaluation fails
        """
        # Read tool_can_use from root config
        root_config = self.request_context.configurator.get_config()
        tool_can_use_str = ""
        if root_config and root_config.tool_can_use:
            tool_can_use_str = "\n".join(
                [
                    f"{tool.tool_name}: {tool.tool_description}"
                    for tool in root_config.tool_can_use
                ]
            )

        # Flatten all interactions to check for shadow content
        all_interactions = extract_interactions_from_request_interaction_data_models(
            request_interaction_data_models
        )

        # Check if any interaction has shadow content
        if has_shadow_content(all_interactions):
            return self._evaluate_with_shadow_comparison(
                request_interaction_data_models,
                tool_can_use_str,
            )

        # No shadow content - use existing evaluation prompt
        return self._evaluate_regular(
            request_interaction_data_models,
            tool_can_use_str,
        )

    def _evaluate_regular(
        self,
        request_interaction_data_models: list[RequestInteractionDataModel],
        tool_can_use_str: str,
    ) -> AgentSuccessEvaluationResult | None:
        """
        Evaluate agent success for the group without shadow comparison.

        Args:
            request_interaction_data_models: All request interaction data models in the group
            tool_can_use_str: Formatted string of available tools

        Returns:
            Optional[AgentSuccessEvaluationResult]: Evaluation result or None if evaluation fails
        """
        messages = construct_agent_success_evaluation_messages_from_sessions(
            prompt_manager=self.request_context.prompt_manager,
            request_interaction_data_models=request_interaction_data_models,
            agent_context_prompt=self.agent_context,
            success_definition_prompt=(
                self.config.success_definition_prompt.strip()
                if self.config.success_definition_prompt
                else ""
            ),
            tool_can_use=tool_can_use_str,
            metadata_definition_prompt=(
                self.config.metadata_definition_prompt.strip()
                if self.config.metadata_definition_prompt
                else None
            ),
        )
        messages_dict = messages

        session_request_count = len(request_interaction_data_models)
        interaction_count = sum(
            len(rdm.interactions) for rdm in request_interaction_data_models
        )
        logger.info(
            "event=agent_success_eval_llm_start session_id=%s evaluation_name=%s "
            "requests=%d interactions=%d model=%s",
            self.service_config.session_id,
            self.config.evaluation_name,
            session_request_count,
            interaction_count,
            self.default_evaluate_model_name,
        )
        log_llm_messages(logger, "Agent success evaluation", messages_dict)

        # Use Pydantic model for structured output
        evaluation_response = self.client.generate_chat_response(
            messages=messages_dict,
            model=self.default_evaluate_model_name,
            response_format=AgentSuccessEvaluationOutput,
        )
        if not evaluation_response:
            logger.info(
                "No evaluation can be generated for session %s",
                self.service_config.session_id,
            )
            return None

        log_model_response(
            logger, "Agent success evaluation response", evaluation_response
        )

        if not isinstance(evaluation_response, AgentSuccessEvaluationOutput):
            logger.warning(
                "Unexpected response type from evaluation LLM: %s",
                type(evaluation_response),
            )
            return None

        return self._build_evaluation_result(
            evaluation_response=evaluation_response,
            request_interaction_data_models=request_interaction_data_models,
        )

    def _evaluate_with_shadow_comparison(
        self,
        request_interaction_data_models: list[RequestInteractionDataModel],
        tool_can_use_str: str,
    ) -> AgentSuccessEvaluationResult | None:
        """
        Evaluate agent success with shadow content comparison at group level.

        Uses a combined prompt that:
        1. Evaluates the regular version for success
        2. Compares regular vs shadow to determine which is better

        The regular and shadow versions are randomly assigned to Request 1/Request 2
        to avoid LLM bias toward one position.

        Args:
            request_interaction_data_models: All request interaction data models in the group
            tool_can_use_str: Formatted string of available tools

        Returns:
            Optional[AgentSuccessEvaluationResult]: Evaluation result with regular_vs_shadow comparison
        """
        # Flatten all interactions from all request data models
        all_interactions = extract_interactions_from_request_interaction_data_models(
            request_interaction_data_models
        )

        # Randomly decide which is Request 1 vs Request 2 to avoid position bias
        regular_is_request_1 = random.choice([True, False])  # noqa: S311

        # Format interactions for regular and shadow versions
        regular_interactions = format_interactions_for_request(
            all_interactions, use_shadow=False
        )
        shadow_interactions = format_interactions_for_request(
            all_interactions, use_shadow=True
        )

        # Assign to Request 1 and Request 2 based on random choice
        if regular_is_request_1:
            request_1_interactions = regular_interactions
            request_2_interactions = shadow_interactions
        else:
            request_1_interactions = shadow_interactions
            request_2_interactions = regular_interactions

        logger.info(
            "Evaluating with shadow comparison. regular_is_request_1=%s",
            regular_is_request_1,
        )

        # Build combined prompt
        messages = construct_agent_success_evaluation_with_comparison_messages(
            prompt_manager=self.request_context.prompt_manager,
            request_1_interactions=request_1_interactions,
            request_2_interactions=request_2_interactions,
            agent_context_prompt=self.agent_context,
            success_definition_prompt=(
                self.config.success_definition_prompt.strip()
                if self.config.success_definition_prompt
                else ""
            ),
            tool_can_use=tool_can_use_str,
            metadata_definition_prompt=(
                self.config.metadata_definition_prompt.strip()
                if self.config.metadata_definition_prompt
                else None
            ),
            interactions_for_images=all_interactions,
        )

        messages_dict = messages

        session_request_count = len(request_interaction_data_models)
        interaction_count = sum(
            len(rdm.interactions) for rdm in request_interaction_data_models
        )
        logger.info(
            "event=agent_success_eval_comparison_llm_start session_id=%s evaluation_name=%s "
            "requests=%d interactions=%d model=%s regular_is_request_1=%s",
            self.service_config.session_id,
            self.config.evaluation_name,
            session_request_count,
            interaction_count,
            self.default_evaluate_model_name,
            regular_is_request_1,
        )
        log_llm_messages(
            logger, "Agent success evaluation with comparison", messages_dict
        )

        # Use Pydantic model for structured output
        evaluation_response = self.client.generate_chat_response(
            messages=messages_dict,
            model=self.default_evaluate_model_name,
            response_format=AgentSuccessEvaluationWithComparisonOutput,
        )

        if not evaluation_response:
            logger.info(
                "No evaluation can be generated for session %s",
                self.service_config.session_id,
            )
            return None

        log_model_response(
            logger,
            "Agent success evaluation with comparison response",
            evaluation_response,
        )

        if not isinstance(
            evaluation_response, AgentSuccessEvaluationWithComparisonOutput
        ):
            logger.warning(
                "Unexpected response type from evaluation LLM: %s",
                type(evaluation_response),
            )
            return None

        # Map comparison result to RegularVsShadow enum
        regular_vs_shadow = self._map_comparison_to_enum(
            better_request=evaluation_response.better_request or "tie",
            is_significantly_better=evaluation_response.is_significantly_better
            or False,
            regular_is_request_1=regular_is_request_1,
        )

        return self._build_evaluation_result(
            evaluation_response=evaluation_response,
            request_interaction_data_models=request_interaction_data_models,
            regular_vs_shadow=regular_vs_shadow,
        )

    def _build_evaluation_result(
        self,
        evaluation_response: AgentSuccessEvaluationOutput
        | AgentSuccessEvaluationWithComparisonOutput,
        request_interaction_data_models: list[RequestInteractionDataModel],
        regular_vs_shadow: RegularVsShadow | None = None,
    ) -> AgentSuccessEvaluationResult:
        """
        Build an AgentSuccessEvaluationResult from LLM evaluation response and session data.

        Args:
            evaluation_response: The parsed LLM evaluation output
            request_interaction_data_models: All request interaction data models in the session
            regular_vs_shadow: Optional comparison result for shadow evaluation

        Returns:
            AgentSuccessEvaluationResult: The constructed evaluation result
        """
        return AgentSuccessEvaluationResult(
            session_id=self.service_config.session_id,
            agent_version=self.service_config.agent_version,
            evaluation_name=self.config.evaluation_name,
            is_success=evaluation_response.is_success,
            failure_type=evaluation_response.failure_type or "",
            failure_reason=evaluation_response.failure_reason or "",
            regular_vs_shadow=regular_vs_shadow,
            number_of_correction_per_session=self._get_correction_count(),
            user_turns_to_resolution=(
                self._count_user_turns(request_interaction_data_models)
                if evaluation_response.is_success
                else None
            ),
            is_escalated=evaluation_response.is_escalated,
        )

    def _count_user_turns(
        self,
        request_interaction_data_models: list[RequestInteractionDataModel],
    ) -> int:
        """
        Count user-side turns across all interactions in the session.

        A user-side turn is any interaction whose role is NOT one of the agent/system roles.

        Args:
            request_interaction_data_models: All request interaction data models in the session

        Returns:
            int: Number of user-side turns
        """
        agent_roles = _AGENT_ROLES
        count = 0
        for rdm in request_interaction_data_models:
            for interaction in rdm.interactions:
                if interaction.role.lower() not in agent_roles:
                    count += 1
        return count

    def _get_correction_count(self) -> int:
        """
        Count user playbooks linked to the current session.

        Returns:
            int: Number of user playbooks for the session, defaulting to 0 on error.
        """
        try:
            count = self.request_context.storage.count_user_playbooks_by_session(  # type: ignore[reportOptionalMemberAccess]
                self.service_config.session_id
            )
            return count if count is not None else 0
        except Exception:
            logger.warning(
                "Failed to count user playbooks for session %s, defaulting to 0",
                self.service_config.session_id,
            )
            return 0

    def _map_comparison_to_enum(
        self,
        better_request: str,
        is_significantly_better: bool,
        regular_is_request_1: bool,
    ) -> RegularVsShadow:
        """
        Map the LLM's comparison output to the RegularVsShadow enum.

        Args:
            better_request: "1", "2", or "tie" from LLM response
            is_significantly_better: Whether the better one is significantly better
            regular_is_request_1: Whether regular version was assigned to Request 1

        Returns:
            RegularVsShadow enum value
        """
        if better_request == "tie":
            return RegularVsShadow.TIED

        if better_request == "1":
            # Request 1 is better
            if regular_is_request_1:
                # Regular is Request 1, so regular is better
                return (
                    RegularVsShadow.REGULAR_IS_BETTER
                    if is_significantly_better
                    else RegularVsShadow.REGULAR_IS_SLIGHTLY_BETTER
                )
            # Shadow is Request 1, so shadow is better
            return (
                RegularVsShadow.SHADOW_IS_BETTER
                if is_significantly_better
                else RegularVsShadow.SHADOW_IS_SLIGHTLY_BETTER
            )
        if better_request == "2":
            # Request 2 is better
            if regular_is_request_1:
                # Regular is Request 1, so shadow (Request 2) is better
                return (
                    RegularVsShadow.SHADOW_IS_BETTER
                    if is_significantly_better
                    else RegularVsShadow.SHADOW_IS_SLIGHTLY_BETTER
                )
            # Shadow is Request 1, so regular (Request 2) is better
            return (
                RegularVsShadow.REGULAR_IS_BETTER
                if is_significantly_better
                else RegularVsShadow.REGULAR_IS_SLIGHTLY_BETTER
            )

        # Default to tied if unexpected value
        logger.warning(
            "Unexpected better_request value: %s, defaulting to TIED", better_request
        )
        return RegularVsShadow.TIED
