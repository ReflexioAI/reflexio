"""Constants for agent success evaluation service"""

from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict, Field

from reflexio.models.structured_output import StrictStructuredOutput


@dataclass(frozen=True)
class AgentSuccessEvaluationConstants:
    """Constants for agent success evaluation prompts and configurations"""

    # Prompt IDs
    AGENT_SUCCESS_EVALUATION_PROMPT_ID = "agent_success_evaluation"


class AgentSuccessEvaluationOutput(StrictStructuredOutput):
    """
    Unified output schema for agent success evaluation.

    For successful evaluations, only is_success=True is required.
    For failed evaluations, all fields are required to provide failure details.

    Attributes:
        is_success (bool): Indicates whether the agent successfully responded to the user
        failure_type (Optional[str]): Type of failure - 'missing_tool', 'wrong_tool', 'insufficient_info_from_tool', or 'wrong_answer'. Required when is_success=False
        failure_reason (Optional[str]): Explanation for the failure and what the agent needs to do differently. Required when is_success=False
    """

    is_success: bool = Field(
        description="Indicates whether the agent successfully responded to the user"
    )
    failure_type: (
        Literal[
            "missing_tool", "wrong_tool", "insufficient_info_from_tool", "wrong_answer"
        ]
        | None
    ) = Field(
        default=None,
        description="Type of improvement the agent needs: 'missing_tool' (agent lacks necessary tools), 'wrong_tool' (agent used incorrect tool), 'insufficient_info_from_tool' (tool lacks necessary information), 'wrong_answer' (agent had info but answered incorrectly). Required when is_success=False",
    )
    failure_reason: str | None = Field(
        default=None,
        description="Explanation for the failure and what the agent needs to do differently. Required when is_success=False",
    )
    is_escalated: bool = Field(
        default=False,
        description="Whether the user was handed off to a human agent or another agent during the session.",
    )
    # OpenAI schema parsing requires explicitly forbidding additional properties
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


# F1 cleanup: ``AgentSuccessEvaluationWithComparisonOutput`` (the combined
# evaluation-plus-comparison schema) was retracted along with the session-level
# shadow comparison branch in ``AgentSuccessEvaluator``. Per-turn shadow
# comparison uses dedicated schemas under ``services/shadow_comparison/``.
