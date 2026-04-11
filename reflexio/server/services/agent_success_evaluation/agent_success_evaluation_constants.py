"""Constants for agent success evaluation service"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class AgentSuccessEvaluationConstants:
    """Constants for agent success evaluation prompts and configurations"""

    # Prompt IDs
    AGENT_SUCCESS_EVALUATION_PROMPT_ID = "agent_success_evaluation"
    AGENT_SUCCESS_EVALUATION_WITH_COMPARISON_PROMPT_ID = (
        "agent_success_evaluation_with_comparison"
    )


class AgentSuccessEvaluationOutput(BaseModel):
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


class AgentSuccessEvaluationWithComparisonOutput(BaseModel):
    """
    Combined output schema for agent success evaluation with regular vs shadow comparison.

    This schema is used when interactions contain shadow_content. It includes:
    1. Success evaluation for the regular version (Request 1 in prompt when regular_is_request_1)
    2. Comparison result between regular and shadow versions

    Attributes:
        is_success (bool): Indicates whether the agent successfully responded (evaluated on regular version)
        failure_type (Optional[str]): Type of failure when is_success=False
        failure_reason (Optional[str]): Explanation for the failure when is_success=False
        better_request (str): Which request is better - "1", "2", or "tie"
        is_significantly_better (bool): Whether the better request is significantly better
        comparison_reason (Optional[str]): Brief explanation of why one is better
    """

    # Success evaluation fields (for regular version)
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
        description="Type of failure. Required when is_success=False",
    )
    failure_reason: str | None = Field(
        default=None,
        description="Explanation for the failure. Required when is_success=False",
    )
    is_escalated: bool = Field(
        default=False,
        description="Whether the user was handed off to a human agent or another agent during the session.",
    )
    # Comparison fields
    better_request: Literal["1", "2", "tie"] = Field(
        description="Which request is better: '1', '2', or 'tie'"
    )
    is_significantly_better: bool = Field(
        description="Whether the better request is significantly better than the other"
    )
    comparison_reason: str | None = Field(
        default=None,
        description="Brief explanation of why one request is better than the other",
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )
