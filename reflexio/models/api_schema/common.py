"""Shared types used by both domain/ and ui/ subpackages.

This module contains types that need to be imported by multiple layers
without creating circular dependencies. Keep it minimal — only types
that are genuinely shared belong here.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = [
    "NEVER_EXPIRES_TIMESTAMP",
    "BlockingIssueKind",
    "BlockingIssue",
    "ToolUsed",
]

# OS-agnostic "never expires" timestamp (January 1, 2100 00:00:00 UTC)
NEVER_EXPIRES_TIMESTAMP = 4102444800


class BlockingIssueKind(StrEnum):
    MISSING_TOOL = "missing_tool"
    PERMISSION_DENIED = "permission_denied"
    EXTERNAL_DEPENDENCY = "external_dependency"
    POLICY_RESTRICTION = "policy_restriction"


class BlockingIssue(BaseModel):
    kind: BlockingIssueKind
    details: str = Field(
        description="What capability is missing and why it blocks the request"
    )


class ToolUsed(BaseModel):
    tool_name: str
    tool_data: dict = Field(
        default_factory=dict
    )  # tool metadata: input, output, latency, etc.
