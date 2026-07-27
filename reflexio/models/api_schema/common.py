"""Shared types used by both domain/ and ui/ subpackages.

This module contains types that need to be imported by multiple layers
without creating circular dependencies. Keep it minimal — only types
that are genuinely shared belong here.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = [
    "NEVER_EXPIRES_TIMESTAMP",
    "BlockingIssue",
    "BlockingIssueKind",
    "ToolUsed",
    "sanitise_for_log",
]

_MAX_LOG_VALUE_LEN = 64


def sanitise_for_log(value: str, max_len: int = _MAX_LOG_VALUE_LEN) -> str:
    """Make one caller-supplied string safe to put in a log line.

    Strips control characters and bounds the length. A newline in a
    caller-controlled value forges a line in a shared multi-tenant log stream,
    and an unbounded value bloats the record. ``request_id`` is a
    ``NonEmptyStr`` with no length cap and no character restrictions, so it
    qualifies.

    Args:
        value (str): The caller-supplied string.
        max_len (int): Maximum length before truncation.

    Returns:
        str: Printable, length-bounded text safe for a single log line.
    """
    cleaned = "".join(char if char.isprintable() else "?" for char in value)
    return cleaned if len(cleaned) <= max_len else f"{cleaned[:max_len]}..."


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
