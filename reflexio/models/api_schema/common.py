"""Shared types used by both domain/ and ui/ subpackages.

This module contains types that need to be imported by multiple layers
without creating circular dependencies. Keep it minimal — only types
that are genuinely shared belong here.
"""

from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

__all__ = [
    "NEVER_EXPIRES_TIMESTAMP",
    "BlockingIssue",
    "BlockingIssueKind",
    "CapturesUnknownFields",
    "ToolUsed",
    "cap_warning_list",
    "sanitise_for_log",
    "summarise_unknown_names",
]

_MAX_LOG_VALUE_LEN = 64

# Unknown key names are caller-controlled: unbounded in length and count, and
# free to contain newlines. They are echoed into both the HTTP response and a
# log line, so they need bounding on THREE axes -- per-name length, names per
# interaction, and total entries in the warning list -- plus control-character
# stripping. Unbounded, 1000 interactions x N long bogus keys produced a
# ~350 KB response body and one enormous log record.
_MAX_REPORTED_NAMES = 5
_MAX_WARNING_ENTRIES = 20
_MAX_STATUS_LEN = 100


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


def summarise_unknown_names(names: list[str]) -> str:
    """Render unknown key names for a caller-facing warning, bounded and safe.

    Args:
        names (list[str]): The unrecognised key names, already sorted.

    Returns:
        str: Comma-separated sanitised names, each truncated, with a "+N more"
            suffix when the list was longer than the cap.
    """
    shown = [sanitise_for_log(name) for name in names[:_MAX_REPORTED_NAMES]]
    remaining = len(names) - len(shown)
    return ", ".join(shown) + (f", +{remaining} more" if remaining > 0 else "")


def cap_warning_list(warnings: list[str]) -> list[str]:
    """Bound the NUMBER of warning entries.

    Per-name caps alone do not bound the total: ``interaction_data_list``
    permits 1000 entries, so one warning each still adds up to a
    multi-hundred-KB response body and a single enormous log record.

    Args:
        warnings (list[str]): Individually-bounded warning strings.

    Returns:
        list[str]: A NEW list of at most ``_MAX_WARNING_ENTRIES`` entries, with
            a trailing overflow entry when any were dropped. Always a copy --
            callers append to the result.
    """
    if len(warnings) <= _MAX_WARNING_ENTRIES:
        return list(warnings)
    dropped = len(warnings) - _MAX_WARNING_ENTRIES
    return [
        *warnings[:_MAX_WARNING_ENTRIES],
        f"...and {dropped} more interaction(s) with the same problem",
    ]


class CapturesUnknownFields(BaseModel):
    """Records unrecognised keys instead of silently discarding them.

    ``extra="allow"`` here is only a means of *seeing* what the caller sent; the
    validator strips the extras straight back out, so nothing unexpected reaches
    storage or a re-serialised request and ``model_dump()`` stays clean. Only
    NAMES are retained -- the values are caller payload, potentially Customer
    Content, and must never reach a log or an error body.

    Deliberately NOT ``extra="forbid"``. That was implemented and reverted:
    rejecting unknown keys broke every first-party plugin publish, because the
    plugins build their wire payload with a denylist and so carry request-level
    bookkeeping such as ``user_id`` on every turn. Their adapters swallow the
    resulting error and never advance the publish watermark, so the same batch
    retried forever and nothing was ever published -- strictly worse silent data
    loss than the one this reporting exists to surface.
    """

    model_config = ConfigDict(extra="allow")

    _unknown_field_names: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def capture_and_strip_unknown_fields(self) -> Self:
        """Record unknown key names, then drop them.

        Returns:
            Self: this model, with unknown keys recorded and removed.
        """
        if extra := self.__pydantic_extra__:
            self._unknown_field_names = sorted(extra)
            self.__pydantic_extra__ = {}
        return self

    def unknown_field_names(self) -> list[str]:
        """Names of unrecognised keys the caller sent, if any."""
        return self._unknown_field_names


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


class ToolUsed(CapturesUnknownFields):
    tool_name: str
    tool_data: dict = Field(
        default_factory=dict
    )  # tool metadata: input, output, latency, etc.
    # Declared because both first-party plugins already send it on every tool
    # entry ("success" / "error", derived from the tool response). It was being
    # silently discarded -- real signal lost, and the largest single source of
    # unknown-field warnings once nested capture began reporting them.
    status: str = ""

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, value: object) -> str:
        """Accept whatever the caller sent; never reject the batch over it.

        Declaring this strictly (``str`` with ``max_length``) turned five
        previously-harmless values -- an int, None, a bool, a dict, an over-long
        string -- into a 422 rejecting the WHOLE publish, where before they were
        silently ignored and the publish returned 200. The plugin adapters
        swallow that and never advance their watermark, so the batch retries
        forever: the exact failure ``CapturesUnknownFields`` exists to avoid.
        Declaring a field must not be a backdoor to the strictness we rejected.

        Args:
            value (object): Whatever the caller put in ``status``.

        Returns:
            str: A bounded string; never raises.
        """
        if value is None:
            return ""
        return str(value)[:_MAX_STATUS_LEN]
