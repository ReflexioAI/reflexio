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


class CapturesUnknownFields(BaseModel):
    """Records unrecognised keys instead of silently discarding them.

    ``extra="allow"`` here is only a means of *seeing* what the caller sent.
    **On the validation path** -- ``model_validate`` / ``__init__``, which is
    how every HTTP request and SDK call builds these models -- the validator
    strips the extras straight back out, so nothing unexpected reaches storage
    or a re-serialised request and ``model_dump()`` stays clean. That guarantee
    is scoped to that path: ``model_construct``, ``model_copy(update=...)`` and
    a plain ``instance.attr = value`` with an undeclared name all bypass
    validation, and under ``extra="allow"`` they store the value in
    ``__pydantic_extra__`` where it *will* reach ``model_dump()``. Under the
    previous ``extra="ignore"`` such an assignment raised instead. No
    first-party code constructs these models that way; if that changes, add a
    ``model_serializer`` that drops extras rather than relying on this note.
    Only NAMES are retained -- the values are caller payload, potentially
    Customer Content, and must never reach a log or an error body.

    Deliberately NOT ``extra="forbid"``. That was implemented and reverted:
    rejecting unknown keys broke every first-party plugin publish, because the
    plugins build their wire payload with a denylist and so carry request-level
    bookkeeping such as ``user_id`` on every turn. Their adapters swallow the
    resulting error and never advance the publish watermark, so the same batch
    retried forever and nothing was ever published -- strictly worse silent data
    loss than the one this reporting exists to surface.
    """

    # ``json_schema_extra`` corrects what ``extra="allow"`` would otherwise
    # publish: pydantic emits ``additionalProperties: true`` for it, telling
    # every OpenAPI consumer that unknown keys are accepted and kept. They are
    # accepted and *stripped*, which is the opposite promise.
    model_config = ConfigDict(
        extra="allow", json_schema_extra={"additionalProperties": False}
    )

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


# Bounds ``ToolUsed.status`` below. That field is coerced rather than rejected,
# so its length is caller-controlled and needs a cap of its own.
_MAX_STATUS_LEN = 100


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
            str: A bounded string. Not literally total -- ``str(value)`` calls
                a caller-defined ``__str__`` when one exists -- but total over
                the inputs that matter: no decoded JSON value can make this
                raise, so no HTTP payload can turn ``status`` into a 422.
        """
        if value is None:
            return ""
        return str(value)[:_MAX_STATUS_LEN]
