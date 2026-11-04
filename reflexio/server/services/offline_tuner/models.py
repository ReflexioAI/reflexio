from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, field_validator

from reflexio.models.structured_output import StrictStructuredOutput


class CandidateRevision(StrictStructuredOutput):
    """The only candidate mutation supported by replay-gated tuning."""

    model_config = ConfigDict(extra="forbid")

    revised_content: str

    @field_validator("revised_content")
    @classmethod
    def require_nonempty_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("revised_content must be non-empty")
        return value

    @classmethod
    def from_untrusted(cls, value: object) -> Self:
        """Revalidate provider output without accepting compatibility aliases."""
        return cls.model_validate(value)
