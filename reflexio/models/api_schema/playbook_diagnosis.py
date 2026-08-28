"""Public retrieved-learning diagnosis contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PlaybookDiagnosis(BaseModel):
    """An evidence-bounded diagnosis, not proof of a causal effect."""

    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "content_defect",
        "application_failure",
        "external_failure",
        "no_issue",
        "unknown",
    ]
    reason: str = Field(min_length=1, max_length=4000)
    evidence_interaction_ids: list[int] = Field(default_factory=list, max_length=20)
