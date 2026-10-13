"""Bounded cross-chunk context made from verifiable original evidence.

Selected quotations are context, not a replacement for full chunk coverage.
Updates append evidence; an apparent contradiction never silently deletes an
earlier correction or its chronology.
"""

import json
from typing import Literal

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

from reflexio.server.services.extraction.input_preparation import (
    ExtractionInputBudgetExceededError,
    SourceSpan,
    resolve_evidence_span,
)


class ProtectedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["instruction", "correction", "action", "outcome", "uncertainty"]
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=8)
    quote: str = Field(min_length=1)
    source_order: int = Field(default=0, ge=0)
    role: str = ""


class CandidateEvidenceQuote(ProtectedEvidence):
    candidate_ref: str = Field(min_length=1)


class ProtectedRecordUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[ProtectedEvidence] = Field(default_factory=list, max_length=128)
    candidate_evidence: list[CandidateEvidenceQuote] = Field(
        default_factory=list, max_length=128
    )


class ProtectedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: tuple[ProtectedEvidence, ...] = ()

    def render(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )

    def merge(
        self,
        update: ProtectedRecordUpdate,
        *,
        visible_spans: tuple[SourceSpan, ...],
        max_tokens: int = 2048,
    ) -> "ProtectedRecord":
        """Accept exact new quotations from this pass without rewriting history."""
        if max_tokens <= 0:
            raise ValueError("Protected record capacity must be positive")
        entries = list(self.evidence)
        seen = {(item.kind, item.evidence_refs, item.quote) for item in entries}
        for item in update.evidence:
            try:
                verified = [
                    resolve_evidence_span(ref, visible_spans)
                    for ref in item.evidence_refs
                ]
            except ValueError as exc:
                raise ValueError(
                    "Protected evidence must quote an original visible source span"
                ) from exc
            first = verified[0]
            if any(
                span.interaction_id != first.interaction_id
                or span.request_id != first.request_id
                or span.field != first.field
                or (index > 0 and verified[index - 1].end != span.start)
                for index, span in enumerate(verified)
            ):
                raise ValueError(
                    "Protected quotation must preserve contiguous original evidence"
                )
            if item.quote not in "".join(span.text for span in verified):
                raise ValueError(
                    "Protected evidence must quote an original visible source span"
                )
            key = (item.kind, item.evidence_refs, item.quote)
            if key not in seen:
                entries.append(
                    item.model_copy(
                        update={"source_order": first.source_order, "role": first.role}
                    )
                )
                seen.add(key)
        entries.sort(key=lambda item: item.source_order)
        merged = ProtectedRecord(evidence=tuple(entries))
        encoding = tiktoken.get_encoding("cl100k_base")
        if len(encoding.encode(merged.render(), disallowed_special=())) > max_tokens:
            raise ExtractionInputBudgetExceededError(
                "Required protected evidence exceeds record capacity"
            )
        return merged
