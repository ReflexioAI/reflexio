"""Call-local prepared history for extraction prompts, leaving originals intact."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.extraction.input_preparation import (
    PreparedChunk,
    PreparedExtractionInput,
    SourceSpan,
    resolve_evidence_span,
)
from reflexio.server.services.extraction.protected_record import (
    ProtectedRecord,
    ProtectedRecordUpdate,
)


def protected_source_origins(
    prepared: PreparedExtractionInput,
    *records: ProtectedRecord,
    source_spans: tuple[SourceSpan, ...] = (),
) -> dict[str, Any]:
    """V3 source attribution is contextual metadata, never a rewritten record."""
    approved = tuple(
        span for chunk in prepared.chunks for span in (*chunk.neighbors, *chunk.spans)
    )
    if not any(span.request_source is not None for span in approved):
        return {}
    origins = {}
    for carried in source_spans:
        span = resolve_evidence_span(carried.evidence_ref, approved)
        if span != carried:
            raise InputBudgetConfigurationError("Carried source origin changed")
        origins[span.evidence_ref] = span.origin_payload()
    for record in records:
        for item in record.evidence:
            for ref in item.evidence_refs:
                span = resolve_evidence_span(ref, approved)
                origins[ref] = span.origin_payload()
    return {
        "protected_source_origins": origins,
        "source_attribution_policy": (
            "Request source identifies the producer or workflow. Transcript role User "
            "does not prove direct human authorship or permission. Generated or internal "
            "requests are not user approval; an empty source is unknown, not human approval."
        ),
    }


class ProtectedQuoteAmbiguityError(InputBudgetConfigurationError):
    """A quotation needs a more specific original range before freezing."""


def protected_quote_view(
    prepared: PreparedExtractionInput, record: ProtectedRecord
) -> tuple[ProtectedRecord, tuple[SourceSpan, ...]]:
    """Expose only the literal quotation, with its exact original offsets."""
    approved = tuple(
        span for chunk in prepared.chunks for span in (*chunk.neighbors, *chunk.spans)
    )
    evidence = []
    shown = {}
    for item in record.evidence:
        spans = [resolve_evidence_span(ref, approved) for ref in item.evidence_refs]
        first = spans[0]
        if any(
            span.interaction_id != first.interaction_id
            or span.request_id != first.request_id
            or span.field != first.field
            or (index > 0 and spans[index - 1].end != span.start)
            for index, span in enumerate(spans)
        ):
            raise InputBudgetConfigurationError("Protected quotation source changed")
        original = "".join(span.text for span in spans)
        offset = original.find(item.quote)
        if offset < 0 or original.find(item.quote, offset + 1) >= 0:
            raise ProtectedQuoteAmbiguityError(
                "Protected quotation lacks a unique original occurrence"
            )
        start, end = first.start + offset, first.start + offset + len(item.quote)
        refs = []
        for span in spans:
            low, high = max(start, span.start), min(end, span.end)
            if low >= high:
                continue
            piece = replace(
                span,
                start=low,
                end=high,
                text=span.text[low - span.start : high - span.start],
            )
            shown[piece.evidence_ref] = piece
            refs.append(piece.evidence_ref)
        evidence.append(item.model_copy(update={"evidence_refs": tuple(refs)}))
    rendered = ProtectedRecord(evidence=tuple(evidence)).merge(
        ProtectedRecordUpdate(), visible_spans=()
    )
    return rendered, tuple(shown.values())


@dataclass(frozen=True)
class OverflowPresentation:
    prepared: PreparedExtractionInput
    chunk: PreparedChunk
    record: ProtectedRecord

    @property
    def visible_spans(self) -> tuple[SourceSpan, ...]:
        selected = {
            span.evidence_ref: span
            for span in (*self.chunk.neighbors, *self.chunk.spans)
        }
        _, protected = protected_quote_view(self.prepared, self.record)
        selected.update({span.evidence_ref: span for span in protected})
        return tuple(selected.values())

    @property
    def text(self) -> str:
        original = json.loads(self.chunk.text)
        for origin in original["sources"].values():
            origin["turn_ref"] = f"T{origin['source_order']}"
        return json.dumps(
            {
                "protected_record": protected_quote_view(self.prepared, self.record)[
                    0
                ].model_dump(mode="json"),
                "original_chunk": original,
                **protected_source_origins(self.prepared, self.record),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def is_present_in(self, messages: list[dict[str, Any]]) -> bool:
        """Custom templates must actually carry this complete source chunk."""
        text = self.text
        representations = (text, json.dumps(text, ensure_ascii=False)[1:-1])
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and any(
                value in content for value in representations
            ):
                return True
            if isinstance(content, list):
                for part in content:
                    if (
                        isinstance(part, dict)
                        and isinstance(part.get("text"), str)
                        and any(value in part["text"] for value in representations)
                    ):
                        return True
        return False


current_overflow_presentation: ContextVar[OverflowPresentation | None] = ContextVar(
    "reflexio_overflow_presentation", default=None
)


@contextmanager
def present_extraction_chunk(
    prepared: PreparedExtractionInput, chunk: PreparedChunk, record: ProtectedRecord
) -> Iterator[None]:
    if not any(chunk is item for item in prepared.chunks):
        raise ValueError("Extraction presentation lacks its prepared chunk")
    token = current_overflow_presentation.set(
        OverflowPresentation(prepared, chunk, record)
    )
    try:
        yield
    finally:
        current_overflow_presentation.reset(token)
