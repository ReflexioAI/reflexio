"""Call-local prepared history for extraction prompts, leaving originals intact."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from reflexio.server.services.extraction.input_preparation import (
    PreparedChunk,
    PreparedExtractionInput,
    SourceSpan,
)
from reflexio.server.services.extraction.protected_record import ProtectedRecord


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
        protected_refs = {
            ref for item in self.record.evidence for ref in item.evidence_refs
        }
        for chunk in self.prepared.chunks:
            for span in (*chunk.neighbors, *chunk.spans):
                if span.evidence_ref in protected_refs:
                    selected[span.evidence_ref] = span
        if not protected_refs <= selected.keys():
            raise ValueError(
                "Protected context references unavailable original evidence"
            )
        return tuple(selected.values())

    @property
    def text(self) -> str:
        original = json.loads(self.chunk.text)
        for origin in original["sources"].values():
            origin["turn_ref"] = f"T{origin['source_order']}"
        return json.dumps(
            {
                "protected_record": self.record.model_dump(mode="json"),
                "original_chunk": original,
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
