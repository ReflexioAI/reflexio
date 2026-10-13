"""Lossless, source-addressed text preparation for extraction overflow.

This module deliberately makes no judgement about which facts matter. Every
nonempty extraction text field is represented before any legacy clipping.
Provider prompt budgets and protected-record judgement belong to the caller.
"""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import tiktoken

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request


class ExtractionInputBudgetExceededError(RuntimeError):
    """Complete source coverage cannot fit the bounded extraction work plan."""


def source_witness_bytes(request: Request, interaction: Interaction) -> bytes:
    """Canonical source witness shared with final transaction validation."""
    return json.dumps(
        {
            "request": request.model_dump(mode="json"),
            "interaction": interaction.model_dump(
                mode="json", exclude={"embedding", "token_count"}
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


@dataclass(frozen=True)
class SourceSpan:
    interaction_id: int
    request_id: str
    role: str
    field: str
    start: int
    end: int
    text: str
    source_order: int = 0
    created_at: int | None = None
    request_created_at: int | None = None
    session_id: str | None = None

    @property
    def evidence_ref(self) -> str:
        return f"i{self.interaction_id}:{self.field}:{self.start}:{self.end}"

    def render(self) -> str:
        # JSON framing keeps source-controlled newlines/labels inside the text.
        return json.dumps(
            {
                "evidence_ref": self.evidence_ref,
                "request_id": self.request_id,
                "role": self.role,
                "source_order": self.source_order,
                "text": self.text,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def evidence_payload(self) -> dict[str, str]:
        return {"evidence_ref": self.evidence_ref, "text": self.text}

    def origin_payload(self) -> dict[str, str | int]:
        payload: dict[str, str | int] = {
            "request_id": self.request_id,
            "role": self.role,
            "source_order": self.source_order,
        }
        if self.created_at is not None:
            payload["created_at"] = datetime.fromtimestamp(
                self.created_at, UTC
            ).isoformat()
        if self.request_created_at is not None:
            payload["request_created_at"] = datetime.fromtimestamp(
                self.request_created_at, UTC
            ).isoformat()
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        return payload


@dataclass(frozen=True)
class PreparedChunk:
    index: int
    spans: tuple[SourceSpan, ...]
    neighbors: tuple[SourceSpan, ...] = ()

    @property
    def text(self) -> str:
        ordered = (*self.neighbors, *self.spans)
        # Share immutable origin metadata within this prompt. Every occurrence
        # still carries its original field/span reference and exact text.
        return json.dumps(
            {
                "sources": {
                    str(span.interaction_id): span.origin_payload() for span in ordered
                },
                "neighbor_evidence": [
                    span.evidence_payload() for span in self.neighbors
                ],
                "evidence": [span.evidence_payload() for span in self.spans],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class PreparedExtractionInput:
    source_digest: str
    chunks: tuple[PreparedChunk, ...]


def _split_field(
    *,
    interaction_id: int,
    request_id: str,
    role: str,
    field: str,
    text: str,
    encoding: tiktoken.Encoding,
    span_tokens: int,
    source_order: int,
    created_at: int | None = None,
    request_created_at: int | None = None,
    session_id: str | None = None,
) -> Iterator[SourceSpan]:
    start = 0
    while start < len(text):
        # Split at character boundaries: decoding partial tokenizer bytes can
        # corrupt Unicode and break exact source-span verification.
        lower = start + 1
        upper = min(len(text), start + span_tokens * 8)
        best = start
        while lower <= upper:
            middle = (lower + upper) // 2
            count = len(encoding.encode(text[start:middle], disallowed_special=()))
            if count <= span_tokens:
                best = middle
                lower = middle + 1
            else:
                upper = middle - 1
        if best == start:
            raise ExtractionInputBudgetExceededError(
                "A source character exceeds span capacity"
            )
        yield SourceSpan(
            interaction_id,
            request_id,
            role,
            field,
            start,
            best,
            text[start:best],
            source_order,
            created_at,
            request_created_at,
            session_id,
        )
        start = best


def prepare_extraction_input(
    sessions: list[RequestInteractionDataModel],
    *,
    user_id: str,
    chunk_tokens: int = 8192,
    span_tokens: int = 512,
    max_chunks: int = 48,
    overlap_tokens: int = 256,
    require_overlap: bool = False,
) -> PreparedExtractionInput:
    """Build bounded chunks without dropping or mutating source text.

    Inputs must already be loaded from the scoped storage reference. This extra
    owner check rejects accidental mixed-owner input; project/tenant checks stay
    at that storage boundary. Chunk capacities include JSON evidence framing.
    """
    if (
        not user_id
        or min(chunk_tokens, span_tokens, max_chunks) <= 0
        or overlap_tokens < 0
    ):
        raise ValueError("Invalid extraction preparation scope or capacity")
    encoding = tiktoken.get_encoding("cl100k_base")
    primary_capacity = (
        chunk_tokens - overlap_tokens if require_overlap else chunk_tokens
    )
    if primary_capacity <= 0:
        raise ExtractionInputBudgetExceededError(
            "Neighbor context exceeds chunk capacity"
        )
    source_hash = hashlib.sha256()
    chunks: list[PreparedChunk] = []
    current: list[SourceSpan] = []
    frame_tokens = len(
        encoding.encode(PreparedChunk(0, ()).text, disallowed_special=())
    )
    current_tokens = frame_tokens
    current_origins: set[int] = set()
    seen: set[int] = set()
    source_order = 0
    for session in sessions:
        if session.request.user_id != user_id:
            raise ValueError("Extraction request owner mismatch")
        for interaction in session.interactions:
            if (
                interaction.user_id != user_id
                or interaction.request_id != session.request.request_id
                or interaction.interaction_id <= 0
                or interaction.interaction_id in seen
            ):
                raise ValueError("Extraction source identity mismatch")
            seen.add(interaction.interaction_id)
            source_order += 1
            source_hash.update(source_witness_bytes(session.request, interaction))
            fields = [
                ("content", interaction.content),
                ("shadow_content", interaction.shadow_content),
                ("expert_content", interaction.expert_content),
                ("user_action_description", interaction.user_action_description),
            ]
            if interaction.user_action.value != "none":
                fields.append(("user_action", interaction.user_action.value))
            fields.extend(
                (
                    f"tools_used[{index}]",
                    json.dumps(
                        tool.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                for index, tool in enumerate(interaction.tools_used)
            )
            for field, text in fields:
                for span in _split_field(
                    interaction_id=interaction.interaction_id,
                    request_id=interaction.request_id,
                    role=interaction.role,
                    field=field,
                    text=text,
                    encoding=encoding,
                    span_tokens=span_tokens,
                    source_order=source_order,
                    created_at=interaction.created_at if require_overlap else None,
                    request_created_at=session.request.created_at
                    if require_overlap
                    else None,
                    session_id=session.session_id if require_overlap else None,
                ):
                    evidence_cost = (
                        len(
                            encoding.encode(
                                json.dumps(
                                    span.evidence_payload(),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                                disallowed_special=(),
                            )
                        )
                        + 1
                    )
                    origin_cost = (
                        len(
                            encoding.encode(
                                json.dumps(
                                    {str(span.interaction_id): span.origin_payload()},
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                                disallowed_special=(),
                            )
                        )
                        + 1
                    )
                    cost = evidence_cost + (
                        origin_cost if span.interaction_id not in current_origins else 0
                    )
                    if frame_tokens + evidence_cost + origin_cost > primary_capacity:
                        raise ExtractionInputBudgetExceededError(
                            "Evidence framing exceeds chunk capacity"
                        )
                    if current and current_tokens + cost > primary_capacity:
                        chunks.append(PreparedChunk(len(chunks), tuple(current)))
                        current, current_tokens = [], frame_tokens
                        current_origins.clear()
                        cost = evidence_cost + origin_cost
                    if len(chunks) >= max_chunks:
                        raise ExtractionInputBudgetExceededError(
                            "Complete evidence exceeds chunk limit"
                        )
                    current.append(span)
                    current_origins.add(span.interaction_id)
                    current_tokens += cost
    if current:
        chunks.append(PreparedChunk(len(chunks), tuple(current)))

    # Neighbor context is source-backed and optional; it never replaces primary
    # coverage. Include only whole spans within both overlap and chunk capacity.
    for index in range(1, len(chunks)):
        neighbor = chunks[index - 1].spans[-1]
        if require_overlap:
            neighbor = _bounded_neighbor(
                neighbor,
                encoding,
                overlap_tokens,
                chunk=chunks[index],
                chunk_tokens=chunk_tokens,
            )
        cost = len(encoding.encode(neighbor.render(), disallowed_special=()))
        candidate = PreparedChunk(index, chunks[index].spans, (neighbor,))
        total = len(encoding.encode(candidate.text, disallowed_special=()))
        if cost <= overlap_tokens and total <= chunk_tokens:
            chunks[index] = candidate
        elif require_overlap:
            raise ExtractionInputBudgetExceededError(
                "Required neighbor context exceeds capacity"
            )
    return PreparedExtractionInput(source_hash.hexdigest(), tuple(chunks))


def _bounded_neighbor(
    span: SourceSpan,
    encoding: tiktoken.Encoding,
    limit: int,
    *,
    chunk: PreparedChunk | None = None,
    chunk_tokens: int | None = None,
) -> SourceSpan:
    from dataclasses import replace

    low, high, chosen = span.start, span.end - 1, None
    while low <= high:
        middle = (low + high) // 2
        candidate = replace(span, start=middle, text=span.text[middle - span.start :])
        fits = len(encoding.encode(candidate.render(), disallowed_special=())) <= limit
        if fits and chunk is not None:
            # A different interaction adds its complete origin (including dates
            # and session) to the v2 source map. Bound the actual presentation,
            # rather than assuming the legacy span rendering includes that cost.
            rendered = PreparedChunk(chunk.index, chunk.spans, (candidate,)).text
            fits = (
                chunk_tokens is not None
                and len(encoding.encode(rendered, disallowed_special=()))
                <= chunk_tokens
            )
        if fits:
            chosen, high = candidate, middle - 1
        else:
            low = middle + 1
    if chosen is None:
        raise ExtractionInputBudgetExceededError(
            "Source neighbor framing exceeds capacity"
        )
    return chosen


def repartition_extraction_input(
    prepared: PreparedExtractionInput, *, chunk_index: int, split_after: int
) -> PreparedExtractionInput:
    """Split one unfinished leaf without changing any original source spans."""
    from dataclasses import replace

    if type(chunk_index) is not int or not 0 <= chunk_index < len(prepared.chunks):
        raise ValueError("Unknown source chunk")
    original = prepared.chunks[chunk_index]
    if type(split_after) is not int or not 0 < split_after < len(original.spans):
        raise ExtractionInputBudgetExceededError("Source chunk cannot be safely split")
    if len(prepared.chunks) >= 48:
        raise ExtractionInputBudgetExceededError(
            "Source repartition exceeds chunk limit"
        )
    first = PreparedChunk(chunk_index, original.spans[:split_after], original.neighbors)
    neighbor = _bounded_neighbor(
        first.spans[-1], tiktoken.get_encoding("cl100k_base"), 256
    )
    second = PreparedChunk(chunk_index + 1, original.spans[split_after:], (neighbor,))
    chunks = (
        *prepared.chunks[:chunk_index],
        first,
        second,
        *prepared.chunks[chunk_index + 1 :],
    )
    return PreparedExtractionInput(
        prepared.source_digest,
        tuple(replace(chunk, index=index) for index, chunk in enumerate(chunks)),
    )
