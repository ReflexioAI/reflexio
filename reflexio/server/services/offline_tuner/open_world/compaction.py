"""Background LiteLLM adapter for source-backed session compaction."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from reflexio.server.llm.litellm_client import LiteLLMClient, ProviderRequestGuardError
from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.offline_tuner.evidence.compaction import (
    COMPACTION_INSTRUCTIONS,
    MAX_COMPACTION_RENDER_TOKENS,
    CompactionError,
    CompactionIdentity,
    CompactionPoint,
    CompactionSource,
    SessionCompaction,
    compact_session,
    compaction_source,
    validate_compaction,
)
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.native_capacity import (
    verified_native_context,
)
from reflexio.server.services.offline_tuner.open_world.analyst import (
    _canonical_user_message,
)
from reflexio.server.services.offline_tuner.open_world.identity import canonical_payload
from reflexio.server.services.offline_tuner.open_world.models import (
    EvidenceDocument,
    EvidenceSpan,
)

from .evidence_documents import _span_id


def _source(span: EvidenceSpan) -> CompactionSource:
    return compaction_source(
        request_id=span.request_id,
        interaction_id=span.interaction_id,
        field_path=span.field_path,
        speaker=span.speaker,
        text=span.immutable_text,
    )


def _apply(document: EvidenceDocument, value: SessionCompaction) -> EvidenceDocument:
    sources = {_source(s).source_id: s for s in document.spans}
    validate_compaction(value, {key: _source(s) for key, s in sources.items()})
    spans: dict[str, EvidenceSpan] = {}
    summary: list[dict[str, str]] = []
    for point in value.points:
        original = sources[point.source_id]
        start = (
            point.start_offset
            if point.start_offset is not None
            else original.immutable_text.index(point.exact_quote)
        )
        field_path = f"{original.field_path}[{start}:{start + len(point.exact_quote)}]"
        span = replace(
            original,
            span_id=_span_id(
                trajectory_digest=document.trajectory_digest,
                tenant_id=original.tenant_id,
                role=document.manifest_role,
                user_id=document.user_id,
                session_id=document.session_id,
                request_id=original.request_id,
                interaction_id=original.interaction_id,
                field_path=field_path,
                immutable_text=point.exact_quote,
                speaker=original.speaker,
            ),
            field_path=field_path,
            immutable_text=point.exact_quote,
        )
        spans[span.span_id] = span
        summary.append(
            {
                "category": point.category,
                "detail": point.detail,
                "evidence_span_id": span.span_id,
            }
        )
    result = replace(
        document,
        spans=tuple(spans.values()),
        compaction_summary=json.dumps(summary, ensure_ascii=False),
    )
    # Meter the actual renderer, including escaping and unusually long source
    # identifiers. Reserve additional framing for the surrounding request.
    render_bound = (
        count_input_tokens(
            _canonical_user_message({"document": canonical_payload(result)})
        )
        + 750
    )
    if render_bound > MAX_COMPACTION_RENDER_TOKENS:
        raise CompactionError(
            "compacted document exceeds its frozen payload reservation"
        )
    return result


CompactionPointCategory = Literal[
    "goal", "constraint", "action", "tool_result", "error", "correction", "unresolved"
]


class _SelectedPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    category: CompactionPointCategory
    detail: str = Field(
        min_length=1,
        max_length=1280,
        description="One brief factual sentence, preferably under 160 characters; MUST be at most 512 characters.",
    )
    evidence_id: str = Field(min_length=1, max_length=32)


class _CompactionSlots(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @property
    def points(self) -> list[_SelectedPoint]:
        return [
            point
            for name in type(self).model_fields
            if (point := getattr(self, name)) is not None
        ]


# Fixed required slots enforce capacity structurally even when a provider
# ignores maxItems. Multiple fields also disable truncated single-list salvage.
_SLOT_FIELDS: dict[str, Any] = {
    f"p{index:02d}": (
        _SelectedPoint | None,
        Field(..., description="One source-backed fact, or null when unused."),
    )
    for index in range(1, 25)
}
_SelectedCompaction = create_model(
    "_SelectedCompaction", __base__=_CompactionSlots, **_SLOT_FIELDS
)


def verified_compaction_context(model: str) -> int:
    """Keep the rolling compactor's original minimum separate from model capacity."""
    context = verified_native_context(model)
    if context < 65_536:
        raise CompactionError("compaction model has no verified usable context budget")
    return context


class SessionCompactor:
    def __init__(self, client: LiteLLMClient, *, project_id: str) -> None:
        self.client = client
        self.project_id = project_id
        self.context_tokens = verified_compaction_context(client.config.model)

    def _complete(
        self,
        payload: str,
        document: EvidenceDocument,
        before_call: Callable[[], None],
    ) -> SessionCompaction:
        # These verified routes support direct answers; qualify preservation
        # for this evidence-selection mode separately from analyst reasoning.
        direct_answer = self.client.config.model in {
            "minimax/MiniMax-M3",
            "zai/glm-5.1",
        }
        sources_by_id = {
            _source(span).source_id: _source(span) for span in document.spans
        }

        def guard(params: dict[str, Any], *_args: Any, **_kwargs: Any) -> None:
            before_call()
            if params.get("model") != self.client.config.model:
                raise ProviderRequestGuardError(
                    "compaction model changed during request"
                )
            if direct_answer and params.get("extra_body", {}).get("thinking") != {
                "type": "disabled"
            }:
                raise ProviderRequestGuardError("compaction thinking mode changed")
            messages = params.get("messages", [])
            # UTF-8 bytes are a conservative tokenizer-independent ceiling;
            # include the injected output schema and reserve output+framing.
            upper_bound = (
                len(json.dumps(messages, ensure_ascii=False).encode())
                + len(json.dumps(params.get("response_format", {})).encode())
                + 16_384
                + 2048
            )
            if upper_bound > self.context_tokens:
                raise ProviderRequestGuardError(
                    "compaction request exceeds model context budget"
                )

        admitted = json.loads(payload)
        excerpts: dict[str, tuple[str, str, int | None]] = {}
        selectors: dict[tuple[str, str, int | None], str] = {}

        def register(source_id: str, text: str, start: int | None) -> str:
            key = (source_id, text, start)
            if key not in selectors:
                selector = f"e{len(selectors) + 1}"
                selectors[key] = selector
                excerpts[selector] = key
            return selectors[key]

        records = [
            {
                "evidence_id": register(r["source_id"], r["text"], r["start"]),
                "speaker": r["speaker"],
                "field_path": r["field_path"],
                "text": r["text"],
                "origin": "current",
            }
            for r in admitted["source_records"]
        ]
        previous = None
        if admitted["previous"]:
            previous_points = [
                {
                    "category": p["category"],
                    "detail": p["detail"],
                    "evidence_id": register(
                        p["source_id"], p["exact_quote"], p.get("start_offset")
                    ),
                }
                for p in admitted["previous"]["points"]
            ]
            previous = {
                f"p{index + 1:02d}": previous_points[index]
                if index < len(previous_points)
                else None
                for index in range(24)
            }
            # Keep rolling memory in the exact output shape. Mixing input-only
            # speaker/text/path fields into previous.points caused real models
            # to copy those forbidden fields into subsequent output.
            records.extend(
                {
                    "evidence_id": register(
                        p["source_id"], p["exact_quote"], p.get("start_offset")
                    ),
                    "speaker": sources_by_id[p["source_id"]].speaker,
                    "field_path": sources_by_id[p["source_id"]].field_path,
                    "text": p["exact_quote"],
                    "origin": "previous",
                }
                for p in admitted["previous"]["points"]
            )
        model_payload = canonical_json_bytes(
            {"previous": previous, "source_records": records}
        ).decode()

        def validate_output(value: Any) -> tuple[str, ...]:
            if not isinstance(value, _CompactionSlots):
                return ("response: expected session compaction",)
            budget_errors = []
            if not value.points:
                return ("slots: retain at least one source-backed fact",)
            budget_errors.extend(
                f"points.{index}.detail: you returned {len(point.detail)} characters. Rewrite as a brief sentence of at most 512 characters."
                for index, point in enumerate(value.points)
                if len(point.detail) > 512
            )
            errors = tuple(
                f"points.{index}.evidence_id: copy an exact ID from source_records or previous points"
                for index, point in enumerate(value.points)
                if point.evidence_id not in excerpts
            )
            if errors or budget_errors:
                return (*budget_errors, *errors)
            try:
                # The repair ladder covers semantic/size failures too,
                # before a checkpoint can persist an unusable result.
                _apply(document, materialize(value))
            except CompactionError as exc:
                return (str(exc),)
            return ()

        def materialize(value: _CompactionSlots) -> SessionCompaction:
            return SessionCompaction(
                points=[
                    CompactionPoint(
                        category=p.category,
                        detail=p.detail,
                        source_id=excerpts[p.evidence_id][0],
                        exact_quote=excerpts[p.evidence_id][1],
                        start_offset=excerpts[p.evidence_id][2],
                    )
                    for p in value.points
                ]
            )

        request_options: dict[str, Any] = (
            {"extra_body": {"thinking": {"type": "disabled"}}} if direct_answer else {}
        )
        result = self.client.generate_chat_response(
            [{"role": "user", "content": model_payload}],
            system_message=COMPACTION_INSTRUCTIONS,
            response_format=_SelectedCompaction,
            temperature=0,
            # Thinking tokens share this ceiling on providers such as MiniMax.
            # An 8K limit truncated valid large-batch JSON in live qualification.
            max_tokens=16_384,
            fallback_models=[],
            provider_request_guard=guard,
            structured_output_validator=validate_output,
            **request_options,
        )
        if not isinstance(result, _CompactionSlots):
            raise CompactionError("provider returned no structured session compaction")
        return materialize(result)

    def compact(
        self,
        document: EvidenceDocument,
        *,
        through_request_id: str | None,
        governance_epochs: tuple[tuple[str, int], ...],
        before_call: Callable[[], None],
        checkpoint: Callable[[str, int, SessionCompaction], None] | None = None,
        cached: Mapping[str, Any]
        | Callable[[str], Mapping[str, Any] | None]
        | None = None,
    ) -> EvidenceDocument:
        if not document.spans:
            raise CompactionError("compaction document has no source evidence")
        identity = CompactionIdentity(
            tenant_id=document.spans[0].tenant_id,
            project_id=self.project_id,
            user_id=document.user_id,
            session_id=document.session_id,
            trajectory_digest=document.trajectory_digest,
            through_request_id=through_request_id,
            governance_epochs=tuple(sorted(governance_epochs)),
            model=self.client.config.model,
        )
        sources = tuple(_source(span) for span in document.spans)
        if len({s.source_id for s in sources}) != len(sources):
            raise CompactionError("source selector collision")
        before_call()
        if callable(cached):
            cached = cached(identity.digest)
        cached_json: str | None = None
        if (
            cached
            and cached.get("identity") == identity.digest
            and cached.get("complete") is True
        ):
            cached_json = json.dumps(cached["compaction"])
        if cached_json is not None:
            try:
                value = SessionCompaction.model_validate_json(cached_json)
            except ValidationError as exc:
                raise CompactionError("invalid cached session compaction") from exc
            result = _apply(document, value)
            before_call()
            if checkpoint:
                checkpoint(identity.digest, -1, value)
            return result

        resume_batches = 0
        resume_value = None
        if cached and cached.get("identity") == identity.digest:
            resume_batches = cached["batches"]
            try:
                resume_value = SessionCompaction.model_validate(cached["compaction"])
            except ValidationError as exc:
                raise CompactionError("invalid cached session compaction") from exc
        value = compact_session(
            sources,
            complete=lambda payload: self._complete(payload, document, before_call),
            max_batch_bytes=min(128_000, self.context_tokens - 48_000),
            before_call=before_call,
            resume_batches=resume_batches,
            resume_value=resume_value,
            checkpoint=(lambda count, value: checkpoint(identity.digest, count, value))
            if checkpoint
            else None,
        )
        result = _apply(document, value)
        before_call()
        if checkpoint:
            checkpoint(identity.digest, -1, value)
        return result
