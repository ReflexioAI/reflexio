"""Exact original evidence presentation, independent of legacy chunk geometry.

The pure renderer also serves prompt costing. Only the validating factory makes
an execution presentation; authorization and processed-leaf receipts belong to
its durable caller. This module does not install a runtime ContextVar or publish.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from reflexio.server.services.extraction.input_preparation import (
    ExtractionSources,
    SourceSpan,
    resolve_evidence_span,
)
from reflexio.server.services.input_compression import EvidencePacket, EvidenceSource

PACKAGE_PROTOCOL = "source-package-v1"
_SOURCE_POLICY = (
    "Only the original evidence text is citable. Labels and omissions are host "
    "navigation, not additional testimony. Request source identifies the producer "
    "or workflow; transcript role User does not establish human authorship or "
    "permission. Preserve conditions, chronology, corrections and uncertainty."
)


def evidence_from_source_span(span: SourceSpan) -> EvidenceSource:
    """Keep original range, producer, role and chronology in the shared DTO."""
    return EvidenceSource(
        source_id=span.evidence_ref,
        request_id=span.request_id,
        interaction_id=span.interaction_id,
        field=span.field,
        speaker=span.role,
        source_order=span.source_order,
        text=span.text,
        start=span.start,
        request_source=span.request_source or "",
        created_at=span.created_at,
        request_created_at=span.request_created_at,
        session_id=span.session_id,
    )


def render_source_package(packet: EvidencePacket, *, source_digest: str) -> str:
    """Render the exact consumer envelope for pure costing and execution."""
    return json.dumps(
        {
            "input_protocol": PACKAGE_PROTOCOL,
            "source_digest": source_digest,
            "source_attribution_policy": _SOURCE_POLICY,
            "evidence_package": packet.model_dump(mode="json", exclude_none=True),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class SourcePackagePresentation:
    source_digest: str
    package_digest: str
    _text: str = field(repr=False)
    visible_spans: tuple[SourceSpan, ...] = field(repr=False)

    @property
    def text(self) -> str:
        return self._text

    def is_present_in(self, messages: list[dict[str, Any]]) -> bool:
        """Custom prompts must include the complete actual package envelope."""
        representations = (self.text, json.dumps(self.text, ensure_ascii=False)[1:-1])
        for message in messages:
            content = message.get("content")
            if isinstance(content, str) and any(s in content for s in representations):
                return True
            if isinstance(content, list) and any(
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and any(s in part["text"] for s in representations)
                for part in content
            ):
                return True
        return False


def source_package_presentation(
    packet: EvidencePacket, *, sources: ExtractionSources
) -> SourcePackagePresentation:
    """Validate exact selected originals, then freeze immutable prompt bytes."""
    if not sources.source_digest:
        raise ValueError("Source package requires its original witness")
    identifiers = [source.source_id for source in packet.evidence]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Source package evidence identity collision")
    positions = {
        (span.interaction_id, span.field): index
        for index, span in enumerate(sources.spans)
    }
    visible = []
    order = []
    for evidence in packet.evidence:
        resolved = resolve_evidence_span(evidence.source_id, sources.spans)
        if evidence != evidence_from_source_span(resolved):
            raise ValueError("Source package original evidence changed")
        visible.append(resolved)
        order.append(
            (positions[(resolved.interaction_id, resolved.field)], resolved.start)
        )
    if order != sorted(order):
        raise ValueError("Source package chronology changed")
    available = set(identifiers)
    if any(
        ref not in available for point in packet.points for ref in point.evidence_refs
    ):
        raise ValueError("Source package selection references unseen evidence")
    text = render_source_package(packet, source_digest=sources.source_digest)
    return SourcePackagePresentation(
        source_digest=sources.source_digest,
        package_digest=hashlib.sha256(text.encode()).hexdigest(),
        _text=text,
        visible_spans=tuple(visible),
    )
