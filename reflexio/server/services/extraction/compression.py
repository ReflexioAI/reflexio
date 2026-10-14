"""Window adapter for shared compression; original extraction storage is intact.

The caller chooses when its rendered downstream prompt needs compression and
supplies the available evidence budget. A result is not a learning/write plan.
"""

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction.input_preparation import (
    collect_extraction_sources,
)
from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionPoint,
    CompressionResult,
    EvidencePacket,
    EvidenceSource,
    packet_tokens,
)
from reflexio.server.services.input_compression_llm import compress_with_client

EXTRACTION_OUTPUT_CONTRACT = "extraction-original-ranges-content-groups-v3"


def _canonical_packet(packet: EvidencePacket) -> EvidencePacket:
    sources = [
        source.model_copy(
            update={
                "source_id": f"i{source.interaction_id}:{source.field}:{source.start}:{source.start + len(source.text)}",
            }
        )
        for source in packet.evidence
    ]
    refs = {
        before.source_id: after.source_id
        for before, after in zip(packet.evidence, sources, strict=True)
    }
    return packet.model_copy(
        update={
            "evidence": sources,
            "points": [
                point.model_copy(
                    update={"evidence_refs": [refs[ref] for ref in point.evidence_refs]}
                )
                for point in packet.points
            ],
        }
    )


def compress_extraction_window(
    groups: list[RequestInteractionDataModel],
    *,
    org_id: str,
    project_id: str,
    user_id: str,
    budget: CompressionBudget,
    client: LiteLLMClient,
    context_limits: Mapping[str, int],
    before_call: Callable[[], None] = lambda: None,
    load: Callable[[str], Mapping[str, Any] | None] = lambda _key: None,
    save: Callable[[str, dict[str, Any]], None] = lambda _key, _value: None,
) -> CompressionResult:
    """Prepare all original extraction fields, including giant tool payloads.

    Groups must be loaded by the caller's project/tenant-scoped storage. The
    existing preparation rechecks request/interaction ownership and uniqueness.
    before_call authorizes the lease, including cache-only/pass-through results;
    durable attempt reservations belong at the scoped provider dispatch guard.
    Do not replace an admitted overflow journal with this different protocol.
    """
    before_call()
    prepared = collect_extraction_sources(
        groups,
        user_id=user_id,
        include_timestamps=True,
        include_request_source=True,
        max_source_bytes=budget.max_source_bytes,
    )
    sources = [
        EvidenceSource(
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
        for span in prepared.spans
    ]
    # Short conversation text must not compete with thousands of tool results
    # at every reduction. Preserve ALL these fields together when inexpensive;
    # never guess which approval, correction or assistant qualification matters.
    preserved = [source for source in sources if source.field == "content"]

    def preserved_points() -> list[CompressionPoint]:
        return (
            [
                CompressionPoint(
                    category="original_conversation",
                    detail="Original conversation; roles do not establish human authorship.",
                    evidence_refs=[source.source_id for source in preserved],
                )
            ]
            if preserved
            else []
        )

    if packet_tokens(
        _canonical_packet(EvidencePacket(evidence=preserved, points=preserved_points()))
    ) > (budget.output_tokens // 2):
        preserved = []
    preserved_ids = {source.source_id for source in preserved}
    field_order = {
        (source.request_id, source.interaction_id, source.field): index
        for index, source in enumerate(sources)
    }

    def with_original_content(packet: EvidencePacket) -> EvidencePacket:
        evidence = sorted(
            [*preserved, *packet.evidence],
            key=lambda source: (
                field_order[(source.request_id, source.interaction_id, source.field)],
                source.start,
            ),
        )
        represented = {
            (source.request_id, source.interaction_id, source.field)
            for source in evidence
        }
        return _canonical_packet(
            packet.model_copy(
                update={
                    "evidence": evidence,
                    "points": [*preserved_points(), *packet.points],
                    "omissions": [
                        *packet.omissions,
                        f"Host: {len(represented)} of {len(sources)} original fields represented; selected excerpts may be partial.",
                    ],
                }
            )
        )

    # Cache choices depend on the space occupied by preserved originals, even
    # though these originals are not paid model input or generated testimony.
    preserved_witness = hashlib.sha256(
        EvidencePacket(evidence=preserved).render().encode()
    ).hexdigest()
    result = compress_with_client(
        [source for source in sources if source.source_id not in preserved_ids],
        scope={"org_id": org_id, "project_id": project_id, "user_id": user_id},
        policy="extraction",
        budget=budget,
        client=client,
        context_limits=context_limits,
        before_call=before_call,
        load=load,
        save=save,
        measure_output=lambda packet: packet_tokens(with_original_content(packet)),
        output_contract=f"{EXTRACTION_OUTPUT_CONTRACT}:{preserved_witness}",
    )
    before_call()
    return replace(
        result,
        packet=with_original_content(result.packet),
        source_digest=prepared.source_digest,
        covered_source_ids=tuple(source.source_id for source in sources),
    )
