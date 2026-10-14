"""Window adapter for shared compression; original extraction storage is intact.

The caller chooses when its rendered downstream prompt needs compression and
supplies the available evidence budget. A result is not a learning/write plan.
"""

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction.input_preparation import (
    ExtractionSources,
    collect_extraction_sources,
)
from reflexio.server.services.extraction.source_package import evidence_from_source_span
from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionError,
    CompressionPoint,
    CompressionResult,
    EvidencePacket,
    packet_tokens,
)
from reflexio.server.services.input_compression_llm import (
    ClientCompressionPlan,
    plan_with_client,
)

EXTRACTION_OUTPUT_CONTRACT = "extraction-original-ranges-content-groups-v4"


@dataclass(frozen=True)
class ExtractionCompressionPlan:
    """A frozen original window and provider-free native compression plan.

    The original collector witness is authoritative, including empty interactions.
    The narrower native cache-source digest is never a substitute for it. Only
    the combined witness belongs in durable admission; source data stays private.
    """

    sources: ExtractionSources = field(repr=False)
    native: ClientCompressionPlan
    witness: str
    preserved_source_ids: tuple[str, ...]
    _execute: Callable[..., CompressionResult] = field(repr=False, compare=False)

    @property
    def source_digest(self) -> str:
        return self.sources.source_digest

    @property
    def source_cache_digest(self) -> str:
        return self.native.compression.source_digest

    def execute(
        self,
        *,
        before_call: Callable[[], None] = lambda: None,
        load: Callable[[str], Mapping[str, Any] | None] = lambda _key: None,
        save: Callable[[str, dict[str, Any]], None] = lambda _key, _value: None,
    ) -> CompressionResult:
        """Check caller authority before/after execution and on each dispatch."""
        return self._execute(before_call=before_call, load=load, save=save)


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


def plan_extraction_window(
    groups: list[RequestInteractionDataModel],
    *,
    org_id: str,
    project_id: str,
    user_id: str,
    budget: CompressionBudget,
    client: LiteLLMClient,
    context_limits: Mapping[str, int],
    measure_output: Callable[[EvidencePacket], int] = packet_tokens,
    estimate_output: Callable[[EvidencePacket], int] | None = None,
    output_contract: str = EXTRACTION_OUTPUT_CONTRACT,
) -> ExtractionCompressionPlan:
    """Collect original fields and plan before legacy geometry or provider I/O.

    The caller supplies already authorized, tenant/project-scoped groups. No
    lease, cache or storage callback runs during preparation. A custom pure
    consumer renderer must have its own versioned output_contract and matching
    token budget; it may count full consumer prompts instead of packet JSON.
    The optional estimate only sizes advisory source menus; exact measurement
    still controls acceptance. Existing packet-budget qualification uses the
    unchanged default v4 contract.
    """
    preparation_started = time.monotonic()
    if any(
        item.interacted_image_url or item.image_encoding
        for group in groups
        for item in group.interactions
    ):
        raise CompressionError("Text compression cannot represent image evidence")
    if not output_contract or (
        (measure_output is not packet_tokens or estimate_output is not None)
        and output_contract == EXTRACTION_OUTPUT_CONTRACT
    ):
        raise CompressionError(
            "Custom extraction renderer requires its own output contract"
        )
    prepared = collect_extraction_sources(
        groups,
        user_id=user_id,
        include_timestamps=True,
        include_request_source=True,
        max_source_bytes=budget.max_source_bytes,
    )
    sources = [evidence_from_source_span(span) for span in prepared.spans]
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

    if (
        preserved
        and measure_output(with_original_content(EvidencePacket()))
        > budget.output_tokens
    ):
        preserved = []
    preserved_ids = {source.source_id for source in preserved}

    # Cache choices depend on the space occupied by preserved originals, even
    # though these originals are not paid model input or generated testimony.
    preserved_witness = hashlib.sha256(
        EvidencePacket(evidence=preserved).render().encode()
    ).hexdigest()
    native = plan_with_client(
        [source for source in sources if source.source_id not in preserved_ids],
        scope={"org_id": org_id, "project_id": project_id, "user_id": user_id},
        policy="extraction",
        budget=budget,
        client=client,
        context_limits=context_limits,
        measure_output=lambda packet: measure_output(with_original_content(packet)),
        estimate_output=(
            (lambda packet: estimate_output(with_original_content(packet)))
            if estimate_output is not None
            else None
        ),
        output_contract=f"{output_contract}:{preserved_witness}",
    )
    witness = hashlib.sha256(
        json.dumps(
            {
                "native": native.witness,
                "original_source_digest": prepared.source_digest,
                "preserved_source_digest": preserved_witness,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    preparation_seconds = time.monotonic() - preparation_started

    def execute(
        *,
        before_call: Callable[[], None],
        load: Callable[[str], Mapping[str, Any] | None],
        save: Callable[[str, dict[str, Any]], None],
    ) -> CompressionResult:
        execution_started = time.monotonic()
        before_call()
        result = native.execute(before_call=before_call, load=load, save=save)
        before_call()
        return replace(
            result,
            packet=with_original_content(result.packet),
            source_digest=prepared.source_digest,
            covered_source_ids=tuple(source.source_id for source in sources),
            elapsed_seconds=preparation_seconds + time.monotonic() - execution_started,
        )

    return ExtractionCompressionPlan(
        prepared,
        native,
        witness,
        tuple(source.source_id for source in preserved),
        execute,
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
    """Prepare and execute the unchanged packet-budget window adapter.

    Use plan_extraction_window for admission planning with explicit downstream
    rendering. Durable quota/checkpoints and publication remain caller-owned.
    """
    return plan_extraction_window(
        groups,
        org_id=org_id,
        project_id=project_id,
        user_id=user_id,
        budget=budget,
        client=client,
        context_limits=context_limits,
    ).execute(before_call=before_call, load=load, save=save)
