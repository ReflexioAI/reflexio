"""Session adapter for the shared, separately qualified compression protocol.

This returns an analyst EvidenceDocument with original quoted spans and a
non-citable summary. It never changes outcome, exposure or publication state.
The legacy compactor/journals remain on their existing versioned protocol.
"""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from typing import Any

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionError,
    CompressionResult,
    EvidencePacket,
    EvidenceSource,
)
from reflexio.server.services.input_compression_llm import (
    compress_with_client,
    plan_with_client,
)
from reflexio.server.services.offline_tuner.open_world.analyst import (
    _canonical_user_message,
)
from reflexio.server.services.offline_tuner.open_world.identity import canonical_payload
from reflexio.server.services.offline_tuner.open_world.models import (
    EvidenceDocument,
    EvidenceSpan,
)
from reflexio.server.services.offline_tuner.open_world.shared_compaction import (
    SharedCompactionJournal,
)
from reflexio.server.services.storage.session_outcome_identity import (
    CanonicalSessionTrajectory,
    canonical_trajectory_bytes,
)

from .evidence_documents import _span_id, project_trajectory_spans

SESSION_OUTPUT_CONTRACT = "offline-analyst-canonical-base64-groups-v2-reserve750"


def _source_key(span: EvidenceSpan) -> str:
    # Span IDs bind the entire trajectory and change when the session appends.
    # Cache individual immutable sources independently; final citable spans are
    # always rebound to the caller's currently authorized frozen trajectory.
    fields = asdict(span)
    fields.pop("span_id")
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def compress_session_evidence(
    document: EvidenceDocument,
    *,
    project_id: str,
    through_request_id: str,
    frozen_trajectory: CanonicalSessionTrajectory,
    governance_epochs: tuple[tuple[str, int], ...],
    budget: CompressionBudget,
    client: LiteLLMClient,
    context_limits: Mapping[str, int],
    before_call: Callable[[], None],
    load: Callable[[str], Mapping[str, Any] | None] = lambda _key: None,
    save: Callable[[str, dict[str, Any]], None] = lambda _key, _value: None,
    journal: SharedCompactionJournal | None = None,
    measure_document: Callable[[EvidenceDocument], int] | None = None,
    consumer_contract: str = "",
) -> tuple[EvidenceDocument, CompressionResult]:
    """Compress only this already authorized and frozen session prefix.

    Do not pass labels, target playbooks, hypotheses or discovery assignments to
    the compressor. The source reader still owns prefix completeness and scope.
    Supply its already-read canonical trajectory, including empty terminal
    requests; the final source identity binds that verified cutover.
    Callers must use a new protocol receipt namespace, not legacy v6 checkpoints.
    """
    if not document.spans or not through_request_id:
        raise CompressionError("Session compression requires frozen source evidence")
    tenants = {span.tenant_id for span in document.spans}
    if len(tenants) != 1:
        raise CompressionError("Session compression contains mixed tenant evidence")
    trajectory_digest = hashlib.sha256(
        canonical_trajectory_bytes(frozen_trajectory)
    ).hexdigest()
    requests = frozen_trajectory["requests"]
    if (
        trajectory_digest != document.trajectory_digest
        or frozen_trajectory["session_id"] != document.session_id
        or not requests
        or requests[-1]["request"]["request_id"] != through_request_id
        or any(
            request["request"]["user_id"] != document.user_id
            or request["request"]["session_id"] != document.session_id
            or any(
                item["user_id"] != document.user_id
                or item["request_id"] != request["request"]["request_id"]
                for item in request["interactions"]
            )
            for request in requests
        )
        or project_trajectory_spans(
            frozen_trajectory,
            trajectory_digest=trajectory_digest,
            tenant_id=next(iter(tenants)),
            role=document.manifest_role,
            user_id=document.user_id,
            session_id=document.session_id,
            include_tools=True,
        )
        != document.spans
    ):
        raise CompressionError(
            "Session compression requires its exact frozen trajectory and terminal prefix"
        )
    before_call()
    sources = [
        EvidenceSource(
            source_id=_source_key(span),
            request_id=span.request_id,
            interaction_id=span.interaction_id,
            field=span.field_path,
            speaker=span.speaker,
            source_order=index,
            text=span.immutable_text,
        )
        for index, span in enumerate(document.spans)
    ]
    # Governance belongs in every cached section's authority, while trajectory
    # digest/prefix stay in the final document and are never model testimony.
    epochs = hashlib.sha256(json.dumps(sorted(governance_epochs)).encode()).hexdigest()

    if measure_document is not None and not consumer_contract:
        raise CompressionError("Native consumer sizing requires its frozen contract")

    def render_size(source: EvidenceDocument, packet: EvidencePacket) -> int:
        if measure_document is not None:
            return measure_document(_apply_packet(source, packet))
        return (
            count_input_tokens(
                _canonical_user_message(
                    {
                        "document": canonical_payload(_apply_packet(source, packet)),
                    }
                )
            )
            + 750
        )

    # Advisory prices must not invalidate unchanged leaves just because an
    # append changes the final trajectory/citation hashes. Actual admission
    # below still measures the real authorized document, never this neutral DTO.
    pricing_document = replace(
        document,
        trajectory_digest="0" * 64,
        session_outcome="unknown" if document.manifest_role == "discovery" else None,
        spans=tuple(
            replace(span, span_id=_source_key(span)) for span in document.spans
        ),
    )

    options: dict[str, Any] = {
        "scope": {
            "org_id": next(iter(tenants)),
            "project_id": project_id,
            "user_id": document.user_id,
            "session_id": document.session_id,
            "governance": epochs,
        },
        "policy": "offline_tuning",
        "budget": budget,
        "client": client,
        "context_limits": context_limits,
        "measure_output": lambda packet: render_size(document, packet),
        "estimate_output": lambda packet: render_size(pricing_document, packet),
        "output_contract": SESSION_OUTPUT_CONTRACT
        + (":" + consumer_contract if consumer_contract else ""),
    }
    if journal is None:
        result = compress_with_client(
            sources, before_call=before_call, load=load, save=save, **options
        )
    else:
        plan = plan_with_client(sources, **options)
        journal.bind(plan)
        with journal.provider_scope():
            result = plan.execute(
                before_call=before_call,
                load=journal.load,
                save=journal.save,
                inspect_node=journal.inspect,
            )
    before_call()
    # Leaf receipts remain tied to immutable authorized fields for append reuse.
    # The final source identity additionally binds the complete frozen cutover,
    # including a terminal request with no citable text.
    identity = hashlib.sha256(
        canonical_trajectory_bytes(
            {
                "schema_version": "shared-session-prefix-v1",
                "source_digest": result.source_digest,
                "trajectory_digest": trajectory_digest,
                "through_request_id": through_request_id,
                "tenant_id": next(iter(tenants)),
                "project_id": project_id,
                "user_id": document.user_id,
                "session_id": document.session_id,
                "governance_epochs": sorted(governance_epochs),
            }
        )
    ).hexdigest()
    prepared = _apply_packet(document, result.packet)
    if journal is not None:
        journal.finish()
    return prepared, replace(result, source_digest=identity)


def _apply_packet(
    document: EvidenceDocument, packet: EvidencePacket
) -> EvidenceDocument:
    if (
        not packet.points
        and not packet.overview
        and len(packet.evidence) == len(document.spans)
        and all(
            e.source_id == _source_key(s) and e.text == s.immutable_text
            for e, s in zip(packet.evidence, document.spans, strict=True)
        )
    ):
        return document
    originals = {_source_key(span): span for span in document.spans}
    spans = []
    evidence_indices = {}
    for evidence in packet.evidence:
        # Fragments have a server-generated suffix; resolve using the complete
        # original identity, not a model-supplied source key or summary detail.
        original = originals.get(evidence.source_id) or originals.get(
            evidence.source_id.rsplit(":", 2)[0]
        )
        if (
            original is None
            or original.immutable_text[
                evidence.start : evidence.start + len(evidence.text)
            ]
            != evidence.text
        ):
            raise CompressionError("Compacted session lost its original source")
        field_path = f"{original.field_path}[{evidence.start}:{evidence.start + len(evidence.text)}]"
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
                immutable_text=evidence.text,
                speaker=original.speaker,
            ),
            field_path=field_path,
            immutable_text=evidence.text,
        )
        spans.append(span)
        evidence_indices[evidence.source_id] = len(spans) - 1
    # Original passages already contain the facts. Repeating generated prose
    # and full citation IDs inside a base64 summary wastes the analyst budget.
    # Keep point grouping (including all qualifiers) as non-citable navigation.
    groups: dict[str, list[list[int]]] = {}
    for point in packet.points:
        groups.setdefault(point.category, []).append(
            [evidence_indices[ref] for ref in point.evidence_refs]
        )
    return replace(
        document,
        spans=tuple(spans),
        compaction_summary=json.dumps(
            {
                "evidence_groups": groups,
                "unresolved": packet.unresolved,
                "omissions": packet.omissions,
                "omission_counts": packet.omission_counts.model_dump(),
            },
            ensure_ascii=False,
        ),
    )
