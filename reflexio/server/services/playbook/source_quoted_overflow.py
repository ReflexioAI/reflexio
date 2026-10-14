"""Recover overflow instructions without publishing model-authored prose.

Selection is semantic and may omit useful instructions. Text and attribution are
mechanical: copy complete original User messages, never selected sub-sentences.
No existing learning is rewritten, retired or judged semantically redundant.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import tiktoken

from reflexio.models.api_schema.domain.entities import LineageContext, UserPlaybook
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
)
from reflexio.server.services.extraction.input_preparation import (
    PreparedExtractionInput,
    source_witness_bytes,
)
from reflexio.server.services.extraction.protected_record import (
    ProtectedRecord,
    ProtectedRecordUpdate,
)

if TYPE_CHECKING:
    from reflexio.models.config_schema import UserPlaybookExtractorConfig
    from reflexio.server.services.deferred_learning_plan import GenerationComputePlan
    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.playbook.service import PlaybookGenerationService

POLICY = "source-quoted-playbooks-v1"
# The existing extraction output reservation, also bounding recovered publication.
PUBLICATION_TOKENS = 8192


def is_source_quoted_playbook(row: UserPlaybook) -> bool:
    """Persisted policy marker also excludes automatic generative successors."""
    try:
        metadata = json.loads(row.notes or "null")
    except (ValueError, TypeError):
        return False
    return isinstance(metadata, dict) and metadata.get("policy") == POLICY


@dataclass(frozen=True)
class SourceMessage:
    interaction_id: int
    request_id: str
    role: str
    source_order: int
    content: str


def project_source_messages(
    prepared: PreparedExtractionInput,
    groups: list[RequestInteractionDataModel],
    record: ProtectedRecord,
    *,
    user_id: str,
) -> list[SourceMessage]:
    """Copy selected complete messages; exclude other actors/fields and oversize.

    A reference can select a message, but cannot remove its negation, conditions
    or surrounding context. Duplicate selections of one source copy it once.
    """
    witness = hashlib.sha256()
    originals = {}
    for group in groups:
        if group.request.user_id != user_id:
            raise ValueError("Source projection request owner changed")
        for item in group.interactions:
            if (
                item.user_id != user_id
                or item.request_id != group.request.request_id
                or item.interaction_id in originals
            ):
                raise ValueError("Source projection interaction owner changed")
            originals[item.interaction_id] = item
            witness.update(source_witness_bytes(group.request, item))
    if witness.hexdigest() != prepared.source_digest:
        raise ValueError("Source projection witness changed")
    spans = tuple(
        s for chunk in prepared.chunks for s in (*chunk.neighbors, *chunk.spans)
    )
    checked = ProtectedRecord().merge(
        ProtectedRecordUpdate(evidence=list(record.evidence)), visible_spans=spans
    )
    selected: dict[int, SourceMessage] = {}
    from reflexio.server.services.extraction.input_preparation import (
        resolve_evidence_span,
    )

    for evidence in checked.evidence:
        if evidence.kind not in {"instruction", "correction"}:
            continue
        span = resolve_evidence_span(evidence.evidence_refs[0], spans)
        if span.role.casefold() != "user" or span.field != "content":
            continue
        item = originals[span.interaction_id]
        if item.role != span.role or item.request_id != span.request_id:
            raise ValueError("Source projection attribution changed")
        selected[item.interaction_id] = SourceMessage(
            item.interaction_id,
            item.request_id,
            item.role,
            span.source_order,
            item.content,
        )
    encoding = tiktoken.get_encoding("cl100k_base")
    result = []
    for message in sorted(selected.values(), key=lambda item: item.source_order):
        cost = len(
            encoding.encode(
                json.dumps(asdict(message), ensure_ascii=False), disallowed_special=()
            )
        )
        if not message.content.strip() or cost > PUBLICATION_TOKENS:
            continue  # Conservative omission; never publish a truncated instruction.
        result.append(message)
    return result


def is_source_quoted_window(coordinator: OverflowCoordinator) -> bool:
    return (
        coordinator.window.kind == "playbook"
        and len(coordinator.require_prepared().chunks) > 1
    )


def _projection(
    coordinator: OverflowCoordinator, service: PlaybookGenerationService
) -> tuple[list[UserPlaybook], dict[str, Any]]:
    config = service.service_config
    if config is None or config.window_interactions is None:
        raise InputBudgetConfigurationError(
            "Source projection requires a scoped window"
        )
    if (
        config.user_id != coordinator.window.user_id
        or not config.window_interactions
        or config.request_id != config.window_interactions[-1].request.request_id
        or config.agent_version != config.window_interactions[-1].request.agent_version
    ):
        raise InputBudgetConfigurationError(
            "Source projection generation binding changed"
        )
    prepared = coordinator.require_prepared()
    if (
        coordinator.checkpoint.get("active_child") is not None
        or coordinator.checkpoint.get("waiting_child") is not None
    ):
        raise InputBudgetConfigurationError(
            "Source projection cannot bypass a child continuation"
        )
    coverage = [coordinator.candidate_key(chunk.text) for chunk in prepared.chunks]
    if (
        not is_source_quoted_window(coordinator)
        or coordinator.checkpoint.get("protected_coverage") != coverage
    ):
        raise InputBudgetConfigurationError(
            "Source projection requires a complete protected scan"
        )
    try:
        messages = project_source_messages(
            prepared,
            config.window_interactions,
            coordinator.record,
            user_id=coordinator.window.user_id,
        )
    except ValueError as exc:
        # A persisted record can pass structural parsing but reference an
        # unshown source or invalid quote. This pure validation failure must
        # use the typed configuration path, preserving the saved journal.
        raise InputBudgetConfigurationError(
            "Source projection evidence or source witness is invalid"
        ) from exc
    rows = [
        UserPlaybook(
            user_id=config.user_id,
            agent_version=config.agent_version,
            request_id=config.request_id,
            playbook_name="source instruction",
            content=message.content,
            trigger=message.content,  # Existing search embeds the trigger only.
            source_span=message.content,
            source_interaction_ids=[message.interaction_id],
            source=config.source,
            notes=json.dumps(
                {
                    "policy": POLICY,
                    "source_request_id": message.request_id,
                    "source_role": message.role,
                    "source_order": message.source_order,
                    "provenance": "Copied original transcript content; human authorship and execution are not inferred.",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        for message in messages
    ]
    # Account for the actual row representation, including the repeated source
    # text and neutral provenance. Dropping one whole message leaves later small
    # messages eligible; no instruction is clipped to fill the remaining space.
    bounded: list[UserPlaybook] = []
    encoding = tiktoken.get_encoding("cl100k_base")
    for row in rows:
        serialized = json.dumps(
            [_row_witness(r) for r in [*bounded, row]],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if (
            len(encoding.encode(serialized, disallowed_special=()))
            <= PUBLICATION_TOKENS
        ):
            bounded.append(row)
    rows = bounded
    receipt = {
        "policy": POLICY,
        "source_digest": coordinator.source_digest,
        "window_id": coordinator.window.window_id,
        "org_id": coordinator.storage.org_id,
        "user_id": config.user_id,
        "project_id": coordinator.window.project_id,
        "agent_version": config.agent_version,
        "request_id": config.request_id,
        "protected_digest": coordinator.step_key(
            POLICY, coordinator.record.model_dump(mode="json")
        ),
        "rows": [_row_witness(row) for row in rows],
    }
    return rows, receipt


def _row_witness(row: UserPlaybook) -> dict[str, Any]:
    # Identity and clock are assigned later; vectors have a separate digest and
    # postcommit tagging may classify rows. Behavioral text stays sealed.
    return row.model_dump(
        mode="json", exclude={"user_playbook_id", "created_at", "embedding", "tags"}
    )


def _embedding_digest(row: UserPlaybook) -> str:
    """Bind precomputed search vectors carried by the durable cached plan."""
    if row.embedding is None or len(row.embedding) != EMBEDDING_DIMENSIONS:
        raise InputBudgetConfigurationError("Source projection lacks a search vector")
    try:
        if any(isinstance(value, bool) for value in row.embedding):
            raise ValueError("Boolean vector component")
        encoded = json.dumps(
            [
                struct.unpack("!f", struct.pack("!f", float(value)))[0]
                for value in row.embedding
            ],
            allow_nan=False,
            separators=(",", ":"),
        )
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputBudgetConfigurationError(
            "Source projection has an invalid search vector"
        ) from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_source_quoted_plan(
    coordinator: OverflowCoordinator,
    service: PlaybookGenerationService,
    config: UserPlaybookExtractorConfig,
) -> GenerationComputePlan:
    from reflexio.server.services.base_generation_service import PreparedGenerationRun
    from reflexio.server.services.deferred_learning_plan import (
        GenerationComputePlan,
        PlaybookWritePlan,
    )
    from reflexio.server.services.extractor_config_utils import get_extractor_name
    from reflexio.server.services.storage.storage_base import AgentRunStatus

    rows, receipt = _projection(coordinator, service)
    checkpoint = coordinator.checkpoint
    previous = checkpoint.get("source_quoted_publication")
    if previous is not None:
        vectors = checkpoint.get("source_quoted_vectors")
        if (
            not isinstance(vectors, list)
            or len(vectors) != len(rows)
            or any(not isinstance(vector, list) for vector in vectors)
        ):
            raise InputBudgetConfigurationError(
                "Source projection lost its sealed search vectors"
            )
        for row, vector in zip(rows, vectors, strict=True):
            row.embedding = vector
    elif rows:
        coordinator.storage.precompute_user_playbook_embeddings(
            rows, expand_documents=False
        )
        if any(not row.embedding for row in rows):
            # SQLite exposes a temporary provider outage as an empty vector.
            # No publication has been sealed yet, so leave this retryable.
            raise EmbeddingUnavailableError(
                "Source projection search vectors are unavailable"
            )
    receipt["embedding_digests"] = [_embedding_digest(row) for row in rows]
    if previous is not None and previous != receipt:
        raise InputBudgetConfigurationError(
            "Source projection publication policy changed"
        )
    checkpoint["source_quoted_publication"] = receipt
    checkpoint["source_quoted_vectors"] = [list(row.embedding) for row in rows]
    checkpoint["source_coverage"] = [
        coordinator.candidate_key(chunk.text)
        for chunk in coordinator.require_prepared().chunks
    ]
    checkpoint["stage"] = "extracted"
    coordinator.save_checkpoint(checkpoint)
    run_id = f"window:{coordinator.window.window_id}"
    with coordinator.storage.commit_scope():
        coordinator.storage.fence_user_extraction(
            coordinator.window.user_id, coordinator.token
        )
        coordinator.storage.validate_extraction_source_digest(
            coordinator.window, coordinator.source_digest
        )
        run = coordinator.storage.get_agent_run(run_id)
        if run is None or run.pending_tool_call_ids:
            raise InputBudgetConfigurationError(
                "Source projection parent is unavailable or awaiting input"
            )
        if run.status in {AgentRunStatus.RUNNING, AgentRunStatus.FAILED}:
            updated = coordinator.storage.update_agent_run_status(
                run_id, AgentRunStatus.AGENT_COMPLETED, expected_statuses=(run.status,)
            )
            if updated is None or updated.status != AgentRunStatus.AGENT_COMPLETED:
                raise InputBudgetConfigurationError(
                    "Source projection parent lost ownership"
                )
        elif run.status != AgentRunStatus.AGENT_COMPLETED:
            raise InputBudgetConfigurationError(
                "Source projection parent cannot publish"
            )
    service._last_extraction_run_ids = [run_id]
    service._last_model_provenance = None  # Selection did not author the copied text.
    service._last_token_totals = coordinator.token_totals
    service._last_extractor_run_stats = {"total": 1, "failed": 0, "timed_out": 0}
    write_plan = (
        PlaybookWritePlan(
            request_id=receipt["request_id"],
            output_pending_status=False,
            skip_aggregation=service.skip_aggregation,
            new_playbooks=rows,
            superseded_ids=[],
            merge_groups=[],
            lineage_contexts=[
                LineageContext(
                    op_kind="create",
                    actor="source_projection",
                    request_id=receipt["request_id"],
                    source_ids=[str(value) for value in row.source_interaction_ids],
                )
                for row in rows
            ],
        )
        if rows
        else None
    )
    plan = GenerationComputePlan(
        prepared=PreparedGenerationRun(
            config, get_extractor_name(config), receipt["user_id"]
        ),
        generated_count=len(rows),
        billable_count=len(rows),
        write_plan=write_plan,
        bookmark_advance=None,
        generation_start=time.perf_counter(),
        extraction_run_ids=[run_id],
        token_totals=coordinator.token_totals,
    )
    validate_source_quoted_plan(coordinator, service, plan)
    return plan


def validate_source_quoted_plan(
    coordinator: OverflowCoordinator,
    service: PlaybookGenerationService,
    plan: GenerationComputePlan | None,
) -> None:
    """Reject old freeform cached plans and any mutation after projection."""
    rows, expected = _projection(coordinator, service)
    write = plan.write_plan if plan is not None else None
    expected["embedding_digests"] = (
        [_embedding_digest(row) for row in write.new_playbooks]
        if write is not None
        else []
    )
    if (
        coordinator.checkpoint.get("source_quoted_publication") != expected
        or plan is None
        or plan.generated_count != len(rows)
        or plan.billable_count != len(rows)
        or plan.bookmark_advance is not None
        or plan.extraction_run_ids != [f"window:{coordinator.window.window_id}"]
        or (bool(rows) != (write is not None))
        or (
            write is not None
            and (
                [_row_witness(row) for row in write.new_playbooks] != expected["rows"]
                or write.superseded_ids != []
                or write.merge_groups != []
                or write.request_id != expected["request_id"]
                or write.output_pending_status
                or write.skip_aggregation != service.skip_aggregation
                or any(row.tags is not None for row in write.new_playbooks)
                or any(row.user_playbook_id != 0 for row in write.new_playbooks)
                or write.lineage_contexts
                != [
                    LineageContext(
                        op_kind="create",
                        actor="source_projection",
                        request_id=expected["request_id"],
                        source_ids=[str(value) for value in row.source_interaction_ids],
                    )
                    for row in rows
                ]
            )
        )
    ):
        raise InputBudgetConfigurationError(
            "Write plan lacks the exact source-quoted publication receipt"
        )
