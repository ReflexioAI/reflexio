"""Inline/manual adapter using the existing retained-output publication owner."""

from __future__ import annotations

from typing import Any

from reflexio.server.llm._litellm_types import (
    ExtractionInputIntegrityError,
    ProviderRequestGuardError,
)
from reflexio.server.llm.token_accounting import sum_trace_tokens
from reflexio.server.llm.tools import ToolLoopResult
from reflexio.server.services.deferred_learning_plan import (
    ExtractorBookmarkPosition,
    PlaybookWritePlan,
)
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
)
from reflexio.server.services.extraction.history_inputs import (
    HistoricalInputs,
    history_from_output,
)
from reflexio.server.services.extraction.prior_answer_search import (
    MAX_KNOWLEDGE_QUERY_CHARS,
)
from reflexio.server.services.extraction.resumable_agent import (
    AgentRunResult,
    ResumableExtractionAgent,
    decode_committed_output,
)
from reflexio.server.services.extraction.source_inputs import (
    capture_source_inputs,
    require_source_inputs,
)
from reflexio.server.services.extractor_config_utils import get_extractor_name
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.server.services.playbook.components.overflow_evidence import (
    historical_playbook_digest,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    build_playbook_prompt_context,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    build_scope_hash,
)
from reflexio.server.services.write_plan_codec import (
    decode_write_plan,
    encode_write_plan,
)

from .input_scope import native_display_fields
from .models import Input, Source
from .native import ProceduralOutput, receipt_to_plan
from .transport import StageClient
from .workflow import digest, run_pipeline


def retained_plan(run: AgentRunRecord) -> PlaybookWritePlan | None:
    """Validate the exact retained inline result; callers still own publication."""
    if (
        run.binding.extractor_kind != "playbook"
        or run.generation_request_snapshot.get("output_schema_name")
        != ProceduralOutput.__name__
        or not isinstance(run.committed_output, dict)
    ):
        raise ExtractionInputIntegrityError(
            "Procedural inline output schema unavailable"
        )
    raw, provenance = decode_committed_output(run.committed_output)
    output = ProceduralOutput.model_validate(raw)
    publication = run.committed_output.get("publication", {})
    sources = publication.get("source_inputs")
    require_source_inputs(sources, run)
    history = history_from_output(run.committed_output)
    if (
        output.source_digest != build_scope_hash(sources)
        or output.source_coverage != [output.source_digest]
        or output.historical_playbooks != history.playbooks
        or output.plan_sha256 != digest(output.write_plan)
        or set(output.receipt["old_state"]) != set(history.playbooks)
    ):
        raise ExtractionInputIntegrityError(
            "Procedural inline input/plan binding changed"
        )
    plan = decode_write_plan(output.write_plan, "playbook")
    if plan is not None and not isinstance(plan, PlaybookWritePlan):
        raise ExtractionInputIntegrityError("Unexpected procedural plan kind")
    settings = run.service_config_snapshot
    if not isinstance(settings, dict):
        raise ExtractionInputIntegrityError(
            "Procedural publication settings unavailable"
        )
    if plan is not None and (
        plan.output_pending_status != settings.get("output_pending_status", False)
        or plan.skip_aggregation != settings.get("skip_aggregation", False)
    ):
        raise ExtractionInputIntegrityError("Procedural publication options changed")
    expected = receipt_to_plan(
        output.receipt,
        user_id=run.binding.user_id or "",
        agent_version=run.binding.agent_version or "",
        request_id=run.binding.request_id,
        source=run.binding.source,
        output_pending_status=plan.output_pending_status if plan else False,
        skip_aggregation=plan.skip_aggregation if plan else False,
        provenance=provenance,
    )
    if (plan is None) != (expected is None):
        raise ExtractionInputIntegrityError("Procedural plan completeness changed")
    if plan is not None and expected is not None:
        # Embeddings/timestamps belong to native preparation. Every business
        # field, consumed identity and lineage remains the reviewed value.
        def business(value: PlaybookWritePlan) -> dict[str, Any]:
            encoded = encode_write_plan(value)
            if encoded is None:
                raise ExtractionInputIntegrityError(
                    "Procedural business plan unavailable"
                )
            for row in encoded["new_playbooks"]:
                if row["user_playbook_id"] not in (None, 0):
                    raise ExtractionInputIntegrityError(
                        "Procedural output already has a stored identity"
                    )
                row.pop("embedding", None)
                row.pop("created_at", None)
            return encoded

        if business(plan) != business(expected):
            raise ExtractionInputIntegrityError(
                "Procedural saved rows differ from reviewed result"
            )
    return plan


def run_inline(extractor: Any, groups: list[Any]) -> AgentRunResult:
    """Retain source, history and reviewed native plan before inline publication."""
    storage = extractor.request_context.storage
    if storage is None:
        raise ExtractionInputIntegrityError("Procedural extraction requires storage")
    current = extractor.service_config
    run = build_extractor_agent_run_record(
        org_id=extractor.request_context.org_id,
        extractor_kind="playbook",
        user_id=current.user_id,
        generation_request_id=current.request_id,
        agent_version=current.agent_version,
        source=current.source,
        request_interaction_data_models=groups,
        extractor_config=extractor.config,
        service_config=current,
        agent_context=extractor.agent_context,
        output_schema_name=ProceduralOutput.__name__,
    )
    sources = capture_source_inputs(storage, run, groups)
    context = build_playbook_prompt_context(groups, label_turns=True)
    visible = {
        row.interaction_id: list(row.evidence_texts)
        for row in context.evidence_sources.values()
    }
    originals = {
        row.interaction_id: (row, group.request)
        for group in groups
        for row in group.interactions
    }
    query = "\n\n".join(piece for pieces in visible.values() for piece in pieces)[
        -MAX_KNOWLEDGE_QUERY_CHARS:
    ]
    from reflexio.models.api_schema.domain.entities import UserPlaybook

    old = PlaybookConsolidator(
        extractor.request_context,
        extractor.client,
        extractor.config.deduplication_config,
    ).retrieve_existing_playbooks(
        [
            UserPlaybook(
                user_id=current.user_id,
                agent_version=current.agent_version,
                request_id=current.request_id,
                trigger=query,
            )
        ],
        user_id=current.user_id,
        agent_version=current.agent_version,
    )
    history = HistoricalInputs(
        playbooks={
            str(row.user_playbook_id): historical_playbook_digest(row) for row in old
        }
    )
    data = Input(
        owner=current.user_id,
        sources=[
            Source(
                interaction_id=row.interaction_id,
                request_id=row.request_id,
                owner=current.user_id,
                role=row.role,
                request_source=row.request_source or "",
                text="\n\n".join(visible[row.interaction_id]),
                created_at=originals[row.interaction_id][0].created_at,
                request_created_at=originals[row.interaction_id][1].created_at,
                piece_fields=native_display_fields(originals[row.interaction_id][0])
                if any(
                    field != "content"
                    for field in native_display_fields(originals[row.interaction_id][0])
                )
                else None,
            )
            for row in context.evidence_sources.values()
        ],
        existing=[row.model_dump(mode="json", exclude={"embedding"}) for row in old],
        agent_context=extractor.agent_context,
        definition=extractor.config.extraction_definition_prompt
        or "Extract useful supported procedural guidance.",
        tool_context="",
    )
    stage_client = StageClient(
        extractor.client, extractor.default_generation_model_name
    )

    def execute(_run: AgentRunRecord) -> ToolLoopResult:
        try:
            receipt = run_pipeline(data, visible, stage_client)
            plan = receipt_to_plan(
                receipt,
                user_id=current.user_id,
                agent_version=current.agent_version,
                request_id=current.request_id,
                source=current.source,
                output_pending_status=getattr(current, "output_pending_status", False),
                skip_aggregation=getattr(current, "skip_aggregation", False),
                provenance=stage_client.latest_provenance,
            )
            if plan is not None and plan.new_playbooks:
                storage.precompute_user_playbook_embeddings(
                    plan.new_playbooks, expand_documents=False
                )
            encoded = encode_write_plan(plan)
            source_digest = build_scope_hash(sources)
            output = ProceduralOutput(
                source_digest=source_digest,
                source_coverage=[source_digest],
                historical_playbooks=history.playbooks,
                receipt=receipt,
                write_plan=encoded,
                plan_sha256=digest(encoded),
            )
            stage_client.trace.finished = True
            return ToolLoopResult(
                ctx=None,
                trace=stage_client.trace,
                finished_reason="structured_output",
                structured_output=output,
                provenance=stage_client.latest_provenance,
            )
        except ProviderRequestGuardError:
            raise
        except Exception as error:
            return ToolLoopResult(
                ctx=None,
                trace=stage_client.trace,
                finished_reason="error",
                error=error,
                error_type=type(error).__name__,
                error_detail="procedural_workflow_failed",
            )

    return ResumableExtractionAgent(client=extractor.client, storage=storage).start(
        run=run,
        publication_origin="initial_inline",
        bookmark_position=ExtractorBookmarkPosition(
            get_extractor_name(extractor.config),
            [item.interaction_id for group in groups for item in group.interactions],
            max(item.created_at for group in groups for item in group.interactions),
            current.user_id,
        ),
        messages=[],
        history_inputs=history,
        output_schema=ProceduralOutput,
        source_inputs=sources,
        workflow=execute,
    )


def extract_inline(extractor: Any, groups: list[Any]) -> list[Any]:
    """Expose normal extraction metadata through the existing extractor outcome."""
    result = run_inline(extractor, groups)
    extractor._last_resumable_run_id = result.run_id
    extractor._last_resumable_run = result.execution_run
    extractor._last_resumable_token_totals = sum_trace_tokens(result.trace)
    extractor._last_model_provenance = result.model_provenance
    if result.output is None or result.execution_run is None:
        raise RuntimeError(
            "Procedural playbook extraction did not complete"
        ) from result.error
    plan = retained_plan(result.execution_run)
    return plan.new_playbooks if plan is not None else []
