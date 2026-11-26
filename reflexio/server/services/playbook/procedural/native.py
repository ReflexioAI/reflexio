"""Durable native execution and publication of the selected procedural method.

The coordinator owns attempts and source/state witnesses; this consumer journals
completed semantic stages and builds an ordinary native write plan. A saved plan
never supplies execution authority: the actual window owner validates it again
inside the existing publication transaction.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from reflexio.models.api_schema.domain.entities import LineageContext, UserPlaybook
from reflexio.server.llm._litellm_types import (
    ExtractionInputIntegrityError,
    ModelProvenance,
)
from reflexio.server.services.deferred_learning_plan import (
    GenerationComputePlan,
    PlaybookWritePlan,
)
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.prior_answer_search import (
    MAX_KNOWLEDGE_QUERY_CHARS,
)
from reflexio.server.services.extraction.resumable_agent import (
    decode_committed_output,
    encode_committed_output,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    build_playbook_prompt_context,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus
from reflexio.server.services.write_plan_codec import (
    decode_write_plan,
    encode_write_plan,
)

from . import workflow
from .catalog import catalog
from .compilation import witnesses
from .models import ContractError, Input, Maintenance, Observation, Review, Source

KEY = "procedural_playbook_v1"
PREPARATION_VERSION = 5


class ProceduralOutput(BaseModel):
    """Retained reviewed output, not a model-authored execution permission."""

    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    source_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_coverage: list[str]
    historical_playbooks: dict[str, str]
    receipt: dict[str, Any]
    write_plan: dict[str, Any] | None
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def accepts_fresh_sources(service: Any, prepared: Any) -> bool:
    """Source-quoted recovery and expert/multimodal work keep independent owners."""
    from reflexio.server.env_utils import env_bool

    from .input_scope import is_ordinary_text

    if len(prepared.chunks) != 1:
        return False
    config = service.service_config
    if config is None or config.window_interactions is None:
        return False
    if not is_ordinary_text(config.window_interactions):
        return False
    return not env_bool("MOCK_LLM_RESPONSE", default=False)


def initial_contract(run: Any) -> dict[str, Any]:
    """Source-preparation v5 is admitted with this exact native parent scope."""
    from reflexio.server.work_scope import current_project_id

    return {
        "version": 1,
        "method": workflow.VERSION,
        "scope": {
            "org_id": run.binding.org_id,
            "project_id": run.project_id
            if run.project_id is not None
            else current_project_id() or "",
            "user_id": run.binding.user_id,
            "run_id": run.id,
            "kind": run.binding.extractor_kind,
            "request_id": run.binding.request_id,
            "agent_version": run.binding.agent_version,
        },
        "reader_steps": {},
    }


def require_preparation_contract(run: Any, manifest: dict[str, Any]) -> None:
    descriptor = manifest.get("checkpoint", {}).get(KEY)
    expected = initial_contract(run)
    if (
        manifest.get("preparation_version") != PREPARATION_VERSION
        or run.binding.extractor_kind != "playbook"
        or not isinstance(descriptor, dict)
        or descriptor.get("adapter") not in (None, "inline")
        or type(descriptor.get("version")) is not int
        or any(
            descriptor.get(key) != expected[key]
            for key in ("version", "method", "scope")
        )
        or run.generation_request_snapshot.get("output_schema_name")
        != ProceduralOutput.__name__
    ):
        raise ExtractionInputIntegrityError("Procedural preparation contract changed")


def is_procedural_window(coordinator: OverflowCoordinator, _service: Any) -> bool:
    """Only an explicitly admitted procedural parent can use this consumer."""
    if coordinator.window.kind != "playbook" or KEY not in coordinator.checkpoint:
        return False
    run = coordinator.saved_run
    if run is None:
        raise ExtractionInputIntegrityError("Procedural preparation parent unavailable")
    require_preparation_contract(run, run.generation_request_snapshot["overflow"])
    return True


def receipt_to_plan(
    receipt: dict[str, Any],
    *,
    user_id: str,
    agent_version: str,
    request_id: str,
    source: str | None,
    output_pending_status: bool,
    skip_aggregation: bool,
    provenance: ModelProvenance | None = None,
) -> PlaybookWritePlan | None:
    """Translate the exact reviewed result without another semantic model call."""
    from reflexio.models.api_schema.domain.entities import Status

    records = receipt["rendered"]
    selected = receipt["accepted"]
    retired = receipt["retired_ids"]
    if (
        receipt.get("method_version") != workflow.VERSION
        or selected != receipt["plan"]["selected"]
        or len(selected) != len(set(selected))
        or set(selected) != set(records)
        or workflow.digest(
            {
                "records": records,
                "keep_existing": receipt["plan"]["keep_existing"],
                "retired_ids": retired,
            }
        )
        != receipt["reviewed_sha256"]
        or ((selected or retired) and receipt["outcome"] != "accept")
    ):
        raise ExtractionInputIntegrityError("Reviewed procedural result changed")
    if not selected and not retired:
        return None
    rows = []
    parents = []
    for key in selected:
        record = records[key]
        rows.append(
            UserPlaybook(
                user_id=user_id,
                agent_version=agent_version,
                request_id=request_id,
                playbook_name="procedural-playbook",
                reader_angle="instruction",
                rationale="See cited evidence.",
                source=source,
                status=Status.PENDING if output_pending_status else None,
                **{
                    field: record[field]
                    for field in (
                        "content",
                        "trigger",
                        "source_interaction_ids",
                        "source_span",
                        "expanded_terms",
                    )
                },
            )
        )
        parents.append(record["replaces"])
    targets = [identity for ids in parents for identity in ids] + retired
    historical = receipt["old_state"]
    if len(targets) != len(set(targets)) or any(
        type(identity) is not int or str(identity) not in historical
        for identity in targets
    ):
        raise ExtractionInputIntegrityError(
            "Procedural targets exceed original history"
        )
    return PlaybookWritePlan(
        request_id=request_id,
        output_pending_status=output_pending_status,
        skip_aggregation=skip_aggregation,
        new_playbooks=rows,
        superseded_ids=targets,
        merge_groups=[(index, ids) for index, ids in enumerate(parents) if ids],
        lineage_contexts=[
            LineageContext(
                op_kind="create",
                actor="procedural_extractor",
                source_ids=[str(identity) for identity in ids],
                request_id=request_id,
                reason="source-backed reviewed complete procedure",
                model_name=provenance.model_name if provenance else None,
                provider=provenance.provider if provenance else None,
            )
            for ids in parents
        ],
        consolidation_provenance=provenance,
    )


class JournalClient:
    """Restore exact completed stages; every physical attempt stays with native guards."""

    def __init__(
        self,
        coordinator: OverflowCoordinator,
        delegate: Any,
        data: Input,
        visible: dict[int, list[str]],
    ):
        self.coordinator, self.delegate = coordinator, delegate
        _, self.quotes, _ = catalog(data, visible)
        self.profiles = deepcopy(delegate.profiles)
        self.model_ladders = deepcopy(delegate.model_ladders)
        self.request_contracts = deepcopy(delegate.request_contracts)
        self.latest_provenance: ModelProvenance | None = None

    def ask[T](
        self,
        stage: str,
        prompt: str,
        body: dict[str, Any],
        models: dict[str, type[BaseModel]],
        assemble: Callable[..., T],
        validate: Callable[[T], Any],
        *,
        evidence_choices: list[str] | None = None,
    ) -> T:
        adapter = TypeAdapter(
            {"source": list[Observation], "maintain": Maintenance, "review": Review}[
                stage
            ]
        )
        key = self.coordinator.step_key(
            "procedural-" + stage,
            {
                "version": workflow.VERSION,
                "prompt": prompt,
                "body": body,
                "models": {
                    name: model.model_json_schema() for name, model in models.items()
                },
                "evidence_choices": evidence_choices,
                "profiles": self.profiles,
                "model_ladders": self.model_ladders,
                "request_contracts": self.request_contracts,
            },
        )
        self.coordinator.validate_current_parent()
        failed = self.coordinator.checkpoint[KEY].get("failed_steps", {}).get(key)
        if failed is not None:
            from .transport import IncompleteBatchError

            failure = (
                IncompleteBatchError
                if failed["error_type"] == "IncompleteBatchError"
                else ContractError
            )
            raise failure("Retained procedural method failure: " + failed["reason"])
        saved = self.coordinator.completed_step(key)
        if saved is None:
            try:
                value = self.delegate.ask(
                    stage,
                    prompt,
                    body,
                    models,
                    assemble,
                    validate,
                    evidence_choices=evidence_choices,
                )
            except (ContractError, ValidationError) as error:
                # Incomplete/invalid terminal declarations are method outcomes,
                # not an invitation to redraw them on the next worker turn.
                # Native provider outages, turn suspensions and guard refusals
                # remain owned by their existing resumable error handling.
                checkpoint = self.coordinator.checkpoint
                checkpoint[KEY].setdefault("failed_steps", {})[key] = {
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "reason": str(error)[:4000],
                }
                self.coordinator.save_checkpoint(checkpoint)
                raise
            provenance = self.delegate.latest_provenance
            if provenance is None:
                raise ExtractionInputIntegrityError(
                    "Procedural stage lacks observed model provenance"
                )
            self.coordinator.complete_step(
                key, adapter.dump_python(value, mode="json"), provenance
            )
            self.latest_provenance = provenance
        else:
            value = adapter.validate_python(saved["output"])
            validate(value)
            self.latest_provenance = ModelProvenance(**saved["model_provenance"])
            self.delegate.latest_provenance = self.latest_provenance
        checkpoint = self.coordinator.checkpoint
        observed = self.latest_provenance
        if observed is None:
            raise ExtractionInputIntegrityError(
                "Procedural stage provenance disappeared"
            )
        checkpoint[KEY]["last_provenance"] = asdict(observed)
        self.coordinator.save_checkpoint(checkpoint)
        if stage == "source":
            for observation in value:
                witnesses(observation.evidence, self.quotes)
            checkpoint = self.coordinator.checkpoint
            checkpoint[KEY]["reader_steps"] = {
                self.coordinator.candidate_key(chunk.text): key
                for chunk in self.coordinator.require_prepared().chunks
            }
            self.coordinator.save_checkpoint(checkpoint)
        return value


def _input(
    coordinator: OverflowCoordinator, service: Any, config: Any
) -> tuple[Input, dict[int, list[str]]]:
    from reflexio.server.services.playbook.components.consolidator import (
        PlaybookConsolidator,
    )

    current = service.service_config
    groups = current.window_interactions
    from reflexio.server.services.extraction.overflow_presentation import (
        current_overflow_presentation,
        present_extraction_chunk,
    )

    prepared = coordinator.require_prepared()
    # Source-quoted recovery owns multiple chunks. This consumer receives the
    # complete admitted single chunk, never the legacy clipped history view.
    if len(prepared.chunks) != 1:
        raise ExtractionInputIntegrityError(
            "Procedural consumer requires its admitted complete source chunk"
        )
    with present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record):
        context = build_playbook_prompt_context(groups, label_turns=True)
        presentation = current_overflow_presentation.get()
        if presentation is None:
            raise ExtractionInputIntegrityError(
                "Native source presentation unavailable"
            )
        piece_fields: dict[int, list[str]] = {}
        for span in presentation.visible_spans:
            piece_fields.setdefault(span.interaction_id, []).append(span.field)
    visible = {
        row.interaction_id: list(row.evidence_texts)
        for row in context.evidence_sources.values()
        if row.evidence_texts
    }
    if not any(piece.strip() for pieces in visible.values() for piece in pieces):
        # Preserve every source witness, but do not ask a model to learn from
        # an entirely blank projection. Mixed useful/blank pieces stay exact.
        visible = {}
    originals = {
        row.interaction_id: (row, group.request)
        for group in groups
        for row in group.interactions
    }
    checkpoint = coordinator.checkpoint
    stored = checkpoint[KEY].get("existing")
    if stored is None:
        query = "\n\n".join(text for pieces in visible.values() for text in pieces)[
            -MAX_KNOWLEDGE_QUERY_CHARS:
        ]
        existing = (
            PlaybookConsolidator(
                service.request_context, service.client, config.deduplication_config
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
            if visible
            else []
        )
        coordinator.remember_historical_playbooks(existing)
        checkpoint = coordinator.checkpoint
        checkpoint[KEY]["existing"] = [
            row.model_dump(mode="json", exclude={"embedding"}) for row in existing
        ]
        checkpoint[KEY]["query_sha256"] = workflow.digest(query)
        coordinator.save_checkpoint(checkpoint)
    else:
        existing = [UserPlaybook.model_validate(row) for row in stored]
        coordinator.remember_historical_playbooks(existing)
    coordinator.storage.validate_extraction_historical_playbooks(
        coordinator.window, coordinator.checkpoint.get("historical_playbooks", {})
    )
    return Input(
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
                piece_fields=piece_fields[row.interaction_id]
                if any(field != "content" for field in piece_fields[row.interaction_id])
                else None,
            )
            for row in context.evidence_sources.values()
            if row.interaction_id in visible
        ],
        existing=[
            row.model_dump(mode="json", exclude={"embedding"}) for row in existing
        ],
        agent_context=coordinator.agent_context,
        definition=config.extraction_definition_prompt
        or "Extract useful supported procedural guidance.",
        tool_context="",
    ), visible


def _retained(coordinator: OverflowCoordinator) -> ProceduralOutput:
    raw = coordinator.checkpoint[KEY].get("output")
    if raw is None:
        raise ExtractionInputIntegrityError("Procedural final plan is unavailable")
    output = ProceduralOutput.model_validate(raw)
    expected = [
        coordinator.candidate_key(chunk.text)
        for chunk in coordinator.require_prepared().chunks
    ]
    steps = coordinator.checkpoint[KEY].get("reader_steps", {})
    blank_source_noop = (
        output.receipt.get("source_projection") == "no_visible_native_evidence"
        and output.write_plan is None
        and not steps
        and coordinator.attempt_count == 0
        and all(
            not span.text.strip()
            for chunk in coordinator.require_prepared().chunks
            for span in (*chunk.neighbors, *chunk.spans)
        )
    )
    if (
        output.source_digest != coordinator.source_digest
        or output.source_coverage != expected
        or (not blank_source_noop and set(steps) != set(expected))
        or any(coordinator.completed_step(key) is None for key in steps.values())
        or output.historical_playbooks
        != coordinator.checkpoint.get("historical_playbooks", {})
        or output.plan_sha256 != workflow.digest(output.write_plan)
    ):
        raise ExtractionInputIntegrityError(
            "Procedural plan source/state binding changed"
        )
    return output


def validate_window_plan(
    coordinator: OverflowCoordinator, service: Any, plan: GenerationComputePlan | None
) -> None:
    """Recheck the exact retained result under the caller's native publication lock."""
    output = _retained(coordinator)
    run = coordinator.saved_run
    if run is None or run.committed_output is None:
        raise ExtractionInputIntegrityError("Procedural parent output is unavailable")
    raw, _ = decode_committed_output(run.committed_output)
    if ProceduralOutput.model_validate(raw) != output:
        raise ExtractionInputIntegrityError("Procedural retained output changed")
    if (
        plan is None
        or plan.extraction_run_ids != [run.id]
        or workflow.digest(encode_write_plan(plan.write_plan)) != output.plan_sha256
        or plan.generated_count
        != len(plan.write_plan.new_playbooks if plan.write_plan else [])
        or plan.billable_count != plan.generated_count
    ):
        raise ExtractionInputIntegrityError(
            "Procedural publication plan changed after review"
        )
    if plan.write_plan is not None and (
        plan.write_plan.request_id != service.service_config.request_id
        or plan.write_plan.output_pending_status != service.output_pending_status
        or plan.write_plan.skip_aggregation != service.skip_aggregation
        or any(
            row.user_id != coordinator.window.user_id
            or row.agent_version != service.service_config.agent_version
            for row in plan.write_plan.new_playbooks
        )
    ):
        raise ExtractionInputIntegrityError("Procedural publication scope changed")


def compute_window(
    coordinator: OverflowCoordinator, service: Any, config: Any
) -> GenerationComputePlan:
    """Run/restore the native three-stage consumer without an extra binary vote."""
    from reflexio.server.services.base_generation_service import PreparedGenerationRun
    from reflexio.server.services.durable_learning.user_lease import (
        owned_agent_run_write,
    )
    from reflexio.server.services.extractor_config_utils import get_extractor_name

    from .transport import StageClient

    started = time.perf_counter()
    from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name

    root_config = service.request_context.configurator.get_config()
    model = resolve_model_name(
        ModelRole.GENERATION,
        llm_config=root_config.llm_config if root_config else None,
        api_key_config=root_config.api_key_config if root_config else None,
    )
    delegate = StageClient(service.client, model=model)
    checkpoint = coordinator.checkpoint
    if KEY not in checkpoint:
        raise ExtractionInputIntegrityError("Procedural consumer was not admitted")
    if "profiles" not in checkpoint[KEY]:
        if coordinator.attempt_count:
            raise ExtractionInputIntegrityError(
                "Procedural profiles unavailable after dispatch"
            )
        checkpoint[KEY].update(
            {
                "profiles": deepcopy(delegate.profiles),
                "model_ladders": deepcopy(delegate.model_ladders),
                "request_contracts": deepcopy(delegate.request_contracts),
                "attempt_start": coordinator.attempt_count,
            }
        )
        coordinator.save_checkpoint(checkpoint)
    elif (
        checkpoint[KEY]["profiles"] != delegate.profiles
        or checkpoint[KEY]["model_ladders"] != delegate.model_ladders
        or checkpoint[KEY].get("request_contracts") != delegate.request_contracts
    ):
        raise ExtractionInputIntegrityError("Procedural stage model settings changed")
    # Restore the selected operation's finite physical-call allowance; a worker
    # turn or process restart never buys another six dispatches.
    delegate.physical_calls = (
        coordinator.attempt_count - checkpoint[KEY]["attempt_start"]
    )
    delegate.calls = delegate.physical_calls
    if checkpoint[KEY].get("output") is None:
        data, visible = _input(coordinator, service, config)
        journal = JournalClient(coordinator, delegate, data, visible)
        receipt = workflow.run_pipeline(data, visible, journal)
        current = service.service_config
        plan = receipt_to_plan(
            receipt,
            user_id=current.user_id,
            agent_version=current.agent_version,
            request_id=current.request_id,
            source=current.source,
            output_pending_status=service.output_pending_status,
            skip_aggregation=service.skip_aggregation,
            provenance=journal.latest_provenance,
        )
        if plan is not None and plan.new_playbooks:
            coordinator.storage.precompute_user_playbook_embeddings(
                plan.new_playbooks, expand_documents=False
            )
        encoded = encode_write_plan(plan)
        checkpoint = coordinator.checkpoint
        output = ProceduralOutput(
            source_digest=coordinator.source_digest,
            source_coverage=[
                coordinator.candidate_key(chunk.text)
                for chunk in coordinator.require_prepared().chunks
            ],
            historical_playbooks=checkpoint.get("historical_playbooks", {}),
            receipt=receipt,
            write_plan=encoded,
            plan_sha256=workflow.digest(encoded),
        )
        checkpoint[KEY]["output"] = output.model_dump(mode="json")
        checkpoint["source_coverage"] = output.source_coverage
        checkpoint["stage"] = "extracted"
        coordinator.save_checkpoint(checkpoint)
    output = _retained(coordinator)
    recorded_provenance = coordinator.checkpoint[KEY].get("last_provenance")
    if recorded_provenance is not None:
        delegate.latest_provenance = ModelProvenance(**recorded_provenance)
    run = coordinator.saved_run
    if run is None:
        raise ExtractionInputIntegrityError("Procedural parent claim unavailable")
    with owned_agent_run_write(coordinator.storage, run):
        current_run = coordinator.storage.get_agent_run(run.id)
        if current_run is None:
            raise ExtractionInputIntegrityError("Procedural parent disappeared")
        if current_run.committed_output is None:
            updated = coordinator.storage.update_agent_run_status(
                run.id,
                AgentRunStatus.AGENT_COMPLETED,
                committed_output=encode_committed_output(
                    output, delegate.latest_provenance
                ),
                expected_statuses=(current_run.status,),
                expected_claimed_by=current_run.claimed_by,
                expected_claimed_at=current_run.claimed_at,
            )
            if updated is None or updated.status != AgentRunStatus.AGENT_COMPLETED:
                raise ExtractionInputIntegrityError(
                    "Procedural completion lost ownership"
                )
            current_run = updated
        coordinator.saved_run = current_run
    native_plan = decode_write_plan(output.write_plan, "playbook")
    if native_plan is not None and not isinstance(native_plan, PlaybookWritePlan):
        raise ExtractionInputIntegrityError("Procedural output changed its plan kind")
    count = len(native_plan.new_playbooks) if native_plan is not None else 0
    service._last_extraction_run_ids = [current_run.id]
    service._last_extraction_run = current_run
    service._last_model_provenance = delegate.latest_provenance
    service._last_token_totals = coordinator.token_totals
    service._last_extractor_run_stats = {"total": 1, "failed": 0, "timed_out": 0}
    result = GenerationComputePlan(
        prepared=PreparedGenerationRun(
            config, get_extractor_name(config), coordinator.window.user_id
        ),
        generated_count=count,
        billable_count=count,
        write_plan=native_plan,
        bookmark_advance=None,
        generation_start=started,
        extraction_run_ids=[current_run.id],
        token_totals=coordinator.token_totals,
        execution_run=current_run,
    )
    validate_window_plan(coordinator, service, result)
    return result
