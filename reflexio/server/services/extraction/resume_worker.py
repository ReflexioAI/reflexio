"""Follow-up resume worker for resolved async extraction tool calls."""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import PlaybookConfig, ProfileExtractorConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.error_reporting import error_tags
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.services.extraction.agent_run_records import build_scope_hash
from reflexio.server.services.extraction.answer_inputs import answer_input_scope
from reflexio.server.services.extraction.history_inputs import HistoricalInputs
from reflexio.server.services.extraction.pending_tool_admission import (
    pending_tool_calls_enabled,
)
from reflexio.server.services.extraction.pending_tool_call_dispatch import (
    PendingToolCallToolContext,
)
from reflexio.server.services.extraction.prior_answer_search import (
    append_prior_knowledge_context,
)
from reflexio.server.services.extraction.publication import (
    PreparedPublication,
    publish_resumed_output,
)
from reflexio.server.services.extraction.resumable_agent import (
    ResumableExtractionAgent,
    create_pending_info_tools_for_extractor_kind,
    decode_committed_output,
)
from reflexio.server.services.extraction.source_inputs import capture_source_inputs
from reflexio.server.services.playbook.components.extractor import PlaybookExtractor
from reflexio.server.services.playbook.playbook_evidence import (
    build_playbook_extraction_contract,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredExtractedPlaybookList,
    StructuredPlaybookList,
    StructuredReferencedExtractedPlaybookList,
    build_playbook_prompt_context,
    construct_expert_playbook_extraction_messages,
    construct_playbook_extraction_messages_from_sessions,
    has_expert_content,
    uses_evidence_grounded_extraction,
)
from reflexio.server.services.playbook.review_window import (
    infer_playbook_review_user_id,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.profile.components.extractor import (
    MAX_EXISTING_PROFILES_FOR_CONTEXT,
    ProfileExtractor,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
    prepare_profile_extraction_prompt,
)
from reflexio.server.services.profile.service import (
    ProfileGenerationService,
    ProfileGenerationServiceConfig,
)
from reflexio.server.services.service_utils import (
    extract_interactions_from_request_interaction_data_models,
)
from reflexio.server.services.storage.error import (
    ReadinessUnavailableError,
    execution_input_is_busy,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    AgentRunStatus,
    BaseStorage,
    LeaseLostError,
    PendingToolCallRecord,
    PendingToolCallStatus,
)
from reflexio.server.services.storage.storage_base.agent_run._agent_run_store import (
    is_overflow_protocol_run,
)
from reflexio.server.services.tagging.tagging_scheduler import schedule_tagging
from reflexio.server.site_var.site_var_manager import SiteVarManager
from reflexio.server.work_scope import WorkScope, bind_work_scope

logger = logging.getLogger(__name__)


def _refuse_overflow_protocol_run(run: AgentRunRecord) -> None:
    if is_overflow_protocol_run(run):
        from reflexio.server.services.extraction.recovery import RecoveryRefusedError

        raise RecoveryRefusedError("overflow_protocol_recovery_unsupported")


class ResumeWorkerError(RuntimeError):
    """Base error for retryable resume-worker failures."""


def _next_retry_at(attempt_count: int) -> datetime:
    delay_seconds = min(300, max(1, 2 ** max(0, attempt_count - 1)))
    return datetime.now(UTC) + timedelta(seconds=delay_seconds)


def _create_llm_client(request_context: RequestContext) -> LiteLLMClient:
    # The tool loop re-resolves the model per call via ModelRole.EXTRACTION_AGENT
    # (see run_tool_loop); this client's model name is only a fallback. We resolve
    # the GENERATION model here so the client still carries a sensible default and
    # the org's api_key_config, which the loop reuses for provider routing.
    model_setting = SiteVarManager().get_site_var("llm_model_setting")
    site_var = model_setting if isinstance(model_setting, dict) else {}
    config = request_context.configurator.get_config()
    api_key_config = config.api_key_config if config else None
    llm_config = config.llm_config if config else None
    model_name = resolve_model_name(
        ModelRole.GENERATION,
        site_var_value=site_var.get("default_generation_model_name"),
        llm_config=llm_config,
        api_key_config=api_key_config,
    )
    return LiteLLMClient(
        LiteLLMConfig(
            model=model_name,
            api_key_config=api_key_config,
            max_tokens=llm_config.generation_max_output_tokens if llm_config else None,
        )
    )


def _storage(request_context: RequestContext) -> BaseStorage:
    storage = request_context.storage
    if storage is None:
        raise ResumeWorkerError("Storage not configured")
    return storage


def _request_interaction_models_from_ids(
    storage: BaseStorage,
    interaction_ids: list[int],
    *,
    fallback_source: str | None,
    fallback_agent_version: str | None,
) -> list[RequestInteractionDataModel]:
    interactions = storage.get_interactions_by_ids(interaction_ids)
    if not interactions:
        raise ResumeWorkerError("No source interactions found for resumed run")

    by_request: dict[str, list[Interaction]] = defaultdict(list)
    for interaction in interactions:
        by_request[interaction.request_id].append(interaction)

    models: list[RequestInteractionDataModel] = []
    for request_id, request_interactions in by_request.items():
        ordered = sorted(
            request_interactions,
            key=lambda item: (item.created_at, item.interaction_id),
        )
        request = storage.get_request(request_id)
        fallback_request = request is None
        if request is None:
            first = ordered[0]
            request = Request(
                request_id=request_id,
                user_id=first.user_id,
                created_at=first.created_at,
                source=fallback_source or "",
                agent_version=fallback_agent_version or "",
                session_id=request_id,
            )
        model = RequestInteractionDataModel(
            session_id=request.session_id,
            request=request,
            interactions=ordered,
        )
        model._fallback_request = fallback_request
        models.append(model)
    return sorted(
        models,
        key=lambda model: min(
            (interaction.created_at for interaction in model.interactions),
            default=0,
        ),
    )


def _select_current_extractor_config(
    request_context: RequestContext,
    run: AgentRunRecord,
) -> ProfileExtractorConfig | PlaybookConfig:
    root_config = request_context.configurator.get_config()
    if run.binding.extractor_kind == "profile":
        config = getattr(root_config, "profile_extractor_config", None)
    elif run.binding.extractor_kind == "playbook":
        config = getattr(root_config, "user_playbook_extractor_config", None)
    else:
        raise ResumeWorkerError(
            f"Unsupported extractor kind {run.binding.extractor_kind!r}"
        )

    if config is not None and isinstance(
        config, ProfileExtractorConfig | PlaybookConfig
    ):
        return config
    raise ResumeWorkerError(
        f"Current extractor config not found for {run.binding.extractor_kind}"
    )


def _log_config_hash_drift(
    run: AgentRunRecord,
    extractor_config: BaseModel,
) -> None:
    current_hash = build_scope_hash(extractor_config.model_dump(mode="json"))
    if run.binding.extractor_config_hash and (
        current_hash != run.binding.extractor_config_hash
    ):
        logger.info(
            "event=resumable_extraction_config_hash_drift run_id=%s extractor_kind=%s "
            "stored_hash=%s current_hash=%s",
            run.id,
            run.binding.extractor_kind,
            run.binding.extractor_config_hash,
            current_hash,
        )


def _run_playbook_contract_selection(
    request_context: RequestContext,
    run: AgentRunRecord,
    *,
    expert: bool,
) -> tuple[bool, bool]:
    """Resolve the playbook schema selected when the durable run was created."""
    output_schema_name = run.generation_request_snapshot.get("output_schema_name")
    if output_schema_name == StructuredReferencedExtractedPlaybookList.__name__:
        return True, True
    if output_schema_name == StructuredExtractedPlaybookList.__name__:
        return True, False
    if output_schema_name == StructuredPlaybookList.__name__:
        return False, False
    # Backward compatibility for durable runs created before the schema name was
    # snapshotted. New runs never consult activation state during resume.
    strict = uses_evidence_grounded_extraction(
        request_context.prompt_manager,
        expert=expert,
    )
    return strict, strict and not expert


def _run_uses_strict_playbook_evidence(
    request_context: RequestContext,
    run: AgentRunRecord,
    *,
    expert: bool,
) -> bool:
    """Backward-compatible boolean view of the snapshotted contract."""
    return _run_playbook_contract_selection(request_context, run, expert=expert)[0]


class ExtractionResumeWorker:
    """Execute deferred extraction under admitted user-work ownership."""

    def __init__(
        self,
        *,
        request_context: RequestContext,
        llm_client: LiteLLMClient | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.request_context = request_context
        self.storage = _storage(request_context)
        self.client = llm_client or _create_llm_client(request_context)
        self.worker_id = worker_id or f"resume_worker_{uuid.uuid4().hex}"
        self._active_claim: AgentRunRecord | None = None

    def _with_resolved_playbook_user_id(self, run: AgentRunRecord) -> AgentRunRecord:
        """Return a playbook run with a proven owner for legacy nullable rows."""
        if run.binding.extractor_kind != "playbook":
            return run
        if run.binding.user_id and run.binding.user_id.strip():
            return run
        user_id = infer_playbook_review_user_id(
            storage=self.storage,
            source_interaction_ids=run.binding.source_interaction_ids,
            subject=f"Playbook extraction run {run.id}",
        )
        return replace(run, binding=replace(run.binding, user_id=user_id))

    def recover_selected(self, selection: Any, *, operation_id: str) -> AgentRunRecord:
        """Recover one previewed run without opening the scheduler's resume gate."""
        from reflexio.server.services.durable_learning.user_lease import (
            UserLeaseBusyError,
            adopt_agent_run_for_execution,
            fence_explicit_extraction,
            user_extraction_lease,
        )
        from reflexio.server.services.extraction.recovery import (
            RecoveryRefusedError,
            validate_selection,
        )

        if selection.org_id != self.request_context.org_id:
            raise RecoveryRefusedError("organization_mismatch")
        with bind_work_scope(WorkScope(selection.org_id, selection.project_id)):
            run = self.storage.get_agent_run(selection.run_id)
            if run is None:
                raise RecoveryRefusedError("run_not_found")
            validate_selection(self.request_context, run, selection)
            user_id = self._with_resolved_playbook_user_id(run).binding.user_id
            if not user_id:
                raise RecoveryRefusedError("owner_unverifiable")
            try:
                # Explicit selected work declines busy admission without changing
                # the previewed run. No logical claim ages while waiting here.
                with user_extraction_lease(
                    self.storage, user_id, max_wait_seconds=0
                ) as owner:
                    pending_config = self.request_context.configurator.get_config().pending_tool_call_config
                    with self.storage.commit_scope():
                        fence_explicit_extraction(self.storage)
                        current = self.storage.get_agent_run(run.id)
                        if current is None:
                            raise RecoveryRefusedError("run_not_found")
                        validate_selection(
                            self.request_context, current, selection, owner=owner
                        )
                        claimed = self.storage.claim_agent_run_for_recovery(
                            expected=current,
                            worker_id=self.worker_id,
                            operation_id=operation_id,
                        )
                        if claimed is None:
                            raise RecoveryRefusedError("claim_conflict")
                        claimed = adopt_agent_run_for_execution(self.storage, claimed)
                    logger.info(
                        "event=extraction_recovery_claimed run_id=%s operation_id=%s",
                        run.id,
                        operation_id,
                    )
                    self._active_claim = claimed
                    try:
                        # Already admitted and adopted: do not reacquire capacity
                        # or a second user lease through another dispatcher.
                        result = self._execute_owned_claim(
                            claimed,
                            finalization_retry=True,
                            pending_config=pending_config,
                            selection=selection,
                        )
                    except Exception as exc:
                        self._mark_selected_recovery_interrupted(run.id, exc)
                        raise
                    facts = self.storage.get_agent_run_recovery_facts(run.id)
                    if (
                        result.status != AgentRunStatus.FINALIZED
                        or not facts.get("receipt_exists")
                        or facts.get("receipt_missing_results") != 0
                    ):
                        raise RecoveryRefusedError("recovery_not_verified")
                    logger.info(
                        "event=extraction_recovery_verified run_id=%s operation_id=%s",
                        run.id,
                        operation_id,
                    )
                    return result
            except UserLeaseBusyError as exc:
                raise RecoveryRefusedError("execution_busy") from exc

    def _execute_owned_claim(
        self,
        run: AgentRunRecord,
        *,
        finalization_retry: bool,
        pending_config: Any,
        selection: Any,
    ) -> AgentRunRecord:
        """Process while the caller retains actual lease and capacity ownership."""
        from reflexio.server.services.durable_learning.user_lease import (
            adopt_agent_run_for_execution,
        )

        expected_status = (
            AgentRunStatus.FINALIZING if finalization_retry else AgentRunStatus.RESUMING
        )
        latest = self.storage.get_agent_run(run.id)
        if (
            latest is None
            or latest.status != expected_status
            or latest.claimed_by != run.claimed_by
            or latest.claimed_at != run.claimed_at
        ):
            return latest or run
        _refuse_overflow_protocol_run(latest)
        latest = adopt_agent_run_for_execution(self.storage, latest)
        self._active_claim = latest
        receipt = (
            self.storage.get_agent_run_finalization_receipt(
                run_id=latest.id,
                entity_type="profile"
                if latest.binding.extractor_kind == "profile"
                else "user_playbook",
            )
            if finalization_retry
            else None
        )
        if selection is not None and receipt is None:
            from reflexio.server.services.extraction.recovery import (
                validate_selection,
            )

            validate_selection(
                self.request_context, latest, selection, owned_claim=True
            )
        elif finalization_retry and latest.recovery_operation_id:
            # Interrupted selected work can reach the ordinary
            # retry queue; retain its stronger safety contract.
            from reflexio.server.services.extraction.recovery import (
                inspect_run,
            )

            # Receipt replay must deliver already-committed receipts,
            # even if sources/config or retry limits changed later.
            if (
                receipt is None
                and not inspect_run(self.request_context, latest, owned_claim=True)[
                    "eligible"
                ]
            ):
                return self._update_claimed_status(
                    latest.id,
                    AgentRunStatus.FAILED,
                    last_error="selected_recovery_refused",
                )
        source_ids = set(latest.binding.source_interaction_ids)
        if (
            receipt is None
            and source_ids
            and source_ids
            != {
                interaction.interaction_id
                for interaction in self.storage.get_interactions_by_ids(
                    list(source_ids)
                )
            }
        ):
            # Never revive saved output whose source was explicitly erased.
            return self._update_claimed_status(
                latest.id,
                AgentRunStatus.CANCELLED,
                last_error="source_interactions_erased",
            )
        if finalization_retry:
            return self._retry_finalization(latest) or latest
        if latest.resume_attempts >= pending_config.max_resume_attempts:
            return self._update_claimed_status(
                latest.id,
                AgentRunStatus.FAILED,
                last_error="Resumable extraction attempt allowance exhausted",
            )
        if latest.max_steps_remaining is not None and latest.max_steps_remaining <= 0:
            # Refusal consumes no model attempt; the adopted witness still
            # fences the terminal transition under the actual user owner.
            return self._update_claimed_status(
                latest.id,
                AgentRunStatus.FAILED,
                last_error="Resumable extraction max-step budget exhausted",
            )
        return self._process_claimed_run(latest, pending_config) or latest

    def _begin_model_attempt(self, run: AgentRunRecord) -> AgentRunRecord:
        """Charge only prepared model work, retaining its exact owned witness."""
        from reflexio.server.services.durable_learning.user_lease import (
            owned_agent_run_write,
        )

        if (
            self._active_claim != run
            or run.claimed_by is None
            or run.claimed_at is None
        ):
            raise LeaseLostError("Resume charge has no retained execution claim")
        with owned_agent_run_write(self.storage, run):
            attempts = self.storage.begin_extraction_resume(
                run.id, run.claimed_by, run.claimed_at.isoformat()
            )
        if attempts is None:
            raise LeaseLostError("Resume charge lost its execution claim")
        charged = replace(run, resume_attempts=attempts)
        self._active_claim = charged
        return charged

    def _mark_selected_recovery_interrupted(
        self, run_id: str, error: Exception
    ) -> None:
        """A selected refusal cannot enter ordinary retries or outlive its owner."""
        from reflexio.server.services.extraction.recovery import RecoveryRefusedError
        from reflexio.server.services.storage.storage_base import LeaseLostError

        active = self._active_claim
        if (
            active is None
            or active.id != run_id
            or active.status != AgentRunStatus.FINALIZING
        ):
            return
        try:
            refused = isinstance(error, RecoveryRefusedError)
            self._update_claimed_status(
                run_id,
                AgentRunStatus.FAILED
                if refused
                else AgentRunStatus.FINALIZATION_FAILED,
                last_error="selected_recovery_refused"
                if refused
                else "selected_recovery_interrupted",
            )
        except LeaseLostError:
            logger.info("event=selected_recovery_cleanup_lost_owner run_id=%s", run_id)

    def _update_claimed_status(
        self, run_id: str, status: AgentRunStatus, **kwargs: Any
    ) -> AgentRunRecord:
        from reflexio.server.services.durable_learning.user_lease import (
            owned_agent_run_write,
        )

        active = self._active_claim
        if (
            active is None
            or active.id != run_id
            or active.status
            not in (
                AgentRunStatus.RESUMING,
                AgentRunStatus.AGENT_COMPLETED,
                AgentRunStatus.FINALIZING,
                AgentRunStatus.FAILED,
            )
        ):
            raise LeaseLostError("Resume transition has no retained execution claim")
        with owned_agent_run_write(self.storage, active):
            updated = self.storage.update_agent_run_status(
                run_id,
                status,
                expected_statuses=(active.status,),
                expected_claimed_by=active.claimed_by,
                expected_claimed_at=active.claimed_at,
                **kwargs,
            )
            if updated is None:
                raise LeaseLostError("Resume transition lost its execution claim")
        self._active_claim = updated
        logger.info(
            "event=extraction_run_transition run_id=%s org_id=%s project_id=%s status=%s operation_id=%s",
            updated.id,
            updated.binding.org_id,
            updated.project_id,
            updated.status,
            updated.recovery_operation_id,
        )
        return updated

    def _settle_finalization_failure(
        self,
        run: AgentRunRecord,
        error: Exception,
        *,
        last_error: str,
        max_finalization_attempts: int,
    ) -> AgentRunRecord:
        from reflexio.server.services.extraction.publication import (
            settle_finalization_failure,
        )

        active = self._active_claim
        if active is None or active.id != run.id:
            raise LeaseLostError("Finalization failure has no retained execution")
        updated = settle_finalization_failure(
            self.storage,
            active,
            error,
            max_attempts=max_finalization_attempts,
            last_error=last_error,
        )
        self._active_claim = updated
        logger.info(
            "event=extraction_run_transition run_id=%s org_id=%s project_id=%s status=%s operation_id=%s",
            updated.id,
            updated.binding.org_id,
            updated.project_id,
            updated.status,
            updated.recovery_operation_id,
        )
        return updated

    def _settle_interrupted_preparation(self, run: AgentRunRecord) -> AgentRunRecord:
        from reflexio.server.services.durable_learning.user_lease import (
            owned_agent_run_write,
        )
        from reflexio.server.services.extraction.publication_preparation import (
            preparation_claim_digest,
        )
        from reflexio.server.services.extraction.publication_receipts import (
            require_output_id,
        )

        limit = self.request_context.configurator.get_config().pending_tool_call_config.max_finalization_attempts
        with owned_agent_run_write(self.storage, run), self.storage._stream_sql() as db:
            output = run.committed_output
            if not isinstance(output, dict):
                raise ResumeWorkerError("Finalization requires retained output")
            publication = output.get("publication")
            output_id = require_output_id(
                publication.get("id") if isinstance(publication, dict) else None
            )
            if (
                self.storage._get_output_publication_receipt(db, run.id, output_id)
                is not None
            ):
                return run
            entry = self.storage.pending_agent_preparation_attempt(run)
            if entry is None:
                return run
            if entry.claim_digest == preparation_claim_digest(run):
                raise LeaseLostError(
                    "This execution already has secondary compute in progress"
                )
            count = run.finalization_attempts + 1
            updated = self._update_claimed_status(
                run.id,
                AgentRunStatus.FAILED if count >= limit else AgentRunStatus.FINALIZING,
                increment_finalization_attempts=True,
                last_error="interrupted_finalization_compute",
            )
            self.storage.settle_agent_preparation_failure(updated, entry.identity)
            return updated

    def _process_claimed_run(
        self, run: AgentRunRecord, pending_config: Any
    ) -> AgentRunRecord | None:
        """Resume and finalize one claimed run. Runs under the run's project.

        Body moved verbatim out of ``run_once`` so a single ``with`` can cover
        every tenant write it makes -- the resume, the finalization, and each of
        the four ``update_agent_run_status`` exit paths.
        """
        _refuse_overflow_protocol_run(run)
        try:
            run = self._with_resolved_playbook_user_id(run)
            resolved_calls = self._load_resolved_tool_calls(run)
            if not resolved_calls:
                raise ResumeWorkerError(
                    f"Run {run.id} has no resolved, unconsumed tool calls"
                )
            completed = self._resume_run(run, resolved_calls)
            if (
                not isinstance(completed, AgentRunRecord)
                or completed.id != run.id
                or completed.status != AgentRunStatus.AGENT_COMPLETED
                or self._active_claim != completed
            ):
                raise LeaseLostError("Resume output has no committed execution witness")
        except ReadinessUnavailableError:
            from reflexio.server.services.durable_learning.user_lease import (
                owned_agent_run_write,
            )

            active = self._active_claim
            if active is None or active.id != run.id:
                raise LeaseLostError(
                    "Readiness deferral has no retained execution"
                ) from None
            if (
                active.status == AgentRunStatus.AGENT_COMPLETED
                and active.committed_output is not None
            ):
                # The returned witness proves output committed. Retry conversion,
                # without rereading a newer owner's output or refunding its call.
                return self._update_claimed_status(
                    run.id,
                    AgentRunStatus.FINALIZATION_FAILED,
                    next_resume_at=datetime.now(UTC) + timedelta(seconds=30),
                    last_error="ReadinessUnavailableError",
                )
            if (
                active.status != AgentRunStatus.RESUMING
                or active.claimed_by is None
                or active.claimed_at is None
            ):
                raise LeaseLostError(
                    "Readiness deferral lost its execution claim"
                ) from None
            if active.resume_attempts == run.resume_attempts:
                # Preparation never charged this execution. Do not refund a
                # previous model call when readiness fails before entry.
                return self._update_claimed_status(
                    run.id,
                    AgentRunStatus.RESUME_READY,
                    next_resume_at=datetime.now(UTC) + timedelta(seconds=30),
                    last_error="ReadinessUnavailableError",
                )
            with owned_agent_run_write(self.storage, active):
                deferred = self.storage.defer_extraction_resume(
                    run.id,
                    active.claimed_by,
                    active.claimed_at.isoformat(),
                    active.resume_attempts,
                    (datetime.now(UTC) + timedelta(seconds=30)).isoformat(),
                )
                # This read returns our own CAS result under the held writer
                # locks; it cannot confer a replacement execution's authority.
                updated = self.storage.get_agent_run(run.id) if deferred else None
                if deferred and (
                    updated is None or updated.status != AgentRunStatus.RESUME_READY
                ):
                    raise LeaseLostError(
                        "Readiness deferral result is unavailable"
                    ) from None
            if updated is not None:
                self._active_claim = updated
            return updated
        except LeaseLostError:
            raise
        except Exception as exc:
            active = self._active_claim
            if active is None or active.id != run.id:
                raise LeaseLostError(
                    "Resume failure has no retained execution claim"
                ) from exc
            if execution_input_is_busy(exc):
                # Keep any admitted attempt charged: absence of a returned
                # response does not establish that no provider work entered.
                # Returned model work uses the interrupted-turn failure path.
                # Input-only refusal adds no failure or backoff; another model
                # attempt must still pass the stored allowance before dispatch.
                return self._update_claimed_status(
                    run.id,
                    AgentRunStatus.FINALIZATION_FAILED
                    if active.status == AgentRunStatus.AGENT_COMPLETED
                    and active.committed_output is not None
                    else AgentRunStatus.RESUME_READY,
                    next_resume_at=datetime.now(UTC),
                    last_error="ExecutionInputBusyError",
                )
            with error_tags(
                subsystem="extraction",
                op="resume_run",
                org_id=self.request_context.org_id,
                run_id=run.id,
                error_type=type(exc).__name__,
            ):
                logger.exception(
                    "event=resumable_extraction_resume_failed run_id=%s",
                    run.id,
                )
            failed_status = (
                AgentRunStatus.FAILED
                if active.resume_attempts >= pending_config.max_resume_attempts
                else AgentRunStatus.RESUME_READY
            )
            return self._update_claimed_status(
                run.id,
                failed_status,
                next_resume_at=_next_retry_at(active.resume_attempts),
                last_error=str(exc),
            )

        try:
            run = self._update_claimed_status(completed.id, AgentRunStatus.FINALIZING)
            run = self._with_resolved_playbook_user_id(run)
            return self._publish_retained_output(run)
        except ReadinessUnavailableError as error:
            return self._settle_finalization_failure(
                run,
                error,
                last_error="ReadinessUnavailableError",
                max_finalization_attempts=pending_config.max_finalization_attempts,
            )
        except LeaseLostError:
            raise
        except Exception as exc:
            with error_tags(
                subsystem="extraction",
                op="finalize_run",
                org_id=self.request_context.org_id,
                run_id=run.id,
                error_type=type(exc).__name__,
            ):
                logger.exception(
                    "event=resumable_extraction_finalization_failed run_id=%s",
                    run.id,
                )
            return self._settle_finalization_failure(
                run,
                exc,
                last_error=str(exc),
                max_finalization_attempts=pending_config.max_finalization_attempts,
            )

    def _retry_finalization(self, run: AgentRunRecord) -> AgentRunRecord | None:
        _refuse_overflow_protocol_run(run)
        pending_config = (
            self.request_context.configurator.get_config().pending_tool_call_config
        )
        try:
            run = self._with_resolved_playbook_user_id(run)
            return self._publish_retained_output(run)
        except ReadinessUnavailableError as error:
            return self._settle_finalization_failure(
                run,
                error,
                last_error="ReadinessUnavailableError",
                max_finalization_attempts=pending_config.max_finalization_attempts,
            )
        except LeaseLostError:
            raise
        except Exception as exc:
            with error_tags(
                subsystem="extraction",
                op="finalize_run_retry",
                org_id=self.request_context.org_id,
                run_id=run.id,
                error_type=type(exc).__name__,
            ):
                logger.error(
                    "event=resumable_extraction_finalization_retry_failed run_id=%s error_type=%s",
                    run.id,
                    type(exc).__name__,
                )
            return self._settle_finalization_failure(
                run,
                exc,
                last_error=type(exc).__name__,
                max_finalization_attempts=pending_config.max_finalization_attempts,
            )

    def _load_resolved_tool_calls(
        self,
        run: AgentRunRecord,
    ) -> list[PendingToolCallRecord]:
        records: list[PendingToolCallRecord] = []
        for dependency in self.storage.list_run_tool_dependencies(run.id):
            if dependency.consumed_at is not None or dependency.resolved_at is None:
                continue
            record = self.storage.get_pending_tool_call(dependency.pending_tool_call_id)
            if record and record.status == PendingToolCallStatus.RESOLVED:
                records.append(record)
        return sorted(
            records,
            key=lambda record: record.resolved_at or datetime.max.replace(tzinfo=UTC),
        )

    def _extra_tools_for_run(
        self,
        run: AgentRunRecord,
    ) -> tuple[list[Any], PendingToolCallToolContext | None]:
        if not pending_tool_calls_enabled(self.request_context):
            return [], None
        pending_config = (
            self.request_context.configurator.get_config().pending_tool_call_config
        )
        tools = create_pending_info_tools_for_extractor_kind(run.binding.extractor_kind)
        return tools, PendingToolCallToolContext(
            storage=self.storage,
            run_id=run.id,
            org_id=self.request_context.org_id,
            extractor_kind=run.binding.extractor_kind,
            user_id=run.binding.user_id,
            config=pending_config,
        )

    def _resume_run(
        self,
        run: AgentRunRecord,
        resolved_calls: list[PendingToolCallRecord],
    ) -> AgentRunRecord:
        _refuse_overflow_protocol_run(run)
        request_interaction_data_models = _request_interaction_models_from_ids(
            self.storage,
            run.binding.source_interaction_ids,
            fallback_source=run.binding.source,
            fallback_agent_version=run.binding.agent_version,
        )
        extractor_config = _select_current_extractor_config(self.request_context, run)
        _log_config_hash_drift(run, extractor_config)
        with answer_input_scope(self.storage, run.id):
            if run.binding.extractor_kind == "profile":
                return self._resume_profile(
                    run,
                    extractor_config,
                    request_interaction_data_models,
                    resolved_calls,
                )
            if run.binding.extractor_kind == "playbook":
                return self._resume_playbook(
                    run,
                    extractor_config,
                    request_interaction_data_models,
                    resolved_calls,
                )
            raise ResumeWorkerError(
                f"Unsupported extractor kind {run.binding.extractor_kind!r}"
            )

    def _resume_profile(
        self,
        run: AgentRunRecord,
        extractor_config: ProfileExtractorConfig | PlaybookConfig,
        request_interaction_data_models: list[RequestInteractionDataModel],
        resolved_calls: list[PendingToolCallRecord],
    ) -> AgentRunRecord:
        if not isinstance(extractor_config, ProfileExtractorConfig):
            raise ResumeWorkerError("Expected profile extractor config")
        if run.binding.user_id is None:
            raise ResumeWorkerError("Profile resume requires user_id")

        existing_profiles = self.storage.get_user_profile(run.binding.user_id)
        context_profiles = sorted(
            existing_profiles,
            key=lambda profile: profile.last_modified_timestamp,
            reverse=True,
        )[:MAX_EXISTING_PROFILES_FOR_CONTEXT]
        agent_context = self.request_context.configurator.get_agent_context()
        prepared_prompt = prepare_profile_extraction_prompt(
            prompt_manager=self.request_context.prompt_manager,
            request_interaction_data_models=request_interaction_data_models,
            agent_context_prompt=agent_context,
            context_prompt=(
                extractor_config.context_prompt.strip()
                if extractor_config.context_prompt
                else ""
            ),
            extraction_definition_prompt=extractor_config.extraction_definition_prompt.strip(),
            existing_profiles=context_profiles,
            user_id=run.binding.user_id,
        )
        return self._resume_agent(
            run=run,
            messages=self._messages_with_prior_knowledge(
                run=run,
                messages=prepared_prompt.messages,
                extractor_config=extractor_config,
                resolved_calls=resolved_calls,
            ),
            output_schema=StructuredProfilesOutput,
            history_inputs=prepared_prompt.history_inputs,
            source_inputs=capture_source_inputs(
                self.storage, run, request_interaction_data_models
            ),
            resolved_calls=resolved_calls,
            log_label="Profile extraction resume",
        )

    def _resume_playbook(
        self,
        run: AgentRunRecord,
        extractor_config: ProfileExtractorConfig | PlaybookConfig,
        request_interaction_data_models: list[RequestInteractionDataModel],
        resolved_calls: list[PendingToolCallRecord],
    ) -> AgentRunRecord:
        if not isinstance(extractor_config, PlaybookConfig):
            raise ResumeWorkerError("Expected playbook extractor config")
        user_id = run.binding.user_id
        if not user_id:
            raise ResumeWorkerError("Playbook resume requires user_id")

        agent_context = self.request_context.configurator.get_agent_context()
        playbook_definition = (
            extractor_config.extraction_definition_prompt.strip()
            if extractor_config.extraction_definition_prompt
            else ""
        )
        all_interactions = extract_interactions_from_request_interaction_data_models(
            request_interaction_data_models
        )
        prompt_manager = self.request_context.prompt_manager
        expert_mode = has_expert_content(all_interactions)
        strict_evidence, evidence_references = _run_playbook_contract_selection(
            self.request_context,
            run,
            expert=expert_mode,
        )
        prompt_context = build_playbook_prompt_context(
            request_interaction_data_models,
            expert=expert_mode,
            label_turns=strict_evidence,
        )
        contract = build_playbook_extraction_contract(
            prompt_manager,
            expert=expert_mode,
            evidence_sources=prompt_context.evidence_sources,
            evidence_units=prompt_context.evidence_units,
            strict_override=strict_evidence,
            evidence_references_override=evidence_references,
        )
        if expert_mode:
            messages = construct_expert_playbook_extraction_messages(
                prompt_manager=prompt_manager,
                request_interaction_data_models=request_interaction_data_models,
                agent_context_prompt=agent_context,
                extraction_definition_prompt=playbook_definition,
                prompt_context=prompt_context,
                citation_instruction=contract.citation_instruction(),
            )
        else:
            root_config = self.request_context.configurator.get_config()
            tool_can_use = ""
            if root_config and root_config.tool_can_use:
                tool_can_use = "\n".join(
                    [
                        f"{tool.tool_name}: {tool.tool_description}"
                        for tool in root_config.tool_can_use
                    ]
                )
            messages = construct_playbook_extraction_messages_from_sessions(
                prompt_manager=prompt_manager,
                request_interaction_data_models=request_interaction_data_models,
                agent_context_prompt=agent_context,
                extraction_definition_prompt=playbook_definition,
                tool_can_use=tool_can_use,
                prompt_context=prompt_context,
                citation_instruction=contract.citation_instruction(),
            )
        return self._resume_agent(
            run=run,
            messages=self._messages_with_prior_knowledge(
                run=run,
                messages=messages,
                extractor_config=extractor_config,
                resolved_calls=resolved_calls,
            ),
            output_schema=contract.schema,
            history_inputs=HistoricalInputs(),
            source_inputs=capture_source_inputs(
                self.storage, run, request_interaction_data_models
            ),
            resolved_calls=resolved_calls,
            log_label="Playbook extraction resume",
            # Same bounded corrective turn a fresh run gets, so a resumed run
            # cannot yield a batch a fresh run would have rejected.
            structured_output_validator=contract.validator,
        )

    def _messages_with_prior_knowledge(
        self,
        *,
        run: AgentRunRecord,
        messages: list[dict[str, Any]],
        extractor_config: BaseModel,
        resolved_calls: list[PendingToolCallRecord] | None = None,
    ) -> list[dict[str, Any]]:
        pending_config = (
            self.request_context.configurator.get_config().pending_tool_call_config
        )
        # Prior-knowledge context is always injected within the resumable path.
        return append_prior_knowledge_context(
            messages=messages,
            storage=self.storage,
            org_id=self.request_context.org_id,
            extractor_kind=run.binding.extractor_kind,
            extractor_config=extractor_config,
            source=run.binding.source,
            agent_version=run.binding.agent_version,
            similarity_threshold=pending_config.for_tool(
                "ask_human"
            ).similarity_threshold,
            # Avoid re-surfacing the run's own just-resolved answers, which are
            # injected separately as explicit resolved-tool-result messages.
            exclude_pending_tool_call_ids={call.id for call in resolved_calls or []},
        )

    def _resume_agent(
        self,
        *,
        run: AgentRunRecord,
        messages: list[dict[str, Any]],
        output_schema: type[BaseModel],
        source_inputs: dict[str, Any],
        history_inputs: HistoricalInputs,
        resolved_calls: list[PendingToolCallRecord],
        log_label: str,
        structured_output_validator: (
            Callable[[BaseModel], Sequence[str]] | None
        ) = None,
    ) -> AgentRunRecord:
        extra_tools, extra_context = self._extra_tools_for_run(run)
        result = ResumableExtractionAgent(
            client=self.client,
            storage=self.storage,
        ).resume(
            run=run,
            messages=messages,
            output_schema=output_schema,
            resolved_tool_calls=resolved_calls,
            source_inputs=source_inputs,
            history_inputs=history_inputs,
            before_model=self._begin_model_attempt,
            extra_tools=extra_tools,
            extra_tool_context=extra_context,
            log_label=log_label,
            structured_output_validator=structured_output_validator,
        )

        witness = result.execution_run
        if (
            witness is None
            or result.run_id != run.id
            or witness.id != run.id
            or witness.claimed_by != run.claimed_by
            or witness.claimed_at != run.claimed_at
            or witness.status
            not in (AgentRunStatus.AGENT_COMPLETED, AgentRunStatus.FAILED)
        ):
            raise LeaseLostError("Resume result did not retain its execution claim")
        from reflexio.server.services.durable_learning.user_lease import (
            owned_agent_run_write,
        )

        with owned_agent_run_write(self.storage, witness):
            pass
        self._active_claim = witness
        if witness.status != AgentRunStatus.AGENT_COMPLETED:
            raise ResumeWorkerError(
                f"{run.binding.extractor_kind.capitalize()} resume did not finish: {result.failure_label}"
            )
        return witness

    def _items_from_committed_output(
        self,
        run: AgentRunRecord,
    ) -> tuple[list[Any], list[str], ModelProvenance | None]:
        if run.committed_output is None:
            raise ResumeWorkerError(
                f"Run {run.id} cannot retry finalization without committed output"
            )
        request_interaction_data_models = _request_interaction_models_from_ids(
            self.storage,
            run.binding.source_interaction_ids,
            fallback_source=run.binding.source,
            fallback_agent_version=run.binding.agent_version,
        )
        extractor_config = _select_current_extractor_config(self.request_context, run)
        output, model_provenance = decode_committed_output(run.committed_output)
        if run.binding.extractor_kind == "profile":
            items, pending_ids = self._profile_items_from_output(
                run,
                extractor_config,
                output,
            )
            return items, pending_ids, model_provenance
        if run.binding.extractor_kind == "playbook":
            items, pending_ids = self._playbook_items_from_output(
                run,
                extractor_config,
                request_interaction_data_models,
                output,
            )
            return items, pending_ids, model_provenance
        raise ResumeWorkerError(
            f"Unsupported extractor kind {run.binding.extractor_kind!r}"
        )

    def _profile_items_from_output(
        self,
        run: AgentRunRecord,
        extractor_config: ProfileExtractorConfig | PlaybookConfig,
        committed_output: dict[str, Any],
    ) -> tuple[list[Any], list[str]]:
        if not isinstance(extractor_config, ProfileExtractorConfig):
            raise ResumeWorkerError("Expected profile extractor config")
        if run.binding.user_id is None:
            raise ResumeWorkerError("Profile finalization retry requires user_id")
        output = StructuredProfilesOutput.model_validate(committed_output)
        raw_profiles = [profile.model_dump() for profile in output.profiles or []]
        return (
            ProfileExtractor._convert_raw_to_user_profiles(
                raw_profiles=raw_profiles,
                user_id=run.binding.user_id,
                request_id=run.binding.request_id,
                source_interaction_ids=list(run.binding.source_interaction_ids),
            ),
            run.pending_tool_call_ids,
        )

    def _playbook_items_from_output(
        self,
        run: AgentRunRecord,
        extractor_config: ProfileExtractorConfig | PlaybookConfig,
        request_interaction_data_models: list[RequestInteractionDataModel],
        committed_output: dict[str, Any],
    ) -> tuple[list[Any], list[str]]:
        if not isinstance(extractor_config, PlaybookConfig):
            raise ResumeWorkerError("Expected playbook extractor config")
        user_id = run.binding.user_id
        if not user_id:
            raise ResumeWorkerError("Playbook finalization retry requires user_id")
        expert_mode = has_expert_content(
            extract_interactions_from_request_interaction_data_models(
                request_interaction_data_models
            )
        )
        strict_evidence, evidence_references = _run_playbook_contract_selection(
            self.request_context,
            run,
            expert=expert_mode,
        )
        prompt_context = build_playbook_prompt_context(
            request_interaction_data_models,
            expert=expert_mode,
            label_turns=strict_evidence,
        )
        contract = build_playbook_extraction_contract(
            self.request_context.prompt_manager,
            expert=expert_mode,
            evidence_sources=prompt_context.evidence_sources,
            evidence_units=prompt_context.evidence_units,
            strict_override=strict_evidence,
            evidence_references_override=evidence_references,
        )
        output = contract.schema.model_validate(committed_output)
        if not isinstance(
            output,
            StructuredPlaybookList
            | StructuredReferencedExtractedPlaybookList
            | StructuredExtractedPlaybookList,
        ):
            raise ResumeWorkerError(
                "Committed playbook output validated to an unexpected schema: "
                f"{type(output).__name__}"
            )
        agent_context = self.request_context.configurator.get_agent_context()
        service_config = PlaybookGenerationServiceConfig(
            request_id=run.binding.request_id,
            agent_version=run.binding.agent_version or "",
            user_id=user_id,
            source=run.binding.source,
            auto_run=False,
            force_extraction=True,
        )
        extractor = PlaybookExtractor(
            request_context=self.request_context,
            llm_client=self.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=agent_context,
        )
        if contract.strict:
            items = extractor._process_structured_response_list(
                output,
                evidence_sources=prompt_context.evidence_sources,
                evidence_units=prompt_context.evidence_units,
            )
        else:
            items = extractor._process_structured_response_list(
                output,
                source_interaction_ids=list(run.binding.source_interaction_ids),
            )
        return items, run.pending_tool_call_ids

    def _publish_retained_output(self, run: AgentRunRecord) -> AgentRunRecord:
        _refuse_overflow_protocol_run(run)
        run = self._settle_interrupted_preparation(run)
        if run.status == AgentRunStatus.FAILED:
            return run
        from reflexio.server.services.extraction.publication_preparation import (
            SavedPublicationPreparation,
            bookmark_from_output,
        )

        def prepare(
            run: AgentRunRecord, saved: SavedPublicationPreparation | None
        ) -> PreparedPublication:
            user_id = run.binding.user_id
            if not user_id:
                raise ResumeWorkerError("Output publication requires its admitted user")
            if run.binding.extractor_kind == "profile":
                service = ProfileGenerationService(
                    llm_client=self.client, request_context=self.request_context
                )
                service.service_config = ProfileGenerationServiceConfig(
                    user_id=user_id,
                    request_id=run.binding.request_id,
                    source=run.binding.source,
                    auto_run=False,
                    force_extraction=True,
                )
            elif run.binding.extractor_kind == "playbook":
                service = PlaybookGenerationService(
                    llm_client=self.client, request_context=self.request_context
                )
                service.service_config = PlaybookGenerationServiceConfig(
                    request_id=run.binding.request_id,
                    agent_version=run.binding.agent_version or "",
                    user_id=user_id,
                    source=run.binding.source,
                    auto_run=False,
                    force_extraction=True,
                )
                service._review_run = run
            else:
                raise ResumeWorkerError("Unsupported publication extractor")
            if saved is not None:
                return PreparedPublication(service, saved.write_plan, saved.bookmark)
            items, _, provenance = self._items_from_committed_output(run)
            if provenance is not None:
                service._last_model_provenance = provenance
            from reflexio.server.services.extraction.publication_receipts import (
                output_publication_origin,
            )

            bookmark = (
                bookmark_from_output(run)
                if items
                and output_publication_origin(run.committed_output) == "initial_inline"
                else None
            )
            return PreparedPublication(
                service, service._resolve_write_plan([items]), bookmark
            )

        updated = publish_resumed_output(
            self.storage,
            run,
            prepare=prepare,
            on_published=self._schedule_finalized_tagging,
        )
        self._active_claim = updated
        return updated

    def _schedule_finalized_tagging(self, run: AgentRunRecord) -> None:
        user_id = run.binding.user_id
        if not user_id:
            return
        if run.binding.extractor_kind not in ("profile", "playbook"):
            return

        # Defer tagging off this worker's drain loop. The pass is idempotent
        # (skips already-tagged entities), so running both profile and playbook
        # tagging is safe regardless of this run's extractor kind.
        try:
            schedule_tagging(
                org_id=self.request_context.org_id,
                user_id=user_id,
                agent_version=run.binding.agent_version or "",
                request_context=self.request_context,
                llm_client=self.client,
            )
        except Exception:
            logger.exception(
                "Failed to schedule tagging for finalized %s run %s",
                run.binding.extractor_kind,
                run.id,
            )
