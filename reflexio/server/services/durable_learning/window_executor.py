"""One explicit-window compute, short fenced commit, then durable effects."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.config_schema import (
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.env_utils import env_bool, env_str
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    ProviderInputOverflowError,
    declared_context_limits,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.operational_metrics import record_health
from reflexio.server.services.durable_learning.window_codec import (
    decode_plan,
    encode_plan,
)
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
    source_interaction_digest,
)
from reflexio.server.services.extraction.input_preparation import (
    prepare_extraction_input,
    repartition_extraction_input,
)
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowCoordinator,
    OverflowSuspendedError,
    legacy_observed_usage,
)
from reflexio.server.services.extraction.protected_pass import build_protected_record
from reflexio.server.services.extraction.resumable_agent import decode_committed_output
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
)
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationRequest,
)
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    AgentRunStatus,
    BaseStorage,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    Window,
    WindowInputsDeletedError,
)

logger = logging.getLogger(__name__)


def load_window_inputs(
    storage: BaseStorage, window: Window
) -> list[RequestInteractionDataModel]:
    if window.invalidated:
        raise WindowInputsDeletedError("Window inputs were erased")
    ids = [m["interaction_id"] for m in window.manifest]
    interactions = {i.interaction_id: i for i in storage.get_interactions_by_ids(ids)}
    groups: list[RequestInteractionDataModel] = []
    for item in window.manifest:
        interaction = interactions.get(item["interaction_id"])
        if interaction is None or interaction.user_id != window.user_id:
            raise WindowInputsDeletedError("Extraction manifest was erased")
        if not groups or groups[-1].request.request_id != interaction.request_id:
            request = storage.get_request(interaction.request_id)
            if request is None or request.user_id != window.user_id:
                raise WindowInputsDeletedError("Extraction request was erased")
            groups.append(
                RequestInteractionDataModel(
                    session_id=request.session_id,
                    request=request,
                    interactions=[],
                    arrival_order=True,
                )
            )
        groups[-1].interactions.append(interaction)
    return groups


class WindowExecutor:
    def __init__(self, context: RequestContext, client: LiteLLMClient):
        if context.storage is None:
            raise ValueError("Extraction requires storage")
        self.context, self.client, self.storage = context, client, context.storage

    def service(self, window: Window) -> tuple[Any, Any, Any]:
        groups = load_window_inputs(self.storage, window)
        attribution = groups[-1].request
        common = {
            "user_id": window.user_id,
            "request_id": attribution.request_id,
            "source": attribution.source,
            "force_extraction": window.force,
        }
        if window.kind == "profile":
            service = ProfileGenerationService(self.client, self.context)
            request = ProfileGenerationRequest(**common)
            config = ProfileExtractorConfig.model_validate(window.policy["extractor"])
        else:
            service = PlaybookGenerationService(
                self.client, self.context, skip_aggregation=window.skip_aggregation
            )
            request = PlaybookGenerationRequest(
                **common, agent_version=attribution.agent_version
            )
            config = UserPlaybookExtractorConfig.model_validate(
                window.policy["extractor"]
            )
        return self._configure(service, request, config, groups, window)

    @staticmethod
    def _configure(
        service: Any,
        request: Any,
        config: Any,
        groups: list[RequestInteractionDataModel],
        window: Window,
    ) -> tuple[Any, Any, Any]:
        service.set_extraction_window(groups, config, window.window_id)
        service._window_skip_should_run = window.policy.get(
            "skip_should_run_check", False
        )
        service.service_config = service._load_generation_service_config(request)
        service.service_config.window_interactions = groups
        service.service_config.extraction_window_id = window.window_id
        service._last_precheck_sessions = groups
        return service, request, config

    @staticmethod
    def _bypass_overflow_admission(
        window: Window,
        known: AgentRunRecord | None,
        manifest: dict[str, Any] | None,
        groups: list[RequestInteractionDataModel],
    ) -> bool:
        if env_bool("MOCK_LLM_RESPONSE", default=False):
            if known is not None:
                # A mode toggle must not replace durable real work with fake
                # results, fabricate source coverage, or refund its quota.
                raise InputBudgetConfigurationError(
                    "Mock mode cannot replace an existing extraction agent run"
                )
            # Simulations have no provider input to budget. Keep their established
            # mock extraction/persistence behavior without admitting a protocol.
            return True
        if any(
            item.interacted_image_url or item.image_encoding
            for group in groups
            for item in group.interactions
        ):
            if manifest is not None:
                raise InputBudgetConfigurationError(
                    "Admitted textual extraction cannot consume image evidence"
                )
            # Preserve the established multimodal prompt and evidence contract.
            # Text-only span witnesses cannot prove facts learned from images.
            return True
        # Enabling admission cannot strand a validated legacy write plan whose
        # computation finished before its persistence retry.
        return manifest is None and window.outcome is not None

    def _coordinator(
        self,
        window: Window,
        token: str,
        service: Any,
        request: Any,
        config: Any,
    ) -> OverflowCoordinator | None:
        known = self.storage.get_agent_run(f"window:{window.window_id}")
        manifest = known.generation_request_snapshot.get("overflow") if known else None
        groups = service.service_config.window_interactions
        if self._bypass_overflow_admission(window, known, manifest, groups):
            return None
        if manifest is None and not env_bool(
            "REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", default=False
        ):
            return None
        # Legacy unfinished agents keep their established continuation protocol.
        if known is not None and manifest is None and known.committed_output is None:
            if known.status == AgentRunStatus.FAILED:
                raise InputBudgetConfigurationError(
                    "Failed legacy extraction needs explicit recovery; historical quota is unknown"
                )
            return None
        # Freeze this geometry before any provider work. Existing checkpoints
        # retain their original partition and cannot reclaim consumed slots.
        chunk_tokens = manifest.get("input_chunk_tokens", 8192) if manifest else 10000
        if type(chunk_tokens) is not int or chunk_tokens not in {8192, 10000}:
            raise InputBudgetConfigurationError("Extraction partition changed")
        preparation_version = manifest.get("preparation_version", 1) if manifest else 2
        if preparation_version not in {1, 2}:
            raise InputBudgetConfigurationError(
                "Unknown extraction preparation version"
            )
        prepared = prepare_extraction_input(
            groups,
            user_id=window.user_id,
            chunk_tokens=chunk_tokens,
            require_overlap=preparation_version == 2,
        )
        if not prepared.chunks:
            if manifest is not None:
                raise InputBudgetConfigurationError(
                    "Prepared extraction lost textual evidence"
                )
            # Image-only requests use the established multimodal extractor.
            # Empty text coverage must never become an all-false chunk vote.
            return None
        if (
            manifest is not None
            and manifest.get("source_digest") != prepared.source_digest
        ):
            raise WindowInputsDeletedError(
                "Extraction source changed after preparation"
            )
        if manifest is not None and manifest.get("chunk_digests") != [
            OverflowCoordinator.candidate_key(chunk.text) for chunk in prepared.chunks
        ]:
            raise InputBudgetConfigurationError("Admitted extraction partition changed")
        replan = manifest.get("checkpoint", {}).get("repartition") if manifest else None
        if replan is not None:
            prepared = repartition_extraction_input(
                prepared,
                chunk_index=replan["chunk_index"],
                split_after=replan["split_after"],
            )
            if replan.get("chunk_digests") != [
                OverflowCoordinator.candidate_key(chunk.text)
                for chunk in prepared.chunks
            ]:
                raise InputBudgetConfigurationError(
                    "Frozen extraction repartition changed"
                )
        expected_run = build_extractor_agent_run_record(
            org_id=self.context.org_id,
            extractor_kind=window.kind,
            user_id=window.user_id,
            generation_request_id=request.request_id,
            agent_version=getattr(request, "agent_version", None),
            source=request.source,
            request_interaction_data_models=groups,
            extractor_config=config,
            service_config=service.service_config,
            agent_context=(known.agent_context_snapshot or "")
            if known is not None
            else self.context.configurator.get_agent_context(),
        )
        if known is not None and known.binding != expected_run.binding:
            raise InputBudgetConfigurationError(
                "Extraction coordinator parent binding changed"
            )
        if manifest is None:
            manifest = OverflowCoordinator.initial_manifest(prepared.source_digest)
            manifest["input_chunk_tokens"] = chunk_tokens
            manifest["preparation_version"] = preparation_version
            if known is not None:
                observed = legacy_observed_usage(known.committed_output)
                if observed is not None:
                    manifest["legacy_observed_usage"] = observed
            manifest["chunk_digests"] = [
                OverflowCoordinator.candidate_key(chunk.text)
                for chunk in prepared.chunks
            ]
            with self.storage.commit_scope():
                self.storage.fence_user_extraction(window.user_id, token)
                self.storage.validate_extraction_source_digest(
                    window, prepared.source_digest
                )
                if known is None:
                    run = replace(expected_run, id=f"window:{window.window_id}")
                    run.generation_request_snapshot["overflow"] = manifest
                    # Source rows remain authoritative; do not repeat giant input
                    # blobs inside a parent snapshot for every provider attempt.
                    if run.service_config_snapshot is not None:
                        run.service_config_snapshot.pop("window_interactions", None)
                    known = self.storage.create_agent_run(run)
                else:
                    digests = {
                        str(item.interaction_id): source_interaction_digest(item)
                        for group in groups
                        for item in group.interactions
                    }
                    manifest = self.storage.adopt_extraction_overflow(
                        window,
                        token,
                        source_digest=prepared.source_digest,
                        source_digests=digests,
                        manifest=manifest,
                    )
        if known is None:
            raise RuntimeError("Extraction coordinator parent is unavailable")
        turn_seconds = max(240, self.client.config.timeout + 60)
        # The coordinator guards every dispatch against this absolute turn
        # deadline. Its enclosing extractor future must leave time for the last
        # admitted call to save usage/output and yield without losing ownership.
        service._window_extractor_timeout_seconds = turn_seconds + 10
        coordinator = OverflowCoordinator(
            self.storage,
            window,
            token,
            manifest,
            source_digest=prepared.source_digest,
            context_limits=declared_context_limits(
                env_str("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", default="{}")
            ),
            prepared=prepared,
            agent_context=known.agent_context_snapshot or "",
            # A turn must admit at least one configured provider call. The
            # worker renews its user lease while it waits; checkpoints still
            # yield before another call can outlive the remaining turn.
            turn_seconds=turn_seconds,
        )
        coordinator.saved_run = known
        return coordinator

    def _prepare_overflow_context(
        self, coordinator: OverflowCoordinator, config: Any
    ) -> None:
        prepared = coordinator.require_prepared()
        checkpoint = coordinator.checkpoint
        coverage = [coordinator.candidate_key(chunk.text) for chunk in prepared.chunks]
        if checkpoint.get("protected_coverage") == coverage:
            return
        if (
            checkpoint.get("extraction_policy")
            and checkpoint.get("source_coverage") == coverage
            and checkpoint.get("stage") in {"extracted", "recovered"}
        ):
            # New-protocol output already scanned every source chunk. Its
            # references can be verified without another provider pass.
            return
        saved_run = coordinator.saved_run
        claims: dict[str, Any] = {}
        if saved_run is not None and saved_run.committed_output is not None:
            payload, _ = decode_committed_output(saved_run.committed_output)
            items = (
                payload.get(
                    "profiles" if coordinator.window.kind == "profile" else "playbooks"
                )
                or []
            )
            claims = {
                f"C{index}": {
                    key: item.get(key)
                    for key in ("content", "trigger", "rationale")
                    if key in item
                }
                for index, item in enumerate(items, 1)
            }
        if (
            len(prepared.chunks) > 1
            or claims
            or (saved_run is not None and saved_run.committed_output is not None)
        ):
            coordinator.record = build_protected_record(
                coordinator=coordinator,
                prepared=prepared,
                client=self.client,
                extraction_criteria=config.extraction_definition_prompt or "",
                agent_context=coordinator.agent_context,
                candidates=claims,
                minimum_following_attempts=(
                    len(claims) * 2 + 1 if claims else len(prepared.chunks) + 3
                ),
            )

    def execute(self, window: Window, token: str) -> None:
        service, request, config = self.service(window)
        coordinator = self._coordinator(window, token, service, request, config)
        if window.outcome is None:
            known_run = self.storage.get_agent_run(f"window:{window.window_id}")
            if known_run is not None and known_run.committed_output is not None:
                # The gate already admitted this durable result. A new vote on
                # retry must not discard successful output before persistence.
                service._window_skip_should_run = True
            if coordinator is None:
                plan = service.compute_generation(request)
            else:
                plan = self._compute_overflow_window(
                    coordinator, service, request, config
                )
                coordinator.require_full_source_coverage()
                service._last_token_totals = coordinator.token_totals
                if plan is not None:
                    plan.token_totals = coordinator.token_totals
                    service._last_token_totals = plan.token_totals
            outcome = encode_plan(plan, service)
            self.storage.save_extraction_outcome(window, token, outcome)
        else:
            plan = decode_plan(
                window.outcome, window.kind, config, window.user_id, service
            )
        with self.storage.commit_scope():
            # Explicit deletion wins over a cached compute as well as live compute.
            self.storage.fence_user_extraction(window.user_id, token)
            self.storage.validate_extraction_inputs(window)
            if coordinator is not None:
                coordinator.require_full_source_coverage()
                self.storage.validate_extraction_source_digest(
                    window, coordinator.source_digest
                )
                self.storage.validate_extraction_answers(
                    coordinator.checkpoint.get("answer_witnesses", {})
                )
                self.storage.validate_extraction_answer_dependencies(
                    window, coordinator.checkpoint.get("answer_dependencies", {})
                )
                self.storage.validate_extraction_historical_playbooks(
                    window, coordinator.checkpoint.get("historical_playbooks", {})
                )
                if plan is not None:
                    plan.token_totals = coordinator.token_totals
                    service._last_token_totals = plan.token_totals
            if plan is not None:
                service.persist_generation(plan)
            elif (
                coordinator is not None
                and coordinator.checkpoint.get("stage") == "gated_false"
            ):
                # The precheck created a parent before it voted. An all-false
                # completed scan must not leave an orphan running agent behind.
                finalized = self.storage.update_agent_run_status(
                    f"window:{window.window_id}",
                    AgentRunStatus.FINALIZED,
                    expected_statuses=(AgentRunStatus.RUNNING, AgentRunStatus.FAILED),
                )
                if finalized is None:
                    raise RuntimeError("Extraction gate parent lost ownership")
            effects = encode_plan(plan, service)
            effects["billing"] = self._billing_snapshot(window, service, plan)
            self.storage.complete_extraction(window, token, effects)
        record_health("worker.commits")

    def _compute_overflow_window(
        self, coordinator: OverflowCoordinator, service: Any, request: Any, config: Any
    ) -> Any:
        try:
            self._prepare_overflow_context(coordinator, config)
            with coordinator.provider_scope():
                return service.compute_generation(request)
        except Exception as exc:
            current: BaseException | None = exc
            seen: set[int] = set()
            while current is not None and id(current) not in seen:
                if isinstance(current, ProviderInputOverflowError):
                    coordinator.repartition_rejected_input(current)
                    raise OverflowSuspendedError(
                        "Extraction has a frozen smaller source plan"
                    ) from None
                seen.add(id(current))
                current = current.__cause__ or current.__context__
            raise

    def deliver(
        self,
        window: Window,
        effects: dict[str, Any],
        *,
        service: Any = None,
        plan: Any = None,
        token: str,
    ) -> None:
        billing = effects.get("billing")
        if billing is not None:
            self._bill(window, billing)
        if service is None:
            try:
                service, _, config = self.service(window)
            except WindowInputsDeletedError:
                # Erasure cancels derived work, not the already committed receipt.
                self.storage.ack_extraction_effects(
                    window.window_id, user_id=window.user_id, token=token
                )
                return
            plan = decode_plan(effects, window.kind, config, window.user_id, service)
        if plan is not None:
            service._finalize_extraction_runs()
        if self.storage.claim_extraction_derived(window, token):
            # These schedulers were and remain best-effort. Claim the handoff
            # before calling them, so an outbox ack failure never repeats it.
            try:
                if plan is not None:
                    service.emit_generation_side_effects(plan)
            except Exception:
                logger.exception(
                    "Derived extraction dispatch failed kind=%s", window.kind
                )
            try:
                from reflexio.server.services.tagging.tagging_scheduler import (
                    schedule_tagging,
                )

                attribution = self.storage.get_request(
                    window.manifest[-1]["request_id"]
                )
                if attribution is not None:
                    schedule_tagging(
                        org_id=self.context.org_id,
                        user_id=window.user_id,
                        agent_version=attribution.agent_version or "",
                        request_context=self.context,
                        llm_client=self.client,
                    )
            except Exception:
                logger.exception("Extraction tagging dispatch failed")
        self.storage.ack_extraction_effects(
            window.window_id, user_id=window.user_id, token=token
        )

    def _billing_snapshot(
        self, window: Window, service: Any, plan: Any
    ) -> dict[str, Any] | None:
        from reflexio.server.llm.token_accounting import RunTokenTotals
        from reflexio.server.metering_signals import (
            count_input_tokens,
            platform_llm_from_config,
        )

        if plan is None:
            return None
        wp = plan.write_plan
        ids = []
        if wp is not None:
            ids = (
                [str(x.profile_id) for x in wp.new_profiles]
                if window.kind == "profile"
                else [str(x.user_playbook_id) for x in wp.new_playbooks]
            )
        if plan.finalization_result is not None:
            ids = plan.finalization_result.learning_ids
        totals = plan.token_totals or RunTokenTotals()
        return {
            "ids": ids,
            "created_at": time.time(),
            "platform_llm": platform_llm_from_config(
                self.context.configurator.get_config()
            ),
            "input_tokens": count_input_tokens(
                service._extraction_input_text(plan.prepared)
            ),
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            # `.get(...)` on the read side, not here: an effects blob written by
            # an older image has no such key, and a KeyError at bill time would
            # strand a window that already committed its learnings.
            "cache_read_input_tokens": totals.cache_read_input_tokens,
            "cache_write_input_tokens": totals.cache_write_input_tokens,
        }

    def _bill(self, window: Window, billing: dict[str, Any]) -> None:
        from reflexio.server.metering_events import (
            record_extraction_tokens,
            record_learnings_generated_records_strict,
        )

        record_learnings_generated_records_strict(
            org_id=self.context.org_id,
            learning_ids=billing["ids"],
            platform_llm=billing["platform_llm"],
            platform_storage=None,
            pipeline=window.kind,
            user_id=window.user_id,
            request_id=window.manifest[-1]["request_id"],
            entity_type="profile" if window.kind == "profile" else "user_playbook",
            created_at=billing["created_at"],
        )
        record_extraction_tokens(
            org_id=self.context.org_id,
            billing_input_tokens=billing["input_tokens"],
            prompt_tokens=billing["prompt_tokens"],
            completion_tokens=billing["completion_tokens"],
            # Tolerant reads: an effects blob written before the cache fields
            # existed is still billable, and a KeyError here would strand a
            # window whose learnings are already committed.
            cache_read_input_tokens=billing.get("cache_read_input_tokens", 0),
            cache_write_input_tokens=billing.get("cache_write_input_tokens", 0),
            platform_llm=billing["platform_llm"],
            platform_storage=None,
            pipeline=window.kind,
            request_id=window.manifest[-1]["request_id"],
            event_key=f"tok:{window.window_id}",
            strict=True,
            created_at=billing["created_at"],
        )
