"""Explicit profile package qualification; ordinary workers keep their executor.

Reuses WindowExecutor's fenced publication, cursor and effects transaction. No
environment flag or automatic scheduler selects this implementation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, replace
from typing import Any

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server.env_utils import env_bool, env_str
from reflexio.server.llm.input_budget import (
    InputBudgetConfigurationError,
    declared_context_limits,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.base_generation_service import PreparedGenerationRun
from reflexio.server.services.deferred_learning_plan import GenerationComputePlan
from reflexio.server.services.durable_learning.window_codec import encode_plan
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
    matches_frozen_extractor_binding,
)
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowCoordinator,
    OverflowSuspendedError,
)
from reflexio.server.services.extraction.pending_tool_admission import (
    pending_tool_calls_enabled,
)
from reflexio.server.services.extraction.prior_answer_search import (
    CapturedPriorKnowledge,
    capture_prior_knowledge_context,
)
from reflexio.server.services.extraction.profile_package_consolidator import (
    PackageProfileConsolidator,
)
from reflexio.server.services.extraction.profile_package_consumer import (
    consume_profile_package,
)
from reflexio.server.services.extraction.profile_package_preparation import (
    ProfilePackagePreparation,
    prepare_profile_package,
    profile_package_original_seed,
)
from reflexio.server.services.extraction.profile_package_state import profile_witness
from reflexio.server.services.extraction.resumable_agent import (
    encode_committed_output,
)
from reflexio.server.services.extractor_config_utils import get_extractor_name
from reflexio.server.services.profile.components.extractor import (
    MAX_EXISTING_PROFILES_FOR_CONTEXT,
    ProfileExtractor,
)
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.storage_base import AgentRunStatus, canonical_json
from reflexio.server.services.storage.storage_base._extraction_stream import Window


class _PackageProfileService(ProfileGenerationService):
    package_consolidator: PackageProfileConsolidator

    def _create_consolidator(self) -> PackageProfileConsolidator:
        return self.package_consolidator


class _ProfilePackageCoordinator(OverflowCoordinator):
    preparation: ProfilePackagePreparation
    executor: ProfilePackageWindowExecutor

    def require_full_source_coverage(self) -> None:
        # Reconstruct actual selections and actual gate/child receipts. A host
        # coverage list by itself never authorizes publication.
        consume_profile_package(
            preparation=self.preparation,
            coordinator=self,
            client=self.executor.client,
            pending_tool_config=self.executor.context.configurator.get_config().pending_tool_call_config,
            cache_only=True,
        )
        if self.checkpoint.get("stage") not in {"extracted", "gated_false"}:
            raise InputBudgetConfigurationError("Profile processing is incomplete")

    def billing_input_text(self) -> str:
        return "\n".join(
            span.text for span in self.preparation.compression.sources.spans
        )


class ProfilePackageWindowExecutor(WindowExecutor):
    """Opt-in caller-owned execution, without adopting old unfinished journals."""

    def _billing_snapshot(
        self,
        window: Window,
        service: Any,
        plan: Any,
        coordinator: OverflowCoordinator | None = None,
    ) -> dict[str, Any] | None:
        if (
            plan is None
            and coordinator is not None
            and coordinator.checkpoint.get("stage") == "gated_false"
        ):
            return self._token_billing_snapshot(
                [], 0, coordinator.token_totals, created_at=time.time()
            )
        return super()._billing_snapshot(window, service, plan, coordinator)

    def service(self, window: Window) -> tuple[Any, Any, Any]:
        if window.kind != "profile" or env_bool("MOCK_LLM_RESPONSE", default=False):
            raise InputBudgetConfigurationError(
                "Profile package requires real textual profile work"
            )
        return super().service(window)

    def _profile_service(self) -> ProfileGenerationService:
        return _PackageProfileService(self.client, self.context)

    def _coordinator(
        self, window: Window, token: str, service: Any, request: Any, config: Any
    ) -> _ProfilePackageCoordinator:
        known = self.storage.get_agent_run(f"window:{window.window_id}")
        manifest = self._overflow_manifest(known)
        if known is not None and (manifest is None or manifest["version"] != 2):
            raise InputBudgetConfigurationError(
                "Existing extraction requires its original protocol"
            )
        if known is None and window.outcome is not None:
            raise InputBudgetConfigurationError(
                "Cached extraction lacks its package parent"
            )
        groups = service.service_config.window_interactions
        limits = declared_context_limits(
            env_str("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", default="{}")
        )
        pending = self.context.configurator.get_config().pending_tool_call_config
        manager = self.context.prompt_manager
        snapshot = (
            known.generation_request_snapshot.get("profile_package_preparation")
            if known
            else None
        )
        if known:
            if not isinstance(snapshot, dict):
                raise InputBudgetConfigurationError(
                    "Frozen profile preparation unavailable"
                )
            profiles = [
                UserProfile.model_validate(row) for row in snapshot["context_profiles"]
            ]
            context = snapshot["agent_context"]
            prior = (
                CapturedPriorKnowledge(canonical_json(snapshot["prior_context"]))
                if snapshot["prior_context"] is not None
                else None
            )
            manager = PromptManager(
                prompt_bank_path=manager.prompt_bank_path,
                extra_prompt_bank_paths=manager.prompt_bank_paths[1:],
                version_override={
                    key: value["active_version"]
                    for key, value in snapshot["templates"].items()
                },
            )
            gate_enabled = snapshot["consumer"]["gate_enabled"]
            gate_model = (
                snapshot["consumer"]["ladders"]["gate"][0]
                if gate_enabled
                else service._resolve_should_run_model()
            )
            dedup_model = snapshot["consumer"]["ladders"]["dedup"][0]
            tools_active = snapshot["pending_tool_config"] is not None
            template = known.generation_request_snapshot["profile_dedup_template"]
        else:
            profiles = sorted(
                service.service_config.existing_data or [],
                key=lambda row: row.last_modified_timestamp,
                reverse=True,
            )[:MAX_EXISTING_PROFILES_FOR_CONTEXT]
            context = self.context.configurator.get_agent_context()
            tools_active = pending_tool_calls_enabled(self.context)
            prior = (
                capture_prior_knowledge_context(
                    messages=profile_package_original_seed(
                        prompt_manager=manager,
                        groups=groups,
                        existing_profiles=profiles,
                        agent_context=context,
                        extractor_config=config,
                    ),
                    storage=self.storage,
                    org_id=self.storage.org_id,
                    extractor_kind="profile",
                    extractor_config=config,
                    source=request.source,
                    agent_version=None,
                    similarity_threshold=pending.for_tool(
                        "attach_pending_info_request"
                    ).similarity_threshold,
                )
                if tools_active
                else None
            )
            gate_enabled = not (
                window.force or window.policy.get("skip_should_run_check", False)
            )
            gate_model = service._resolve_should_run_model()
            # Resolve the same native dedup model as the ordinary service.
            from reflexio.server.services.profile.components.consolidator import (
                ProfileConsolidator,
            )

            native = ProfileConsolidator(
                request_context=self.context, llm_client=self.client
            )
            dedup_model = native.model_name
            template = manager.get_prompt_template_identity(native._get_prompt_id())
        expected = build_extractor_agent_run_record(
            org_id=self.storage.org_id,
            extractor_kind="profile",
            user_id=window.user_id,
            generation_request_id=request.request_id,
            agent_version=None,
            source=request.source,
            request_interaction_data_models=groups,
            extractor_config=config,
            service_config=service.service_config,
            agent_context=context,
        )
        if known is not None and not matches_frozen_extractor_binding(
            known, expected, window.policy["extractor"]
        ):
            raise InputBudgetConfigurationError("Profile package admission changed")
        preparation = prepare_profile_package(
            groups,
            org_id=self.storage.org_id,
            project_id=window.project_id,
            user_id=window.user_id,
            source=request.source,
            client=self.client,
            context_limits=limits,
            prompt_manager=manager,
            extractor_config=config,
            existing_profiles=profiles,
            agent_context=context,
            gate_enabled=gate_enabled,
            gate_model=gate_model,
            dedup_model=dedup_model,
            prior_context=prior,
            pending_tools_active=tools_active,
            pending_tool_config=pending,
            frozen_pending_tool_config=(
                snapshot["pending_tool_config"] if snapshot is not None else None
            ),
            frozen_extractor_config=(
                known.generation_request_snapshot["extractor_config"] if known else None
            ),
        )
        if snapshot is not None and preparation.snapshot != snapshot:
            raise InputBudgetConfigurationError("Frozen profile preparation changed")
        plan = preparation.compression
        descriptor = {
            "name": "source-package",
            "version": 1,
            "plan_digest": plan.witness,
            "source_cache_digest": plan.source_cache_digest,
            "contract_digest": preparation.contract_digest,
            "scope": {
                "org_id": self.storage.org_id,
                "project_id": window.project_id,
                "user_id": window.user_id,
                "window_id": window.window_id,
                "request_id": request.request_id,
                "agent_version": None,
                "kind": "profile",
            },
            "section_digests": list(plan.native.compression.section_digests),
            "consumer": preparation.snapshot["consumer"],
            "stage_caps": preparation.snapshot["stage_caps"],
        }
        if known is not None and (
            manifest is None or manifest["input_protocol"] != descriptor
        ):
            raise InputBudgetConfigurationError("Profile package admission changed")
        with self.storage.commit_scope():
            self.storage.fence_user_extraction(window.user_id, token)
            self.storage.validate_extraction_inputs(window)
            self.storage.validate_extraction_source_digest(window, plan.source_digest)
            self.storage.validate_extraction_answers(
                prior.answer_witnesses if prior else {}
            )
            witnesses = {row.profile_id: profile_witness(row) for row in profiles}
            self.storage.validate_extraction_historical_profiles(window, witnesses)
            if known is None:
                manifest = OverflowCoordinator.initial_package_manifest(
                    plan.source_digest, descriptor
                )
                manifest["checkpoint"].update(
                    answer_witnesses=prior.answer_witnesses if prior else {},
                    historical_profiles=witnesses,
                )
                expected = replace(expected, id=f"window:{window.window_id}")
                expected.generation_request_snapshot.update(
                    overflow=manifest,
                    profile_package_preparation=preparation.snapshot,
                    profile_dedup_template=template,
                )
                if expected.service_config_snapshot is not None:
                    expected.service_config_snapshot.pop("window_interactions", None)
                    # The selected prompt context already has its own frozen
                    # witness. Do not rewrite every other OLD row on each CAS.
                    expected.service_config_snapshot.pop("existing_data", None)
                known = self.storage.create_agent_run(expected)
                if (
                    known.generation_request_snapshot
                    != expected.generation_request_snapshot
                ):
                    raise InputBudgetConfigurationError(
                        "Profile parent admission lost ownership"
                    )
        if manifest is None:
            raise InputBudgetConfigurationError("Profile manifest unavailable")
        coordinator = _ProfilePackageCoordinator(
            self.storage,
            window,
            token,
            manifest,
            source_digest=plan.source_digest,
            context_limits=limits,
            turn_seconds=max(240, self.client.config.timeout + 60),
            agent_context=context,
        )
        coordinator.saved_run = known
        coordinator.preparation = preparation
        coordinator.executor = self
        service.package_consolidator = PackageProfileConsolidator(
            request_context=self.context,
            llm_client=self.client,
            coordinator=coordinator,
            preparation=preparation,
            template_identity=template,
        )
        return coordinator

    def _compute_overflow_window(
        self, coordinator: Any, service: Any, request: Any, config: Any
    ) -> GenerationComputePlan | None:
        saved = coordinator.checkpoint.get("profile_package_plan")
        if saved is not None:
            from reflexio.server.services.durable_learning.window_codec import (
                decode_plan,
            )

            coordinator.require_full_source_coverage()
            result = consume_profile_package(
                preparation=coordinator.preparation,
                coordinator=coordinator,
                client=self.client,
                pending_tool_config=self.context.configurator.get_config().pending_tool_call_config,
                cache_only=True,
            )
            self._validate_host_binding(coordinator, result)
            self._complete_parent(coordinator, result)
            return decode_plan(
                saved, "profile", config, coordinator.window.user_id, service
            )
        result = consume_profile_package(
            preparation=coordinator.preparation,
            coordinator=coordinator,
            client=self.client,
            pending_tool_config=self.context.configurator.get_config().pending_tool_call_config,
        )
        checkpoint = coordinator.checkpoint
        if not result.gate_decision:
            checkpoint.update(
                stage="gated_false",
                profile_package_plan={"version": 1, "skipped": True},
            )
            coordinator.save_checkpoint(checkpoint)
            return None
        if result.agent is None or result.agent.output is None:
            raise OverflowSuspendedError("Profile package awaits its durable child")
        raw = result.agent.output.model_dump(mode="json").get("profiles") or []
        converted = checkpoint.get("profile_package_candidates")
        if converted is None:
            provenance = (
                asdict(result.agent.model_provenance)
                if result.agent.model_provenance
                else None
            )
            checkpoint.setdefault("candidate_models", {}).update(
                {coordinator.candidate_key(row): provenance for row in raw}
            )
            coordinator.save_checkpoint(checkpoint)
            extractor = ProfileExtractor(
                self.context,
                self.client,
                config,
                service.service_config,
                coordinator.agent_context,
            )
            with coordinator.provider_scope():
                profiles = extractor._convert_raw_to_user_profiles(
                    raw,
                    request.user_id,
                    request.request_id,
                    [row["interaction_id"] for row in coordinator.window.manifest],
                )
            checkpoint = coordinator.checkpoint
            converted = [row.model_dump(mode="json") for row in profiles]
            checkpoint["profile_package_candidates"] = converted
            coordinator.save_checkpoint(checkpoint)
        profiles = [UserProfile.model_validate(row) for row in converted]
        service._last_model_provenance = result.agent.model_provenance
        coordinator.candidate_models.update(
            {row.profile_id: result.agent.model_provenance for row in profiles}
        )
        service._last_extraction_run_ids = [f"window:{coordinator.window.window_id}"]
        service._last_extractor_run_stats = {"total": 1, "failed": 0, "timed_out": 0}
        with coordinator.provider_scope():
            write_plan = service._resolve_write_plan([profiles])
        plan = GenerationComputePlan(
            PreparedGenerationRun(config, get_extractor_name(config), request.user_id),
            len(raw),
            len(write_plan.new_profiles) if write_plan else 0,
            write_plan,
            None,
            time.perf_counter(),
            list(service._last_extraction_run_ids),
            coordinator.token_totals,
        )
        checkpoint = coordinator.checkpoint
        checkpoint.update(
            stage="extracted", profile_package_plan=encode_plan(plan, service)
        )
        checkpoint["profile_package_plan_digest"] = self._plan_digest(
            coordinator, result, checkpoint
        )
        coordinator.save_checkpoint(checkpoint)
        self._complete_parent(coordinator, result)
        return plan

    @staticmethod
    def _plan_digest(coordinator: Any, result: Any, checkpoint: dict[str, Any]) -> str:
        try:
            encoded = json.dumps(
                {
                    "contract": coordinator.preparation.contract_digest,
                    "source": coordinator.source_digest,
                    "child_output": result.agent.output.model_dump(mode="json")
                    if result.agent and result.agent.output
                    else None,
                    "child_provenance": asdict(result.agent.model_provenance)
                    if result.agent and result.agent.model_provenance
                    else None,
                    "gate": result.gate_decision,
                    **{
                        key: checkpoint.get(key)
                        for key in (
                            "profile_package_candidates",
                            "profile_dedup_inputs",
                            "profile_dedup_decision",
                            "historical_profiles",
                            "profile_package_plan",
                        )
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise InputBudgetConfigurationError(
                "Profile write plan is not finite JSON"
            ) from exc
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _validate_host_binding(self, coordinator: Any, result: Any) -> None:
        checkpoint = coordinator.checkpoint
        if not result.gate_decision:
            if checkpoint.get("profile_package_plan") != {
                "version": 1,
                "skipped": True,
            }:
                raise InputBudgetConfigurationError(
                    "Negative profile gate has a write plan"
                )
            return
        if checkpoint.get("profile_package_plan_digest") != self._plan_digest(
            coordinator, result, checkpoint
        ):
            raise InputBudgetConfigurationError(
                "Frozen profile write plan binding changed"
            )

    def _complete_parent(self, coordinator: Any, result: Any) -> None:
        if not result.gate_decision:
            return
        if result.agent is None or result.agent.output is None:
            raise InputBudgetConfigurationError("Completed profile child unavailable")
        with self.storage.commit_scope():
            coordinator.require_full_source_coverage()
            self.storage.validate_extraction_historical_profiles(
                coordinator.window,
                coordinator.checkpoint.get("historical_profiles", {}),
            )
            completed = self.storage.update_agent_run_status(
                f"window:{coordinator.window.window_id}",
                AgentRunStatus.AGENT_COMPLETED,
                committed_output=encode_committed_output(
                    result.agent.output, result.agent.model_provenance
                ),
                expected_statuses=(
                    AgentRunStatus.RUNNING,
                    AgentRunStatus.FAILED,
                    AgentRunStatus.AGENT_COMPLETED,
                ),
            )
            if completed is None or completed.status != AgentRunStatus.AGENT_COMPLETED:
                raise InputBudgetConfigurationError(
                    "Profile package completion lost ownership"
                )

    def _validate_overflow_plan(
        self, coordinator: Any, service: Any, plan: Any
    ) -> None:
        result = consume_profile_package(
            preparation=coordinator.preparation,
            coordinator=coordinator,
            client=self.client,
            pending_tool_config=self.context.configurator.get_config().pending_tool_call_config,
            cache_only=True,
        )
        self._validate_host_binding(coordinator, result)
        self.storage.validate_extraction_historical_profiles(
            coordinator.window, coordinator.checkpoint.get("historical_profiles", {})
        )
        if encode_plan(plan, service) != coordinator.checkpoint.get(
            "profile_package_plan"
        ):
            raise InputBudgetConfigurationError("Cached profile write plan changed")
