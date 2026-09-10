"""One explicit-window compute, short fenced commit, then durable effects."""

from __future__ import annotations

import logging
import time
from typing import Any

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.config_schema import (
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.durable_learning.window_codec import (
    decode_plan,
    encode_plan,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
)
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationRequest,
)
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.storage_base import BaseStorage
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

    def execute(self, window: Window, token: str) -> None:
        service, request, config = self.service(window)
        if window.outcome is None:
            known_run = self.storage.get_agent_run(f"window:{window.window_id}")
            if known_run is not None and known_run.committed_output is not None:
                # The gate already admitted this durable result. A new vote on
                # retry must not discard successful output before persistence.
                service._window_skip_should_run = True
            plan = service.compute_generation(request)
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
            if plan is not None:
                service.persist_generation(plan)
            effects = encode_plan(plan, service)
            effects["billing"] = self._billing_snapshot(window, service, plan)
            self.storage.complete_extraction(window, token, effects)

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
        from reflexio.server.billing_signals import (
            count_input_tokens,
            platform_llm_from_config,
        )
        from reflexio.server.llm.token_accounting import RunTokenTotals

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
        }

    def _bill(self, window: Window, billing: dict[str, Any]) -> None:
        from reflexio.server.billing_meter import (
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
            platform_llm=billing["platform_llm"],
            platform_storage=None,
            pipeline=window.kind,
            request_id=window.manifest[-1]["request_id"],
            event_key=f"tok:{window.window_id}",
            strict=True,
            created_at=billing["created_at"],
        )
