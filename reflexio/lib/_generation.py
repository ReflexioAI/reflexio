import uuid
from typing import Any

from reflexio.lib._base import (
    STORAGE_NOT_CONFIGURED_MSG,
    ReflexioBase,
    _require_storage,
)
from reflexio.models.api_schema.service_schemas import (
    ManualPlaybookGenerationRequest,
    ManualPlaybookGenerationResponse,
    ManualProfileGenerationRequest,
    ManualProfileGenerationResponse,
    RerunPlaybookGenerationRequest,
    RerunPlaybookGenerationResponse,
    RerunProfileGenerationRequest,
    RerunProfileGenerationResponse,
)


class GenerationMixin(ReflexioBase):
    def run_playbook_aggregation(
        self,
        agent_version: str,
        playbook_name: str | None = None,  # noqa: ARG002 — deprecated, accepted but ignored
    ) -> dict:
        """Run playbook aggregation for a given agent version.

        Args:
            agent_version (str): The agent version
            playbook_name (str | None): Deprecated compatibility input. Aggregation is
                singleton (one playbook kind per org), so name-based selection is ignored.

        Returns:
            dict: Aggregation stats (clusters_found, user_playbooks_processed, playbooks_generated)

        Raises:
            ValueError: If storage is not configured
        """
        if not self._is_storage_configured():
            raise ValueError(STORAGE_NOT_CONFIGURED_MSG)
        from reflexio.server.extensions import get_service
        from reflexio.server.services.playbook.aggregation_prompt_processing import (
            AGGREGATION_PROMPT_PROCESSOR,
        )
        from reflexio.server.services.playbook.aggregation_scheduler import (
            aggregation_min_interval_seconds,
        )
        from reflexio.server.services.playbook.components.aggregator import (
            PlaybookAggregator,
        )
        from reflexio.server.services.playbook.playbook_service_utils import (
            PlaybookAggregatorRequest,
        )

        aggregation_prompt_processor = get_service(AGGREGATION_PROMPT_PROCESSOR)
        min_interval_seconds = aggregation_min_interval_seconds()
        aggregator_kwargs = {}
        if aggregation_prompt_processor is not None:
            aggregator_kwargs["aggregation_prompt_processor"] = (
                aggregation_prompt_processor
            )

        aggregator_request = PlaybookAggregatorRequest(
            agent_version=agent_version,
            rerun=True,
        )
        storage = self.request_context.storage
        if storage is None:
            raise ValueError(STORAGE_NOT_CONFIGURED_MSG)
        claim = None
        if getattr(storage, "supports_incremental_playbook_aggregation", False) is True:
            storage.schedule_playbook_aggregation(agent_version)
            claim = storage.claim_due_playbook_aggregation(
                owner=f"admin-rerun:{uuid.uuid4().hex}",
                lease_seconds=1800,
                agent_version=agent_version,
            )
            if claim is None:
                raise RuntimeError(
                    "another playbook aggregation is already running for this organization"
                )
        if claim is not None:
            aggregator_kwargs["aggregation_claim"] = claim
        playbook_aggregator = PlaybookAggregator(
            llm_client=self.llm_client,
            request_context=self.request_context,
            agent_version=agent_version,
            **aggregator_kwargs,
        )
        try:
            result = playbook_aggregator.run(aggregator_request)
        except Exception:
            if claim is not None:
                storage.finish_playbook_aggregation_claim(
                    claim,
                    success=False,
                    retry_after_seconds=60,
                    backlog_retry_after_seconds=1,
                    min_interval_seconds=min_interval_seconds,
                )
            raise
        if claim is not None and not storage.finish_playbook_aggregation_claim(
            claim,
            success=True,
            retry_after_seconds=60,
            backlog_retry_after_seconds=1,
            min_interval_seconds=min_interval_seconds,
        ):
            raise RuntimeError("playbook aggregation rerun lost its database fence")
        return result

    def _run_generation_service(
        self,
        request: Any,
        request_type: type,
        service_cls: type,
        output_pending: bool,
        run_method: str,
    ) -> Any:
        """Shared logic for rerun and manual generation endpoints."""
        if isinstance(request, dict):
            request = request_type(**request)
        service = service_cls(
            llm_client=self.llm_client,
            request_context=self.request_context,
            allow_manual_trigger=True,
            output_pending_status=output_pending,
        )
        return getattr(service, run_method)(request)

    @_require_storage(RerunProfileGenerationResponse, msg_field="msg")
    def rerun_profile_generation(
        self,
        request: RerunProfileGenerationRequest | dict,
    ) -> RerunProfileGenerationResponse:
        """Rerun profile generation for one or all users with filtered interactions.

        Args:
            request (Union[RerunProfileGenerationRequest, dict]): The rerun request

        Returns:
            RerunProfileGenerationResponse: Response containing success status, message, and count of profiles generated
        """
        from reflexio.server.services.profile.service import (
            ProfileGenerationService,
        )

        return self._run_generation_service(
            request,
            RerunProfileGenerationRequest,
            ProfileGenerationService,
            output_pending=True,
            run_method="run_rerun",
        )

    @_require_storage(ManualProfileGenerationResponse, msg_field="msg")
    def manual_profile_generation(
        self,
        request: ManualProfileGenerationRequest | dict,
    ) -> ManualProfileGenerationResponse:
        """Manually trigger profile generation with window-sized interactions and CURRENT output.

        Args:
            request (Union[ManualProfileGenerationRequest, dict]): The request

        Returns:
            ManualProfileGenerationResponse: Response containing success status, message, and count of profiles generated
        """
        from reflexio.server.services.profile.service import (
            ProfileGenerationService,
        )

        return self._run_generation_service(
            request,
            ManualProfileGenerationRequest,
            ProfileGenerationService,
            output_pending=False,
            run_method="run_manual_regular",
        )

    @_require_storage(RerunPlaybookGenerationResponse, msg_field="msg")
    def rerun_playbook_generation(
        self,
        request: RerunPlaybookGenerationRequest | dict,
    ) -> RerunPlaybookGenerationResponse:
        """Rerun playbook generation with filtered interactions.

        Args:
            request (Union[RerunPlaybookGenerationRequest, dict]): The rerun request

        Returns:
            RerunPlaybookGenerationResponse: Response containing success status, message, and count of playbooks generated
        """
        from reflexio.server.services.playbook.service import (
            PlaybookGenerationService,
        )

        return self._run_generation_service(
            request,
            RerunPlaybookGenerationRequest,
            PlaybookGenerationService,
            output_pending=True,
            run_method="run_rerun",
        )

    @_require_storage(ManualPlaybookGenerationResponse, msg_field="msg")
    def manual_playbook_generation(
        self,
        request: ManualPlaybookGenerationRequest | dict,
    ) -> ManualPlaybookGenerationResponse:
        """Manually trigger playbook generation with window-sized interactions and CURRENT output.

        Args:
            request (Union[ManualPlaybookGenerationRequest, dict]): The generation request

        Returns:
            ManualPlaybookGenerationResponse: Response containing success status, message, and count of playbooks generated
        """
        from reflexio.server.services.playbook.service import (
            PlaybookGenerationService,
        )

        return self._run_generation_service(
            request,
            ManualPlaybookGenerationRequest,
            PlaybookGenerationService,
            output_pending=False,
            run_method="run_manual_regular",
        )
