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
from reflexio.server.services.playbook.playbook_aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_generation_service import (
    PlaybookGenerationService,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.profile.profile_generation_service import (
    ProfileGenerationService,
)


class GenerationMixin(ReflexioBase):
    def run_playbook_aggregation(self, agent_version: str, playbook_name: str) -> dict:
        """Run playbook aggregation for a given agent version.

        Args:
            agent_version (str): The agent version
            playbook_name (str): The playbook name

        Returns:
            dict: Aggregation stats (clusters_found, user_playbooks_processed, playbooks_generated)

        Raises:
            ValueError: If storage is not configured
        """
        if not self._is_storage_configured():
            raise ValueError(STORAGE_NOT_CONFIGURED_MSG)
        playbook_aggregator = PlaybookAggregator(
            llm_client=self.llm_client,
            request_context=self.request_context,
            agent_version=agent_version,
        )
        aggregator_request = PlaybookAggregatorRequest(
            agent_version=agent_version,
            playbook_name=playbook_name,
            rerun=True,
        )
        return playbook_aggregator.run(aggregator_request)

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
        return self._run_generation_service(
            request,
            ManualPlaybookGenerationRequest,
            PlaybookGenerationService,
            output_pending=False,
            run_method="run_manual_regular",
        )
