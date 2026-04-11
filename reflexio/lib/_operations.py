import logging

from reflexio.lib._base import STORAGE_NOT_CONFIGURED_MSG, ReflexioBase
from reflexio.models.api_schema.service_schemas import (
    CancelOperationRequest,
    CancelOperationResponse,
    GetOperationStatusRequest,
    GetOperationStatusResponse,
    OperationStatus,
    OperationStatusInfo,
)
from reflexio.server.services.operation_state_utils import OperationStateManager

logger = logging.getLogger(__name__)


class OperationsMixin(ReflexioBase):
    def get_operation_status(
        self, request: GetOperationStatusRequest | dict
    ) -> GetOperationStatusResponse:
        """Get the status of an operation.

        Args:
            request (Union[GetOperationStatusRequest, dict]): Request containing service_name

        Returns:
            GetOperationStatusResponse: Response containing operation status info
        """
        if not self._is_storage_configured():
            return GetOperationStatusResponse(
                success=True, operation_status=None, msg=STORAGE_NOT_CONFIGURED_MSG
            )
        try:
            # Convert dict to request object if needed
            if isinstance(request, dict):
                request = GetOperationStatusRequest(**request)

            # Build the progress key: {service_name}::{org_id}::progress
            org_id = self.request_context.org_id
            progress_key = f"{request.service_name}::{org_id}::progress"

            # Get operation state from storage
            state_entry = self._get_storage().get_operation_state(progress_key)

            if not state_entry:
                return GetOperationStatusResponse(
                    success=False,
                    msg=f"No operation found for service: {request.service_name}",
                )

            # Extract the actual operation_state from the storage wrapper
            # Storage returns: {"service_name": "...", "operation_state": {...}, "updated_at": "..."}
            operation_state = state_entry.get("operation_state", state_entry)

            # Auto-recover stale IN_PROGRESS operations so the frontend
            # doesn't show "in progress" forever after a crash/restart
            if operation_state.get("status") == OperationStatus.IN_PROGRESS.value:
                from datetime import UTC, datetime

                from reflexio.server.services.operation_state_utils import (
                    BATCH_STALE_PROGRESS_SECONDS,
                )

                started_at = operation_state.get("started_at")
                if started_at is not None:
                    current_time = int(datetime.now(UTC).timestamp())
                    elapsed = current_time - started_at
                    if elapsed > BATCH_STALE_PROGRESS_SECONDS:
                        logger.warning(
                            "Stale %s operation detected during status poll "
                            "(started %d seconds ago), auto-marking as FAILED",
                            request.service_name,
                            elapsed,
                        )
                        operation_state["status"] = OperationStatus.FAILED.value
                        operation_state["completed_at"] = current_time
                        operation_state["error_message"] = (
                            f"Auto-recovered: operation was stuck for {elapsed}s "
                            f"(threshold: {BATCH_STALE_PROGRESS_SECONDS}s)"
                        )
                        self._get_storage().update_operation_state(
                            progress_key, operation_state
                        )

            # Convert to OperationStatusInfo
            operation_status_info = OperationStatusInfo(**operation_state)

            return GetOperationStatusResponse(
                success=True,
                operation_status=operation_status_info,
                msg="Retrieved operation status successfully",
            )

        except Exception as e:
            return GetOperationStatusResponse(
                success=False, msg=f"Failed to get operation status: {str(e)}"
            )

    def cancel_operation(
        self, request: CancelOperationRequest | dict
    ) -> CancelOperationResponse:
        """Cancel an in-progress operation (rerun or manual generation).

        Sets a cancellation flag so the batch loop stops before the next user.
        The current LLM call finishes, but no new users are started.

        Args:
            request (Union[CancelOperationRequest, dict]): Request containing optional service_name.
                If service_name is None, cancels both profile_generation and playbook_generation.

        Returns:
            CancelOperationResponse: Response with list of services that were cancelled
        """
        if not self._is_storage_configured():
            return CancelOperationResponse(
                success=False, msg=STORAGE_NOT_CONFIGURED_MSG
            )
        try:
            if isinstance(request, dict):
                request = CancelOperationRequest(**request)

            # Determine which services to cancel
            if request.service_name:
                service_names = [request.service_name]
            else:
                service_names = ["profile_generation", "playbook_generation"]

            cancelled_services = []
            for svc in service_names:
                mgr = OperationStateManager(
                    storage=self._get_storage(),
                    org_id=self.request_context.org_id,
                    service_name=svc,
                )
                if mgr.request_cancellation():
                    cancelled_services.append(svc)

            if cancelled_services:
                return CancelOperationResponse(
                    success=True,
                    cancelled_services=cancelled_services,
                    msg=f"Cancellation requested for: {', '.join(cancelled_services)}",
                )
            return CancelOperationResponse(
                success=True,
                cancelled_services=[],
                msg="No in-progress operations found to cancel",
            )

        except Exception as e:
            return CancelOperationResponse(
                success=False, msg=f"Failed to cancel operation: {str(e)}"
            )
