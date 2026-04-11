"""Unit tests for OperationsMixin.

Tests get_operation_status and cancel_operation with mocked storage
and OperationStateManager.
"""

import time
from unittest.mock import MagicMock, patch

from reflexio.lib._operations import OperationsMixin
from reflexio.models.api_schema.service_schemas import (
    CancelOperationRequest,
    GetOperationStatusRequest,
    OperationStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixin(*, storage_configured: bool = True) -> OperationsMixin:
    """Create an OperationsMixin instance with mocked internals."""
    mixin = object.__new__(OperationsMixin)
    mock_storage = MagicMock()

    mock_request_context = MagicMock()
    mock_request_context.org_id = "test_org"
    mock_request_context.storage = mock_storage if storage_configured else None
    mock_request_context.is_storage_configured.return_value = storage_configured

    mixin.request_context = mock_request_context
    return mixin


def _get_storage(mixin: OperationsMixin) -> MagicMock:
    return mixin.request_context.storage


# ---------------------------------------------------------------------------
# get_operation_status
# ---------------------------------------------------------------------------


class TestGetOperationStatus:
    def test_success_completed(self):
        """Return OperationStatusInfo for a COMPLETED operation."""
        mixin = _make_mixin()
        now = int(time.time())
        _get_storage(mixin).get_operation_state.return_value = {
            "operation_state": {
                "service_name": "profile_generation",
                "status": OperationStatus.COMPLETED.value,
                "started_at": now - 60,
                "completed_at": now,
                "total_users": 5,
                "processed_users": 5,
                "progress_percentage": 100.0,
            }
        }

        request = GetOperationStatusRequest(service_name="profile_generation")
        response = mixin.get_operation_status(request)

        assert response.success is True
        assert response.operation_status is not None
        assert response.operation_status.status == OperationStatus.COMPLETED
        assert response.operation_status.service_name == "profile_generation"

    def test_stale_auto_recovery(self):
        """Stale IN_PROGRESS operation is auto-marked as FAILED."""
        mixin = _make_mixin()
        stale_started_at = int(time.time()) - 9999  # well past threshold
        _get_storage(mixin).get_operation_state.return_value = {
            "operation_state": {
                "service_name": "profile_generation",
                "status": OperationStatus.IN_PROGRESS.value,
                "started_at": stale_started_at,
            }
        }

        request = GetOperationStatusRequest(service_name="profile_generation")
        response = mixin.get_operation_status(request)

        assert response.success is True
        assert response.operation_status is not None
        assert response.operation_status.status == OperationStatus.FAILED
        assert "Auto-recovered" in (response.operation_status.error_message or "")
        # Verify storage was updated
        _get_storage(mixin).update_operation_state.assert_called_once()

    def test_not_found(self):
        """No operation state returns failure with descriptive message."""
        mixin = _make_mixin()
        _get_storage(mixin).get_operation_state.return_value = None

        request = GetOperationStatusRequest(service_name="profile_generation")
        response = mixin.get_operation_status(request)

        assert response.success is False
        assert "No operation found" in (response.msg or "")

    def test_storage_not_configured(self):
        """When storage is not configured, return success with None status."""
        mixin = _make_mixin(storage_configured=False)

        request = GetOperationStatusRequest(service_name="profile_generation")
        response = mixin.get_operation_status(request)

        assert response.success is True
        assert response.operation_status is None
        assert response.msg is not None

    def test_dict_input(self):
        """Accept a plain dict and auto-convert to request object."""
        mixin = _make_mixin()
        now = int(time.time())
        _get_storage(mixin).get_operation_state.return_value = {
            "operation_state": {
                "service_name": "playbook_generation",
                "status": OperationStatus.COMPLETED.value,
                "started_at": now - 30,
                "completed_at": now,
            }
        }

        response = mixin.get_operation_status({"service_name": "playbook_generation"})

        assert response.success is True
        assert response.operation_status is not None


# ---------------------------------------------------------------------------
# cancel_operation
# ---------------------------------------------------------------------------


class TestCancelOperation:
    def test_cancel_single_service(self):
        """Cancel a single named service."""
        mixin = _make_mixin()

        with patch("reflexio.lib._operations.OperationStateManager") as mock_mgr_cls:
            mock_mgr_instance = MagicMock()
            mock_mgr_instance.request_cancellation.return_value = True
            mock_mgr_cls.return_value = mock_mgr_instance

            request = CancelOperationRequest(service_name="profile_generation")
            response = mixin.cancel_operation(request)

        assert response.success is True
        assert "profile_generation" in response.cancelled_services
        assert len(response.cancelled_services) == 1

    def test_cancel_both_services(self):
        """When service_name is None, cancel both profile and playbook generation."""
        mixin = _make_mixin()

        with patch("reflexio.lib._operations.OperationStateManager") as mock_mgr_cls:
            mock_mgr_instance = MagicMock()
            mock_mgr_instance.request_cancellation.return_value = True
            mock_mgr_cls.return_value = mock_mgr_instance

            request = CancelOperationRequest(service_name=None)
            response = mixin.cancel_operation(request)

        assert response.success is True
        assert len(response.cancelled_services) == 2
        assert "profile_generation" in response.cancelled_services
        assert "playbook_generation" in response.cancelled_services

    def test_no_operations_found(self):
        """No in-progress operations to cancel returns empty list."""
        mixin = _make_mixin()

        with patch("reflexio.lib._operations.OperationStateManager") as mock_mgr_cls:
            mock_mgr_instance = MagicMock()
            mock_mgr_instance.request_cancellation.return_value = False
            mock_mgr_cls.return_value = mock_mgr_instance

            request = CancelOperationRequest(service_name=None)
            response = mixin.cancel_operation(request)

        assert response.success is True
        assert response.cancelled_services == []
        assert "No in-progress" in (response.msg or "")

    def test_storage_not_configured(self):
        """Cancel fails gracefully when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        request = CancelOperationRequest(service_name="profile_generation")
        response = mixin.cancel_operation(request)

        assert response.success is False
        assert response.msg is not None

    def test_cancel_with_dict_input(self):
        """Accept a plain dict and auto-convert to CancelOperationRequest."""
        mixin = _make_mixin()

        with patch("reflexio.lib._operations.OperationStateManager") as mock_mgr_cls:
            mock_mgr_instance = MagicMock()
            mock_mgr_instance.request_cancellation.return_value = True
            mock_mgr_cls.return_value = mock_mgr_instance

            response = mixin.cancel_operation({"service_name": "profile_generation"})

        assert response.success is True
        assert "profile_generation" in response.cancelled_services

    def test_cancel_exception_handling(self):
        """Test that exceptions in cancel_operation are caught gracefully."""
        mixin = _make_mixin()

        with patch("reflexio.lib._operations.OperationStateManager") as mock_mgr_cls:
            mock_mgr_cls.side_effect = RuntimeError("storage error")

            request = CancelOperationRequest(service_name="profile_generation")
            response = mixin.cancel_operation(request)

        assert response.success is False
        assert "Failed to cancel" in (response.msg or "")


class TestGetOperationStatusEdgeCases:
    """Additional edge cases for get_operation_status."""

    def test_in_progress_not_stale(self):
        """IN_PROGRESS operation within threshold is returned as-is."""
        mixin = _make_mixin()
        now = int(time.time())
        _get_storage(mixin).get_operation_state.return_value = {
            "operation_state": {
                "service_name": "profile_generation",
                "status": OperationStatus.IN_PROGRESS.value,
                "started_at": now - 10,  # Started 10 seconds ago, well within threshold
            }
        }

        request = GetOperationStatusRequest(service_name="profile_generation")
        response = mixin.get_operation_status(request)

        assert response.success is True
        assert response.operation_status is not None
        assert response.operation_status.status == OperationStatus.IN_PROGRESS

    def test_in_progress_without_started_at(self):
        """IN_PROGRESS operation without started_at is not auto-recovered."""
        mixin = _make_mixin()
        now = int(time.time())
        _get_storage(mixin).get_operation_state.return_value = {
            "operation_state": {
                "service_name": "profile_generation",
                "status": OperationStatus.IN_PROGRESS.value,
                "started_at": now,
                "total_users": 5,
                "processed_users": 2,
                "progress_percentage": 40.0,
                # started_at present and recent, so no auto-recovery
            }
        }

        request = GetOperationStatusRequest(service_name="profile_generation")
        response = mixin.get_operation_status(request)

        assert response.success is True
        assert response.operation_status is not None
        assert response.operation_status.status == OperationStatus.IN_PROGRESS
        # Storage should NOT have been updated since it's not stale
        _get_storage(mixin).update_operation_state.assert_not_called()

    def test_exception_returns_failure(self):
        """Exception during get_operation_status returns failure response."""
        mixin = _make_mixin()
        _get_storage(mixin).get_operation_state.side_effect = RuntimeError(
            "db connection failed"
        )

        request = GetOperationStatusRequest(service_name="profile_generation")
        response = mixin.get_operation_status(request)

        assert response.success is False
        assert "Failed to get operation status" in (response.msg or "")
