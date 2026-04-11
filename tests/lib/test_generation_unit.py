"""Unit tests for GenerationMixin.

Tests run_playbook_aggregation, _run_generation_service,
rerun_profile_generation, manual_profile_generation, rerun_playbook_generation,
manual_playbook_generation, and storage-not-configured error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.lib._base import STORAGE_NOT_CONFIGURED_MSG
from reflexio.lib._generation import GenerationMixin
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixin(*, storage_configured: bool = True) -> GenerationMixin:
    """Create a GenerationMixin instance with mocked internals, bypassing __init__."""
    mixin = object.__new__(GenerationMixin)
    mock_storage = MagicMock()

    mock_request_context = MagicMock()
    mock_request_context.org_id = "test_org"
    mock_request_context.storage = mock_storage if storage_configured else None
    mock_request_context.is_storage_configured.return_value = storage_configured

    mixin.request_context = mock_request_context
    mixin.llm_client = MagicMock()
    return mixin


# ---------------------------------------------------------------------------
# run_playbook_aggregation
# ---------------------------------------------------------------------------


class TestRunPlaybookAggregation:
    @patch("reflexio.lib._generation.PlaybookAggregator")
    def test_calls_aggregator_run_with_correct_args(self, mock_agg_cls):
        """Constructs PlaybookAggregator and calls run() with correct request."""
        mixin = _make_mixin()
        mock_agg_instance = MagicMock()
        mock_agg_cls.return_value = mock_agg_instance

        mixin.run_playbook_aggregation(agent_version="v2", playbook_name="my_fb")

        mock_agg_cls.assert_called_once_with(
            llm_client=mixin.llm_client,
            request_context=mixin.request_context,
            agent_version="v2",
        )
        mock_agg_instance.run.assert_called_once()
        request_arg = mock_agg_instance.run.call_args[0][0]
        assert request_arg.agent_version == "v2"
        assert request_arg.playbook_name == "my_fb"
        assert request_arg.rerun is True

    def test_raises_when_storage_not_configured(self):
        """Raises ValueError when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)

        with pytest.raises(ValueError, match=STORAGE_NOT_CONFIGURED_MSG):
            mixin.run_playbook_aggregation(agent_version="v1", playbook_name="fb")


# ---------------------------------------------------------------------------
# _run_generation_service
# ---------------------------------------------------------------------------


class TestRunGenerationService:
    def test_dict_to_request_conversion(self):
        """Converts dict input to the specified request_type before calling service."""
        mixin = _make_mixin()
        mock_service_cls = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_cls.return_value = mock_service_instance
        mock_service_instance.run_rerun.return_value = "result"

        result = mixin._run_generation_service(
            request={"agent_version": "v1"},
            request_type=RerunPlaybookGenerationRequest,
            service_cls=mock_service_cls,
            output_pending=True,
            run_method="run_rerun",
        )

        assert result == "result"
        mock_service_cls.assert_called_once_with(
            llm_client=mixin.llm_client,
            request_context=mixin.request_context,
            allow_manual_trigger=True,
            output_pending_status=True,
        )
        # Verify the request was converted from dict to the correct type
        call_arg = mock_service_instance.run_rerun.call_args[0][0]
        assert isinstance(call_arg, RerunPlaybookGenerationRequest)
        assert call_arg.agent_version == "v1"

    def test_direct_request_passthrough(self):
        """Passes a proper request object through without conversion."""
        mixin = _make_mixin()
        mock_service_cls = MagicMock()
        mock_service_instance = MagicMock()
        mock_service_cls.return_value = mock_service_instance
        mock_service_instance.run_manual_regular.return_value = "ok"

        original_request = ManualProfileGenerationRequest()
        result = mixin._run_generation_service(
            request=original_request,
            request_type=ManualProfileGenerationRequest,
            service_cls=mock_service_cls,
            output_pending=False,
            run_method="run_manual_regular",
        )

        assert result == "ok"
        # The original request object should be passed through unchanged
        call_arg = mock_service_instance.run_manual_regular.call_args[0][0]
        assert call_arg is original_request


# ---------------------------------------------------------------------------
# rerun_profile_generation
# ---------------------------------------------------------------------------


class TestRerunProfileGeneration:
    def test_calls_profile_service_run_rerun(self):
        """Delegates to ProfileGenerationService.run_rerun with correct args."""
        mixin = _make_mixin()

        with patch.object(
            mixin,
            "_run_generation_service",
            return_value=MagicMock(spec=RerunProfileGenerationResponse),
        ) as mock_run:
            request = RerunProfileGenerationRequest(user_id="u1")
            mixin.rerun_profile_generation(request)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1] or {}
            call_args = mock_run.call_args[0] or ()
            # Verify service_cls and run_method via positional or keyword args
            # The method passes them as keyword args
            assert call_kwargs.get("run_method") == "run_rerun" or (
                len(call_args) >= 5 and call_args[4] == "run_rerun"
            )

    def test_storage_not_configured(self):
        """Returns failure response when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)
        request = RerunProfileGenerationRequest()

        response = mixin.rerun_profile_generation(request)

        assert response.success is False
        assert STORAGE_NOT_CONFIGURED_MSG in (response.msg or "")


# ---------------------------------------------------------------------------
# manual_profile_generation
# ---------------------------------------------------------------------------


class TestManualProfileGeneration:
    def test_calls_profile_service_run_manual_regular(self):
        """Delegates to ProfileGenerationService.run_manual_regular."""
        mixin = _make_mixin()

        with patch.object(
            mixin,
            "_run_generation_service",
            return_value=MagicMock(spec=ManualProfileGenerationResponse),
        ) as mock_run:
            request = ManualProfileGenerationRequest()
            mixin.manual_profile_generation(request)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1] or {}
            call_args = mock_run.call_args[0] or ()
            assert call_kwargs.get("run_method") == "run_manual_regular" or (
                len(call_args) >= 5 and call_args[4] == "run_manual_regular"
            )

    def test_storage_not_configured(self):
        """Returns failure response when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)
        request = ManualProfileGenerationRequest()

        response = mixin.manual_profile_generation(request)

        assert response.success is False
        assert STORAGE_NOT_CONFIGURED_MSG in (response.msg or "")


# ---------------------------------------------------------------------------
# rerun_playbook_generation
# ---------------------------------------------------------------------------


class TestRerunPlaybookGeneration:
    def test_calls_playbook_service_run_rerun(self):
        """Delegates to PlaybookGenerationService.run_rerun."""
        mixin = _make_mixin()

        with patch.object(
            mixin,
            "_run_generation_service",
            return_value=MagicMock(spec=RerunPlaybookGenerationResponse),
        ) as mock_run:
            request = RerunPlaybookGenerationRequest(agent_version="v1")
            mixin.rerun_playbook_generation(request)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1] or {}
            call_args = mock_run.call_args[0] or ()
            assert call_kwargs.get("run_method") == "run_rerun" or (
                len(call_args) >= 5 and call_args[4] == "run_rerun"
            )

    def test_storage_not_configured(self):
        """Returns failure response when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)
        request = RerunPlaybookGenerationRequest(agent_version="v1")

        response = mixin.rerun_playbook_generation(request)

        assert response.success is False
        assert STORAGE_NOT_CONFIGURED_MSG in (response.msg or "")


# ---------------------------------------------------------------------------
# manual_playbook_generation
# ---------------------------------------------------------------------------


class TestManualPlaybookGeneration:
    def test_calls_playbook_service_run_manual_regular(self):
        """Delegates to PlaybookGenerationService.run_manual_regular."""
        mixin = _make_mixin()

        with patch.object(
            mixin,
            "_run_generation_service",
            return_value=MagicMock(spec=ManualPlaybookGenerationResponse),
        ) as mock_run:
            request = ManualPlaybookGenerationRequest(agent_version="v2")
            mixin.manual_playbook_generation(request)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1] or {}
            call_args = mock_run.call_args[0] or ()
            assert call_kwargs.get("run_method") == "run_manual_regular" or (
                len(call_args) >= 5 and call_args[4] == "run_manual_regular"
            )

    def test_storage_not_configured(self):
        """Returns failure response when storage is not configured."""
        mixin = _make_mixin(storage_configured=False)
        request = ManualPlaybookGenerationRequest(agent_version="v2")

        response = mixin.manual_playbook_generation(request)

        assert response.success is False
        assert STORAGE_NOT_CONFIGURED_MSG in (response.msg or "")
