"""
Unit tests for BaseGenerationService class.

Tests the abstract base class by creating a concrete implementation for testing.
"""
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportArgumentType=false

import tempfile
import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import Request
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Status,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.base_generation_service import (
    BaseGenerationService,
    ExtractorExecutionError,
    StatusChangeOperation,
    _cheap_should_run_reject,
    _is_pure_slash_command,
)
from reflexio.server.services.extraction.outcome import ExtractionOutcome
from reflexio.server.services.storage.storage_base import AgentRunStatus

# ===============================
# Test Data Classes
# ===============================


@dataclass
class MockExtractorConfig:
    """Mock extractor config for testing."""

    extractor_name: str
    request_sources_enabled: list[str] | None = None
    manual_trigger: bool = False
    window_size_override: int | None = None
    stride_size_override: int | None = None


@dataclass
class MockServiceConfig:
    """Mock service config for testing."""

    user_id: str = "test_user"
    request_id: str = "test_request"
    request_interaction_data_models: list | None = None
    source: str | None = None
    allow_manual_trigger: bool = False
    extractor_names: list[str] | None = None
    auto_run: bool = True
    force_extraction: bool = False


class MockExtractor:
    """Mock extractor for testing sequential execution."""

    def __init__(self, result=None, should_raise=False, exception_message="Test error"):
        self.result = result
        self.should_raise = should_raise
        self.exception_message = exception_message
        self.run_called = False

    def run(self):
        self.run_called = True
        if self.should_raise:
            raise Exception(self.exception_message)
        return self.result


# ===============================
# Concrete Test Implementation
# ===============================


class ConcreteGenerationService(BaseGenerationService):
    """Concrete implementation of BaseGenerationService for testing."""

    def __init__(
        self, llm_client, request_context, extractor_config=None, extractor_configs=None
    ):
        super().__init__(llm_client, request_context)
        if extractor_config is not None:
            self._extractor_config = extractor_config
        elif extractor_configs:
            self._extractor_config = extractor_configs[0]
        else:
            self._extractor_config = None
        self._processed_results = []
        # For upgrade/downgrade testing
        self._items_by_status = {}
        self._deleted_count = 0
        self._updated_count = 0

    def _load_extractor_config(self):
        return self._extractor_config

    def _load_generation_service_config(self, request):
        return request

    def _create_extractor(self, extractor_config, service_config):
        # Return mock extractor that returns the config name as result
        return MockExtractor(result={"extractor_name": extractor_config.extractor_name})

    def _get_service_name(self):
        return "test_generation_service"

    def _process_results(self, results):
        self._processed_results = results

    # Rerun hooks
    def _get_rerun_user_ids(self, request):
        # Get unique user IDs from request interactions
        interactions = getattr(request, "interactions", [])
        user_ids = set()
        for interaction in interactions:
            user_ids.add(interaction.user_id)
        return list(user_ids)

    def _build_rerun_request_params(self, request):
        return {"test_param": "test_value"}

    def _create_run_request_for_item(self, user_id, request):
        return MockServiceConfig(
            user_id=user_id,
            request_id=f"rerun_{user_id}",
            source=getattr(request, "source", None),
        )

    def _create_rerun_response(self, success, msg, count):
        return {"success": success, "message": msg, "count": count}

    def _get_generated_count(self, request, processed_user_ids=None):
        return len(self._processed_results)

    # Upgrade/downgrade hooks
    def _has_items_with_status(self, status, request):
        return (
            status in self._items_by_status and len(self._items_by_status[status]) > 0
        )

    def _delete_items_by_status(self, status, request):
        if status in self._items_by_status:
            count = len(self._items_by_status[status])
            self._items_by_status[status] = []
            self._deleted_count = count
            return count
        return 0

    def _update_items_status(self, old_status, new_status, request, user_ids=None):
        if old_status in self._items_by_status:
            items = self._items_by_status.pop(old_status, [])
            if new_status not in self._items_by_status:
                self._items_by_status[new_status] = []
            self._items_by_status[new_status].extend(items)
            self._updated_count = len(items)
            return len(items)
        return 0

    def _create_status_change_response(self, operation, success, counts, msg):
        return {
            "operation": operation.value,
            "success": success,
            "counts": counts,
            "message": msg,
        }

    # In-progress tracking hooks
    def _get_base_service_name(self):
        return "test_generation"

    def _should_track_in_progress(self):
        return False  # Disabled by default for tests

    def _get_lock_scope_id(self, request):
        return getattr(request, "user_id", None)


# ===============================
# Fixtures
# ===============================


@pytest.fixture
def temp_storage():
    """Create a temporary directory for storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def llm_client():
    """Create a mock LLM client."""
    config = LiteLLMConfig(model="gpt-4o-mini")
    return LiteLLMClient(config)


@pytest.fixture
def request_context(temp_storage):
    """Create a request context with temporary storage."""
    return RequestContext(org_id="test_org", storage_base_dir=temp_storage)


@pytest.fixture
def base_service(llm_client, request_context):
    """Create a concrete generation service for testing."""
    return ConcreteGenerationService(llm_client, request_context)


# ===============================
# Test: _filter_extractor_config_by_service_config
# ===============================


class TestFilterExtractorConfigByServiceConfig:
    """Tests for the _filter_extractor_config_by_service_config method."""

    def test_no_filtering_without_source_attribute(self, base_service):
        """Test that config is not filtered if service_config has no source attribute."""
        config = MockExtractorConfig(extractor_name="extractor1")

        class NoSourceConfig:
            pass

        service_config = NoSourceConfig()
        result = base_service._filter_extractor_config_by_service_config(
            config, service_config
        )
        assert result is config

    def test_filter_by_source_enabled(self, base_service):
        """Test filtering extractor by request_sources_enabled."""
        config = MockExtractorConfig(
            extractor_name="extractor1", request_sources_enabled=["api", "web"]
        )

        service_config = MockServiceConfig(source="api")
        result = base_service._filter_extractor_config_by_service_config(
            config, service_config
        )

        assert result is config

    def test_filter_by_source_disabled(self, base_service):
        """Test that source-mismatched extractor is filtered out."""
        config = MockExtractorConfig(
            extractor_name="extractor1", request_sources_enabled=["mobile"]
        )

        service_config = MockServiceConfig(source="api")
        result = base_service._filter_extractor_config_by_service_config(
            config, service_config
        )

        assert result is None

    def test_filter_by_manual_trigger(self, base_service):
        """Test filtering extractor by manual_trigger flag."""
        config = MockExtractorConfig(extractor_name="extractor1", manual_trigger=True)

        service_config = MockServiceConfig(allow_manual_trigger=False)
        result = base_service._filter_extractor_config_by_service_config(
            config, service_config
        )

        assert result is None

    def test_manual_trigger_allowed_when_allow_manual_trigger_true(self, base_service):
        """Test that manual_trigger extractor is allowed when requested."""
        config = MockExtractorConfig(extractor_name="extractor1", manual_trigger=True)

        service_config = MockServiceConfig(allow_manual_trigger=True)
        result = base_service._filter_extractor_config_by_service_config(
            config, service_config
        )

        assert result is config

    def test_filter_by_matching_extractor_name(self, base_service):
        """Test explicit extractor name filters."""
        config = MockExtractorConfig(extractor_name="extractor1")

        service_config = MockServiceConfig(extractor_names=["extractor1"])
        result = base_service._filter_extractor_config_by_service_config(
            config, service_config
        )

        assert result is config

    def test_filter_by_non_matching_extractor_name(self, base_service):
        """Test explicit extractor name mismatch filters out the config."""
        config = MockExtractorConfig(extractor_name="extractor1")

        service_config = MockServiceConfig(extractor_names=["extractor2"])
        result = base_service._filter_extractor_config_by_service_config(
            config, service_config
        )

        assert result is None

    def test_combined_filtering(self, base_service):
        """Test that all filter conditions are applied together."""
        config = MockExtractorConfig(
            extractor_name="extractor1",
            request_sources_enabled=["api"],
            manual_trigger=False,
        )

        service_config = MockServiceConfig(
            source="api",
            allow_manual_trigger=False,
            extractor_names=["extractor1"],
        )
        result = base_service._filter_extractor_config_by_service_config(
            config, service_config
        )

        assert result is config

    def test_none_source_in_service_config(self, base_service):
        """Test filtering when source is None in service_config."""
        config = MockExtractorConfig(
            extractor_name="extractor1", request_sources_enabled=["api"]
        )

        service_config = MockServiceConfig(source=None)
        result = base_service._filter_extractor_config_by_service_config(
            config, service_config
        )

        assert result is config


# ===============================
# Test: _filter_config_by_stride
# ===============================


class StrideEnabledService(ConcreteGenerationService):
    """Concrete service with stride_size pre-filtering enabled."""

    def _get_extractor_state_service_name(self):
        return "test_extractor"


class TestFilterConfigByStride:
    """Tests for the _filter_config_by_stride method."""

    def _make_request_interaction_models(self, n_interactions: int):
        """Create mock RequestInteractionDataModel objects with n interactions."""
        from reflexio.models.api_schema.internal_schema import (
            RequestInteractionDataModel,
        )
        from reflexio.models.api_schema.service_schemas import Request

        interactions = [
            Interaction(
                interaction_id=i,
                user_id="test_user",
                content=f"message {i}",
                request_id="req1",
                created_at=1000 + i,
                role="User",
            )
            for i in range(n_interactions)
        ]
        request = Request(
            request_id="req1",
            user_id="test_user",
            created_at=1000,
            source="api",
        )
        return [
            RequestInteractionDataModel(
                session_id="req1",
                request=request,
                interactions=interactions,
            )
        ]

    def test_returns_config_when_no_service_name(self, llm_client, request_context):
        """Verify _filter_config_by_stride returns config unchanged when
        _get_extractor_state_service_name() returns None (e.g., AgentSuccessEvaluationService).
        """
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_config=MockExtractorConfig(extractor_name="ext1"),
        )
        service.service_config = MockServiceConfig()

        config = MockExtractorConfig(extractor_name="ext1")
        result = service._filter_config_by_stride(config)
        assert result is config

    def test_returns_config_when_auto_run_false(self, llm_client, request_context):
        """Verify config passes when auto_run=False (rerun/manual mode)."""
        service = StrideEnabledService(
            llm_client,
            request_context,
        )
        service.service_config = MockServiceConfig(auto_run=False)

        config = MockExtractorConfig(extractor_name="ext1")
        result = service._filter_config_by_stride(config)
        assert result is config

    def test_returns_config_when_force_extraction_true(
        self, llm_client, request_context
    ):
        """Verify config passes when force_extraction=True (agent-curated publish)."""
        service = StrideEnabledService(
            llm_client,
            request_context,
        )
        service.service_config = MockServiceConfig(force_extraction=True)

        config = MockExtractorConfig(extractor_name="ext1")
        result = service._filter_config_by_stride(config)
        assert result is config

    def _setup_stride_size_service(
        self, llm_client, request_context, n_new_interactions
    ):
        """Create a StrideEnabledService with mocked storage for stride_size tests.

        Mocks storage and configurator before creating the service so that
        self.storage in the service references the mock.
        """
        # Mock configurator and storage BEFORE creating service so __init__ picks them up
        request_context.configurator = MagicMock()
        request_context.configurator.get_config.return_value = None

        mock_storage = MagicMock()
        new_interactions = self._make_request_interaction_models(n_new_interactions)
        mock_storage.get_operation_state_with_new_request_interaction = MagicMock(
            return_value=({}, new_interactions)
        )
        request_context.storage = mock_storage

        service = StrideEnabledService(
            llm_client,
            request_context,
        )
        return service  # noqa: RET504

    def test_filters_config_when_stride_size_not_met(self, llm_client, request_context):
        """Verify config is dropped when new interaction count < stride_size."""
        service = self._setup_stride_size_service(llm_client, request_context, 2)
        service.service_config = MockServiceConfig(auto_run=True, source="api")

        config = MockExtractorConfig(extractor_name="ext1")
        result = service._filter_config_by_stride(config)
        assert result is None

    def test_passes_config_when_stride_size_met(self, llm_client, request_context):
        """Verify config passes through when new interaction count >= stride_size."""
        service = self._setup_stride_size_service(llm_client, request_context, 6)
        service.service_config = MockServiceConfig(auto_run=True, source="api")

        config = MockExtractorConfig(extractor_name="ext1")
        result = service._filter_config_by_stride(config)
        assert result is config

    def test_handles_source_skip(self, llm_client, request_context):
        """Verify config that fails source filtering in stride_size check is skipped."""
        service = self._setup_stride_size_service(llm_client, request_context, 10)
        service.service_config = MockServiceConfig(auto_run=True, source="api")

        config = MockExtractorConfig(
            extractor_name="ext1", request_sources_enabled=["mobile"]
        )
        result = service._filter_config_by_stride(config)
        assert result is None

    def test_uses_per_extractor_stride_size_override(self, llm_client, request_context):
        """Verify per-extractor stride_size override is respected."""
        service = self._setup_stride_size_service(llm_client, request_context, 3)
        service.service_config = MockServiceConfig(auto_run=True, source="api")

        config = MockExtractorConfig(extractor_name="ext1", stride_size_override=2)
        result = service._filter_config_by_stride(config)
        assert result is config


# ===============================
# Test: _should_run_before_extraction (skip_should_run_check flag)
# ===============================


class TestShouldRunBeforeExtraction:
    """Tests for the skip_should_run_check config flag in _should_run_before_extraction."""

    def test_skip_should_run_check_bypasses_llm_call(self, llm_client, request_context):
        """When skip_should_run_check=True, extraction proceeds without LLM check."""
        from reflexio.models.config_schema import Config

        config = Config(storage_config={"type": "sqlite"}, skip_should_run_check=True)
        request_context.configurator._config = config

        service = ConcreteGenerationService(llm_client, request_context)
        service.service_config = MockServiceConfig(auto_run=True)

        config = MockExtractorConfig(extractor_name="test")
        result = service._should_run_before_extraction(config)

        assert result is True

    def test_default_skip_should_run_check_does_not_bypass(self):
        """When skip_should_run_check=False (default), the flag guard does not fire."""
        from reflexio.models.config_schema import Config

        config = Config(storage_config={"type": "sqlite"})
        assert config.skip_should_run_check is False

    def test_should_run_before_extraction_returns_true_when_force_extraction(
        self, llm_client, request_context
    ):
        """force_extraction=True bypasses cheap pre-filter and LLM should_run vote."""
        service = ConcreteGenerationService(llm_client, request_context)
        service.service_config = MockServiceConfig(auto_run=True, force_extraction=True)

        precheck_spy = MagicMock()
        service._collect_scoped_interactions_for_precheck = precheck_spy  # type: ignore[method-assign]
        llm_call_spy = MagicMock()
        llm_client.generate_chat_response = llm_call_spy

        result = service._should_run_before_extraction(
            MockExtractorConfig(extractor_name="test")
        )

        assert result is True
        precheck_spy.assert_not_called()
        llm_call_spy.assert_not_called()


# ===============================
# Test: run()
# ===============================


class TestRun:
    """Tests for the main run() method."""

    def test_run_with_valid_request(self, llm_client, request_context):
        """Test run() with a valid request containing interactions."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_config=MockExtractorConfig(extractor_name="extractor1"),
        )

        request = MockServiceConfig(
            user_id="test_user",
            request_id="test_request",
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)

        assert len(service._processed_results) == 1
        assert service._processed_results[0]["extractor_name"] == "extractor1"

    def test_run_with_none_request(self, base_service):
        """Test that run() handles None request gracefully."""
        base_service.run(None)
        # Should not raise, just return early
        assert len(base_service._processed_results) == 0

    def test_run_without_interaction_data(self, llm_client, request_context):
        """Test run() when request has no interaction data.

        Note: After the refactor, extractors handle their own data collection.
        When request_interaction_data_models=None, extractors will attempt to
        collect their own interactions rather than the service returning early.
        """
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_config=MockExtractorConfig(extractor_name="extractor1"),
        )

        request = MockServiceConfig(
            user_id="test_user",
            request_id="test_request",
            request_interaction_data_models=None,
        )

        service.run(request)

        # After refactor: extractors run and try to get their own data
        # The mock extractor returns a result, so we expect 1 result
        assert len(service._processed_results) == 1

    def test_run_without_extractor_config(self, llm_client, request_context):
        """Test run() when no extractor config is available."""
        service = ConcreteGenerationService(llm_client, request_context)

        request = MockServiceConfig(
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)

        # Should return early without processing
        assert len(service._processed_results) == 0

    def test_run_stores_service_config(self, llm_client, request_context):
        """Test that run() stores the service_config for later access."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_config=MockExtractorConfig(extractor_name="extractor1"),
        )

        request = MockServiceConfig(
            user_id="test_user",
            request_id="test_request",
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)

        assert service.service_config is not None
        assert service.service_config.user_id == "test_user"

    def test_run_filters_extractor_config(self, llm_client, request_context):
        """Test that run() applies config filtering before creating extractors."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_config=MockExtractorConfig(
                extractor_name="extractor1", request_sources_enabled=["api"]
            ),
        )

        request = MockServiceConfig(
            source="api",
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)

        assert len(service._processed_results) == 1
        assert service._processed_results[0]["extractor_name"] == "extractor1"

    def test_run_skips_filtered_extractor_config(self, llm_client, request_context):
        """Test that run() skips source-mismatched config."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_config=MockExtractorConfig(
                extractor_name="extractor1", request_sources_enabled=["mobile"]
            ),
        )

        request = MockServiceConfig(
            source="api",
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)

        assert len(service._processed_results) == 0

    def test_run_raises_when_configured_extractor_fails(
        self, llm_client, request_context
    ):
        """Test that run() raises when the configured extractor fails."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_config=MockExtractorConfig(extractor_name="extractor1"),
        )
        service._create_extractor = MagicMock(
            side_effect=lambda extractor_config, service_config: MockExtractor(  # noqa: ARG005
                should_raise=True
            )
        )

        request = MockServiceConfig(
            user_id="test_user",
            request_id="test_request",
            request_interaction_data_models=[MagicMock()],
        )

        with pytest.raises(ExtractorExecutionError):
            service.run(request)


# ===============================
# Test: _count_interactions()
# ===============================


# ===============================
# Helper: Mock Operation State Storage
# ===============================


def create_mock_operation_state_storage():
    """Create a mock storage that tracks operation state properly."""
    state_store = {}

    def get_operation_state(service_name):
        return state_store.get(service_name)

    def upsert_operation_state(service_name, state):
        state_store[service_name] = {"operation_state": state}

    def update_operation_state(service_name, state):
        state_store[service_name] = {"operation_state": state}

    return get_operation_state, upsert_operation_state, update_operation_state


# ===============================
# Test: run_rerun()
# ===============================


class TestRunRerun:
    """Tests for the run_rerun() method."""

    def test_rerun_with_valid_interactions(self, llm_client, request_context):
        """Test rerun with valid interactions."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Set up mock storage that tracks state properly
        get_state, upsert_state, update_state = create_mock_operation_state_storage()
        service.storage.get_operation_state = get_state
        service.storage.upsert_operation_state = upsert_state
        service.storage.update_operation_state = update_state

        request = MagicMock()
        request.interactions = [
            Interaction(user_id="user1", request_id="req1", content="test1"),
            Interaction(user_id="user1", request_id="req1", content="test2"),
        ]

        response = service.run_rerun(request)

        assert response["success"] is True
        assert "Completed" in response["message"]

    def test_rerun_blocks_if_in_progress(self, llm_client, request_context):
        """Test that rerun blocks if another operation is in progress."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Mock operation state to show in-progress (with recent started_at so stale detection doesn't trigger)
        from datetime import UTC, datetime

        service.storage.get_operation_state = MagicMock(
            return_value={
                "operation_state": {
                    "status": "in_progress",
                    "started_at": int(datetime.now(UTC).timestamp()),
                }
            }
        )

        request = MagicMock()
        request.interactions = [
            Interaction(user_id="user1", request_id="req1", content="test1")
        ]

        response = service.run_rerun(request)

        assert response["success"] is False
        assert "already in progress" in response["message"]

    def test_rerun_with_no_interactions(self, llm_client, request_context):
        """Test rerun when no interactions match filters."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service.storage.get_operation_state = MagicMock(return_value=None)

        request = MagicMock()
        request.interactions = []

        response = service.run_rerun(request)

        assert response["success"] is False
        assert "No interactions found" in response["message"]

    def test_rerun_groups_by_user(self, llm_client, request_context):
        """Test that rerun processes interactions grouped by user."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Set up mock storage that tracks state properly
        get_state, upsert_state, update_state = create_mock_operation_state_storage()
        service.storage.get_operation_state = get_state
        service.storage.upsert_operation_state = upsert_state
        service.storage.update_operation_state = update_state

        request = MagicMock()
        request.interactions = [
            Interaction(user_id="user1", request_id="req1", content="test1"),
            Interaction(user_id="user2", request_id="req2", content="test2"),
            Interaction(user_id="user1", request_id="req3", content="test3"),
        ]

        response = service.run_rerun(request)

        assert response["success"] is True
        # Should process 2 users (user1 and user2)
        assert "2 user" in response["message"]


# ===============================
# Test: run_upgrade()
# ===============================


class TestRunUpgrade:
    """Tests for the run_upgrade() method."""

    def test_upgrade_promotes_pending_items(self, llm_client, request_context):
        """Test that upgrade promotes pending items to current."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Set up items: pending items exist
        service._items_by_status = {
            Status.PENDING: ["item1", "item2"],
            None: ["old_item"],  # Current items
            Status.ARCHIVED: ["archived_item"],
        }

        request = MagicMock()
        response = service.run_upgrade(request)

        assert response["success"] is True
        assert response["operation"] == "upgrade"
        assert response["counts"]["promoted"] == 2

    def test_upgrade_fails_without_pending_items(self, llm_client, request_context):
        """Test that upgrade fails when no pending items exist."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service._items_by_status = {
            None: ["current_item"],
        }

        request = MagicMock()
        response = service.run_upgrade(request)

        assert response["success"] is False
        assert "No pending items" in response["message"]

    def test_upgrade_archives_current_items(self, llm_client, request_context):
        """Test that upgrade archives current items."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service._items_by_status = {
            Status.PENDING: ["new_item"],
            None: ["current1", "current2", "current3"],
        }

        request = MagicMock()
        response = service.run_upgrade(request)

        assert response["success"] is True
        assert response["counts"]["archived"] == 3

    def test_upgrade_deletes_old_archived_items(self, llm_client, request_context):
        """Test that upgrade deletes old archived items."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service._items_by_status = {
            Status.PENDING: ["new_item"],
            Status.ARCHIVED: ["old1", "old2"],
        }

        request = MagicMock()
        response = service.run_upgrade(request)

        assert response["success"] is True
        assert response["counts"]["deleted"] == 2

    def test_upgrade_with_archive_current_false_skips_archive(
        self, llm_client, request_context
    ):
        """Test that upgrade with archive_current=False only promotes pending items."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service._items_by_status = {
            Status.PENDING: ["new1", "new2"],
            None: ["current1", "current2", "current3"],
            Status.ARCHIVED: ["archived1"],
        }

        request = MagicMock()
        request.archive_current = False
        response = service.run_upgrade(request)

        assert response["success"] is True
        assert response["counts"]["promoted"] == 2
        assert response["counts"]["archived"] == 0
        assert response["counts"]["deleted"] == 0
        # Current items (3 original + 2 promoted) should all have None status
        assert len(service._items_by_status.get(None, [])) == 5
        # Archived items should still exist (not deleted)
        assert len(service._items_by_status.get(Status.ARCHIVED, [])) == 1

    def test_upgrade_default_behavior_archives_current(
        self, llm_client, request_context
    ):
        """Test that upgrade without archive_current attribute archives as before."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service._items_by_status = {
            Status.PENDING: ["new1"],
            None: ["current1", "current2"],
            Status.ARCHIVED: ["archived1"],
        }

        # Request without archive_current attribute (simulates old callers)
        request = MagicMock(spec=[])
        response = service.run_upgrade(request)

        assert response["success"] is True
        assert response["counts"]["promoted"] == 1
        assert response["counts"]["archived"] == 2
        assert response["counts"]["deleted"] == 1


# ===============================
# Test: run_downgrade()
# ===============================


class TestRunDowngrade:
    """Tests for the run_downgrade() method."""

    def test_downgrade_restores_archived_items(self, llm_client, request_context):
        """Test that downgrade restores archived items to current."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service._items_by_status = {
            None: ["current_item"],
            Status.ARCHIVED: ["archived1", "archived2"],
        }

        request = MagicMock()
        response = service.run_downgrade(request)

        assert response["success"] is True
        assert response["operation"] == "downgrade"
        assert response["counts"]["restored"] == 2

    def test_downgrade_fails_without_archived_items(self, llm_client, request_context):
        """Test that downgrade fails when no archived items exist."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service._items_by_status = {
            None: ["current_item"],
        }

        request = MagicMock()
        response = service.run_downgrade(request)

        assert response["success"] is False
        assert "No archived items" in response["message"]

    def test_downgrade_demotes_current_items(self, llm_client, request_context):
        """Test that downgrade demotes current items to archived."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service._items_by_status = {
            None: ["current1", "current2"],
            Status.ARCHIVED: ["archived1"],
        }

        request = MagicMock()
        response = service.run_downgrade(request)

        assert response["success"] is True
        assert response["counts"]["demoted"] == 2


# ===============================
# Test: StatusChangeOperation Enum
# ===============================


class TestStatusChangeOperation:
    """Tests for the StatusChangeOperation enum."""

    def test_upgrade_value(self):
        """Test UPGRADE enum value."""
        assert StatusChangeOperation.UPGRADE.value == "upgrade"

    def test_downgrade_value(self):
        """Test DOWNGRADE enum value."""
        assert StatusChangeOperation.DOWNGRADE.value == "downgrade"


# ===============================
# Test: Error Handling
# ===============================


class TestErrorHandling:
    """Tests for error handling in BaseGenerationService."""

    def test_run_handles_exception_in_load_config(self, llm_client, request_context):
        """Test that run() handles exceptions during config loading."""

        class FailingService(ConcreteGenerationService):
            def _load_generation_service_config(self, request):
                raise ValueError("Config loading failed")

        service = FailingService(llm_client, request_context)
        request = MockServiceConfig(request_interaction_data_models=[MagicMock()])

        # Should not raise, just log warning
        service.run(request)
        assert len(service._processed_results) == 0

    def test_run_handles_exception_in_extractor(self, llm_client, request_context):
        """Test that run() raises ExtractorExecutionError when all extractors fail."""

        class FailingExtractorService(ConcreteGenerationService):
            def _create_extractor(self, extractor_config, service_config):
                return MockExtractor(should_raise=True)

        service = FailingExtractorService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )
        request = MockServiceConfig(request_interaction_data_models=[MagicMock()])

        with pytest.raises(ExtractorExecutionError):
            service.run(request)
        assert len(service._processed_results) == 0

    def test_rerun_handles_item_processing_exception(self, llm_client, request_context):
        """Test that rerun handles exceptions during item processing."""

        class FailingRunService(ConcreteGenerationService):
            def run(self, request):
                if hasattr(request, "user_id") and request.user_id == "failing_user":
                    raise Exception("Processing failed")
                super().run(request)

        service = FailingRunService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Set up mock storage that tracks state properly
        get_state, upsert_state, update_state = create_mock_operation_state_storage()
        service.storage.get_operation_state = get_state
        service.storage.upsert_operation_state = upsert_state
        service.storage.update_operation_state = update_state

        request = MagicMock()
        request.interactions = [
            Interaction(user_id="failing_user", request_id="req1", content="test1"),
            Interaction(user_id="success_user", request_id="req2", content="test2"),
        ]

        response = service.run_rerun(request)

        # Should still complete successfully for other users
        assert response["success"] is True


# ===============================
# Test: In-Progress Lock Mechanism
# ===============================


class InProgressTrackingService(ConcreteGenerationService):
    """Concrete implementation with in-progress tracking enabled."""

    def __init__(self, llm_client, request_context, extractor_configs=None):
        super().__init__(llm_client, request_context, extractor_configs)
        self._generation_count = 0  # Tracks _run_generation calls

    def _should_track_in_progress(self):
        return True  # Enable in-progress tracking

    def _get_base_service_name(self):
        return "test_generation"

    def _get_lock_scope_id(self, request):
        return getattr(request, "user_id", "unknown")

    def _run_generation(self, request):
        """Override to track generation calls."""
        self._generation_count += 1
        # Don't call super() to avoid needing real extractors


class TestInProgressLockMechanism:
    """Tests for the in-progress lock acquisition and release mechanism."""

    def test_lock_acquired_when_no_existing_lock(self, llm_client, request_context):
        """Test that lock is acquired when no lock exists."""
        service = InProgressTrackingService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Mock storage to simulate no existing lock and successful lock acquisition
        service.storage.try_acquire_in_progress_lock = MagicMock(
            return_value={"acquired": True}
        )
        # Return state showing we own the lock with no pending
        service.storage.get_operation_state = MagicMock(
            return_value={
                "operation_state": {
                    "in_progress": True,
                    "current_request_id": "request_1",
                    "pending_request_id": None,
                }
            }
        )
        service.storage.upsert_operation_state = MagicMock()

        request = MockServiceConfig(user_id="test_user", request_id="request_1")
        service.run(request)

        # Verify lock acquisition was attempted and generation ran
        service.storage.try_acquire_in_progress_lock.assert_called_once()
        assert service._generation_count == 1

    def test_lock_not_acquired_when_another_operation_in_progress(
        self, llm_client, request_context
    ):
        """Test that lock is not acquired when another operation is running."""
        service = InProgressTrackingService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Mock storage to simulate existing lock (not acquired)
        service.storage.try_acquire_in_progress_lock = MagicMock(
            return_value={"acquired": False}
        )

        request = MockServiceConfig(user_id="test_user", request_id="request_2")
        service.run(request)

        # Verify generation was NOT run (lock not acquired)
        assert service._generation_count == 0

    def test_stale_lock_is_overridden(self, llm_client, request_context):
        """Test that stale locks (>5 min) are overridden."""
        service = InProgressTrackingService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Mock storage to simulate stale lock that gets acquired
        # The storage.try_acquire_in_progress_lock handles stale lock detection
        service.storage.try_acquire_in_progress_lock = MagicMock(
            return_value={"acquired": True, "was_stale": True}
        )
        service.storage.get_operation_state = MagicMock(
            return_value={
                "operation_state": {
                    "in_progress": True,
                    "current_request_id": "request_3",
                    "pending_request_id": None,
                }
            }
        )
        service.storage.upsert_operation_state = MagicMock()

        request = MockServiceConfig(user_id="test_user", request_id="request_3")
        service.run(request)

        # Verify lock was acquired (stale lock overridden)
        assert service._generation_count == 1

    def test_pending_request_triggers_rerun(self, llm_client, request_context):
        """Test that pending_request_id triggers a re-run after completion."""
        service = InProgressTrackingService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Track call count for get_operation_state (used by release_lock)
        release_call_count = [0]

        def mock_get_state(state_key):
            release_call_count[0] += 1
            if release_call_count[0] == 1:
                # First call: return pending request to trigger re-run
                return {
                    "operation_state": {
                        "in_progress": True,
                        "current_request_id": "request_1",
                        "pending_request_id": "request_2",
                    }
                }
            # Subsequent calls: no more pending requests
            return {
                "operation_state": {
                    "in_progress": True,
                    "current_request_id": "request_2",
                    "pending_request_id": None,
                }
            }

        service.storage.try_acquire_in_progress_lock = MagicMock(
            return_value={"acquired": True}
        )
        service.storage.get_operation_state = mock_get_state
        service.storage.upsert_operation_state = MagicMock()

        request = MockServiceConfig(user_id="test_user", request_id="request_1")
        service.run(request)

        # Verify _run_generation was called twice (initial + re-run for pending request)
        assert service._generation_count == 2

    def test_lock_cleared_on_exception(self, llm_client, request_context):
        """Test that lock is cleared when an exception occurs during generation."""

        class FailingInProgressService(InProgressTrackingService):
            def _run_generation(self, request):
                raise Exception("Generation failed!")

        service = FailingInProgressService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        service.storage.try_acquire_in_progress_lock = MagicMock(
            return_value={"acquired": True}
        )
        service.storage.upsert_operation_state = MagicMock()

        request = MockServiceConfig(user_id="test_user", request_id="request_1")

        # Should raise but lock should be cleared
        with pytest.raises(Exception, match="Generation failed!"):
            service.run(request)

        # Verify lock was cleared (upsert with in_progress=False)
        clear_call = service.storage.upsert_operation_state.call_args
        assert clear_call is not None
        state_arg = clear_call[0][1]
        assert state_arg["in_progress"] is False
        assert state_arg["current_request_id"] is None
        assert state_arg["pending_request_id"] is None

    def test_release_lock_no_pending_clears_state(self, llm_client, request_context):
        """Test that releasing lock with no pending request clears the state."""
        from reflexio.server.services.operation_state_utils import (
            OperationStateManager,
        )

        service = InProgressTrackingService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Mock storage to return state with matching request_id and no pending
        service.storage.get_operation_state = MagicMock(
            return_value={
                "operation_state": {
                    "in_progress": True,
                    "current_request_id": "my_request",
                    "pending_request_id": None,
                }
            }
        )
        service.storage.upsert_operation_state = MagicMock()

        mgr = OperationStateManager(service.storage, service.org_id, "test_generation")
        result = mgr.release_lock("my_request", scope_id="test_user")

        # Should return None (no pending) and clear the lock
        assert result is None
        service.storage.upsert_operation_state.assert_called_once()
        state_arg = service.storage.upsert_operation_state.call_args[0][1]
        assert state_arg["in_progress"] is False
        assert state_arg["current_request_id"] is None
        assert state_arg["pending_request_id"] is None

    def test_release_lock_with_pending_transfers_ownership(
        self, llm_client, request_context
    ):
        """Test that releasing lock with pending request transfers ownership."""
        from reflexio.server.services.operation_state_utils import (
            OperationStateManager,
        )

        service = InProgressTrackingService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Mock storage to return state with pending request
        service.storage.get_operation_state = MagicMock(
            return_value={
                "operation_state": {
                    "in_progress": True,
                    "current_request_id": "my_request",
                    "pending_request_id": "new_request",
                }
            }
        )
        service.storage.upsert_operation_state = MagicMock()

        mgr = OperationStateManager(service.storage, service.org_id, "test_generation")
        result = mgr.release_lock("my_request", scope_id="test_user")

        # Should return pending_request_id and transfer ownership
        assert result == "new_request"
        service.storage.upsert_operation_state.assert_called_once()
        state_arg = service.storage.upsert_operation_state.call_args[0][1]
        assert state_arg["in_progress"] is True
        assert state_arg["current_request_id"] == "new_request"
        assert state_arg["pending_request_id"] is None

    def test_release_lock_ignores_if_not_owner(self, llm_client, request_context):
        """Test that release does nothing if caller is not the current owner."""
        from reflexio.server.services.operation_state_utils import (
            OperationStateManager,
        )

        service = InProgressTrackingService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Mock storage to return state owned by different request
        service.storage.get_operation_state = MagicMock(
            return_value={
                "operation_state": {
                    "in_progress": True,
                    "current_request_id": "other_request",
                    "pending_request_id": "another_pending",
                }
            }
        )
        service.storage.upsert_operation_state = MagicMock()

        mgr = OperationStateManager(service.storage, service.org_id, "test_generation")
        result = mgr.release_lock("my_request", scope_id="test_user")

        # Should return None and NOT update state (not the owner)
        assert result is None
        service.storage.upsert_operation_state.assert_not_called()


class TestPayloadAwareDrain:
    """Drain loop must rerun against the QUEUED request's payload, not the
    original holder's. This is the R2 fix (reflexio-enterprise#59).
    """

    def test_drain_uses_queued_payload_not_original(self, llm_client, request_context):
        """When a queue entry comes off, _run_generation runs with the queued
        payload's user_id/request_id — the previous bug ran with the holder's."""
        from pydantic import BaseModel

        class PydanticTestRequest(BaseModel):
            user_id: str
            request_id: str

        seen_user_ids: list[str] = []

        class TrackingService(InProgressTrackingService):
            def _run_generation(self, request):
                seen_user_ids.append(getattr(request, "user_id", None))
                # Don't call super() — base just bumps a counter.

        service = TrackingService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # First call returns state with a queued entry; second call shows
        # empty queue so drain stops.
        call_count = [0]

        def mock_get_state(state_key):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "operation_state": {
                        "in_progress": True,
                        "current_request_id": "request_holder",
                        "pending_request_queue": [
                            {
                                "request_id": "request_queued",
                                "payload": {
                                    "user_id": "queued_user",
                                    "request_id": "request_queued",
                                },
                            },
                        ],
                    }
                }
            return {
                "operation_state": {
                    "in_progress": True,
                    "current_request_id": "request_queued",
                    "pending_request_queue": [],
                }
            }

        service.storage.try_acquire_in_progress_lock = MagicMock(
            return_value={"acquired": True}
        )
        service.storage.get_operation_state = mock_get_state
        service.storage.upsert_operation_state = MagicMock()

        original = PydanticTestRequest(
            user_id="holder_user", request_id="request_holder"
        )
        service.run(original)

        assert seen_user_ids == ["holder_user", "queued_user"], (
            f"Drain ran with wrong user_ids: {seen_user_ids} "
            "(expected holder first, then queued)"
        )

    def test_drain_legacy_pending_request_id_falls_back_to_original(
        self, llm_client, request_context
    ):
        """Mid-deploy back-compat: if the storage row lacks the queue field
        but has the legacy pending_request_id, the rerun should still happen
        — using the original request, matching pre-fix behaviour."""
        from pydantic import BaseModel

        class PydanticTestRequest(BaseModel):
            user_id: str
            request_id: str

        seen_user_ids: list[str] = []

        class TrackingService(InProgressTrackingService):
            def _run_generation(self, request):
                seen_user_ids.append(getattr(request, "user_id", None))

        service = TrackingService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        call_count = [0]

        def mock_get_state(state_key):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "operation_state": {
                        "in_progress": True,
                        "current_request_id": "request_holder",
                        "pending_request_id": "request_legacy",
                        # No pending_request_queue field
                    }
                }
            return {
                "operation_state": {
                    "in_progress": True,
                    "current_request_id": "request_legacy",
                    "pending_request_queue": [],
                }
            }

        service.storage.try_acquire_in_progress_lock = MagicMock(
            return_value={"acquired": True}
        )
        service.storage.get_operation_state = mock_get_state
        service.storage.upsert_operation_state = MagicMock()

        original = PydanticTestRequest(
            user_id="holder_user", request_id="request_holder"
        )
        service.run(original)

        # Legacy entry has no payload → fallback to original holder's request.
        assert seen_user_ids == ["holder_user", "holder_user"]


# ===============================
# Test: Extractor Names Filtering in Rerun
# ===============================


class TestRerunWithExtractorNamesFilter:
    """Tests for extractor_names filtering during rerun operations."""

    def test_rerun_respects_extractor_names_filter(self, llm_client, request_context):
        """Test that rerun only runs extractors specified in extractor_names."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="extractor1"),
                MockExtractorConfig(extractor_name="extractor2"),
                MockExtractorConfig(extractor_name="extractor3"),
            ],
        )

        # Set up mock storage
        get_state, upsert_state, update_state = create_mock_operation_state_storage()
        service.storage.get_operation_state = get_state
        service.storage.upsert_operation_state = upsert_state
        service.storage.update_operation_state = update_state

        # Create request with extractor_names filter
        request = MagicMock()
        request.interactions = [
            Interaction(user_id="user1", request_id="req1", content="test1")
        ]
        request.extractor_names = ["extractor1", "extractor3"]

        # Override _create_run_request_for_item to pass extractor_names
        original_create = service._create_run_request_for_item

        def create_with_names(user_id, req):
            result = original_create(user_id, req)
            result.extractor_names = getattr(req, "extractor_names", None)
            return result

        service._create_run_request_for_item = create_with_names

        response = service.run_rerun(request)

        assert response["success"] is True


# ===============================
# Test: Cancellation in batch operations
# ===============================


class TestCancellationInBatch:
    """Tests for cancellation during batch operations."""

    def test_batch_stops_on_cancellation(self, llm_client, request_context):
        """Test that _run_batch_with_progress stops when cancellation is requested."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Set up mock storage that tracks operation state and simulates cancellation
        # via a separate cancellation key (matching the new implementation)
        state_store = {}
        cancellation_key = "test_generation::test_org::cancellation"

        def get_state(key):
            return state_store.get(key)

        def upsert_state(key, state):
            state_store[key] = {"operation_state": state}

        def update_state(key, state):
            # After processing user1, simulate cancellation being requested
            # by writing to the separate cancellation key
            if (
                state.get("current_user_id") is None
                and len(state.get("processed_user_ids", [])) >= 1
            ):
                state_store[cancellation_key] = {
                    "operation_state": {"cancellation_requested": True}
                }
            state_store[key] = {"operation_state": state}

        service.storage.get_operation_state = get_state
        service.storage.upsert_operation_state = upsert_state
        service.storage.update_operation_state = update_state

        request = MagicMock()
        request.interactions = [
            Interaction(user_id="user1", request_id="req1", content="test1"),
            Interaction(user_id="user2", request_id="req2", content="test2"),
            Interaction(user_id="user3", request_id="req3", content="test3"),
        ]

        response = service.run_rerun(request)

        assert response["success"] is True
        # user1 processed, then cancellation detected before user2/user3
        assert response["count"] < 3  # Cancellation should stop before all users

    def test_fresh_rerun_works_after_cancel(self, llm_client, request_context):
        """Test that a new rerun works after a cancelled operation (status = cancelled, not in_progress)."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        # Set up initial state as CANCELLED
        state_store = {}
        progress_key = "test_generation::test_org::progress"
        state_store[progress_key] = {
            "operation_state": {
                "status": "cancelled",
            }
        }

        def get_state(key):
            return state_store.get(key)

        def upsert_state(key, state):
            state_store[key] = {"operation_state": state}

        def update_state(key, state):
            state_store[key] = {"operation_state": state}

        service.storage.get_operation_state = get_state
        service.storage.upsert_operation_state = upsert_state
        service.storage.update_operation_state = update_state

        request = MagicMock()
        request.interactions = [
            Interaction(user_id="user1", request_id="req1", content="test1"),
        ]

        # check_in_progress should NOT block since status is "cancelled" (not "in_progress")
        response = service.run_rerun(request)
        assert response["success"] is True

    def test_is_batch_mode_reset_after_batch(self, llm_client, request_context):
        """Test that _is_batch_mode is reset to False after batch completes."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        get_state, upsert_state, update_state = create_mock_operation_state_storage()
        service.storage.get_operation_state = get_state
        service.storage.upsert_operation_state = upsert_state
        service.storage.update_operation_state = update_state

        request = MagicMock()
        request.interactions = [
            Interaction(user_id="user1", request_id="req1", content="test1"),
        ]

        assert service._is_batch_mode is False
        service.run_rerun(request)
        assert service._is_batch_mode is False  # Reset after batch finishes


# ===============================
# Test: Sequential Execution
# ===============================


class TestSequentialExecution:
    """Tests for the single configured extractor execution in _run_generation."""

    def test_sequential_single_extractor(self, llm_client, request_context):
        """Test sequential execution with a single extractor."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        request = MockServiceConfig(
            user_id="test_user",
            request_id="test_request",
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)

        # Single extractor should produce one result
        assert len(service._processed_results) == 1

    def test_legacy_multiple_configs_run_first_config_only(
        self, llm_client, request_context
    ):
        """Test that legacy config lists normalize to the first config."""
        call_order = []

        class TrackingExtractor:
            def __init__(self, name, result):
                self.name = name
                self.result = result

            def run(self):
                call_order.append(self.name)
                return self.result

        class TrackingService(ConcreteGenerationService):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._process_calls = []

            def _create_extractor(self, extractor_config, service_config):
                return TrackingExtractor(
                    extractor_config.extractor_name,
                    {"name": extractor_config.extractor_name},
                )

            def _process_results(self, results):
                self._process_calls.append(list(results))

        service = TrackingService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="ext1"),
                MockExtractorConfig(extractor_name="ext2"),
                MockExtractorConfig(extractor_name="ext3"),
            ],
        )

        request = MockServiceConfig(
            user_id="test_user",
            request_id="test_request",
        )

        service.run(request)

        assert call_order == ["ext1"]
        assert len(service._process_calls) == 1
        assert service._process_calls[0] == [{"name": "ext1"}]

    def test_configured_extractor_failure_raises(self, llm_client, request_context):
        """Test that failure in the configured extractor fails the run."""

        class PartialService(ConcreteGenerationService):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._process_calls = []

            def _create_extractor(self, extractor_config, service_config):
                should_raise = extractor_config.extractor_name == "failing"
                return MockExtractor(
                    result={"name": extractor_config.extractor_name},
                    should_raise=should_raise,
                )

            def _process_results(self, results):
                self._process_calls.append(list(results))

        service = PartialService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="failing"),
                MockExtractorConfig(extractor_name="ext3"),
            ],
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")

        with pytest.raises(ExtractorExecutionError):
            service.run(request)
        assert service._process_calls == []

    def test_sequential_all_fail_raises(self, llm_client, request_context):
        """Test that all extractors failing raises ExtractorExecutionError."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="ext1"),
                MockExtractorConfig(extractor_name="ext2"),
            ],
        )
        service._create_extractor = MagicMock(
            side_effect=lambda ec, sc: MockExtractor(should_raise=True)  # noqa: ARG005
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")

        with pytest.raises(ExtractorExecutionError):
            service.run(request)

    def test_sequential_uses_single_loaded_config_for_extractors(
        self, llm_client, request_context
    ):
        """Test that service_config is loaded once for an extraction run."""
        load_config_calls = []

        class RefetchService(ConcreteGenerationService):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._process_calls = []

            def _load_generation_service_config(self, request):
                config = super()._load_generation_service_config(request)
                load_config_calls.append(config)
                return config

            def _create_extractor(self, extractor_config, service_config):
                return MockExtractor(result={"name": extractor_config.extractor_name})

            def _process_results(self, results):
                self._process_calls.append(list(results))

        service = RefetchService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="ext1"),
                MockExtractorConfig(extractor_name="ext2"),
            ],
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")
        service.run(request)

        assert len(load_config_calls) == 1

    def test_sequential_does_not_set_incremental_flag(
        self, llm_client, request_context
    ):
        """Extractor configs run independently without incremental service state."""
        observed_incremental = []

        class IncrementalTracker(ConcreteGenerationService):
            def _create_extractor(self, extractor_config, service_config):
                observed_incremental.append(
                    getattr(service_config, "is_incremental", False)
                )
                return MockExtractor(result={"name": extractor_config.extractor_name})

            def _process_results(self, results):
                pass

        service = IncrementalTracker(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="ext1"),
                MockExtractorConfig(extractor_name="ext2"),
                MockExtractorConfig(extractor_name="ext3"),
            ],
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")
        service.run(request)

        assert observed_incremental == [False]

    def test_sequential_does_not_pass_previously_extracted(
        self, llm_client, request_context
    ):
        """Extractor configs do not receive cross-extractor results."""
        observed_previously = []

        class PreviousTracker(ConcreteGenerationService):
            def _create_extractor(self, extractor_config, service_config):
                observed_previously.append(
                    list(getattr(service_config, "previously_extracted", []))
                )
                return MockExtractor(result={"name": extractor_config.extractor_name})

            def _process_results(self, results):
                pass

        service = PreviousTracker(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="ext1"),
                MockExtractorConfig(extractor_name="ext2"),
                MockExtractorConfig(extractor_name="ext3"),
            ],
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")
        service.run(request)

        assert observed_previously == [[]]

    def test_sequential_none_results_do_not_create_incremental_state(
        self, llm_client, request_context
    ):
        """None results do not change the independent extractor state model."""
        observed_previously = []

        class NoneResultTracker(ConcreteGenerationService):
            def _create_extractor(self, extractor_config, service_config):
                observed_previously.append(
                    list(getattr(service_config, "previously_extracted", []))
                )
                # ext2 returns None
                if extractor_config.extractor_name == "ext2":
                    return MockExtractor(result=None)
                return MockExtractor(result={"name": extractor_config.extractor_name})

            def _process_results(self, results):
                pass

        service = NoneResultTracker(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="ext1"),
                MockExtractorConfig(extractor_name="ext2"),
                MockExtractorConfig(extractor_name="ext3"),
            ],
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")
        service.run(request)

        assert observed_previously == [[]]

    def test_sequential_completed_outcome_is_unwrapped(
        self, llm_client, request_context
    ):
        """ExtractionOutcome.completed contributes its item list as one result."""

        class OutcomeService(ConcreteGenerationService):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._process_calls = []

            def _create_extractor(self, extractor_config, service_config):
                return MockExtractor(
                    result=ExtractionOutcome.completed(
                        [{"name": extractor_config.extractor_name}]
                    )
                )

            def _process_results(self, results):
                self._process_calls.append(list(results))

        service = OutcomeService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="ext1")],
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")
        service.run(request)

        assert service._process_calls == [[[{"name": "ext1"}]]]

    def test_sequential_empty_outcome_is_success_without_processing(
        self, llm_client, request_context
    ):
        """ExtractionOutcome.empty is a successful run with no persisted output."""

        class EmptyOutcomeService(ConcreteGenerationService):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._process_calls = []

            def _create_extractor(self, extractor_config, service_config):
                return MockExtractor(result=ExtractionOutcome.empty())

            def _process_results(self, results):
                self._process_calls.append(list(results))

        service = EmptyOutcomeService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="ext1")],
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")
        service.run(request)

        assert service._process_calls == []
        assert service._last_extractor_run_stats == {
            "total": 1,
            "failed": 0,
            "timed_out": 0,
        }

    def test_extraction_outcome_run_finalizes_after_processing(
        self, llm_client, request_context
    ):
        """A resumable run is marked finalized only after result processing."""

        class OutcomeService(ConcreteGenerationService):
            def _create_extractor(self, extractor_config, service_config):
                return MockExtractor(
                    result=ExtractionOutcome.completed([{"name": "x"}], run_id="run_1")
                )

        service = OutcomeService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="ext1")],
        )
        service.storage = MagicMock()
        service.storage.get_agent_run.return_value = SimpleNamespace(
            pending_tool_call_ids=[],
            committed_output={"items": []},
            finalization_attempts=0,
        )

        service.run(MockServiceConfig(user_id="test_user", request_id="test_request"))

        service.storage.update_agent_run_status.assert_called_with(
            "run_1",
            AgentRunStatus.FINALIZED,
            pending_tool_call_ids=[],
        )

    def test_extraction_outcome_finalization_failure_marks_run_retryable(
        self, llm_client, request_context
    ):
        """Finalization failure is tracked on the agent run for later retry."""

        class FailingFinalizeService(ConcreteGenerationService):
            def _create_extractor(self, extractor_config, service_config):
                return MockExtractor(
                    result=ExtractionOutcome.completed([{"name": "x"}], run_id="run_1")
                )

            def _process_results(self, results):
                raise RuntimeError("persist failed")

        service = FailingFinalizeService(
            llm_client,
            request_context,
            extractor_configs=[MockExtractorConfig(extractor_name="ext1")],
        )
        service.storage = MagicMock()
        service.storage.get_agent_run.return_value = SimpleNamespace(
            pending_tool_call_ids=[],
            committed_output={"items": []},
            finalization_attempts=0,
        )

        service.run(MockServiceConfig(user_id="test_user", request_id="test_request"))

        _, status = service.storage.update_agent_run_status.call_args.args[:2]
        kwargs = service.storage.update_agent_run_status.call_args.kwargs
        assert status == AgentRunStatus.FINALIZATION_FAILED
        assert kwargs["last_error"] == "persist failed"
        assert kwargs["increment_finalization_attempts"] is True

    def test_configured_extractor_timeout_fails_generation(
        self, llm_client, request_context, monkeypatch
    ):
        """Test that a timed-out configured extractor fails generation."""
        monkeypatch.setattr(
            "reflexio.server.services.base_generation_service.EXTRACTOR_TIMEOUT_SECONDS",
            0.01,
        )

        class SlowExtractor:
            def run(self):
                time.sleep(0.1)
                return {"name": "slow"}

        class TimeoutService(ConcreteGenerationService):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._process_calls = []

            def _create_extractor(self, extractor_config, service_config):
                if extractor_config.extractor_name == "slow":
                    return SlowExtractor()
                return MockExtractor(result={"name": extractor_config.extractor_name})

            def _process_results(self, results):
                self._process_calls.append(list(results))

        service = TimeoutService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="slow"),
                MockExtractorConfig(extractor_name="fast"),
            ],
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")

        with pytest.raises(ExtractorExecutionError):
            service.run(request)
        assert service._process_calls == []
        assert service._last_extractor_run_stats["total"] == 1
        assert service._last_extractor_run_stats["failed"] == 1
        assert service._last_extractor_run_stats["timed_out"] == 1


class TestPureSlashCommand:
    """Cases for the ``_is_pure_slash_command`` helper."""

    @pytest.mark.parametrize(
        "content",
        [
            "/learn",
            "  /tag ",
            "/claude-smart:tag",
            "/commit",
            "/review",
        ],
    )
    def test_bare_dispatch_is_pure(self, content):
        assert _is_pure_slash_command(content) is True

    @pytest.mark.parametrize(
        "content",
        [
            "/btw this is a side note I want recorded",
            "/claude-smart:tag the previous turn was wrong because X",
            "/commit fix the foo in bar",
            "not a slash command at all",
            "",
            "   ",
        ],
    )
    def test_non_pure_returns_false(self, content):
        assert _is_pure_slash_command(content) is False

    def test_handles_bare_slash_with_no_body(self):
        """A lone ``/`` has no command-name token; treat as non-pure.

        Pins the edge-case behavior after the switch from
        ``startswith("/")`` to the token-regex approach — the old rule
        would have flagged this as a slash command, the new rule does
        not, and we want regressions here to be loud.
        """
        assert _is_pure_slash_command("/") is False
        assert _is_pure_slash_command("  /  ") is False


class TestCheapShouldRunReject:
    """Regression tests for the ``all_slash_commands`` rule after the /btw fix."""

    @staticmethod
    def _batch(*contents: str) -> list[RequestInteractionDataModel]:
        return [
            RequestInteractionDataModel(
                session_id="s",
                request=Request(request_id="r", user_id="u"),
                interactions=[
                    Interaction(user_id="u", request_id="r", role="User", content=c)
                    for c in contents
                ],
            )
        ]

    def test_btw_with_note_is_not_rejected(self):
        batch = self._batch("/btw some side note with real content from the user")
        assert _cheap_should_run_reject(batch) is None

    def test_namespaced_tag_with_note_is_not_rejected(self):
        batch = self._batch(
            "/claude-smart:tag the last turn misunderstood what I was asking"
        )
        assert _cheap_should_run_reject(batch) is None

    def test_bare_slash_command_still_rejected(self):
        # Bare `/learn` also trips the length rule first; either rejection
        # reason is fine — the contract is that it must be dropped.
        batch = self._batch("/learn")
        assert _cheap_should_run_reject(batch) is not None

    def test_mixed_bare_slash_commands_still_rejected(self):
        batch = self._batch("/learn", "/claude-smart:tag", "/review")
        assert _cheap_should_run_reject(batch) is not None

    def test_mixed_bare_and_content_bearing_passes(self):
        batch = self._batch("/learn", "/btw actually keep this whole batch around")
        assert _cheap_should_run_reject(batch) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
