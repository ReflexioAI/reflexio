"""
Unit tests for BaseGenerationService class.

Tests the abstract base class by creating a concrete implementation for testing.
"""

import tempfile
import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

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
)

# ===============================
# Test Data Classes
# ===============================


@dataclass
class MockExtractorConfig:
    """Mock extractor config for testing."""

    extractor_name: str
    request_sources_enabled: list[str] | None = None
    manual_trigger: bool = False
    batch_size_override: int | None = None
    batch_interval_override: int | None = None


@dataclass
class MockServiceConfig:
    """Mock service config for testing."""

    user_id: str = "test_user"
    request_id: str = "test_request"
    request_interaction_data_models: list | None = None
    source: str | None = None
    allow_manual_trigger: bool = False
    extractor_names: list[str] | None = None
    is_incremental: bool = False
    previously_extracted: list = field(default_factory=list)
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

    def __init__(self, llm_client, request_context, extractor_configs=None):
        super().__init__(llm_client, request_context)
        self._extractor_configs = extractor_configs or []
        self._processed_results = []
        # For upgrade/downgrade testing
        self._items_by_status = {}
        self._deleted_count = 0
        self._updated_count = 0

    def _load_extractor_configs(self):
        return self._extractor_configs

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

    def _update_config_for_incremental(self, previously_extracted):
        self.service_config.is_incremental = True
        self.service_config.previously_extracted = list(previously_extracted)

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
# Test: _filter_extractor_configs_by_service_config
# ===============================


class TestFilterExtractorConfigsByServiceConfig:
    """Tests for the _filter_extractor_configs_by_service_config method."""

    def test_no_filtering_without_source_attribute(self, base_service):
        """Test that configs are not filtered if service_config has no source attribute."""
        configs = [
            MockExtractorConfig(extractor_name="extractor1"),
            MockExtractorConfig(extractor_name="extractor2"),
        ]

        # Create service config without source attribute
        class NoSourceConfig:
            pass

        service_config = NoSourceConfig()
        result = base_service._filter_extractor_configs_by_service_config(
            configs, service_config
        )
        assert len(result) == 2

    def test_filter_by_source_enabled(self, base_service):
        """Test filtering extractors by request_sources_enabled."""
        configs = [
            MockExtractorConfig(
                extractor_name="extractor1", request_sources_enabled=["api", "web"]
            ),
            MockExtractorConfig(
                extractor_name="extractor2", request_sources_enabled=["mobile"]
            ),
            MockExtractorConfig(extractor_name="extractor3"),  # No source restriction
        ]

        service_config = MockServiceConfig(source="api")
        result = base_service._filter_extractor_configs_by_service_config(
            configs, service_config
        )

        # extractor1 (api in enabled list) and extractor3 (no restriction) should pass
        assert len(result) == 2
        extractor_names = [c.extractor_name for c in result]
        assert "extractor1" in extractor_names
        assert "extractor3" in extractor_names
        assert "extractor2" not in extractor_names

    def test_filter_by_manual_trigger(self, base_service):
        """Test filtering extractors by manual_trigger flag."""
        configs = [
            MockExtractorConfig(extractor_name="extractor1", manual_trigger=True),
            MockExtractorConfig(extractor_name="extractor2", manual_trigger=False),
            MockExtractorConfig(extractor_name="extractor3"),  # Default False
        ]

        # allow_manual_trigger=False - manual_trigger=True extractors should be skipped
        service_config = MockServiceConfig(allow_manual_trigger=False)
        result = base_service._filter_extractor_configs_by_service_config(
            configs, service_config
        )

        assert len(result) == 2
        extractor_names = [c.extractor_name for c in result]
        assert "extractor2" in extractor_names
        assert "extractor3" in extractor_names
        assert "extractor1" not in extractor_names

    def test_manual_trigger_allowed_when_allow_manual_trigger_true(self, base_service):
        """Test that manual_trigger extractors are allowed when allow_manual_trigger=True."""
        configs = [
            MockExtractorConfig(extractor_name="extractor1", manual_trigger=True),
            MockExtractorConfig(extractor_name="extractor2", manual_trigger=False),
        ]

        service_config = MockServiceConfig(allow_manual_trigger=True)
        result = base_service._filter_extractor_configs_by_service_config(
            configs, service_config
        )

        assert len(result) == 2
        extractor_names = [c.extractor_name for c in result]
        assert "extractor1" in extractor_names
        assert "extractor2" in extractor_names

    def test_filter_by_extractor_names(self, base_service):
        """Test filtering extractors by extractor_names list in service_config."""
        configs = [
            MockExtractorConfig(extractor_name="extractor1"),
            MockExtractorConfig(extractor_name="extractor2"),
            MockExtractorConfig(extractor_name="extractor3"),
        ]

        service_config = MockServiceConfig(extractor_names=["extractor1", "extractor3"])
        result = base_service._filter_extractor_configs_by_service_config(
            configs, service_config
        )

        assert len(result) == 2
        extractor_names = [c.extractor_name for c in result]
        assert "extractor1" in extractor_names
        assert "extractor3" in extractor_names
        assert "extractor2" not in extractor_names

    def test_combined_filtering(self, base_service):
        """Test that all filter conditions are applied together."""
        configs = [
            MockExtractorConfig(
                extractor_name="extractor1",
                request_sources_enabled=["api"],
                manual_trigger=False,
            ),
            MockExtractorConfig(
                extractor_name="extractor2",
                request_sources_enabled=["mobile"],
                manual_trigger=False,
            ),
            MockExtractorConfig(
                extractor_name="extractor3",
                request_sources_enabled=["api"],
                manual_trigger=True,
            ),
        ]

        # Source=api, allow_manual_trigger=False, filter by name
        service_config = MockServiceConfig(
            source="api",
            allow_manual_trigger=False,
            extractor_names=["extractor1", "extractor3"],
        )
        result = base_service._filter_extractor_configs_by_service_config(
            configs, service_config
        )

        # Only extractor1 passes all filters:
        # - extractor2: wrong source
        # - extractor3: manual_trigger=True but allow_manual_trigger=False
        assert len(result) == 1
        assert result[0].extractor_name == "extractor1"

    def test_empty_configs_list(self, base_service):
        """Test filtering with empty configs list."""
        configs = []
        service_config = MockServiceConfig(source="api")
        result = base_service._filter_extractor_configs_by_service_config(
            configs, service_config
        )
        assert len(result) == 0

    def test_none_source_in_service_config(self, base_service):
        """Test filtering when source is None in service_config."""
        configs = [
            MockExtractorConfig(
                extractor_name="extractor1", request_sources_enabled=["api"]
            ),
            MockExtractorConfig(extractor_name="extractor2"),
        ]

        service_config = MockServiceConfig(source=None)
        result = base_service._filter_extractor_configs_by_service_config(
            configs, service_config
        )

        # Both should pass since source is None (no filtering by source)
        assert len(result) == 2


# ===============================
# Test: _filter_configs_by_batch_interval
# ===============================


class BatchIntervalEnabledService(ConcreteGenerationService):
    """Concrete service with batch_interval pre-filtering enabled."""

    def _get_extractor_state_service_name(self):
        return "test_extractor"


class TestFilterConfigsByBatchInterval:
    """Tests for the _filter_configs_by_batch_interval method."""

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
                role="user",
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

    def test_returns_all_configs_when_no_service_name(
        self, llm_client, request_context
    ):
        """Verify _filter_configs_by_batch_interval returns all configs unchanged when
        _get_extractor_state_service_name() returns None (e.g., AgentSuccessEvaluationService).
        """
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="ext1"),
                MockExtractorConfig(extractor_name="ext2"),
            ],
        )
        service.service_config = MockServiceConfig()

        configs = [
            MockExtractorConfig(extractor_name="ext1"),
            MockExtractorConfig(extractor_name="ext2"),
        ]
        result = service._filter_configs_by_batch_interval(configs)
        assert len(result) == 2
        assert result is configs  # Same list object returned, no filtering

    def test_returns_all_configs_when_auto_run_false(self, llm_client, request_context):
        """Verify all configs pass when auto_run=False (rerun/manual mode)."""
        service = BatchIntervalEnabledService(
            llm_client,
            request_context,
            extractor_configs=[],
        )
        service.service_config = MockServiceConfig(auto_run=False)

        configs = [
            MockExtractorConfig(extractor_name="ext1"),
            MockExtractorConfig(extractor_name="ext2"),
        ]
        result = service._filter_configs_by_batch_interval(configs)
        assert len(result) == 2
        assert result is configs  # Same list object returned, no filtering

    def test_returns_all_configs_when_force_extraction_true(
        self, llm_client, request_context
    ):
        """Verify all configs pass when force_extraction=True (agent-curated publish)."""
        service = BatchIntervalEnabledService(
            llm_client,
            request_context,
            extractor_configs=[],
        )
        service.service_config = MockServiceConfig(force_extraction=True)

        configs = [
            MockExtractorConfig(extractor_name="ext1"),
            MockExtractorConfig(extractor_name="ext2"),
        ]
        result = service._filter_configs_by_batch_interval(configs)
        assert len(result) == 2
        assert result is configs  # Same list object returned, no filtering

    def _setup_batch_interval_service(
        self, llm_client, request_context, n_new_interactions
    ):
        """Create a BatchIntervalEnabledService with mocked storage for batch_interval tests.

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

        service = BatchIntervalEnabledService(
            llm_client,
            request_context,
            extractor_configs=[],
        )
        return service  # noqa: RET504

    def test_filters_configs_when_batch_interval_not_met(
        self, llm_client, request_context
    ):
        """Verify configs are dropped when new interaction count < batch_interval."""
        service = self._setup_batch_interval_service(llm_client, request_context, 2)
        service.service_config = MockServiceConfig(auto_run=True, source="api")

        configs = [
            MockExtractorConfig(extractor_name="ext1"),
        ]
        result = service._filter_configs_by_batch_interval(configs)
        assert len(result) == 0

    def test_passes_configs_when_batch_interval_met(self, llm_client, request_context):
        """Verify configs pass through when new interaction count >= batch_interval."""
        service = self._setup_batch_interval_service(llm_client, request_context, 6)
        service.service_config = MockServiceConfig(auto_run=True, source="api")

        configs = [
            MockExtractorConfig(extractor_name="ext1"),
        ]
        result = service._filter_configs_by_batch_interval(configs)
        assert len(result) == 1
        assert result[0].extractor_name == "ext1"

    def test_handles_source_skip(self, llm_client, request_context):
        """Verify configs that fail source filtering in batch_interval check are skipped."""
        service = self._setup_batch_interval_service(llm_client, request_context, 10)
        service.service_config = MockServiceConfig(auto_run=True, source="api")

        # ext1 has sources_enabled=["mobile"] but triggering source is "api" -> should be skipped
        # ext2 has no source restriction -> should pass batch_interval check
        configs = [
            MockExtractorConfig(
                extractor_name="ext1", request_sources_enabled=["mobile"]
            ),
            MockExtractorConfig(extractor_name="ext2"),
        ]
        result = service._filter_configs_by_batch_interval(configs)
        # ext1 skipped by source filter, ext2 passes batch_interval
        assert len(result) == 1
        assert result[0].extractor_name == "ext2"

    def test_uses_per_extractor_batch_interval_override(
        self, llm_client, request_context
    ):
        """Verify per-extractor batch_interval override is respected."""
        service = self._setup_batch_interval_service(llm_client, request_context, 3)
        service.service_config = MockServiceConfig(auto_run=True, source="api")

        # 3 new interactions: ext1 (batch_interval=2 -> passes), ext2 (default batch_interval=5 -> fails)
        configs = [
            MockExtractorConfig(extractor_name="ext1", batch_interval_override=2),
            MockExtractorConfig(extractor_name="ext2"),
        ]
        result = service._filter_configs_by_batch_interval(configs)
        assert len(result) == 1
        assert result[0].extractor_name == "ext1"


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

        configs = [MockExtractorConfig(extractor_name="test")]
        result = service._should_run_before_extraction(configs)

        assert result is True

    def test_default_skip_should_run_check_does_not_bypass(self):
        """When skip_should_run_check=False (default), the flag guard does not fire."""
        from reflexio.models.config_schema import Config

        config = Config(storage_config={"type": "sqlite"})
        assert config.skip_should_run_check is False


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
            extractor_configs=[
                MockExtractorConfig(extractor_name="extractor1"),
                MockExtractorConfig(extractor_name="extractor2"),
            ],
        )

        request = MockServiceConfig(
            user_id="test_user",
            request_id="test_request",
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)

        # _process_results called once with all results after all extractors complete
        assert len(service._processed_results) == 2

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
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
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

    def test_run_without_extractor_configs(self, llm_client, request_context):
        """Test run() when no extractor configs are available."""
        service = ConcreteGenerationService(
            llm_client, request_context, extractor_configs=[]
        )

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
            extractor_configs=[MockExtractorConfig(extractor_name="extractor1")],
        )

        request = MockServiceConfig(
            user_id="test_user",
            request_id="test_request",
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)

        assert service.service_config is not None
        assert service.service_config.user_id == "test_user"

    def test_run_filters_extractor_configs(self, llm_client, request_context):
        """Test that run() applies config filtering before creating extractors."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(
                    extractor_name="extractor1", request_sources_enabled=["api"]
                ),
                MockExtractorConfig(
                    extractor_name="extractor2", request_sources_enabled=["mobile"]
                ),
            ],
        )

        request = MockServiceConfig(
            source="api",
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)

        # Only extractor1 should run since source is "api"
        assert len(service._processed_results) == 1
        assert service._processed_results[0]["extractor_name"] == "extractor1"

    def test_run_raises_when_all_extractors_fail(self, llm_client, request_context):
        """Test that run() raises when all extractors fail."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="extractor1"),
                MockExtractorConfig(extractor_name="extractor2"),
            ],
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

    def test_run_partial_success_when_some_extractors_fail(
        self, llm_client, request_context
    ):
        """Test that run() succeeds when at least one extractor returns a result."""
        service = ConcreteGenerationService(
            llm_client,
            request_context,
            extractor_configs=[
                MockExtractorConfig(extractor_name="extractor1"),
                MockExtractorConfig(extractor_name="extractor2"),
            ],
        )
        service._create_extractor = MagicMock(
            side_effect=lambda extractor_config, service_config: MockExtractor(  # noqa: ARG005
                result={"extractor_name": extractor_config.extractor_name},
                should_raise=extractor_config.extractor_name == "extractor1",
            )
        )

        request = MockServiceConfig(
            user_id="test_user",
            request_id="test_request",
            request_interaction_data_models=[MagicMock()],
        )

        service.run(request)
        assert len(service._processed_results) == 1
        assert service._processed_results[0]["extractor_name"] == "extractor2"


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
    """Tests for the sequential extractor execution in _run_generation."""

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

    def test_sequential_multiple_extractors_all_succeed(
        self, llm_client, request_context
    ):
        """Test that all extractors run sequentially and each result is saved."""
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

        # Extractors ran sequentially
        assert call_order == ["ext1", "ext2", "ext3"]
        # _process_results called once with all 3 results
        assert len(service._process_calls) == 1
        assert len(service._process_calls[0]) == 3

    def test_sequential_partial_failure(self, llm_client, request_context):
        """Test that failure in one extractor doesn't stop others."""

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
                MockExtractorConfig(extractor_name="ext1"),
                MockExtractorConfig(extractor_name="failing"),
                MockExtractorConfig(extractor_name="ext3"),
            ],
        )

        request = MockServiceConfig(user_id="test_user", request_id="test_request")
        service.run(request)

        # _process_results called once with the 2 successful results
        assert len(service._process_calls) == 1
        assert len(service._process_calls[0]) == 2

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

    def test_sequential_refetches_config_for_subsequent_extractors(
        self, llm_client, request_context
    ):
        """Test that service_config is refetched after the first extractor."""
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

        # Config loaded once initially, then re-loaded before ext2
        assert len(load_config_calls) == 2

    def test_sequential_sets_incremental_flag(self, llm_client, request_context):
        """Test that is_incremental is set on service_config for subsequent extractors."""
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

        # First extractor: not incremental. Subsequent: incremental
        assert observed_incremental == [False, True, True]

    def test_sequential_passes_previously_extracted(self, llm_client, request_context):
        """Test that previously_extracted accumulates results across extractors."""
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

        # First: empty, second: 1 result, third: 2 results
        assert len(observed_previously[0]) == 0
        assert len(observed_previously[1]) == 1
        assert len(observed_previously[2]) == 2

    def test_sequential_none_results_not_accumulated(self, llm_client, request_context):
        """Test that None results from extractors are not added to previously_extracted."""
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

        # ext1: empty, ext2: 1 (from ext1), ext3: 1 (ext2 returned None, not accumulated)
        assert len(observed_previously[0]) == 0
        assert len(observed_previously[1]) == 1
        assert len(observed_previously[2]) == 1

    def test_sequential_timeout_does_not_block_following_extractors(
        self, llm_client, request_context, monkeypatch
    ):
        """Test that timed-out extractors are skipped and later extractors still run."""
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
        service.run(request)

        assert len(service._process_calls) == 1
        assert service._process_calls[0] == [{"name": "fast"}]
        assert service._last_extractor_run_stats["failed"] == 1
        assert service._last_extractor_run_stats["timed_out"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
