"""
Unit tests for extractor_interaction_utils module.

Tests the shared utility functions used by all extractors for:
- Window/stride parameter extraction
- Source filtering
- Stride checking
"""

from dataclasses import dataclass

from reflexio.models.api_schema.internal_schema import (
    RequestInteractionDataModel,
)
from reflexio.models.api_schema.service_schemas import (
    Interaction,
    Request,
)
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
    iter_sliding_windows,
    should_extractor_run_by_batch_interval,
)

# ===============================
# Test Data Classes
# ===============================


@dataclass
class MockExtractorConfig:
    """Mock extractor config for testing."""

    extractor_name: str
    batch_size_override: int | None = None
    batch_interval_override: int | None = None
    request_sources_enabled: list[str] | None = None


@dataclass
class MockPlaybookConfig:
    """Mock playbook config with playbook_name."""

    playbook_name: str
    batch_size: int | None = None
    batch_interval: int | None = None
    request_sources_enabled: list[str] | None = None


# ===============================
# Test: get_extractor_window_params
# ===============================


class TestGetExtractorWindowParams:
    """Tests for window/stride parameter extraction."""

    def test_extractor_override_takes_precedence(self):
        """Test that extractor-level overrides take precedence over globals."""
        config = MockExtractorConfig(
            extractor_name="test",
            batch_size_override=50,
            batch_interval_override=10,
        )

        window, stride = get_extractor_window_params(
            config,
            global_batch_size=100,
            global_batch_interval=20,
        )

        assert window == 50
        assert stride == 10

    def test_global_fallback_when_extractor_not_set(self):
        """Test fallback to global values when extractor doesn't override."""
        config = MockExtractorConfig(extractor_name="test")

        window, stride = get_extractor_window_params(
            config,
            global_batch_size=100,
            global_batch_interval=20,
        )

        assert window == 100
        assert stride == 20

    def test_partial_override(self):
        """Test partial override - only window size set on extractor."""
        config = MockExtractorConfig(
            extractor_name="test",
            batch_size_override=50,
        )

        window, stride = get_extractor_window_params(
            config,
            global_batch_size=100,
            global_batch_interval=20,
        )

        assert window == 50
        assert stride == 20

    def test_defaults_when_nothing_set(self):
        """Test defaults (window=10, stride=5) are returned when no values are set anywhere."""
        config = MockExtractorConfig(extractor_name="test")

        window, stride = get_extractor_window_params(
            config,
            global_batch_size=None,
            global_batch_interval=None,
        )

        assert window == 10
        assert stride == 5

    def test_zero_values_respected(self):
        """Test that zero values ARE respected (0 is a valid override value)."""
        config = MockExtractorConfig(
            extractor_name="test",
            batch_size_override=0,
            batch_interval_override=0,
        )

        window, stride = get_extractor_window_params(
            config,
            global_batch_size=100,
            global_batch_interval=20,
        )

        # 0 should be treated as a valid override value (not None)
        assert window == 0
        assert stride == 0


# ===============================
# Test: get_effective_source_filter
# ===============================


class TestGetEffectiveSourceFilter:
    """Tests for source filtering logic."""

    def test_none_sources_enabled_returns_no_filter(self):
        """Test that request_sources_enabled=None means get ALL sources."""
        config = MockExtractorConfig(
            extractor_name="test",
            request_sources_enabled=None,
        )

        should_skip, effective_source = get_effective_source_filter(
            config,
            triggering_source="api",
        )

        assert should_skip is False
        assert effective_source is None  # No filtering

    def test_source_in_enabled_list_returns_that_source(self):
        """Test filtering when triggering source is in enabled list."""
        config = MockExtractorConfig(
            extractor_name="test",
            request_sources_enabled=["api", "web"],
        )

        should_skip, effective_source = get_effective_source_filter(
            config,
            triggering_source="api",
        )

        assert should_skip is False
        assert effective_source == ["api"]

    def test_source_not_in_enabled_list_returns_skip(self):
        """Test safety skip when source is not in enabled list."""
        config = MockExtractorConfig(
            extractor_name="test",
            request_sources_enabled=["mobile", "desktop"],
        )

        should_skip, effective_source = get_effective_source_filter(
            config,
            triggering_source="api",
        )

        assert should_skip is True
        assert effective_source is None

    def test_none_triggering_source_with_enabled_list(self):
        """Test when triggering source is None but enabled list is set (rerun flow)."""
        config = MockExtractorConfig(
            extractor_name="test",
            request_sources_enabled=["api", "web"],
        )

        should_skip, effective_source = get_effective_source_filter(
            config,
            triggering_source=None,
        )

        # When triggering_source is None (rerun flow) and sources_enabled is set,
        # the function returns all enabled sources so caller can filter by them
        assert should_skip is False
        assert effective_source == ["api", "web"]

    def test_empty_enabled_list(self):
        """Test with empty enabled list - treated same as None (all sources enabled)."""
        config = MockExtractorConfig(
            extractor_name="test",
            request_sources_enabled=[],
        )

        should_skip, effective_source = get_effective_source_filter(
            config,
            triggering_source="api",
        )

        # Empty list means all sources enabled (same as None)
        assert should_skip is False
        assert effective_source is None


# ===============================
# Test: should_extractor_run_by_batch_interval
# ===============================


class TestShouldExtractorRunByBatchInterval:
    """Tests for batch_interval-based execution checking."""

    def test_run_when_count_equals_batch_interval(self):
        """Test that extractor runs when count equals batch_interval."""
        result = should_extractor_run_by_batch_interval(
            new_interaction_count=10, batch_interval_size=10
        )
        assert result is True

    def test_run_when_count_exceeds_batch_interval(self):
        """Test that extractor runs when count exceeds batch_interval."""
        result = should_extractor_run_by_batch_interval(
            new_interaction_count=15, batch_interval_size=10
        )
        assert result is True

    def test_skip_when_count_below_batch_interval(self):
        """Test that extractor skips when count is below batch_interval."""
        result = should_extractor_run_by_batch_interval(
            new_interaction_count=5, batch_interval_size=10
        )
        assert result is False

    def test_run_when_batch_interval_is_none(self):
        """Test that extractor always runs when batch_interval is None."""
        result = should_extractor_run_by_batch_interval(
            new_interaction_count=1, batch_interval_size=None
        )
        assert result is True

    def test_run_when_batch_interval_is_zero(self):
        """Test that extractor always runs when batch_interval is 0."""
        result = should_extractor_run_by_batch_interval(
            new_interaction_count=1, batch_interval_size=0
        )
        assert result is True

    def test_skip_when_zero_interactions(self):
        """Test skip when there are no new interactions."""
        result = should_extractor_run_by_batch_interval(
            new_interaction_count=0, batch_interval_size=10
        )
        assert result is False

    def test_skip_with_zero_interactions_even_without_batch_interval(self):
        """Test that zero interactions skips even without batch_interval configured."""
        result = should_extractor_run_by_batch_interval(
            new_interaction_count=0, batch_interval_size=None
        )
        # No interactions means nothing to process - always skip
        assert result is False


# ===============================
# Test: iter_sliding_windows
# ===============================


def _create_mock_request_interaction_model(
    num_interactions: int, source: str = "api", session_id: str = "test_group"
) -> RequestInteractionDataModel:
    """Helper to create a RequestInteractionDataModel with specified number of interactions."""
    request = Request(
        user_id="user1",
        agent_version="v1",
        request_id=f"req_{num_interactions}",
        source=source,
    )
    interactions = [
        Interaction(
            interaction_id=i,
            user_id="user1",
            content=f"message_{i}",
            request_id=request.request_id,
            created_at=1000 + i,
        )
        for i in range(num_interactions)
    ]
    return RequestInteractionDataModel(
        session_id=session_id, request=request, interactions=interactions
    )


class TestIterSlidingWindows:
    """Tests for sliding window iteration over RequestInteractionDataModel."""

    def test_empty_list_yields_nothing(self):
        """Test that empty input yields nothing."""
        windows = list(iter_sliding_windows([], batch_size=10, batch_interval_size=5))
        assert windows == []

    def test_single_model_fits_in_window(self):
        """Test single model that fits in window yields one window."""
        models = [_create_mock_request_interaction_model(5)]
        windows = list(iter_sliding_windows(models, batch_size=10, batch_interval_size=5))

        assert len(windows) == 1
        assert windows[0][0] == 0  # window index
        assert windows[0][1] == models  # contains original model

    def test_multiple_models_fit_in_one_window(self):
        """Test multiple models that together fit in one window."""
        models = [
            _create_mock_request_interaction_model(3),
            _create_mock_request_interaction_model(4),
        ]
        windows = list(iter_sliding_windows(models, batch_size=10, batch_interval_size=5))

        assert len(windows) == 1
        assert windows[0][0] == 0
        assert len(windows[0][1]) == 2  # both models included

    def test_basic_sliding_window(self):
        """Test basic sliding window with multiple windows needed."""
        # 3 models with 10 interactions each = 30 total
        # Model boundaries: [0-9], [10-19], [20-29]
        # batch_size=15, stride=10 should yield 3 windows:
        # Window 0: covers [0-14] → models[0] and models[1] (overlaps both)
        # Window 1: covers [10-24] → models[1] and models[2] (model[0] ends at 10, exclusive)
        # Window 2: covers [20-34] → models[2] only (model[1] ends at 20, exclusive)
        models = [
            _create_mock_request_interaction_model(10),
            _create_mock_request_interaction_model(10),
            _create_mock_request_interaction_model(10),
        ]
        windows = list(iter_sliding_windows(models, batch_size=15, batch_interval_size=10))

        assert len(windows) == 3
        # Window 0: covers [0-14], includes models[0] and models[1]
        assert windows[0][0] == 0
        assert len(windows[0][1]) == 2
        # Window 1: covers [10-24], includes models[1] and models[2]
        assert windows[1][0] == 1
        assert len(windows[1][1]) == 2
        # Window 2: covers [20-34], includes only models[2]
        assert windows[2][0] == 2
        assert len(windows[2][1]) == 1

    def test_non_overlapping_windows(self):
        """Test non-overlapping windows when stride >= window_size."""
        models = [
            _create_mock_request_interaction_model(10),
            _create_mock_request_interaction_model(10),
            _create_mock_request_interaction_model(10),
        ]
        windows = list(iter_sliding_windows(models, batch_size=10, batch_interval_size=10))

        assert len(windows) == 3
        # Each window should contain exactly one model
        for i, (idx, window_models) in enumerate(windows):
            assert idx == i
            assert len(window_models) == 1

    def test_stride_larger_than_window(self):
        """Test when stride is larger than window (gaps between windows)."""
        models = [
            _create_mock_request_interaction_model(10),
            _create_mock_request_interaction_model(10),
            _create_mock_request_interaction_model(10),
        ]
        # batch_size=5, stride=15 means windows at positions 0-4, 15-19
        windows = list(iter_sliding_windows(models, batch_size=5, batch_interval_size=15))

        assert len(windows) == 2
        # Window 0: covers 0-4, only models[0]
        assert windows[0][0] == 0
        assert len(windows[0][1]) == 1
        # Window 1: covers 15-19, only models[1]
        assert windows[1][0] == 1
        assert len(windows[1][1]) == 1

    def test_invalid_window_size_zero(self):
        """Test that batch_size=0 yields single window with all data."""
        models = [_create_mock_request_interaction_model(10)]
        windows = list(iter_sliding_windows(models, batch_size=0, batch_interval_size=5))

        assert len(windows) == 1
        assert windows[0][1] == models

    def test_invalid_window_size_negative(self):
        """Test that negative window_size yields single window with all data."""
        models = [_create_mock_request_interaction_model(10)]
        windows = list(iter_sliding_windows(models, batch_size=-5, batch_interval_size=5))

        assert len(windows) == 1
        assert windows[0][1] == models

    def test_stride_zero_defaults_to_window_size(self):
        """Test that stride=0 defaults to window_size."""
        models = [
            _create_mock_request_interaction_model(10),
            _create_mock_request_interaction_model(10),
        ]
        # stride=0 should default to batch_size=10, yielding 2 non-overlapping windows
        windows = list(iter_sliding_windows(models, batch_size=10, batch_interval_size=0))

        assert len(windows) == 2

    def test_stride_none_defaults_to_window_size(self):
        """Test that stride=None defaults to window_size."""
        models = [
            _create_mock_request_interaction_model(10),
            _create_mock_request_interaction_model(10),
        ]
        # stride=None should default to batch_size=10
        windows = list(iter_sliding_windows(models, batch_size=10, batch_interval_size=None))

        assert len(windows) == 2

    def test_models_with_varying_sizes(self):
        """Test with models having different numbers of interactions."""
        # Model boundaries: [0-4], [5-24], [25-29]
        models = [
            _create_mock_request_interaction_model(5),  # interactions 0-4
            _create_mock_request_interaction_model(20),  # interactions 5-24
            _create_mock_request_interaction_model(5),  # interactions 25-29
        ]
        # Total: 30 interactions
        # batch_size=15, stride=10
        windows = list(iter_sliding_windows(models, batch_size=15, batch_interval_size=10))

        assert len(windows) == 3
        # Window 0: covers [0-14], models[0] (0-4) and models[1] (5-24) overlap
        assert len(windows[0][1]) == 2
        # Window 1: covers [10-24], only models[1] (5-24) overlaps
        # (models[0] ends at 5, exclusive; models[2] starts at 25)
        assert len(windows[1][1]) == 1
        # Window 2: covers [20-34], models[1] (5-24) and models[2] (25-29) overlap
        assert len(windows[2][1]) == 2

    def test_preserves_model_order(self):
        """Test that models maintain their order within windows."""
        models = [
            _create_mock_request_interaction_model(5),
            _create_mock_request_interaction_model(5),
            _create_mock_request_interaction_model(5),
        ]
        windows = list(iter_sliding_windows(models, batch_size=10, batch_interval_size=5))

        # First window should have models[0] and models[1] in order
        assert windows[0][1][0] is models[0]
        assert windows[0][1][1] is models[1]

    def test_model_with_zero_interactions_included(self):
        """Test that models with zero interactions are still considered."""
        models = [
            _create_mock_request_interaction_model(10),
            _create_mock_request_interaction_model(0),  # empty model
            _create_mock_request_interaction_model(10),
        ]
        # Total: 20 interactions, empty model at position 10
        windows = list(iter_sliding_windows(models, batch_size=15, batch_interval_size=10))

        assert len(windows) == 2

    def test_all_empty_models_yields_nothing(self):
        """Test that all empty models yields nothing."""
        models = [
            _create_mock_request_interaction_model(0),
            _create_mock_request_interaction_model(0),
        ]
        windows = list(iter_sliding_windows(models, batch_size=10, batch_interval_size=5))

        assert windows == []

    def test_window_indices_are_sequential(self):
        """Test that window indices are sequential starting from 0."""
        models = [_create_mock_request_interaction_model(10) for _ in range(5)]
        windows = list(iter_sliding_windows(models, batch_size=10, batch_interval_size=10))

        indices = [w[0] for w in windows]
        assert indices == list(range(5))
