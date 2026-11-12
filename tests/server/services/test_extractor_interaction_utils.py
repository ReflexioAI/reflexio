"""
Unit tests for extractor_interaction_utils module.

Tests the shared utility functions used by all extractors for:
- Window/stride parameter extraction
- Source filtering
- Stride checking
"""

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from reflexio.config_env import StoredConfigurationError
from reflexio.models.config_schema import Config, validate_stored_config
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
    should_extractor_run_by_stride,
)

# ===============================
# Test Data Classes
# ===============================


@dataclass
class MockExtractorConfig:
    """Mock extractor config for testing."""

    extractor_name: str
    window_size_override: int | None = None
    stride_size_override: int | None = None
    request_sources_enabled: list[str] | None = None


@dataclass
class MockPlaybookConfig:
    """Mock playbook config with playbook_name."""

    playbook_name: str
    window_size: int | None = None
    stride_size: int | None = None
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
            window_size_override=50,
            stride_size_override=10,
        )

        window, stride = get_extractor_window_params(
            config,
            global_window_size=100,
            global_stride_size=20,
        )

        assert window == 50
        assert stride == 10

    def test_global_fallback_when_extractor_not_set(self):
        """Test fallback to global values when extractor doesn't override."""
        config = MockExtractorConfig(extractor_name="test")

        window, stride = get_extractor_window_params(
            config,
            global_window_size=100,
            global_stride_size=20,
        )

        assert window == 100
        assert stride == 20

    def test_partial_override(self):
        """Test partial override - only window size set on extractor."""
        config = MockExtractorConfig(
            extractor_name="test",
            window_size_override=50,
        )

        window, stride = get_extractor_window_params(
            config,
            global_window_size=100,
            global_stride_size=20,
        )

        assert window == 50
        assert stride == 20

    def test_defaults_when_nothing_set(self):
        """Test defaults (window=10, stride=8) are returned when no values are set anywhere."""
        config = MockExtractorConfig(extractor_name="test")

        window, stride = get_extractor_window_params(
            config,
            global_window_size=None,
            global_stride_size=None,
        )

        assert window == 10
        assert stride == 8

    def test_zero_values_respected(self):
        """Test that zero values ARE respected (0 is a valid override value)."""
        config = MockExtractorConfig(
            extractor_name="test",
            window_size_override=0,
            stride_size_override=0,
        )

        window, stride = get_extractor_window_params(
            config,
            global_window_size=100,
            global_stride_size=20,
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
# Test: should_extractor_run_by_stride
# ===============================


class TestShouldExtractorRunByStride:
    """Tests for stride_size-based execution checking."""

    def test_run_when_count_equals_stride_size(self):
        """Test that extractor runs when count equals stride_size."""
        result = should_extractor_run_by_stride(
            new_interaction_count=10, stride_size=10
        )
        assert result is True

    def test_run_when_count_exceeds_stride_size(self):
        """Test that extractor runs when count exceeds stride_size."""
        result = should_extractor_run_by_stride(
            new_interaction_count=15, stride_size=10
        )
        assert result is True

    def test_skip_when_count_below_stride_size(self):
        """Test that extractor skips when count is below stride_size."""
        result = should_extractor_run_by_stride(new_interaction_count=5, stride_size=10)
        assert result is False

    def test_run_when_stride_size_is_none(self):
        """Test that extractor always runs when stride_size is None."""
        result = should_extractor_run_by_stride(
            new_interaction_count=1, stride_size=None
        )
        assert result is True

    def test_run_when_stride_size_is_zero(self):
        """Test that extractor always runs when stride_size is 0."""
        result = should_extractor_run_by_stride(new_interaction_count=1, stride_size=0)
        assert result is True

    def test_skip_when_zero_interactions(self):
        """Test skip when there are no new interactions."""
        result = should_extractor_run_by_stride(new_interaction_count=0, stride_size=10)
        assert result is False

    def test_skip_with_zero_interactions_even_without_stride_size(self):
        """Test that zero interactions skips even without stride_size configured."""
        result = should_extractor_run_by_stride(
            new_interaction_count=0, stride_size=None
        )
        # No interactions means nothing to process - always skip
        assert result is False


class TestEffectiveWindowAdmission:
    """Automatic configuration cannot promise an impossible extraction stride."""

    @pytest.mark.parametrize(
        "kind", ["profile_extractor_config", "user_playbook_extractor_config"]
    )
    @pytest.mark.parametrize(
        "override",
        [
            {"window_size_override": 2},
            {"stride_size_override": 20},
            {"window_size_override": 3, "stride_size_override": 4},
        ],
    )
    def test_automatic_overrides_are_validated_after_fallback(self, kind, override):
        with pytest.raises(ValidationError, match=kind):
            Config.model_validate(
                {
                    "storage_config": None,
                    kind: {
                        "extraction_definition_prompt": "Extract direct evidence",
                        **override,
                    },
                }
            )

    @pytest.mark.parametrize(
        "kind", ["profile_extractor_config", "user_playbook_extractor_config"]
    )
    def test_stored_automatic_conflict_is_sanitized(self, kind):
        with pytest.raises(StoredConfigurationError, match="configuration") as error:
            validate_stored_config(
                {
                    "storage_config": None,
                    kind: {
                        "extraction_definition_prompt": "Extract direct evidence",
                        "window_size_override": 2,
                    },
                    "agent_context_prompt": "must-not-appear-in-error",
                }
            )
        assert "must-not-appear-in-error" not in str(error.value)

    def test_manual_profile_uses_width_without_an_automatic_stride_gate(self):
        config = Config.model_validate(
            {
                "storage_config": None,
                "profile_extractor_config": {
                    "extraction_definition_prompt": "Extract direct evidence",
                    "manual_trigger": True,
                    "window_size_override": 2,
                },
            }
        )
        assert get_extractor_window_params(
            config.profile_extractor_config, config.window_size, config.stride_size
        ) == (2, 8)

    def test_independent_valid_overrides_and_disabled_extractors(self):
        config = Config.model_validate(
            {
                "storage_config": None,
                "profile_extractor_config": None,
                "user_playbook_extractor_config": {
                    "extraction_definition_prompt": "Extract direct evidence",
                    "window_size_override": 4,
                    "stride_size_override": 2,
                },
            }
        )
        assert get_extractor_window_params(
            config.user_playbook_extractor_config,
            config.window_size,
            config.stride_size,
        ) == (4, 2)
        assert config.profile_extractor_config is None

    def test_durable_admission_still_checks_bypassed_model_validation(self):
        from reflexio.models.api_schema.domain.entities import (
            PublishUserInteractionRequest,
        )
        from reflexio.server.services.durable_learning.admission import (
            extraction_admission,
        )

        config = Config(storage_config=None)
        profile = config.profile_extractor_config
        assert profile is not None
        unchecked = config.model_copy(
            update={
                "profile_extractor_config": profile.model_copy(
                    update={"window_size_override": 2}
                )
            }
        )
        request = PublishUserInteractionRequest.model_validate(
            {
                "user_id": "user",
                "session_id": "session",
                "interaction_data_list": [{"role": "user", "content": "source"}],
            }
        )
        with pytest.raises(ValueError, match="stride_size must be <= window_size"):
            extraction_admission(unchecked, request)
