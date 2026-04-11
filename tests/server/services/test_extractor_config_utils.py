"""
Unit tests for extractor_config_utils module.
"""

from dataclasses import dataclass

import pytest

from reflexio.server.services.extractor_config_utils import filter_extractor_configs

# ===============================
# Test Data Classes
# ===============================


@dataclass
class MockExtractorConfig:
    """Mock extractor config for testing."""

    extractor_name: str
    request_sources_enabled: list[str] | None = None
    manual_trigger: bool = False


@dataclass
class MockPlaybookConfig:
    """Mock playbook config with extractor_name."""

    extractor_name: str
    request_sources_enabled: list[str] | None = None
    manual_trigger: bool = False


@dataclass
class MockEvaluationConfig:
    """Mock evaluation config with evaluation_name instead of extractor_name."""

    evaluation_name: str
    request_sources_enabled: list[str] | None = None


# ===============================
# Test: filter_extractor_configs
# ===============================


class TestFilterExtractorConfigs:
    """Tests for the filter_extractor_configs utility function."""

    def test_no_filtering_when_no_parameters(self):
        """Test that all configs pass when no filtering parameters are provided."""
        configs = [
            MockExtractorConfig(extractor_name="extractor1"),
            MockExtractorConfig(extractor_name="extractor2"),
        ]

        result = filter_extractor_configs(configs)

        assert len(result) == 2

    def test_filter_by_source_enabled(self):
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

        result = filter_extractor_configs(configs, source="api")

        # extractor1 (api in enabled list) and extractor3 (no restriction) should pass
        assert len(result) == 2
        extractor_names = [c.extractor_name for c in result]
        assert "extractor1" in extractor_names
        assert "extractor3" in extractor_names
        assert "extractor2" not in extractor_names

    def test_filter_by_manual_trigger(self):
        """Test filtering extractors by manual_trigger flag."""
        configs = [
            MockExtractorConfig(extractor_name="extractor1", manual_trigger=True),
            MockExtractorConfig(extractor_name="extractor2", manual_trigger=False),
            MockExtractorConfig(extractor_name="extractor3"),  # Default False
        ]

        # allow_manual_trigger=False - manual_trigger=True extractors should be skipped
        result = filter_extractor_configs(configs, allow_manual_trigger=False)

        assert len(result) == 2
        extractor_names = [c.extractor_name for c in result]
        assert "extractor2" in extractor_names
        assert "extractor3" in extractor_names
        assert "extractor1" not in extractor_names

    def test_manual_trigger_allowed_when_allow_manual_trigger_true(self):
        """Test that manual_trigger extractors are allowed when allow_manual_trigger=True."""
        configs = [
            MockExtractorConfig(extractor_name="extractor1", manual_trigger=True),
            MockExtractorConfig(extractor_name="extractor2", manual_trigger=False),
        ]

        result = filter_extractor_configs(configs, allow_manual_trigger=True)

        assert len(result) == 2
        extractor_names = [c.extractor_name for c in result]
        assert "extractor1" in extractor_names
        assert "extractor2" in extractor_names

    def test_filter_by_extractor_names(self):
        """Test filtering extractors by extractor_names list."""
        configs = [
            MockExtractorConfig(extractor_name="extractor1"),
            MockExtractorConfig(extractor_name="extractor2"),
            MockExtractorConfig(extractor_name="extractor3"),
        ]

        result = filter_extractor_configs(
            configs, extractor_names=["extractor1", "extractor3"]
        )

        assert len(result) == 2
        extractor_names = [c.extractor_name for c in result]
        assert "extractor1" in extractor_names
        assert "extractor3" in extractor_names
        assert "extractor2" not in extractor_names

    def test_combined_filtering(self):
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
        result = filter_extractor_configs(
            configs,
            source="api",
            allow_manual_trigger=False,
            extractor_names=["extractor1", "extractor3"],
        )

        # Only extractor1 passes all filters:
        # - extractor2: wrong source
        # - extractor3: manual_trigger=True but allow_manual_trigger=False
        assert len(result) == 1
        assert result[0].extractor_name == "extractor1"

    def test_empty_configs_list(self):
        """Test filtering with empty configs list."""
        result = filter_extractor_configs([], source="api")
        assert len(result) == 0

    def test_none_source_allows_all(self):
        """Test that None source allows all configs regardless of request_sources_enabled."""
        configs = [
            MockExtractorConfig(
                extractor_name="extractor1", request_sources_enabled=["api"]
            ),
            MockExtractorConfig(extractor_name="extractor2"),
        ]

        result = filter_extractor_configs(configs, source=None)

        # Both should pass since source is None (no filtering by source)
        assert len(result) == 2

    def test_empty_source_string(self):
        """Test filtering with empty string source."""
        configs = [
            MockExtractorConfig(
                extractor_name="extractor1", request_sources_enabled=["api"]
            ),
            MockExtractorConfig(extractor_name="extractor2"),
        ]

        # Empty string source should not trigger source filtering
        result = filter_extractor_configs(configs, source="")

        assert len(result) == 2

    def test_playbook_config_with_playbook_name(self):
        """Test filtering playbook configs that use playbook_name."""
        configs = [
            MockPlaybookConfig(
                extractor_name="playbook1", request_sources_enabled=["api"]
            ),
            MockPlaybookConfig(
                extractor_name="playbook2", request_sources_enabled=["mobile"]
            ),
        ]

        result = filter_extractor_configs(configs, source="api")

        assert len(result) == 1
        assert result[0].extractor_name == "playbook1"

    def test_evaluation_config_with_evaluation_name(self):
        """Test filtering evaluation configs that use evaluation_name."""
        configs = [
            MockEvaluationConfig(
                evaluation_name="eval1", request_sources_enabled=["api"]
            ),
            MockEvaluationConfig(
                evaluation_name="eval2", request_sources_enabled=["mobile"]
            ),
        ]

        result = filter_extractor_configs(configs, source="api")

        assert len(result) == 1
        assert result[0].evaluation_name == "eval1"

    def test_filter_by_extractor_names_with_playbook_config(self):
        """Test filtering by names works for playbook configs."""
        configs = [
            MockPlaybookConfig(extractor_name="playbook1"),
            MockPlaybookConfig(extractor_name="playbook2"),
        ]

        result = filter_extractor_configs(configs, extractor_names=["playbook1"])

        assert len(result) == 1
        assert result[0].extractor_name == "playbook1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
