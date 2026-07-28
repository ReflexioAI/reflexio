"""Default-on playbook aggregation configuration."""

from __future__ import annotations

from reflexio.models.config_schema import (
    Config,
    PlaybookAggregatorConfig,
    StorageConfigSQLite,
    UserPlaybookExtractorConfig,
    validate_stored_config,
)


def test_missing_and_legacy_null_aggregation_config_use_defaults() -> None:
    missing = UserPlaybookExtractorConfig(extraction_definition_prompt="Extract rules")
    stored = validate_stored_config(
        {
            "storage_config": {"db_path": None},
            "user_playbook_extractor_config": {
                "extraction_definition_prompt": "Extract rules",
                "aggregation_config": None,
            },
        }
    )
    legacy_null = stored.user_playbook_extractor_config

    assert legacy_null is not None
    for config in (missing, legacy_null):
        assert config.aggregation_config == PlaybookAggregatorConfig()
        assert config.aggregation_config.min_cluster_size == 2
        assert config.aggregation_config.reaggregation_trigger_count == 2


def test_default_root_config_enables_playbook_aggregation() -> None:
    config = Config(storage_config=StorageConfigSQLite())

    assert config.user_playbook_extractor_config is not None
    assert config.user_playbook_extractor_config.aggregation_config == (
        PlaybookAggregatorConfig()
    )


def test_min_cluster_size_one_remains_the_explicit_off_switch() -> None:
    config = UserPlaybookExtractorConfig(
        extraction_definition_prompt="Extract rules",
        aggregation_config=PlaybookAggregatorConfig(min_cluster_size=1),
    )

    assert config.aggregation_config.min_cluster_size == 1
