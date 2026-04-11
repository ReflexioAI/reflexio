import json
from pathlib import Path

import pytest

from reflexio.models.config_schema import (
    Config,
    LLMConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.configurator.configurator import DefaultConfigurator


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def test_org_id():
    return "test_org"


@pytest.fixture
def configurator(temp_dir, test_org_id):
    return DefaultConfigurator(org_id=test_org_id, base_dir=temp_dir)


def test_init_creates_config_file(temp_dir, test_org_id):
    # Test that initialization creates config file if it doesn't exist
    _config = DefaultConfigurator(org_id=test_org_id, base_dir=temp_dir)
    config_file = Path(temp_dir) / "configs" / f"config_{test_org_id}.json"

    assert config_file.exists()
    with open(config_file, encoding="utf-8") as f:
        loaded_config = Config.model_validate(json.load(f))
        assert isinstance(loaded_config.storage_config, StorageConfigSQLite)
        assert loaded_config.profile_extractor_configs is not None
        assert len(loaded_config.profile_extractor_configs) == 1
        assert (
            loaded_config.profile_extractor_configs[0].extractor_name
            == "default_profile_extractor"
        )
        assert loaded_config.user_playbook_extractor_configs is not None
        assert len(loaded_config.user_playbook_extractor_configs) == 1
        assert (
            loaded_config.user_playbook_extractor_configs[0].extractor_name
            == "default_playbook_extractor"
        )


def test_get_config_with_default(configurator):
    # Test getting non-existent config returns default value
    config = configurator.get_config()
    # Since get_config() returns the full Config object, we can check if a field exists
    # or has a default value by accessing it directly
    assert hasattr(config, "storage_config")
    assert isinstance(config.storage_config, StorageConfigSQLite)


def test_set_and_get_config_by_name(configurator):
    # Test setting and getting config values using set_config_by_name
    test_cases = [
        (
            "storage_config",
            StorageConfigSQLite(
                db_path="/tmp/test.db",  # noqa: S108
            ),
        ),
        (
            "profile_extractor_configs",
            [
                ProfileExtractorConfig(
                    extractor_name="test_extractor",
                    should_extract_profile_prompt_override="test",
                    context_prompt="test",
                    extraction_definition_prompt="test",
                    metadata_definition_prompt="test",
                )
            ],
        ),
    ]

    for key, value in test_cases:
        configurator.set_config_by_name(key, value)
        config = configurator.get_config()
        assert getattr(config, key) == value


def test_config_persistence(temp_dir, test_org_id):
    # Test that config values persist after recreating the configurator
    config1 = DefaultConfigurator(org_id=test_org_id, base_dir=temp_dir)
    new_config = Config(
        storage_config=StorageConfigSQLite(
            db_path="/tmp/test.db",  # noqa: S108
        ),
        profile_extractor_configs=[],
    )
    config1.set_config(new_config)

    # Create new instance to read from the same file
    config2 = DefaultConfigurator(org_id=test_org_id, base_dir=temp_dir)
    assert isinstance(config2.config.storage_config, StorageConfigSQLite)
    assert config2.config.storage_config.db_path == "/tmp/test.db"  # noqa: S108


def test_llm_config_persists_pre_retrieval_model_name(temp_dir, test_org_id):
    configurator = DefaultConfigurator(org_id=test_org_id, base_dir=temp_dir)
    new_config = Config(
        storage_config=StorageConfigSQLite(db_path="/tmp/test.db"),  # noqa: S108
        llm_config=LLMConfig(pre_retrieval_model_name="gpt-5.4-nano"),
    )

    configurator.set_config(new_config)

    reloaded = DefaultConfigurator(org_id=test_org_id, base_dir=temp_dir)
    assert reloaded.config.llm_config is not None
    assert reloaded.config.llm_config.pre_retrieval_model_name == "gpt-5.4-nano"


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-k", "test_init_creates_config_file"])
