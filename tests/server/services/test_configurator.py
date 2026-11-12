from pathlib import Path

import pytest

from reflexio.config_file import LocalConfigFile
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
    loaded_config = Config.model_validate(LocalConfigFile(config_file).read_document())
    assert isinstance(loaded_config.storage_config, StorageConfigSQLite)
    assert loaded_config.profile_extractor_config is not None
    assert loaded_config.profile_extractor_config.extraction_definition_prompt
    assert loaded_config.user_playbook_extractor_config is not None
    assert loaded_config.user_playbook_extractor_config.extraction_definition_prompt


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
            "profile_extractor_config",
            ProfileExtractorConfig(
                should_extract_profile_prompt_override="test",
                context_prompt="test",
                extraction_definition_prompt="test",
                tagging_definition_prompt="test",
            ),
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
        profile_extractor_config=None,
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


def test_manual_window_round_trip_does_not_acquire_automatic_stride_gate(
    configurator, temp_dir, test_org_id
):
    from reflexio.server.services.extractor_interaction_utils import (
        get_extractor_window_params,
    )

    config = configurator.get_config()
    profile = config.profile_extractor_config
    assert profile is not None
    # Manual API calls do not require manual_trigger=True. Preserve that policy.
    patch = {
        **profile.model_dump(),
        "window_size_override": 2,
        "stride_size_override": None,
        "manual_trigger": False,
    }
    configurator.set_config(
        configurator.prepare_config_patch({"profile_extractor_config": patch})
    )
    reloaded = DefaultConfigurator(org_id=test_org_id, base_dir=temp_dir).get_config()
    window = get_extractor_window_params(
        reloaded.profile_extractor_config, reloaded.window_size, reloaded.stride_size
    )
    assert window == (2, 8)
    assert reloaded.profile_extractor_config is not None
    assert reloaded.profile_extractor_config.manual_trigger is False
    with pytest.raises(ValueError, match="stride_size must be <= window_size"):
        window.require_automatic()
