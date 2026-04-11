# ruff: noqa: S101
#!/usr/bin/env python3
"""
Simple test script to verify the new config storage classes work correctly.
"""

import tempfile

from reflexio.models.config_schema import (
    Config,
    StorageConfigSQLite,
)
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)


def test_local_file_config_storage():
    """Test LocalFileConfigStorage functionality."""
    print("Testing LocalFileConfigStorage...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        org_id = "test_org_123"

        # Create storage instance
        storage = LocalFileConfigStorage(org_id=org_id, base_dir=temp_dir)

        # Test get_default_config
        default_config = storage.get_default_config()
        print(
            f"  Default config storage type: {type(default_config.storage_config).__name__}"
        )
        assert isinstance(default_config.storage_config, StorageConfigSQLite)

        # Test load_config (should create default if file doesn't exist)
        loaded_config = storage.load_config()
        print(
            f"  Loaded config storage type: {type(loaded_config.storage_config).__name__}"
        )
        assert isinstance(loaded_config.storage_config, StorageConfigSQLite)

        # Test save_config
        test_config = Config(
            storage_config=StorageConfigSQLite(),
            agent_context_prompt="Test context prompt",
        )
        storage.save_config(test_config)
        print("  Config saved successfully")

        # Test load_config again (should load the saved config)
        reloaded_config = storage.load_config()
        print(
            f"  Reloaded config agent context: {reloaded_config.agent_context_prompt}"
        )
        assert reloaded_config.agent_context_prompt == "Test context prompt"

    print("  LocalFileConfigStorage tests passed!")


def test_configurator_integration():
    """Test that the configurator works with the new storage classes."""
    print("Testing configurator integration...")

    # Test with local storage
    with tempfile.TemporaryDirectory() as temp_dir:
        from reflexio.server.services.configurator.configurator import (
            DefaultConfigurator,
        )

        org_id = "test_org_789"
        configurator = DefaultConfigurator(org_id=org_id, base_dir=temp_dir)

        storage_config = configurator.get_current_storage_configuration()
        print(f"  Configurator storage type: {type(storage_config).__name__}")
        assert isinstance(storage_config, StorageConfigSQLite)

        # Test setting and getting config
        configurator.set_config_by_name(
            "agent_context_prompt", "Integration test prompt"
        )
        context = configurator.get_agent_context()
        print(f"  Agent context: {context}")
        assert context == "Integration test prompt"

    print("  Configurator integration tests passed!")


def test_load_config():
    """Test that the configurator loads the config correctly."""
    print("Testing load_config...")

    new_config = Config(
        storage_config=StorageConfigSQLite(
            db_path="/tmp/test.db",
        ),
        profile_extractor_configs=[],
    )

    json_config = new_config.model_dump_json()
    config = Config.model_validate_json(json_config)

    assert isinstance(config.storage_config, StorageConfigSQLite)
    assert config.storage_config.db_path == "/tmp/test.db"

    print("  Load config tests passed!")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
