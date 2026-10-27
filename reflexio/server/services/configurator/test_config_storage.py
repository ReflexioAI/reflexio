# ruff: noqa: S101
#!/usr/bin/env python3
"""
Simple test script to verify the new config storage classes work correctly.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from reflexio.cli.config_upgrade import convert_config_file
from reflexio.config_env import StoredConfigurationError
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


def test_legacy_list_shape_requires_explicit_conversion():
    """Only explicit offline conversion accepts aliases, preserving originals."""
    with tempfile.TemporaryDirectory() as temp_dir:
        org_id = "legacy_org"
        storage = LocalFileConfigStorage(org_id=org_id, base_dir=temp_dir)

        legacy_payload = {
            "storage_config": StorageConfigSQLite().model_dump(),
            "profile_extractor_configs": [
                {
                    "extractor_name": "legacy_profile",
                    "extraction_definition_prompt": "Extract legacy profile data.",
                }
            ],
            "user_playbook_extractor_configs": [
                {
                    "extractor_name": "legacy_playbook",
                    "extraction_definition_prompt": "Extract legacy playbook rules.",
                }
            ],
            "agent_success_configs": [
                {
                    "evaluation_name": "legacy_success",
                    "success_definition_prompt": "Evaluate legacy agent success.",
                }
            ],
        }
        path = Path(storage.config_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        original = json.dumps(legacy_payload).encode()
        path.write_bytes(original)

        with pytest.raises(StoredConfigurationError):
            storage.load_config()
        assert path.read_bytes() == original
        assert not (path.parent / ".config-migrations").exists()
        converted = LocalFileConfigStorage(
            org_id=org_id, base_dir=str(Path(temp_dir) / "converted")
        )
        destination = Path(converted.config_file)
        destination.parent.mkdir(parents=True)
        convert_config_file(path, destination)
        assert path.read_bytes() == original
        storage = converted

        loaded = storage.load_config()

        assert loaded.profile_extractor_config is not None
        assert "extractor_name" not in loaded.profile_extractor_config.model_dump()
        assert (
            loaded.profile_extractor_config.extraction_definition_prompt
            == "Extract legacy profile data."
        )
        assert loaded.user_playbook_extractor_config is not None
        assert (
            "extractor_name" not in loaded.user_playbook_extractor_config.model_dump()
        )
        assert (
            loaded.user_playbook_extractor_config.extraction_definition_prompt
            == "Extract legacy playbook rules."
        )
        assert loaded.agent_success_config is not None
        assert "evaluation_name" not in loaded.agent_success_config.model_dump()
        assert (
            loaded.agent_success_config.success_definition_prompt
            == "Evaluate legacy agent success."
        )
        storage.save_config(loaded)
        saved = json.loads(Path(storage.config_file).read_text(encoding="utf-8"))
        assert "extractor_name" not in saved["profile_extractor_config"]
        assert "extractor_name" not in saved["user_playbook_extractor_config"]
        assert "evaluation_name" not in saved["agent_success_config"]
        assert storage.load_config() == loaded


def test_load_config_preserves_retired_key_without_rewriting():
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalFileConfigStorage(org_id="legacy_reflection", base_dir=temp_dir)
        payload = {
            "storage_config": StorageConfigSQLite().model_dump(mode="json"),
            "agent_context_prompt": "keep this setting",
            "reflection_config": {"enabled": False},
        }
        path = Path(storage.config_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        original = path.read_bytes()

        loaded = storage.load_config()

        assert loaded.agent_context_prompt == "keep this setting"
        assert path.read_bytes() == original
        assert not (path.parent / ".config-migrations").exists()


def test_load_config_never_attempts_cleanup_publication():
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalFileConfigStorage(
            org_id="readonly_reflection", base_dir=temp_dir
        )
        payload = {
            "storage_config": StorageConfigSQLite().model_dump(mode="json"),
            "agent_context_prompt": "still available",
            "reflection_config": {"enabled": True},
        }
        path = Path(storage.config_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        original = path.read_bytes()

        with patch.object(
            storage,
            "_save_config_to_local_dir",
            side_effect=OSError("read only"),
        ) as publish:
            loaded = storage.load_config()

        publish.assert_not_called()
        assert loaded.agent_context_prompt == "still available"
        assert path.read_bytes() == original


def test_load_config():
    """Test that the configurator loads the config correctly."""
    print("Testing load_config...")

    new_config = Config(
        storage_config=StorageConfigSQLite(
            db_path="/tmp/test.db",
        ),
        profile_extractor_config=None,
    )

    json_config = new_config.model_dump_json()
    config = Config.model_validate_json(json_config)

    assert isinstance(config.storage_config, StorageConfigSQLite)
    assert config.storage_config.db_path == "/tmp/test.db"

    print("  Load config tests passed!")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
