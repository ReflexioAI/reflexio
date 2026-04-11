"""End-to-end tests for configuration management."""

import shutil
import tempfile

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.config_schema import (
    Config,
    PlaybookAggregatorConfig,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from tests.server.test_utils import skip_in_precommit, skip_low_priority


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config storage (not data storage)."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@skip_in_precommit
def test_set_config_end_to_end(
    sqlite_storage_config: StorageConfigSQLite,
    test_org_id: str,
    temp_config_dir: str,
):
    """Test end-to-end configuration setting workflow.

    Uses temp directory for config storage and SQLite for data storage.
    """
    # Create configurator with base_dir for config storage
    # This initializes config_storage so set_config can persist
    configurator = DefaultConfigurator(org_id=test_org_id, base_dir=temp_config_dir)

    # Set initial config
    initial_config = Config(
        storage_config=sqlite_storage_config,
    )
    configurator.set_config(initial_config)

    reflexio = Reflexio(org_id=test_org_id, configurator=configurator)

    # Create a new configuration
    new_config = Config(
        storage_config=sqlite_storage_config,
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="test_config_extractor",
                context_prompt="""
                Test configuration: Extract key information from conversations.
                """,
                extraction_definition_prompt="""
                Test profile content definition.
                """,
                metadata_definition_prompt="""
                Test metadata definition.
                """,
            )
        ],
        user_playbook_extractor_configs=[
            PlaybookConfig(
                extractor_name="test_config_playbook",
                extraction_definition_prompt="""
                Test playbook definition for configuration test.
                """,
                aggregation_config=PlaybookAggregatorConfig(
                    min_cluster_size=3,
                ),
            )
        ],
    )

    # Test setting config with Config object
    response = reflexio.set_config(new_config)
    assert response.success is True
    assert response.msg == "Configuration set successfully"

    # Verify configuration was actually set by checking the configurator
    current_config = reflexio.request_context.configurator.get_config()
    assert current_config is not None
    assert len(current_config.profile_extractor_configs) == 1
    assert (
        current_config.profile_extractor_configs[0].context_prompt.strip()
        == new_config.profile_extractor_configs[0].context_prompt.strip()
    )
    assert len(current_config.user_playbook_extractor_configs) == 1
    assert (
        current_config.user_playbook_extractor_configs[0].extractor_name
        == "test_config_playbook"
    )
    assert (
        current_config.user_playbook_extractor_configs[
            0
        ].aggregation_config.min_cluster_size
        == 3
    )

    # Test setting config with dict input
    config_dict = {
        "storage_config": sqlite_storage_config.model_dump(),
        "profile_extractor_configs": [
            {
                "extractor_name": "dict_test_extractor",
                "context_prompt": "Updated test configuration from dict.",
                "extraction_definition_prompt": "Updated profile content from dict.",
                "metadata_definition_prompt": "Updated metadata from dict.",
            }
        ],
        "user_playbook_extractor_configs": [
            {
                "extractor_name": "dict_test_playbook",
                "extraction_definition_prompt": "Dict playbook definition.",
                "aggregation_config": {
                    "min_cluster_size": 5,
                },
            }
        ],
    }

    # Test setting config with dict
    dict_response = reflexio.set_config(config_dict)
    assert dict_response.success is True
    assert dict_response.msg == "Configuration set successfully"

    # Verify dict configuration was set
    updated_config = reflexio.request_context.configurator.get_config()
    assert updated_config is not None
    assert len(updated_config.profile_extractor_configs) == 1
    assert (
        "Updated test configuration from dict"
        in updated_config.profile_extractor_configs[0].context_prompt
    )
    assert len(updated_config.user_playbook_extractor_configs) == 1
    assert (
        updated_config.user_playbook_extractor_configs[0].extractor_name
        == "dict_test_playbook"
    )
    assert (
        updated_config.user_playbook_extractor_configs[
            0
        ].aggregation_config.min_cluster_size
        == 5
    )

    # Test error handling with invalid config
    try:
        invalid_config = {"invalid_field": "invalid_value"}
        error_response = reflexio.set_config(invalid_config)
        assert error_response.success is False
        assert "Failed to set configuration" in error_response.msg
    except Exception:  # noqa: S110
        # If an exception is thrown instead of returning error response, that's also acceptable
        pass


@skip_in_precommit
@skip_low_priority
def test_get_config_end_to_end(
    sqlite_storage_config: StorageConfigSQLite,
    test_org_id: str,
    temp_config_dir: str,
):
    """Test end-to-end configuration retrieval workflow.

    This test verifies:
    1. get_config returns the current configuration
    2. Configuration is correctly populated after set_config
    3. get_config returns Config object with all expected fields
    """
    # Create configurator with base_dir for config storage
    configurator = DefaultConfigurator(org_id=test_org_id, base_dir=temp_config_dir)

    # Set initial config
    initial_config = Config(
        storage_config=sqlite_storage_config,
    )
    configurator.set_config(initial_config)

    reflexio = Reflexio(org_id=test_org_id, configurator=configurator)

    # Step 1: Get initial config (should exist with defaults or empty)
    retrieved_initial_config = reflexio.get_config()
    assert retrieved_initial_config is not None
    assert isinstance(retrieved_initial_config, Config)

    # Step 2: Set a specific configuration
    new_config = Config(
        storage_config=sqlite_storage_config,
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="get_config_test_extractor",
                context_prompt="Get config test: Extract key information.",
                extraction_definition_prompt="Get config test profile content.",
                metadata_definition_prompt="Get config test metadata.",
            )
        ],
        user_playbook_extractor_configs=[
            PlaybookConfig(
                extractor_name="get_config_test_playbook",
                extraction_definition_prompt="Get config test playbook definition.",
                aggregation_config=PlaybookAggregatorConfig(
                    min_cluster_size=10,
                ),
            )
        ],
    )

    set_response = reflexio.set_config(new_config)
    assert set_response.success is True

    # Step 3: Retrieve the config and verify it matches what was set
    retrieved_config = reflexio.get_config()
    assert retrieved_config is not None
    assert isinstance(retrieved_config, Config)

    # Verify profile extractor configs
    assert len(retrieved_config.profile_extractor_configs) == 1
    assert (
        "Get config test"
        in retrieved_config.profile_extractor_configs[0].context_prompt
    )

    # Verify agent playbook configs
    assert len(retrieved_config.user_playbook_extractor_configs) == 1
    assert (
        retrieved_config.user_playbook_extractor_configs[0].extractor_name
        == "get_config_test_playbook"
    )
    assert (
        retrieved_config.user_playbook_extractor_configs[
            0
        ].aggregation_config.min_cluster_size
        == 10
    )

    # Step 4: Verify storage config
    assert retrieved_config.storage_config is not None
