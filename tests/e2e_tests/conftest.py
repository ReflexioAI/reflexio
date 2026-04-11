"""Shared fixtures and utilities for end-to-end integration tests."""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    AgentSuccessConfig,
    Config,
    PlaybookAggregatorConfig,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
    ToolUseConfig,
)
from reflexio.server.services.configurator.configurator import DefaultConfigurator

_TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
_SCENARIO_DIR = _TEST_DATA_DIR / "scenarios" / "e2e"

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _zero_group_evaluation_delay():
    """Remove the 600s completion-delay gate in group evaluation for e2e tests.

    `run_group_evaluation` skips a session if its latest request is newer than
    `_EFFECTIVE_DELAY_SECONDS` ago. In real usage this prevents evaluating an
    in-progress session; in tests it blocks every assertion that calls the
    runner immediately after publishing interactions. Patch to 0 so the gate
    passes.
    """
    with patch(
        "reflexio.server.services.agent_success_evaluation.group_evaluation_runner._EFFECTIVE_DELAY_SECONDS",
        0,
    ):
        yield


@pytest.fixture
def sqlite_storage_config() -> StorageConfigSQLite:
    """Create a StorageConfigSQLite instance for e2e testing."""
    return StorageConfigSQLite()


@pytest.fixture
def test_org_id(worker_id: str) -> str:
    """Test organization ID unique per worker to avoid parallel test conflicts.

    Uses pytest-xdist's worker_id fixture to create unique org IDs when running
    tests in parallel. For single-process runs, worker_id is 'master'.
    """
    return f"e2e_test_org_{worker_id}"


@pytest.fixture
def reflexio_instance(
    sqlite_storage_config: StorageConfigSQLite, test_org_id: str
) -> Reflexio:
    """Create a Reflexio instance with SQLite storage for testing."""
    # Set up configuration for profile extraction
    config = Config(
        storage_config=sqlite_storage_config,
        agent_context_prompt="this is a sales agent",
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="test_profile_extractor",
                context_prompt="""
Conversation between sales agent and user, extract any information from the interaction if contains any information listed under definition
""",
                extraction_definition_prompt="""
name, age, intent of the conversations
""",
                metadata_definition_prompt="""
choice of ['basic_info', 'conversation_intent']
""",
            )
        ],
        user_playbook_extractor_configs=[
            PlaybookConfig(
                extractor_name="test_playbook",
                extraction_definition_prompt="""
playbook should be something user told you to do differently in the next session. something sales rep did that makes user not satisfied.
playbook content is what agent should do differently in the next session based on the conversation history and be actionable as much as possible.
for example:
if user mentions "I don't like the way you talked to me", summarize conversation history and playbook content should be what is the way agent talk which is not preferred by user.
""",
                aggregation_config=PlaybookAggregatorConfig(
                    min_cluster_size=3,
                ),
            )
        ],
        agent_success_configs=[
            AgentSuccessConfig(
                evaluation_name="test_agent_success",
                success_definition_prompt="sales agent is responding to user apporperately",
            )
        ],
        tool_can_use=[
            ToolUseConfig(
                tool_name="search",
                tool_description="Search for information",
            )
        ],
    )
    # Create configurator with the config directly
    configurator = DefaultConfigurator(org_id=test_org_id, config=config)
    return Reflexio(org_id=test_org_id, configurator=configurator)


@pytest.fixture
def reflexio_instance_profile_only(
    sqlite_storage_config: StorageConfigSQLite, test_org_id: str
) -> Reflexio:
    """Create an Reflexio instance with only profile extraction config."""
    config = Config(
        storage_config=sqlite_storage_config,
        agent_context_prompt="this is a sales agent",
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="test_profile_extractor",
                context_prompt="""
Conversation between sales agent and user, extract any information from the interaction if contains any information listed under definition
""",
                extraction_definition_prompt="""
name, age, intent of the conversations
""",
                metadata_definition_prompt="""
choice of ['basic_info', 'conversation_intent']
""",
            )
        ],
    )
    configurator = DefaultConfigurator(org_id=test_org_id, config=config)
    return Reflexio(org_id=test_org_id, configurator=configurator)


@pytest.fixture
def reflexio_instance_playbook_only(
    sqlite_storage_config: StorageConfigSQLite, test_org_id: str
) -> Reflexio:
    """Create an Reflexio instance with only playbook config."""
    config = Config(
        storage_config=sqlite_storage_config,
        agent_context_prompt="this is a sales agent",
        user_playbook_extractor_configs=[
            PlaybookConfig(
                extractor_name="test_playbook",
                extraction_definition_prompt="""
playbook should be something user told you to do differently in the next session. something sales rep did that makes user not satisfied.
playbook content is what agent should do differently in the next session based on the conversation history and be actionable as much as possible.
for example:
if user mentions "I don't like the way you talked to me", summarize conversation history and playbook content should be what is the way agent talk which is not preferred by user.
""",
                aggregation_config=PlaybookAggregatorConfig(
                    min_cluster_size=3,
                ),
            )
        ],
    )
    configurator = DefaultConfigurator(org_id=test_org_id, config=config)
    return Reflexio(org_id=test_org_id, configurator=configurator)


@pytest.fixture
def reflexio_instance_agent_success_only(
    sqlite_storage_config: StorageConfigSQLite, test_org_id: str
) -> Reflexio:
    """Create an Reflexio instance with only agent success config."""
    config = Config(
        storage_config=sqlite_storage_config,
        agent_context_prompt="this is a sales agent",
        agent_success_configs=[
            AgentSuccessConfig(
                evaluation_name="test_agent_success",
                success_definition_prompt="sales agent is responding to user apporperately",
            )
        ],
        tool_can_use=[
            ToolUseConfig(
                tool_name="search",
                tool_description="Search for information",
            )
        ],
    )
    configurator = DefaultConfigurator(org_id=test_org_id, config=config)
    return Reflexio(org_id=test_org_id, configurator=configurator)


@pytest.fixture
def sample_interaction_requests() -> list[InteractionData]:
    """Load interactions from the customer_support YAML scenario (Priya's conversation).

    Returns 16 interactions with profile signals (name, job, location, preferences)
    and playbook corrections (agent mistakes, user-specific instructions) to ensure
    both the stride gate and LLM should_run gate pass for profile and playbook extraction.
    """
    from tests.test_data.scenarios.yaml_loader import build_interactions, load_scenario

    scenario = load_scenario(_SCENARIO_DIR / "customer_support.yaml")
    participants = scenario["participants"]
    priya_conv = scenario["conversations"]["priya"]
    return build_interactions(priya_conv, participants)


def save_user_playbooks(reflexio_instance: Reflexio):
    """Load mock playbooks from CSV file."""
    user_playbooks = []
    csv_path = _TEST_DATA_DIR / "mock_playbooks.csv"

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        user_playbooks.extend(
            UserPlaybook(
                agent_version=row["agent_version"],
                request_id=row["request_id"],
                content=row["content"],
                playbook_name=row["playbook_name"],
            )
            for row in reader
        )
    reflexio_instance.request_context.storage.save_user_playbooks(user_playbooks)


def _get_playbook_names(instance: Reflexio) -> list[str]:
    """Extract playbook names from the Reflexio instance's config."""
    config = instance.request_context.configurator.get_config()
    if config and config.user_playbook_extractor_configs:
        return [fc.extractor_name for fc in config.user_playbook_extractor_configs]
    return []


def _cleanup_storage(instance: Reflexio):
    """Helper function to cleanup storage for an Reflexio instance."""
    try:
        # Only delete user_playbooks and agent_playbooks created by this instance's config
        for name in _get_playbook_names(instance):
            instance.request_context.storage.delete_all_user_playbooks_by_playbook_name(
                name
            )
            instance.request_context.storage.delete_all_agent_playbooks_by_playbook_name(
                name
            )
        instance.request_context.storage.delete_all_interactions()
        instance.request_context.storage.delete_all_profiles()
        instance.request_context.storage.delete_all_profile_change_logs()
        instance.request_context.storage.delete_all_agent_success_evaluation_results()
        instance.request_context.storage.delete_all_requests()
        instance.request_context.storage.delete_all_operation_states()
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")


@pytest.fixture
def cleanup_after_test(reflexio_instance):
    """Fixture to clean up test data before and after each test."""
    # Cleanup before test to ensure clean state
    _cleanup_storage(reflexio_instance)
    yield  # This allows the test to run
    # Cleanup after test
    _cleanup_storage(reflexio_instance)


@pytest.fixture
def cleanup_profile_only(reflexio_instance_profile_only):
    """Fixture to clean up test data for profile_only instance."""
    _cleanup_storage(reflexio_instance_profile_only)
    yield
    _cleanup_storage(reflexio_instance_profile_only)


@pytest.fixture
def cleanup_playbook_only(reflexio_instance_playbook_only):
    """Fixture to clean up test data for playbook_only instance."""
    _cleanup_storage(reflexio_instance_playbook_only)
    yield
    _cleanup_storage(reflexio_instance_playbook_only)


@pytest.fixture
def cleanup_agent_success_only(reflexio_instance_agent_success_only):
    """Fixture to clean up test data for agent_success_only instance."""
    _cleanup_storage(reflexio_instance_agent_success_only)
    yield
    _cleanup_storage(reflexio_instance_agent_success_only)


@pytest.fixture
def reflexio_instance_playbook_source_filtering(
    sqlite_storage_config: StorageConfigSQLite, test_org_id: str
) -> Reflexio:
    """Create an Reflexio instance with playbook configs using request_sources_enabled filtering."""
    config = Config(
        storage_config=sqlite_storage_config,
        agent_context_prompt="this is a sales agent",
        user_playbook_extractor_configs=[
            # AgentPlaybook config only enabled for "api" source
            PlaybookConfig(
                extractor_name="api_playbook",
                extraction_definition_prompt="""
playbook should be something user told you to do differently in the next session.
""",
                request_sources_enabled=["api"],
            ),
            # AgentPlaybook config only enabled for "webhook" source
            PlaybookConfig(
                extractor_name="webhook_playbook",
                extraction_definition_prompt="""
playbook should be something user told you to do differently in the next session.
""",
                request_sources_enabled=["webhook"],
            ),
            # AgentPlaybook config enabled for all sources (no filter)
            PlaybookConfig(
                extractor_name="all_sources_playbook",
                extraction_definition_prompt="""
playbook should be something user told you to do differently in the next session.
""",
                request_sources_enabled=None,
            ),
        ],
    )
    configurator = DefaultConfigurator(org_id=test_org_id, config=config)
    return Reflexio(org_id=test_org_id, configurator=configurator)


@pytest.fixture
def cleanup_playbook_source_filtering(reflexio_instance_playbook_source_filtering):
    """Fixture to clean up test data for playbook source filtering instance."""
    _cleanup_storage(reflexio_instance_playbook_source_filtering)
    yield
    _cleanup_storage(reflexio_instance_playbook_source_filtering)


@pytest.fixture
def reflexio_instance_manual_profile(
    sqlite_storage_config: StorageConfigSQLite, test_org_id: str
) -> Reflexio:
    """Create an Reflexio instance with manual profile generation config.

    This config has:
    - batch_size set (required for manual generation)
    - allow_manual_trigger=True on the extractor
    """
    config = Config(
        storage_config=sqlite_storage_config,
        agent_context_prompt="this is a sales agent",
        batch_size=10,  # Required for manual generation
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="manual_trigger_extractor",
                context_prompt="""
Conversation between sales agent and user, extract any information from the interaction if contains any information listed under definition
""",
                extraction_definition_prompt="""
name, age, intent of the conversations
""",
                metadata_definition_prompt="""
choice of ['basic_info', 'conversation_intent']
""",
                allow_manual_trigger=True,  # Required for manual generation
            )
        ],
    )
    configurator = DefaultConfigurator(org_id=test_org_id, config=config)
    return Reflexio(org_id=test_org_id, configurator=configurator)


@pytest.fixture
def cleanup_manual_profile(reflexio_instance_manual_profile):
    """Fixture to clean up test data for manual profile instance."""
    _cleanup_storage(reflexio_instance_manual_profile)
    yield
    _cleanup_storage(reflexio_instance_manual_profile)


@pytest.fixture
def reflexio_instance_manual_playbook(
    sqlite_storage_config: StorageConfigSQLite, test_org_id: str
) -> Reflexio:
    """Create an Reflexio instance with manual playbook generation config.

    This config has:
    - batch_size set (required for manual generation)
    - allow_manual_trigger=True on the extractor
    """
    config = Config(
        storage_config=sqlite_storage_config,
        agent_context_prompt="this is a sales agent",
        batch_size=10,  # Required for manual generation
        user_playbook_extractor_configs=[
            PlaybookConfig(
                extractor_name="manual_trigger_playbook",
                extraction_definition_prompt="""
playbook should be something user told you to do differently in the next session. something sales rep did that makes user not satisfied.
playbook content is what agent should do differently in the next session based on the conversation history and be actionable as much as possible.
""",
                allow_manual_trigger=True,  # Required for manual generation
            )
        ],
    )
    configurator = DefaultConfigurator(org_id=test_org_id, config=config)
    return Reflexio(org_id=test_org_id, configurator=configurator)


@pytest.fixture
def cleanup_manual_playbook(reflexio_instance_manual_playbook):
    """Fixture to clean up test data for manual playbook instance."""
    _cleanup_storage(reflexio_instance_manual_playbook)
    yield
    _cleanup_storage(reflexio_instance_manual_playbook)


@pytest.fixture
def reflexio_instance_multiple_profile_extractors(
    sqlite_storage_config: StorageConfigSQLite, test_org_id: str
) -> Reflexio:
    """Create an Reflexio instance with multiple profile extractors.

    This config has multiple extractors for testing extractor_names filtering:
    - extractor_basic_info: Extracts basic info
    - extractor_preferences: Extracts preferences
    - extractor_intent: Extracts conversation intent
    """
    config = Config(
        storage_config=sqlite_storage_config,
        agent_context_prompt="this is a sales agent",
        batch_size=20,
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="extractor_basic_info",
                context_prompt="Extract basic information about the user.",
                extraction_definition_prompt="name, company, role",
                metadata_definition_prompt="choice of ['basic_info']",
            ),
            ProfileExtractorConfig(
                extractor_name="extractor_preferences",
                context_prompt="Extract user preferences from the conversation.",
                extraction_definition_prompt="communication style, preferred contact method",
                metadata_definition_prompt="choice of ['preferences']",
            ),
            ProfileExtractorConfig(
                extractor_name="extractor_intent",
                context_prompt="Extract user intent from the conversation.",
                extraction_definition_prompt="conversation goal, buying intent",
                metadata_definition_prompt="choice of ['intent']",
            ),
        ],
    )
    configurator = DefaultConfigurator(org_id=test_org_id, config=config)
    return Reflexio(org_id=test_org_id, configurator=configurator)


@pytest.fixture
def cleanup_multiple_profile_extractors(
    reflexio_instance_multiple_profile_extractors,
):
    """Fixture to clean up test data for multiple profile extractors instance."""
    _cleanup_storage(reflexio_instance_multiple_profile_extractors)
    yield
    _cleanup_storage(reflexio_instance_multiple_profile_extractors)


@pytest.fixture
def reflexio_instance_multiple_playbook_extractors(
    sqlite_storage_config: StorageConfigSQLite, test_org_id: str
) -> Reflexio:
    """Create an Reflexio instance with multiple playbook extractors.

    This config has multiple extractors with different source filters:
    - api_only_playbook: Only runs for 'api' source
    - webhook_only_playbook: Only runs for 'webhook' source
    - general_playbook: Runs for all sources
    """
    config = Config(
        storage_config=sqlite_storage_config,
        agent_context_prompt="this is a sales agent",
        batch_size=20,
        user_playbook_extractor_configs=[
            PlaybookConfig(
                extractor_name="api_only_playbook",
                extraction_definition_prompt="Extract playbook from API interactions.",
                request_sources_enabled=["api"],
            ),
            PlaybookConfig(
                extractor_name="webhook_only_playbook",
                extraction_definition_prompt="Extract playbook from webhook interactions.",
                request_sources_enabled=["webhook"],
            ),
            PlaybookConfig(
                extractor_name="general_playbook",
                extraction_definition_prompt="Extract general playbook from all sources.",
                request_sources_enabled=None,  # All sources
            ),
        ],
    )
    configurator = DefaultConfigurator(org_id=test_org_id, config=config)
    return Reflexio(org_id=test_org_id, configurator=configurator)


@pytest.fixture
def cleanup_multiple_playbook_extractors(
    reflexio_instance_multiple_playbook_extractors,
):
    """Fixture to clean up test data for multiple playbook extractors instance."""
    _cleanup_storage(reflexio_instance_multiple_playbook_extractors)
    yield
    _cleanup_storage(reflexio_instance_multiple_playbook_extractors)
