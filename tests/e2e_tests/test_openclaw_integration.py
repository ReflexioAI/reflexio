"""E2E tests for OpenClaw integration — multi-user, aggregation, search.

Tests the full data lifecycle: publish interactions from multiple "instances"
(different user_ids), verify extraction, run aggregation, and verify search
returns both user and agent playbooks.
"""

import os

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.retriever_schema import (
    GetAgentPlaybooksRequest,
    GetUserPlaybooksRequest,
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
    UnifiedSearchRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AddAgentPlaybookRequest,
    AddUserPlaybookRequest,
    AgentPlaybook,
    InteractionData,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    Config,
    PlaybookAggregatorConfig,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from tests.server.test_utils import skip_in_precommit

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ALPHA_USER = "instance-alpha"
_BETA_USER = "instance-beta"
_AGENT_VERSION = "openclaw-v1"
_PLAYBOOK_NAME = "openclaw_playbook"


def _make_correction_interactions(topic: str) -> list[InteractionData]:
    """Build a minimal conversation containing a correction signal for a given topic."""
    return [
        InteractionData(
            role="user",
            content=f"Please help me with {topic}.",
        ),
        InteractionData(
            role="assistant",
            content=f"I'll assist you with {topic} using the default approach.",
        ),
        InteractionData(
            role="user",
            content=(
                f"No, don't do it that way for {topic}. "
                "Next time please ask for my preference first before proceeding."
            ),
        ),
        InteractionData(
            role="assistant",
            content="Understood, I'll ask for your preference first next time.",
        ),
    ]


def _make_preference_interactions(preference: str) -> list[InteractionData]:
    """Build a minimal conversation that reveals a user preference."""
    return [
        InteractionData(
            role="user",
            content=f"I prefer {preference} whenever possible.",
        ),
        InteractionData(
            role="assistant",
            content=f"Noted! I'll keep {preference} in mind for our future sessions.",
        ),
        InteractionData(
            role="user",
            content="Great, please always remember that about me.",
        ),
    ]


def _make_reflexio_with_playbook(
    org_id: str,
    storage_config: StorageConfigSQLite,
    min_cluster_size: int = 2,
) -> Reflexio:
    """Return a Reflexio instance configured for playbook extraction and aggregation."""
    config = Config(
        storage_config=storage_config,
        agent_context_prompt="AI coding assistant that helps developers",
        user_playbook_extractor_configs=[
            PlaybookConfig(
                extractor_name=_PLAYBOOK_NAME,
                extraction_definition_prompt=(
                    "Extract any correction the user gave the assistant — "
                    "something the assistant did wrong that the user asked to change. "
                    "Playbook content should be an actionable instruction for the next session."
                ),
                aggregation_config=PlaybookAggregatorConfig(
                    min_cluster_size=min_cluster_size,
                ),
            )
        ],
    )
    configurator = DefaultConfigurator(org_id=org_id, config=config)
    return Reflexio(org_id=org_id, configurator=configurator)


def _make_reflexio_with_profile(
    org_id: str,
    storage_config: StorageConfigSQLite,
) -> Reflexio:
    """Return a Reflexio instance configured for profile extraction only."""
    config = Config(
        storage_config=storage_config,
        agent_context_prompt="AI coding assistant that learns about the user",
        profile_extractor_configs=[
            ProfileExtractorConfig(
                extractor_name="openclaw_profile",
                context_prompt="Extract user preferences and work habits from the conversation.",
                extraction_definition_prompt=(
                    "coding language preferences, tool preferences, workflow preferences"
                ),
                metadata_definition_prompt="choice of ['preference', 'workflow']",
            )
        ],
    )
    configurator = DefaultConfigurator(org_id=org_id, config=config)
    return Reflexio(org_id=org_id, configurator=configurator)


def _cleanup(instance: Reflexio, playbook_name: str | None = None) -> None:
    """Delete all test data created by an instance."""
    storage = instance.request_context.storage
    try:
        if playbook_name:
            storage.delete_all_user_playbooks_by_playbook_name(playbook_name)
            storage.delete_all_agent_playbooks_by_playbook_name(playbook_name)
        storage.delete_all_interactions()
        storage.delete_all_profiles()
        storage.delete_all_requests()
        storage.delete_all_operation_states()
    except Exception as exc:
        print(f"cleanup error (ignored): {exc}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openclaw_playbook_instance(
    sqlite_storage_config: StorageConfigSQLite,
    test_org_id: str,
) -> Reflexio:
    """Reflexio instance configured for OpenClaw playbook extraction."""
    instance = _make_reflexio_with_playbook(test_org_id, sqlite_storage_config)
    _cleanup(instance, _PLAYBOOK_NAME)
    yield instance
    _cleanup(instance, _PLAYBOOK_NAME)


@pytest.fixture
def openclaw_profile_instance(
    sqlite_storage_config: StorageConfigSQLite,
    test_org_id: str,
) -> Reflexio:
    """Reflexio instance configured for OpenClaw profile extraction."""
    instance = _make_reflexio_with_profile(test_org_id, sqlite_storage_config)
    _cleanup(instance)
    yield instance
    _cleanup(instance)


# ---------------------------------------------------------------------------
# TestOpenClawMultiUser
# ---------------------------------------------------------------------------


class TestOpenClawMultiUser:
    """Each OpenClaw agent instance publishes as a separate Reflexio user."""

    @skip_in_precommit
    def test_publish_with_different_user_ids(
        self,
        openclaw_playbook_instance: Reflexio,
    ) -> None:
        """Publish interactions for instance-alpha and instance-beta.

        Verifies both are stored and scoped correctly: each user sees only
        their own user playbooks.
        """
        instance = openclaw_playbook_instance
        storage = instance.request_context.storage

        alpha_turns = _make_correction_interactions("file formatting")
        beta_turns = _make_correction_interactions("code review comments")

        alpha_resp = instance.publish_interaction(
            {
                "user_id": _ALPHA_USER,
                "interaction_data_list": alpha_turns,
                "agent_version": _AGENT_VERSION,
                "source": "openclaw",
            }
        )
        assert alpha_resp.success is True

        beta_resp = instance.publish_interaction(
            {
                "user_id": _BETA_USER,
                "interaction_data_list": beta_turns,
                "agent_version": _AGENT_VERSION,
                "source": "openclaw",
            }
        )
        assert beta_resp.success is True

        # Both sets of interactions should be present in storage
        all_interactions = storage.get_all_interactions()
        assert len(all_interactions) == len(alpha_turns) + len(beta_turns)

        # User playbooks are scoped by user_id via the linked request
        alpha_playbooks = storage.get_user_playbooks(
            playbook_name=_PLAYBOOK_NAME,
            user_id=_ALPHA_USER,
        )
        beta_playbooks = storage.get_user_playbooks(
            playbook_name=_PLAYBOOK_NAME,
            user_id=_BETA_USER,
        )

        # Each user should have their own extracted playbooks
        assert alpha_playbooks, "instance-alpha should have extracted user playbooks"
        assert beta_playbooks, "instance-beta should have extracted user playbooks"

        # Playbooks from one user must not bleed into the other
        alpha_ids = {p.user_playbook_id for p in alpha_playbooks}
        beta_ids = {p.user_playbook_id for p in beta_playbooks}
        assert not alpha_ids.intersection(
            beta_ids
        ), "user playbooks must not overlap between instances"

    @skip_in_precommit
    def test_user_profiles_scoped_to_instance(
        self,
        openclaw_profile_instance: Reflexio,
    ) -> None:
        """Profiles extracted from one instance are not visible to another.

        Publish interactions with preferences from instance-alpha, then verify
        only instance-alpha profiles are returned when filtering by user_id.
        """
        instance = openclaw_profile_instance
        storage = instance.request_context.storage

        alpha_turns = _make_preference_interactions("concise commit messages")

        publish_resp = instance.publish_interaction(
            {
                "user_id": _ALPHA_USER,
                "interaction_data_list": alpha_turns,
                "agent_version": _AGENT_VERSION,
                "source": "openclaw",
            }
        )
        assert publish_resp.success is True

        # Alpha's profiles should be present
        alpha_profiles = storage.get_user_profile(_ALPHA_USER)
        assert alpha_profiles, "instance-alpha should have extracted profiles"

        # Beta never published — no profiles for that user_id
        beta_profiles = storage.get_user_profile(_BETA_USER)
        assert not beta_profiles, (
            "instance-beta should have no profiles; "
            f"found: {[p.content for p in beta_profiles]}"
        )


# ---------------------------------------------------------------------------
# TestAgentPlaybookAggregation
# ---------------------------------------------------------------------------


class TestAgentPlaybookAggregation:
    """Agent playbooks aggregate corrections from all instances."""

    @skip_in_precommit
    def test_aggregation_produces_agent_playbooks(
        self,
        openclaw_playbook_instance: Reflexio,
    ) -> None:
        """Publish corrections from 2 instances, aggregate, verify agent playbooks created.

        Uses mock LLM mode for clustering (avoids needing real embeddings).
        """
        instance = openclaw_playbook_instance
        storage = instance.request_context.storage

        # Publish similar corrections from two users to meet min_cluster_size=2
        for user_id in (_ALPHA_USER, _BETA_USER):
            resp = instance.publish_interaction(
                {
                    "user_id": user_id,
                    "interaction_data_list": _make_correction_interactions(
                        "code formatting"
                    ),
                    "agent_version": _AGENT_VERSION,
                    "source": "openclaw",
                }
            )
            assert resp.success is True

        # Confirm user playbooks were generated for both users
        user_playbooks = storage.get_user_playbooks(playbook_name=_PLAYBOOK_NAME)
        assert user_playbooks, "user playbooks must exist before aggregation"

        original_mock = os.environ.get("MOCK_LLM_RESPONSE")
        try:
            os.environ["MOCK_LLM_RESPONSE"] = "true"
            instance.run_playbook_aggregation(
                agent_version=_AGENT_VERSION,
                playbook_name=_PLAYBOOK_NAME,
            )
        finally:
            if original_mock is None:
                os.environ.pop("MOCK_LLM_RESPONSE", None)
            else:
                os.environ["MOCK_LLM_RESPONSE"] = original_mock

        agent_playbooks = storage.get_agent_playbooks(
            playbook_name=_PLAYBOOK_NAME,
            playbook_status_filter=[PlaybookStatus.PENDING],
        )
        assert agent_playbooks, (
            "aggregation should produce at least one agent playbook with PENDING status"
        )
        assert all(
            p.playbook_status == PlaybookStatus.PENDING for p in agent_playbooks
        )

    @skip_in_precommit
    def test_all_instances_see_agent_playbooks(
        self,
        openclaw_playbook_instance: Reflexio,
    ) -> None:
        """After aggregation, search from any instance returns agent playbooks.

        Seeds agent playbooks directly, then verifies both alpha and beta
        searches return the same shared playbook data.
        """
        instance = openclaw_playbook_instance

        # Seed an agent playbook directly (simulates post-aggregation state)
        seed_resp = instance.add_agent_playbook(
            AddAgentPlaybookRequest(
                agent_playbooks=[
                    AgentPlaybook(
                        agent_version=_AGENT_VERSION,
                        playbook_name=_PLAYBOOK_NAME,
                        content="Always ask the user before applying auto-formatting.",
                        playbook_status=PlaybookStatus.PENDING,
                    )
                ]
            )
        )
        assert seed_resp.success is True

        # Search from alpha's perspective
        alpha_resp = instance.search_agent_playbooks(
            SearchAgentPlaybookRequest(
                query="ask before formatting",
                playbook_name=_PLAYBOOK_NAME,
                agent_version=_AGENT_VERSION,
            )
        )
        assert alpha_resp.success is True
        assert alpha_resp.agent_playbooks, (
            "instance-alpha should see agent playbooks after aggregation"
        )

        # Search from beta's perspective (same query)
        beta_resp = instance.search_agent_playbooks(
            SearchAgentPlaybookRequest(
                query="ask before formatting",
                playbook_name=_PLAYBOOK_NAME,
                agent_version=_AGENT_VERSION,
            )
        )
        assert beta_resp.success is True
        assert beta_resp.agent_playbooks, (
            "instance-beta should see the same agent playbooks"
        )

        # Both searches should return the same playbook ids (agent playbooks are global)
        alpha_ids = {p.agent_playbook_id for p in alpha_resp.agent_playbooks}
        beta_ids = {p.agent_playbook_id for p in beta_resp.agent_playbooks}
        assert alpha_ids == beta_ids, (
            "both instances should see the same shared agent playbooks"
        )


# ---------------------------------------------------------------------------
# TestUnifiedSearch
# ---------------------------------------------------------------------------


class TestUnifiedSearch:
    """Unified search returns both user and agent playbooks."""

    @skip_in_precommit
    def test_search_returns_both_playbook_types(
        self,
        openclaw_playbook_instance: Reflexio,
    ) -> None:
        """Seed user playbooks and agent playbooks, verify unified search returns both."""
        instance = openclaw_playbook_instance

        # Seed a user playbook
        user_seed = instance.add_user_playbook(
            AddUserPlaybookRequest(
                user_playbooks=[
                    UserPlaybook(
                        user_id=_ALPHA_USER,
                        agent_version=_AGENT_VERSION,
                        request_id="seed-request-unified",
                        playbook_name=_PLAYBOOK_NAME,
                        content=(
                            "Always confirm the target branch before creating a PR."
                        ),
                    )
                ]
            )
        )
        assert user_seed.success is True

        # Seed an agent playbook
        agent_seed = instance.add_agent_playbook(
            AddAgentPlaybookRequest(
                agent_playbooks=[
                    AgentPlaybook(
                        agent_version=_AGENT_VERSION,
                        playbook_name=_PLAYBOOK_NAME,
                        content=(
                            "When creating PRs, ask the user to confirm branch and reviewers."
                        ),
                        playbook_status=PlaybookStatus.PENDING,
                    )
                ]
            )
        )
        assert agent_seed.success is True

        # Unified search should surface both
        search_resp = instance.unified_search(
            UnifiedSearchRequest(
                query="pull request branch",
                user_id=_ALPHA_USER,
                playbook_name=_PLAYBOOK_NAME,
                agent_version=_AGENT_VERSION,
            ),
            org_id=instance.org_id,
        )
        assert search_resp.success is True
        assert search_resp.user_playbooks, (
            "unified search should return user playbooks"
        )
        assert search_resp.agent_playbooks, (
            "unified search should return agent playbooks"
        )

    @skip_in_precommit
    def test_search_relevance(
        self,
        openclaw_playbook_instance: Reflexio,
    ) -> None:
        """Search returns relevant results, not irrelevant ones.

        Seeds a deployment-related playbook and a terminal-related playbook,
        then asserts the deployment query returns the deployment playbook
        in top results.
        """
        instance = openclaw_playbook_instance

        deployment_content = (
            "Before deploying, always run the test suite and confirm with the user."
        )
        terminal_content = (
            "When opening a terminal session, ask which shell the user prefers."
        )

        instance.add_user_playbook(
            AddUserPlaybookRequest(
                user_playbooks=[
                    UserPlaybook(
                        user_id=_ALPHA_USER,
                        agent_version=_AGENT_VERSION,
                        request_id="seed-deploy",
                        playbook_name=_PLAYBOOK_NAME,
                        content=deployment_content,
                    ),
                    UserPlaybook(
                        user_id=_ALPHA_USER,
                        agent_version=_AGENT_VERSION,
                        request_id="seed-terminal",
                        playbook_name=_PLAYBOOK_NAME,
                        content=terminal_content,
                    ),
                ]
            )
        )

        search_resp = instance.search_user_playbooks(
            SearchUserPlaybookRequest(
                query="deploy",
                user_id=_ALPHA_USER,
                playbook_name=_PLAYBOOK_NAME,
                top_k=5,
            )
        )
        assert search_resp.success is True
        assert search_resp.user_playbooks, "should find at least one playbook"

        top_result = search_resp.user_playbooks[0]
        assert "deploy" in top_result.content.lower(), (
            f"top result should be the deployment playbook; got: {top_result.content!r}"
        )


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Integration handles errors gracefully."""

    @skip_in_precommit
    def test_search_with_empty_storage_returns_empty(
        self,
        openclaw_playbook_instance: Reflexio,
    ) -> None:
        """Search against an empty storage returns an empty result without raising.

        Verifies the OpenClaw search path does not error when there are no
        playbooks to return (cold-start / fresh-install scenario).
        """
        instance = openclaw_playbook_instance

        # Storage was cleaned before the test; search should gracefully return empty
        user_resp = instance.search_user_playbooks(
            SearchUserPlaybookRequest(
                query="does not exist",
                user_id=_ALPHA_USER,
                playbook_name=_PLAYBOOK_NAME,
            )
        )
        assert user_resp.success is True
        assert user_resp.user_playbooks == []

        agent_resp = instance.search_agent_playbooks(
            SearchAgentPlaybookRequest(
                query="does not exist",
                playbook_name=_PLAYBOOK_NAME,
            )
        )
        assert agent_resp.success is True
        assert agent_resp.agent_playbooks == []

    @skip_in_precommit
    def test_publish_with_minimal_interaction_does_not_crash(
        self,
        openclaw_playbook_instance: Reflexio,
    ) -> None:
        """A very short interaction (single turn) publishes without raising.

        OpenClaw may buffer single-turn interactions before the session ends.
        The pipeline must not error on minimal input even if no playbook is extracted.
        """
        instance = openclaw_playbook_instance

        single_turn = [InteractionData(role="user", content="Hello.")]

        resp = instance.publish_interaction(
            {
                "user_id": _ALPHA_USER,
                "interaction_data_list": single_turn,
                "agent_version": _AGENT_VERSION,
                "source": "openclaw",
            }
        )
        assert resp.success is True
        # The pipeline may or may not extract a playbook from one turn — both are valid.
        # What matters is it does not raise.

    @skip_in_precommit
    def test_get_user_playbooks_for_unknown_user_returns_empty(
        self,
        openclaw_playbook_instance: Reflexio,
    ) -> None:
        """Getting playbooks for a user_id that has never published returns an empty list.

        In the OpenClaw multi-instance scenario, a new agent installation may
        call get_user_playbooks before ever publishing interactions.
        """
        instance = openclaw_playbook_instance

        resp = instance.get_user_playbooks(
            GetUserPlaybooksRequest(
                user_id="brand-new-instance-never-seen-before",
                playbook_name=_PLAYBOOK_NAME,
            )
        )
        assert resp.success is True
        assert resp.user_playbooks == []

    @skip_in_precommit
    def test_get_agent_playbooks_returns_empty_before_aggregation(
        self,
        openclaw_playbook_instance: Reflexio,
    ) -> None:
        """Getting agent playbooks before any aggregation run returns an empty list.

        This matches the expected cold-start behaviour: search.js calls
        getAgentPlaybooks and must handle [] gracefully.
        """
        instance = openclaw_playbook_instance

        resp = instance.get_agent_playbooks(
            GetAgentPlaybooksRequest(
                playbook_name=_PLAYBOOK_NAME,
                agent_version=_AGENT_VERSION,
            )
        )
        assert resp.success is True
        assert resp.agent_playbooks == []
