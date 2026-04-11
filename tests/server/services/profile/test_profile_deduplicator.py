"""
Unit tests for ProfileDeduplicator.

Tests the deduplicator's responsibilities for:
- Pydantic output schema validation
- Profile deduplication with LLM and hybrid search
- Profile formatting for prompts
- Building deduplicated results
- Merging custom features
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest


# Disable mock mode for deduplicator tests so LLM mocks are actually used
@pytest.fixture(autouse=True)
def disable_mock_llm_response(monkeypatch):
    """Disable MOCK_LLM_RESPONSE env var so deduplicator tests use their own mocks."""
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)


from reflexio.models.api_schema.service_schemas import (
    ProfileTimeToLive,
    UserProfile,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.deduplication_utils import parse_item_id
from reflexio.server.services.profile.profile_deduplicator import (
    ProfileDeduplicationOutput,
    ProfileDeduplicator,
    ProfileDuplicateGroup,
)

# ===============================
# Fixtures
# ===============================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock(spec=LiteLLMClient)
    client.get_embeddings.return_value = [[0.1] * 10, [0.2] * 10, [0.3] * 10]
    return client


@pytest.fixture
def mock_request_context():
    """Create a mock request context with prompt manager and storage."""
    context = MagicMock(
        spec_set=["prompt_manager", "storage", "configurator", "org_id"]
    )
    context.prompt_manager = MagicMock()
    context.prompt_manager.render_prompt.return_value = "test prompt"
    context.storage = MagicMock()
    context.storage.search_user_profile.return_value = []
    # Set up configurator chain for model resolution
    mock_config = MagicMock()
    mock_config.api_key_config = None
    context.configurator.get_config.return_value = mock_config
    return context


@pytest.fixture
def mock_site_var_manager():
    """Mock the SiteVarManager to return model settings."""
    with patch("reflexio.server.services.deduplication_utils.SiteVarManager") as mock:
        instance = mock.return_value
        instance.get_site_var.return_value = {"default_generation_model_name": "gpt-4"}
        yield mock


@pytest.fixture
def sample_profiles():
    """Create sample UserProfile objects for testing."""
    timestamp = int(datetime.now(UTC).timestamp())
    return [
        UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id="test_user",
            content="User prefers dark mode for coding",
            last_modified_timestamp=timestamp,
            generated_from_request_id="req_1",
            profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
            source="extractor_a",
        ),
        UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id="test_user",
            content="User likes dark theme in their IDE",
            last_modified_timestamp=timestamp,
            generated_from_request_id="req_2",
            profile_time_to_live=ProfileTimeToLive.ONE_WEEK,
            source="extractor_b",
        ),
        UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id="test_user",
            content="User is a Python developer",
            last_modified_timestamp=timestamp,
            generated_from_request_id="req_3",
            profile_time_to_live=ProfileTimeToLive.ONE_YEAR,
            source="extractor_a",
        ),
    ]


# ===============================
# Test: Pydantic Models
# ===============================


class TestPydanticModels:
    """Tests for the Pydantic output schema models."""

    def test_duplicate_group_creation(self):
        """Test that ProfileDuplicateGroup can be created with valid data."""
        group = ProfileDuplicateGroup(
            item_ids=["NEW-0", "NEW-1", "EXISTING-0"],
            merged_content="User prefers dark mode",
            merged_time_to_live="one_month",
            reasoning="Both profiles are about dark mode preferences",
        )
        assert group.item_ids == ["NEW-0", "NEW-1", "EXISTING-0"]
        assert group.merged_content == "User prefers dark mode"
        assert group.merged_time_to_live == "one_month"

    def test_duplicate_group_forbids_extra_fields(self):
        """Test that ProfileDuplicateGroup allows extra fields at runtime (for LLM robustness)
        but forbids them in JSON schema (for LLM structured output)."""
        # extra="allow" means Pydantic accepts extra fields at runtime
        group = ProfileDuplicateGroup(
            item_ids=["NEW-0"],
            merged_content="test",
            merged_time_to_live="one_day",
            reasoning="test",
            extra_field="not allowed",
        )
        assert group.item_ids == ["NEW-0"]
        # JSON schema should forbid additional properties (used for LLM structured output)
        schema = ProfileDuplicateGroup.model_json_schema()
        assert schema.get("additionalProperties") is False

    def test_deduplication_output_creation(self):
        """Test that ProfileDeduplicationOutput can be created."""
        output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-0", "NEW-1"],
                    merged_content="merged",
                    merged_time_to_live="one_week",
                    reasoning="duplicates",
                )
            ],
            unique_ids=["NEW-2", "NEW-3"],
        )
        assert len(output.duplicate_groups) == 1
        assert output.unique_ids == ["NEW-2", "NEW-3"]

    def test_deduplication_output_empty_defaults(self):
        """Test that ProfileDeduplicationOutput has empty list defaults."""
        output = ProfileDeduplicationOutput()
        assert output.duplicate_groups == []
        assert output.unique_ids == []

    def test_deduplication_output_from_dict(self):
        """Test that ProfileDeduplicationOutput can be validated from dict."""
        data = {
            "duplicate_groups": [
                {
                    "item_ids": ["NEW-0", "NEW-1", "EXISTING-0"],
                    "merged_content": "test",
                    "merged_time_to_live": "one_day",
                    "reasoning": "reason",
                }
            ],
            "unique_ids": ["NEW-2"],
        }
        output = ProfileDeduplicationOutput.model_validate(data)
        assert len(output.duplicate_groups) == 1
        assert output.unique_ids == ["NEW-2"]

    def test_parse_item_id_valid(self):
        """Test parse_item_id with valid inputs."""
        assert parse_item_id("NEW-0") == ("NEW", 0)
        assert parse_item_id("EXISTING-1") == ("EXISTING", 1)
        assert parse_item_id("new-5") == ("NEW", 5)

    def test_parse_item_id_invalid(self):
        """Test parse_item_id returns None for invalid inputs."""
        assert parse_item_id("INVALID-0") is None
        assert parse_item_id("NOHYPHEN") is None
        assert parse_item_id("NEW-abc") is None


# ===============================
# Test: ProfileDeduplicator Init
# ===============================


class TestProfileDeduplicatorInit:
    """Tests for ProfileDeduplicator initialization."""

    def test_init_sets_attributes(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test that __init__ sets all required attributes."""
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        assert deduplicator.request_context == mock_request_context
        assert deduplicator.client == mock_llm_client
        assert deduplicator.model_name == "gpt-4"

    def test_init_uses_auto_detected_model_when_not_specified(
        self, mock_request_context, mock_llm_client, monkeypatch
    ):
        """Test that init falls back to auto-detected model if not in site var."""
        # Clear all provider keys so only OPENAI_API_KEY is detected
        for key in [
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
            "MINIMAX_API_KEY",
            "DASHSCOPE_API_KEY",
            "XAI_API_KEY",
            "MOONSHOT_API_KEY",
            "ZAI_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch(
            "reflexio.server.services.deduplication_utils.SiteVarManager"
        ) as mock:
            instance = mock.return_value
            instance.get_site_var.return_value = {}
            deduplicator = ProfileDeduplicator(
                request_context=mock_request_context,
                llm_client=mock_llm_client,
            )
            assert deduplicator.model_name == "gpt-5-mini"


# ===============================
# Test: Format Profiles For Prompt
# ===============================


class TestFormatProfilesForPrompt:
    """Tests for profile formatting for LLM prompt."""

    def test_format_profiles_basic(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that profiles are formatted correctly with NEW prefix."""
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._format_items_for_prompt(sample_profiles)

        assert "[NEW-0]" in result
        assert "[NEW-1]" in result
        assert "[NEW-2]" in result
        assert "User prefers dark mode for coding" in result
        assert "User likes dark theme in their IDE" in result
        assert "one_month" in result
        assert "one_week" in result
        assert "extractor_a" in result
        assert "extractor_b" in result

    def test_format_profiles_uses_ttl_value(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test formatting shows TTL value from profile."""
        timestamp = int(datetime.now(UTC).timestamp())
        profiles = [
            UserProfile(
                profile_id="1",
                user_id="user",
                content="test content",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req",
                profile_time_to_live=ProfileTimeToLive.ONE_QUARTER,
            )
        ]
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._format_items_for_prompt(profiles)
        assert "TTL: one_quarter" in result

    def test_format_profiles_with_missing_source(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test formatting with profiles that have no source."""
        timestamp = int(datetime.now(UTC).timestamp())
        profiles = [
            UserProfile(
                profile_id="1",
                user_id="user",
                content="test content",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req",
                source=None,
            )
        ]
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._format_items_for_prompt(profiles)
        assert "Source: unknown" in result

    def test_format_existing_profiles(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that existing profiles are formatted with EXISTING prefix."""
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._format_profiles_with_prefix(sample_profiles, "EXISTING")
        assert "[EXISTING-0]" in result
        assert "[EXISTING-1]" in result

    def test_format_empty_profiles(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test formatting empty profile list returns (None)."""
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._format_profiles_with_prefix([], "NEW")
        assert result == "(None)"


# ===============================
# Test: Merge Custom Features
# ===============================


class TestMergeCustomFeatures:
    """Tests for custom features merging."""

    def test_merge_custom_features_empty(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test merging when no profiles have custom features."""
        timestamp = int(datetime.now(UTC).timestamp())
        profiles = [
            UserProfile(
                profile_id="1",
                user_id="user",
                content="test",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req",
                custom_features=None,
            ),
            UserProfile(
                profile_id="2",
                user_id="user",
                content="test2",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req",
                custom_features=None,
            ),
        ]
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._merge_custom_features(profiles)
        assert result is None

    def test_merge_custom_features_single(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test merging when only one profile has custom features."""
        timestamp = int(datetime.now(UTC).timestamp())
        profiles = [
            UserProfile(
                profile_id="1",
                user_id="user",
                content="test",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req",
                custom_features={"key1": "value1"},
            ),
            UserProfile(
                profile_id="2",
                user_id="user",
                content="test2",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req",
                custom_features=None,
            ),
        ]
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._merge_custom_features(profiles)
        assert result == {"key1": "value1"}

    def test_merge_custom_features_multiple(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test merging custom features from multiple profiles."""
        timestamp = int(datetime.now(UTC).timestamp())
        profiles = [
            UserProfile(
                profile_id="1",
                user_id="user",
                content="test",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req",
                custom_features={"key1": "value1", "key2": "old_value"},
            ),
            UserProfile(
                profile_id="2",
                user_id="user",
                content="test2",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req",
                custom_features={"key2": "new_value", "key3": "value3"},
            ),
        ]
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._merge_custom_features(profiles)
        assert result == {"key1": "value1", "key2": "new_value", "key3": "value3"}


# ===============================
# Test: Build Deduplicated Results
# ===============================


class TestBuildDeduplicatedResults:
    """Tests for building deduplicated profile results."""

    def test_build_deduplicated_results_merges_duplicates(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that duplicates are merged into a single profile."""
        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-0", "NEW-1"],
                    merged_content="User prefers dark mode in their IDE",
                    merged_time_to_live="one_month",
                    reasoning="Both about dark mode preferences",
                )
            ],
            unique_ids=["NEW-2"],
        )

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result_profiles, delete_ids, superseded = (
            deduplicator._build_deduplicated_results(
                new_profiles=sample_profiles,
                existing_profiles=[],
                dedup_output=dedup_output,
                user_id="test_user",
                request_id="test_request",
            )
        )

        assert len(result_profiles) == 2  # 1 merged + 1 unique
        assert len(delete_ids) == 0
        assert len(superseded) == 0

        # Find the merged profile
        merged_profile = next(
            (
                p
                for p in result_profiles
                if p.content == "User prefers dark mode in their IDE"
            ),
            None,
        )
        assert merged_profile is not None
        assert merged_profile.profile_time_to_live == ProfileTimeToLive.ONE_MONTH

    def test_build_deduplicated_results_preserves_unique(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that unique profiles are preserved."""
        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[],
            unique_ids=["NEW-0", "NEW-1", "NEW-2"],
        )

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result_profiles, delete_ids, superseded = (
            deduplicator._build_deduplicated_results(
                new_profiles=sample_profiles,
                existing_profiles=[],
                dedup_output=dedup_output,
                user_id="test_user",
                request_id="test_request",
            )
        )

        assert len(result_profiles) == 3

    def test_build_deduplicated_results_handles_invalid_ttl(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that invalid TTL from LLM falls back to template TTL."""
        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-0", "NEW-1"],
                    merged_content="merged content",
                    merged_time_to_live="invalid_ttl",
                    reasoning="test",
                )
            ],
            unique_ids=["NEW-2"],
        )

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result_profiles, _, _ = deduplicator._build_deduplicated_results(
            new_profiles=sample_profiles,
            existing_profiles=[],
            dedup_output=dedup_output,
            user_id="test_user",
            request_id="test_request",
        )

        merged_profile = next(
            (p for p in result_profiles if p.content == "merged content"),
            None,
        )
        assert merged_profile is not None
        # Should fall back to template profile's TTL (first profile in group)
        assert merged_profile.profile_time_to_live == ProfileTimeToLive.ONE_MONTH

    def test_build_deduplicated_results_handles_unmentioned_profiles(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that profiles not mentioned by LLM are added as-is."""
        # LLM only mentions indices 0 and 1, not 2
        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-0", "NEW-1"],
                    merged_content="merged",
                    merged_time_to_live="one_week",
                    reasoning="test",
                )
            ],
            unique_ids=[],  # LLM forgot to mention index 2
        )

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result_profiles, _, _ = deduplicator._build_deduplicated_results(
            new_profiles=sample_profiles,
            existing_profiles=[],
            dedup_output=dedup_output,
            user_id="test_user",
            request_id="test_request",
        )

        # Should still include all profiles (1 merged + 1 unmentioned)
        assert len(result_profiles) == 2

    def test_build_deduplicated_results_collects_existing_to_delete(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that existing profiles marked for deletion are collected."""
        timestamp = int(datetime.now(UTC).timestamp())
        existing_profile = UserProfile(
            profile_id="existing_1",
            user_id="test_user",
            content="Old dark mode preference",
            last_modified_timestamp=timestamp,
            generated_from_request_id="old_req",
        )

        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-0", "EXISTING-0"],
                    merged_content="User prefers dark mode (updated)",
                    merged_time_to_live="one_month",
                    reasoning="New profile supersedes existing",
                )
            ],
            unique_ids=["NEW-1", "NEW-2"],
        )

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result_profiles, delete_ids, superseded = (
            deduplicator._build_deduplicated_results(
                new_profiles=sample_profiles,
                existing_profiles=[existing_profile],
                dedup_output=dedup_output,
                user_id="test_user",
                request_id="test_request",
            )
        )

        assert len(delete_ids) == 1
        assert delete_ids[0] == "existing_1"
        assert len(superseded) == 1
        assert superseded[0].profile_id == "existing_1"


# ===============================
# Test: Deduplicate Main Method
# ===============================


class TestDeduplicate:
    """Tests for the main deduplicate() method."""

    def test_deduplicate_returns_original_when_empty(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
    ):
        """Test that empty input returns empty output."""
        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        profiles, delete_ids, superseded = deduplicator.deduplicate(
            new_profiles=[],
            user_id="test_user",
            request_id="test_request",
        )

        assert profiles == []
        assert delete_ids == []
        assert superseded == []

    def test_deduplicate_returns_original_when_no_duplicates_found(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that original profiles are returned when LLM finds no duplicates."""
        mock_llm_client.generate_chat_response.return_value = (
            ProfileDeduplicationOutput(
                duplicate_groups=[],
                unique_ids=["NEW-0", "NEW-1", "NEW-2"],
            )
        )

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        profiles, delete_ids, superseded = deduplicator.deduplicate(
            new_profiles=sample_profiles,
            user_id="test_user",
            request_id="test_request",
        )

        assert profiles == sample_profiles
        assert delete_ids == []
        assert superseded == []

    def test_deduplicate_returns_original_when_llm_fails(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that original profiles are returned when LLM call fails."""
        mock_llm_client.generate_chat_response.side_effect = Exception("LLM Error")

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        profiles, delete_ids, superseded = deduplicator.deduplicate(
            new_profiles=sample_profiles,
            user_id="test_user",
            request_id="test_request",
        )

        assert profiles == sample_profiles
        assert delete_ids == []
        assert superseded == []

    def test_deduplicate_merges_duplicates(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test that duplicates are properly merged."""
        mock_llm_client.generate_chat_response.return_value = (
            ProfileDeduplicationOutput(
                duplicate_groups=[
                    ProfileDuplicateGroup(
                        item_ids=["NEW-0", "NEW-1"],
                        merged_content="User prefers dark mode",
                        merged_time_to_live="one_month",
                        reasoning="Both about dark mode",
                    )
                ],
                unique_ids=["NEW-2"],
            )
        )

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        profiles, delete_ids, superseded = deduplicator.deduplicate(
            new_profiles=sample_profiles,
            user_id="test_user",
            request_id="test_request",
        )

        # Should have 2 profiles: 1 merged + 1 unique
        assert len(profiles) == 2
        assert len(delete_ids) == 0

    def test_deduplicate_with_existing_profiles_to_delete(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """Test deduplication that supersedes existing profiles."""
        timestamp = int(datetime.now(UTC).timestamp())
        existing_profile = UserProfile(
            profile_id="existing_1",
            user_id="test_user",
            content="Old dark mode preference",
            last_modified_timestamp=timestamp,
            generated_from_request_id="old_req",
        )

        # Mock storage to return existing profile via hybrid search
        mock_request_context.storage.search_user_profile.return_value = [
            existing_profile
        ]

        mock_llm_client.generate_chat_response.return_value = (
            ProfileDeduplicationOutput(
                duplicate_groups=[
                    ProfileDuplicateGroup(
                        item_ids=["NEW-0", "EXISTING-0"],
                        merged_content="User prefers dark mode (updated)",
                        merged_time_to_live="one_month",
                        reasoning="New supersedes existing",
                    )
                ],
                unique_ids=["NEW-1", "NEW-2"],
            )
        )

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        profiles, delete_ids, superseded = deduplicator.deduplicate(
            new_profiles=sample_profiles,
            user_id="test_user",
            request_id="test_request",
        )

        assert len(profiles) == 3  # 1 merged + 2 unique
        assert len(delete_ids) == 1
        assert delete_ids[0] == "existing_1"
        assert len(superseded) == 1


# ===============================
# Test: Integration
# ===============================


class TestIntegration:
    """Integration tests for the complete deduplication flow."""

    def test_full_deduplication_flow(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
    ):
        """Test a complete deduplication flow with realistic data."""
        timestamp = int(datetime.now(UTC).timestamp())

        # Create profiles from different extractors with duplicates
        new_profiles = [
            UserProfile(
                profile_id="p1",
                user_id="user",
                content="User works in finance industry",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req1",
                profile_time_to_live=ProfileTimeToLive.ONE_YEAR,
                source="industry_extractor",
                custom_features={"sector": "finance"},
            ),
            UserProfile(
                profile_id="p2",
                user_id="user",
                content="User is in the financial services sector",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req2",
                profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
                source="job_extractor",
                custom_features={"job_type": "analyst"},
            ),
            UserProfile(
                profile_id="p3",
                user_id="user",
                content="User prefers Python programming",
                last_modified_timestamp=timestamp,
                generated_from_request_id="req3",
                profile_time_to_live=ProfileTimeToLive.INFINITY,
                source="tech_extractor",
            ),
        ]

        mock_llm_client.generate_chat_response.return_value = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-0", "NEW-1"],
                    merged_content="User works in the financial services industry",
                    merged_time_to_live="one_year",
                    reasoning="Both profiles describe the user's industry as finance/financial services",
                )
            ],
            unique_ids=["NEW-2"],
        )

        deduplicator = ProfileDeduplicator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result_profiles, delete_ids, superseded = deduplicator.deduplicate(
            new_profiles=new_profiles,
            user_id="user",
            request_id="test_request",
        )

        # Verify structure
        assert len(result_profiles) == 2
        assert len(delete_ids) == 0

        # Find merged profile
        merged = next(
            (p for p in result_profiles if "financial services industry" in p.content),
            None,
        )
        assert merged is not None
        assert merged.user_id == "user"
        assert merged.profile_time_to_live == ProfileTimeToLive.ONE_YEAR
        # Custom features should be merged
        assert merged.custom_features == {"sector": "finance", "job_type": "analyst"}

        # Find unique profile
        unique = next((p for p in result_profiles if "Python" in p.content), None)
        assert unique is not None
        assert unique.content == "User prefers Python programming"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
