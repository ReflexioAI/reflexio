"""
Unit tests for ProfileConsolidator.

Tests the consolidator's responsibilities for:
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


# Disable mock mode for consolidator tests so LLM mocks are actually used
@pytest.fixture(autouse=True)
def disable_mock_llm_response(monkeypatch):
    """Disable MOCK_LLM_RESPONSE env var so consolidator tests use their own mocks."""
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)


from reflexio.models.api_schema.service_schemas import (
    ProfileTimeToLive,
    UserProfile,
)
from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    StructuredOutputRepairError,
)
from reflexio.server.services.deduplication_utils import parse_item_id
from reflexio.server.services.profile.components.consolidator import (
    ProfileConsolidator,
    ProfileDeduplicationOutput,
    ProfileDeletionDirective,
    ProfileDuplicateGroup,
    _dedup_failure_kind,
    _format_profile_timestamp,
    validate_profile_dedup_output,
)

# ===============================
# Fixtures
# ===============================


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock(spec=LiteLLMClient)

    def _generate_with_provenance(*args, **kwargs):
        value = client.generate_chat_response(*args, **kwargs)
        if isinstance(value, CompletionResult):
            return value
        return CompletionResult(value=value, provenance=ModelProvenance())

    client.generate_chat_response_with_provenance.side_effect = (
        _generate_with_provenance
    )
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
            extra_field="not allowed",  # pyright: ignore[reportCallIssue]
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
        assert output.deletions == []

    def test_deletion_directive_creation(self):
        """Test that ProfileDeletionDirective can be created with valid data."""
        directive = ProfileDeletionDirective(
            new_id="NEW-0",
            existing_ids=["EXISTING-0", "EXISTING-1"],
            reasoning="User asked to forget this topic",
        )
        assert directive.new_id == "NEW-0"
        assert directive.existing_ids == ["EXISTING-0", "EXISTING-1"]
        assert directive.reasoning == "User asked to forget this topic"

    def test_deletion_directive_json_schema_forbids_extra(self):
        """Test that ProfileDeletionDirective's JSON schema forbids additional properties."""
        schema = ProfileDeletionDirective.model_json_schema()
        assert schema.get("additionalProperties") is False

    def test_deduplication_output_with_deletions(self):
        """Test that ProfileDeduplicationOutput accepts deletions."""
        output = ProfileDeduplicationOutput(
            duplicate_groups=[],
            unique_ids=[],
            deletions=[
                ProfileDeletionDirective(
                    new_id="NEW-0",
                    existing_ids=["EXISTING-0"],
                    reasoning="deletion request",
                )
            ],
        )
        assert len(output.deletions) == 1
        assert output.deletions[0].new_id == "NEW-0"

    def test_deduplication_output_deletions_from_dict(self):
        """Test that ProfileDeduplicationOutput with deletions validates from dict."""
        data = {
            "duplicate_groups": [],
            "unique_ids": [],
            "deletions": [
                {
                    "new_id": "NEW-0",
                    "existing_ids": ["EXISTING-0"],
                    "reasoning": "forget request",
                }
            ],
        }
        output = ProfileDeduplicationOutput.model_validate(data)
        assert len(output.deletions) == 1
        assert output.deletions[0].existing_ids == ["EXISTING-0"]

    def test_deduplication_output_from_dict(self):
        """Test that ProfileDeduplicationOutput can be validated from dict."""
        data = {
            "duplicate_groups": [
                {
                    "item_ids": ["NEW-0", "NEW-1", "EXISTING-0"],
                    "merged_content": "test",
                    "merged_time_to_live": "one_day",
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

    def test_parse_item_id_strips_prompt_brackets(self):
        """Weak models echo the rendered label ``[NEW-0]``; tolerate it."""
        assert parse_item_id("[NEW-0]") == ("NEW", 0)
        assert parse_item_id("[EXISTING-3]") == ("EXISTING", 3)
        assert parse_item_id(" [new-5] ") == ("NEW", 5)

    def test_parse_item_id_invalid(self):
        """Test parse_item_id returns None for invalid inputs."""
        assert parse_item_id("INVALID-0") is None
        assert parse_item_id("NOHYPHEN") is None
        assert parse_item_id("NEW-abc") is None


# ===============================
# Test: ProfileConsolidator Init
# ===============================


class TestProfileConsolidatorInit:
    """Tests for ProfileConsolidator initialization."""

    def test_init_sets_attributes(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test that __init__ sets all required attributes."""
        deduplicator = ProfileConsolidator(
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
            "CLAUDE_SMART_USE_LOCAL_CLI",
            "CLAUDE_SMART_USE_LOCAL_EMBEDDING",
        ]:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with patch(
            "reflexio.server.services.deduplication_utils.SiteVarManager"
        ) as mock:
            instance = mock.return_value
            instance.get_site_var.return_value = {}
            deduplicator = ProfileConsolidator(
                request_context=mock_request_context,
                llm_client=mock_llm_client,
            )
            assert deduplicator.model_name == "gpt-5.5"


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
        deduplicator = ProfileConsolidator(
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
        deduplicator = ProfileConsolidator(
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
        deduplicator = ProfileConsolidator(
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
        deduplicator = ProfileConsolidator(
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
        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._format_profiles_with_prefix([], "NEW")
        assert result == "(None)"

    def test_format_profiles_includes_last_modified_utc(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test that formatted profiles include the last-modified timestamp in UTC."""
        # 1704067200 == 2024-01-01 00:00:00 UTC
        profiles = [
            UserProfile(
                profile_id="1",
                user_id="user",
                content="test content",
                last_modified_timestamp=1704067200,
                generated_from_request_id="req",
                profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
                source="extractor_a",
            )
        ]
        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        result = deduplicator._format_profiles_with_prefix(profiles, "NEW")
        assert "Last Modified: 2024-01-01 00:00 UTC" in result

    def test_format_profiles_timestamp_fallback_on_invalid(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Test formatting degrades gracefully when the timestamp is out of range."""
        # Absurdly large value that overflows datetime.fromtimestamp on every
        # supported platform, but is still a valid ``int`` for the Pydantic
        # model field.
        profiles = [
            UserProfile(
                profile_id="1",
                user_id="user",
                content="test content",
                last_modified_timestamp=99999999999999999,
                generated_from_request_id="req",
                profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
                source="extractor_a",
            )
        ]
        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        # Must not raise.
        result = deduplicator._format_profiles_with_prefix(profiles, "NEW")
        assert "Last Modified: unknown" in result

    def test_format_profile_timestamp_helper_happy_path(self):
        """The helper formats a valid timestamp identically to the old inline call."""
        assert _format_profile_timestamp(1704067200) == "2024-01-01 00:00 UTC"

    def test_format_profile_timestamp_helper_fallback(self):
        """The helper returns the sentinel when the timestamp is out of range."""
        assert _format_profile_timestamp(99999999999999999) == "unknown"


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
        deduplicator = ProfileConsolidator(
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
        deduplicator = ProfileConsolidator(
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
        deduplicator = ProfileConsolidator(
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
                )
            ],
            unique_ids=["NEW-2"],
        )

        deduplicator = ProfileConsolidator(
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
        assert uuid.UUID(merged_profile.profile_id).version == 4
        assert str(uuid.UUID(merged_profile.profile_id)) == merged_profile.profile_id
        assert merged_profile.profile_time_to_live == ProfileTimeToLive.ONE_MONTH

    @pytest.mark.parametrize(
        ("item_ids", "expected_source_ids"),
        [
            (["NEW-0", "NEW-1"], [101, 102, 103]),
            (["EXISTING-0", "EXISTING-1"], [102, 201, 202, 203]),
            (["NEW-0", "EXISTING-0"], [101, 102, 201]),
        ],
        ids=["new-only", "existing-only", "mixed"],
    )
    def test_merged_profile_preserves_source_interaction_ids(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
        item_ids,
        expected_source_ids,
    ):
        """Merged profiles retain the ordered union of member provenance."""
        sample_profiles[0].source_interaction_ids = [101, 102]
        sample_profiles[1].source_interaction_ids = [102, 103]
        existing_profiles = [
            UserProfile(
                profile_id="existing-0",
                user_id="test_user",
                content="Existing profile zero",
                last_modified_timestamp=int(datetime.now(UTC).timestamp()),
                generated_from_request_id="existing-request",
                source_interaction_ids=[102, 201],
            ),
            UserProfile(
                profile_id="existing-1",
                user_id="test_user",
                content="Existing profile one",
                last_modified_timestamp=int(datetime.now(UTC).timestamp()),
                generated_from_request_id="existing-request",
                source_interaction_ids=[201, 202, 203],
            ),
        ]
        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=item_ids,
                    merged_content="Merged profile",
                    merged_time_to_live="one_month",
                )
            ],
        )

        result_profiles, _, _ = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )._build_deduplicated_results(
            new_profiles=sample_profiles,
            existing_profiles=existing_profiles,
            dedup_output=dedup_output,
            user_id="test_user",
            request_id="test_request",
        )

        merged_profile = next(
            profile
            for profile in result_profiles
            if profile.content == "Merged profile"
        )
        assert merged_profile.source_interaction_ids == expected_source_ids

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

        deduplicator = ProfileConsolidator(
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
                )
            ],
            unique_ids=["NEW-2"],
        )

        deduplicator = ProfileConsolidator(
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
                )
            ],
            unique_ids=[],  # LLM forgot to mention index 2
        )

        deduplicator = ProfileConsolidator(
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
                )
            ],
            unique_ids=["NEW-1", "NEW-2"],
        )

        deduplicator = ProfileConsolidator(
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

    def test_build_deduplicated_results_handles_deletion_directive(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """A deletion directive erases the EXISTING profile without writing a replacement.

        This is the core bug fix: "forget that I am interested in X" used to
        produce a merged "Previously interested in X, but requested removal"
        profile. With the deletion channel, the NEW directive is consumed and
        the EXISTING profile is deleted outright.
        """
        timestamp = int(datetime.now(UTC).timestamp())
        existing_profile = UserProfile(
            profile_id="existing_old_interest",
            user_id="test_user",
            content="User is interested in self-improving agents",
            last_modified_timestamp=timestamp,
            generated_from_request_id="old_req",
        )

        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[],
            unique_ids=["NEW-1", "NEW-2"],
            deletions=[
                ProfileDeletionDirective(
                    new_id="NEW-0",
                    existing_ids=["EXISTING-0"],
                    reasoning=(
                        "NEW-0 is a meta-request to forget EXISTING-0; "
                        "not a fact about the user."
                    ),
                )
            ],
        )

        deduplicator = ProfileConsolidator(
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

        # EXISTING profile is marked for deletion.
        assert delete_ids == ["existing_old_interest"]
        assert len(superseded) == 1
        assert superseded[0].profile_id == "existing_old_interest"

        # NEW-0 (the directive) was consumed — not re-added by the safety fallback.
        assert all(p.content != sample_profiles[0].content for p in result_profiles), (
            "Deletion directive NEW profile should not appear in result_profiles"
        )

        # Only NEW-1 and NEW-2 (the unrelated unique profiles) remain.
        assert len(result_profiles) == 2
        assert {p.content for p in result_profiles} == {
            sample_profiles[1].content,
            sample_profiles[2].content,
        }

    def test_build_deduplicated_results_deletion_directive_no_match(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """A deletion directive with empty existing_ids still consumes the NEW.

        If the LLM emits a deletion directive but matches no EXISTING profile,
        the NEW profile must still be suppressed — a meta-statement like
        "Requested removal of X" is not a fact worth storing on its own.
        """
        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[],
            unique_ids=["NEW-1", "NEW-2"],
            deletions=[
                ProfileDeletionDirective(
                    new_id="NEW-0",
                    existing_ids=[],
                    reasoning="No matching existing profile found.",
                )
            ],
        )

        deduplicator = ProfileConsolidator(
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

        assert delete_ids == []
        assert superseded == []
        # NEW-0 must not survive into result_profiles.
        assert all(p.content != sample_profiles[0].content for p in result_profiles)
        assert len(result_profiles) == 2


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
        deduplicator = ProfileConsolidator(
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

        deduplicator = ProfileConsolidator(
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

        deduplicator = ProfileConsolidator(
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
                    )
                ],
                unique_ids=["NEW-2"],
            )
        )

        deduplicator = ProfileConsolidator(
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
                    )
                ],
                unique_ids=["NEW-1", "NEW-2"],
            )
        )

        deduplicator = ProfileConsolidator(
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

    def test_deduplicate_applies_deletions_when_no_duplicate_groups(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
    ):
        """A deletion-only LLM response must still erase the EXISTING profile.

        Regression guard: the public `deduplicate()` used to short-circuit when
        `duplicate_groups` was empty, which silently dropped deletion directives
        and returned the 'Requested removal of ...' NEW profile as a new fact —
        the exact zombie-profile failure the deletion channel was meant to fix.
        """
        timestamp = int(datetime.now(UTC).timestamp())
        existing_profile = UserProfile(
            profile_id="existing_forgettable",
            user_id="test_user",
            content="User is interested in self-improving agents",
            last_modified_timestamp=timestamp,
            generated_from_request_id="old_req",
        )
        directive_profile = UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id="test_user",
            content=(
                "Requested removal of interest in self-improving agents "
                "from stored profiles"
            ),
            last_modified_timestamp=timestamp,
            generated_from_request_id="req_directive",
            profile_time_to_live=ProfileTimeToLive.ONE_DAY,
            source="extractor_a",
        )

        mock_request_context.storage.search_user_profile.return_value = [
            existing_profile
        ]
        mock_llm_client.generate_chat_response.return_value = (
            ProfileDeduplicationOutput(
                duplicate_groups=[],
                unique_ids=[],
                deletions=[
                    ProfileDeletionDirective(
                        new_id="NEW-0",
                        existing_ids=["EXISTING-0"],
                        reasoning="Meta-request to forget EXISTING-0.",
                    )
                ],
            )
        )

        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        profiles, delete_ids, superseded = deduplicator.deduplicate(
            new_profiles=[directive_profile],
            user_id="test_user",
            request_id="test_request",
        )

        assert delete_ids == ["existing_forgettable"]
        assert len(superseded) == 1
        assert superseded[0].profile_id == "existing_forgettable"
        # The directive must be consumed — not leak back as a stored fact.
        assert profiles == []

    def test_deduplicate_strips_markers_on_llm_exception(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
    ):
        """When the LLM call raises, fallback must strip canonical deletion markers.

        Regression guard: if the LLM fails, returning `new_profiles` verbatim
        would persist "Requested removal of …" markers as regular facts — the
        exact zombie-profile outcome the deletion-directive channel was built
        to prevent. The fallback must suppress markers while preserving
        ordinary profiles.
        """
        timestamp = int(datetime.now(UTC).timestamp())
        ordinary = UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id="test_user",
            content="User prefers dark mode",
            last_modified_timestamp=timestamp,
            generated_from_request_id="req_ok",
            profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
            source="extractor_a",
        )
        marker = UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id="test_user",
            content=(
                "Requested removal of interest in self-improving agents "
                "from stored profiles"
            ),
            last_modified_timestamp=timestamp,
            generated_from_request_id="req_forget",
            profile_time_to_live=ProfileTimeToLive.ONE_DAY,
            source="extractor_a",
        )

        mock_request_context.storage.search_user_profile.return_value = []
        mock_llm_client.generate_chat_response.side_effect = RuntimeError(
            "LLM unavailable"
        )

        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        profiles, delete_ids, superseded = deduplicator.deduplicate(
            new_profiles=[ordinary, marker],
            user_id="test_user",
            request_id="test_request",
        )

        assert delete_ids == []
        assert superseded == []
        assert [p.profile_id for p in profiles] == [ordinary.profile_id]

    def test_deduplicate_strips_markers_on_empty_output(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
    ):
        """Empty dedup output (no groups, no deletions) still strips markers.

        If the LLM returns nothing to act on but a marker profile is present in
        `new_profiles`, the fallback must drop the marker rather than persist
        it as a fact.
        """
        timestamp = int(datetime.now(UTC).timestamp())
        ordinary = UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id="test_user",
            content="User prefers dark mode",
            last_modified_timestamp=timestamp,
            generated_from_request_id="req_ok",
            profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
            source="extractor_a",
        )
        marker = UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id="test_user",
            content="Requested removal of preference for tabs over spaces",
            last_modified_timestamp=timestamp,
            generated_from_request_id="req_forget",
            profile_time_to_live=ProfileTimeToLive.ONE_DAY,
            source="extractor_a",
        )

        mock_request_context.storage.search_user_profile.return_value = []
        mock_llm_client.generate_chat_response.return_value = (
            ProfileDeduplicationOutput(
                duplicate_groups=[],
                unique_ids=[],
                deletions=[],
            )
        )

        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        profiles, delete_ids, superseded = deduplicator.deduplicate(
            new_profiles=[ordinary, marker],
            user_id="test_user",
            request_id="test_request",
        )

        assert delete_ids == []
        assert superseded == []
        assert [p.profile_id for p in profiles] == [ordinary.profile_id]


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

        mock_llm_client.generate_chat_response.return_value = (
            ProfileDeduplicationOutput(
                duplicate_groups=[
                    ProfileDuplicateGroup(
                        item_ids=["NEW-0", "NEW-1"],
                        merged_content="User works in the financial services industry",
                        merged_time_to_live="one_year",
                    )
                ],
                unique_ids=["NEW-2"],
            )
        )

        deduplicator = ProfileConsolidator(
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


# ===============================
# Test: Rerun mode status_filter behavior (R1 fix)
# ===============================


class TestRetrieveExistingProfilesStatusFilter:
    """Tests for the status_filter selection in _retrieve_existing_profiles.

    These cover the R1 fix: when output_pending_status=True (rerun preview
    mode), the consolidator must NOT search against existing CURRENT
    profiles, otherwise newly-extracted profiles get flagged as duplicates
    and the downstream deletion step in
    ProfileGenerationService._process_results collapses the user's CURRENT
    profile set to zero during what is supposed to be a non-destructive
    preview.
    """

    def _make_profile(self, content: str, profile_id: str | None = None) -> UserProfile:
        return UserProfile(
            profile_id=profile_id or str(uuid.uuid4()),
            user_id="user",
            content=content,
            last_modified_timestamp=int(datetime.now(UTC).timestamp()),
            generated_from_request_id="req",
            profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
        )

    def test_normal_mode_searches_current_profiles(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Default (output_pending_status=False) searches CURRENT profiles."""
        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        new_profiles = [self._make_profile("User likes dark mode")]

        deduplicator._retrieve_existing_profiles(new_profiles, "user")

        # Verify storage was called with status_filter=[None] (CURRENT)
        assert mock_request_context.storage.search_user_profile.call_count == 1
        call_kwargs = mock_request_context.storage.search_user_profile.call_args.kwargs
        assert call_kwargs["status_filter"] == [None]
        search_request = (
            mock_request_context.storage.search_user_profile.call_args.args[0]
        )
        assert search_request.threshold == 0.4

    def test_rerun_mode_searches_pending_profiles_only(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """output_pending_status=True searches PENDING profiles, not CURRENT."""
        from reflexio.models.api_schema.service_schemas import Status

        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
            output_pending_status=True,
        )
        new_profiles = [self._make_profile("User likes dark mode")]

        deduplicator._retrieve_existing_profiles(new_profiles, "user")

        # Verify storage was called with status_filter=[Status.PENDING]
        assert mock_request_context.storage.search_user_profile.call_count == 1
        call_kwargs = mock_request_context.storage.search_user_profile.call_args.kwargs
        assert call_kwargs["status_filter"] == [Status.PENDING]

    def test_rerun_mode_default_init_value_is_false(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """ProfileConsolidator.output_pending_status defaults to False to
        preserve the pre-R1-fix behavior for normal extraction callers."""
        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )
        assert deduplicator.output_pending_status is False

    def test_fallback_embeddings_use_storage_model_and_query_prefix(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """Fallback batch embeddings stay compatible with stored profile vectors."""
        mock_request_context.storage._get_embedding = None
        mock_request_context.storage.embedding_model_name = "local/test-embedding-model"
        mock_request_context.storage.embedding_dimensions = 768
        mock_llm_client.get_embeddings.return_value = [[0.1] * 768]

        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )

        deduplicator._retrieve_existing_profiles(
            [self._make_profile("User likes dark mode")],
            "user",
        )

        mock_llm_client.get_embeddings.assert_called_once_with(
            ["User likes dark mode"],
            model="local/test-embedding-model",
            dimensions=768,
        )

    def test_rerun_mode_dedup_against_existing_pending(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        """When in rerun mode and an existing PENDING profile is similar to
        a new one, the LLM call sees the PENDING profile in EXISTING set --
        so dedup against pending is preserved (preview semantics)."""
        from reflexio.models.api_schema.service_schemas import Status

        existing_pending = self._make_profile("User uses dark theme", "p-existing-1")
        existing_pending.status = Status.PENDING

        # Storage returns the existing PENDING profile when searched
        mock_request_context.storage.search_user_profile.return_value = [
            existing_pending
        ]

        deduplicator = ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
            output_pending_status=True,
        )

        new_profiles = [self._make_profile("User likes dark mode", "p-new-1")]
        existing = deduplicator._retrieve_existing_profiles(new_profiles, "user")

        # Storage was filtered to PENDING
        call_kwargs = mock_request_context.storage.search_user_profile.call_args.kwargs
        assert call_kwargs["status_filter"] == [Status.PENDING]
        # The PENDING profile was retrieved as a candidate for dedup
        assert len(existing) == 1
        assert existing[0].profile_id == "p-existing-1"
        assert existing[0].status == Status.PENDING


# ===============================
# Test: Supersede-without-replacement invariant
# ===============================


def _make_existing(content: str, profile_id: str) -> UserProfile:
    """Build an EXISTING-side profile for group-resolution tests."""
    return UserProfile(
        profile_id=profile_id,
        user_id="test_user",
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="req_existing",
        profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
        source="extractor_a",
    )


class TestSupersedeAlwaysHasReplacement:
    """
    Contract tests for the core dedup invariant: a duplicate group either
    writes a merged replacement profile or marks nothing for deletion.

    Regression guard for the bug where a group's EXISTING members were queued
    for supersede (and its NEW members marked handled) *before* the merge
    template was resolved, so an unresolvable group deleted stored profiles
    with no replacement written and stranded the NEW facts too.
    """

    @pytest.fixture
    def existing_profiles(self):
        return [
            _make_existing("User uses dark theme in the IDE", "p-existing-0"),
            _make_existing("User codes in Python daily", "p-existing-1"),
        ]

    @staticmethod
    def _consolidator(mock_request_context, mock_llm_client):
        return ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )

    @pytest.mark.parametrize(
        ("item_ids", "expect_supersede"),
        [
            # EXISTING-only group: legal per the prompt ("ANY mix of NEW and
            # EXISTING"). Collapses into a merged replacement.
            (["EXISTING-0", "EXISTING-1"], True),
            # Out-of-range NEW falls back to the EXISTING member as template.
            (["NEW-7", "EXISTING-0"], True),
            # Nothing resolvable: must not supersede anything.
            (["NEW-7", "EXISTING-9"], False),
            (["garbage", "also-bad"], False),
            # Mixed group: unchanged behavior.
            (["NEW-0", "EXISTING-0"], True),
        ],
    )
    def test_group_never_supersedes_without_writing_a_replacement(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        sample_profiles,
        existing_profiles,
        item_ids,
        expect_supersede,
    ):
        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=item_ids,
                    merged_content="MERGED",
                    merged_time_to_live="one_month",
                )
            ],
        )

        result_profiles, delete_ids, superseded = self._consolidator(
            mock_request_context, mock_llm_client
        )._build_deduplicated_results(
            new_profiles=sample_profiles,
            existing_profiles=existing_profiles,
            dedup_output=dedup_output,
            user_id="test_user",
            request_id="test_request",
        )

        merged = [p for p in result_profiles if p.content == "MERGED"]

        # The invariant: superseding anything requires a written replacement.
        if delete_ids:
            assert merged, (
                f"group {item_ids} superseded {delete_ids} without writing a "
                "replacement profile"
            )
        assert len(delete_ids) == len(superseded)
        assert bool(delete_ids) is expect_supersede

        # No NEW profile may be silently dropped, whatever happened to the group.
        assert len(result_profiles) >= len(sample_profiles) - len(item_ids)
        for idx, profile in enumerate(sample_profiles):
            consumed_by_group = f"NEW-{idx}" in item_ids
            if not consumed_by_group:
                assert profile in result_profiles, (
                    f"NEW-{idx} was not part of group {item_ids} but went missing"
                )

    def test_existing_only_group_carries_metadata_into_the_merge(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        existing_profiles,
    ):
        """An EXISTING-only merge must not drop custom_features/extractor_names."""
        existing_profiles[0].custom_features = {"theme": "dark"}
        existing_profiles[0].extractor_names = ["extractor_a"]
        existing_profiles[1].extractor_names = ["extractor_b"]

        dedup_output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["EXISTING-0", "EXISTING-1"],
                    merged_content="MERGED",
                    merged_time_to_live="one_year",
                )
            ],
        )

        result_profiles, delete_ids, _ = self._consolidator(
            mock_request_context, mock_llm_client
        )._build_deduplicated_results(
            new_profiles=[],
            existing_profiles=existing_profiles,
            dedup_output=dedup_output,
            user_id="test_user",
            request_id="test_request",
        )

        assert len(result_profiles) == 1
        merged = result_profiles[0]
        assert merged.content == "MERGED"
        assert merged.custom_features == {"theme": "dark"}
        assert merged.extractor_names == ["extractor_a", "extractor_b"]
        assert merged.profile_time_to_live == ProfileTimeToLive.ONE_YEAR
        # Both originals superseded, one replacement written.
        assert set(delete_ids) == {"p-existing-0", "p-existing-1"}

    def test_deletion_directives_still_supersede_without_replacement(
        self,
        mock_request_context,
        mock_llm_client,
        mock_site_var_manager,
        existing_profiles,
    ):
        """
        The invariant is scoped to duplicate_groups. A deletion directive is
        an intentional erase-with-no-replacement and must stay that way.
        """
        directive = UserProfile(
            profile_id=str(uuid.uuid4()),
            user_id="test_user",
            content="Requested removal of dark theme preference",
            last_modified_timestamp=int(datetime.now(UTC).timestamp()),
            generated_from_request_id="req_1",
            profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
            source="extractor_a",
        )
        dedup_output = ProfileDeduplicationOutput(
            deletions=[
                ProfileDeletionDirective(
                    new_id="NEW-0",
                    existing_ids=["EXISTING-0"],
                    reasoning="meta-request to forget",
                )
            ],
        )

        result_profiles, delete_ids, superseded = self._consolidator(
            mock_request_context, mock_llm_client
        )._build_deduplicated_results(
            new_profiles=[directive],
            existing_profiles=existing_profiles,
            dedup_output=dedup_output,
            user_id="test_user",
            request_id="test_request",
        )

        assert delete_ids == ["p-existing-0"]
        assert len(superseded) == 1
        assert result_profiles == []


# ===============================
# Test: Dedup output validator (repair-ladder feedback)
# ===============================


class TestValidateProfileDedupOutput:
    """
    Tests for the semantic validator fed to the structured-output repair
    ladder. These cover the failure class the ladder can actually correct:
    output that parses but references unusable ids.
    """

    @staticmethod
    def _validate(output, new_count=3, existing_count=2):
        return validate_profile_dedup_output(
            output,
            new_profile_count=new_count,
            existing_profile_count=existing_count,
        )

    def test_well_formed_output_passes(self):
        output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-0", "EXISTING-1"],
                    merged_content="merged",
                    merged_time_to_live="one_month",
                )
            ],
            unique_ids=["NEW-1", "NEW-2"],
        )
        assert self._validate(output) == []

    def test_out_of_range_ids_are_reported(self):
        output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-7", "EXISTING-9"],
                    merged_content="merged",
                    merged_time_to_live="one_month",
                )
            ],
            unique_ids=["NEW-0", "NEW-1", "NEW-2"],
        )
        errors = self._validate(output)
        assert any("NEW-7" in e and "out of range" in e for e in errors)
        assert any("EXISTING-9" in e and "out of range" in e for e in errors)

    def test_unparseable_id_is_reported(self):
        output = ProfileDeduplicationOutput(unique_ids=["NEW-0", "NEW-1", "banana"])
        errors = self._validate(output)
        assert any("banana" in e for e in errors)

    def test_missing_new_coverage_is_reported(self):
        """The prompt's own invariant: every NEW profile referenced exactly once."""
        output = ProfileDeduplicationOutput(unique_ids=["NEW-0"])
        errors = self._validate(output)
        assert any("NEW-1" in e and "Missing" in e for e in errors)
        assert any("NEW-2" in e and "Missing" in e for e in errors)

    def test_duplicate_new_coverage_is_reported(self):
        output = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-0", "NEW-1"],
                    merged_content="merged",
                    merged_time_to_live="one_month",
                )
            ],
            unique_ids=["NEW-0", "NEW-2"],
        )
        errors = self._validate(output)
        assert any("more than once" in e and "NEW-0" in e for e in errors)

    def test_deletion_directive_ids_are_range_checked(self):
        output = ProfileDeduplicationOutput(
            unique_ids=["NEW-1", "NEW-2"],
            deletions=[
                ProfileDeletionDirective(
                    new_id="NEW-0",
                    existing_ids=["EXISTING-4"],
                    reasoning="forget",
                )
            ],
        )
        errors = self._validate(output)
        assert any("EXISTING-4" in e and "out of range" in e for e in errors)

    def test_wrong_prefix_is_reported(self):
        """unique_ids must be NEW ids — an EXISTING id there is a real error."""
        output = ProfileDeduplicationOutput(
            unique_ids=["NEW-0", "NEW-1", "NEW-2", "EXISTING-0"]
        )
        errors = self._validate(output)
        assert any("EXISTING-0" in e and "must be an NEW-<n> id" in e for e in errors)


# ===============================
# Test: Failure instrumentation and repair-ladder wiring
# ===============================


class TestDedupFailureHandling:
    """Tests for the observability and degradation behavior of deduplicate()."""

    @pytest.fixture
    def consolidator(
        self, mock_request_context, mock_llm_client, mock_site_var_manager
    ):
        mock_request_context.org_id = "org-test"
        return ProfileConsolidator(
            request_context=mock_request_context,
            llm_client=mock_llm_client,
        )

    def test_validator_is_passed_to_the_client(self, consolidator, sample_profiles):
        """The dedup call must opt into the corrective repair ladder."""
        consolidator.client.generate_chat_response.return_value = (
            ProfileDeduplicationOutput(unique_ids=["NEW-0", "NEW-1", "NEW-2"])
        )
        consolidator.deduplicate(sample_profiles, "test_user", "req-1")

        kwargs = consolidator.client.generate_chat_response.call_args.kwargs
        assert callable(kwargs["structured_output_validator"])

    def test_total_failure_is_reported_and_degrades_to_undeduped(
        self, consolidator, sample_profiles
    ):
        consolidator.client.generate_chat_response.side_effect = LiteLLMClientError(
            "Connection timed out after 120.0 seconds"
        )

        with patch(
            "reflexio.server.services.profile.components.consolidator.capture_anomaly"
        ) as mock_capture:
            profiles, delete_ids, superseded = consolidator.deduplicate(
                sample_profiles, "test_user", "req-1"
            )

        assert len(profiles) == len(sample_profiles)
        assert delete_ids == []
        assert superseded == []
        mock_capture.assert_called_once()
        assert mock_capture.call_args.args[0] == "profile.dedup.failed"
        assert mock_capture.call_args.kwargs["failure_kind"] == "timeout"
        assert mock_capture.call_args.kwargs["new_profile_count"] == 3

    def test_exhausted_ladder_degrades_to_first_parsed_attempt(
        self, consolidator, sample_profiles
    ):
        """
        When the ladder exhausts but an attempt parsed, that attempt is used
        rather than throwing away all dedup work.
        """
        partial = ProfileDeduplicationOutput(
            duplicate_groups=[
                ProfileDuplicateGroup(
                    item_ids=["NEW-0", "NEW-1"],
                    merged_content="MERGED",
                    merged_time_to_live="one_month",
                )
            ],
            # NEW-2 deliberately uncovered — the semantic error the validator
            # rejected this attempt for.
        )

        def _call(**kwargs):
            kwargs["structured_output_validator"](partial)
            raise StructuredOutputRepairError(
                "Structured output repair exhausted",
                failure_kind="semantic",
                model="minimax/MiniMax-M3",
            )

        consolidator.client.generate_chat_response.side_effect = _call

        with patch(
            "reflexio.server.services.profile.components.consolidator.capture_anomaly"
        ) as mock_capture:
            profiles, _, _ = consolidator.deduplicate(
                sample_profiles, "test_user", "req-1"
            )

        # The merge was applied, and the uncovered NEW-2 still survived via the
        # safety fallback — nothing was lost.
        contents = {p.content for p in profiles}
        assert "MERGED" in contents
        assert sample_profiles[2].content in contents
        assert (
            mock_capture.call_args.args[0] == "profile.dedup.degraded_to_first_attempt"
        )
        assert mock_capture.call_args.kwargs["failure_kind"] == "repair_semantic"


class TestDedupFailureKind:
    """Tests for the anomaly failure-class tag."""

    def test_repair_error_carries_its_failure_kind(self):
        exc = StructuredOutputRepairError(
            "exhausted", failure_kind="parse", model="minimax/MiniMax-M3"
        )
        assert _dedup_failure_kind(exc) == "repair_parse"

    def test_timeout_is_detected_from_the_message(self):
        exc = LiteLLMClientError("Connection timed out after 120.0 seconds")
        assert _dedup_failure_kind(exc) == "timeout"

    def test_other_client_errors_are_grouped(self):
        assert _dedup_failure_kind(LiteLLMClientError("boom")) == "llm_client_error"

    def test_unexpected_errors_use_their_type_name(self):
        assert _dedup_failure_kind(ValueError("boom")) == "ValueError"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
