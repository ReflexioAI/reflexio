"""
Unit tests for ProfileConsolidator.

Tests the consolidator's responsibilities for:
- Internal compiler schema and reference validation
- Native owner-scoped retrieval, pending previews and readiness refusal
- Source-bound operation wire and coherent replacement/forget behavior
- Repair exhaustion without unchecked proposal publication
- Provenance, optional search metadata and native metadata primitives
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
from reflexio.models.config_schema import LLMConfig
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
    validate_profile_dedup_output,
)


@pytest.mark.parametrize("phase", ["embedding", "search"])
def test_readiness_refusal_never_consolidates_incomplete_history(
    mock_request_context, mock_llm_client, mock_site_var_manager, phase
):
    from reflexio.server.services.storage.error import ReadinessUnavailableError

    error = ReadinessUnavailableError("embedding_readiness_unavailable")
    storage = mock_request_context.storage
    storage._get_embedding.return_value = [0.1] * 10
    if phase == "embedding":
        storage._get_embedding.side_effect = error
    else:
        storage.search_user_profile.side_effect = [[], error]
    consolidator = ProfileConsolidator(mock_request_context, mock_llm_client)
    consolidator.review_window = _review_window()
    candidates = [
        UserProfile(
            profile_id=str(i),
            user_id="u",
            content=text,
            last_modified_timestamp=100,
            generated_from_request_id="request",
        )
        for i, text in enumerate(["Short replies", "Examples"])
    ]
    with pytest.raises(ReadinessUnavailableError) as caught:
        consolidator.deduplicate(candidates, "u", "request")
    assert caught.value is error
    mock_llm_client.generate_chat_response_with_provenance.assert_not_called()


# ===============================
# Fixtures
# ===============================


@pytest.mark.parametrize("has_source", [False, True])
def test_native_mock_bypass_preserves_profiles_without_planner(
    monkeypatch,
    mock_request_context,
    mock_llm_client,
    mock_site_var_manager,
    has_source,
):
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "true")
    consolidator = ProfileConsolidator(mock_request_context, mock_llm_client)
    if has_source:
        consolidator.review_window = _review_window()
    profile = UserProfile(
        profile_id="mock-profile",
        user_id="u",
        content="Prefers concise answers",
        last_modified_timestamp=100,
        generated_from_request_id="request",
    )
    assert consolidator.deduplicate([profile], "u", "request") == ([profile], [], [])
    mock_llm_client.generate_chat_response_with_provenance.assert_not_called()
    mock_request_context.storage.search_user_profile.assert_not_called()
    assert consolidator.model_provenance is None


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
    context.org_id = "org-test"
    context.prompt_manager = MagicMock()
    context.prompt_manager.render_prompt.return_value = "test prompt"
    context.storage = MagicMock()
    context.storage.search_user_profile.return_value = []
    # Set up configurator chain for model resolution
    mock_config = MagicMock()
    mock_config.api_key_config = None
    mock_config.llm_config = LLMConfig()
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


# ===============================
# Test: Deduplicate Main Method
# ===============================


# ===============================
# Test: Integration
# ===============================


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


# Selected operation compiler behavior replaces tests for the retired merge actor.
# These fixtures are code-owned source conversations; no customer or model data.
def _review_window(text="I prefer dark mode. I use Python."):
    return [
        {
            "session_id": "session",
            "request_id": "request",
            "source": "chat",
            "interactions": [{"interaction_id": 1, "role": "user", "content": text}],
        }
    ]


@pytest.fixture
def selected(mock_request_context, mock_llm_client, mock_site_var_manager):
    consolidator = ProfileConsolidator(mock_request_context, mock_llm_client)
    consolidator.review_window = _review_window()
    return consolidator


def _claim(identity="fresh", text="User prefers dark mode.", owner="owner"):
    return UserProfile(
        profile_id=identity,
        user_id=owner,
        content=text,
        last_modified_timestamp=100,
        generated_from_request_id="request",
        source_span="I prefer dark mode.",
        source_interaction_ids=[1],
        profile_time_to_live=ProfileTimeToLive.ONE_MONTH,
        expiration_timestamp=2592100,
    )


def _operation_reply(client, make_groups, *, raise_after_validation=False):
    """Exercise the actual emitted schema and validation seam without a provider."""
    import json

    def respond(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        payload = json.loads(prompt.rsplit("\n\n", 1)[1])
        output = kwargs["response_format"].model_validate(
            {
                "snapshot": payload["snapshot"],
                "groups": make_groups(payload),
            }
        )
        errors = kwargs["structured_output_validator"](output)
        if errors or raise_after_validation:
            raise StructuredOutputRepairError(
                "Structured output repair exhausted",
                failure_kind="semantic",
                model="provider-free-test",
            )
        return output

    client.generate_chat_response.side_effect = respond


def _assert_group(
    payload, *, new_id="NEW-0", content="User prefers dark mode.", replaces=None
):
    excerpt = payload["original_window"][0]["interactions"][0]["content"]["excerpts"][0]
    clause = {
        "action": "assert",
        "source_new_id": new_id,
        "content": content,
        "evidence": [{"excerpt_id": excerpt["excerpt_id"], "stance": "assertion"}],
    }
    if replaces is not None:
        clause["replaces"] = replaces
    return {"new_ids": [new_id], "writes": [{"clauses": [clause]}]}


def _all_of(row):
    return {"row_id": row["id"], "start_token": 0, "end_token": row["token_count"]}


def test_selected_empty_input_calls_no_model_or_storage(selected):
    assert selected.deduplicate([], "owner", "request") == ([], [], [])
    selected.client.generate_chat_response.assert_not_called()
    selected.request_context.storage.search_user_profile.assert_not_called()


def test_selected_wire_authors_fresh_fact_and_optional_metadata(selected):
    def groups(payload):
        group = _assert_group(payload)
        group["writes"][0]["search_terms"] = ["dark mode", "IDE"]
        return [group]

    _operation_reply(selected.client, groups)
    rows, retired, superseded = selected.deduplicate([_claim()], "owner", "request")
    assert [row.content for row in rows] == ["User prefers dark mode."]
    assert rows[0].expanded_terms == "dark mode, IDE"
    assert rows[0].source_interaction_ids == [1]
    assert rows[0].user_id == "owner"
    assert retired == superseded == []
    assert selected.lineage_sources_by_profile_id[rows[0].profile_id] == []
    selected.request_context.prompt_manager.render_prompt.assert_not_called()
    selected.client.generate_chat_response.assert_called_once()


def test_invalid_optional_metadata_does_not_reject_core_fact(selected):
    def groups(payload):
        group = _assert_group(payload)
        group["writes"][0]["search_terms"] = {"invalid": "object"}
        return [group]

    _operation_reply(selected.client, groups)
    rows, retired, _ = selected.deduplicate([_claim()], "owner", "request")
    assert [row.content for row in rows] == ["User prefers dark mode."]
    assert rows[0].expanded_terms is None
    assert retired == []


def test_selected_correction_compiles_replacement_before_retirement(selected):
    old = _claim("old", "User prefers light mode.")
    selected.request_context.storage.search_user_profile.return_value = [old]
    _operation_reply(
        selected.client,
        lambda payload: [
            _assert_group(
                payload,
                replaces=[_all_of(payload["EXISTING"][0])],
            )
        ],
    )
    rows, retired, superseded = selected.deduplicate([_claim()], "owner", "request")
    assert [row.content for row in rows] == ["User prefers dark mode."]
    assert retired == ["old"]
    assert superseded == [old]
    assert old.content == "User prefers light mode."
    assert selected.lineage_sources_by_profile_id[rows[0].profile_id] == ["old"]


def test_selected_foreign_owner_cannot_be_retired(selected):
    old = _claim("foreign", "User prefers light mode.", owner="another-user")
    selected.request_context.storage.search_user_profile.return_value = [old]
    _operation_reply(
        selected.client,
        lambda payload: [
            _assert_group(
                payload,
                replaces=[_all_of(payload["EXISTING"][0])],
            )
        ],
    )
    assert selected.deduplicate([_claim()], "owner", "request") == ([], [], [])
    assert old.user_id == "another-user"
    assert old.content == "User prefers light mode."


def test_selected_explicit_forget_consumes_marker_without_replacement(selected):
    selected.review_window = _review_window("Forget my dark mode preference.")
    old = _claim("old")
    selected.request_context.storage.search_user_profile.return_value = [old]

    def groups(payload):
        excerpt = payload["original_window"][0]["interactions"][0]["content"][
            "excerpts"
        ][0]
        return [
            {
                "new_ids": ["NEW-0"],
                "writes": [],
                "forget": [
                    {
                        "source_new_id": "NEW-0",
                        "targets": [_all_of(payload["EXISTING"][0])],
                        "evidence": [
                            {
                                "excerpt_id": excerpt["excerpt_id"],
                                "stance": "forget_instruction",
                            }
                        ],
                    }
                ],
            }
        ]

    _operation_reply(selected.client, groups)
    marker = _claim(text="Requested removal of dark mode preference")
    marker.source_span = "Forget my dark mode preference."
    assert selected.deduplicate([marker], "owner", "request") == ([], ["old"], [old])


@pytest.mark.parametrize("failure", ["timeout", "unexpected_type", "legacy_wire"])
def test_selected_unusable_planner_never_publishes_unreviewed_facts(selected, failure):
    old = _claim("old", "User prefers light mode.")
    selected.request_context.storage.search_user_profile.return_value = [old]
    if failure == "timeout":
        selected.client.generate_chat_response.side_effect = LiteLLMClientError(
            "Connection timed out"
        )
    else:
        selected.client.generate_chat_response.return_value = (
            "invalid"
            if failure == "unexpected_type"
            else ProfileDeduplicationOutput(unique_ids=["NEW-0"])
        )
    assert selected.deduplicate([_claim()], "owner", "request") == ([], [], [])
    assert old.content == "User prefers light mode."
    assert "planner_no_usable_output" in selected.enrichment_diagnostics[0]


def test_selected_timeout_reports_failure_class(selected):
    selected.client.generate_chat_response.side_effect = LiteLLMClientError(
        "Connection timed out"
    )
    with patch(
        "reflexio.server.services.profile.components.consolidator.capture_anomaly"
    ) as capture:
        assert selected.deduplicate([_claim()], "owner", "request") == ([], [], [])
    capture.assert_called_once()
    assert capture.call_args.args[0] == "profile.dedup.failed"
    assert capture.call_args.kwargs["failure_kind"] == "timeout"


def test_selected_missing_new_coverage_rejects_entire_wire_without_fallback(selected):
    claims = [_claim(), _claim("other", "User uses Python.")]
    claims[1].source_span = "I use Python."
    _operation_reply(selected.client, lambda payload: [_assert_group(payload)])
    rows, retired, _ = selected.deduplicate(claims, "owner", "request")
    assert rows == []
    assert retired == []
    assert selected.client.generate_chat_response.call_args.kwargs[
        "structured_output_validator"
    ]


@pytest.mark.parametrize("same_angle", [False, True])
def test_native_provenance_primitive_preserves_separate_quotes_and_perspectives(
    same_angle,
):
    from reflexio.server.services.profile.components.consolidator import (
        _merged_provenance,
    )

    first, second = _claim(), _claim("second")
    first.source_span = 'Keep answers short. "Exact quote" \\ path'
    second.source_span = "Include examples while staying brief."
    first.notes, second.notes = "First caveat", "Second caveat"
    first.reader_angle = "preferences"
    second.reader_angle = "preferences" if same_angle else "communication"
    span, notes, angle = _merged_provenance([first, second])
    assert span is not None and notes is not None
    assert span.startswith("Original source evidence values (separate excerpts):")
    assert first.source_span in span and second.source_span in span
    assert first.notes in notes and second.notes in notes
    assert angle == ("preferences" if same_angle else None)
    if not same_angle:
        assert "preferences" in notes and "communication" in notes


def test_selected_profile_prompt_uses_authentic_json_metadata(selected):
    import json

    row = _claim()
    row.source = "user_chat"
    formatted = json.loads(selected._format_profiles_with_prefix([row], "EXISTING"))
    assert formatted[0]["id"] == "EXISTING-0"
    assert formatted[0]["modified"] == row.last_modified_timestamp
    assert formatted[0]["expires"] == row.expiration_timestamp
    assert formatted[0]["ttl"] == "one_month"
    assert formatted[0]["source"] == "user_chat"
    assert json.loads(selected._format_profiles_with_prefix([], "EXISTING")) == []


def test_selected_exhausted_repair_salvages_validated_first_group(selected):
    _operation_reply(
        selected.client,
        lambda payload: [_assert_group(payload)],
        raise_after_validation=True,
    )
    rows, retired, superseded = selected.deduplicate([_claim()], "owner", "request")
    assert [row.content for row in rows] == ["User prefers dark mode."]
    assert retired == superseded == []


def test_retired_base_cannot_execute_as_an_alternative_engine():
    import inspect

    from reflexio.server.services.profile.components.consolidator import (
        _ProfileConsolidatorBase,
    )

    assert inspect.isabstract(_ProfileConsolidatorBase)
    assert not inspect.isabstract(ProfileConsolidator)
