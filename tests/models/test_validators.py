"""
Tests for Pydantic validators in reflexio_commons.

Covers:
1. SSRF prevention (SafeHttpUrl) — cloud metadata, private IPs, localhost
2. Prompt injection mitigation (SanitizedStr) — control character stripping
3. Data integrity (NonEmptyStr, EmbeddingVector, numeric constraints)
4. Time range validation
5. Email validation
6. Cross-field model validators
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.retriever_schema import (
    ConversationTurn,
    GetDashboardStatsRequest,
    GetInteractionsRequest,
    GetRequestsRequest,
    GetUserPlaybooksRequest,
    PeriodStats,
    SearchInteractionRequest,
    SearchUserProfileRequest,
    TimeSeriesDataPoint,
)
from reflexio.models.api_schema.service_schemas import (
    AddAgentPlaybookRequest,
    AddUserPlaybookRequest,
    AgentPlaybook,
    DeleteAgentPlaybookRequest,
    DeleteRequestRequest,
    DeleteUserInteractionRequest,
    DeleteUserPlaybookRequest,
    Interaction,
    InteractionData,
    OperationStatus,
    OperationStatusInfo,
    PublishUserInteractionRequest,
    RerunPlaybookGenerationRequest,
    RerunProfileGenerationRequest,
    UserProfile,
)
from reflexio.models.config_schema import (
    DEFAULT_BATCH_INTERVAL,
    AgentSuccessConfig,
    AnthropicConfig,
    AzureOpenAIConfig,
    Config,
    CustomEndpointConfig,
    OpenAIConfig,
    PlaybookAggregatorConfig,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
    ToolUseConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def strict_mode(monkeypatch):
    """Enable strict URL validation mode (blocks private IPs and localhost)."""
    monkeypatch.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "true")


@pytest.fixture
def non_strict_mode(monkeypatch):
    """Ensure strict URL validation mode is disabled."""
    monkeypatch.delenv("REFLEXIO_BLOCK_PRIVATE_URLS", raising=False)


# =============================================================================
# SSRF Prevention Tests — SafeHttpUrl
# =============================================================================


class TestSSRFPrevention:
    """Tests for SSRF prevention via SafeHttpUrl on config models."""

    def test_always_blocks_aws_metadata_ip(self):
        """Cloud metadata IP 169.254.169.254 is ALWAYS blocked."""
        with pytest.raises(ValidationError, match="cloud metadata"):
            CustomEndpointConfig(
                model="x", api_key="k", api_base="http://169.254.169.254/latest"
            )

    def test_always_blocks_gcp_metadata_hostname(self):
        """GCP metadata hostname is ALWAYS blocked."""
        with pytest.raises(ValidationError, match="cloud metadata"):
            CustomEndpointConfig(
                model="x",
                api_key="k",
                api_base="http://metadata.google.internal/computeMetadata/v1",
            )

    def test_allows_public_urls(self):
        """Public URLs are always accepted."""
        config = CustomEndpointConfig(
            model="gpt-4", api_key="sk-test", api_base="https://api.openai.com/v1"
        )
        assert str(config.api_base).startswith("https://api.openai.com")

    def test_allows_localhost_by_default(self, non_strict_mode):
        """Localhost is allowed when REFLEXIO_BLOCK_PRIVATE_URLS is not set."""
        config = CustomEndpointConfig(
            model="local-model", api_key="k", api_base="http://localhost:8080/v1"
        )
        assert "localhost" in str(config.api_base)

    def test_allows_private_ip_by_default(self, non_strict_mode):
        """Private IPs are allowed when REFLEXIO_BLOCK_PRIVATE_URLS is not set."""
        config = CustomEndpointConfig(
            model="local-model", api_key="k", api_base="http://192.168.1.100:8080/v1"
        )
        assert "192.168.1.100" in str(config.api_base)

    def test_blocks_localhost_in_strict_mode(self, strict_mode):
        """Localhost is blocked when REFLEXIO_BLOCK_PRIVATE_URLS=true."""
        with pytest.raises(ValidationError, match="localhost"):
            CustomEndpointConfig(
                model="x", api_key="k", api_base="http://localhost:8080/v1"
            )

    def test_blocks_private_ip_in_strict_mode(self, strict_mode):
        """Private IPs are blocked when REFLEXIO_BLOCK_PRIVATE_URLS=true."""
        with pytest.raises(ValidationError, match="private"):
            CustomEndpointConfig(
                model="x", api_key="k", api_base="http://192.168.1.1/v1"
            )

    def test_blocks_loopback_ip_in_strict_mode(self, strict_mode):
        """Loopback IP 127.0.0.1 is blocked in strict mode."""
        with pytest.raises(ValidationError):
            CustomEndpointConfig(
                model="x", api_key="k", api_base="http://127.0.0.1:8080/v1"
            )

    def test_azure_endpoint_ssrf_prevention(self):
        """AzureOpenAIConfig.endpoint is also protected against SSRF."""
        with pytest.raises(ValidationError, match="cloud metadata"):
            AzureOpenAIConfig(
                api_key="test-key",
                endpoint="http://169.254.169.254/latest/meta-data/",
            )

    def test_azure_endpoint_allows_valid_url(self):
        """Valid Azure endpoints are accepted."""
        config = AzureOpenAIConfig(
            api_key="test-key",
            endpoint="https://my-resource.openai.azure.com/",
        )
        assert "openai.azure.com" in str(config.endpoint)


# =============================================================================
# Image URL SSRF Tests
# =============================================================================


class TestImageURLSSRF:
    """Tests for SSRF prevention on interacted_image_url."""

    def test_empty_image_url_allowed(self):
        """Empty string is the default and should be allowed."""
        data = InteractionData(interacted_image_url="")
        assert data.interacted_image_url == ""

    def test_blocks_file_scheme(self):
        """file:// scheme is blocked."""
        with pytest.raises(ValidationError, match="scheme must be http"):
            InteractionData(interacted_image_url="file:///etc/passwd")

    def test_blocks_ftp_scheme(self):
        """ftp:// scheme is blocked."""
        with pytest.raises(ValidationError, match="scheme must be http"):
            InteractionData(interacted_image_url="ftp://evil.com/image.png")

    def test_allows_data_uri(self):
        """data: URIs are allowed for inline images."""
        data = InteractionData(
            interacted_image_url="data:image/png;base64,iVBORw0KGgo="
        )
        assert data.interacted_image_url.startswith("data:image/png")

    def test_allows_https_url(self):
        """Public HTTPS URLs are allowed."""
        data = InteractionData(interacted_image_url="https://example.com/image.png")
        assert "example.com" in data.interacted_image_url

    def test_blocks_metadata_ip_in_image_url(self):
        """Cloud metadata IP is blocked in image URLs."""
        with pytest.raises(ValidationError, match="cloud metadata"):
            InteractionData(
                interacted_image_url="http://169.254.169.254/latest/meta-data/"
            )

    def test_interaction_model_also_validates(self):
        """Interaction entity model also validates image URLs."""
        with pytest.raises(ValidationError, match="scheme must be http"):
            Interaction(
                user_id="test",
                request_id="req-1",
                interacted_image_url="file:///etc/passwd",
            )


# =============================================================================
# Prompt Injection Mitigation Tests — SanitizedStr
# =============================================================================


class TestPromptInjectionMitigation:
    """Tests for control character stripping in prompt fields."""

    def test_strips_null_bytes(self):
        """NULL bytes (\x00) are stripped from prompt fields."""
        config = ProfileExtractorConfig(
            extractor_name="test",
            extraction_definition_prompt="Extract\x00preferences",
        )
        assert "\x00" not in config.extraction_definition_prompt
        assert config.extraction_definition_prompt == "Extractpreferences"

    def test_strips_escape_sequences(self):
        """Escape sequences (\x1b) are stripped from prompt fields."""
        config = ProfileExtractorConfig(
            extractor_name="test",
            extraction_definition_prompt="Hello\x1b[31mRED\x1b[0m",
        )
        assert "\x1b" not in config.extraction_definition_prompt

    def test_preserves_tabs_and_newlines(self):
        """Tabs and newlines are legitimate and preserved."""
        prompt = "Step 1:\tDo this\nStep 2:\tDo that"
        config = ProfileExtractorConfig(
            extractor_name="test",
            extraction_definition_prompt=prompt,
        )
        assert "\t" in config.extraction_definition_prompt
        assert "\n" in config.extraction_definition_prompt

    def test_strips_bell_character(self):
        """Bell character (\x07) is stripped."""
        config = PlaybookConfig(
            extractor_name="test",
            extraction_definition_prompt="Alert\x07user",
        )
        assert "\x07" not in config.extraction_definition_prompt

    def test_agent_success_prompt_sanitized(self):
        """AgentSuccessConfig.success_definition_prompt is also sanitized."""
        config = AgentSuccessConfig(
            evaluation_name="test",
            success_definition_prompt="Check\x00success\x1b[0m",
        )
        assert "\x00" not in config.success_definition_prompt
        assert "\x1b" not in config.success_definition_prompt


# =============================================================================
# Data Integrity Tests — NonEmptyStr
# =============================================================================


class TestNonEmptyStr:
    """Tests for NonEmptyStr validation across models."""

    def test_rejects_empty_string(self):
        """Empty string is rejected."""
        with pytest.raises(ValidationError, match="empty"):
            DeleteRequestRequest(request_id="")

    def test_rejects_whitespace_only(self):
        """Whitespace-only string is rejected."""
        with pytest.raises(ValidationError, match="empty"):
            DeleteRequestRequest(request_id="   ")

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        req = DeleteRequestRequest(request_id="  req-123  ")
        assert req.request_id == "req-123"

    def test_accepts_valid_string(self):
        """Valid non-empty string is accepted."""
        req = DeleteRequestRequest(request_id="req-123")
        assert req.request_id == "req-123"

    def test_config_api_key_non_empty(self):
        """API key fields reject empty strings."""
        with pytest.raises(ValidationError, match="empty"):
            AnthropicConfig(api_key="")

    def test_storage_config_non_empty(self):
        """StorageConfigDisk.dir_path rejects empty string."""
        from reflexio.models.config_schema import StorageConfigDisk

        with pytest.raises(ValidationError, match="empty"):
            StorageConfigDisk(dir_path="")

    def test_tool_use_config_non_empty(self):
        """ToolUseConfig fields reject empty strings."""
        with pytest.raises(ValidationError):
            ToolUseConfig(tool_name="", tool_description="test")


# =============================================================================
# Data Integrity Tests — EmbeddingVector
# =============================================================================


class TestEmbeddingVector:
    """Tests for embedding dimension validation."""

    def test_empty_embedding_allowed(self):
        """Empty embedding is allowed (not yet computed)."""
        interaction = Interaction(user_id="test", request_id="req-1", embedding=[])
        assert interaction.embedding == []

    def test_correct_dimension_allowed(self):
        """512-dimension embedding is accepted."""
        embedding = [0.1] * 512
        interaction = Interaction(
            user_id="test", request_id="req-1", embedding=embedding
        )
        assert len(interaction.embedding) == 512

    def test_wrong_dimension_rejected(self):
        """Non-512 non-empty embedding is rejected."""
        with pytest.raises(ValidationError, match="512"):
            Interaction(user_id="test", request_id="req-1", embedding=[1.0, 2.0, 3.0])

    def test_user_profile_embedding_validation(self):
        """UserProfile also validates embedding dimensions."""
        with pytest.raises(ValidationError, match="512"):
            UserProfile(
                profile_id="p1",
                user_id="u1",
                content="test",
                last_modified_timestamp=1000,
                generated_from_request_id="r1",
                embedding=[1.0] * 10,
            )

    def test_playbook_embedding_validation(self):
        """AgentPlaybook also validates embedding dimensions."""
        with pytest.raises(ValidationError, match="512"):
            AgentPlaybook(
                agent_version="v1",
                content="test",
                embedding=[1.0] * 100,
            )


# =============================================================================
# Numeric Constraint Tests
# =============================================================================


class TestNumericConstraints:
    """Tests for numeric field constraints."""

    def test_threshold_lower_bound(self):
        """Threshold cannot be below 0.0."""
        with pytest.raises(ValidationError):
            SearchUserProfileRequest(user_id="test", threshold=-0.1)

    def test_threshold_upper_bound(self):
        """Threshold cannot exceed 1.0."""
        with pytest.raises(ValidationError):
            SearchUserProfileRequest(user_id="test", threshold=1.1)

    def test_threshold_boundary_values(self):
        """Threshold boundary values 0.0 and 1.0 are accepted."""
        r1 = SearchUserProfileRequest(user_id="test", threshold=0.0)
        assert r1.threshold == 0.0
        r2 = SearchUserProfileRequest(user_id="test", threshold=1.0)
        assert r2.threshold == 1.0

    def test_top_k_must_be_positive(self):
        """top_k must be > 0."""
        with pytest.raises(ValidationError):
            SearchUserProfileRequest(user_id="test", top_k=0)

    def test_top_k_negative_rejected(self):
        """Negative top_k is rejected."""
        with pytest.raises(ValidationError):
            GetInteractionsRequest(user_id="test", top_k=-5)

    def test_limit_must_be_positive(self):
        """limit must be > 0."""
        with pytest.raises(ValidationError):
            GetUserPlaybooksRequest(limit=0)

    def test_offset_non_negative(self):
        """offset must be >= 0."""
        with pytest.raises(ValidationError):
            GetRequestsRequest(offset=-1)

    def test_offset_zero_allowed(self):
        """offset=0 is allowed."""
        req = GetRequestsRequest(offset=0)
        assert req.offset == 0

    def test_days_back_must_be_positive(self):
        """days_back must be > 0."""
        with pytest.raises(ValidationError):
            GetDashboardStatsRequest(days_back=0)

    def test_delete_id_must_be_positive(self):
        """Delete request IDs must be > 0."""
        with pytest.raises(ValidationError):
            DeleteAgentPlaybookRequest(agent_playbook_id=0)
        with pytest.raises(ValidationError):
            DeleteUserPlaybookRequest(user_playbook_id=-1)
        with pytest.raises(ValidationError):
            DeleteUserInteractionRequest(user_id="test", interaction_id=0)

    def test_progress_percentage_range(self):
        """progress_percentage must be 0-100."""
        with pytest.raises(ValidationError):
            OperationStatusInfo(
                service_name="test",
                status=OperationStatus.IN_PROGRESS,
                started_at=1000,
                progress_percentage=101.0,
            )

    def test_sampling_rate_range(self):
        """sampling_rate must be 0.0-1.0."""
        with pytest.raises(ValidationError):
            AgentSuccessConfig(
                evaluation_name="test",
                success_definition_prompt="Check success",
                sampling_rate=1.5,
            )

    def test_period_stats_non_negative(self):
        """PeriodStats counts must be >= 0."""
        with pytest.raises(ValidationError):
            PeriodStats(
                total_profiles=-1,
                total_interactions=0,
                total_playbooks=0,
                success_rate=50.0,
            )

    def test_success_rate_percentage(self):
        """success_rate must be 0-100."""
        with pytest.raises(ValidationError):
            PeriodStats(
                total_profiles=0,
                total_interactions=0,
                total_playbooks=0,
                success_rate=150.0,
            )

    def test_timeseries_value_non_negative(self):
        """TimeSeriesDataPoint.value must be >= 0."""
        with pytest.raises(ValidationError):
            TimeSeriesDataPoint(timestamp=1000, value=-1)

    def test_timeseries_timestamp_positive(self):
        """TimeSeriesDataPoint.timestamp must be > 0."""
        with pytest.raises(ValidationError):
            TimeSeriesDataPoint(timestamp=0, value=5)

    def test_playbook_aggregator_config_constraints(self):
        """PlaybookAggregatorConfig thresholds must be >= 1."""
        with pytest.raises(ValidationError):
            PlaybookAggregatorConfig(min_cluster_size=0)
        with pytest.raises(ValidationError):
            PlaybookAggregatorConfig(reaggregation_trigger_count=0)

    def test_window_override_must_be_positive(self):
        """Batch size/interval overrides must be > 0 when set."""
        with pytest.raises(ValidationError):
            ProfileExtractorConfig(
                extractor_name="test",
                extraction_definition_prompt="test",
                batch_size_override=0,
            )
        with pytest.raises(ValidationError):
            ProfileExtractorConfig(
                extractor_name="test",
                extraction_definition_prompt="test",
                batch_interval_override=-1,
            )
        # None is allowed (use global setting)
        config = ProfileExtractorConfig(
            extractor_name="test",
            extraction_definition_prompt="test",
            batch_size_override=None,
        )
        assert config.batch_size_override is None


# =============================================================================
# List Minimum Length Tests
# =============================================================================


class TestListMinLength:
    """Tests for minimum list length constraints on request models."""

    def test_publish_interaction_requires_data(self):
        """PublishUserInteractionRequest requires at least one interaction."""
        with pytest.raises(ValidationError):
            PublishUserInteractionRequest(user_id="test", interaction_data_list=[])

    def test_add_user_playbook_requires_data(self):
        """AddUserPlaybookRequest requires at least one user playbook."""
        with pytest.raises(ValidationError):
            AddUserPlaybookRequest(user_playbooks=[])

    def test_add_agent_playbook_requires_data(self):
        """AddAgentPlaybookRequest requires at least one agent playbook."""
        with pytest.raises(ValidationError):
            AddAgentPlaybookRequest(agent_playbooks=[])


# =============================================================================
# Time Range Validation Tests
# =============================================================================


class TestTimeRangeValidation:
    """Tests for time range validation on request models."""

    def test_end_before_start_rejected(self):
        """end_time before start_time is rejected."""
        with pytest.raises(ValidationError, match="end_time must be after"):
            RerunProfileGenerationRequest(
                start_time=datetime(2024, 6, 1, tzinfo=UTC),
                end_time=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_equal_times_rejected(self):
        """Equal start_time and end_time is rejected."""
        same_time = datetime(2024, 6, 1, tzinfo=UTC)
        with pytest.raises(ValidationError, match="end_time must be after"):
            RerunProfileGenerationRequest(start_time=same_time, end_time=same_time)

    def test_valid_time_range_accepted(self):
        """Valid time range (end > start) is accepted."""
        req = RerunProfileGenerationRequest(
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
            end_time=datetime(2024, 6, 1, tzinfo=UTC),
        )
        assert req.start_time < req.end_time

    def test_none_times_accepted(self):
        """None values for both times are accepted."""
        req = RerunProfileGenerationRequest()
        assert req.start_time is None
        assert req.end_time is None

    def test_only_start_time_accepted(self):
        """Only start_time without end_time is accepted."""
        req = SearchInteractionRequest(
            user_id="test",
            start_time=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert req.start_time is not None
        assert req.end_time is None

    def test_rerun_playbook_time_range(self):
        """RerunPlaybookGenerationRequest also validates time range."""
        with pytest.raises(ValidationError, match="end_time must be after"):
            RerunPlaybookGenerationRequest(
                agent_version="v1",
                start_time=datetime(2024, 6, 1, tzinfo=UTC),
                end_time=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_search_interaction_time_range(self):
        """SearchInteractionRequest validates time range."""
        with pytest.raises(ValidationError, match="end_time must be after"):
            SearchInteractionRequest(
                user_id="test",
                start_time=datetime(2024, 6, 1, tzinfo=UTC),
                end_time=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_get_requests_time_range(self):
        """GetRequestsRequest validates time range."""
        with pytest.raises(ValidationError, match="end_time must be after"):
            GetRequestsRequest(
                start_time=datetime(2024, 6, 1, tzinfo=UTC),
                end_time=datetime(2024, 1, 1, tzinfo=UTC),
            )


# =============================================================================
# Cross-Field Model Validator Tests
# =============================================================================


class TestCrossFieldValidators:
    """Tests for model-level cross-field validators."""

    def test_openai_config_requires_at_least_one_auth(self):
        """OpenAIConfig requires at least api_key or azure_config."""
        with pytest.raises(ValidationError, match="(?i)at least one"):
            OpenAIConfig()

    def test_openai_config_with_api_key(self):
        """OpenAIConfig with only api_key is valid."""
        config = OpenAIConfig(api_key="sk-test")
        assert config.api_key == "sk-test"

    def test_openai_config_with_azure(self):
        """OpenAIConfig with only azure_config is valid."""
        config = OpenAIConfig(
            azure_config=AzureOpenAIConfig(
                api_key="test",
                endpoint="https://my-resource.openai.azure.com/",
            )
        )
        assert config.azure_config is not None

    def test_config_stride_le_window(self):
        """Config: batch_interval must be <= batch_size."""
        with pytest.raises(ValidationError, match="batch_interval"):
            Config(
                storage_config=StorageConfigSQLite(),
                batch_size=10,
                batch_interval=20,
            )

    def test_config_stride_equal_window_ok(self):
        """Config: batch_interval == batch_size is OK."""
        config = Config(
            storage_config=StorageConfigSQLite(),
            batch_size=10,
            batch_interval=10,
        )
        assert config.batch_interval == 10

    def test_config_stride_default_ok(self):
        """Config: omitting batch_interval uses DEFAULT_BATCH_INTERVAL."""
        config = Config(
            storage_config=StorageConfigSQLite(),
            batch_size=10,
        )
        assert config.batch_interval == DEFAULT_BATCH_INTERVAL

    def test_config_defaults_extractors_when_omitted(self):
        """Config: omitted extractor lists still get the default extractors."""
        config = Config(storage_config=StorageConfigSQLite())

        assert config.profile_extractor_configs is not None
        assert config.user_playbook_extractor_configs is not None
        assert config.profile_extractor_configs[0].extractor_name == (
            "default_profile_extractor"
        )
        assert config.user_playbook_extractor_configs[0].extractor_name == (
            "default_playbook_extractor"
        )

    def test_config_populates_default_extractors_when_null(self):
        """Config: null extractor lists get populated with defaults."""
        config = Config(
            storage_config=StorageConfigSQLite(),
            profile_extractor_configs=None,
            user_playbook_extractor_configs=None,
        )

        assert config.profile_extractor_configs is not None
        assert len(config.profile_extractor_configs) == 1
        assert (
            config.profile_extractor_configs[0].extractor_name
            == "default_profile_extractor"
        )
        assert config.user_playbook_extractor_configs is not None
        assert len(config.user_playbook_extractor_configs) == 1
        assert (
            config.user_playbook_extractor_configs[0].extractor_name
            == "default_playbook_extractor"
        )


# =============================================================================
# ConversationTurn Tests
# =============================================================================


class TestConversationTurn:
    """Tests for ConversationTurn validation."""

    def test_empty_role_rejected(self):
        """ConversationTurn.role must be non-empty."""
        with pytest.raises(ValidationError):
            ConversationTurn(role="", content="hello")

    def test_empty_content_rejected(self):
        """ConversationTurn.content must be non-empty."""
        with pytest.raises(ValidationError):
            ConversationTurn(role="user", content="")

    def test_valid_turn_accepted(self):
        """Valid ConversationTurn is accepted."""
        turn = ConversationTurn(role="user", content="Hello, how are you?")
        assert turn.role == "user"


# =============================================================================
# Backward Compatibility Tests — AliasChoices
# =============================================================================


class TestBackwardCompatibility:
    """Tests that old field names still work via model_validator migration."""

    def test_config_accepts_old_window_names(self):
        """Config: extraction_window_size/stride still accepted as aliases."""
        config = Config(
            storage_config=StorageConfigSQLite(),
            extraction_window_size=15,
            extraction_window_stride=7,
        )
        assert config.batch_size == 15
        assert config.batch_interval == 7

    def test_config_new_names_preferred(self):
        """Config: new field names batch_size/batch_interval work directly."""
        config = Config(
            storage_config=StorageConfigSQLite(),
            batch_size=20,
            batch_interval=8,
        )
        assert config.batch_size == 20
        assert config.batch_interval == 8

    def test_profile_extractor_old_override_names(self):
        """ProfileExtractorConfig: old extraction_window_*_override names still work."""
        config = ProfileExtractorConfig(
            extractor_name="test",
            extraction_definition_prompt="test",
            extraction_window_size_override=30,
            extraction_window_stride_override=10,
        )
        assert config.batch_size_override == 30
        assert config.batch_interval_override == 10

    def test_playbook_config_old_override_names(self):
        """PlaybookConfig: old extraction_window_*_override names still work."""
        config = PlaybookConfig(
            extractor_name="test",
            extraction_definition_prompt="test",
            extraction_window_size_override=25,
            extraction_window_stride_override=5,
        )
        assert config.batch_size_override == 25
        assert config.batch_interval_override == 5

    def test_success_config_old_override_names(self):
        """AgentSuccessConfig: old extraction_window_*_override names still work."""
        config = AgentSuccessConfig(
            evaluation_name="test",
            success_definition_prompt="test",
            extraction_window_size_override=40,
            extraction_window_stride_override=15,
        )
        assert config.batch_size_override == 40
        assert config.batch_interval_override == 15

    def test_aggregator_config_old_field_names(self):
        """PlaybookAggregatorConfig: old field names still work via AliasChoices."""
        config = PlaybookAggregatorConfig(
            min_feedback_threshold=5,
            refresh_count=3,
            similarity_threshold=0.7,
        )
        assert config.min_cluster_size == 5
        assert config.reaggregation_trigger_count == 3
        assert config.clustering_similarity == 0.7

    def test_aggregator_config_new_field_names(self):
        """PlaybookAggregatorConfig: new field names work directly."""
        config = PlaybookAggregatorConfig(
            min_cluster_size=4,
            reaggregation_trigger_count=6,
            clustering_similarity=0.8,
        )
        assert config.min_cluster_size == 4
        assert config.reaggregation_trigger_count == 6
        assert config.clustering_similarity == 0.8

    def test_aggregator_config_direction_overlap_threshold_default(self):
        """PlaybookAggregatorConfig: direction_overlap_threshold defaults to 0.6."""
        config = PlaybookAggregatorConfig()
        assert config.direction_overlap_threshold == 0.6
