from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from .api_schema.validators import (
    NonEmptyStr,
    SafeHttpUrl,
    SanitizedNonEmptyStr,
)

# Embedding vector dimensions. Changing this requires a DB migration and re-embedding,
# so it is intentionally a constant rather than a configurable setting.
EMBEDDING_DIMENSIONS = 512

# Default sliding window parameters for extraction
DEFAULT_BATCH_SIZE = 10
DEFAULT_BATCH_INTERVAL = 5


class ExtractionPreset(StrEnum):
    """Named extraction presets that bundle batch_size and batch_interval.

    Each preset targets a specific conversation pattern:
    - quick_chat: Short conversations (support bots, quick Q&A)
    - standard: General-purpose conversational agents (default)
    - long_form: Long conversations (coding assistants, research)
    - high_volume: High-traffic agents (1000+ daily interactions)
    """

    QUICK_CHAT = "quick_chat"
    STANDARD = "standard"
    LONG_FORM = "long_form"
    HIGH_VOLUME = "high_volume"


# Preset parameter values: (batch_size, batch_interval)
_PRESET_VALUES: dict[ExtractionPreset, tuple[int, int]] = {
    ExtractionPreset.QUICK_CHAT: (5, 3),
    ExtractionPreset.STANDARD: (DEFAULT_BATCH_SIZE, DEFAULT_BATCH_INTERVAL),
    ExtractionPreset.LONG_FORM: (25, 10),
    ExtractionPreset.HIGH_VOLUME: (15, 8),
}


# ---------------------------------------------------------------------------
# Field migration maps (old stored JSON name → new Python attr name)
# ---------------------------------------------------------------------------
_CONFIG_FIELD_MIGRATION: dict[str, str] = {
    "extraction_window_size": "batch_size",
    "extraction_window_stride": "batch_interval",
    "playbook_configs": "user_playbook_extractor_configs",
    "agent_feedback_configs": "user_playbook_extractor_configs",
}

_AGGREGATOR_FIELD_MIGRATION: dict[str, str] = {
    "min_feedback_threshold": "min_cluster_size",
    "refresh_count": "reaggregation_trigger_count",
    "similarity_threshold": "clustering_similarity",
}

_EXTRACTOR_OVERRIDE_MIGRATION: dict[str, str] = {
    "extraction_window_size_override": "batch_size_override",
    "extraction_window_stride_override": "batch_interval_override",
}

_PROFILE_CONFIG_FIELD_MIGRATION: dict[str, str] = {
    "profile_content_definition_prompt": "extraction_definition_prompt",
}

_PLAYBOOK_CONFIG_FIELD_MIGRATION: dict[str, str] = {
    "feedback_definition_prompt": "extraction_definition_prompt",
    "playbook_definition_prompt": "extraction_definition_prompt",
    "feedback_aggregator_config": "aggregation_config",
    "playbook_aggregator_config": "aggregation_config",
    "playbook_name": "extractor_name",
    "feedback_name": "extractor_name",
}


def _migrate_dict(data: Any, mapping: dict[str, str]) -> Any:
    """Rename old field names to new ones in a raw dict before Pydantic validates.

    Creates a shallow copy to avoid mutating the caller's dict.
    """
    if isinstance(data, dict):
        data = dict(data)
        for old, new in mapping.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
    return data


class SearchMode(StrEnum):
    """Search mode for hybrid search functionality.

    Controls how search queries are processed:
    - VECTOR: Pure vector similarity search using embeddings
    - FTS: Pure full-text search using PostgreSQL tsvector
    - HYBRID: Combined search using Reciprocal Rank Fusion (RRF)
    """

    VECTOR = "vector"
    FTS = "fts"
    HYBRID = "hybrid"


@dataclass
class SearchOptions:
    """Engine-level search parameters that are pre-computed or not part of the API request."""

    query_embedding: list[float] | None = field(default=None)
    search_mode: SearchMode = field(default=SearchMode.HYBRID)
    rrf_k: int = field(default=60)
    vector_weight: float = field(default=1.0)
    fts_weight: float = field(default=1.0)


class StorageConfigTest(IntEnum):
    UNKNOWN = 0
    INCOMPLETE = 1
    FAILED = 2
    SUCCEEDED = 3


class StorageConfigSQLite(BaseModel):
    """SQLite storage configuration."""

    db_path: str | None = None  # None = use SQLITE_FILE_DIRECTORY env var default


class StorageConfigSupabase(BaseModel):
    url: NonEmptyStr
    key: NonEmptyStr
    db_url: NonEmptyStr


class StorageConfigDisk(BaseModel):
    """Disk-based storage with file-based entities and QMD search."""

    dir_path: NonEmptyStr
    qmd_binary: str = "qmd"


StorageConfig = StorageConfigSQLite | StorageConfigSupabase | StorageConfigDisk | None


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI specific configuration."""

    api_key: NonEmptyStr
    endpoint: SafeHttpUrl  # e.g., "https://your-resource.openai.azure.com/"
    api_version: str = "2024-02-15-preview"
    deployment_name: str | None = None  # Optional, can be specified per request


class OpenAIConfig(BaseModel):
    """OpenAI API configuration (direct or Azure)."""

    api_key: str | None = None  # Direct OpenAI API key
    azure_config: AzureOpenAIConfig | None = None  # Azure OpenAI configuration

    @model_validator(mode="after")
    def check_at_least_one_auth(self) -> Self:
        """Validate that at least one of api_key or azure_config is provided."""
        if self.api_key is not None and not self.api_key.strip():
            self.api_key = None
        if not self.api_key and not self.azure_config:
            raise ValueError(
                "At least one of 'api_key' or 'azure_config' must be provided"
            )
        return self


class AnthropicConfig(BaseModel):
    """Anthropic API configuration."""

    api_key: NonEmptyStr


class OpenRouterConfig(BaseModel):
    """OpenRouter API configuration."""

    api_key: NonEmptyStr


class GeminiConfig(BaseModel):
    """Google Gemini API configuration."""

    api_key: NonEmptyStr


class MiniMaxConfig(BaseModel):
    """MiniMax API configuration."""

    api_key: NonEmptyStr


class DeepSeekConfig(BaseModel):
    """DeepSeek API configuration."""

    api_key: NonEmptyStr


class DashScopeConfig(BaseModel):
    """Alibaba DashScope (Qwen) API configuration."""

    api_key: NonEmptyStr
    api_base: str | None = None  # None = default; set for intl vs China endpoint


class ZAIConfig(BaseModel):
    """Zhipu AI (GLM) API configuration."""

    api_key: NonEmptyStr


class MoonshotConfig(BaseModel):
    """Moonshot (Kimi) API configuration."""

    api_key: NonEmptyStr


class XAIConfig(BaseModel):
    """xAI (Grok) API configuration."""

    api_key: NonEmptyStr


class CustomEndpointConfig(BaseModel):
    """Custom OpenAI-compatible endpoint configuration.

    Args:
        model (str): Model name to use (e.g., 'openai/mistral', 'mistral'). Passed as-is to LiteLLM.
        api_key (str): API key for the custom endpoint.
        api_base (SafeHttpUrl): Base URL of the custom endpoint (e.g., 'http://localhost:8000/v1').
            Validated against SSRF: always blocks cloud metadata endpoints;
            blocks private IPs when REFLEXIO_BLOCK_PRIVATE_URLS=true.
    """

    model: NonEmptyStr
    api_key: NonEmptyStr
    api_base: SafeHttpUrl


class APIKeyConfig(BaseModel):
    """
    API key configuration for LLM providers.

    Supports OpenAI (direct and Azure), Anthropic, OpenRouter, Google Gemini, MiniMax,
    DeepSeek, DashScope (Qwen), Zhipu AI (GLM), Moonshot (Kimi), xAI (Grok), and custom
    OpenAI-compatible endpoints. When custom_endpoint is configured with non-empty fields,
    it takes priority over all other providers for LLM completion calls (but not embeddings).
    """

    custom_endpoint: CustomEndpointConfig | None = None
    openai: OpenAIConfig | None = None
    anthropic: AnthropicConfig | None = None
    openrouter: OpenRouterConfig | None = None
    gemini: GeminiConfig | None = None
    minimax: MiniMaxConfig | None = None
    deepseek: DeepSeekConfig | None = None
    dashscope: DashScopeConfig | None = None
    zai: ZAIConfig | None = None
    moonshot: MoonshotConfig | None = None
    xai: XAIConfig | None = None


class DeduplicationConfig(BaseModel):
    """Configuration for playbook deduplication search parameters.

    Controls the hybrid search behavior when looking for existing playbooks
    to deduplicate against.

    Args:
        search_threshold: Minimum similarity score for search results (0.0-1.0).
        search_top_k: Maximum number of existing playbooks to retrieve per new playbook.
    """

    search_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for deduplication search results.",
    )
    search_top_k: int = Field(
        default=5,
        ge=1,
        description="Maximum number of existing playbooks to retrieve per new playbook.",
    )


class ProfileExtractorConfig(BaseModel):
    extractor_name: NonEmptyStr
    extraction_definition_prompt: SanitizedNonEmptyStr
    context_prompt: str | None = None
    metadata_definition_prompt: str | None = None
    should_extract_profile_prompt_override: str | None = None
    request_sources_enabled: list[str] | None = (
        None  # default enabled for all sources, if set, only extract profiles from the enabled request sources
    )
    manual_trigger: bool = False  # require manual triggering (rerun) to run extraction and skip auto extraction if set to True
    batch_size_override: int | None = Field(default=None, gt=0)
    batch_interval_override: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_field_names(cls, data: Any) -> Any:
        data = _migrate_dict(data, _PROFILE_CONFIG_FIELD_MIGRATION)
        return _migrate_dict(data, _EXTRACTOR_OVERRIDE_MIGRATION)


class PlaybookAggregatorConfig(BaseModel):
    min_cluster_size: int = Field(default=2, ge=1)
    reaggregation_trigger_count: int = Field(default=2, ge=1)
    clustering_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for clustering. Higher = tighter clusters.",
    )
    direction_overlap_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Token overlap threshold for grouping playbooks by direction.",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_field_names(cls, data: Any) -> Any:
        return _migrate_dict(data, _AGGREGATOR_FIELD_MIGRATION)


class UserPlaybookExtractorConfig(BaseModel):
    extractor_name: NonEmptyStr
    extraction_definition_prompt: SanitizedNonEmptyStr
    context_prompt: str | None = None
    metadata_definition_prompt: str | None = None
    aggregation_config: PlaybookAggregatorConfig | None = None
    deduplication_config: DeduplicationConfig | None = None
    request_sources_enabled: list[str] | None = (
        None  # default enabled for all sources, if set, only extract user playbooks from the enabled request sources
    )
    batch_size_override: int | None = Field(default=None, gt=0)
    batch_interval_override: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_field_names(cls, data: Any) -> Any:
        data = _migrate_dict(data, _PLAYBOOK_CONFIG_FIELD_MIGRATION)
        return _migrate_dict(data, _EXTRACTOR_OVERRIDE_MIGRATION)


# Backward-compatible alias (deprecated — use UserPlaybookExtractorConfig)
PlaybookConfig = UserPlaybookExtractorConfig


class ToolUseConfig(BaseModel):
    tool_name: NonEmptyStr
    tool_description: NonEmptyStr


# define what success looks like for agent
class AgentSuccessConfig(BaseModel):
    evaluation_name: NonEmptyStr
    success_definition_prompt: SanitizedNonEmptyStr
    metadata_definition_prompt: str | None = None
    sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0
    )  # fraction of batch of interactions to be sampled for success evaluation
    batch_size_override: int | None = Field(default=None, gt=0)
    batch_interval_override: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_field_names(cls, data: Any) -> Any:
        return _migrate_dict(data, _EXTRACTOR_OVERRIDE_MIGRATION)


class LLMConfig(BaseModel):
    """
    LLM model configuration overrides.

    These settings override the default model names from llm_model_setting.json site variable.
    If a field is None, the default from site variable is used.
    """

    should_run_model_name: str | None = None  # Model for "should run extraction" checks
    generation_model_name: str | None = (
        None  # Model for generation and evaluation tasks
    )
    embedding_model_name: str | None = None  # Model for embedding generation
    pre_retrieval_model_name: str | None = (
        None  # Model for pre-retrieval query reformulation
    )


def _default_profile_extractor_configs() -> list[ProfileExtractorConfig]:
    return [
        ProfileExtractorConfig(
            extractor_name="default_profile_extractor",
            extraction_definition_prompt=(
                "Extract key information about the user and their working "
                "environment: name, role, preferences, and stable facts the "
                "agent needs to know to serve the user correctly — including "
                "data/schema details (table names, column types, units, join "
                "paths), metric definitions the user enforces, and tool "
                "quirks or workarounds the user relies on. Do NOT extract "
                "behavioral rules for the agent (those belong in the "
                "playbook extractor)."
            ),
        ),
    ]


def _default_user_playbook_extractor_configs() -> list[UserPlaybookExtractorConfig]:
    return [
        UserPlaybookExtractorConfig(
            extractor_name="default_playbook_extractor",
            extraction_definition_prompt="Extract playbook rules about agent performance, including areas where the agent was helpful, areas for improvement, and any issues encountered during the interaction.",
        ),
    ]


class Config(BaseModel):
    # define where user configuration is stored at
    storage_config: StorageConfig
    storage_config_test: StorageConfigTest | None = StorageConfigTest.UNKNOWN
    # define agent working environment, tool can use and action space
    agent_context_prompt: str | None = None
    # tools agent can use (shared across success evaluation and playbook extraction)
    tool_can_use: list[ToolUseConfig] | None = None
    # user level memory
    profile_extractor_configs: list[ProfileExtractorConfig] | None = Field(
        default_factory=_default_profile_extractor_configs
    )
    # user playbook extraction
    user_playbook_extractor_configs: list[UserPlaybookExtractorConfig] | None = Field(
        default_factory=_default_user_playbook_extractor_configs
    )
    # agent level success
    agent_success_configs: list[AgentSuccessConfig] | None = None
    # extraction preset — selects bundled batch_size/batch_interval values
    extraction_preset: ExtractionPreset | None = None
    # extraction parameters
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, gt=0)
    batch_interval: int = Field(default=DEFAULT_BATCH_INTERVAL, gt=0)
    # API key configuration for LLM providers
    api_key_config: APIKeyConfig | None = None
    # LLM model configuration overrides
    llm_config: LLMConfig | None = None
    # Skip the LLM pre-extraction eligibility check (always run extraction)
    skip_should_run_check: bool = False
    # Enable storage-time document expansion for improved FTS recall
    enable_document_expansion: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_field_names(cls, data: Any) -> Any:
        """Rename old field names from stored JSON to current names.

        Also strips None values for fields that have non-optional defaults,
        so rows missing these columns fall back to defaults instead of
        failing validation.
        """
        data = _migrate_dict(data, _CONFIG_FIELD_MIGRATION)
        if isinstance(data, dict):
            for key in ("batch_size", "batch_interval"):
                if key in data and data[key] is None:
                    del data[key]
        return data

    @model_validator(mode="after")
    def apply_extraction_preset(self) -> Self:
        """Apply preset values when batch_size/batch_interval are at defaults.

        If a preset is selected but the user also explicitly set batch_size or
        batch_interval, the explicit values win (checked via model_fields_set).
        """
        if self.extraction_preset is None:
            return self

        preset_values = _PRESET_VALUES.get(self.extraction_preset)
        if preset_values is None:
            return self

        preset_batch_size, preset_batch_interval = preset_values
        if "batch_size" not in self.model_fields_set:
            self.batch_size = preset_batch_size
        if "batch_interval" not in self.model_fields_set:
            self.batch_interval = preset_batch_interval

        return self

    @model_validator(mode="after")
    def check_batch_interval_le_batch_size(self) -> Self:
        """Validate that batch_interval <= batch_size."""
        if self.batch_interval > self.batch_size:
            raise ValueError("batch_interval must be <= batch_size")
        return self

    @model_validator(mode="after")
    def ensure_default_extractors(self) -> Self:
        """Populate default extractors if none are configured.

        When Config is deserialized from saved JSON with null/empty extractor
        lists, the default_factory doesn't run. This validator ensures defaults
        are always present so extraction works out of the box.
        """
        if not self.profile_extractor_configs:
            self.profile_extractor_configs = _default_profile_extractor_configs()
        if not self.user_playbook_extractor_configs:
            self.user_playbook_extractor_configs = (
                _default_user_playbook_extractor_configs()
            )
        return self
