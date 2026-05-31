from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .api_schema.validators import (
    NonEmptyStr,
    SafeHttpUrl,
    SanitizedNonEmptyStr,
)

# Embedding vector dimensions. Changing this requires a DB migration and re-embedding,
# so it is intentionally a constant rather than a configurable setting.
EMBEDDING_DIMENSIONS = 512

# Default sliding window parameters for extraction
DEFAULT_WINDOW_SIZE = 10
DEFAULT_STRIDE_SIZE = 5

# Deprecated aliases kept for older imports.
DEFAULT_BATCH_SIZE = DEFAULT_WINDOW_SIZE
DEFAULT_BATCH_INTERVAL = DEFAULT_STRIDE_SIZE


class ExtractionPreset(StrEnum):
    """Named extraction presets that bundle window_size and stride_size.

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


# Preset parameter values: (window_size, stride_size)
_PRESET_VALUES: dict[ExtractionPreset, tuple[int, int]] = {
    ExtractionPreset.QUICK_CHAT: (5, 3),
    ExtractionPreset.STANDARD: (DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE_SIZE),
    ExtractionPreset.LONG_FORM: (25, 10),
    ExtractionPreset.HIGH_VOLUME: (15, 8),
}


# ---------------------------------------------------------------------------
# Field migration maps (old stored JSON name → new Python attr name)
# ---------------------------------------------------------------------------
_CONFIG_FIELD_MIGRATION: dict[str, str] = {
    "batch_size": "window_size",
    "batch_interval": "stride_size",
    "extraction_window_size": "window_size",
    "extraction_window_stride": "stride_size",
}

_AGGREGATOR_FIELD_MIGRATION: dict[str, str] = {
    "min_feedback_threshold": "min_cluster_size",
    "refresh_count": "reaggregation_trigger_count",
    "similarity_threshold": "clustering_similarity",
}

_EXTRACTOR_OVERRIDE_MIGRATION: dict[str, str] = {
    "batch_size_override": "window_size_override",
    "batch_interval_override": "stride_size_override",
    "extraction_window_size_override": "window_size_override",
    "extraction_window_stride_override": "stride_size_override",
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


# Retired list-valued config fields and the singular field that replaced them.
# The first configured entry wins when an old list contains multiple items.
_LEGACY_SINGLE_CONFIG_FIELDS: tuple[tuple[str, str], ...] = (
    ("profile_extractor_configs", "profile_extractor_config"),
    ("user_playbook_extractor_configs", "user_playbook_extractor_config"),
    ("playbook_configs", "user_playbook_extractor_config"),
    ("agent_feedback_configs", "user_playbook_extractor_config"),
    ("agent_success_configs", "agent_success_config"),
)


def _first_config_entry(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def normalize_legacy_config_shape(data: dict[str, Any]) -> dict[str, Any]:
    """Map retired list-valued config fields onto current singular fields.

    This is a stored-data upgrade path applied at storage load boundaries: any
    config persisted before the single-extractor refactor still carries list
    keys (e.g. ``agent_success_configs``) that ``Config`` would otherwise drop
    as unknown fields, silently losing the user's customization. Legacy keys are
    removed from the returned payload and the first configured entry wins.

    Returns a shallow copy; the caller's dict is not mutated.
    """
    normalized = dict(data)
    for legacy_field, current_field in _LEGACY_SINGLE_CONFIG_FIELDS:
        if legacy_field not in normalized:
            continue
        if current_field not in normalized:
            normalized[current_field] = _first_config_entry(normalized[legacy_field])
        del normalized[legacy_field]
    return normalized


class _ExtractorWindowOverrideCompatMixin:
    @property
    def batch_size_override(self) -> int | None:
        """Deprecated alias for window_size_override."""
        return self.window_size_override  # type: ignore[attr-defined]

    @batch_size_override.setter
    def batch_size_override(self, value: int | None) -> None:
        self.window_size_override = value  # type: ignore[attr-defined]

    @property
    def batch_interval_override(self) -> int | None:
        """Deprecated alias for stride_size_override."""
        return self.stride_size_override  # type: ignore[attr-defined]

    @batch_interval_override.setter
    def batch_interval_override(self, value: int | None) -> None:
        self.stride_size_override = value  # type: ignore[attr-defined]


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

    db_path: str | None = None  # None = use LOCAL_STORAGE_PATH env var default


class StorageConfigSupabase(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    url: NonEmptyStr
    key: NonEmptyStr
    db_url: NonEmptyStr
    schema_name: str | None = Field(default=None, alias="schema")


class StorageConfigPostgres(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    storage_type: Literal["postgres"] = Field(default="postgres", alias="type")
    db_url: NonEmptyStr
    schema_name: str | None = Field(default=None, alias="schema")
    pool_size: int = Field(default=10, ge=1)
    # Seconds a query waits for a free pooled connection before failing. Bounds the
    # back-pressure applied when concurrent queries exceed pool_size.
    pool_acquire_timeout: float = Field(default=30.0, gt=0)


class StorageConfigManagedSupabase(BaseModel):
    """Redacted API response for platform-managed Supabase storage."""

    managed_by: Literal["platform"]
    schema_present: bool = True


StorageConfig = (
    StorageConfigSQLite
    | StorageConfigSupabase
    | StorageConfigPostgres
    | StorageConfigManagedSupabase
    | None
)


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


class ProfileExtractorConfig(_ExtractorWindowOverrideCompatMixin, BaseModel):
    extractor_name: NonEmptyStr
    extraction_definition_prompt: SanitizedNonEmptyStr
    context_prompt: str | None = None
    metadata_definition_prompt: str | None = None
    should_extract_profile_prompt_override: str | None = None
    request_sources_enabled: list[str] | None = (
        None  # default enabled for all sources, if set, only extract profiles from the enabled request sources
    )
    manual_trigger: bool = False  # require manual triggering (rerun) to run extraction and skip auto extraction if set to True
    window_size_override: int | None = Field(default=None, gt=0)
    stride_size_override: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_field_names(cls, data: Any) -> Any:
        data = _migrate_dict(data, _PROFILE_CONFIG_FIELD_MIGRATION)
        return _migrate_dict(data, _EXTRACTOR_OVERRIDE_MIGRATION)


class PlaybookAggregatorConfig(BaseModel):
    min_cluster_size: int = Field(default=2, ge=1)
    reaggregation_trigger_count: int = Field(default=2, ge=1)
    clustering_similarity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold for clustering. Higher = tighter clusters. "
            "Default 0.3 is a compromise that works for both cloud embeddings "
            "(OpenAI text-embedding-3-*, Gemini) and the local zero-padded "
            "MiniLM-L6-v2 embedder. Cloud embeddings typically tolerate 0.4-0.6; "
            "the local embedder's 384-dim vectors zero-padded to 512 produce "
            "lower cosine similarities and need ~0.15-0.3 to cluster at all."
        ),
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


class UserPlaybookExtractorConfig(_ExtractorWindowOverrideCompatMixin, BaseModel):
    extractor_name: NonEmptyStr
    extraction_definition_prompt: SanitizedNonEmptyStr
    context_prompt: str | None = None
    metadata_definition_prompt: str | None = None
    aggregation_config: PlaybookAggregatorConfig | None = None
    deduplication_config: DeduplicationConfig | None = None
    request_sources_enabled: list[str] | None = (
        None  # default enabled for all sources, if set, only extract user playbooks from the enabled request sources
    )
    window_size_override: int | None = Field(default=None, gt=0)
    stride_size_override: int | None = Field(default=None, gt=0)

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
class AgentSuccessConfig(_ExtractorWindowOverrideCompatMixin, BaseModel):
    evaluation_name: NonEmptyStr
    success_definition_prompt: SanitizedNonEmptyStr
    metadata_definition_prompt: str | None = None
    sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0
    )  # fraction of window of interactions to be sampled for success evaluation
    window_size_override: int | None = Field(default=None, gt=0)
    stride_size_override: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_field_names(cls, data: Any) -> Any:
        return _migrate_dict(data, _EXTRACTOR_OVERRIDE_MIGRATION)


class ReflectionConfig(BaseModel):
    """Configuration for the sliding-window reflection step.

    Reflection runs inside ``GenerationService.run`` as its own
    sliding-window step (window = global ``window_size``, stride = global
    ``stride_size``, bookmark via ``OperationStateManager``). When
    the gate opens and at least one Assistant interaction in the window
    cites a current user playbook / user profile row, the LLM is asked
    whether any cited rows should be replaced. When ``enabled`` is
    False the step short-circuits.

    Args:
        enabled (bool): Master switch. When False, no LLM call is made.
        model (str | None): Optional model name override. Falls back to
            ``LLMConfig.generation_model_name`` and then the site
            default for ``ModelRole.GENERATION`` when None.
        post_horizon_size (int): Minimum interactions after a citation before
            reflection judges it with full confidence. Citations near the recent
            edge of the window with fewer than this many follow-up turns get a
            'last_chance' judgment with the prompt biased toward no_change.
            Set to 0 to disable the filter (legacy behavior).
    """

    enabled: bool = True
    model: str | None = None
    post_horizon_size: int = Field(
        default=3,
        description=(
            "Minimum interactions after a citation before reflection judges "
            "it with full confidence. Citations near the recent edge of the "
            "window with fewer than this many follow-up turns get a "
            "'last_chance' judgment with the prompt biased toward no_change. "
            "Set to 0 to disable the filter (legacy behavior)."
        ),
        ge=0,
    )


class PlaybookOptimizerConfig(BaseModel):
    """Configuration for GEPA-backed playbook content optimization.

    The optimizer is opt-in (``enabled=False`` by default) and requires
    *exactly one* assistant backend to actually do anything. The two
    backends are mutually exclusive — see ``check_single_assistant_backend``
    below.
    """

    # --- gating ------------------------------------------------------------
    enabled: bool = False
    optimize_agent_playbooks: bool = False
    optimize_user_playbooks: bool = False
    auto_update_pending_agent_playbooks: bool = True
    auto_update_user_playbooks: bool = False

    # --- GEPA budget -------------------------------------------------------
    max_metric_calls: int = Field(default=20, gt=0)
    max_turns: int = Field(default=4, gt=0)
    early_stop_score: float = Field(default=0.9, ge=0.0, le=1.0)
    reflection_minibatch_size: int = Field(default=2, gt=0)
    max_validation_windows: int = Field(default=2, gt=0)
    min_commit_windows: int = Field(default=2, gt=0)
    min_commit_score: float = Field(default=0.75, ge=0.0, le=1.0)
    min_commit_likert: int = Field(default=4, ge=1, le=5)
    use_merge: bool = True
    max_merge_invocations: int = Field(default=5, ge=0)
    reflection_model: str | None = None

    # --- assistant backend: webhook ---------------------------------------
    webhook_url: str | None = None
    webhook_auth_header: str | None = None
    # The webhook_* timeout/retry/backoff fields apply to BOTH backends —
    # the prefix is preserved purely to avoid a config-schema migration.
    webhook_timeout_seconds: int = Field(default=60, gt=0)
    webhook_max_retries: int = Field(default=3, ge=0)
    webhook_backoff_base_seconds: float = Field(default=1.0, ge=0.0)

    # --- assistant backend: local script ----------------------------------
    # Absolute path to the executable. The optimizer spawns
    # [assistant_script_path, *assistant_script_args] per turn, hands the
    # rollout payload over stdin, and reads {"content": "..."} from stdout.
    # See playbook_optimizer/assistant_webhook.py::LocalScriptAssistant.
    assistant_script_path: str | None = None
    assistant_script_args: list[str] = Field(default_factory=list)

    # --- scheduler --------------------------------------------------------
    scheduler_jitter_seconds: float = Field(default=1.0, ge=0.0)
    cooldown_after_aborts_seconds: int = Field(default=3600, ge=0)
    abort_cooldown_threshold: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def check_single_assistant_backend(self) -> Self:
        # Two backends configured at once would create ambiguous behavior
        # (which one wins?), so reject at load time rather than picking one
        # silently in _create_assistant.
        if self.webhook_url and self.assistant_script_path:
            raise ValueError(
                "Configure only one playbook optimizer assistant backend: "
                "webhook_url or assistant_script_path"
            )
        return self


@dataclass(frozen=True)
class EffectivePendingToolCallConfig:
    """Resolved pending-tool-call settings after applying tool overrides."""

    max_pending_followups_per_scope: int
    pending_ttl_seconds: int
    dedup_cache_seconds: int
    prior_answer_valid_seconds: int
    similarity_threshold: float


class PendingToolCallToolOverrideConfig(BaseModel):
    """Optional per-tool pending-call limits."""

    max_pending_followups_per_scope: int | None = Field(default=None, gt=0)
    pending_ttl_seconds: int | None = Field(default=None, gt=0)
    dedup_cache_seconds: int | None = Field(default=None, gt=0)
    prior_answer_valid_seconds: int | None = Field(default=None, gt=0)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PendingToolCallConfig(BaseModel):
    """Configuration for non-blocking pending tool calls."""

    enabled: bool = False
    max_pending_followups_per_scope: int = Field(default=10, gt=0)
    pending_ttl_seconds: int = Field(default=86_400, gt=0)
    dedup_cache_seconds: int = Field(default=300, gt=0)
    prior_answer_valid_seconds: int = Field(default=2_592_000, gt=0)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    resume_poll_interval_seconds: float = Field(default=5.0, gt=0)
    resume_claim_ttl_seconds: int = Field(default=600, gt=0)
    max_resume_attempts: int = Field(default=3, ge=0)
    max_finalization_attempts: int = Field(default=3, ge=0)
    hmac_secrets: list[str] = Field(default_factory=list)
    tool_overrides: dict[str, PendingToolCallToolOverrideConfig] = Field(
        default_factory=dict
    )

    def for_tool(self, tool_name: str) -> EffectivePendingToolCallConfig:
        """Return base settings with an optional exact tool-name override."""
        override = self.tool_overrides.get(tool_name)

        def _value(name: str) -> Any:
            if override is not None:
                override_value = getattr(override, name)
                if override_value is not None:
                    return override_value
            return getattr(self, name)

        return EffectivePendingToolCallConfig(
            max_pending_followups_per_scope=_value("max_pending_followups_per_scope"),
            pending_ttl_seconds=_value("pending_ttl_seconds"),
            dedup_cache_seconds=_value("dedup_cache_seconds"),
            prior_answer_valid_seconds=_value("prior_answer_valid_seconds"),
            similarity_threshold=_value("similarity_threshold"),
        )


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


def _default_profile_extractor_config() -> ProfileExtractorConfig:
    return ProfileExtractorConfig(
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
    )


def _default_user_playbook_extractor_config() -> UserPlaybookExtractorConfig:
    return UserPlaybookExtractorConfig(
        extractor_name="default_playbook_extractor",
        extraction_definition_prompt="Extract playbook rules about agent performance, including areas where the agent was helpful, areas for improvement, and any issues encountered during the interaction.",
    )


class Config(BaseModel):
    # define where user configuration is stored at
    storage_config: StorageConfig
    storage_config_test: StorageConfigTest | None = StorageConfigTest.UNKNOWN
    # define agent working environment, tool can use and action space
    agent_context_prompt: str | None = None
    # tools agent can use (shared across success evaluation and playbook extraction)
    tool_can_use: list[ToolUseConfig] | None = None
    # user level memory
    profile_extractor_config: ProfileExtractorConfig | None = Field(
        default_factory=_default_profile_extractor_config
    )
    # user playbook extraction
    user_playbook_extractor_config: UserPlaybookExtractorConfig | None = Field(
        default_factory=_default_user_playbook_extractor_config
    )
    # agent level success
    agent_success_config: AgentSuccessConfig | None = None
    # extraction preset — selects bundled window_size/stride_size values
    extraction_preset: ExtractionPreset | None = None
    # extraction parameters
    window_size: int = Field(default=DEFAULT_WINDOW_SIZE, gt=0)
    stride_size: int = Field(default=DEFAULT_STRIDE_SIZE, gt=0)
    # API key configuration for LLM providers
    api_key_config: APIKeyConfig | None = None
    # LLM model configuration overrides
    llm_config: LLMConfig | None = None
    # Post-publish reflection service configuration
    reflection_config: ReflectionConfig = Field(default_factory=ReflectionConfig)
    # Optional GEPA-backed playbook content optimizer
    playbook_optimizer_config: PlaybookOptimizerConfig = Field(
        default_factory=PlaybookOptimizerConfig
    )
    # Optional non-blocking async information tools for classic extraction.
    pending_tool_call_config: PendingToolCallConfig = Field(
        default_factory=PendingToolCallConfig
    )
    # Skip the LLM pre-extraction eligibility check (always run extraction)
    skip_should_run_check: bool = False
    # Enable storage-time document expansion for improved FTS recall
    enable_document_expansion: bool = False
    # Whether this org has opted into shadow-mode runs. Drives /healthz/eval
    # liveness derivation and the /api/get_evaluation_overview hero state
    # machine. When True, each publish optionally schedules a parallel
    # "without Reflexio" generation for side-by-side comparison.
    shadow_mode_enabled: bool = False
    eval_sample_n_per_stratum: int = Field(
        default=200,
        gt=0,
        description=(
            "F3: stratified-sample cap per (day × group) stratum in the regen "
            "pipeline. Strata with fewer items are kept whole. Predictable cost "
            "regardless of traffic volume."
        ),
    )
    eval_concurrency_limit: int = Field(
        default=10,
        gt=0,
        description=(
            "F3: max simultaneous LLM judge calls in flight per regen job, "
            "enforced via a ThreadPoolExecutor. Bound to respect provider "
            "rate limits."
        ),
    )
    shadow_comparison_judge_prompt_version: NonEmptyStr = Field(
        default="v1.0.0",
        description=(
            "F1: pinned judge prompt version for per-turn shadow comparison. "
            "Verdicts are stored with the version that produced them; the "
            "dashboard filters to this org's current pinned version so a "
            "future rubric bump doesn't silently mix epochs into the headline."
        ),
    )

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
            for key in (
                "window_size",
                "stride_size",
                "reflection_config",
                "playbook_optimizer_config",
                "pending_tool_call_config",
            ):
                if key in data and data[key] is None:
                    del data[key]
        return data

    @model_validator(mode="after")
    def apply_extraction_preset(self) -> Self:
        """Apply preset values when window_size/stride_size are at defaults.

        If a preset is selected but the user also explicitly set window_size or
        stride_size, the explicit values win (checked via model_fields_set).
        """
        if self.extraction_preset is None:
            return self

        preset_values = _PRESET_VALUES.get(self.extraction_preset)
        if preset_values is None:
            return self

        preset_window_size, preset_stride_size = preset_values
        if "window_size" not in self.model_fields_set:
            self.window_size = preset_window_size
        if "stride_size" not in self.model_fields_set:
            self.stride_size = preset_stride_size

        return self

    @model_validator(mode="after")
    def check_stride_size_le_window_size(self) -> Self:
        """Validate that stride_size <= window_size."""
        if self.stride_size > self.window_size:
            raise ValueError("stride_size must be <= window_size")
        return self

    @model_validator(mode="after")
    def check_pending_tool_calls_storage_backend(self) -> Self:
        """Pending tool calls require a database-backed storage backend.

        ``storage_config is None`` is allowed: in enterprise deployments storage
        is configured centrally (via ``REFLEXIO_STORAGE``) and the per-org config
        blob carries ``None`` rather than a concrete backend. The only removed
        non-database backend (``disk``) is no longer representable as a
        ``StorageConfig``, so a ``None`` here always denotes a deployment-managed
        database backend (sqlite/supabase/postgres).
        """
        if (
            self.pending_tool_call_config.enabled
            and self.storage_config is not None
            and not isinstance(
                self.storage_config,
                (
                    StorageConfigSQLite,
                    StorageConfigSupabase,
                    StorageConfigPostgres,
                    StorageConfigManagedSupabase,
                ),
            )
        ):
            raise ValueError(
                "pending_tool_call_config.enabled requires sqlite, supabase, or postgres storage"
            )
        return self

    @property
    def batch_size(self) -> int:
        """Deprecated alias for window_size."""
        return self.window_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.window_size = value

    @property
    def batch_interval(self) -> int:
        """Deprecated alias for stride_size."""
        return self.stride_size

    @batch_interval.setter
    def batch_interval(self, value: int) -> None:
        self.stride_size = value
