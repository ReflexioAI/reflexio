from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from reflexio.config_env import (
    DEFAULT_STORAGE_POOL_ACQUIRE_TIMEOUT,
    DEFAULT_STORAGE_POOL_SIZE,
    StoredConfigurationError,
)
from reflexio.housekeeping_policy import strip_legacy_housekeeping_fields
from reflexio.pending_tool_policy import (
    strip_legacy_pending_tool_timing,
    strip_retired_pending_followup_warning,
)

from .api_schema.validators import (
    NonEmptyStr,
    SafeHttpUrl,
    SanitizedNonEmptyStr,
)

DEFAULT_MAX_MERGE_INVOCATIONS = 5
_SchedulerJitter = Annotated[
    float, Field(ge=0, le=threading.TIMEOUT_MAX, allow_inf_nan=False)
]
_SchedulerThreshold = Annotated[int, Field(ge=1)]
_SchedulerCooldown = Annotated[int, Field(ge=0, le=threading.TIMEOUT_MAX)]


class OptimizerSchedulingPolicy(BaseModel):
    """Validated, immutable admission-time policy for deferred optimizer work."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    scheduler_jitter_seconds: _SchedulerJitter = 1.0
    abort_cooldown_threshold: _SchedulerThreshold = 2
    cooldown_after_aborts_seconds: _SchedulerCooldown = 3600


_DEFAULT_OPTIMIZER_SCHEDULING_POLICY = OptimizerSchedulingPolicy()


# Embedding vector dimensions. Changing this requires a DB migration and re-embedding,
# so it is intentionally a constant rather than a configurable setting.
EMBEDDING_DIMENSIONS = 512

# Default sliding window parameters for extraction
DEFAULT_WINDOW_SIZE = 10
DEFAULT_STRIDE_SIZE = 8


def _reject_obsolete_fields(data: Any, fields: tuple[str, ...]) -> Any:
    """Reject known obsolete inputs before stored extra-ignore can hide them."""
    if isinstance(data, dict):
        rejected = sorted(set(data).intersection(fields))
        if rejected:
            raise ValueError(
                "Obsolete configuration fields: "
                + ", ".join(rejected)
                + ". Use canonical names; explicitly upgrade saved configuration."
            )
    return data


_OBSOLETE_STORAGE_FIELDS = (
    "storage_type",
    "schema_name",
    "url",
    "key",
    "read_url",
    "read_key",
)


def reject_obsolete_config_names(data: Any) -> Any:
    """Admit root names before project/stored filtering can hide obsolete input."""
    return _reject_obsolete_fields(
        data,
        (
            "batch_size",
            "batch_interval",
            "extraction_window_size",
            "extraction_window_stride",
            "extraction_preset",
            "profile_extractor_configs",
            "user_playbook_extractor_configs",
            "playbook_configs",
            "agent_feedback_configs",
            "agent_success_configs",
            "shadow_comparison_judge_prompt_version",
        ),
    )


class _StorageConfigModel(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    @model_validator(mode="before")
    @classmethod
    def reject_obsolete_storage_fields(cls, data: Any) -> Any:
        return _reject_obsolete_fields(data, _OBSOLETE_STORAGE_FIELDS)


def _validate_storage_identity(data: Any) -> Any:
    _reject_obsolete_fields(data, _OBSOLETE_STORAGE_FIELDS)
    if (
        isinstance(data, dict)
        and "type" not in data
        and any(key in data for key in ("db_url", "read_db_url", "managed_by"))
    ):
        raise ValueError("Remote storage requires an explicit canonical type")
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
    fresh: bool = field(default=False)
    rrf_k: int = field(default=60)
    vector_weight: float = field(default=1.0)
    fts_weight: float = field(default=1.0)
    # Automatic consolidation excludes verbatim recovery rows before top-k.
    exclude_source_quoted: bool = field(default=False)


class StorageConfigTest(IntEnum):
    UNKNOWN = 0
    INCOMPLETE = 1
    FAILED = 2
    SUCCEEDED = 3


class StorageConfigSQLite(_StorageConfigModel):
    """SQLite storage configuration.

    ``extra="forbid"`` is load-bearing, not stylistic. ``StorageConfig`` is an
    *untagged* union and this is its only member with no required fields, so
    without it this model is a catch-all: any payload that fails the stricter
    members silently lands here instead of raising. A Supabase config that lost
    its ``db_url`` used to validate as SQLite, dropping ``url``/``key``/``schema``
    on the floor -- which routed a managed tenant to a local file, produced a
    storage object with no ``_rpc``, and surfaced only as a background sweep
    erroring every few seconds (org 56, 2026-08).

    An empty payload still resolves here on purpose: that is the legitimate OSS
    local-mode default (``local_file_config_storage._default_storage_config``).
    """

    model_config = ConfigDict(
        validate_by_name=False,
        validate_by_alias=True,
        serialize_by_alias=True,
        extra="forbid",
    )

    # Publish the canonical backend tag with typed saves. Historical empty local
    # references still select SQLite, but no writer needs an implicit tag.
    storage_type: Literal["sqlite"] = Field(default="sqlite", alias="type")
    db_path: str | None = None  # None = use LOCAL_STORAGE_PATH env var default


class StorageConfigSupabase(_StorageConfigModel):
    # extra="forbid" keeps this distinguishable from the other union members --
    # see StorageConfigSQLite for why an untagged union needs it.
    model_config = ConfigDict(
        validate_by_name=False,
        validate_by_alias=True,
        serialize_by_alias=True,
        extra="forbid",
    )

    # Native Supabase and Postgres share DSN/schema fields but retain distinct
    # pool policies. Persist their identity instead of inferring it from HTTP
    # credentials that no data operation uses anymore.
    storage_type: Literal["supabase"] = Field(default="supabase", alias="type")
    db_url: NonEmptyStr
    schema_name: str | None = Field(default=None, alias="schema")


class StorageConfigPostgres(_StorageConfigModel):
    # extra="forbid" keeps this distinguishable from the other union members --
    # see StorageConfigSQLite for why an untagged union needs it. Without it a
    # Supabase config missing only its ``key`` validated as Postgres and
    # connected by a path nobody intended.
    model_config = ConfigDict(
        validate_by_name=False,
        validate_by_alias=True,
        serialize_by_alias=True,
        extra="forbid",
    )

    storage_type: Literal["postgres"] = Field(default="postgres", alias="type")
    db_url: NonEmptyStr
    schema_name: str | None = Field(default=None, alias="schema")
    pool_size: int = Field(default=DEFAULT_STORAGE_POOL_SIZE, ge=1)
    # Seconds a query waits for a free pooled connection before failing. Bounds the
    # back-pressure applied when concurrent queries exceed pool_size.
    pool_acquire_timeout: float = Field(
        default=DEFAULT_STORAGE_POOL_ACQUIRE_TIMEOUT, gt=0
    )
    read_db_url: NonEmptyStr | None = None
    read_pool_size: int | None = Field(default=None, ge=1)
    read_pool_acquire_timeout: float | None = Field(default=None, gt=0)


class StorageConfigManagedSupabase(_StorageConfigModel):
    """Redacted platform data reference; historical class name stays compatible."""

    model_config = ConfigDict(
        validate_by_name=False,
        validate_by_alias=True,
        serialize_by_alias=True,
        extra="forbid",
    )
    storage_type: Literal["supabase", "postgres"] = Field(
        default="supabase", alias="type"
    )
    managed_by: Literal["platform"]
    schema_present: bool = True


StorageConfig = Annotated[
    StorageConfigSQLite
    | StorageConfigSupabase
    | StorageConfigPostgres
    | StorageConfigManagedSupabase
    | None,
    BeforeValidator(_validate_storage_identity),
]

# Used by validate_stored_config to check storage_config strictly, outside the
# lenient whole-Config pass that would otherwise relax these models' extra rules.
_STORAGE_CONFIG_ADAPTER: TypeAdapter[StorageConfig] = TypeAdapter(
    StorageConfig, config=ConfigDict(hide_input_in_errors=True)
)


def validate_storage_config(data: Any) -> StorageConfig:
    """Validate a canonical reference before any deployment binding replaces it."""
    return _STORAGE_CONFIG_ADAPTER.validate_python(data)


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI specific configuration."""

    api_key: NonEmptyStr
    endpoint: SafeHttpUrl  # e.g., "https://your-resource.openai.azure.com/"
    api_version: str = "2024-02-15-preview"


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

    Controls bounded candidate generation when looking for existing playbooks
    to deduplicate against. The candidate threshold is intentionally independent
    from the embedding model's user-facing retrieval default because an LLM
    performs the final duplicate decision.

    Args:
        strategy: Select legacy four-action or independently audited three-action consolidation.
        search_threshold: Minimum similarity score for deduplication candidates
            (0.0-1.0), independent from user-facing retrieval defaults.
        search_top_k: Maximum number of existing playbooks to retrieve per new playbook.
    """

    strategy: Literal["legacy", "audited"] = Field(
        default="legacy",
        description="Playbook consolidation strategy. audited uses Create/Replace/Remove with independent audit; legacy restores the existing four actions.",
    )
    search_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score for deduplication candidates. This is a "
            "dedup-specific override, independent from the embedding model's "
            "user-facing retrieval default."
        ),
    )
    search_top_k: int = Field(
        default=5,
        ge=1,
        description="Maximum number of existing playbooks to retrieve per new playbook.",
    )


SINGLETON_PROFILE_EXTRACTOR_NAME = "profile"
SINGLETON_USER_PLAYBOOK_NAME = "playbook"
SINGLETON_AGENT_SUCCESS_EVALUATION_NAME = "agent_success"
DEFAULT_AGENT_SUCCESS_SAMPLING_RATE = 0.05
DEFAULT_AGENT_SUCCESS_DEFINITION_PROMPT = (
    "Evaluate whether the AI agent successfully handled the user's session.\n\n"
    "Mark the session successful when, by the end of the conversation, the agent:\n"
    "1. Identified and addressed the user's main goal or question.\n"
    "2. Provided a correct, useful, and actionable response or completed the requested action.\n"
    "Mark the session unsuccessful when the agent failed to understand the request,\n"
    "gave incorrect or unhelpful guidance, did not complete an available action,\n"
    "ignored important constraints, or left the user unsatisfied."
)


class ProfileExtractorConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    extraction_definition_prompt: SanitizedNonEmptyStr
    context_prompt: str | None = None
    tagging_definition_prompt: str | None = None
    should_extract_profile_prompt_override: str | None = None
    request_sources_enabled: list[str] | None = (
        None  # default enabled for all sources, if set, only extract profiles from the enabled request sources
    )
    manual_trigger: bool = False  # require manual triggering (rerun) to run extraction and skip auto extraction if set to True
    window_size_override: int | None = Field(default=None, gt=0)
    stride_size_override: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def reject_obsolete_names(cls, data: Any) -> Any:
        return _reject_obsolete_fields(
            data,
            (
                "profile_content_definition_prompt",
                "batch_size_override",
                "batch_interval_override",
                "extraction_window_size_override",
                "extraction_window_stride_override",
            ),
        )


class PlaybookAggregatorConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    min_cluster_size: int = Field(default=2, ge=1)
    reaggregation_trigger_count: int = Field(default=2, ge=1)
    clustering_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold for clustering. Higher = tighter clusters. "
            "When omitted, Reflexio uses the embedding model's calibrated default."
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
    def reject_obsolete_names(cls, data: Any) -> Any:
        return _reject_obsolete_fields(
            data,
            (
                "min_feedback_threshold",
                "refresh_count",
                "similarity_threshold",
            ),
        )


class UserPlaybookExtractorConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    extraction_definition_prompt: SanitizedNonEmptyStr
    tagging_definition_prompt: str | None = None
    aggregation_config: PlaybookAggregatorConfig = Field(
        default_factory=PlaybookAggregatorConfig
    )
    deduplication_config: DeduplicationConfig | None = None
    request_sources_enabled: list[str] | None = (
        None  # default enabled for all sources, if set, only extract user playbooks from the enabled request sources
    )
    window_size_override: int | None = Field(default=None, gt=0)
    stride_size_override: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def reject_obsolete_names(cls, data: Any) -> Any:
        data = _reject_obsolete_fields(
            data,
            (
                "feedback_definition_prompt",
                "playbook_definition_prompt",
                "feedback_aggregator_config",
                "playbook_aggregator_config",
                "playbook_name",
                "feedback_name",
                "batch_size_override",
                "batch_interval_override",
                "extraction_window_size_override",
                "extraction_window_stride_override",
            ),
        )
        if isinstance(data, dict):
            data = dict(data)
        if (
            isinstance(data, dict)
            and "aggregation_config" in data
            and data["aggregation_config"] is None
        ):
            # Historical configs persisted the absent optional aggregator as
            # JSON null. Aggregation is now default-on, so treat both a missing
            # field and the legacy null representation as "use defaults".
            data.pop("aggregation_config")
        return data


# Backward-compatible alias (deprecated — use UserPlaybookExtractorConfig)
PlaybookConfig = UserPlaybookExtractorConfig


class ToolUseConfig(BaseModel):
    tool_name: NonEmptyStr
    tool_description: NonEmptyStr


# define what success looks like for agent
class AgentSuccessConfig(BaseModel):
    success_definition_prompt: SanitizedNonEmptyStr
    tagging_definition_prompt: str | None = None
    request_sources_enabled: list[str] | None = (
        None  # default enabled for all sources, if set, only evaluate requests from the enabled request sources
    )
    sampling_rate: float = Field(
        default=DEFAULT_AGENT_SUCCESS_SAMPLING_RATE,
        ge=0.0,
        le=1.0,
        description="Fraction of sessions to evaluate automatically.",
    )
    evaluation_only_sampling_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of evaluation-only sessions to evaluate automatically."
            " None inherits sampling_rate."
        ),
    )
    retrieved_learning_sampling_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of sessions on which the retrieved-learning relevance and"
            " impact judge runs. None inherits sampling_rate. Its verdicts back"
            " the retrieved-learning evaluation API and the Evaluations page;"
            " set it independently to grade served learnings more or less"
            " densely than the session-success judge. The offline playbook"
            " tuner samples and judges its own sessions and does not read"
            " these verdicts, so this rate does not affect tuning."
        ),
    )


def _default_agent_success_config() -> AgentSuccessConfig:
    return AgentSuccessConfig(
        success_definition_prompt=DEFAULT_AGENT_SUCCESS_DEFINITION_PROMPT,
        sampling_rate=DEFAULT_AGENT_SUCCESS_SAMPLING_RATE,
    )


class RetrievalFloorConfig(BaseModel):
    """Read-path relevance floor: drop search results below a per-arm cross-encoder score.

    Floors are raw, model-specific cross-encoder logits, not probabilities. ``None``
    selects the calibrated default for the reranker discovered from the inference
    service. Numeric values are exact organization overrides, including ``0.0``.
    """

    candidate_pool_size: int = Field(
        default=0,
        ge=0,
        description="Candidates per arm before flooring; zero disables flooring.",
    )
    profile_floor: float | None = None
    user_playbook_floor: float | None = None
    agent_playbook_floor: float | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_obsolete_policy(cls, data: Any) -> Any:
        return _reject_obsolete_fields(data, ("enabled", "pool_size"))


class PlaybookOptimizerConfig(BaseModel):
    """Configuration for GEPA-backed playbook content optimization.

    The optimizer is opt-in (``enabled=False`` by default) and requires
    *exactly one* assistant backend to actually do anything. The two
    backends are mutually exclusive — see ``check_single_assistant_backend``
    below.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    # --- gating ------------------------------------------------------------
    enabled: bool = False
    agent_playbooks_enabled: bool = False
    user_playbooks_enabled: bool = False

    # --- GEPA budget -------------------------------------------------------
    max_metric_calls: int = Field(default=20, gt=0)
    max_turns: int = Field(default=4, gt=0)
    reflection_minibatch_size: int = Field(default=2, gt=0)
    max_validation_windows: int = Field(default=2, gt=0)
    min_commit_windows: int = Field(default=2, gt=0)
    min_commit_score: float = Field(default=0.75, ge=0.0, le=1.0)
    min_commit_likert: int = Field(default=4, ge=1, le=5)
    # One budget owns both merge admission and its limit; zero disables merging.
    max_merge_invocations: int = Field(default=DEFAULT_MAX_MERGE_INVOCATIONS, ge=0)
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

    # Scheduling and optimizer search budgets have separate owners.
    scheduler_jitter_seconds: _SchedulerJitter = (
        _DEFAULT_OPTIMIZER_SCHEDULING_POLICY.scheduler_jitter_seconds
    )
    abort_cooldown_threshold: _SchedulerThreshold = (
        _DEFAULT_OPTIMIZER_SCHEDULING_POLICY.abort_cooldown_threshold
    )
    cooldown_after_aborts_seconds: _SchedulerCooldown = (
        _DEFAULT_OPTIMIZER_SCHEDULING_POLICY.cooldown_after_aborts_seconds
    )

    @property
    def scheduling_policy(self) -> OptimizerSchedulingPolicy:
        return OptimizerSchedulingPolicy(
            scheduler_jitter_seconds=self.scheduler_jitter_seconds,
            abort_cooldown_threshold=self.abort_cooldown_threshold,
            cooldown_after_aborts_seconds=self.cooldown_after_aborts_seconds,
        )

    @model_validator(mode="before")
    @classmethod
    def reject_obsolete_policy(cls, data: Any) -> Any:
        return _reject_obsolete_fields(
            data,
            (
                "optimize_user_playbooks",
                "auto_update_user_playbooks",
                "optimize_agent_playbooks",
                "auto_update_pending_agent_playbooks",
                "use_merge",
                "early_stop_score",
            ),
        )

    def allows_target(self, kind: Literal["agent_playbook", "user_playbook"]) -> bool:
        """Admit only targets with an enabled optimization and adoption path."""
        if not self.enabled:
            return False
        if kind == "agent_playbook":
            return self.agent_playbooks_enabled
        return self.user_playbooks_enabled

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


class LineageGCConfig(BaseModel):
    """Configuration for the tombstone garbage-collection job (enabled by default).

    Purpose
    -------
    Retains tombstone content for ``tombstone_grace_window_days`` days after the
    row's *retirement instant* (``retired_at``) to support audit, replay of dedup
    and aggregation runs, and rollback.  After the window expires the row is
    hard-deleted and a ``hard_delete`` lineage event is recorded.

    Age basis
    ---------
    The GC ages on ``retired_at`` — the INTEGER epoch set when a row is tombstoned
    (merged, superseded, or archived).  Rows with ``retired_at = NULL`` (created
    before the column was added) are never eligible; they have no retirement clock.

    Grace window
    ------------
    90 days is the default grace window.  This matches common 90-day soft-delete
    retention policies and satisfies GDPR Art. 5(1)(e) storage-limitation for
    personal data in profiles.  The value is a per-deployment policy knob; ratify
    with your DPO before shortening it in production.  The 90-day floor also
    preserves tombstones long enough for B3 changelog replay and rollback
    consumers — raise ``tombstone_grace_window_days`` further if your replay
    horizon exceeds 90 days.

    Enabled by default
    ------------------
    GC is ON by default so tombstones created by the soft-delete flags (also ON by
    default) are reclaimed automatically.  Disabling GC while soft-delete is enabled
    allows tombstone counts to grow without bound — only do this deliberately (e.g.
    extended audit hold) and with a plan to re-enable.

    Disabling
    ---------
    Set ``enabled = False`` in your deployment config to hold all tombstones
    indefinitely (e.g. for an extended audit window or rollback standby period).
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    enabled: bool = True
    tombstone_grace_window_days: int = Field(default=90, gt=0)

    @model_validator(mode="before")
    @classmethod
    def discard_legacy_execution_settings(cls, data: Any) -> Any:
        """Accept former defaults, reject custom execution policy inputs."""
        return strip_legacy_housekeeping_fields({"lineage_gc": data})["lineage_gc"]


class ExpiryReclamationConfig(BaseModel):
    """Direct-delete reclamation of expired plain rows (non-audited).

    Independent of ``lineage_gc``: these rows carry no PII/audit/grace obligation,
    so they can be reclaimed whenever this is enabled even if tombstone GC is off.

    Opt-in by default (``enabled=False``) so operators control a staged rollout
    of the direct-delete Class B sweeps and are not surprised by deletions on
    upgrade.  ``lineage_gc.enabled`` (which defaults to True) is unaffected and
    continues to drive the Class A profile-expiry and tombstone-GC paths.
    """

    enabled: bool = False


class GovernanceRetentionConfig(BaseModel):
    """Audit-event retention policy. **Enterprise-only:** reclamation is performed
    by an enterprise per-org reclamation sweep registered via
    ``register_per_org_sweep``. In an OSS-only deployment these knobs are accepted
    but inert (the OSS lineage scheduler does not reclaim audit events) and the
    server logs a startup warning when retention is enabled.
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    audit_events_max_age_days: int = Field(
        default=0,
        ge=0,
        description="Maximum audit-event age in days; zero retains indefinitely.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_obsolete_retention_policy(cls, data: Any) -> Any:
        return _reject_obsolete_fields(
            data, ("audit_events_retention_enabled", "audit_events_retention_days")
        )

    audit_events_delete_batch_limit: int = Field(
        default=500,
        gt=0,
        le=(1 << 31) - 1,
        description="Maximum expired rows deleted per table in one governance sweep.",
    )


@dataclass(frozen=True)
class EffectivePendingToolCallConfig:
    """Resolved pending-tool-call settings after applying tool overrides."""

    pending_ttl_seconds: int
    dedup_cache_seconds: int
    prior_answer_valid_seconds: int
    similarity_threshold: float


class PendingToolCallToolOverrideConfig(BaseModel):
    """Optional per-tool pending-call limits."""

    pending_ttl_seconds: int | None = Field(default=None, gt=0)
    dedup_cache_seconds: int | None = Field(default=None, gt=0)
    prior_answer_valid_seconds: int | None = Field(default=None, gt=0)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PendingToolCallConfig(BaseModel):
    """Configuration for non-blocking pending tool calls."""

    enabled: bool = False
    pending_ttl_seconds: int = Field(default=86_400, gt=0)
    dedup_cache_seconds: int = Field(default=300, gt=0)
    prior_answer_valid_seconds: int = Field(default=2_592_000, gt=0)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    max_resume_attempts: int = Field(default=3, ge=0)
    max_finalization_attempts: int = Field(default=3, ge=0)
    hmac_secrets: list[str] = Field(default_factory=list)
    tool_overrides: dict[str, PendingToolCallToolOverrideConfig] = Field(
        default_factory=dict
    )

    @model_validator(mode="before")
    @classmethod
    def discard_retired_execution_policy(cls, data: Any) -> Any:
        return strip_retired_pending_followup_warning(
            strip_legacy_pending_tool_timing(data)
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
    # Wall-clock seconds allowed for one generation/evaluation LLM call. None
    # takes the module default in reflexio.lib._base, which is deliberately
    # higher than LiteLLMConfig.timeout: this client backs background
    # generation, aggregation and session evaluation, and real evaluations were
    # measured at 145-265s against a local-CLI bridge (claude-smart#162).
    generation_timeout_seconds: int | None = Field(default=None, gt=0)


class RetrievalExperimentConfig(BaseModel):
    """The single retrieval holdout experiment currently serving traffic."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: NonEmptyStr = Field(max_length=128)
    holdout_percentage: float = Field(gt=0, lt=100)


class RetrievalExperimentRecord(RetrievalExperimentConfig):
    """Immutable lifecycle metadata retained after an experiment stops."""

    started_at: int = Field(ge=0)
    ended_at: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Self:
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at must be greater than or equal to started_at")
        return self


def _default_profile_extractor_config() -> ProfileExtractorConfig:
    return ProfileExtractorConfig(
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
        extraction_definition_prompt="Extract playbook rules about agent performance, including areas where the agent was helpful, areas for improvement, and any issues encountered during the interaction.",
    )


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

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
    agent_success_config: AgentSuccessConfig | None = Field(
        default_factory=_default_agent_success_config
    )
    # extraction parameters
    window_size: int = Field(default=DEFAULT_WINDOW_SIZE, gt=0)
    stride_size: int = Field(default=DEFAULT_STRIDE_SIZE, gt=0)
    # Captured at admission; 0 disables idle finalization for new publications.
    extraction_idle_timeout_seconds: int = Field(default=21600, ge=0)
    # API key configuration for LLM providers
    api_key_config: APIKeyConfig | None = None
    # LLM model configuration overrides
    llm_config: LLMConfig | None = None
    # One active user-randomized retrieval holdout experiment. Lifecycle
    # endpoints own this field and the append-only history below so generic
    # settings saves cannot silently erase an experiment.
    retrieval_experiment_config: RetrievalExperimentConfig | None = None
    retrieval_experiment_history: list[RetrievalExperimentRecord] = Field(
        default_factory=list
    )
    # Read-path relevance floor (per-arm cross-encoder score cutoff)
    retrieval_floor: RetrievalFloorConfig = Field(default_factory=RetrievalFloorConfig)
    # Optional GEPA-backed playbook content optimizer
    playbook_optimizer_config: PlaybookOptimizerConfig = Field(
        default_factory=PlaybookOptimizerConfig
    )
    # Tombstone GC job gate (opt-in, off by default — see LineageGCConfig)
    lineage_gc: LineageGCConfig = Field(default_factory=LineageGCConfig)
    # Direct-delete reclamation of expired plain rows (share links, pending tool calls)
    expiry_reclamation: ExpiryReclamationConfig = Field(
        default_factory=ExpiryReclamationConfig
    )
    governance_retention: GovernanceRetentionConfig = Field(
        default_factory=GovernanceRetentionConfig
    )
    # Optional non-blocking async information tools for classic extraction.
    pending_tool_call_config: PendingToolCallConfig = Field(
        default_factory=PendingToolCallConfig
    )
    # Skip the LLM pre-extraction eligibility check (always run extraction)
    skip_should_run_check: bool = False
    # Enable storage-time document expansion for improved FTS recall
    enable_document_expansion: bool = False
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

    @model_validator(mode="before")
    @classmethod
    def reject_obsolete_names(cls, data: Any) -> Any:
        """Admit canonical names and preserve established null-default semantics."""
        data = reject_obsolete_config_names(data)
        if isinstance(data, dict):
            data = dict(data)
        if isinstance(data, dict):
            for key in (
                "window_size",
                "stride_size",
                "playbook_optimizer_config",
                "lineage_gc",
                "expiry_reclamation",
                "governance_retention",
                "pending_tool_call_config",
                "retrieval_floor",
            ):
                if key in data and data[key] is None:
                    del data[key]
        return data

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


def validate_stored_config(data: dict[str, Any]) -> Config:
    """Validate persisted config with schema-evolution read compatibility.

    Persisted JSON can contain fields that were valid when it was written but
    have since been deleted. Ignore only those unknown fields while retaining
    normal validation for recognized values. Missing fields continue to use
    their current schema defaults. Invalid recognized values raise a sanitized
    StoredConfigurationError so loaders cannot replace saved policy with defaults.

    API writes intentionally construct ``Config`` directly and therefore keep
    the model's strict ``extra="forbid"`` behavior.
    """
    try:
        # Normal reads validate canonical input; only explicit upgrades translate.
        normalized = data
        # extra="ignore" cascades into nested models; validate storage strictly
        # first so a malformed remote config cannot become a local backend.
        storage_config = normalized.get("storage_config")
        if isinstance(storage_config, dict):
            try:
                _STORAGE_CONFIG_ADAPTER.validate_python(storage_config)
            except ValidationError:
                raise StoredConfigurationError(
                    "Stored configuration failed validation at storage_config. "
                    "Correct the saved values before continuing."
                ) from None
        return Config.model_validate(normalized, extra="ignore")
    except ValidationError as exc:
        # Only the first location component is an authoritative Config field.
        # Nested components can be supplied map keys, even when their spelling
        # happens to match a property declared elsewhere in the schema.
        paths = sorted(
            {
                str(error["loc"][0])
                for error in exc.errors(include_input=False, include_context=False)
                if error["loc"] and error["loc"][0] in Config.model_fields
            }
        ) or ["configuration"]
        raise StoredConfigurationError(
            "Stored configuration failed validation at "
            + ", ".join(paths[:10])
            + ". Correct the saved values before continuing."
        ) from None
