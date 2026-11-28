"""Shared distribution upgrade diagnostics; removed names never resolve values."""

from collections.abc import Mapping
from dataclasses import dataclass

# The canonical scope column of ``REMOVED_ENV``: removed name -> the key that
# ``reject_removed_env(..., canonical=...)`` matches (empty when no canonical
# reader owns the name). It stays a literal dict, not one derived from the
# records, because reflexio-enterprise's configuration_inventory scanner reads
# this assignment with ast.literal_eval. Callers extend it with their own names.
REMOVED_ENV_ALIASES = {
    "RATELIMIT_STORAGE_OPTIONS": "",
    "RATELIMIT_HEADERS_ENABLED": "",
    "RATELIMIT_HEADER_LIMIT": "",
    "RATELIMIT_HEADER_REMAINING": "",
    "RATELIMIT_HEADER_RESET": "",
    "RATELIMIT_HEADER_RETRY_AFTER": "",
    "RATELIMIT_HEADER_RETRY_AFTER_VALUE": "",
    "RATELIMIT_IN_MEMORY_FALLBACK_ENABLED": "RATELIMIT_IN_MEMORY_FALLBACK",
    "REFLEXIO_EMBEDDING_REGENERATION_BATCH_SIZE": "regeneration request batch_size",
    "REFLEXIO_MAX_INTERACTION_CONTENT_TOKENS": "",
    "REFLEXIO_LLM_SEED": "",
    "REFLEXIO_AGENT_VERSION": "",
    "REFLEXIO_INSTANCE_ID": "",
    "REFLEXIO_DEFAULT_ORG_ID": "config bind-local",
    "REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS": "",
    "REFLEXIO_LEARNING_ALARM_SUPPRESS_SECONDS": "",
    "POOLED_CREDITS_ENABLED": "BILLING_ENFORCEMENT_MODE",
    "TRANSFORMERS_OFFLINE": "HF_HUB_OFFLINE",
    "REFLEXIO_FERNET_KEYS": "FERNET_KEYS",
    "REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS": "REFLEXIO_EMBED_BATCH_SIZE",
    "REFLEXIO_ROW_LIMIT_OFFLINE_TUNER_REWARD_LABEL": "",
    "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOK_SOURCE_USER_PLAYBOOKS": "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOKS",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_JOBS": "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_CANDIDATES": "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_EVALUATIONS": "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_EVENTS": "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY",
    "REFLEXIO_API_QUERY_EMBEDDING_ENABLED": "REFLEXIO_INFERENCE_WORKLOAD",
    "NOMIC_EMBED_DEVICE": "REFLEXIO_EMBEDDING_DEVICE",
    "REFLEXIO_SEARCH_RERANK_SERVICE_URL": "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL",
    "REFLEXIO_SEARCH_DEADLINE_ENABLED": "REFLEXIO_SEARCH_TIMEOUT_SECONDS",
    "REFLEXIO_SEARCH_CONCURRENCY_TIMEOUT_SECONDS": "REFLEXIO_SEARCH_TIMEOUT_SECONDS",
    "REFLEXIO_POSTGRES_STATEMENT_TIMEOUT_MS": "REFLEXIO_DB_LEASE_TIMEOUT_SECONDS",
    "CLAUDE_SMART_USE_LOCAL_EMBEDDING": "REFLEXIO_EMBEDDING_PROVIDER",
    "PUBLIC_SCHEMA_AUDIT_ENABLED": "PUBLIC_SCHEMA_AUDIT_INTERVAL_SECONDS",
    "REFLEXIO_SEARCH_RECENCY_ENABLED": "REFLEXIO_SEARCH_RECENCY_POOL_SIZE",
    "REFLEXIO_SEARCH_JUDGE": "REFLEXIO_SEARCH_JUDGE_API_KEY",
    "REFLEXIO_LLM_FAIL_CLOSED_PROVIDERS": "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY",
    "DEBUG_LOG_TO_CONSOLE": "REFLEXIO_LOG_PROFILE",
    "DATA_SUPABASE_DB_URL": "DATA_DB_URL",
    "POSTGRES_DB_URL": "DATA_DB_URL",
    "WEB_CONCURRENCY": "server --workers N",
    "REFLEXIO_SERVER_WORKERS": "server --workers N",
    "INTERACTION_CLEANUP_THRESHOLD": "REFLEXIO_ROW_LIMIT_INTERACTIONS",
    "REFLEXIO_EMBED_TORCH_THREADS": "REFLEXIO_NATIVE_COMPUTE_THREADS",
    "REFLEXIO_MISSING_VECTOR_BACKFILL_ENABLED": "REFLEXIO_MISSING_VECTOR_BACKFILL_CAP",
    "REFLEXIO_PUBLISH_TIMING_ENABLED": "REFLEXIO_PUBLISH_TIMING_MIN_MS",
    "REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS": "REFLEXIO_PUBLISH_TIMING_MIN_MS",
    "CONTROL_PLANE_INGEST_URL": "CONTROL_PLANE_URL",
    "TOPUP_RECONCILE_ENABLED": "CREDIT_INVOICE_RECONCILE_INTERVAL_SECONDS",
    "TOPUP_RECONCILE_INTERVAL_SECONDS": "CREDIT_INVOICE_RECONCILE_INTERVAL_SECONDS",
    "BILLING_METERING_SWEEPS_ENABLED": "BILLING_METERING_MAINTENANCE_INTERVAL_SECONDS",
    "BILLING_METERING_SWEEP_INTERVAL_SECONDS": "BILLING_METERING_MAINTENANCE_INTERVAL_SECONDS",
    "BILLING_PERIOD_CLOSE_ENABLED": "BILLING_PERIOD_CLOSE_INTERVAL_SECONDS",
    "PERIOD_CLOSE_INTERVAL_SECONDS": "BILLING_PERIOD_CLOSE_INTERVAL_SECONDS",
}


@dataclass(frozen=True)
class RemovedEnv:
    """Everything the upgrade diagnostic knows about one removed variable.

    Attributes:
        name (str): The removed environment variable; its canonical scope is
            ``REMOVED_ENV_ALIASES[name]``.
        replacement (str | None): Printed as ``name -> replacement``; None prints
            ``name (remove assignment)``. When set, it equals the scope.
        migration (str | None): Replaces the derived per-name text after ``name``.
        guidance (str): Appended once after every per-name item. Records sharing
            one text share one paragraph; paragraphs follow registry order.
    """

    name: str
    replacement: str | None = None
    migration: str | None = None
    guidance: str = ""

    @property
    def item(self) -> str:
        """Return the per-name migration text."""
        if self.migration is not None:
            return f"{self.name} {self.migration}"
        return _item(self.name, self.replacement or "")


def _renamed(name: str, replacement: str, guidance: str = "") -> RemovedEnv:
    return RemovedEnv(name, replacement, guidance=guidance)


def _item(name: str, replacement: str) -> str:
    return f"{name} -> {replacement}" if replacement else f"{name} (remove assignment)"


_WORKERS_GUIDANCE = (
    "; preserve the actual worker count with explicit server --workers N; "
    "worker diagnostics now use that launcher argument directly. "
    "REFLEXIO_SERVER_WORKERS never selected execution capacity"
)
_OPTIMIZATION_HISTORY = "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY"
_OPTIMIZATION_GUIDANCE = (
    "; optimizer history now has one aggregate physical-row budget "
    "and reclaims complete eligible job graphs. Select its capacity "
    "explicitly; zero disables it. Independent old caps, including "
    "zero and blank assignments, are never silently converted"
)

# Registry order is the order of appended guidance; keep new guidance-bearing
# records where their paragraph belongs and plain renames after them.
REMOVED_ENV: tuple[RemovedEnv, ...] = (
    RemovedEnv(
        "RATELIMIT_STORAGE_OPTIONS",
        guidance="; nonempty environment strings never supplied usable storage options; remove the invalid assignment and preserve supported backend URI options",
    ),
    RemovedEnv(
        "RATELIMIT_HEADERS_ENABLED",
        guidance="; remove custom quota metadata settings; HTTP 429 now supplies standard Retry-After and successful responses omit quota metadata",
    ),
    RemovedEnv(
        "RATELIMIT_HEADER_LIMIT",
        guidance="; remove custom quota metadata settings; HTTP 429 now supplies standard Retry-After and successful responses omit quota metadata",
    ),
    RemovedEnv(
        "RATELIMIT_HEADER_REMAINING",
        guidance="; remove custom quota metadata settings; HTTP 429 now supplies standard Retry-After and successful responses omit quota metadata",
    ),
    RemovedEnv(
        "RATELIMIT_HEADER_RESET",
        guidance="; remove custom quota metadata settings; HTTP 429 now supplies standard Retry-After and successful responses omit quota metadata",
    ),
    RemovedEnv(
        "RATELIMIT_HEADER_RETRY_AFTER",
        guidance="; remove custom quota metadata settings; HTTP 429 now supplies standard Retry-After and successful responses omit quota metadata",
    ),
    RemovedEnv(
        "RATELIMIT_HEADER_RETRY_AFTER_VALUE",
        guidance="; remove custom quota metadata settings; HTTP 429 now supplies standard Retry-After and successful responses omit quota metadata",
    ),
    _renamed(
        "RATELIMIT_IN_MEMORY_FALLBACK_ENABLED",
        "RATELIMIT_IN_MEMORY_FALLBACK",
        "; unset disables fallback, inherit uses original rules, and a rate expression selects custom rules; do not copy the old boolean",
    ),
    RemovedEnv(
        "REFLEXIO_INSTANCE_ID",
        guidance=(
            "; WAL directories now own their persisted replica identity. "
            "Before removing the binding, point REFLEXIO_USAGE_WAL_PATH at the "
            "existing resolved WAL leaf; preserve its files and instance_id. "
            "Do not move or reinitialize pending usage"
        ),
    ),
    RemovedEnv(
        "REFLEXIO_AGENT_VERSION",
        guidance=(
            "; agent identity belongs to the caller: use explicit agent_version "
            "in SDK calls or --agent-version in the CLI, then remove the "
            "process assignment. Omitted publish identity keeps agent-v0; "
            "filter/regeneration commands require an explicit target. "
            "No stored request identity is rewritten"
        ),
    ),
    _renamed(
        "REFLEXIO_DEFAULT_ORG_ID",
        "config bind-local",
        "; stop the local backend. For a custom installation keep its saved config "
        "and database, remove the assignment, and run reflexio config bind-local "
        "with the previous organization as the argument before starting services. "
        "Use the same launch environment file to preserve its identity",
    ),
    _renamed(
        "REFLEXIO_EMBEDDING_REGENERATION_BATCH_SIZE",
        "regeneration request batch_size",
        "; preserve the intended size in the existing regeneration request "
        "batch_size field before removing this assignment. The request owns "
        "job paging; existing jobs retain their stored batch size",
    ),
    RemovedEnv(
        "REFLEXIO_MAX_INTERACTION_CONTENT_TOKENS",
        guidance=(
            "; complete interaction evidence now uses model-owned request capacity. "
            "Remove the old per-turn slicing assignment; preserve existing model, "
            "context declaration and output policy. No stored source is rewritten"
        ),
    ),
    _renamed(
        "REFLEXIO_SEARCH_RERANK_SERVICE_URL",
        "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL",
        "; remove the duplicate assignment and keep the query URL. "
        "Use the search workload for reserved reranking "
        "or the query workload for query embeddings alone",
    ),
    RemovedEnv(
        "REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS",
        guidance=(
            "; LLM attempts now use one accepted-work deadline, including startup. "
            "Remove the grace assignment and configure the intended request timeout directly; "
            "worker retirement does not extend result acceptance"
        ),
    ),
    RemovedEnv(
        "REFLEXIO_LEARNING_ALARM_SUPPRESS_SECONDS",
        guidance=(
            "; discovery notifications now follow failure episodes and current health. "
            "Remove the suppression assignment; do not copy its duration. "
            "The queue-age alert threshold remains independent"
        ),
    ),
    _renamed(
        "POOLED_CREDITS_ENABLED",
        "BILLING_ENFORCEMENT_MODE",
        "; remove the obsolete pooling assignment; do not copy its boolean value. "
        "Usage credits follow the plan and customer cohort. Preserve the intended "
        "BILLING_ENFORCEMENT_MODE (observe or active); no ledger is rewritten",
    ),
    _renamed(
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_OFFLINE",
        "; preserve the effective HF_HUB_OFFLINE policy before removing "
        "the alias. A nonempty canonical value already took precedence, "
        "including 0; do not overwrite it with an ignored alias. "
        "Resolve image defaults and secret bindings explicitly",
    ),
    _renamed(
        "REFLEXIO_FERNET_KEYS",
        "FERNET_KEYS",
        "; preserve all existing configuration and connector decryption "
        "keys in the canonical ring, choosing its first encryption key "
        "explicitly. Do not overwrite an existing FERNET_KEYS ring "
        "with only the connector key. Existing ciphertext is not rewritten",
    ),
    RemovedEnv(
        "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOK_SOURCE_USER_PLAYBOOKS",
        # Scoped under the agent-playbook cap, but nothing replaces it.
        guidance=(
            "; source provenance now follows its parent playbooks. Remove "
            "the source-link cap assignment; do not copy its value. Existing "
            "agent/user playbook limits keep their original units and values"
        ),
    ),
    _renamed("WEB_CONCURRENCY", "server --workers N", _WORKERS_GUIDANCE),
    _renamed("REFLEXIO_SERVER_WORKERS", "server --workers N", _WORKERS_GUIDANCE),
    _renamed(
        "REFLEXIO_LLM_FAIL_CLOSED_PROVIDERS",
        "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY",
        "; capacities now apply to every resolved provider; preserve "
        "per-provider sizes with REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY_OVERRIDES. "
        "Dispatch without a permit is no longer supported",
    ),
    _renamed(
        "REFLEXIO_SEARCH_DEADLINE_ENABLED",
        "REFLEXIO_SEARCH_TIMEOUT_SECONDS",
        "; preserve the former true work budget with 5 seconds or select "
        "0 to inherit the original HTTP backstop (former false). "
        "Accepted-response writes use the remaining HTTP budget",
    ),
    _renamed(
        "REFLEXIO_SEARCH_CONCURRENCY_TIMEOUT_SECONDS",
        "REFLEXIO_SEARCH_TIMEOUT_SECONDS",
        "; search admission and execution now share one ingress budget. "
        "Remove the queue-timeout assignment and choose the total "
        "REFLEXIO_SEARCH_TIMEOUT_SECONDS budget (0 inherits HTTP). "
        "Independent early queue rejection is no longer supported; "
        "do not automatically copy its value",
    ),
    _renamed(
        "REFLEXIO_POSTGRES_STATEMENT_TIMEOUT_MS",
        "REFLEXIO_DB_LEASE_TIMEOUT_SECONDS",
        "; ordinary storage SQL now consumes the remaining checked-out "
        "connection lease, including transaction cleanup. Remove the "
        "statement assignment and deliberately choose the whole-borrow "
        "REFLEXIO_DB_LEASE_TIMEOUT_SECONDS budget. Units and semantics "
        "differ; do not automatically copy the old value. Independent deployment-wide "
        "per-statement policy is no longer supported",
    ),
    _renamed(
        "REFLEXIO_SEARCH_RECENCY_ENABLED",
        "REFLEXIO_SEARCH_RECENCY_POOL_SIZE",
        "; a zero recency pool disables scored recency retrieval and "
        "ranking; preserve positive pool sizes to enable it, with "
        "20 as the unchanged default",
    ),
    *(
        _renamed(
            f"REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_{kind}",
            _OPTIMIZATION_HISTORY,
            _OPTIMIZATION_GUIDANCE,
        )
        for kind in ("JOBS", "CANDIDATES", "EVALUATIONS", "EVENTS")
    ),
    _renamed(
        "REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS",
        "REFLEXIO_EMBED_BATCH_SIZE",
        "; remove the queue-cap assignment, do not copy its value. "
        "REFLEXIO_EMBED_BATCH_SIZE now owns native encoder and queue-turn "
        "capacity together; preserve its existing value",
    ),
    _renamed(
        "PUBLIC_SCHEMA_AUDIT_ENABLED",
        "PUBLIC_SCHEMA_AUDIT_INTERVAL_SECONDS",
        "; zero audit cadence disables periodic schema audits; preserve "
        "a positive cadence to enable them, with 86400 as the unchanged "
        "default. Values are seconds, not booleans",
    ),
    _renamed(
        "CLAUDE_SMART_USE_LOCAL_EMBEDDING",
        "REFLEXIO_EMBEDDING_PROVIDER",
        "; preserve the selected embedding model, explicit service endpoint "
        "and effective service timeout. The old false value was not an "
        "embeddings-off switch; use canonical off only when deliberately "
        "disabling embeddings. See the embedding upgrade guide",
    ),
    RemovedEnv(
        "REFLEXIO_LLM_SEED",
        migration=(
            "(remove assignment): sampling belongs to the "
            "request. Pass seed explicitly when needed and temperature=0 "
            "to retain the former temperature override. Model restrictions "
            "still apply; existing evaluation records are preserved"
        ),
    ),
    RemovedEnv("REFLEXIO_ROW_LIMIT_OFFLINE_TUNER_REWARD_LABEL"),
    _renamed("REFLEXIO_API_QUERY_EMBEDDING_ENABLED", "REFLEXIO_INFERENCE_WORKLOAD"),
    _renamed("NOMIC_EMBED_DEVICE", "REFLEXIO_EMBEDDING_DEVICE"),
    _renamed("DEBUG_LOG_TO_CONSOLE", "REFLEXIO_LOG_PROFILE"),
    _renamed("DATA_SUPABASE_DB_URL", "DATA_DB_URL"),
    _renamed("POSTGRES_DB_URL", "DATA_DB_URL"),
    _renamed("INTERACTION_CLEANUP_THRESHOLD", "REFLEXIO_ROW_LIMIT_INTERACTIONS"),
    _renamed("REFLEXIO_EMBED_TORCH_THREADS", "REFLEXIO_NATIVE_COMPUTE_THREADS"),
    _renamed(
        "REFLEXIO_MISSING_VECTOR_BACKFILL_ENABLED",
        "REFLEXIO_MISSING_VECTOR_BACKFILL_CAP",
    ),
    _renamed("REFLEXIO_PUBLISH_TIMING_ENABLED", "REFLEXIO_PUBLISH_TIMING_MIN_MS"),
    _renamed("REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS", "REFLEXIO_PUBLISH_TIMING_MIN_MS"),
    _renamed("CONTROL_PLANE_INGEST_URL", "CONTROL_PLANE_URL"),
    _renamed("TOPUP_RECONCILE_ENABLED", "CREDIT_INVOICE_RECONCILE_INTERVAL_SECONDS"),
    _renamed(
        "TOPUP_RECONCILE_INTERVAL_SECONDS",
        "CREDIT_INVOICE_RECONCILE_INTERVAL_SECONDS",
    ),
    _renamed(
        "BILLING_METERING_SWEEPS_ENABLED",
        "BILLING_METERING_MAINTENANCE_INTERVAL_SECONDS",
    ),
    _renamed(
        "BILLING_METERING_SWEEP_INTERVAL_SECONDS",
        "BILLING_METERING_MAINTENANCE_INTERVAL_SECONDS",
    ),
    _renamed(
        "REFLEXIO_SEARCH_JUDGE",
        "REFLEXIO_SEARCH_JUDGE_API_KEY",
        "; when explicitly opting into the existing Jev route, bind the approved "
        "credential instead of copying the old mode value. Leave the dedicated "
        "binding unset for native search",
    ),
    _renamed("BILLING_PERIOD_CLOSE_ENABLED", "BILLING_PERIOD_CLOSE_INTERVAL_SECONDS"),
    _renamed("PERIOD_CLOSE_INTERVAL_SECONDS", "BILLING_PERIOD_CLOSE_INTERVAL_SECONDS"),
)

_BY_NAME = {record.name: record for record in REMOVED_ENV}


def reject_removed_env(
    env: Mapping[str, object],
    aliases: Mapping[str, str] = REMOVED_ENV_ALIASES,
    *,
    canonical: str | None = None,
) -> None:
    """Reject obsolete assignments, including blanks, without logging secrets.

    Checking before dotenv loading prevents an obsolete process-level database
    target from being silently replaced by a file's canonical target. Scoped
    checks protect direct policy readers as well as ordinary startup.

    Registered names take their text from ``REMOVED_ENV``; caller-supplied
    names absent from it are described by their ``aliases`` value alone.
    """
    obsolete = sorted(
        name
        for name in aliases
        if (canonical is None or aliases[name] == canonical) and name in env
    )
    if not obsolete:
        return
    migration = "; ".join(
        _BY_NAME[name].item if name in _BY_NAME else _item(name, aliases[name])
        for name in obsolete
    )
    guidance: list[str] = []
    for record in REMOVED_ENV:
        if record.name in obsolete and record.guidance not in guidance:
            guidance.append(record.guidance)
    migration += "".join(guidance)
    raise ValueError(
        f"Removed environment names: {migration}. Follow the name-specific "
        "guidance above and remove the old assignments; "
        "see the configuration upgrade guide. Values are never copied."
    )
