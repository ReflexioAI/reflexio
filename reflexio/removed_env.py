"""Shared distribution upgrade diagnostics; removed names never resolve values."""

from collections.abc import Mapping

REMOVED_ENV_ALIASES = {
    "REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS": "REFLEXIO_EMBED_BATCH_SIZE",
    "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOK_SOURCE_USER_PLAYBOOKS": "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOKS",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_JOBS": "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_CANDIDATES": "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_EVALUATIONS": "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY",
    "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_EVENTS": "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_HISTORY",
    "REFLEXIO_API_QUERY_EMBEDDING_ENABLED": "REFLEXIO_INFERENCE_WORKLOAD",
    "NOMIC_EMBED_DEVICE": "REFLEXIO_EMBEDDING_DEVICE",
    "REFLEXIO_SEARCH_DEADLINE_ENABLED": "REFLEXIO_SEARCH_TIMEOUT_SECONDS",
    "REFLEXIO_SEARCH_CONCURRENCY_TIMEOUT_SECONDS": "REFLEXIO_SEARCH_TIMEOUT_SECONDS",
    "REFLEXIO_POSTGRES_STATEMENT_TIMEOUT_MS": "REFLEXIO_DB_LEASE_TIMEOUT_SECONDS",
    "CLAUDE_SMART_USE_LOCAL_EMBEDDING": "REFLEXIO_EMBEDDING_PROVIDER",
    "PUBLIC_SCHEMA_AUDIT_ENABLED": "PUBLIC_SCHEMA_AUDIT_INTERVAL_SECONDS",
    "REFLEXIO_SEARCH_RECENCY_ENABLED": "REFLEXIO_SEARCH_RECENCY_POOL_SIZE",
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
    """
    obsolete = sorted(
        name
        for name in aliases
        if name in env and (canonical is None or aliases[name] == canonical)
    )
    if obsolete:
        source_cap = "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOK_SOURCE_USER_PLAYBOOKS"
        migration = "; ".join(
            f"{name} (remove assignment)"
            if name == source_cap
            else f"{name} -> {aliases[name]}"
            for name in obsolete
        )
        if "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOK_SOURCE_USER_PLAYBOOKS" in obsolete:
            migration += (
                "; source provenance now follows its parent playbooks. Remove "
                "the source-link cap assignment; do not copy its value. Existing "
                "agent/user playbook limits keep their original units and values"
            )
        if {"WEB_CONCURRENCY", "REFLEXIO_SERVER_WORKERS"}.intersection(obsolete):
            migration += (
                "; preserve the actual worker count with explicit server --workers N; "
                "worker diagnostics now use that launcher argument directly. "
                "REFLEXIO_SERVER_WORKERS never selected execution capacity"
            )
        if "REFLEXIO_LLM_FAIL_CLOSED_PROVIDERS" in obsolete:
            migration += (
                "; capacities now apply to every resolved provider; preserve "
                "per-provider sizes with REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY_OVERRIDES. "
                "Dispatch without a permit is no longer supported"
            )
        if "REFLEXIO_SEARCH_DEADLINE_ENABLED" in obsolete:
            migration += (
                "; preserve the former true work budget with 5 seconds or select "
                "0 to inherit the original HTTP backstop (former false). "
                "Accepted-response writes use the remaining HTTP budget"
            )
        if "REFLEXIO_SEARCH_CONCURRENCY_TIMEOUT_SECONDS" in obsolete:
            migration += (
                "; search admission and execution now share one ingress budget. "
                "Remove the queue-timeout assignment and choose the total "
                "REFLEXIO_SEARCH_TIMEOUT_SECONDS budget (0 inherits HTTP). "
                "Independent early queue rejection is no longer supported; "
                "do not automatically copy its value"
            )
        if "REFLEXIO_POSTGRES_STATEMENT_TIMEOUT_MS" in obsolete:
            migration += (
                "; ordinary storage SQL now consumes the remaining checked-out "
                "connection lease, including transaction cleanup. Remove the "
                "statement assignment and deliberately choose the whole-borrow "
                "REFLEXIO_DB_LEASE_TIMEOUT_SECONDS budget. Units and semantics "
                "differ; do not automatically copy the old value. Independent deployment-wide "
                "per-statement policy is no longer supported"
            )
        if "REFLEXIO_SEARCH_RECENCY_ENABLED" in obsolete:
            migration += (
                "; a zero recency pool disables scored recency retrieval and "
                "ranking; preserve positive pool sizes to enable it, with "
                "20 as the unchanged default"
            )
        if any(
            name.startswith("REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_")
            for name in obsolete
        ):
            migration += (
                "; optimizer history now has one aggregate physical-row budget "
                "and reclaims complete eligible job graphs. Select its capacity "
                "explicitly; zero disables it. Independent old caps, including "
                "zero and blank assignments, are never silently converted"
            )
        if "REFLEXIO_EMBED_MICRO_BATCH_MAX_TEXTS" in obsolete:
            migration += (
                "; remove the queue-cap assignment, do not copy its value. "
                "REFLEXIO_EMBED_BATCH_SIZE now owns native encoder and queue-turn "
                "capacity together; preserve its existing value"
            )
        if "PUBLIC_SCHEMA_AUDIT_ENABLED" in obsolete:
            migration += (
                "; zero audit cadence disables periodic schema audits; preserve "
                "a positive cadence to enable them, with 86400 as the unchanged "
                "default. Values are seconds, not booleans"
            )
        if "CLAUDE_SMART_USE_LOCAL_EMBEDDING" in obsolete:
            migration += (
                "; preserve the selected embedding model, explicit service endpoint "
                "and effective service timeout. The old false value was not an "
                "embeddings-off switch; use canonical off only when deliberately "
                "disabling embeddings. See the embedding upgrade guide"
            )
        raise ValueError(
            f"Removed environment names: {migration}. Follow the name-specific "
            "guidance above and remove the old assignments; "
            "see the configuration upgrade guide. Values are never copied."
        )
