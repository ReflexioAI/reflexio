"""Shared distribution upgrade diagnostics; removed names never resolve values."""

from collections.abc import Mapping

REMOVED_ENV_ALIASES = {
    "DEBUG_LOG_TO_CONSOLE": "REFLEXIO_LOG_PROFILE",
    "DATA_SUPABASE_DB_URL": "DATA_DB_URL",
    "POSTGRES_DB_URL": "DATA_DB_URL",
    "WEB_CONCURRENCY": "REFLEXIO_SERVER_WORKERS",
    "INTERACTION_CLEANUP_THRESHOLD": "REFLEXIO_ROW_LIMIT_INTERACTIONS",
    "REFLEXIO_EMBED_TORCH_THREADS": "REFLEXIO_INFERENCE_TORCH_THREADS",
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
        migration = "; ".join(f"{name} -> {aliases[name]}" for name in obsolete)
        if "WEB_CONCURRENCY" in obsolete:
            migration += (
                "; preserve the actual worker count with explicit server --workers N; "
                "REFLEXIO_SERVER_WORKERS is diagnostic metadata only"
            )
        raise ValueError(
            f"Removed environment names: {migration}. Migrate the effective "
            "policy to the canonical setting and remove the old assignments; "
            "see the configuration upgrade guide. Values are never copied."
        )
