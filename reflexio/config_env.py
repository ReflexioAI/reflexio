"""Lightweight canonical environment resolution shared by CLI and server.

No dotenv loading, server initialization, or cached process state: callers pass
an already selected environment mapping, and canonical nonblank values win.
"""

from collections.abc import Mapping


def resolve_data_db_url(env: Mapping[str, str]) -> str:
    """Resolve one data database target, accepting backend-specific old names.

    Blank values are unset. With no selected backend, retain the historical
    Supabase-then-Postgres fallback used by import-time compatibility constants.
    A selected backend never falls through to the other backend's old name.
    """
    canonical = env.get("DATA_DB_URL", "").strip()
    if canonical:
        return canonical
    backend = env.get("REFLEXIO_STORAGE", "").strip().lower()
    if backend == "postgres":
        return env.get("POSTGRES_DB_URL", "").strip()
    if backend == "supabase":
        return env.get("DATA_SUPABASE_DB_URL", "").strip()
    return (
        env.get("DATA_SUPABASE_DB_URL", "").strip()
        or env.get("POSTGRES_DB_URL", "").strip()
    )
