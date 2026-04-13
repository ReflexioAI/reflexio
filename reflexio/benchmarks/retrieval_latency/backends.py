"""
Backend setup helpers for the retrieval latency benchmark.

Each supported backend is exposed as a context manager that:

1. Builds a ``Config`` with the appropriate ``StorageConfig``.
2. Instantiates a ``Reflexio`` facade and obtains the underlying storage.
3. Yields a :class:`BackendHandle` (reflexio instance + storage + org_id).
4. Tears down temporary resources on exit.

SQLite is always available. Supabase is only available when the local
stack is running (``supabase start``) and the expected env vars are set; in
all other cases the helper yields ``None`` so the benchmark loop can skip
cleanly with a clear warning.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.config_schema import (
    Config,
    StorageConfigSQLite,
    StorageConfigSupabase,
)
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.storage.storage_base import BaseStorage

logger = logging.getLogger(__name__)

BENCH_ORG_ID = "bench_org"


@dataclass
class BackendHandle:
    """
    Handle to a live Reflexio facade wired to a specific storage backend.

    Attributes:
        name (str): Short backend identifier, e.g. ``"sqlite"``.
        reflexio (Reflexio): Service-layer facade — call ``search_profiles``
            etc. directly on this for the service layer benchmark.
        storage (BaseStorage): Underlying storage instance, needed for
            swapping ``_get_embedding`` during seeding and the timed loop.
        org_id (str): Organization ID used for this benchmark run.
    """

    name: str
    reflexio: Reflexio
    storage: BaseStorage
    org_id: str


def _make_reflexio(config: Config, org_id: str) -> Reflexio:
    """
    Build a Reflexio instance from an in-memory ``Config``.

    Args:
        config (Config): Fully populated benchmark config.
        org_id (str): Organization ID for the Reflexio instance.

    Returns:
        Reflexio: A facade with storage wired up through the configurator.
    """
    configurator = DefaultConfigurator(org_id=org_id, config=config)
    return Reflexio(org_id=org_id, configurator=configurator)


@contextmanager
def sqlite_backend() -> Iterator[BackendHandle]:
    """
    Construct an isolated SQLite-backed Reflexio instance in a tmpdir.

    Mirrors the contract-test storage fixture pattern: fresh directory,
    fresh database, no shared state with other runs.

    Yields:
        BackendHandle: A live handle with name ``"sqlite"``.
    """
    with tempfile.TemporaryDirectory(prefix="reflexio-bench-sqlite-") as tmp:
        db_path = str(Path(tmp) / "reflexio.db")
        config = Config(storage_config=StorageConfigSQLite(db_path=db_path))
        reflexio = _make_reflexio(config, BENCH_ORG_ID)
        storage = reflexio._get_storage()
        yield BackendHandle(
            name="sqlite",
            reflexio=reflexio,
            storage=storage,
            org_id=BENCH_ORG_ID,
        )


def _supabase_env_ready() -> StorageConfigSupabase | None:
    """
    Read Supabase env vars and return a populated config, or ``None``.

    Falls back to the default local ``supabase start`` URL when only the
    keys are provided.

    Returns:
        StorageConfigSupabase | None: A config if all three fields can be
        resolved, otherwise ``None``.
    """
    url = os.environ.get("SUPABASE_URL", "http://127.0.0.1:54321")
    key = os.environ.get("SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_KEY")
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not (key and db_url):
        return None
    return StorageConfigSupabase(url=url, key=key, db_url=db_url)


@contextmanager
def supabase_backend() -> Iterator[BackendHandle | None]:
    """
    Construct a Supabase-backed Reflexio instance if the local stack is up.

    Yields ``None`` (and logs a warning) when the Supabase env vars are
    missing or the connection fails, so the benchmark loop can skip this
    backend gracefully without aborting the whole run.

    Yields:
        BackendHandle | None: Live handle, or ``None`` if unavailable.
    """
    storage_config = _supabase_env_ready()
    if storage_config is None:
        logger.warning(
            "Supabase backend skipped: set SUPABASE_ANON_KEY and SUPABASE_DB_URL "
            "(and optionally SUPABASE_URL) to enable it."
        )
        yield None
        return

    config = Config(storage_config=storage_config)
    try:
        reflexio = _make_reflexio(config, BENCH_ORG_ID)
        storage = reflexio._get_storage()
    except Exception as err:  # noqa: BLE001
        logger.warning(
            "Supabase backend skipped: failed to connect — %s. "
            "Is `supabase start` running?",
            err,
        )
        yield None
        return

    try:
        yield BackendHandle(
            name="supabase",
            reflexio=reflexio,
            storage=storage,
            org_id=BENCH_ORG_ID,
        )
    finally:
        # Best-effort cleanup: wipe the benchmark org's rows so repeated
        # runs don't accumulate forever. Swallow errors — a failing cleanup
        # must not mask a real measurement failure.
        try:
            reflexio.delete_all_profiles_bulk()
            reflexio.delete_all_user_playbooks_bulk()
            reflexio.delete_all_agent_playbooks_bulk()
        except Exception as err:  # noqa: BLE001
            logger.warning("Supabase cleanup failed: %s", err)


BACKENDS = {
    "sqlite": sqlite_backend,
    "supabase": supabase_backend,
}
