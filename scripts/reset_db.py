#!/usr/bin/env python3
"""Reset the local SQLite database to a clean state.

Deletes the existing database file (and WAL/SHM sidecars) then
re-creates all tables, indexes, and FTS virtual tables from scratch.

Usage:
    uv run python scripts/reset_db.py
    uv run python scripts/reset_db.py --org acme
    uv run python scripts/reset_db.py --db-path /custom/path/reflexio.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # scripts/
_PROJECT_ROOT = _THIS_DIR.parent  # repo root

sys.path.insert(0, str(_PROJECT_ROOT))

from reflexio.cli.bootstrap_config import default_org_id
from reflexio.server import LOCAL_STORAGE_PATH
from reflexio.server.services.storage.sqlite_storage._dataset_path import (
    resolve_sqlite_db_path,
)


def _default_db_path(org_id: str) -> Path:
    """Return the database *org_id* owns, adopting a legacy file if it holds one."""
    return Path(resolve_sqlite_db_path(LOCAL_STORAGE_PATH, org_id))


def reset_db(db_path: Path, org_id: str) -> None:
    """Delete the SQLite database and its WAL/SHM sidecars, then recreate empty tables."""
    # Remove existing files
    removed: list[str] = []
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = db_path.parent / (db_path.name + suffix)
        if p.exists():
            p.unlink()
            removed.append(p.name)

    if removed:
        print(f"Removed: {', '.join(removed)}")
    else:
        print("No existing database found — creating fresh.")

    # Re-create by importing and instantiating storage (runs DDL automatically)
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    # Recreated under the same identity the path was resolved for -- a hardcoded
    # org would rebuild a database nobody reads.
    storage = SQLiteStorage(org_id=org_id, db_path=str(db_path))
    storage.conn.close()
    print(f"Clean database created at: {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset local SQLite database to a clean state."
    )
    parser.add_argument(
        "--org",
        default=None,
        help="Dataset identity to reset (default: the configured default org).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to the database file (default: derived from --org).",
    )
    args = parser.parse_args()

    org_id: str = args.org or default_org_id()
    db_path: Path = args.db_path or _default_db_path(org_id)

    print(f"This will DELETE all data in: {db_path}")
    confirm = input("Continue? [y/N] ")
    if confirm.lower() != "y":
        print("Aborted.")
        sys.exit(1)

    reset_db(db_path, org_id)


if __name__ == "__main__":
    main()
