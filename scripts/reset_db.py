#!/usr/bin/env python3
"""Reset the local SQLite database to a clean state.

Deletes the existing database file (and WAL/SHM sidecars) then
re-creates all tables, indexes, and FTS virtual tables from scratch.

Usage:
    uv run python scripts/reset_db.py
    uv run python scripts/reset_db.py --db-path /custom/path/reflexio.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # scripts/
_PROJECT_ROOT = _THIS_DIR.parent  # repo root

sys.path.insert(0, str(_PROJECT_ROOT))

from reflexio.server import SQLITE_FILE_DIRECTORY


def _default_db_path() -> Path:
    return Path(SQLITE_FILE_DIRECTORY) / "reflexio.db"


def reset_db(db_path: Path) -> None:
    """Delete the SQLite database and its WAL/SHM sidecars, then recreate empty tables."""
    # Remove existing files
    removed: list[str] = []
    for suffix in ("", "-wal", "-shm"):
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

    storage = SQLiteStorage(org_id="default", db_path=str(db_path))
    storage.conn.close()
    print(f"Clean database created at: {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset local SQLite database to a clean state."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help=f"Path to the database file (default: {_default_db_path()})",
    )
    args = parser.parse_args()

    db_path: Path = args.db_path or _default_db_path()

    print(f"This will DELETE all data in: {db_path}")
    confirm = input("Continue? [y/N] ")
    if confirm.lower() != "y":
        print("Aborted.")
        sys.exit(1)

    reset_db(db_path)


if __name__ == "__main__":
    main()
