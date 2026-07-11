"""Append-only JSONL sink for rows removed by row-count retention."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def retention_archive_enabled() -> bool:
    """Return whether archive-before-delete retention is enabled."""
    return os.environ.get("REFLEXIO_RETENTION_ARCHIVE", "").lower() in {
        "1",
        "true",
    }


def resolve_archive_directory(database_path: str) -> Path:
    """Resolve the archive directory from the override or SQLite DB path."""
    override = os.environ.get("REFLEXIO_RETENTION_ARCHIVE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(database_path).expanduser().parent / "archive"


def append_archive_rows(
    archive_dir: Path, table_name: str, rows: list[dict[str, Any]]
) -> None:
    """Append sanitized rows to ``table_name``'s archive file.

    Each JSON line is emitted with one ``write`` call. Values unsupported by
    the JSON encoder fall back to their ``repr`` while embeddings are omitted.
    """
    if not rows:
        return
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_at = int(time.time())
    path = archive_dir / f"{table_name}.jsonl"
    with path.open("a", encoding="utf-8") as archive_file:
        for row in rows:
            sanitized = {key: value for key, value in row.items() if key != "embedding"}
            record = {
                "table": table_name,
                "archived_at": archived_at,
                "row": sanitized,
            }
            archive_file.write(
                json.dumps(record, default=repr, separators=(",", ":")) + "\n"
            )
