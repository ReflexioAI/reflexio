"""Append-only JSONL sink for rows removed by row-count retention."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_RETENTION_ARCHIVE_MAX_BYTES = 10 * 1024**3


class RetentionArchiveFullError(OSError):
    """Raised when another archive line would exceed the configured ceiling."""


def retention_archive_enabled() -> bool:
    """Return whether archive-before-delete retention is enabled."""
    return os.environ.get("REFLEXIO_RETENTION_ARCHIVE", "").lower() in {
        "1",
        "true",
    }


def resolve_archive_directory(database_path: str) -> Path:
    """Resolve the archive directory from the override or SQLite DB path.

    Args:
        database_path: Path to the active SQLite database.

    Returns:
        Directory that owns the per-table JSONL archive files.
    """
    override = os.environ.get("REFLEXIO_RETENTION_ARCHIVE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(database_path).expanduser().parent / "archive"


def retention_archive_max_bytes() -> int:
    """Return the total archive-directory ceiling in bytes.

    Returns:
        Positive byte ceiling. Defaults to 10 GiB.

    Raises:
        ValueError: If ``REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES`` is not a
            positive integer. Retention then fails closed instead of silently
            running without a usable ceiling.
    """
    raw = os.environ.get("REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES")
    if raw is None or not raw.strip():
        return DEFAULT_RETENTION_ARCHIVE_MAX_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES must be a positive integer"
        ) from exc
    if value <= 0:
        raise ValueError(
            "REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES must be a positive integer"
        )
    return value


def archive_size_bytes(archive_dir: Path) -> int:
    """Return total bytes currently used by JSONL archive files.

    Args:
        archive_dir: Directory containing per-table JSONL files.

    Returns:
        Sum of all ``*.jsonl`` file sizes, or zero before first write.
    """
    if not archive_dir.is_dir():
        return 0
    return sum(path.stat().st_size for path in archive_dir.glob("*.jsonl"))


def append_archive_rows(
    archive_dir: Path, table_name: str, rows: list[dict[str, Any]]
) -> None:
    """Append sanitized rows to ``table_name``'s archive file.

    Each JSON line is emitted with one ``write`` call. Values unsupported by
    the JSON encoder fall back to their ``repr`` while embeddings are omitted.

    Args:
        archive_dir: Directory containing per-table JSONL files.
        table_name: Physical table being archived.
        rows: Complete database rows selected for retention removal.

    Raises:
        RetentionArchiveFullError: If another line would exceed the total
            archive-directory ceiling. The caller must then keep live rows.
    """
    if not rows:
        return
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_at = int(time.time())
    max_bytes = retention_archive_max_bytes()
    projected_bytes = archive_size_bytes(archive_dir)
    path = archive_dir / f"{table_name}.jsonl"
    with path.open("ab") as archive_file:
        for row in rows:
            sanitized = {key: value for key, value in row.items() if key != "embedding"}
            record = {
                "table": table_name,
                "archived_at": archived_at,
                "row": sanitized,
            }
            encoded_line = (
                json.dumps(record, default=repr, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            if projected_bytes + len(encoded_line) > max_bytes:
                raise RetentionArchiveFullError(
                    "Retention archive reached its configured ceiling "
                    f"({max_bytes} bytes); live-row trimming was stopped"
                )
            archive_file.write(encoded_line)
            projected_bytes += len(encoded_line)
