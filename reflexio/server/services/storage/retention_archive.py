"""Append-only JSONL sink for rows removed by row-count retention."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_RETENTION_ARCHIVE_MAX_BYTES = 10 * 1024**3
RETENTION_ARCHIVE_DELETE_BATCH = 1_000
logger = logging.getLogger(__name__)
_warned_archive_dirs: set[Path] = set()


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
    """Return the total archive-directory hard ceiling in bytes.

    Returns:
        Positive archive ceiling. Defaults to 10 GiB.

    Invalid values log an error and fall back to 10 GiB so observability
    configuration can never stop live-row retention.
    """
    raw = os.environ.get("REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES")
    if raw is None or not raw.strip():
        return DEFAULT_RETENTION_ARCHIVE_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.error(
            "Invalid REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES=%r; using %d",
            raw,
            DEFAULT_RETENTION_ARCHIVE_MAX_BYTES,
        )
        return DEFAULT_RETENTION_ARCHIVE_MAX_BYTES
    if value <= 0:
        logger.error(
            "Invalid REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES=%r; using %d",
            raw,
            DEFAULT_RETENTION_ARCHIVE_MAX_BYTES,
        )
        return DEFAULT_RETENTION_ARCHIVE_MAX_BYTES
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
) -> bool:
    """Append sanitized rows to ``table_name``'s archive file.

    Each JSON line is emitted with one ``write`` call. Values unsupported by
    the JSON encoder fall back to their ``repr`` while embeddings are omitted.

    Args:
        archive_dir: Directory containing per-table JSONL files.
        table_name: Physical table being archived.
        rows: Complete database rows selected for retention removal.

    The configured threshold is a hard archive ceiling. A batch that would
    cross it is skipped whole so disk usage stays bounded; callers must still
    continue live-row retention.

    Returns:
        True when all rows were appended, or False when the ceiling skipped
        the batch.
    """
    if not rows:
        return True
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_at = int(time.time())
    max_bytes = retention_archive_max_bytes()
    projected_bytes = archive_size_bytes(archive_dir)
    resolved_archive_dir = archive_dir.resolve()
    if projected_bytes <= max_bytes:
        _warned_archive_dirs.discard(resolved_archive_dir)
    path = archive_dir / f"{table_name}.jsonl"
    encoded_lines: list[bytes] = []
    for row in rows:
        sanitized = {key: value for key, value in row.items() if key != "embedding"}
        record = {
            "table": table_name,
            "archived_at": archived_at,
            "row": sanitized,
        }
        encoded_lines.append(
            (json.dumps(record, default=repr, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    batch_bytes = sum(len(line) for line in encoded_lines)
    if projected_bytes + batch_bytes > max_bytes:
        logger.error(
            "Retention archive ceiling reached; skipping evidence archive while "
            "live-row retention continues: directory=%s table=%s rows_skipped=%d "
            "size_bytes=%d batch_bytes=%d ceiling_bytes=%d",
            archive_dir,
            table_name,
            len(rows),
            projected_bytes,
            batch_bytes,
            max_bytes,
        )
        _warned_archive_dirs.add(resolved_archive_dir)
        return False
    with path.open("ab") as archive_file:
        for encoded_line in encoded_lines:
            archive_file.write(encoded_line)
            projected_bytes += len(encoded_line)
    return True
