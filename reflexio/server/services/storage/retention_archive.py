"""Bounded FIFO JSONL archive for rows removed by row-count retention."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_RETENTION_ARCHIVE_MAX_BYTES = 10 * 1024**3
RETENTION_ARCHIVE_DELETE_BATCH = 1_000
STALE_TEMPORARY_FILE_SECONDS = 60 * 60
logger = logging.getLogger(__name__)


def retention_archive_enabled() -> bool:
    """Return whether archive-before-delete retention is enabled.

    Returns:
        True when the archive environment flag is enabled.
    """
    return os.environ.get("REFLEXIO_RETENTION_ARCHIVE", "").lower() in {
        "1",
        "true",
    }


def resolve_archive_directory(database_path: str) -> Path:
    """Return the archive directory beside the SQLite database.

    Args:
        database_path: Path to the SQLite database.

    Returns:
        A sibling ``archive`` directory.
    """
    return Path(database_path).expanduser().parent / "archive"


def retention_archive_max_bytes() -> int:
    """Return the positive archive ceiling, defaulting to 10 GiB.

    Returns:
        Configured positive ceiling in bytes, or the default for invalid input.
    """
    raw = os.environ.get("REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES")
    if raw is None or not raw.strip():
        return DEFAULT_RETENTION_ARCHIVE_MAX_BYTES
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        logger.error(
            "Invalid REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES=%r; using %d",
            raw,
            DEFAULT_RETENTION_ARCHIVE_MAX_BYTES,
        )
        return DEFAULT_RETENTION_ARCHIVE_MAX_BYTES
    return value


def _encode_records(rows_by_table: Mapping[str, list[dict[str, Any]]]) -> bytes:
    archived_at = int(time.time())
    lines = []
    for table_name, rows in rows_by_table.items():
        for row in rows:
            record = {
                "table": table_name,
                "archived_at": archived_at,
                "row": {key: value for key, value in row.items() if key != "embedding"},
            }
            lines.append(json.dumps(record, default=repr, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


def _segment_sizes(archive_dir: Path) -> list[tuple[Path, int]]:
    """Return existing archive segments and sizes in FIFO order."""
    segments = []
    for path in sorted(archive_dir.glob("*.jsonl")):
        try:
            segments.append((path, path.stat().st_size))
        except FileNotFoundError:
            continue
    return segments


def _remove_stale_temporary_files(archive_dir: Path) -> None:
    """Remove abandoned writes without disturbing another active process."""
    stale_before = time.time() - STALE_TEMPORARY_FILE_SECONDS
    for path in archive_dir.glob("*.tmp"):
        try:
            if path.stat().st_mtime < stale_before:
                path.unlink(missing_ok=True)
        except FileNotFoundError:
            continue


def append_archive_batch(
    archive_dir: Path, rows_by_table: Mapping[str, list[dict[str, Any]]]
) -> bool:
    """Append one retention batch and evict oldest segments to stay bounded.

    One segment contains the target rows and any cascade-deleted rows, so FIFO
    eviction never splits a retention batch. The completed newest segment is
    installed before old segments are removed; a crash can temporarily exceed
    the ceiling but cannot replace newest evidence with older evidence.

    Args:
        archive_dir: Directory that owns the FIFO archive segments.
        rows_by_table: Deleted database rows grouped by source table.

    Returns:
        True when the batch was archived. False only when one batch is itself
        larger than the configured ceiling.
    """
    encoded = _encode_records(rows_by_table)
    if not encoded:
        return True
    max_bytes = retention_archive_max_bytes()
    if len(encoded) > max_bytes:
        return False

    archive_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_temporary_files(archive_dir)
    segment = archive_dir / f"{time.time_ns():020d}-{uuid4().hex}.jsonl"
    temporary = segment.with_suffix(".tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(segment)
    finally:
        temporary.unlink(missing_ok=True)

    segments = _segment_sizes(archive_dir)
    total_bytes = sum(size for _, size in segments)
    evicted_files = 0
    evicted_bytes = 0
    for oldest, size in segments:
        if total_bytes <= max_bytes:
            break
        if oldest == segment:
            continue
        try:
            oldest.unlink()
        except FileNotFoundError:
            total_bytes -= size
            continue
        total_bytes -= size
        evicted_files += 1
        evicted_bytes += size
    if evicted_files:
        logger.info(
            "Retention archive FIFO evicted oldest evidence: files=%d bytes=%d "
            "size_bytes=%d ceiling_bytes=%d",
            evicted_files,
            evicted_bytes,
            total_bytes,
            max_bytes,
        )
    return True
