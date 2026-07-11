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
logger = logging.getLogger(__name__)


def retention_archive_enabled() -> bool:
    """Return whether archive-before-delete retention is enabled."""
    return os.environ.get("REFLEXIO_RETENTION_ARCHIVE", "").lower() in {
        "1",
        "true",
    }


def resolve_archive_directory(database_path: str) -> Path:
    """Return the configured archive directory or the SQLite-local default."""
    override = os.environ.get("REFLEXIO_RETENTION_ARCHIVE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(database_path).expanduser().parent / "archive"


def retention_archive_max_bytes() -> int:
    """Return the positive archive ceiling, defaulting to 10 GiB."""
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


def append_archive_batch(
    archive_dir: Path, rows_by_table: Mapping[str, list[dict[str, Any]]]
) -> bool:
    """Append one retention batch and evict oldest segments to stay bounded.

    One segment contains the target rows and any cascade-deleted rows, so FIFO
    eviction never splits a retention batch. The completed newest segment is
    installed before old segments are removed; a crash can temporarily exceed
    the ceiling but cannot replace newest evidence with older evidence.

    Returns:
        True when the batch was archived. False only when one batch is itself
        larger than the configured ceiling.
    """
    encoded = _encode_records(rows_by_table)
    if not encoded:
        return True
    max_bytes = retention_archive_max_bytes()
    row_count = sum(len(rows) for rows in rows_by_table.values())
    if len(encoded) > max_bytes:
        logger.error(
            "Retention archive batch exceeds the archive ceiling; skipping evidence "
            "while live-row retention continues: rows_skipped=%d batch_bytes=%d "
            "ceiling_bytes=%d",
            row_count,
            len(encoded),
            max_bytes,
        )
        return False

    archive_dir.mkdir(parents=True, exist_ok=True)
    segment = archive_dir / f"{time.time_ns():020d}-{uuid4().hex}.jsonl"
    temporary = segment.with_suffix(".tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(segment)
    finally:
        temporary.unlink(missing_ok=True)

    segments = sorted(
        archive_dir.glob("*.jsonl"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    total_bytes = sum(path.stat().st_size for path in segments)
    evicted_files = 0
    evicted_bytes = 0
    for oldest in segments:
        if total_bytes <= max_bytes:
            break
        if oldest == segment:
            continue
        size = oldest.stat().st_size
        oldest.unlink()
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
