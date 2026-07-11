"""Dependency-free filesystem portability checks for retention archives."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "reflexio/server/services/storage/retention_archive.py"
)
SPEC = importlib.util.spec_from_file_location(
    "retention_archive_portability", MODULE_PATH
)
assert SPEC and SPEC.loader
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class RetentionArchiveFilesystemTest(unittest.TestCase):
    """Exercise the archive's real filesystem operations on each CI OS."""

    def test_atomic_segments_fifo_and_default_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_dir = archive.resolve_archive_directory(str(root / "reflexio.db"))
            self.assertEqual(archive_dir, root / "archive")
            with mock.patch.dict(
                os.environ, {"REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES": "100000"}
            ):
                self.assertTrue(
                    archive.append_archive_batch(archive_dir, {"rows": [{"id": "old"}]})
                )
                first_size = next(archive_dir.glob("*.jsonl")).stat().st_size
            with mock.patch.dict(
                os.environ,
                {"REFLEXIO_RETENTION_ARCHIVE_MAX_BYTES": str(first_size + 1)},
            ):
                self.assertTrue(
                    archive.append_archive_batch(archive_dir, {"rows": [{"id": "new"}]})
                )
            self.assertEqual(len(list(archive_dir.glob("*.jsonl"))), 1)
            self.assertEqual(list(archive_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
