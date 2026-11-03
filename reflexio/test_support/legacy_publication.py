"""Recorded synthetic original-v1 databases for current-owner upgrade tests."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parents[2] / "tests/fixtures/legacy_publication_v1"


def restore_legacy_publication_fixture(phase: str, destination: Path) -> Path:
    """Restore an exact, hash-checked predecessor into a test-owned path."""
    metadata = json.loads((_FIXTURES / "provenance.json").read_text())
    descriptor = metadata["fixtures"][phase]
    packed = (_FIXTURES / f"{phase}.sqlite.gz").read_bytes()
    if hashlib.sha256(packed).hexdigest() != descriptor["gzip_sha256"]:
        raise ValueError("Recorded predecessor compressed fixture changed")
    raw = gzip.decompress(packed)
    if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("Recorded predecessor fixture changed")
    destination.write_bytes(raw)
    return destination
