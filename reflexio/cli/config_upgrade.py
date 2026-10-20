"""Explicit offline JSON conversion; never discovers an operator config path."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

from reflexio.config_migration import canonicalize_config_document


def convert_config_file(source: Path, destination: Path) -> None:
    """Write a canonical copy atomically, retaining source bytes and metadata.

    The caller explicitly selects both paths. An existing destination is
    replaced; using the input itself, including a symlink to it, is rejected.
    The temporary output uses tempfile's restrictive permissions.
    """
    if source.resolve() == destination.resolve():
        raise ValueError(
            "Input and output paths must differ; keep the original configuration."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Configuration must be a JSON object")
        converted = canonicalize_config_document(payload)
        encoded = json.dumps(converted, indent=2, allow_nan=False) + "\n"
    except ValueError:
        raise ValueError(
            "Input is not a valid configuration for safe conversion."
        ) from None
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
