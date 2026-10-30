"""Explicit offline JSON conversion; never discovers an operator config path."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from reflexio.config_file import LocalConfigFile, encode_local_config_document
from reflexio.config_migration import canonicalize_config_document


def convert_config_file(source: Path, destination: Path) -> None:
    """Write a canonical copy atomically, retaining source bytes and metadata.

    The output records its current local-file generation; pure logical
    canonicalize_config_document remains independent of this storage format.
    The caller explicitly selects both paths. An existing destination is
    replaced; using the input itself, including a symlink to it, is rejected.
    The temporary output uses tempfile's restrictive permissions.
    """
    if source.resolve() == destination.resolve():
        raise ValueError(
            "Input and output paths must differ; keep the original configuration."
        )
    try:
        payload = LocalConfigFile(source).read_document_for_conversion()
        converted = canonicalize_config_document(payload)
        encoded = encode_local_config_document(converted)
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


def prepare_local_config(
    org_id: str | None = None, *, base_dir: str | None = None
) -> bool:
    """Prepare only the selected local org before strict startup readers.

    The serving launcher owns stopping its predecessor first. Runtime reads
    never call this method, and enterprise auth rows use their migration owner.
    """
    from reflexio.cli.bootstrap_config import default_org_id
    from reflexio.config_file import LocalConfigFile
    from reflexio.server.services.configurator.local_file_config_storage import (
        LocalFileConfigStorage,
    )

    storage = LocalFileConfigStorage(org_id or default_org_id(), base_dir=base_dir)
    return LocalConfigFile(storage.config_file).prepare()
