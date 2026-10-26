"""One persistence owner for strict reads, atomic writes and local file upgrades.

File-generation metadata belongs to this storage format, not Config or its wire
API. Historical conversion runs only through explicit pre-launch preparation.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from filelock import FileLock

from reflexio.config_document import decode_config_document, encode_config_document
from reflexio.config_env import StoredConfigurationError

FILE_FORMAT_VERSION = 1
_METADATA_KEY = "_reflexio_config_file"


def _decode(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document = decode_config_document(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise StoredConfigurationError(
            "Saved configuration is not finite JSON"
        ) from None
    if _METADATA_KEY not in document:
        return document, {}
    metadata = document.pop(_METADATA_KEY)
    if (
        not isinstance(metadata, dict)
        or set(metadata) - {"version", "original_sha256"}
        or type(metadata.get("version")) is not int
        or not 1 <= metadata["version"] <= FILE_FORMAT_VERSION
        or (
            "original_sha256" in metadata
            and (
                not isinstance(metadata["original_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", metadata["original_sha256"]) is None
            )
        )
    ):
        raise StoredConfigurationError(
            "Unrecognized local configuration file generation; use the matching SDK "
            "or restore the original file before restarting"
        )
    return document, metadata


def _reject_ambiguous_unmarked_policy(data: dict[str, Any]) -> None:
    optimizer = data.get("playbook_optimizer_config")
    if isinstance(optimizer, dict) and "agent_playbooks_enabled" in optimizer:
        raise StoredConfigurationError(
            "Unmarked configuration contains agent_playbooks_enabled, which "
            "was ignored by the previous schema. Stop old writers and use "
            "config upgrade-file to explicitly select canonical policy."
        )


def encode_local_config_document(
    data: dict[str, Any], metadata: dict[str, Any] | None = None
) -> str:
    """Encode a deliberate current file, keeping format out of the logical model."""
    return (
        encode_config_document(
            {
                **data,
                _METADATA_KEY: {**(metadata or {}), "version": FILE_FORMAT_VERSION},
            },
            indent=2,
        )
        + "\n"
    )


def _sync_directory(directory: Path) -> None:
    if os.name == "posix":
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _publish_bytes(target: Path, raw: bytes, *, create_only: bool = False) -> bool:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if create_only:
            try:
                os.link(temporary, target)
            except FileExistsError:
                return False
        else:
            temporary.replace(target)
        _sync_directory(target.parent)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class LocalConfigFile:
    """Serialize cooperating writers and preserve exact originals on upgrade."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _target(self) -> Path:
        for part in (self.path, *self.path.parents):
            if part.is_symlink() and not part.exists():
                raise StoredConfigurationError(
                    "Saved configuration link has a missing target; restore it before starting"
                )
        return self.path.resolve()

    def read_document(self) -> dict[str, Any]:
        """Read one complete file without transforming or writing configuration."""
        target = self._target()
        if target.exists() and not target.is_file():
            raise StoredConfigurationError("Saved configuration must be a regular file")
        document, _metadata = _decode(target.read_bytes())
        return document

    def publish_document(self, document: str, *, create_only: bool = False) -> bool:
        """Publish an explicit full replacement; this is not a read/merge patch."""
        from reflexio.models.config_schema import validate_stored_config

        data, _supplied_metadata = _decode(document.encode("utf-8"))
        validate_stored_config(data)
        target = self._target()
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with FileLock(f"{target}.lock", timeout=30, mode=0o600):
            if create_only and target.exists():
                return False
            metadata: dict[str, Any] = {"version": FILE_FORMAT_VERSION}
            if target.exists():
                if not target.is_file():
                    raise StoredConfigurationError(
                        "Saved configuration must be a regular file"
                    )
                _previous, metadata = _decode(target.read_bytes())
                metadata = {**metadata, "version": FILE_FORMAT_VERSION}
            if self._target() != target:
                raise StoredConfigurationError(
                    "Saved configuration link changed; retry the write"
                )
            encoded = encode_local_config_document(data, metadata)
            return _publish_bytes(target, encoded.encode(), create_only=create_only)

    def edit_document(self, edit: Callable[[dict[str, Any]], None]) -> bool:
        """Apply a storage-only edit to the latest strict raw snapshot.

        Callbacks run synchronously under the publication lock and must not do
        external I/O. Explicit full replacements keep their separate semantics.
        """
        from reflexio.models.config_schema import validate_stored_config

        target = self._target()
        if not target.is_file():
            raise StoredConfigurationError("Saved configuration must be a regular file")
        with FileLock(f"{target}.lock", timeout=30, mode=0o600):
            original = target.read_bytes()
            data, metadata = _decode(original)
            if not metadata:
                _reject_ambiguous_unmarked_policy(data)
            validate_stored_config(data)
            edit(data)
            validate_stored_config(data)
            previous, _ = _decode(original)
            if data == previous:
                return False
            if self._target() != target or target.read_bytes() != original:
                raise StoredConfigurationError(
                    "Saved configuration changed during editing; retry the write"
                )
            encoded = encode_local_config_document(data, metadata)
            return _publish_bytes(target, encoded.encode())

    def prepare(self) -> bool:
        """Upgrade an older generation before serving, retaining its original bytes.

        The launcher must stop old serving writers first. The lock serializes
        current product writers; it is not a CAS against arbitrary file editors.
        """
        from reflexio.config_migration import canonicalize_agent_optimizer_policy
        from reflexio.models.config_schema import validate_stored_config

        target = self._target()
        if not target.exists():
            return False
        if not target.is_file():
            raise StoredConfigurationError("Saved configuration must be a regular file")
        with FileLock(f"{target}.lock", timeout=30, mode=0o600):
            original = target.read_bytes()
            data, metadata = _decode(original)
            if metadata.get("version") == FILE_FORMAT_VERSION:
                validate_stored_config(data)
                return False
            _reject_ambiguous_unmarked_policy(data)
            # Generation0 is the supported unmarked canonical schema. Only
            # this newly retired pair is converted, not arbitrary old aliases.
            canonical = dict(data)
            if "playbook_optimizer_config" in canonical:
                canonical["playbook_optimizer_config"] = (
                    canonicalize_agent_optimizer_policy(
                        canonical["playbook_optimizer_config"]
                    )
                )
            validate_stored_config(canonical)
            digest = hashlib.sha256(original).hexdigest()
            encoded = encode_local_config_document(
                canonical, {"original_sha256": digest}
            ).encode()
            root = target.parent / ".config-migrations"
            if root.is_symlink():
                raise StoredConfigurationError(
                    "Configuration backup directory must not be a link"
                )
            root.mkdir(mode=0o700, exist_ok=True)
            _sync_directory(root.parent)
            directory = root / target.name
            if directory.is_symlink():
                raise StoredConfigurationError(
                    "Configuration backup directory must not be a link"
                )
            directory.mkdir(mode=0o700, exist_ok=True)
            _sync_directory(root)
            backup = directory / f"{digest}.json"
            if backup.is_symlink() or (backup.exists() and not backup.is_file()):
                raise StoredConfigurationError(
                    "Configuration backup must be a regular file"
                )
            if (
                not _publish_bytes(backup, original, create_only=True)
                and backup.read_bytes() != original
            ):
                raise StoredConfigurationError(
                    "Saved configuration original does not match its receipt"
                )
            if self._target() != target or target.read_bytes() != original:
                raise StoredConfigurationError(
                    "Saved configuration changed during preparation; retry startup"
                )
            _publish_bytes(target, encoded)
            return True
