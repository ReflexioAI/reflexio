"""Organization selection shared by the OSS CLI and local server."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

DEFAULT_ORG_ID = "self-host-org"
_ORG_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class LocalInstanceError(ValueError):
    """A local installation cannot safely select its saved configuration."""


def is_valid_local_org_id(org_id: str) -> bool:
    """Whether a namespace can be used as a local filename component."""
    return _ORG_ID.fullmatch(org_id) is not None


def _legacy_org_id() -> str | None:
    return os.environ.get("REFLEXIO_DEFAULT_ORG_ID", "").strip() or None


def _instance_file() -> Path:
    from reflexio.cli.env_loader import get_env_path
    from reflexio.cli.paths import reflexio_home

    address = os.fsencode(get_env_path().absolute())
    key = hashlib.sha256(address).hexdigest()
    return reflexio_home() / "instances" / f"{key}.json"


def _config_file(org_id: str) -> Path:
    from reflexio.cli.paths import reflexio_home

    if not is_valid_local_org_id(org_id):
        raise LocalInstanceError("Local instance has an invalid organization name.")
    path = (reflexio_home() / "configs").resolve() / f"config_{org_id}.json"
    if path.is_symlink() or not path.is_file():
        raise LocalInstanceError(
            "Local instance requires its existing saved organization configuration."
        )
    return path


def _read_binding(path: Path) -> str | None:
    if path.parent.is_symlink() or path.is_symlink():
        raise LocalInstanceError("Local instance records must not be linked.")
    try:
        if path.exists() and not path.is_file():
            raise LocalInstanceError("Local instance record must be a regular file.")
        record = json.loads(path.read_bytes())
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as exc:
        raise LocalInstanceError("Could not read the local instance record.") from exc
    if (
        not isinstance(record, dict)
        or set(record) != {"org_id"}
        or not isinstance(record["org_id"], str)
        or not is_valid_local_org_id(record["org_id"])
    ):
        raise LocalInstanceError("Local instance record has an invalid organization.")
    return record["org_id"]


def configured_local_org_id() -> str:
    """Resolve the local launcher's namespace for config and storage."""
    legacy = _legacy_org_id()
    bound = _read_binding(_instance_file())
    if bound is None:
        return legacy or DEFAULT_ORG_ID
    if legacy is not None and legacy != bound:
        raise LocalInstanceError(
            "The legacy organization selector conflicts with this local instance."
        )
    _verify_bound_config(bound)
    return bound


def _verify_bound_config(org_id: str) -> None:
    """Refuse before startup if the instance lost its saved SQLite file."""
    from reflexio.config_file import LocalConfigFile
    from reflexio.models.config_schema import (
        StorageConfigSQLite,
        validate_stored_config,
    )

    config = validate_stored_config(
        LocalConfigFile(_config_file(org_id)).read_document()
    )
    if isinstance(config.storage_config, StorageConfigSQLite):
        _verify_sqlite(org_id, config.storage_config.db_path)


def _verify_sqlite(org_id: str, db_path: str | None) -> None:
    from reflexio.server.services.storage.sqlite_storage._dataset_path import (
        DatasetIdentityError,
        verify_existing_dataset_owner,
    )

    if db_path is None or not Path(db_path).is_absolute():
        raise LocalInstanceError(
            "A bound SQLite dataset requires its saved absolute path."
        )
    try:
        verify_existing_dataset_owner(Path(db_path), org_id)
    except DatasetIdentityError as exc:
        raise LocalInstanceError(str(exc)) from exc


def bind_local_instance() -> str:
    """Persist the currently selected installation's organization after verification."""
    from filelock import FileLock, Timeout

    from reflexio.config_file import LocalConfigFile, _publish_bytes, _sync_directory
    from reflexio.models.config_schema import (
        StorageConfigSQLite,
        validate_stored_config,
    )

    instance = _instance_file()
    if instance.parent.is_symlink():
        raise LocalInstanceError("Local instance records must not be linked.")
    instance.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _sync_directory(instance.parent.parent)
    lock = FileLock(f"{instance}.lock", timeout=30, mode=0o600)
    try:
        lock.acquire()
    except Timeout as exc:
        raise LocalInstanceError(
            "Local instance binding is already in progress."
        ) from exc
    try:
        org_id = configured_local_org_id()
        config = validate_stored_config(
            LocalConfigFile(_config_file(org_id)).read_document()
        )
        if isinstance(config.storage_config, StorageConfigSQLite):
            _pin_sqlite(org_id, config.storage_config.db_path)

        record = json.dumps({"org_id": org_id}).encode() + b"\n"
        if not _publish_bytes(instance, record, create_only=True):
            if _read_binding(instance) != org_id:
                raise LocalInstanceError(
                    "Another organization already owns this local instance."
                )
            _sync_directory(instance.parent)
        return org_id
    finally:
        lock.release()


def _pin_sqlite(org_id: str, db_path: str | None) -> None:
    if db_path is not None:
        _verify_sqlite(org_id, db_path)
        return

    from reflexio.models.config_schema import (
        StorageConfigSQLite,
        validate_stored_config,
    )
    from reflexio.server import LOCAL_STORAGE_PATH
    from reflexio.server.services.configurator.local_file_config_storage import (
        LocalFileConfigStorage,
    )
    from reflexio.server.services.storage.sqlite_storage._dataset_path import (
        LEGACY_DB_FILENAME,
        derive_db_path,
        stored_identity_labels,
    )

    derived = derive_db_path(LOCAL_STORAGE_PATH, org_id)
    selected = derived
    if not derived.exists():
        selected = Path(LOCAL_STORAGE_PATH) / LEGACY_DB_FILENAME
        if (stored_identity_labels(selected) or set()) - {org_id}:
            raise LocalInstanceError(
                "The legacy SQLite file contains multiple organizations."
            )
    absolute = str(selected.absolute())
    _verify_sqlite(org_id, absolute)

    def pin(document: dict, _generation: int) -> None:
        saved = validate_stored_config(document).storage_config
        if not isinstance(saved, StorageConfigSQLite) or saved.db_path not in (
            None,
            absolute,
        ):
            raise LocalInstanceError("The saved SQLite target changed during binding.")
        document["storage_config"]["db_path"] = absolute

    LocalFileConfigStorage(org_id).prepare_for_startup(edit=pin)
