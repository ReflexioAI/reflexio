"""Base class for DiskStorage — directory setup, file I/O helpers, and QMD integration."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    UserPlaybook,
)
from reflexio.models.config_schema import StorageConfigDisk
from reflexio.server import LOCAL_STORAGE_PATH
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.storage_base import BaseStorage

from ._file_io import (
    deserialize_embedding,
    deserialize_entity,
    serialize_embedding,
    serialize_entity,
)
from ._qmd_client import QMDClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Format dispatch tables — add new formats here (one line each)
# ---------------------------------------------------------------------------

_SERIALIZERS: dict[str, Callable[[BaseModel], str]] = {
    ".md": lambda model: serialize_entity(model),
    ".json": lambda model: model.model_dump_json(indent=2),
}

_DESERIALIZERS: dict[str, Callable[[str, type], object]] = {
    ".md": lambda text, cls: deserialize_entity(text, cls),
    ".json": lambda text, cls: cls.model_validate_json(text),
}

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_SERIALIZERS)

# Default file extension for new entity files — change here to switch format
_DEFAULT_ENTITY_EXT = ".md"


class DiskStorageBase(BaseStorage):
    """Storage backend that persists entities as files with YAML frontmatter.

    Directory layout::

        {base_dir}/disk_{org_id}/
            profiles/{user_id}/{profile_id}
            interactions/{user_id}/{interaction_id}
            requests/{request_id}
            user_playbooks/{user_playbook_id}
            agent_playbooks/{agent_playbook_id}
            evaluations/{result_id}
            operation_states/{service_name}.json
            change_logs/profile/{id}.json
            change_logs/playbook_aggregation/{id}.json
            .embeddings/{entity_type}/{entity_id}.json

    Entity files use the extension defined by ``_DEFAULT_ENTITY_EXT``.
    Search operations are delegated to QMD (tobi/qmd) for BM25, vector,
    and hybrid search over the entity files.
    """

    # Subdirectory names
    _PROFILES = "profiles"
    _INTERACTIONS = "interactions"
    _REQUESTS = "requests"
    _USER_PLAYBOOKS = "user_playbooks"
    _AGENT_PLAYBOOKS = "agent_playbooks"
    _EVALUATIONS = "evaluations"
    _OPERATION_STATES = "operation_states"
    _CHANGE_LOGS_PROFILE = "change_logs/profile"
    _CHANGE_LOGS_PLAYBOOK_AGG = "change_logs/playbook_aggregation"
    _EMBEDDINGS = ".embeddings"

    def __init__(
        self,
        org_id: str,
        base_dir: str | None = None,
        config: StorageConfigDisk | None = None,
    ) -> None:
        self.config: StorageConfigDisk | None = config
        qmd_binary = "qmd"
        if self.config:
            base_dir = self.config.dir_path
            qmd_binary = self.config.qmd_binary
            if not base_dir:
                err_msg = "DiskStorage received empty directory"
                logger.error(err_msg)
                raise StorageError(err_msg)
            if not Path(base_dir).is_absolute():
                err_msg = f"DiskStorage received a non absolute path {base_dir}"
                logger.error(err_msg)
                raise StorageError(err_msg)
            try:
                if not Path(base_dir).exists():
                    Path(base_dir).mkdir(parents=True, exist_ok=True)
            except OSError as e:
                err_msg = f"DiskStorage cannot create directory at {base_dir}"
                logger.error(err_msg)
                raise StorageError(err_msg) from e

        if base_dir is None:
            base_dir = LOCAL_STORAGE_PATH
        try:
            if not Path(base_dir).exists():
                Path(base_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            err_msg = f"DiskStorage cannot create directory at {base_dir}"
            logger.error(err_msg)
            raise StorageError(err_msg) from e
        if not Path(base_dir).is_dir():
            err_msg = f"DiskStorage specified an invalid directory at {base_dir}"
            logger.error(err_msg)
            raise StorageError(err_msg)

        logger.info("DiskStorage for org %s uses directory %s", org_id, base_dir)
        super().__init__(org_id, base_dir)
        self._lock = threading.RLock()

        self._org_dir = Path(base_dir) / f"disk_{self._safe_filename(org_id)}"
        self._ensure_dirs()

        # Initialize QMD client for search
        self._qmd = QMDClient(
            collection_path=self._org_dir,
            collection_name=f"reflexio_{self._safe_filename(org_id)}",
            qmd_binary=qmd_binary,
        )

    def _ensure_dirs(self) -> None:
        """Create all required subdirectories."""
        for subdir in (
            self._PROFILES,
            self._INTERACTIONS,
            self._REQUESTS,
            self._USER_PLAYBOOKS,
            self._AGENT_PLAYBOOKS,
            self._EVALUATIONS,
            self._OPERATION_STATES,
            self._CHANGE_LOGS_PROFILE,
            self._CHANGE_LOGS_PLAYBOOK_AGG,
            self._EMBEDDINGS,
        ):
            (self._org_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Entity I/O — format dispatched by file extension via _SERIALIZERS/_DESERIALIZERS
    # ------------------------------------------------------------------

    def _write_entity(self, path: Path, model: BaseModel) -> None:
        """Atomic write: serialize entity, write to .tmp, then rename.

        Format is determined by file extension (e.g., ``.md`` → frontmatter,
        ``.json`` → JSON).  Add new formats to ``_SERIALIZERS``.

        Args:
            path: Target entity file path.
            model: The Pydantic entity to serialize.
        """
        serializer = _SERIALIZERS.get(path.suffix)
        if not serializer:
            raise StorageError(f"Unsupported file format: {path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(serializer(model), encoding="utf-8")
        tmp.rename(path)

    def _read_entity(self, path: Path, model_class: type[T]) -> T:
        """Read and parse a single entity file.

        Format is determined by file extension.  Add new formats to
        ``_DESERIALIZERS``.

        Args:
            path: The entity file to read.
            model_class: The Pydantic model class to instantiate.

        Returns:
            An instance of model_class populated from the file.
        """
        deserializer = _DESERIALIZERS.get(path.suffix)
        if not deserializer:
            raise StorageError(f"Unsupported file format: {path.suffix}")
        entity = deserializer(path.read_text(encoding="utf-8"), model_class)
        # Load sidecar embedding if present
        if "embedding" in model_class.model_fields and (
            embedding := self._read_embedding(path)
        ):
            entity.embedding = embedding
        return entity  # type: ignore[return-value]

    def _list_entities(self, directory: Path, model_class: type[T]) -> list[T]:
        """List all entities in a flat directory.

        Scans for all supported file extensions.

        Args:
            directory: Directory to scan.
            model_class: The Pydantic model class for each file.

        Returns:
            List of parsed entities.
        """
        if not directory.exists():
            return []
        return [
            self._read_entity(p, model_class)
            for p in sorted(directory.iterdir())
            if p.suffix in _SUPPORTED_EXTENSIONS and p.is_file()
        ]

    def _list_entities_recursive(
        self, directory: Path, model_class: type[T]
    ) -> list[T]:
        """List all entities recursively (for user-scoped directories).

        Scans for all supported file extensions.

        Args:
            directory: Root directory to scan recursively.
            model_class: The Pydantic model class for each file.

        Returns:
            List of parsed entities from all subdirectories.
        """
        if not directory.exists():
            return []
        return [
            self._read_entity(p, model_class)
            for p in sorted(directory.rglob("*"))
            if p.suffix in _SUPPORTED_EXTENSIONS and p.is_file()
        ]

    def _entity_path(self, directory: Path, name: str) -> Path:
        """Build an entity file path with the configured extension.

        Args:
            directory: Parent directory for the entity.
            name: Entity identifier (will be sanitized).

        Returns:
            Path like ``directory / safe_name + _DEFAULT_ENTITY_EXT``.
        """
        return directory / f"{self._safe_filename(name)}{_DEFAULT_ENTITY_EXT}"

    def _scan_entities(self, directory: Path, *, recursive: bool = False) -> list[Path]:
        """Scan a directory for entity files with any supported extension.

        Args:
            directory: Directory to scan.
            recursive: If True, scan subdirectories recursively.

        Returns:
            Sorted list of matching file paths.
        """
        if not directory.exists():
            return []
        scanner = directory.rglob("*") if recursive else directory.iterdir()
        return sorted(
            p for p in scanner if p.suffix in _SUPPORTED_EXTENSIONS and p.is_file()
        )

    # ------------------------------------------------------------------
    # Dict I/O (for operation states — plain dicts, not Pydantic models)
    # ------------------------------------------------------------------

    def _write_dict(self, path: Path, data_dict: dict) -> None:
        """Write a plain dict as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data_dict, indent=2, default=str))
        tmp.rename(path)

    # ------------------------------------------------------------------
    # Embedding sidecar I/O
    # ------------------------------------------------------------------

    def _write_embedding(self, entity_path: Path, embedding: list[float]) -> None:
        """Write an embedding vector as a sidecar JSON file.

        Sidecar path is derived from the entity file path:
        .embeddings/{entity_type}/{entity_id}.json

        Args:
            entity_path: The entity file path.
            embedding: The embedding vector.
        """
        if not embedding:
            return
        # Derive sidecar path: .../profiles/user_abc/prof_42.md
        # → .embeddings/profiles/user_abc/prof_42.json
        relative = entity_path.relative_to(self._org_dir)
        sidecar = self._org_dir / self._EMBEDDINGS / relative.with_suffix(".json")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tmp = sidecar.with_suffix(".tmp")
        tmp.write_text(serialize_embedding(embedding))
        tmp.rename(sidecar)

    def _read_embedding(self, entity_path: Path) -> list[float]:
        """Read an embedding vector from a sidecar JSON file.

        Args:
            entity_path: The entity file path.

        Returns:
            The embedding vector, or empty list if sidecar doesn't exist.
        """
        relative = entity_path.relative_to(self._org_dir)
        sidecar = self._org_dir / self._EMBEDDINGS / relative.with_suffix(".json")
        if not sidecar.exists():
            return []
        try:
            return deserialize_embedding(sidecar.read_text())
        except (json.JSONDecodeError, ValueError):
            return []

    def _delete_embedding(self, entity_path: Path) -> None:
        """Delete the sidecar embedding file for an entity."""
        relative = entity_path.relative_to(self._org_dir)
        sidecar = self._org_dir / self._EMBEDDINGS / relative.with_suffix(".json")
        if sidecar.exists():
            sidecar.unlink()

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------

    def _next_id(self, directory: Path) -> int:
        """Get next auto-increment ID by scanning entity filenames."""
        max_id = 0
        for p in self._scan_entities(directory, recursive=True):
            try:
                file_id = int(p.stem)
                if file_id > max_id:
                    max_id = file_id
            except ValueError:
                continue
        return max_id + 1

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Sanitize a string for use as a safe filename, preventing path traversal.

        Args:
            name (str): The raw name to sanitize

        Returns:
            str: A safe filename

        Raises:
            StorageError: If the name is empty or resolves to empty after sanitization
        """
        if not name:
            raise StorageError("Filename must not be empty")
        name = name.replace("\x00", "")
        name = re.sub(r"[^a-zA-Z0-9\-_.]", "_", name)
        name = re.sub(r"\.{2,}", ".", name)
        name = name.strip(".")
        if not name:
            raise StorageError("Filename resolved to empty after sanitization")
        return name

    def _user_dir(self, parent_dir: Path, user_id: str) -> Path:
        """Return a sanitized user subdirectory under a parent directory."""
        return parent_dir / self._safe_filename(user_id)

    def _current_timestamp(self) -> str:
        """Return a timezone-aware ISO timestamp."""
        return datetime.now(UTC).isoformat()

    # Directory accessors
    def _profiles_dir(self) -> Path:
        return self._org_dir / self._PROFILES

    def _interactions_dir(self) -> Path:
        return self._org_dir / self._INTERACTIONS

    def _requests_dir(self) -> Path:
        return self._org_dir / self._REQUESTS

    def _user_playbooks_dir(self) -> Path:
        return self._org_dir / self._USER_PLAYBOOKS

    def _agent_playbooks_dir(self) -> Path:
        return self._org_dir / self._AGENT_PLAYBOOKS

    def _evaluations_dir(self) -> Path:
        return self._org_dir / self._EVALUATIONS

    def _operation_states_dir(self) -> Path:
        return self._org_dir / self._OPERATION_STATES

    def _profile_change_logs_dir(self) -> Path:
        return self._org_dir / self._CHANGE_LOGS_PROFILE

    def _playbook_agg_change_logs_dir(self) -> Path:
        return self._org_dir / self._CHANGE_LOGS_PLAYBOOK_AGG

    def _clear_dir(self, directory: Path) -> None:
        """Remove all contents of a directory and recreate it."""
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    def _should_delete_playbook(
        self,
        playbook: AgentPlaybook | UserPlaybook,
        playbook_name: str,
        agent_version: str | None,
    ) -> bool:
        """Helper to determine if a playbook should be deleted."""
        if playbook.playbook_name != playbook_name:
            return False
        return not (
            agent_version is not None and playbook.agent_version != agent_version
        )

    def _trigger_qmd_update(self) -> None:
        """Trigger QMD index update after file writes.

        Called after batch operations to keep the search index in sync.
        """
        try:
            self._qmd.update_index()
        except (OSError, subprocess.SubprocessError):
            logger.warning(
                "QMD index update failed; search may be stale", exc_info=True
            )
