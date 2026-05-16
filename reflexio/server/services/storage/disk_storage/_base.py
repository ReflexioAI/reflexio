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
from typing import Any, TypeVar, cast

import yaml
from pydantic import BaseModel

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    Interaction,
    PlaybookAggregationChangeLog,
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
    ProfileChangeLog,
    Request,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.config_schema import StorageConfigDisk
from reflexio.server import LOCAL_STORAGE_PATH
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.retention import RETENTION_TARGETS_BY_NAME
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
    _AGENT_PLAYBOOK_SOURCE_MAP = "playbook_optimizer/source_map"
    _PLAYBOOK_OPT_JOBS = "playbook_optimizer/jobs"
    _PLAYBOOK_OPT_CANDIDATES = "playbook_optimizer/candidates"
    _PLAYBOOK_OPT_EVALUATIONS = "playbook_optimizer/evaluations"
    _PLAYBOOK_OPT_EVENTS = "playbook_optimizer/events"
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
            self._AGENT_PLAYBOOK_SOURCE_MAP,
            self._PLAYBOOK_OPT_JOBS,
            self._PLAYBOOK_OPT_CANDIDATES,
            self._PLAYBOOK_OPT_EVALUATIONS,
            self._PLAYBOOK_OPT_EVENTS,
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
        text = serializer(model)
        if isinstance(model, UserProfile):
            created_at = self._entity_metadata_value(path, "created_at")
            if created_at is None:
                created_at = int(datetime.now(UTC).timestamp())
            text = self._with_entity_metadata(
                text, path.suffix, {"created_at": created_at}
            )
        tmp.write_text(text, encoding="utf-8")
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
        entity = cast(T, deserializer(path.read_text(encoding="utf-8"), model_class))
        # Load sidecar embedding if present
        if "embedding" in model_class.model_fields and (
            embedding := self._read_embedding(path)
        ):
            cast(Any, entity).embedding = embedding
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

    def _entity_metadata_value(self, path: Path, key: str) -> Any | None:
        """Read a metadata value directly from an entity file."""
        if not path.exists():
            return None
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8") or "{}")
                return data.get(key) if isinstance(data, dict) else None
            if path.suffix != ".md":
                return None
            frontmatter, _body = self._split_markdown_frontmatter(
                path.read_text(encoding="utf-8")
            )
            return frontmatter.get(key)
        except (OSError, json.JSONDecodeError, ValueError, yaml.YAMLError):
            return None

    def _with_entity_metadata(
        self, text: str, suffix: str, metadata: dict[str, Any]
    ) -> str:
        """Return serialized entity text with extra metadata preserved."""
        if suffix == ".json":
            data = json.loads(text or "{}")
            if isinstance(data, dict):
                data.update(metadata)
                return json.dumps(data, indent=2, default=str)
            return text
        if suffix != ".md":
            return text
        frontmatter, body = self._split_markdown_frontmatter(text)
        frontmatter.update(metadata)
        yaml_text = yaml.dump(
            frontmatter,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        ).rstrip("\n")
        return f"---\n{yaml_text}\n---{body}"

    @staticmethod
    def _split_markdown_frontmatter(text: str) -> tuple[dict[str, Any], str]:
        if not text.startswith("---"):
            return {}, text
        close_idx = text.find("\n---\n", 3)
        if close_idx == -1:
            if not text.endswith("\n---"):
                return {}, text
            close_idx = len(text) - 4
            body = ""
        else:
            body = text[close_idx + 4 :]
        yaml_text = text[3 : close_idx + 1]
        data = yaml.safe_load(yaml_text) or {}
        return data if isinstance(data, dict) else {}, body

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

    def count_retention_target_rows(self, target_name: str) -> int:
        if target_name == "agent_playbook_source_user_playbooks":
            with self._lock:
                return len(self._source_map_rows())
        spec = self._disk_retention_spec(target_name)
        if spec is None:
            return 0
        directory, _model, recursive = spec
        with self._lock:
            return len(self._scan_entities(directory, recursive=recursive))

    def delete_oldest_retention_target_rows(self, target_name: str, count: int) -> int:
        if count <= 0:
            return 0
        if target_name == "agent_playbook_source_user_playbooks":
            return self._delete_oldest_source_map_files(count)
        spec = self._disk_retention_spec(target_name)
        if spec is None:
            return 0
        directory, model, recursive = spec
        target = RETENTION_TARGETS_BY_NAME[target_name]

        with self._lock:
            entries: list[tuple[Any, Path, Any]] = []
            for path in self._scan_entities(directory, recursive=recursive):
                entity = self._read_entity(path, model)
                entries.append(
                    (
                        self._disk_retention_order_value(
                            target_name, target.order_column, path, entity
                        ),
                        path,
                        entity,
                    )
                )
            entries.sort(key=lambda item: (item[0], item[1].name))
            selected = entries[:count]
            if not selected:
                return 0
            self._delete_disk_retention_dependencies(target_name, selected)
            for _order, path, _entity in selected:
                if path.exists():
                    self._delete_embedding(path)
                    path.unlink()
        self._trigger_qmd_update()
        return len(selected)

    def _disk_retention_spec(
        self, target_name: str
    ) -> tuple[Path, type[BaseModel], bool] | None:
        specs: dict[str, tuple[Path, type[BaseModel], bool]] = {
            "profiles": (self._profiles_dir(), UserProfile, True),
            "interactions": (self._interactions_dir(), Interaction, True),
            "requests": (self._requests_dir(), Request, False),
            "user_playbooks": (self._user_playbooks_dir(), UserPlaybook, False),
            "agent_playbooks": (self._agent_playbooks_dir(), AgentPlaybook, False),
            "agent_success_evaluation_result": (
                self._evaluations_dir(),
                AgentSuccessEvaluationResult,
                False,
            ),
            "profile_change_logs": (
                self._profile_change_logs_dir(),
                ProfileChangeLog,
                False,
            ),
            "playbook_aggregation_change_logs": (
                self._playbook_agg_change_logs_dir(),
                PlaybookAggregationChangeLog,
                False,
            ),
            "playbook_optimization_jobs": (
                self._playbook_opt_jobs_dir(),
                PlaybookOptimizationJob,
                False,
            ),
            "playbook_optimization_candidates": (
                self._playbook_opt_candidates_dir(),
                PlaybookOptimizationCandidate,
                False,
            ),
            "playbook_optimization_evaluations": (
                self._playbook_opt_evaluations_dir(),
                PlaybookOptimizationEvaluation,
                False,
            ),
            "playbook_optimization_events": (
                self._playbook_opt_events_dir(),
                PlaybookOptimizationEvent,
                False,
            ),
        }
        if target_name not in RETENTION_TARGETS_BY_NAME:
            raise ValueError(f"Unknown retention target: {target_name}")
        return specs.get(target_name)

    def _disk_retention_order_value(
        self,
        target_name: str,
        order_column: str,
        path: Path,
        entity: BaseModel,
    ) -> float:
        value = (
            self._entity_metadata_value(path, order_column)
            if target_name == "profiles"
            else getattr(entity, order_column, None)
        )
        if value is None:
            return path.stat().st_mtime
        return self._normalize_retention_order_value(value)

    @staticmethod
    def _normalize_retention_order_value(value: Any) -> float:
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
            try:
                return datetime.fromisoformat(value).timestamp()
            except ValueError:
                return 0.0
        return 0.0

    def _delete_disk_retention_dependencies(
        self, target_name: str, entries: list[tuple[Any, Path, Any]]
    ) -> None:
        entities = [entity for _order, _path, entity in entries]
        if target_name == "requests":
            request_ids = {entity.request_id for entity in entities}
            self._delete_interactions_for_request_ids(request_ids)
        elif target_name == "user_playbooks":
            user_playbook_ids = {int(entity.user_playbook_id) for entity in entities}
            self._delete_source_windows_for_user_playbook_ids(user_playbook_ids)
        elif target_name == "agent_playbooks":
            agent_playbook_ids = {int(entity.agent_playbook_id) for entity in entities}
            self._delete_source_map_files(agent_playbook_ids)
        elif target_name == "playbook_optimization_jobs":
            job_ids = {int(entity.job_id) for entity in entities}
            self._delete_optimizer_files_for_job_ids(job_ids)
        elif target_name == "playbook_optimization_candidates":
            candidate_ids = {int(entity.candidate_id) for entity in entities}
            self._delete_optimizer_evaluation_files_for_candidate_ids(candidate_ids)

    def _delete_interactions_for_request_ids(self, request_ids: set[str]) -> None:
        if not request_ids:
            return
        for path in self._scan_entities(self._interactions_dir(), recursive=True):
            interaction = self._read_entity(path, Interaction)
            if interaction.request_id in request_ids:
                self._delete_embedding(path)
                path.unlink()

    def _delete_source_map_files(self, agent_playbook_ids: set[int]) -> None:
        for agent_playbook_id in agent_playbook_ids:
            path = self._entity_path(
                self._agent_playbook_source_map_dir(), str(agent_playbook_id)
            )
            if path.exists():
                path.unlink()

    def _delete_source_windows_for_user_playbook_ids(
        self, user_playbook_ids: set[int]
    ) -> None:
        if not user_playbook_ids:
            return
        for path in self._scan_entities(self._agent_playbook_source_map_dir()):
            windows = self._read_source_windows_data(path)
            filtered = [
                item
                for item in windows
                if int(item.get("user_playbook_id", 0)) not in user_playbook_ids
            ]
            if filtered:
                self._write_dict(path, {"source_windows": filtered})
            else:
                path.unlink()

    def _delete_optimizer_files_for_job_ids(self, job_ids: set[int]) -> None:
        if not job_ids:
            return
        for directory, model in (
            (self._playbook_opt_evaluations_dir(), PlaybookOptimizationEvaluation),
            (self._playbook_opt_events_dir(), PlaybookOptimizationEvent),
            (self._playbook_opt_candidates_dir(), PlaybookOptimizationCandidate),
        ):
            for path in self._scan_entities(directory):
                entity = self._read_entity(path, model)
                if int(entity.job_id) in job_ids:
                    path.unlink()

    def _delete_optimizer_evaluation_files_for_candidate_ids(
        self, candidate_ids: set[int]
    ) -> None:
        if not candidate_ids:
            return
        for path in self._scan_entities(self._playbook_opt_evaluations_dir()):
            entity = self._read_entity(path, PlaybookOptimizationEvaluation)
            if int(entity.candidate_id) in candidate_ids:
                path.unlink()

    def _delete_oldest_source_map_files(self, count: int) -> int:
        return self._delete_oldest_source_map_rows(count)

    def _source_map_rows(self) -> list[tuple[float, int, int, Path, dict[str, Any]]]:
        rows: list[tuple[float, int, int, Path, dict[str, Any]]] = []
        for path in self._scan_entities(self._agent_playbook_source_map_dir()):
            try:
                agent_playbook_id = int(path.stem)
            except ValueError:
                continue
            fallback_created_at = path.stat().st_mtime
            for window in self._read_source_windows_data(path):
                user_playbook_id = int(window.get("user_playbook_id", 0))
                created_at = self._normalize_retention_order_value(
                    window.get("created_at", fallback_created_at)
                )
                rows.append(
                    (created_at, agent_playbook_id, user_playbook_id, path, window)
                )
        return rows

    def _read_source_windows_data(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [
                {
                    "user_playbook_id": int(user_playbook_id),
                    "source_interaction_ids": [],
                    "created_at": path.stat().st_mtime,
                }
                for user_playbook_id in data
            ]
        if not isinstance(data, dict):
            return []
        windows = data.get("source_windows")
        if isinstance(windows, list):
            return [item for item in windows if isinstance(item, dict)]
        user_playbook_ids = data.get("user_playbook_ids")
        if isinstance(user_playbook_ids, list):
            return [
                {
                    "user_playbook_id": int(user_playbook_id),
                    "source_interaction_ids": [],
                    "created_at": path.stat().st_mtime,
                }
                for user_playbook_id in user_playbook_ids
            ]
        return []

    def _delete_oldest_source_map_rows(self, count: int) -> int:
        with self._lock:
            selected = self._source_map_rows()[:]
            selected.sort(key=lambda row: (row[0], row[1], row[2]))
            selected = selected[:count]
            by_path: dict[Path, set[int]] = {}
            for _created_at, _agent_id, user_playbook_id, path, _window in selected:
                by_path.setdefault(path, set()).add(user_playbook_id)
            for path, user_playbook_ids in by_path.items():
                remaining = [
                    window
                    for window in self._read_source_windows_data(path)
                    if int(window.get("user_playbook_id", 0)) not in user_playbook_ids
                ]
                if remaining:
                    self._write_dict(path, {"source_windows": remaining})
                elif path.exists():
                    path.unlink()
        return len(selected)

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

    def _agent_playbook_source_map_dir(self) -> Path:
        return self._org_dir / self._AGENT_PLAYBOOK_SOURCE_MAP

    def _playbook_opt_jobs_dir(self) -> Path:
        return self._org_dir / self._PLAYBOOK_OPT_JOBS

    def _playbook_opt_candidates_dir(self) -> Path:
        return self._org_dir / self._PLAYBOOK_OPT_CANDIDATES

    def _playbook_opt_evaluations_dir(self) -> Path:
        return self._org_dir / self._PLAYBOOK_OPT_EVALUATIONS

    def _playbook_opt_events_dir(self) -> Path:
        return self._org_dir / self._PLAYBOOK_OPT_EVENTS

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
