"""Disk storage CRUD for ``shadow_comparison_verdicts`` (F1).

Verdicts persist as one JSON file per row under
``{org_dir}/shadow_comparison_verdicts/{verdict_id}.json``. The verdict_id
autoincrement is implemented via a monotonic counter file ``.counter`` in
the same directory; the same ``self._lock`` (RLock) that guards every
other DiskStorage write makes this thread-safe within a single process,
matching the rest of the backend.

Mirrors :class:`reflexio.server.services.storage.sqlite_storage._shadow_verdicts.ShadowVerdictsMixin`
in semantics:
* ``save`` ignores the input ``verdict_id`` and returns a copy with the
  storage-assigned key.
* ``get_shadow_comparison_verdicts`` filters by inclusive ``[from_ts, to_ts]``
  window AND ``judge_prompt_version`` and returns results sorted by
  ``created_at`` ascending.
* ``delete_shadow_comparison_verdicts_by_session`` returns the number of
  files deleted.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from reflexio.models.api_schema.eval_overview_schema import ShadowComparisonVerdict

logger = logging.getLogger(__name__)

_SHADOW_VERDICTS_DIR = "shadow_comparison_verdicts"
_COUNTER_FILENAME = ".counter"


class ShadowVerdictsMixin:
    """Disk implementation of the shadow_comparison_verdicts contract.

    Overrides the abstract ``NotImplementedError`` defaults from
    :class:`reflexio.server.services.storage.storage_base._shadow_verdicts.ShadowVerdictsMixin`
    with a JSON-file implementation.
    """

    # Attributes provided by DiskStorageBase via MRO; declared here so
    # pyright sees the correct types when reading the helpers below.
    _org_dir: Path
    _lock: threading.RLock

    def _verdicts_dir(self) -> Path:
        """Return the verdicts subdirectory, creating it on demand."""
        path = self._org_dir / _SHADOW_VERDICTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _next_verdict_id(self) -> int:
        """Allocate the next autoincrement verdict_id via a counter file.

        Returns:
            int: The newly-assigned 1-based id.
        """
        counter_path = self._verdicts_dir() / _COUNTER_FILENAME
        try:
            current = int(counter_path.read_text())
        except (FileNotFoundError, ValueError):
            current = 0
        new = current + 1
        counter_path.write_text(str(new))
        return new

    def save_shadow_comparison_verdict(
        self, verdict: ShadowComparisonVerdict
    ) -> ShadowComparisonVerdict:
        """
        Persist a verdict and return the row with the assigned ``verdict_id``.

        The input ``verdict_id`` is ignored; storage assigns the next
        autoincrement key — matching the SQLite contract.

        Args:
            verdict (ShadowComparisonVerdict): Verdict to persist.

        Returns:
            ShadowComparisonVerdict: The verdict with ``verdict_id``
                populated from the autoincrement counter.
        """
        with self._lock:
            new_id = self._next_verdict_id()
            materialized = verdict.model_copy(update={"verdict_id": new_id})
            path = self._verdicts_dir() / f"{new_id}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(materialized.model_dump_json(indent=2), encoding="utf-8")
            tmp.rename(path)
            return materialized

    def get_shadow_comparison_verdict(
        self, verdict_id: int
    ) -> ShadowComparisonVerdict | None:
        """
        Fetch a single verdict by its autoincrement primary key.

        Args:
            verdict_id (int): The storage-assigned key.

        Returns:
            ShadowComparisonVerdict | None: The verdict if present, else
                ``None``.
        """
        with self._lock:
            path = self._verdicts_dir() / f"{verdict_id}.json"
            if not path.is_file():
                return None
            return ShadowComparisonVerdict.model_validate_json(
                path.read_text(encoding="utf-8")
            )

    def get_shadow_comparison_verdicts(
        self,
        from_ts: int,
        to_ts: int,
        judge_prompt_version: str,
    ) -> list[ShadowComparisonVerdict]:
        """
        Fetch verdicts in ``[from_ts, to_ts]`` for one pinned prompt version.

        Args:
            from_ts (int): Inclusive lower bound on ``created_at``, Unix
                epoch seconds (UTC).
            to_ts (int): Inclusive upper bound on ``created_at``, Unix
                epoch seconds (UTC).
            judge_prompt_version (str): Pinned ``shadow_comparison`` prompt
                version. Filtering by this prevents rubric-mixing in the
                dashboard headline.

        Returns:
            list[ShadowComparisonVerdict]: Matching verdicts in
                chronological order (ascending ``created_at``).
        """
        with self._lock:
            results: list[ShadowComparisonVerdict] = []
            for path in self._iter_verdict_files():
                verdict = self._load_verdict_file(path)
                if verdict is None:
                    continue
                if verdict.judge_prompt_version != judge_prompt_version:
                    continue
                ts = int(verdict.created_at.timestamp())
                if from_ts <= ts <= to_ts:
                    results.append(verdict)
            results.sort(key=lambda v: v.created_at)
            return results

    def delete_shadow_comparison_verdicts_by_session(self, session_id: str) -> int:
        """
        Delete every verdict belonging to one session.

        Args:
            session_id (str): The session whose verdicts should be removed.

        Returns:
            int: Number of files deleted.
        """
        with self._lock:
            count = 0
            for path in self._iter_verdict_files():
                verdict = self._load_verdict_file(path)
                if verdict is None:
                    continue
                if verdict.session_id == session_id:
                    path.unlink()
                    count += 1
            return count

    def _iter_verdict_files(self) -> Iterator[Path]:
        """Yield verdict JSON file paths, skipping the counter and any tmp files."""
        directory = self._verdicts_dir()
        for path in directory.iterdir():
            if path.name.startswith("."):
                continue
            if path.suffix != ".json":
                continue
            yield path

    @staticmethod
    def _load_verdict_file(path: Path) -> ShadowComparisonVerdict | None:
        """Read and parse a verdict JSON file, returning None on corruption.

        Corrupt or partially-written files are logged and skipped so a single
        bad file doesn't take down list/delete operations across the whole
        directory.

        Args:
            path (Path): The verdict JSON file to read.

        Returns:
            ShadowComparisonVerdict | None: The parsed verdict, or ``None``
                if the file is missing or malformed.
        """
        try:
            return ShadowComparisonVerdict.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, ValidationError):
            logger.warning("Skipping unreadable verdict file: %s", path, exc_info=True)
            return None
