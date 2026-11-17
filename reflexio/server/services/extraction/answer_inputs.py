"""Run-local answer evidence; digests are data, never execution authority."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from threading import Lock
from typing import TYPE_CHECKING, Any

from reflexio.server.llm.input_budget import ExtractionInputIntegrityError
from reflexio.server.work_scope import current_project_id

if TYPE_CHECKING:
    from reflexio.server.services.storage.storage_base import (
        BaseStorage,
        PendingToolCallRecord,
    )


def require_publication_answer_inputs(
    output: Any,
) -> tuple[dict[str, str], dict[str, str]]:
    """Validate retained input structure; hashes are evidence, never ownership."""
    from reflexio.server.services.extraction.publication_receipts import (
        require_output_id,
    )

    publication = output.get("publication") if isinstance(output, dict) else None
    dependencies = (
        publication.get("dependency_inputs") if isinstance(publication, dict) else None
    )
    answers = (
        publication.get("answer_inputs") if isinstance(publication, dict) else None
    )
    if (
        not isinstance(output, dict)
        or type(output.get("_reflexio_envelope_version")) is not int
        or output.get("_reflexio_envelope_version") != 1
        or not isinstance(publication, dict)
        or not isinstance(dependencies, dict)
        or not isinstance(answers, dict)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for captured in (dependencies, answers)
            for key, value in captured.items()
        )
        or any(answers.get(key) != digest for key, digest in dependencies.items())
    ):
        raise ExtractionInputIntegrityError(
            "Committed output publication inputs unavailable"
        )
    try:
        require_output_id(publication.get("id"))
    except ValueError as exc:
        raise ExtractionInputIntegrityError(
            "Committed output publication identity invalid"
        ) from exc
    return dependencies, answers


class _AnswerInputs:
    def __init__(self, storage: BaseStorage, run_id: str) -> None:
        self.storage = storage
        self.org_id = storage.org_id
        self.project_id = current_project_id()
        self.run_id = run_id
        self.values: dict[str, str] = {}
        self.active = True
        self.lock = Lock()

    def require_target(self, storage: BaseStorage, run_id: str | None = None) -> None:
        if (
            storage is not self.storage
            or storage.org_id != self.org_id
            or current_project_id() != self.project_id
            or (run_id is not None and self.run_id != run_id)
        ):
            raise ExtractionInputIntegrityError("Answer capture target changed")

    def snapshot(self) -> dict[str, str]:
        with self.lock:
            if not self.active:
                raise ExtractionInputIntegrityError("Answer input capture ended")
            return dict(self.values)


_capture: ContextVar[_AnswerInputs | None] = ContextVar(
    "extraction_answer_inputs", default=None
)


@contextmanager
def answer_input_scope(storage: BaseStorage, run_id: str) -> Iterator[None]:
    existing = _capture.get()
    if (
        existing is not None
        and existing.storage is storage
        and existing.run_id == run_id
    ):
        existing.require_target(storage, run_id)
        existing.snapshot()  # A copied context cannot reopen an ended capture.
        yield
        return
    current = _AnswerInputs(storage, run_id)
    token = _capture.set(current)
    try:
        yield
    finally:
        with current.lock:
            current.active = False
        _capture.reset(token)


def has_answer_input_capture(storage: BaseStorage) -> bool:
    current = _capture.get()
    if current is None:
        return False
    current.require_target(storage)
    current.snapshot()
    return True


def record_answer_input(record: PendingToolCallRecord) -> None:
    current = _capture.get()
    if current is None:
        return
    from datetime import UTC, datetime

    from reflexio.server.services.extraction.overflow_answers import (
        answer_witness,
        require_current_answer,
    )
    from reflexio.server.services.storage.storage_base import build_scope_hash

    current.require_target(current.storage)
    witness = answer_witness(asdict(record))
    require_current_answer(witness, current.org_id, datetime.now(UTC))
    digest = build_scope_hash(witness)
    with current.lock:
        if not current.active:
            raise ExtractionInputIntegrityError("Answer input capture ended")
        if record.id in current.values and current.values[record.id] != digest:
            raise ExtractionInputIntegrityError(
                "Answer changed within extraction input"
            )
        current.values[record.id] = digest


def snapshot_answer_inputs(storage: BaseStorage, run_id: str) -> dict[str, str]:
    current = _capture.get()
    if current is None:
        raise ExtractionInputIntegrityError("Answer input capture is unavailable")
    current.require_target(storage, run_id)
    return current.snapshot()
