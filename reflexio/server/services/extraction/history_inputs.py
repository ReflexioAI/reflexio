"""Actual ordinary-consolidation inputs, separate from execution authority."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.work_scope import current_project_id

if TYPE_CHECKING:
    from reflexio.models.api_schema.service_schemas import UserPlaybook, UserProfile
    from reflexio.server.services.storage.storage_base import (
        AgentRunRecord,
        BaseStorage,
    )


@dataclass
class HistoricalInputs:
    profiles: dict[str, str] = field(default_factory=dict)
    playbooks: dict[str, str] = field(default_factory=dict)


def require_historical_inputs(inputs: HistoricalInputs | None) -> HistoricalInputs:
    """Detach declared original inputs; absence is never ordinary empty history."""
    if not isinstance(inputs, HistoricalInputs) or any(
        not isinstance(values, dict) for values in (inputs.profiles, inputs.playbooks)
    ):
        raise InputBudgetConfigurationError("Original historical inputs unavailable")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        for values in (inputs.profiles, inputs.playbooks)
        for key, digest in values.items()
    ):
        raise InputBudgetConfigurationError("Original historical inputs invalid")
    try:
        if any(str(int(key)) != key for key in inputs.playbooks):
            raise ValueError("noncanonical identity")
    except ValueError as exc:
        raise InputBudgetConfigurationError(
            "Historical playbook identity invalid"
        ) from exc
    return deepcopy(inputs)


def history_from_output(output: Any) -> HistoricalInputs:
    publication = output.get("publication") if isinstance(output, dict) else None
    inputs = (
        publication.get("history_inputs") if isinstance(publication, dict) else None
    )
    if not isinstance(inputs, dict) or set(inputs) != {"profiles", "playbooks"}:
        raise InputBudgetConfigurationError("Original historical inputs unavailable")
    return require_historical_inputs(
        HistoricalInputs(inputs["profiles"], inputs["playbooks"])
    )


def merge_historical_inputs(
    original: HistoricalInputs, prepared: HistoricalInputs
) -> HistoricalInputs:
    merged = require_historical_inputs(original)
    other = require_historical_inputs(prepared)
    for previous, current in (
        (merged.profiles, other.profiles),
        (merged.playbooks, other.playbooks),
    ):
        if any(
            key in previous and previous[key] != digest
            for key, digest in current.items()
        ):
            raise InputBudgetConfigurationError(
                "Historical learning changed between model stages"
            )
        previous.update(current)
    return merged


class _Capture:
    def __init__(self, storage: BaseStorage, run: AgentRunRecord) -> None:
        self.storage = storage
        self.org_id = storage.org_id
        self.project_id = current_project_id()
        self.user_id = run.binding.user_id
        self.inputs = HistoricalInputs()
        self.active = True
        self.lock = Lock()

    def record(
        self,
        storage: BaseStorage | None,
        user_id: str | None,
        kind: str,
        identity: str,
        digest: str,
    ) -> None:
        with self.lock:
            if (
                not self.active
                or storage is None
                or storage is not self.storage
                or storage.org_id != self.org_id
                or current_project_id() != self.project_id
                or not self.user_id
                or user_id != self.user_id
                or not identity
            ):
                raise InputBudgetConfigurationError(
                    "Historical input capture scope changed"
                )
            values = (
                self.inputs.profiles if kind == "profile" else self.inputs.playbooks
            )
            if identity in values and values[identity] != digest:
                raise InputBudgetConfigurationError(
                    "Historical learning changed during preparation"
                )
            values[identity] = digest

    def finish(self) -> HistoricalInputs:
        with self.lock:
            if not self.active:
                raise InputBudgetConfigurationError("Historical input capture ended")
            self.active = False
            return HistoricalInputs(
                dict(self.inputs.profiles), dict(self.inputs.playbooks)
            )


_current: ContextVar[_Capture | None] = ContextVar(
    "ordinary_publication_history", default=None
)


@contextmanager
def historical_input_scope(
    storage: BaseStorage, run: AgentRunRecord
) -> Iterator[_Capture]:
    # Each preparation owns one capture. A nested/copy cannot reset the evidence.
    if _current.get() is not None:
        raise InputBudgetConfigurationError(
            "Historical preparation capture already active"
        )
    if (
        run.binding.org_id != storage.org_id
        or not run.binding.user_id
        or (run.project_id or "") != (current_project_id() or "")
    ):
        raise InputBudgetConfigurationError("Historical preparation binding changed")
    capture = _Capture(storage, run)
    token = _current.set(capture)
    try:
        yield capture
    finally:
        with capture.lock:
            capture.active = False
        _current.reset(token)


def record_historical_profiles(
    storage: BaseStorage | None, profiles: list[UserProfile]
) -> None:
    capture = _current.get()
    if capture is None:
        return
    from reflexio.server.services.extraction.profile_package_state import (
        profile_witness,
    )

    for profile in profiles:
        if not profile.profile_id:
            raise InputBudgetConfigurationError(
                "Historical profile identity unavailable"
            )
        capture.record(
            storage,
            profile.user_id,
            "profile",
            profile.profile_id,
            profile_witness(profile),
        )


def record_historical_playbooks(
    storage: BaseStorage | None, playbooks: list[UserPlaybook]
) -> None:
    capture = _current.get()
    if capture is None:
        return
    from reflexio.server.services.playbook.components.overflow_evidence import (
        historical_playbook_digest,
    )

    for playbook in playbooks:
        if playbook.user_playbook_id is None:
            raise InputBudgetConfigurationError(
                "Historical playbook identity unavailable"
            )
        capture.record(
            storage,
            playbook.user_id,
            "playbook",
            str(playbook.user_playbook_id),
            historical_playbook_digest(playbook),
        )
