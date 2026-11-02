"""Fixed private progress state for one user's renewable work owner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, cast

WorkClass = Literal["learning", "finalization", "continuation"]
ResumeClass = Literal["finalization", "continuation"]


@dataclass(frozen=True)
class ResumeCursor:
    """Immutable-ID sweep boundary; arrivals beyond it wait for the next sweep."""

    after_id: str | None = None
    through_id: str | None = None

    def __post_init__(self) -> None:
        for value in (self.after_id, self.through_id):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError("Invalid private user-work cursor")
        # Ordering belongs to the writer database's selection query, not Python
        # string collation. Both SQLite and native databases persist this shape.
        if self.after_id is not None and self.through_id is None:
            raise ValueError("Private user-work cursor has no sweep boundary")


@dataclass(frozen=True)
class UserWorkPosition:
    """Per-user class fairness and two bounded resumable-work sweep cursors.

    This record contains identifiers only. It is not a configuration map and must
    be persisted under the actual user lease before execution can release it.
    """

    last_work_class: WorkClass = "learning"
    finalization: ResumeCursor = field(default_factory=ResumeCursor)
    continuation: ResumeCursor = field(default_factory=ResumeCursor)

    @property
    def next_work_class(self) -> WorkClass:
        if self.last_work_class == "learning":
            return "finalization"
        if self.last_work_class == "finalization":
            return "continuation"
        return "learning"

    def selected(
        self, work_class: WorkClass, *, cursor: ResumeCursor | None = None
    ) -> UserWorkPosition:
        """Consume an inspected opportunity, including an empty/refused result."""
        if work_class != self.next_work_class:
            raise ValueError("Private user-work class selected out of order")
        if work_class == "learning":
            if cursor is not None:
                raise ValueError("Learning work has no resumable-run cursor")
            return replace(self, last_work_class=work_class)
        if cursor is None:
            raise ValueError("Resumable selection must preserve its sweep cursor")
        if work_class == "finalization":
            return replace(self, last_work_class=work_class, finalization=cursor)
        return replace(self, last_work_class=work_class, continuation=cursor)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, value: Any) -> UserWorkPosition:
        """Only a never-initialized row defaults; malformed state fails closed."""
        if value is None:
            return cls()
        raw = json.loads(value) if isinstance(value, str) else value
        if (
            not isinstance(raw, dict)
            or set(raw) != {"last_work_class", "finalization", "continuation"}
            or not isinstance(raw["last_work_class"], str)
            or raw["last_work_class"]
            not in {"learning", "finalization", "continuation"}
        ):
            raise ValueError("Invalid private user-work position")
        cursors = {}
        for name in ("finalization", "continuation"):
            item = raw[name]
            if not isinstance(item, dict) or set(item) != {"after_id", "through_id"}:
                raise ValueError("Invalid private user-work cursor")
            cursors[name] = ResumeCursor(**item)
        return cls(
            cast(WorkClass, raw["last_work_class"]),
            cursors["finalization"],
            cursors["continuation"],
        )
