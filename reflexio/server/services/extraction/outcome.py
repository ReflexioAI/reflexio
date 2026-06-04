"""Shared extraction outcome types for resumable extraction paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from reflexio.server.llm.token_accounting import RunTokenTotals


@dataclass(frozen=True)
class ExtractionOutcome[T]:
    """Result wrapper returned by extractors that need explicit empty results."""

    status: Literal["completed", "empty"]
    items: list[T] = field(default_factory=list)
    run_id: str | None = None
    token_totals: RunTokenTotals | None = None

    @classmethod
    def completed(
        cls,
        items: list[T],
        *,
        run_id: str | None = None,
        token_totals: RunTokenTotals | None = None,
    ) -> ExtractionOutcome[T]:
        return cls(status="completed", items=items, run_id=run_id, token_totals=token_totals)

    @classmethod
    def empty(cls, *, run_id: str | None = None) -> ExtractionOutcome[T]:
        return cls(status="empty", items=[], run_id=run_id)
