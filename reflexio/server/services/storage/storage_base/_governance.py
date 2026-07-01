from __future__ import annotations

from abc import ABC


class GovernanceMixin(ABC):  # noqa: B024 - drained residual kept composed for structural continuity
    """Mixin for backend-neutral governance storage primitives.

    Fully drained: every governance-contract abstract method now lives in a
    dedicated sub-mixin (audit, purge, subject-barrier, erase-execution,
    rebuild-hide). This residual mixin stays composed for structural continuity.
    """
