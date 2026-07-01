"""Residual SQLite profile module.

All public methods have been extracted into the ``profiles`` package
(``ProfileStoreMixin``, ``InteractionStoreMixin``, ``ProfileSearchMixin``). Only
the shared module-level ``_build_tags_sql`` helper (used by both ProfileStore and
ProfileSearch) and the now-drained ``ProfileMixin`` class remain; Task 6 removes
the class and its composition.
"""

from typing import Any


def _build_tags_sql(alias: str, tags: list[str] | None) -> tuple[str, list[Any]]:
    if not tags:
        return "", []
    placeholders = ",".join("?" for _ in tags)
    return (
        f"EXISTS (SELECT 1 FROM json_each({alias}.tags) WHERE value IN ({placeholders}))",
        list(tags),
    )


class ProfileMixin:
    """Drained residual mixin — all methods moved to the ``profiles`` package."""
