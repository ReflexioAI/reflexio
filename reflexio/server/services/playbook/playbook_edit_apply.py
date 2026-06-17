"""Shared archive+insert primitive for applying a playbook edit.

Extracted from ReflectionService._replace_playbook so the offline tuner and
the online reflection path share one lifecycle (insert-then-archive, optimistic
concurrency).
"""

from typing import TYPE_CHECKING

from reflexio.models.api_schema.domain.entities import UserPlaybook

if TYPE_CHECKING:
    from reflexio.server.services.storage.storage_base import BaseStorage


def apply_playbook_edit(
    storage: "BaseStorage",
    *,
    incumbent_id: int,
    new_playbook: UserPlaybook,
    source: str,
    expect_current: bool = True,
) -> int:
    """Insert a replacement playbook then archive the incumbent.

    Args:
        storage: A BaseStorage instance providing ``save_user_playbooks``,
            ``get_user_playbook_by_id``, and ``archive_user_playbook_by_id``.
        incumbent_id: ``user_playbook_id`` of the playbook being replaced.
        new_playbook: The replacement playbook (inserted as CURRENT, i.e.
            ``status=None``).
        source: Provenance label stored on the new playbook row.
        expect_current: When ``True`` (default), check that the incumbent is
            still CURRENT before archiving; if it has already been archived by
            a concurrent writer, skip the archive and return ``-1``.  Pass
            ``False`` to always archive regardless (reflection's existing
            behaviour).

    Returns:
        The ``user_playbook_id`` of the newly inserted playbook, or ``-1`` if
        ``expect_current`` is ``True`` and the incumbent was no longer CURRENT.
    """
    new_playbook.source = source
    storage.save_user_playbooks([new_playbook])
    new_id: int = new_playbook.user_playbook_id

    if expect_current:
        incumbent = storage.get_user_playbook_by_id(incumbent_id)
        if incumbent is None or incumbent.status is not None:
            return -1

    user_id = new_playbook.user_id or ""
    archived = storage.archive_user_playbook_by_id(
        user_id=user_id,
        user_playbook_id=incumbent_id,
    )
    return new_id if archived else -1
