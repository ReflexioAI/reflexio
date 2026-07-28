"""Shared atomic supersede primitive for applying a playbook edit.

Online and background playbook repair paths share one lifecycle
(insert-then-supersede, no orphan).
"""

from typing import TYPE_CHECKING

from reflexio.models.api_schema.domain.entities import LineageContext, UserPlaybook

if TYPE_CHECKING:
    from reflexio.server.services.storage.storage_base import BaseStorage


class _LostSupersedeRaceError(Exception):
    """Internal signal used to roll back a provisional successor."""


def apply_playbook_edit(
    storage: "BaseStorage",
    *,
    incumbent_id: int,
    new_playbook: UserPlaybook,
    source: str,
    request_id: str,
    skip_embedding: bool = False,
    revise_context: LineageContext | None = None,
) -> int:
    """Insert a replacement playbook then atomically supersede the incumbent.

    Uses ``storage.supersede_record`` (atomic conditional CAS) so a lost race
    never leaves an orphan CURRENT row:

    - Insert the new playbook as CURRENT.
    - Call ``supersede_record(incumbent_id → new_id)``, which only succeeds when
      the incumbent is still CURRENT (``status IS NULL``).
    - If ``supersede_record`` returns ``False`` (incumbent already gone), roll
      back the transaction and return ``-1``.

    Args:
        storage: A BaseStorage instance providing ``save_user_playbooks``,
            ``supersede_record``, and ``commit_scope``.
        incumbent_id: ``user_playbook_id`` of the playbook being replaced.
        new_playbook: The replacement playbook (inserted as CURRENT, i.e.
            ``status=None``).
        source: Provenance label stored on the new playbook row and in the
            lineage event actor field.
        request_id: Operation-run correlation id for the lineage event. Must be
            non-empty; use an operation-scoped id. Raises ``ValueError``
            immediately (before any storage write) when empty, preventing
            orphaned successor rows.
        skip_embedding: Forwarded to ``save_user_playbooks``. Defaults to
            ``False`` (precompute the embedding before opening the transaction).

    Returns:
        The ``user_playbook_id`` of the newly inserted playbook, or ``-1`` if
        the incumbent was not CURRENT (no mutation; no orphan left behind).

    Raises:
        ValueError: If ``request_id`` is empty or None.
    """
    if not request_id:
        raise ValueError(
            "apply_playbook_edit: request_id must be non-empty (operation-run correlation id)"
        )
    new_playbook.source = source
    if not skip_embedding:
        storage.precompute_user_playbook_embeddings([new_playbook])

    ctx = revise_context or LineageContext(
        op_kind="revise", actor=source, request_id=request_id
    )
    try:
        with storage.commit_scope():
            storage.save_user_playbooks([new_playbook], skip_embedding=True)
            new_id = new_playbook.user_playbook_id
            if not storage.supersede_record(
                entity_type="user_playbook",
                incumbent_id=str(incumbent_id),
                successor_id=str(new_id),
                context=ctx,
            ):
                raise _LostSupersedeRaceError
    except _LostSupersedeRaceError:
        new_playbook.user_playbook_id = 0
        return -1
    return new_id
