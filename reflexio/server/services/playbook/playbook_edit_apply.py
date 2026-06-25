"""Shared atomic supersede primitive for applying a playbook edit.

Extracted from ReflectionService._replace_playbook so online and background
playbook repair paths share one lifecycle (insert-then-supersede, no orphan).
"""

from typing import TYPE_CHECKING

from reflexio.models.api_schema.domain.entities import LineageContext, UserPlaybook

if TYPE_CHECKING:
    from reflexio.server.services.storage.storage_base import BaseStorage


def apply_playbook_edit(
    storage: "BaseStorage",
    *,
    incumbent_id: int,
    new_playbook: UserPlaybook,
    source: str,
    request_id: str,
) -> int:
    """Insert a replacement playbook then atomically supersede the incumbent.

    Uses ``storage.supersede_record`` (atomic conditional CAS) so a lost race
    never leaves an orphan CURRENT row:

    - Insert the new playbook as CURRENT.
    - Call ``supersede_record(incumbent_id → new_id)``, which only succeeds when
      the incumbent is still CURRENT (``status IS NULL``).
    - If ``supersede_record`` returns ``False`` (incumbent already gone), delete
      the just-inserted successor and return ``-1``.

    Args:
        storage: A BaseStorage instance providing ``save_user_playbooks``,
            ``supersede_record``, and ``delete_user_playbooks_by_ids``.
        incumbent_id: ``user_playbook_id`` of the playbook being replaced.
        new_playbook: The replacement playbook (inserted as CURRENT, i.e.
            ``status=None``).
        source: Provenance label stored on the new playbook row and in the
            lineage event actor field.
        request_id: Operation-run correlation id for the lineage event. Must be
            non-empty; use the reflection run id (``ReflectionServiceRequest.request_id``)
            or another operation-scoped id. Raises ``ValueError`` immediately
            (before any storage write) when empty, preventing orphaned successor rows.

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
    storage.save_user_playbooks([new_playbook])
    new_id: int = new_playbook.user_playbook_id

    ctx = LineageContext(op_kind="revise", actor=source, request_id=request_id)
    superseded = storage.supersede_record(
        entity_type="user_playbook",
        incumbent_id=str(incumbent_id),
        successor_id=str(new_id),
        context=ctx,
    )
    if not superseded:
        # lost the race: delete the just-inserted successor so no orphan CURRENT row
        # remains. It was never live, so this is a rollback — not an audited erasure.
        storage.delete_user_playbooks_by_ids([new_id], emit_hard_delete=False)
        return -1
    return new_id
