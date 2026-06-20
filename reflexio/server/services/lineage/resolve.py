from __future__ import annotations

from typing import Any, Literal

from reflexio.models.api_schema.domain.entities import RecordRef
from reflexio.server.tracing import capture_anomaly

EntityType = Literal["user_playbook", "agent_playbook", "profile"]

_GETTER = {
    "user_playbook": lambda s, i: s.get_user_playbook_by_id(
        int(i), include_tombstones=True
    ),
    "agent_playbook": lambda s, i: s.get_agent_playbook_by_id(
        int(i), include_tombstones=True
    ),
    "profile": lambda s, i: s.get_profile_by_id(str(i), include_tombstones=True),
}
_MAX_HOPS = 8  # Phase A: chains are 1-hop; cap guards malformed pointers


def resolve_current(
    storage: Any, entity_type: EntityType, record_id: Any
) -> RecordRef | None:
    """Follow merged_into/superseded_by pointers to the live survivor.

    Args:
        storage: Any storage backend exposing get_*_by_id with include_tombstones.
        entity_type: One of "user_playbook", "agent_playbook", "profile".
        record_id: The primary key of the record to resolve (int for playbooks, str for profiles).

    Returns:
        RecordRef pointing to the live survivor (is_purged=True if its body is blank),
        or None if the record doesn't exist, there's a cycle, or the chain exceeds _MAX_HOPS.

    Raises:
        ValueError: If ``entity_type`` is not a recognized entity type.
    """
    get = _GETTER.get(entity_type)
    if get is None:
        raise ValueError(f"unknown entity_type: {entity_type!r}")
    visited: set[str] = set()
    cur = get(storage, record_id)
    if cur is None:
        return None
    while True:
        cur_id = str(_pk(cur, entity_type))
        if cur_id in visited or len(visited) >= _MAX_HOPS:
            # A cycle or an over-long chain means a malformed pointer graph;
            # we return None (callers treat as unresolvable) but surface it to
            # Sentry so the otherwise-silent breakage is observable.
            reason = "cycle" if cur_id in visited else "max_hops_exceeded"
            capture_anomaly(
                f"lineage.resolve_current.{reason}",
                entity_type=entity_type,
                entity_id=str(record_id),
                hops=len(visited),
            )
            return None  # cycle or runaway chain
        visited.add(cur_id)
        nxt = cur.merged_into if cur.merged_into is not None else cur.superseded_by
        if nxt is None:
            return RecordRef(id=cur_id, is_purged=(not cur.content))
        nxt_row = get(storage, nxt)
        if nxt_row is None:
            return RecordRef(
                id=cur_id, is_purged=(not cur.content)
            )  # dangling pointer — stop here
        cur = nxt_row


def _pk(row: Any, entity_type: EntityType) -> Any:
    """Extract the primary key from a row based on its entity type.

    Args:
        row: The entity row object.
        entity_type: One of "user_playbook", "agent_playbook", "profile".

    Returns:
        The primary key value.

    Raises:
        ValueError: If ``entity_type`` is not a recognized entity type.
    """
    pk = {
        "user_playbook": "user_playbook_id",
        "agent_playbook": "agent_playbook_id",
        "profile": "profile_id",
    }.get(entity_type)
    if pk is None:
        raise ValueError(f"unknown entity_type: {entity_type!r}")
    return getattr(row, pk)
