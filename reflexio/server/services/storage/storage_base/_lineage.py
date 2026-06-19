from abc import abstractmethod

from reflexio.models.api_schema.domain.entities import LineageEvent


class LineageEventMixin:
    """Abstract storage interface for the append-only, content-free lineage log."""

    @abstractmethod
    def append_lineage_event(self, event: LineageEvent) -> int:
        """Append an event; idempotent on (entity_id, op, request_id). Return the row id.

        Args:
            event (LineageEvent): The fully-formed event to persist. ``event_id``
                may be 0; the storage layer assigns a real id on insert. On a
                duplicate ``(entity_id, op, request_id)`` the existing row is
                returned unchanged.

        Returns:
            int: The assigned or existing ``event_id``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_lineage_events(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        org_id: str | None = None,
    ) -> list[LineageEvent]:
        """Retrieve lineage events, optionally filtered.

        Args:
            entity_type (str | None): Filter to events for this entity type. If
                None, no entity_type filter is applied.
            entity_id (str | None): Filter to events for this entity id. If None,
                no entity_id filter is applied.
            org_id (str | None): Filter to events for this org. If None, no
                org_id filter is applied.

        Returns:
            list[LineageEvent]: Matching events ordered by ``event_id`` ascending.
        """
        raise NotImplementedError
