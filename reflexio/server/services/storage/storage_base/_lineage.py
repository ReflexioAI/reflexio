from abc import abstractmethod
from typing import Literal

from reflexio.models.api_schema.domain.entities import LineageContext, LineageEvent

EntityType = Literal["user_playbook", "agent_playbook", "profile"]


class LineageEventMixin:
    """Abstract storage interface for the append-only, content-free lineage log."""

    @abstractmethod
    def append_lineage_event(self, event: LineageEvent) -> int:
        """Append an event; idempotent on (org_id, entity_type, entity_id, op, request_id).

        Args:
            event (LineageEvent): The fully-formed event to persist. ``event_id``
                may be 0; the storage layer assigns a real id on insert. On a
                duplicate ``(org_id, entity_type, entity_id, op, request_id)`` the existing row
                is returned unchanged.

        Returns:
            int: The assigned or existing ``event_id``.

        Note:
            This method deliberately does NOT enforce a non-empty ``request_id``.
            System and GC events (e.g. ``hard_delete`` from TTL GC, ``status_change``
            from internal transitions) legitimately use an auto-generated UUID that
            need not be tied to a user-facing request id. Callers that require
            request-scoped lineage (``merge_records``, ``supersede_record``) enforce
            non-empty ``request_id`` themselves.
        """
        raise NotImplementedError

    @abstractmethod
    def get_lineage_events(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        org_id: str | None = None,
        request_id: str | None = None,
    ) -> list[LineageEvent]:
        """Retrieve lineage events, optionally filtered.

        Args:
            entity_type (str | None): Filter to events for this entity type. If
                None, no entity_type filter is applied.
            entity_id (str | None): Filter to events for this entity id. If None,
                no entity_id filter is applied.
            org_id (str | None): Filter to events for this org. If None, no
                org_id filter is applied.
            request_id (str | None): Filter to events for this request id. If
                None, no request_id filter is applied.

        Returns:
            list[LineageEvent]: Matching events ordered by ``event_id`` ascending.

        Note:
            Enterprise/Supabase overrides must also apply the ``request_id`` filter
            to maintain contract parity with the SQLite implementation (B3b T3).
        """
        raise NotImplementedError

    @abstractmethod
    def merge_records(
        self,
        *,
        entity_type: EntityType,
        survivor_id: str,
        source_ids: list[str],
        context: LineageContext,
    ) -> None:
        """Soft-delete each source into the survivor in one atomic transaction.

        Sets ``status=MERGED`` and ``merged_into=survivor_id`` on each source
        whose status is not already a tombstone (MERGED or SUPERSEDED). Appends
        a single ``merge`` lineage event keyed on ``survivor_id``. Idempotent —
        re-running on already-tombstoned sources is a no-op.

        Args:
            entity_type (str): One of ``"user_playbook"``, ``"agent_playbook"``,
                or ``"profile"``.
            survivor_id (str): The id of the record that survives the merge.
            source_ids (list[str]): Ids of records to tombstone as merged.
            context (LineageContext): Caller-supplied intent (actor, reason, etc.).

        Raises:
            ValueError: If ``context.request_id`` is empty or whitespace-only.
        """
        raise NotImplementedError

    @abstractmethod
    def supersede_record(
        self,
        *,
        entity_type: EntityType,
        incumbent_id: str,
        successor_id: str,
        context: LineageContext,
    ) -> bool:
        """Atomically replace the incumbent with the successor if incumbent is CURRENT.

        Sets ``status=SUPERSEDED`` and ``superseded_by=successor_id`` on the
        incumbent **only** when its ``status IS NULL`` (CURRENT). Appends a
        ``revise`` lineage event keyed on ``successor_id`` when the guard
        succeeds. Returns ``False`` without mutating anything when the incumbent
        is not CURRENT (its status is already set).

        Args:
            entity_type (str): One of ``"user_playbook"``, ``"agent_playbook"``,
                or ``"profile"``.
            incumbent_id (str): The id of the record to supersede.
            successor_id (str): The id of the record that replaces the incumbent.
            context (LineageContext): Caller-supplied intent (actor, reason, etc.).

        Returns:
            bool: ``True`` if the incumbent was CURRENT and was superseded;
                ``False`` if the incumbent was not CURRENT and no mutation occurred.

        Raises:
            ValueError: If ``context.request_id`` is empty or whitespace-only.
        """
        raise NotImplementedError

    @abstractmethod
    def gc_expired_tombstones(
        self, *, entity_type: str, older_than_epoch: int, limit: int = 1000
    ) -> int:
        """Hard-delete tombstone rows that are older than the given epoch cutoff.

        Emits one ``hard_delete`` lineage event per deleted row before deleting it,
        all within a single atomic transaction. Rows on legal hold are skipped.

        Args:
            entity_type (str): One of ``"user_playbook"``, ``"agent_playbook"``,
                or ``"profile"``.
            older_than_epoch (int): Unix timestamp. Rows whose age column value
                is strictly less than this cutoff are eligible.
            limit (int): Maximum number of rows to delete in one call. Defaults
                to 1000.

        Returns:
            int: The number of rows physically deleted.

        Raises:
            ValueError: If ``entity_type`` is not a recognized entity type.
        """
        raise NotImplementedError

    def list_org_ids(self) -> list[str]:
        """Return every distinct org_id known to this storage instance.

        Used by :class:`LineageGCScheduler` to enumerate all tenants so GC
        runs for every org, not just the bootstrap org.

        Returns:
            list[str]: Distinct org ids, order unspecified.

        Raises:
            NotImplementedError: If the backend has not yet implemented this
                method (enterprise backends owe this in B2 Task 6).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement list_org_ids"
        )
