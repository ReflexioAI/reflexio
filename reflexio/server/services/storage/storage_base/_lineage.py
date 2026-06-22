from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

from reflexio.models.api_schema.domain.entities import LineageContext, LineageEvent

EntityType = Literal["user_playbook", "agent_playbook", "profile"]


@dataclass
class LegalHold:
    """A legal hold record that protects entities from hard deletion.

    A hold is *active* while ``released_at`` is None. Releasing a hold sets
    ``released_at`` (and ``released_by``) but never deletes the row — the hold
    history is itself a compliance artifact and must survive.

    Attributes:
        id (int): Auto-assigned hold id.
        org_id (str): The organisation that owns the held entities.
        scope (str): One of ``'org'``, ``'user'``, or ``'entity'``.
        entity_type (str | None): For ``'entity'`` scope, the held entity's type.
        entity_id (str | None): For ``'entity'`` scope, the held entity's id.
        user_id (str | None): For ``'user'`` scope, the held user's id.
        matter_id (str): Caller-supplied identifier grouping related holds.
        legal_basis (str): One of ``'litigation_hold'``, ``'regulatory_order'``,
            or ``'legal_obligation'``.
        reason (str): Free-text justification for the hold.
        placed_by (str): Actor who placed the hold.
        placed_at (int): Unix epoch when the hold was placed.
        released_at (int | None): Unix epoch when released; None while active.
        released_by (str | None): Actor who released the hold; None while active.
    """

    id: int
    org_id: str
    scope: str  # 'org' | 'user' | 'entity'
    entity_type: str | None
    entity_id: str | None
    user_id: str | None
    matter_id: str
    legal_basis: str  # 'litigation_hold' | 'regulatory_order' | 'legal_obligation'
    reason: str
    placed_by: str
    placed_at: int  # epoch
    released_at: int | None  # None = active
    released_by: str | None


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

    @abstractmethod
    def place_hold(
        self,
        *,
        org_id: str,
        scope: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        user_id: str | None = None,
        matter_id: str,
        legal_basis: str,
        reason: str,
        placed_by: str,
        placed_at: int | None = None,  # defaults to now
    ) -> int:
        """Place a legal hold; return the new hold id.

        Args:
            org_id (str): The organisation that owns the held entities.
            scope (str): One of ``'org'``, ``'user'``, or ``'entity'``.
            entity_type (str | None): For ``'entity'`` scope, the held entity's type.
            entity_id (str | None): For ``'entity'`` scope, the held entity's id.
            user_id (str | None): For ``'user'`` scope, the held user's id.
            matter_id (str): Caller-supplied identifier grouping related holds.
            legal_basis (str): One of ``'litigation_hold'``, ``'regulatory_order'``,
                or ``'legal_obligation'``.
            reason (str): Free-text justification.
            placed_by (str): Actor placing the hold.
            placed_at (int | None): Unix epoch; defaults to now when None.

        Returns:
            int: The new hold's id.
        """
        raise NotImplementedError

    @abstractmethod
    def release_hold(
        self,
        *,
        org_id: str,
        hold_id: int | None = None,
        matter_id: str | None = None,
        scope: str | None = None,
        user_id: str | None = None,
        released_by: str,
        released_at: int | None = None,  # defaults to now
    ) -> int:
        """Release holds by id OR by matter+scope+optional user; return count released.

        Exactly one of ``hold_id`` or ``matter_id`` must be supplied. When
        releasing by ``matter_id``, the optional ``scope`` and ``user_id`` further
        narrow which active holds are released. Already-released holds are skipped.

        Args:
            org_id (str): The organisation that owns the holds.
            hold_id (int | None): Release this specific hold id.
            matter_id (str | None): Release active holds with this matter id.
            scope (str | None): Narrow matter-release to this scope.
            user_id (str | None): Narrow matter-release to this user.
            released_by (str): Actor releasing the holds.
            released_at (int | None): Unix epoch; defaults to now when None.

        Returns:
            int: The number of holds released.

        Raises:
            ValueError: If neither ``hold_id`` nor ``matter_id`` is supplied.
        """
        raise NotImplementedError

    @abstractmethod
    def get_holds(
        self,
        org_id: str,
        *,
        active_only: bool = True,
    ) -> list[LegalHold]:
        """Return holds for ``org_id``.

        Args:
            org_id (str): The organisation whose holds to return.
            active_only (bool): When True (default), only return rows where
                ``released_at IS NULL``. When False, include released holds too.

        Returns:
            list[LegalHold]: Matching holds.
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
