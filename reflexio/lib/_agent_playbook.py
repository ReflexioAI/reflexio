from __future__ import annotations

import logging
from collections import defaultdict
from typing import Literal, cast

from reflexio.lib._base import (
    STORAGE_NOT_CONFIGURED_MSG,
    ReflexioBase,
    _require_storage,
)
from reflexio.models.api_schema.domain.entities import (
    PlaybookAggregationChangeLog,
    agent_playbook_to_snapshot,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.retriever_schema import (
    GetAgentPlaybooksRequest,
    GetAgentPlaybooksResponse,
    SearchAgentPlaybookRequest,
    SearchAgentPlaybookResponse,
    UpdateAgentPlaybookRequest,
    UpdateAgentPlaybookResponse,
    UpdatePlaybookStatusRequest,
    UpdatePlaybookStatusResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AddAgentPlaybookRequest,
    AddAgentPlaybookResponse,
    AgentPlaybook,
    BulkDeleteResponse,
    DeleteAgentPlaybookRequest,
    DeleteAgentPlaybookResponse,
    DeleteAgentPlaybooksByIdsRequest,
    PlaybookAggregationChangeLogResponse,
)
from reflexio.models.config_schema import SearchOptions
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._playbook import (
    AGGREGATE_REASON_PREFIX,
)
from reflexio.server.tracing import profile_step


class AgentPlaybookMixin(ReflexioBase):
    def get_playbook_aggregation_change_logs(
        self,
        playbook_name: str,
        agent_version: str,
    ) -> PlaybookAggregationChangeLogResponse:
        """Get playbook aggregation change logs, served from the lineage reconstruction.

        The change-log view is rebuilt on demand from ``lineage_event`` rows via
        :func:`reconstruct_playbook_aggregation_change_log`. The legacy
        ``playbook_aggregation_change_logs`` table is no longer read. Results are
        filtered to entries matching ``playbook_name`` and ``agent_version``.
        ``updated_agent_playbooks`` is always ``[]`` (tolerated parity delta — Decision 3).

        Args:
            playbook_name (str): Filter — only logs for this playbook name are returned.
            agent_version (str): Filter — only logs for this agent version are returned.

        Returns:
            PlaybookAggregationChangeLogResponse: Response containing the reconstructed
                change logs filtered by playbook_name and agent_version.
        """
        if not self._is_storage_configured():
            return PlaybookAggregationChangeLogResponse(success=True, change_logs=[])
        # Legacy table no longer read; served by reconstruction filtered by
        # playbook_name + agent_version. updated_agent_playbooks is always []
        # (tolerated parity delta).
        return reconstruct_playbook_aggregation_change_log(
            self._get_storage(),
            playbook_name=playbook_name,
            agent_version=agent_version,
        )

    @_require_storage(DeleteAgentPlaybookResponse)
    def delete_agent_playbook(
        self,
        request: DeleteAgentPlaybookRequest | dict,
    ) -> DeleteAgentPlaybookResponse:
        """Delete an agent playbook by ID.

        Args:
            request (DeleteAgentPlaybookRequest): The delete request containing agent_playbook_id

        Returns:
            DeleteAgentPlaybookResponse: Response containing success status and message
        """
        if isinstance(request, dict):
            request = DeleteAgentPlaybookRequest(**request)
        self._get_storage().delete_agent_playbook(request.agent_playbook_id)
        return DeleteAgentPlaybookResponse(success=True, message="Deleted successfully")

    @_require_storage(BulkDeleteResponse)
    def delete_all_playbooks_bulk(self) -> BulkDeleteResponse:
        """Delete all playbooks (both user and agent).

        Cascading variant — wipes both playbook stores. For per-entity
        semantics use :meth:`delete_all_agent_playbooks_bulk` (agent only)
        or :meth:`UserPlaybookMixin.delete_all_user_playbooks_bulk`
        (user only).

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        self._get_storage().delete_all_agent_playbooks()
        self._get_storage().delete_all_user_playbooks()
        return BulkDeleteResponse(success=True, message="Deleted successfully")

    @_require_storage(BulkDeleteResponse)
    def delete_all_agent_playbooks_bulk(self) -> BulkDeleteResponse:
        """Delete all agent playbooks (only agent playbooks, not user playbooks).

        Unlike :meth:`delete_all_playbooks_bulk` (which cascades to both
        user and agent playbooks), this method scopes the deletion
        strictly to agent playbooks. Use this from CLI or API callers
        that want per-entity semantics.

        Returns:
            BulkDeleteResponse: Response containing success status and message.
        """
        self._get_storage().delete_all_agent_playbooks()
        return BulkDeleteResponse(success=True, message="Deleted successfully")

    @_require_storage(BulkDeleteResponse)
    def delete_agent_playbooks_by_ids_bulk(
        self,
        request: DeleteAgentPlaybooksByIdsRequest | dict,
    ) -> BulkDeleteResponse:
        """Delete agent playbooks by their IDs.

        Args:
            request (DeleteAgentPlaybooksByIdsRequest): The delete request containing agent_playbook_ids

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        if isinstance(request, dict):
            request = DeleteAgentPlaybooksByIdsRequest(**request)
        self._get_storage().delete_agent_playbooks_by_ids(request.agent_playbook_ids)
        return BulkDeleteResponse(
            success=True,
            deleted_count=len(request.agent_playbook_ids),
            message=f"Deleted {len(request.agent_playbook_ids)} item(s)",
        )

    def add_agent_playbook(
        self,
        request: AddAgentPlaybookRequest | dict,
    ) -> AddAgentPlaybookResponse:
        """Add agent playbooks directly to storage.

        Args:
            request (Union[AddAgentPlaybookRequest, dict]): The add request containing agent playbooks

        Returns:
            AddAgentPlaybookResponse: Response containing success status, message, and count of added playbooks
        """
        if not self._is_storage_configured():
            return AddAgentPlaybookResponse(
                success=False, message=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = AddAgentPlaybookRequest(**request)

        try:
            # Normalize playbooks - only keep required fields, reset others to defaults.
            # Top-level structured fields (trigger, rationale) are preserved so CLI
            # callers and the aggregation pipeline don't lose them.
            normalized_playbooks = [
                AgentPlaybook(
                    agent_version=fb.agent_version,
                    playbook_name=fb.playbook_name,
                    content=fb.content,
                    trigger=fb.trigger,
                    rationale=fb.rationale,
                    playbook_status=fb.playbook_status,
                    playbook_metadata=(fb.playbook_metadata or ""),
                )
                for fb in request.agent_playbooks
            ]

            self._get_storage().save_agent_playbooks(normalized_playbooks)
            return AddAgentPlaybookResponse(
                success=True,
                added_count=len(normalized_playbooks),
                message=f"Added {len(normalized_playbooks)} item(s)",
            )
        except Exception as e:
            return AddAgentPlaybookResponse(success=False, message=str(e))

    def get_agent_playbooks(
        self,
        request: GetAgentPlaybooksRequest | dict,
    ) -> GetAgentPlaybooksResponse:
        """Get agent playbooks.

        Args:
            request (Union[GetAgentPlaybooksRequest, dict]): The get request

        Returns:
            GetAgentPlaybooksResponse: Response containing agent playbooks
        """
        if not self._is_storage_configured():
            return GetAgentPlaybooksResponse(
                success=True, agent_playbooks=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = GetAgentPlaybooksRequest(**request)

        try:
            agent_playbooks = self._get_storage().get_agent_playbooks(
                limit=request.limit or 100,
                playbook_name=request.playbook_name,
                agent_version=request.agent_version,
                status_filter=request.status_filter,
                playbook_status_filter=[request.playbook_status_filter]
                if request.playbook_status_filter
                else None,
                tags=request.tags,
            )
            return GetAgentPlaybooksResponse(
                success=True,
                agent_playbooks=agent_playbooks,
                msg=f"Found {len(agent_playbooks)} agent playbook(s)",
            )
        except Exception as e:
            return GetAgentPlaybooksResponse(
                success=False, agent_playbooks=[], msg=str(e)
            )

    def search_agent_playbooks(
        self,
        request: SearchAgentPlaybookRequest | dict,
    ) -> SearchAgentPlaybookResponse:
        """Search agent playbooks with advanced filtering and semantic search.

        Args:
            request (Union[SearchAgentPlaybookRequest, dict]): The search request

        Returns:
            SearchAgentPlaybookResponse: Response containing matching agent playbooks
        """
        if not self._is_storage_configured():
            return SearchAgentPlaybookResponse(
                success=True, agent_playbooks=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = SearchAgentPlaybookRequest(**request)

        try:
            query = (
                self._reformulate_query(
                    request.query, enabled=bool(request.enable_reformulation)
                )
                or request.query
            )
            search_request = request.model_copy(update={"query": query})
            query_embedding = self._maybe_get_query_embedding(
                search_request.query, search_request.search_mode
            )
            options = (
                SearchOptions(query_embedding=query_embedding)
                if query_embedding
                else None
            )
            with profile_step(
                "search.storage",
                entity_type="agent_playbooks",
                search_mode=search_request.search_mode,
                top_k=search_request.top_k,
            ) as span:
                agent_playbooks = self._get_storage().search_agent_playbooks(
                    search_request, options
                )
                span.set_data("result_count", len(agent_playbooks))
            return SearchAgentPlaybookResponse(
                success=True,
                agent_playbooks=agent_playbooks,
                msg=f"Found {len(agent_playbooks)} matching agent playbook(s)",
            )
        except Exception as e:
            return SearchAgentPlaybookResponse(
                success=False, agent_playbooks=[], msg=str(e)
            )

    @_require_storage(UpdatePlaybookStatusResponse, msg_field="msg")
    def update_agent_playbook_status(
        self,
        request: UpdatePlaybookStatusRequest | dict,
    ) -> UpdatePlaybookStatusResponse:
        """Update the status of a specific agent playbook.

        Args:
            request (Union[UpdatePlaybookStatusRequest, dict]): The update request

        Returns:
            UpdatePlaybookStatusResponse: Response containing success status and message
        """
        if isinstance(request, dict):
            request = UpdatePlaybookStatusRequest(**request)
        self._get_storage().update_agent_playbook_status(
            agent_playbook_id=request.agent_playbook_id,
            playbook_status=request.playbook_status,
        )
        return UpdatePlaybookStatusResponse(
            success=True, msg="Playbook status updated successfully"
        )

    @_require_storage(UpdateAgentPlaybookResponse, msg_field="msg")
    def update_agent_playbook(
        self,
        request: UpdateAgentPlaybookRequest | dict,
    ) -> UpdateAgentPlaybookResponse:
        """Update editable fields of an agent playbook.

        Args:
            request (Union[UpdateAgentPlaybookRequest, dict]): The update request

        Returns:
            UpdateAgentPlaybookResponse: Response containing success status and message
        """
        if isinstance(request, dict):
            request = UpdateAgentPlaybookRequest(**request)
        self._get_storage().update_agent_playbook(
            agent_playbook_id=request.agent_playbook_id,
            playbook_name=request.playbook_name,
            content=request.content,
            trigger=request.trigger,
            rationale=request.rationale,
            playbook_status=request.playbook_status,
        )
        return UpdateAgentPlaybookResponse(
            success=True, msg="Agent playbook updated successfully"
        )


# ---------------------------------------------------------------------------
# Standalone read-side reconstruction (Phase B3b Task 2)
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Prefix for aggregate lineage event reasons — single source of truth in
# AGGREGATE_REASON_PREFIX (storage_base/_playbook.py); imported here for the parser.
_PREFIX = AGGREGATE_REASON_PREFIX


def reconstruct_playbook_aggregation_change_log(
    storage: BaseStorage,
    *,
    limit: int = 100,
    playbook_name: str | None = None,
    agent_version: str | None = None,
) -> PlaybookAggregationChangeLogResponse:
    """Rebuild the PlaybookAggregationChangeLog view from lineage events.

    Uses two immutable / stable signals to classify every aggregation run:

    * **added(R)** — entity_ids of ``aggregate`` lineage events with
      ``request_id == R``.  Each ``aggregate`` event records one playbook
      saved in the run.  The ``reason`` field encodes the run mode:
      ``"aggregate:full_archive"`` or ``"aggregate:incremental"``; any other
      value defaults to ``"incremental"``.

    * **removed(R)** — entity_ids of ``status_change`` lineage events with
      ``to_status == "superseded"`` and ``request_id == R``.  This is the
      exact signature emitted by ``supersede_agent_playbooks_by_ids`` and
      ``supersede_agent_playbooks_by_playbook_name`` (the soft-delete paths).
      APPROVED playbooks are skipped by those helpers, so they have no removal
      signal and are correctly absent from reconstruction.removed.

    Groups are formed over the union of request_ids from both signals.
    Request_id ``""`` is skipped — it would merge unrelated runs.
    A row is emitted only when ``added or removed`` is non-empty.

    When a removed playbook's tombstone has been physically purged (GC),
    it is silently omitted from ``removed_agent_playbooks`` rather than crashing.

    ``updated_agent_playbooks = []`` (Decision 3 — tolerated delta; updates
    are folded into added/removed in the B3b reconstruction model).

    An agent_playbook *added* in run R is included in R's
    ``added_agent_playbooks`` even if a later run superseded (tombstoned) it
    (resolved with ``include_tombstones=True``) — so R is not dropped from the
    change log, matching the legacy table and ``reconstruct_profile_change_log``.
    As with the removed side, once a tombstone is physically purged (GC) the
    snapshot resolves to ``None`` and is silently omitted.

    Version-semantics note (E1): for remove-only runs with mixed agent_version
    across the superseded set, ``agent_version`` is that of the first resolved
    removed snapshot.  This is a query-VISIBILITY loss — version-filtered
    queries will omit those removals — not just a label ambiguity.

    Run-scalars (``playbook_name``, ``agent_version``) are read from the
    reconstructed content: ``added[0]`` is preferred, else ``removed[0]``.
    For a remove-only run whose superseded playbooks span multiple versions,
    ``agent_version`` is that of the first resolved removed snapshot — a
    tolerated, documented ambiguity (the aggregate event carries no version
    since remove-only runs emit no aggregate event).

    Args:
        storage (BaseStorage): Storage instance to query.
        limit (int): Maximum number of reconstructed entries to return.
            Defaults to 100.
        playbook_name (str | None): When provided, only logs whose
            ``playbook_name`` matches are returned. Defaults to ``None``
            (no filter).
        agent_version (str | None): When provided, only logs whose
            ``agent_version`` matches are returned. Defaults to ``None``
            (no filter).

    Returns:
        PlaybookAggregationChangeLogResponse: ``success=True`` with
            reconstructed rows ordered most-recent-first (by max event
            ``created_at`` in each request_id group), filtered by
            ``playbook_name``/``agent_version`` when supplied, and capped
            at ``limit``.
    """
    if limit <= 0:
        return PlaybookAggregationChangeLogResponse(success=True, change_logs=[])

    all_events = storage.get_lineage_events(
        entity_type="agent_playbook", org_id=storage.org_id
    )

    added_by_req: dict[str, list[str]] = defaultdict(list)
    removed_by_req: dict[str, list[str]] = defaultdict(list)
    run_mode_by_req: dict[str, Literal["full_archive", "incremental"]] = {}
    sort_key: dict[str, tuple[int, int]] = {}

    for evt in all_events:
        req = evt.request_id
        if not req:
            continue  # skip empty — never merge unrelated runs
        cur = sort_key.get(req, (0, 0))
        if (evt.created_at, evt.event_id) > cur:
            sort_key[req] = (evt.created_at, evt.event_id)
        if evt.op == "aggregate":
            added_by_req[req].append(evt.entity_id)
            if req not in run_mode_by_req:
                reason = evt.reason or ""
                if reason.startswith(_PREFIX):
                    suffix = reason[len(_PREFIX) :]
                    run_mode_by_req[req] = cast(
                        Literal["full_archive", "incremental"],
                        suffix
                        if suffix in ("full_archive", "incremental")
                        else "incremental",
                    )
                else:
                    run_mode_by_req[req] = "incremental"
        elif evt.op == "status_change" and evt.to_status == Status.SUPERSEDED.value:
            removed_by_req[req].append(evt.entity_id)

    candidate_reqs = set(added_by_req) | set(removed_by_req)
    sorted_reqs = sorted(
        candidate_reqs,
        key=lambda r: sort_key.get(r, (0, 0)),
        reverse=True,
    )

    logs: list[PlaybookAggregationChangeLog] = []
    # PERFORMANCE NOTE (M3): as of Track B retirement this serves the live
    # ``/api/playbook_aggregation_change_logs`` endpoint (the legacy table is no
    # longer read), so it IS on the request path. Two bounds keep it from scaling
    # with full org history: (1) the lineage-event read is paginated in the
    # Supabase backend (``get_lineage_events`` loops ``.range()`` windows), so
    # PostgREST ``max_rows`` can no longer silently truncate to the oldest rows;
    # (2) name/version filtering AND the ``limit`` are applied INSIDE this loop and
    # we ``break`` once ``limit`` matches are collected, so the per-entity
    # ``get_agent_playbook_by_id`` (N+1) resolution stops at the requested page
    # instead of resolving every run in the org. ``sorted_reqs`` is most-recent
    # first, so breaking after ``limit`` matches yields the ``limit`` most-recent
    # matches. Remaining follow-up (tracked, NOT done here): a batch ``*_by_ids``
    # fetch shared with ``reconstruct_profile_change_log``.
    for req in sorted_reqs:
        added = []
        for eid in added_by_req[req]:
            try:
                # include_tombstones: a playbook added in this run but later
                # superseded must still appear in this run's added side (so the
                # run is not dropped) — mirrors the removed-side resolve below.
                pb = storage.get_agent_playbook_by_id(int(eid), include_tombstones=True)
            except (ValueError, TypeError):
                logger.warning(
                    "reconstruct: malformed entity_id %r in added_by_req, skipping", eid
                )
                continue
            if pb is not None:
                added.append(agent_playbook_to_snapshot(pb))

        removed = []
        for eid in removed_by_req[req]:
            try:
                pb = storage.get_agent_playbook_by_id(int(eid), include_tombstones=True)
            except (ValueError, TypeError):
                logger.warning(
                    "reconstruct: malformed entity_id %r in removed_by_req, skipping",
                    eid,
                )
                continue
            if pb is not None:
                removed.append(agent_playbook_to_snapshot(pb))

        if not added and not removed:
            continue

        # agent_version is the run's REPRESENTATIVE snapshot version (added[0], else
        # removed[0]). For a remove-only run whose superseded playbooks span multiple
        # versions, the label is that of the first resolved removed snapshot — a
        # tolerated, documented ambiguity (the aggregate event carries no version).
        first = added[0] if added else removed[0]
        # Filter INSIDE the loop so the ``limit`` is applied to the post-filter
        # set (not a pre-filter slice) AND we can stop once the page is full.
        if playbook_name is not None and first.playbook_name != playbook_name:
            continue
        if agent_version is not None and first.agent_version != agent_version:
            continue

        ts, _ = sort_key.get(req, (0, 0))
        logs.append(
            PlaybookAggregationChangeLog(
                playbook_name=first.playbook_name,
                agent_version=first.agent_version,
                run_mode=run_mode_by_req.get(req, "incremental"),
                added_agent_playbooks=added,
                removed_agent_playbooks=removed,
                updated_agent_playbooks=[],
                created_at=ts,
            )
        )
        if len(logs) >= limit:  # limit > 0 guaranteed above
            break

    return PlaybookAggregationChangeLogResponse(success=True, change_logs=logs)
