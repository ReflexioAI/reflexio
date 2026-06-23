import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

from datetime import UTC, datetime

from reflexio.lib._base import (
    STORAGE_NOT_CONFIGURED_MSG,
    ReflexioBase,
    _require_storage,
)
from reflexio.lib._lineage_parity import ParityReadStorage
from reflexio.models.api_schema.domain.entities import ProfileChangeLog
from reflexio.models.api_schema.retriever_schema import (
    GetProfileStatisticsResponse,
    GetUserProfilesRequest,
    GetUserProfilesResponse,
    RerankUserProfilesRequest,
    RerankUserProfilesResponse,
    SearchUserProfileRequest,
    SearchUserProfileResponse,
    StorageStatsRequest,
    StorageStatsResponse,
    UpdateUserProfileRequest,
    UpdateUserProfileResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AddUserProfileRequest,
    AddUserProfileResponse,
    BulkDeleteResponse,
    DeleteProfilesByIdsRequest,
    DeleteUserProfileRequest,
    DeleteUserProfileResponse,
    DowngradeProfilesRequest,
    DowngradeProfilesResponse,
    ProfileChangeLogResponse,
    Status,
    UpgradeProfilesRequest,
    UpgradeProfilesResponse,
)
from reflexio.server.services.profile.profile_generation_service import (
    ProfileGenerationService,
)
from reflexio.server.tracing import profile_step


class ProfilesMixin(ReflexioBase):
    def search_user_profiles(
        self,
        request: SearchUserProfileRequest | dict,
        status_filter: list[Status | None] | None = None,
    ) -> SearchUserProfileResponse:
        """Search for user profiles.

        Args:
            request (SearchUserProfileRequest): The search request
            status_filter (Optional[list[Optional[Status]]]): Filter profiles by status. Defaults to [None] for current profiles only.

        Returns:
            SearchUserProfileResponse: Response containing matching profiles
        """
        if not self._is_storage_configured():
            return SearchUserProfileResponse(
                success=True, user_profiles=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = SearchUserProfileRequest(**request)
        if status_filter is None:
            status_filter = [None]  # Default to current profiles
        rewritten = self._reformulate_query(
            request.query, enabled=bool(request.enable_reformulation)
        )
        if rewritten:
            request = request.model_copy(update={"query": rewritten})
        query_embedding = self._maybe_get_query_embedding(
            request.query, request.search_mode
        )
        logger.info(
            "search_user_profiles: query=%r, search_mode=%s, embedding_generated=%s",
            request.query,
            request.search_mode,
            query_embedding is not None,
        )
        with profile_step(
            "search.storage",
            entity_type="profiles",
            search_mode=request.search_mode,
            top_k=request.top_k,
        ) as span:
            profiles = self._get_storage().search_user_profile(
                request, status_filter=status_filter, query_embedding=query_embedding
            )
            span.set_data("result_count", len(profiles))
        return SearchUserProfileResponse(
            success=True,
            user_profiles=profiles,
            msg=f"Found {len(profiles)} matching profile(s)",
        )

    def rerank_user_profiles(
        self,
        request: RerankUserProfilesRequest | dict,
    ) -> RerankUserProfilesResponse:
        """Rerank a list of profile ids by query relevance using a cross-encoder.

        Fetches each profile's full content (filtered by ``user_id``), scores
        ``(query, content)`` pairs with ``cross-encoder/ms-marco-MiniLM-L-6-v2``,
        and returns the top_k profiles sorted by descending score. Profile ids
        that don't exist for the user are silently dropped.

        Args:
            request (Union[RerankUserProfilesRequest, dict]): The rerank
                request — must contain ``user_id``, ``query``, and
                ``profile_ids``.

        Returns:
            RerankUserProfilesResponse: Profiles sorted by descending
                relevance score, capped at ``request.top_k``.
        """
        if not self._is_storage_configured():
            return RerankUserProfilesResponse(
                success=True, user_profiles=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = RerankUserProfilesRequest(**request)
        if not request.profile_ids:
            return RerankUserProfilesResponse(
                success=True, user_profiles=[], msg="No profile_ids provided"
            )

        # Fetch every profile for the user — including PENDING and ARCHIVED —
        # because callers may want to rerank historical context, not just
        # the currently-published set.
        all_profiles = self._get_storage().get_user_profile(
            request.user_id, status_filter=[None, Status.PENDING, Status.ARCHIVED]
        )
        wanted = set(request.profile_ids)
        candidates = [p for p in all_profiles if p.profile_id in wanted]
        dropped = len(request.profile_ids) - len(candidates)

        # Lazy import keeps test collection fast; the cross-encoder pulls in
        # torch + sentence-transformers on first call.
        from reflexio.server.llm.rerank import score_pairs

        scores = score_pairs(request.query, [p.content for p in candidates])
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        top = [profile for profile, _score in ranked[: request.top_k]]
        msg = f"Reranked {len(candidates)} profile(s); dropped {dropped} unknown id(s)"
        return RerankUserProfilesResponse(success=True, user_profiles=top, msg=msg)

    def storage_stats(
        self,
        request: StorageStatsRequest | dict,
    ) -> StorageStatsResponse:
        """Return lightweight metadata about a user's stored profiles + playbooks.

        Provides counts and the last-modified timestamp range across every
        status, suitable for sizing ``top_k`` before retrieval.

        Args:
            request (Union[StorageStatsRequest, dict]): The stats request —
                must contain ``user_id``.

        Returns:
            StorageStatsResponse: Counts and timestamp range for the user.
        """
        if not self._is_storage_configured():
            return StorageStatsResponse(
                success=True,
                profile_count=0,
                playbook_count=0,
                msg=STORAGE_NOT_CONFIGURED_MSG,
            )
        if isinstance(request, dict):
            request = StorageStatsRequest(**request)
        storage = self._get_storage()
        # Walk every status — agent callers care about total surface area,
        # not just CURRENT entries.
        all_statuses: list[Status | None] = [None, Status.PENDING, Status.ARCHIVED]
        profiles = storage.get_user_profile(request.user_id, status_filter=all_statuses)
        oldest_ts: datetime | None = None
        newest_ts: datetime | None = None
        if profiles:
            timestamps = [p.last_modified_timestamp for p in profiles]
            oldest_ts = datetime.fromtimestamp(min(timestamps), tz=UTC)
            newest_ts = datetime.fromtimestamp(max(timestamps), tz=UTC)
        playbook_count = storage.count_user_playbooks(
            user_id=request.user_id, status_filter=all_statuses
        )
        return StorageStatsResponse(
            success=True,
            profile_count=len(profiles),
            playbook_count=playbook_count,
            oldest_profile_modified=oldest_ts,
            newest_profile_modified=newest_ts,
            msg=f"Found {len(profiles)} profile(s) and {playbook_count} playbook(s)",
        )

    def get_profile_change_logs(self) -> ProfileChangeLogResponse:
        """Get profile change logs, served from the lineage reconstruction.

        B3 Task 3: the change-log view is rebuilt on demand from ``lineage_event``
        linkage joined to survivor/tombstone content via
        :func:`reconstruct_profile_change_log`, rather than read from the legacy
        ``profile_change_logs`` table. The legacy table is still written (Task 6
        pending), so this read-side repoint is reversible. ``mentioned_profiles``
        is always ``[]`` in the reconstructed shape (unchanged from legacy).

        Returns:
            ProfileChangeLogResponse: Response containing the reconstructed
                profile change logs.
        """
        if not self._is_storage_configured():
            return ProfileChangeLogResponse(success=True, profile_change_logs=[])
        return reconstruct_profile_change_log(self._get_storage())

    @_require_storage(DeleteUserProfileResponse)
    def delete_profile(
        self,
        request: DeleteUserProfileRequest | dict,
    ) -> DeleteUserProfileResponse:
        """Delete user profiles.

        Args:
            request (DeleteUserProfileRequest): The delete request

        Returns:
            DeleteUserProfileResponse: Response containing success status and message
        """
        if isinstance(request, dict):
            request = DeleteUserProfileRequest(**request)
        self._get_storage().delete_user_profile(request)
        return DeleteUserProfileResponse(success=True, message="Deleted successfully")

    @_require_storage(UpdateUserProfileResponse, msg_field="msg")
    def update_user_profile(
        self,
        request: UpdateUserProfileRequest | dict,
    ) -> UpdateUserProfileResponse:
        """Apply a partial update to an existing user profile.

        Fetches the current profile by ``(user_id, profile_id)``, applies the
        non-None fields from ``request``, refreshes ``last_modified_timestamp``,
        and persists the whole record via
        :meth:`BaseStorage.update_user_profile_by_id`. The storage layer
        regenerates the embedding for the updated content.

        Args:
            request (Union[UpdateUserProfileRequest, dict]): The update request.
                ``user_id`` and ``profile_id`` are required; ``content`` and
                ``custom_features`` are optional — only non-None fields are
                applied.

        Returns:
            UpdateUserProfileResponse: ``success=True`` when the profile was
                updated, ``success=False`` with a descriptive ``msg`` when it
                could not be found.
        """
        if isinstance(request, dict):
            request = UpdateUserProfileRequest(**request)
        storage = self._get_storage()
        profiles = storage.get_user_profile(request.user_id)
        existing = next(
            (p for p in profiles if p.profile_id == request.profile_id), None
        )
        if existing is None:
            return UpdateUserProfileResponse(
                success=False,
                msg=(
                    f"Profile not found: user_id={request.user_id!r} "
                    f"profile_id={request.profile_id!r}"
                ),
            )
        if request.content is not None:
            existing.content = request.content
        if request.custom_features is not None:
            existing.custom_features = request.custom_features
        existing.last_modified_timestamp = int(datetime.now(UTC).timestamp())
        storage.update_user_profile_by_id(request.user_id, request.profile_id, existing)
        return UpdateUserProfileResponse(
            success=True, msg="User profile updated successfully"
        )

    @_require_storage(BulkDeleteResponse)
    def delete_all_profiles_bulk(self) -> BulkDeleteResponse:
        """Delete all profiles.

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        self._get_storage().delete_all_profiles()
        return BulkDeleteResponse(success=True, message="Deleted successfully")

    @_require_storage(BulkDeleteResponse)
    def delete_profiles_by_ids(
        self,
        request: DeleteProfilesByIdsRequest | dict,
    ) -> BulkDeleteResponse:
        """Delete profiles by their IDs.

        Args:
            request (DeleteProfilesByIdsRequest): The delete request containing profile_ids

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        if isinstance(request, dict):
            request = DeleteProfilesByIdsRequest(**request)
        deleted = self._get_storage().delete_profiles_by_ids(request.profile_ids)
        return BulkDeleteResponse(
            success=True, deleted_count=deleted, message=f"Deleted {deleted} item(s)"
        )

    def add_user_profile(
        self,
        request: AddUserProfileRequest | dict,
    ) -> AddUserProfileResponse:
        """Add user profiles directly to storage, bypassing inference.

        Mirrors :meth:`add_user_playbook` — useful for seeding a known
        fact about a user (testing, migration, manual fact injection)
        without going through the interaction-based generation pipeline.
        The storage layer's ``add_user_profile`` populates the embedding
        automatically.

        Args:
            request (Union[AddUserProfileRequest, dict]): The add
                request containing user profiles. Profiles must each
                have a non-empty ``content`` field.

        Returns:
            AddUserProfileResponse: Response containing success status,
                message, and count of profiles added.
        """
        if not self._is_storage_configured():
            return AddUserProfileResponse(
                success=False, message=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = AddUserProfileRequest(**request)

        # Group by user_id since storage.add_user_profile takes
        # (user_id, list[UserProfile]) and we want one storage call
        # per user.
        by_user: dict[str, list] = {}
        for p in request.user_profiles:
            by_user.setdefault(p.user_id, []).append(p)

        # Per-user try/except so we can surface partial-success in
        # the response message instead of silently losing track of
        # which users were persisted before a later failure.
        persisted_profiles = 0
        for persisted_users, (user_id, profiles) in enumerate(by_user.items()):
            try:
                self._get_storage().add_user_profile(user_id, profiles)
            except Exception:
                # Log the full exception for operators (storage errors
                # may contain SQL text, file paths, table names); return
                # a generic message to the caller to avoid information
                # disclosure over HTTP.
                logger.exception("add_user_profile failed for user_id=%s", user_id)
                if persisted_users == 0:
                    message = "Failed to add user profile"
                else:
                    message = (
                        f"Partially persisted {persisted_profiles} profile(s) "
                        f"for {persisted_users} user(s) before failing on "
                        f"user {user_id}"
                    )
                return AddUserProfileResponse(success=False, message=message)
            persisted_profiles += len(profiles)

        return AddUserProfileResponse(
            success=True,
            added_count=persisted_profiles,
            message=f"Added {persisted_profiles} profile(s)",
        )

    def get_profiles(
        self,
        request: GetUserProfilesRequest | dict,
        status_filter: list[Status | None] | None = None,
    ) -> GetUserProfilesResponse:
        """Get user profiles.

        Args:
            request (GetUserProfilesRequest): The get request
            status_filter (Optional[list[Optional[Status]]]): Filter profiles by status. Defaults to [None] for current profiles only.
                If provided, takes precedence over request.status_filter.

        Returns:
            GetUserProfilesResponse: Response containing user profiles
        """
        if not self._is_storage_configured():
            return GetUserProfilesResponse(
                success=True, user_profiles=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = GetUserProfilesRequest(**request)

        # Priority: parameter > request.status_filter > default [None]
        if status_filter is None:
            if hasattr(request, "status_filter") and request.status_filter is not None:
                status_filter = request.status_filter
            else:
                status_filter = [None]  # Default to current profiles

        profiles = self._get_storage().get_user_profile(
            request.user_id, status_filter=status_filter, tags=request.tags
        )
        profiles = sorted(
            profiles, key=lambda x: x.last_modified_timestamp, reverse=True
        )

        # Apply time filters
        if request.start_time:
            profiles = [
                p
                for p in profiles
                if p.last_modified_timestamp >= int(request.start_time.timestamp())
            ]
        if request.end_time:
            profiles = [
                p
                for p in profiles
                if p.last_modified_timestamp <= int(request.end_time.timestamp())
            ]

        # Apply top_k limit
        if request.top_k:
            profiles = sorted(
                profiles, key=lambda x: x.last_modified_timestamp, reverse=True
            )[: request.top_k]

        return GetUserProfilesResponse(
            success=True,
            user_profiles=profiles,
            msg=f"Found {len(profiles)} profile(s)",
        )

    def get_all_profiles(
        self,
        limit: int = 100,
        status_filter: list[Status | None] | None = None,
    ) -> GetUserProfilesResponse:
        """Get all user profiles across all users.

        Args:
            limit (int, optional): Maximum number of profiles to return. Defaults to 100.
            status_filter (Optional[list[Optional[Status]]]): Filter profiles by status. Defaults to [None] for current profiles only.

        Returns:
            GetUserProfilesResponse: Response containing all user profiles
        """
        if not self._is_storage_configured():
            return GetUserProfilesResponse(
                success=True, user_profiles=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if status_filter is None:
            status_filter = [None]  # Default to current profiles
        profiles = self._get_storage().get_all_profiles(
            limit=limit, status_filter=status_filter
        )
        profiles = sorted(
            profiles, key=lambda x: x.last_modified_timestamp, reverse=True
        )
        return GetUserProfilesResponse(
            success=True,
            user_profiles=profiles,
            msg=f"Found {len(profiles)} profile(s)",
        )

    def upgrade_all_profiles(
        self,
        request: UpgradeProfilesRequest | dict | None = None,
    ) -> UpgradeProfilesResponse:
        """Upgrade all profiles by deleting old ARCHIVED, archiving CURRENT, and promoting PENDING.

        Args:
            request (Union[UpgradeProfilesRequest, dict], optional): The upgrade request

        Returns:
            UpgradeProfilesResponse: Response containing success status and counts
        """
        if not self._is_storage_configured():
            return UpgradeProfilesResponse(
                success=False, message=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = UpgradeProfilesRequest(**request)
        elif request is None:
            request = UpgradeProfilesRequest(user_id=None, only_affected_users=False)

        service = ProfileGenerationService(
            llm_client=self.llm_client,
            request_context=self.request_context,
        )
        return service.run_upgrade(request)  # type: ignore[reportArgumentType]

    def downgrade_all_profiles(
        self,
        request: DowngradeProfilesRequest | dict | None = None,
    ) -> DowngradeProfilesResponse:
        """Downgrade all profiles by archiving CURRENT and restoring ARCHIVED.

        Args:
            request (Union[DowngradeProfilesRequest, dict], optional): The downgrade request

        Returns:
            DowngradeProfilesResponse: Response containing success status and counts
        """
        if not self._is_storage_configured():
            return DowngradeProfilesResponse(
                success=False, message=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = DowngradeProfilesRequest(**request)
        elif request is None:
            request = DowngradeProfilesRequest(user_id=None, only_affected_users=False)

        service = ProfileGenerationService(
            llm_client=self.llm_client,
            request_context=self.request_context,
        )
        return service.run_downgrade(request)  # type: ignore[reportArgumentType]

    def get_profile_statistics(self) -> GetProfileStatisticsResponse:
        """Get profile count statistics by status.

        Returns:
            GetProfileStatisticsResponse: Response containing profile counts
        """
        if not self._is_storage_configured():
            return GetProfileStatisticsResponse(
                success=True,
                current_count=0,
                pending_count=0,
                archived_count=0,
                expiring_soon_count=0,
                msg=STORAGE_NOT_CONFIGURED_MSG,
            )
        try:
            stats = self._get_storage().get_profile_statistics()
            return GetProfileStatisticsResponse(
                success=True, msg="Retrieved profile statistics successfully", **stats
            )
        except Exception as e:
            return GetProfileStatisticsResponse(
                success=False, msg=f"Failed to get profile statistics: {str(e)}"
            )


# ---------------------------------------------------------------------------
# Standalone read-side reconstruction (Phase B3 Task 2)
# ---------------------------------------------------------------------------


def reconstruct_profile_change_log(
    storage: ParityReadStorage,
    *,
    limit: int = 100,
) -> ProfileChangeLogResponse:
    """Rebuild the ProfileChangeLog view using time-travel-stable signals.

    Uses two immutable / stable signals to classify every dedup run:

    * **added(R)** — profiles whose ``generated_from_request_id == R``.  This
      column is set at creation and never changes, so it correctly classifies
      a profile as "added in run R" even if it is later tombstoned in run R2.
      Tombstones are included so the content is available.

    * **removed(R)** — entity_ids of ``status_change`` lineage events with
      ``to_status == "superseded"`` and ``request_id == R``.  This is the
      exact signature emitted by ``supersede_profiles_by_ids`` (the dedup
      soft-delete path).  It is distinct from reflection which emits
      ``op="revise"``, so reflection events are never mis-counted as removals.

    Groups are formed over the union of request_ids from both signals.
    Request_id ``""`` is skipped — it would merge unrelated runs.
    A row is emitted only when ``added or removed`` is non-empty (matching
    legacy semantics: ``add_profile_change_log`` was called only when
    ``all_new_profiles or superseded_profiles``).

    ``mentioned_profiles = []`` — always empty (Stage-1 shape; the legacy
    path also always wrote ``[]``).

    When a removed profile's tombstone has been physically purged (GDPR GC),
    it is silently omitted from ``removed_profiles`` rather than crashing.

    Args:
        storage (ParityReadStorage): Storage read surface to query (any
            BaseStorage backend or the read-only parity reader).
        limit (int): Maximum number of reconstructed entries to return.
            Defaults to 100.

    Returns:
        ProfileChangeLogResponse: ``success=True`` with reconstructed rows
            ordered most-recent first (by max event ``created_at`` in each
            request_id group), capped at ``limit``.
    """
    if limit <= 0:
        return ProfileChangeLogResponse(success=True, profile_change_logs=[])

    all_events = storage.get_lineage_events(
        entity_type="profile", org_id=storage.org_id
    )

    # Dedup soft-delete signature: status_change to_status=="superseded".
    # Each such event records one profile removed in the dedup run ``request_id``.
    # Distinct from reflection which emits op="revise" — so revise events are
    # never counted as removals here.
    removal_by_req: dict[str, list[str]] = defaultdict(list)
    sort_key: dict[str, tuple[int, int]] = {}  # request_id -> (created_at, event_id)
    for evt in all_events:
        key = evt.request_id
        if not key:
            continue  # skip empty-string request_ids — never merge unrelated runs
        cur = sort_key.get(key, (0, 0))
        if (evt.created_at, evt.event_id) > cur:
            sort_key[key] = (evt.created_at, evt.event_id)
        if evt.op == "status_change" and evt.to_status == "superseded":
            removal_by_req[key].append(evt.entity_id)

    # Resolve the "added" side for every run in ONE bulk read, grouped by each
    # profile's immutable generated_from_request_id. This replaces a read per
    # candidate request_id, which fanned out over the org's whole dedup history
    # before the limit slice below — a hot-path N+1 on network-backed storage now
    # that the live endpoint serves this reconstruction.
    added_by_req: dict[str, list] = defaultdict(list)
    for profile in storage.get_all_generated_profiles():
        added_by_req[profile.generated_from_request_id].append(profile)

    # Candidate request_ids are the UNION of:
    #   (a) lineage EVENT request_ids — runs that produced a dedup removal
    #       (status_change/superseded event); and
    #   (b) request_ids stamped on profile rows (the keys of added_by_req) —
    #       discovers ADD-ONLY dedup runs (new profiles, nothing superseded) that
    #       emit no lineage event. ``get_all_generated_profiles`` already excludes
    #       the empty-string sentinel, so unrelated runs are never merged.
    #
    # This closes the reconstruction-completeness gap for add-only runs: the
    # legacy `add_profile_change_log` fired whenever `all_new_profiles or
    # superseded_profiles` was non-empty, so a run with only adds was still
    # recorded. The (b) path mirrors that.
    #
    # For add-only runs (in set (b) but not (a)), no event timestamp exists.
    # We derive their sort key from the max `last_modified_timestamp` of the
    # profiles in that group — set at creation, giving a sensible most-recent-first
    # ordering relative to event-timestamped runs. The secondary key is 0.
    candidate_req_ids: set[str] = set(sort_key.keys()) | set(added_by_req.keys())

    def _effective_sort_key(req_id: str) -> tuple[int, int]:
        """Return (timestamp, event_id) for sorting; for add-only runs fall back to
        the max last_modified_timestamp of the profiles in that group."""
        if req_id in sort_key:
            return sort_key[req_id]
        profiles = added_by_req.get(req_id, [])
        max_ts = max((p.last_modified_timestamp for p in profiles), default=0)
        return (max_ts, 0)

    sorted_keys = sorted(
        candidate_req_ids,
        key=_effective_sort_key,
        reverse=True,
    )[:limit]

    logs: list[ProfileChangeLog] = []
    for req_id in sorted_keys:
        # added: profiles whose generated_from_request_id == req_id (any status,
        # include tombstones — a profile added in R1 and tombstoned in R2 is still
        # "added in R1").
        added = added_by_req[req_id]

        # removed: dedup-superseded profiles from this run's lineage events.
        removed: list = []
        for entity_id in removal_by_req.get(req_id, []):
            profile = storage.get_profile_by_id(entity_id, include_tombstones=True)
            if profile is not None:
                removed.append(profile)
            # Tombstone physically purged (GDPR GC) → silently omit; no crash.

        if not added and not removed:
            # No dedup activity for this request_id — skip to match legacy semantics.
            continue

        ts, _ = _effective_sort_key(req_id)
        user_id = added[0].user_id if added else removed[0].user_id
        logs.append(
            ProfileChangeLog(
                id=0,
                user_id=user_id,
                request_id=req_id,
                created_at=ts,
                added_profiles=added,
                removed_profiles=removed,
                mentioned_profiles=[],
            )
        )

    return ProfileChangeLogResponse(success=True, profile_change_logs=logs)
