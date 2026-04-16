import logging

logger = logging.getLogger(__name__)

from datetime import UTC, datetime

from reflexio.lib._base import (
    STORAGE_NOT_CONFIGURED_MSG,
    ReflexioBase,
    _require_storage,
)
from reflexio.models.api_schema.retriever_schema import (
    GetProfileStatisticsResponse,
    GetUserProfilesRequest,
    GetUserProfilesResponse,
    SearchUserProfileRequest,
    SearchUserProfileResponse,
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


class ProfilesMixin(ReflexioBase):
    def search_profiles(
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
            "search_profiles: query=%r, search_mode=%s, embedding_generated=%s",
            request.query,
            request.search_mode,
            query_embedding is not None,
        )
        profiles = self._get_storage().search_user_profile(
            request, status_filter=status_filter, query_embedding=query_embedding
        )
        return SearchUserProfileResponse(
            success=True,
            user_profiles=profiles,
            msg=f"Found {len(profiles)} matching profile(s)",
        )

    def get_profile_change_logs(self) -> ProfileChangeLogResponse:
        """Get profile change logs.

        Returns:
            ProfileChangeLogResponse: Response containing profile change logs
        """
        if not self._is_storage_configured():
            return ProfileChangeLogResponse(success=True, profile_change_logs=[])
        changelogs = self._get_storage().get_profile_change_logs()
        return ProfileChangeLogResponse(success=True, profile_change_logs=changelogs)

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
            request.user_id, status_filter=status_filter
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
