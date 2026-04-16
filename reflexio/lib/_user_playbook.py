from reflexio.lib._base import (
    STORAGE_NOT_CONFIGURED_MSG,
    ReflexioBase,
    _require_storage,
)
from reflexio.models.api_schema.retriever_schema import (
    GetUserPlaybooksRequest,
    GetUserPlaybooksResponse,
    SearchUserPlaybookRequest,
    SearchUserPlaybookResponse,
    UpdateUserPlaybookRequest,
    UpdateUserPlaybookResponse,
)
from reflexio.models.api_schema.service_schemas import (
    AddUserPlaybookRequest,
    AddUserPlaybookResponse,
    BulkDeleteResponse,
    DeleteUserPlaybookRequest,
    DeleteUserPlaybookResponse,
    DeleteUserPlaybooksByIdsRequest,
    DowngradeUserPlaybooksRequest,
    DowngradeUserPlaybooksResponse,
    UpgradeUserPlaybooksRequest,
    UpgradeUserPlaybooksResponse,
)
from reflexio.models.config_schema import SearchOptions
from reflexio.server.services.playbook.playbook_generation_service import (
    PlaybookGenerationService,
)


class UserPlaybookMixin(ReflexioBase):
    def get_user_playbooks(
        self,
        request: GetUserPlaybooksRequest | dict,
    ) -> GetUserPlaybooksResponse:
        """Get user playbooks.

        Args:
            request (Union[GetUserPlaybooksRequest, dict]): The get request

        Returns:
            GetUserPlaybooksResponse: Response containing user playbooks
        """
        if not self._is_storage_configured():
            return GetUserPlaybooksResponse(
                success=True, user_playbooks=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = GetUserPlaybooksRequest(**request)

        try:
            user_playbooks = self._get_storage().get_user_playbooks(
                limit=request.limit or 100,
                user_id=request.user_id,
                playbook_name=request.playbook_name,
                agent_version=request.agent_version,
                status_filter=request.status_filter,
            )
            return GetUserPlaybooksResponse(
                success=True,
                user_playbooks=user_playbooks,
                msg=f"Found {len(user_playbooks)} user playbook(s)",
            )
        except Exception as e:
            return GetUserPlaybooksResponse(
                success=False, user_playbooks=[], msg=str(e)
            )

    def add_user_playbook(
        self,
        request: AddUserPlaybookRequest | dict,
    ) -> AddUserPlaybookResponse:
        """Add user playbooks directly to storage.

        Args:
            request (Union[AddUserPlaybookRequest, dict]): The add request containing user playbooks

        Returns:
            AddUserPlaybookResponse: Response containing success status, message, and count of added playbooks
        """
        if not self._is_storage_configured():
            return AddUserPlaybookResponse(
                success=False, message=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = AddUserPlaybookRequest(**request)

        try:
            self._get_storage().save_user_playbooks(list(request.user_playbooks))
            return AddUserPlaybookResponse(
                success=True,
                added_count=len(request.user_playbooks),
                message=f"Added {len(request.user_playbooks)} item(s)",
            )
        except Exception as e:
            return AddUserPlaybookResponse(success=False, message=str(e))

    def search_user_playbooks(
        self,
        request: SearchUserPlaybookRequest | dict,
    ) -> SearchUserPlaybookResponse:
        """Search user playbooks with advanced filtering and semantic search.

        Args:
            request (Union[SearchUserPlaybookRequest, dict]): The search request

        Returns:
            SearchUserPlaybookResponse: Response containing matching user playbooks
        """
        if not self._is_storage_configured():
            return SearchUserPlaybookResponse(
                success=True, user_playbooks=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = SearchUserPlaybookRequest(**request)

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
            user_playbooks = self._get_storage().search_user_playbooks(
                search_request, options
            )
            return SearchUserPlaybookResponse(
                success=True,
                user_playbooks=user_playbooks,
                msg=f"Found {len(user_playbooks)} matching user playbook(s)",
            )
        except Exception as e:
            return SearchUserPlaybookResponse(
                success=False, user_playbooks=[], msg=str(e)
            )

    @_require_storage(DeleteUserPlaybookResponse)
    def delete_user_playbook(
        self,
        request: DeleteUserPlaybookRequest | dict,
    ) -> DeleteUserPlaybookResponse:
        """Delete a user playbook by ID.

        Args:
            request (DeleteUserPlaybookRequest): The delete request containing user_playbook_id

        Returns:
            DeleteUserPlaybookResponse: Response containing success status and message
        """
        if isinstance(request, dict):
            request = DeleteUserPlaybookRequest(**request)
        self._get_storage().delete_user_playbook(request.user_playbook_id)
        return DeleteUserPlaybookResponse(success=True, message="Deleted successfully")

    @_require_storage(BulkDeleteResponse)
    def delete_user_playbooks_by_ids_bulk(
        self,
        request: DeleteUserPlaybooksByIdsRequest | dict,
    ) -> BulkDeleteResponse:
        """Delete user playbooks by their IDs.

        Args:
            request (DeleteUserPlaybooksByIdsRequest): The delete request containing user_playbook_ids

        Returns:
            BulkDeleteResponse: Response containing success status and deleted count
        """
        if isinstance(request, dict):
            request = DeleteUserPlaybooksByIdsRequest(**request)
        deleted = self._get_storage().delete_user_playbooks_by_ids(
            request.user_playbook_ids
        )
        return BulkDeleteResponse(
            success=True, deleted_count=deleted, message=f"Deleted {deleted} item(s)"
        )

    @_require_storage(BulkDeleteResponse)
    def delete_all_user_playbooks_bulk(self) -> BulkDeleteResponse:
        """Delete all user playbooks (only user playbooks, not agent playbooks).

        Unlike :meth:`delete_all_playbooks_bulk` on ``AgentPlaybookMixin``
        (which cascades to both user and agent playbooks), this method
        scopes the deletion strictly to user playbooks. Use this from
        CLI or API callers that want per-entity semantics.

        Returns:
            BulkDeleteResponse: Response containing success status and message.
        """
        self._get_storage().delete_all_user_playbooks()
        return BulkDeleteResponse(success=True, message="Deleted successfully")

    @_require_storage(UpdateUserPlaybookResponse, msg_field="msg")
    def update_user_playbook(
        self,
        request: UpdateUserPlaybookRequest | dict,
    ) -> UpdateUserPlaybookResponse:
        """Update editable fields of a user playbook.

        Args:
            request (Union[UpdateUserPlaybookRequest, dict]): The update request

        Returns:
            UpdateUserPlaybookResponse: Response containing success status and message
        """
        if isinstance(request, dict):
            request = UpdateUserPlaybookRequest(**request)
        self._get_storage().update_user_playbook(
            user_playbook_id=request.user_playbook_id,
            playbook_name=request.playbook_name,
            content=request.content,
            trigger=request.trigger,
            rationale=request.rationale,
            blocking_issue=request.blocking_issue,
        )
        return UpdateUserPlaybookResponse(
            success=True, msg="User playbook updated successfully"
        )

    def upgrade_all_user_playbooks(
        self,
        request: UpgradeUserPlaybooksRequest | dict | None = None,
    ) -> UpgradeUserPlaybooksResponse:
        """Upgrade all user playbooks by deleting old ARCHIVED, archiving CURRENT, and promoting PENDING.

        Args:
            request (Union[UpgradeUserPlaybooksRequest, dict], optional): The upgrade request

        Returns:
            UpgradeUserPlaybooksResponse: Response containing success status and counts
        """
        if not self._is_storage_configured():
            return UpgradeUserPlaybooksResponse(
                success=False, message=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = UpgradeUserPlaybooksRequest(**request)
        elif request is None:
            request = UpgradeUserPlaybooksRequest()

        service = PlaybookGenerationService(
            llm_client=self.llm_client,
            request_context=self.request_context,
        )
        return service.run_upgrade(request)  # type: ignore[reportArgumentType]

    def downgrade_all_user_playbooks(
        self,
        request: DowngradeUserPlaybooksRequest | dict | None = None,
    ) -> DowngradeUserPlaybooksResponse:
        """Downgrade all user playbooks by archiving CURRENT and restoring ARCHIVED.

        Args:
            request (Union[DowngradeUserPlaybooksRequest, dict], optional): The downgrade request

        Returns:
            DowngradeUserPlaybooksResponse: Response containing success status and counts
        """
        if not self._is_storage_configured():
            return DowngradeUserPlaybooksResponse(
                success=False, message=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = DowngradeUserPlaybooksRequest(**request)
        elif request is None:
            request = DowngradeUserPlaybooksRequest()

        service = PlaybookGenerationService(
            llm_client=self.llm_client,
            request_context=self.request_context,
        )
        return service.run_downgrade(request)  # type: ignore[reportArgumentType]
