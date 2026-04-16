from reflexio.lib._base import (
    STORAGE_NOT_CONFIGURED_MSG,
    ReflexioBase,
    _require_storage,
)
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


class AgentPlaybookMixin(ReflexioBase):
    def get_playbook_aggregation_change_logs(
        self, playbook_name: str, agent_version: str
    ) -> PlaybookAggregationChangeLogResponse:
        """Get playbook aggregation change logs.

        Args:
            playbook_name (str): Playbook name to filter by
            agent_version (str): Agent version to filter by

        Returns:
            PlaybookAggregationChangeLogResponse: Response containing change logs
        """
        if not self._is_storage_configured():
            return PlaybookAggregationChangeLogResponse(success=True, change_logs=[])
        change_logs = self._get_storage().get_playbook_aggregation_change_logs(
            playbook_name=playbook_name, agent_version=agent_version
        )
        return PlaybookAggregationChangeLogResponse(
            success=True, change_logs=change_logs
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
            # Top-level structured fields (trigger, rationale, blocking_issue) are
            # preserved so CLI callers and the aggregation pipeline don't lose them.
            normalized_playbooks = [
                AgentPlaybook(
                    agent_version=fb.agent_version,
                    playbook_name=fb.playbook_name,
                    content=fb.content,
                    trigger=fb.trigger,
                    rationale=fb.rationale,
                    blocking_issue=fb.blocking_issue,
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
            agent_playbooks = self._get_storage().search_agent_playbooks(
                search_request, options
            )
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
            blocking_issue=request.blocking_issue,
            playbook_status=request.playbook_status,
        )
        return UpdateAgentPlaybookResponse(
            success=True, msg="Agent playbook updated successfully"
        )
