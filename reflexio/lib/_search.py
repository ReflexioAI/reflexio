import logging

from reflexio.lib._base import STORAGE_NOT_CONFIGURED_MSG, ReflexioBase
from reflexio.models.api_schema.retriever_schema import (
    GetAgentSuccessEvaluationResultsRequest,
    GetAgentSuccessEvaluationResultsResponse,
    GetRequestsRequest,
    GetRequestsResponse,
    GetRetrievedLearningEvaluationResultsRequest,
    GetRetrievedLearningEvaluationResultsResponse,
    RequestData,
    Session,
    UnifiedSearchRequest,
    UnifiedSearchResponse,
)
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.site_var.site_var_manager import SiteVarManager
from reflexio.server.tracing import profile_step

_LOGGER = logging.getLogger(__name__)


class SearchMixin(ReflexioBase):
    def get_agent_success_evaluation_results(
        self,
        request: GetAgentSuccessEvaluationResultsRequest | dict,
    ) -> GetAgentSuccessEvaluationResultsResponse:
        """Get agent success evaluation results.

        Args:
            request (Union[GetAgentSuccessEvaluationResultsRequest, dict]): The get request

        Returns:
            GetAgentSuccessEvaluationResultsResponse: Response containing agent success evaluation results
        """
        if not self._is_storage_configured():
            return GetAgentSuccessEvaluationResultsResponse(
                success=True,
                agent_success_evaluation_results=[],
                msg=STORAGE_NOT_CONFIGURED_MSG,
            )
        if isinstance(request, dict):
            request = GetAgentSuccessEvaluationResultsRequest(**request)

        try:
            if request.start_time or request.end_time:
                results = (
                    self._get_storage().get_agent_success_evaluation_results_in_window(
                        from_ts=(
                            int(request.start_time.timestamp())
                            if request.start_time
                            else 0
                        ),
                        to_ts=(
                            int(request.end_time.timestamp())
                            if request.end_time
                            else 2**31 - 1
                        ),
                        limit=request.limit or 100,
                        agent_version=request.agent_version,
                        include_embedding=False,
                    )
                )
            else:
                results = self._get_storage().get_agent_success_evaluation_results(
                    limit=request.limit or 100,
                    agent_version=request.agent_version,
                    include_embedding=False,
                )
            return GetAgentSuccessEvaluationResultsResponse(
                success=True,
                agent_success_evaluation_results=results,
                msg=f"Found {len(results)} evaluation result(s)",
            )
        except Exception as e:
            return GetAgentSuccessEvaluationResultsResponse(
                success=False, agent_success_evaluation_results=[], msg=str(e)
            )

    def get_retrieved_learning_evaluation_results(
        self,
        request: GetRetrievedLearningEvaluationResultsRequest | dict,
    ) -> GetRetrievedLearningEvaluationResultsResponse:
        """Get per-learning retrieved-learning evaluation verdicts.

        Args:
            request (GetRetrievedLearningEvaluationResultsRequest | dict): The
                read request (optional user/session/interaction-time filters + limit).

        Returns:
            GetRetrievedLearningEvaluationResultsResponse: Matching verdicts
            ordered by target interaction time for time-filtered reads and by
            result creation time otherwise.
        """
        if not self._is_storage_configured():
            return GetRetrievedLearningEvaluationResultsResponse(
                success=True, results=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = GetRetrievedLearningEvaluationResultsRequest(**request)
        try:
            results = self._get_storage().get_retrieved_learning_evaluation_results(
                user_id=request.user_id,
                session_id=request.session_id,
                from_ts=(
                    int(request.start_time.timestamp()) if request.start_time else None
                ),
                to_ts=int(request.end_time.timestamp()) if request.end_time else None,
                limit=request.limit,
            )
            return GetRetrievedLearningEvaluationResultsResponse(
                success=True,
                results=results,
                msg=f"Found {len(results)} retrieved-learning evaluation result(s)",
            )
        except Exception:
            _LOGGER.exception("Failed to read retrieved-learning evaluation results")
            return GetRetrievedLearningEvaluationResultsResponse(
                success=False,
                results=[],
                msg="Failed to read retrieved-learning evaluation results",
            )

    def get_requests(
        self,
        request: GetRequestsRequest | dict,
    ) -> GetRequestsResponse:
        """Get requests with their associated interactions, grouped by session.

        Args:
            request (Union[GetRequestsRequest, dict]): The get request

        Returns:
            GetRequestsResponse: Response containing requests grouped by session with their interactions
        """
        if not self._is_storage_configured():
            return GetRequestsResponse(
                success=True, sessions=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        if isinstance(request, dict):
            request = GetRequestsRequest(**request)

        try:
            # Get requests with interactions from storage (already grouped by session)
            grouped_results = self._get_storage().get_sessions(
                user_id=request.user_id,
                request_id=request.request_id,
                session_id=request.session_id,
                source=request.source,
                start_time=(
                    int(request.start_time.timestamp()) if request.start_time else None
                ),
                end_time=(
                    int(request.end_time.timestamp()) if request.end_time else None
                ),
                top_k=request.top_k,
                offset=request.offset or 0,
            )

            # Transform the dictionary into Session objects
            sessions = []
            for group_name, request_interaction_data_models in grouped_results.items():
                # Convert each RequestInteractionDataModel to RequestData
                request_data_list = [
                    RequestData(
                        request=request_interaction.request,
                        interactions=request_interaction.interactions,
                    )
                    for request_interaction in request_interaction_data_models
                ]
                sessions.append(
                    Session(session_id=group_name, requests=request_data_list)
                )

            # Pagination is session-based (top_k counts sessions): there may be
            # more pages when this page filled the session limit.
            total_returned = sum(len(s.requests) for s in sessions)
            effective_limit = request.top_k or 100
            has_more = len(sessions) >= effective_limit

            return GetRequestsResponse(
                success=True,
                sessions=sessions,
                has_more=has_more,
                msg=f"Found {total_returned} request(s) in {len(sessions)} session(s)",
            )
        except Exception as e:
            return GetRequestsResponse(success=False, sessions=[], msg=str(e))

    def unified_search(
        self,
        request: UnifiedSearchRequest | dict,
        org_id: str,
    ) -> UnifiedSearchResponse:
        """Search across all entity types (profiles, feedbacks, raw_feedbacks) in parallel.

        Args:
            request (Union[UnifiedSearchRequest, dict]): The unified search request
            org_id (str): Organization ID (used for feature flag checks)

        Returns:
            UnifiedSearchResponse: Combined results from all entity types
        """
        if not self._is_storage_configured():
            return UnifiedSearchResponse(success=True, msg=STORAGE_NOT_CONFIGURED_MSG)
        if isinstance(request, dict):
            request = UnifiedSearchRequest(**request)

        with profile_step(
            "search.prepare",
            enabled=bool(request.enable_reformulation),
            has_conversation_history=bool(request.conversation_history),
            search_mode=request.search_mode,
        ):
            config = self.request_context.configurator.get_config()
            config_llm_config = config.llm_config if config else None

            # Resolve pre_retrieval_model_name: config override -> site var -> auto-detect.
            model_setting = SiteVarManager().get_site_var("llm_model_setting")
            site_var = model_setting if isinstance(model_setting, dict) else {}
            api_key_config = config.api_key_config if config else None

            pre_retrieval_model_name = resolve_model_name(
                ModelRole.PRE_RETRIEVAL,
                site_var_value=site_var.get("pre_retrieval_model_name"),
                config_override=config_llm_config.pre_retrieval_model_name
                if config_llm_config
                else None,
                api_key_config=api_key_config,
            )
            storage = self._get_storage()
            prompt_manager = self.request_context.prompt_manager
            retrieval_floor = config.retrieval_floor if config else None
            recency = getattr(storage, "recency", None)

        from reflexio.server.services.unified_search_service import run_unified_search

        return run_unified_search(
            request=request,
            org_id=org_id,
            storage=storage,
            llm_client=self.llm_client,
            prompt_manager=prompt_manager,
            pre_retrieval_model_name=pre_retrieval_model_name,
            retrieval_floor=retrieval_floor,
            recency=recency,
        )
