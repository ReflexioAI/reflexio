from typing import TYPE_CHECKING

from reflexio.lib._base import STORAGE_NOT_CONFIGURED_MSG, ReflexioBase
from reflexio.models.api_schema.braintrust_schema import (
    BraintrustStatusResponse,
    ConnectBraintrustRequest,
    ConnectBraintrustResponse,
    SelectProjectsRequest,
    SelectProjectsResponse,
    SyncBraintrustResponse,
)
from reflexio.models.api_schema.eval_overview_schema import (
    ContextTile,
    GetEvaluationOverviewRequest,
    GetEvaluationOverviewResponse,
    HeroBlock,
    NumberWithDelta,
    PercentWithDelta,
    ScoreDistribution,
)
from reflexio.models.api_schema.retriever_schema import (
    DashboardStats,
    GetDashboardStatsRequest,
    GetDashboardStatsResponse,
    GetPlaybookApplicationStatsRequest,
    GetPlaybookApplicationStatsResponse,
    PeriodStats,
    TimeSeriesDataPoint,
)

if TYPE_CHECKING:
    from reflexio.server.services.braintrust.service import (
        BraintrustConnectorService,
    )


class DashboardMixin(ReflexioBase):
    def get_dashboard_stats(
        self, request: GetDashboardStatsRequest | dict
    ) -> GetDashboardStatsResponse:
        """Get dashboard statistics including counts and time-series data.

        Args:
            request (Union[GetDashboardStatsRequest, dict]): Request containing days_back and granularity

        Returns:
            GetDashboardStatsResponse: Response containing dashboard statistics
        """
        if not self._is_storage_configured():
            # Return empty stats when storage is not configured
            empty_period = PeriodStats(
                total_profiles=0,
                total_interactions=0,
                total_playbooks=0,
                success_rate=0.0,
            )
            empty_stats = DashboardStats(
                current_period=empty_period,
                previous_period=empty_period,
                interactions_time_series=[],
                profiles_time_series=[],
                playbooks_time_series=[],
                evaluations_time_series=[],
            )
            return GetDashboardStatsResponse(
                success=True, stats=empty_stats, msg=STORAGE_NOT_CONFIGURED_MSG
            )
        try:
            # Convert dict to request object if needed
            if isinstance(request, dict):
                request = GetDashboardStatsRequest(**request)

            # Get stats from storage layer
            stats_dict = self._get_storage().get_dashboard_stats(
                days_back=request.days_back or 30
            )

            # Convert dict to Pydantic models
            current_period = PeriodStats(**stats_dict["current_period"])
            previous_period = PeriodStats(**stats_dict["previous_period"])

            interactions_time_series = [
                TimeSeriesDataPoint(**ts)
                for ts in stats_dict["interactions_time_series"]
            ]
            profiles_time_series = [
                TimeSeriesDataPoint(**ts) for ts in stats_dict["profiles_time_series"]
            ]
            playbooks_time_series = [
                TimeSeriesDataPoint(**ts) for ts in stats_dict["playbooks_time_series"]
            ]
            evaluations_time_series = [
                TimeSeriesDataPoint(**ts)
                for ts in stats_dict["evaluations_time_series"]
            ]

            # Build dashboard stats object
            dashboard_stats = DashboardStats(
                current_period=current_period,
                previous_period=previous_period,
                interactions_time_series=interactions_time_series,
                profiles_time_series=profiles_time_series,
                playbooks_time_series=playbooks_time_series,
                evaluations_time_series=evaluations_time_series,
            )

            return GetDashboardStatsResponse(
                success=True,
                stats=dashboard_stats,
                msg="Retrieved dashboard stats successfully",
            )

        except Exception as e:
            return GetDashboardStatsResponse(
                success=False, msg=f"Failed to get dashboard stats: {str(e)}"
            )

    def get_playbook_application_stats(
        self, request: GetPlaybookApplicationStatsRequest | dict
    ) -> GetPlaybookApplicationStatsResponse:
        """Get per-rule citation counts from the interactions table.

        Aggregates the JSON citations column on interactions in the look-back
        window and groups by (kind, real_id) so the dashboard can show how
        often each individual playbook or profile has been applied.

        Args:
            request (Union[GetPlaybookApplicationStatsRequest, dict]): Request
                containing days_back.

        Returns:
            GetPlaybookApplicationStatsResponse: Response containing the
                aggregated stats sorted by applied_count descending.
        """
        if not self._is_storage_configured():
            return GetPlaybookApplicationStatsResponse(
                success=True, stats=[], msg=STORAGE_NOT_CONFIGURED_MSG
            )
        try:
            if isinstance(request, dict):
                request = GetPlaybookApplicationStatsRequest(**request)
            stats = self._get_storage().get_playbook_application_stats(
                days_back=request.days_back
            )
            return GetPlaybookApplicationStatsResponse(
                success=True,
                stats=stats,
                msg="Retrieved playbook application stats successfully",
            )
        except Exception as e:
            return GetPlaybookApplicationStatsResponse(
                success=False,
                stats=[],
                msg=f"Failed to get playbook application stats: {str(e)}",
            )

    # ==============================
    # Braintrust connector (Plan C-backend)
    # ==============================

    def braintrust_connect(
        self, request: ConnectBraintrustRequest | dict
    ) -> ConnectBraintrustResponse:
        """Step 1 of the Braintrust connect flow — validate key, list workspaces."""
        if isinstance(request, dict):
            request = ConnectBraintrustRequest(**request)
        if not self._is_storage_configured():
            return ConnectBraintrustResponse(
                success=False, msg=STORAGE_NOT_CONFIGURED_MSG
            )
        return self._braintrust_service().connect(request)

    def braintrust_select_projects(
        self, request: SelectProjectsRequest | dict
    ) -> SelectProjectsResponse:
        """Step 2 — persist the connection with selected projects (key encrypted)."""
        if isinstance(request, dict):
            request = SelectProjectsRequest(**request)
        if not self._is_storage_configured():
            return SelectProjectsResponse(success=False, msg=STORAGE_NOT_CONFIGURED_MSG)
        return self._braintrust_service().select_projects(request)

    def braintrust_status(self) -> BraintrustStatusResponse:
        """Return whether this org is connected to Braintrust + sync state."""
        if not self._is_storage_configured():
            return BraintrustStatusResponse(connected=False)
        return self._braintrust_service().status()

    def braintrust_disconnect(self) -> None:
        """Delete the persisted Braintrust connection."""
        if not self._is_storage_configured():
            return
        self._braintrust_service().disconnect()

    def braintrust_sync(self) -> SyncBraintrustResponse:
        """Manual one-shot sync (cron-driven sync is a follow-up)."""
        if not self._is_storage_configured():
            return SyncBraintrustResponse(success=False, msg=STORAGE_NOT_CONFIGURED_MSG)
        return self._braintrust_service().sync_once()

    def _braintrust_service(self) -> "BraintrustConnectorService":
        from reflexio.server.services.braintrust.service import (
            BraintrustConnectorService,
        )

        return BraintrustConnectorService(
            storage=self._get_storage(),
            org_id=self.request_context.org_id,
        )

    def get_evaluation_overview(
        self, request: GetEvaluationOverviewRequest | dict
    ) -> GetEvaluationOverviewResponse:
        """Build the /evaluations overview payload (hero + tiles + attribution + distribution).

        Args:
            request (GetEvaluationOverviewRequest | dict): Window, bucket
                granularity, and shadow-inclusion flag.

        Returns:
            GetEvaluationOverviewResponse: Full payload — the redesigned
                /evaluations page is expected to render directly from this.
        """
        if isinstance(request, dict):
            request = GetEvaluationOverviewRequest(**request)
        if not self._is_storage_configured():
            return _empty_overview_response()
        from reflexio.server.services.evaluation_overview.service import (
            EvaluationOverviewService,
        )

        service = EvaluationOverviewService(
            storage=self._get_storage(),
            config=self.request_context.configurator.get_config(),
        )
        return service.run(request)


def _empty_overview_response() -> GetEvaluationOverviewResponse:
    """Return a default response when storage is not configured.

    Used by lib wrappers to keep the endpoint stable even before storage is
    wired up (matches the defensive pattern used by other dashboard methods).
    """
    return GetEvaluationOverviewResponse(
        hero=HeroBlock(
            state="empty",
            regular_success_rate_pp=0.0,
            shadow_success_rate_pp=None,
            delta_pp=None,
            buckets=[],
        ),
        context_tiles=ContextTile(
            success=PercentWithDelta(current=0.0, delta_pp=0.0),
            corrections=NumberWithDelta(current=0.0, delta=0.0),
            turns=NumberWithDelta(current=0.0, delta=0.0),
            escalation=PercentWithDelta(current=0.0, delta_pp=0.0),
        ),
        rule_attribution=[],
        score_distribution=ScoreDistribution(
            current_bins=[0, 0, 0, 0, 0, 0],
            baseline_bins=[0, 0, 0, 0, 0, 0],
            labels=["0", "1", "2", "3", "4", "5+"],
        ),
        braintrust_tiles=[],
    )
