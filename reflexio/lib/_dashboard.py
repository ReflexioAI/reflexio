from reflexio.lib._base import STORAGE_NOT_CONFIGURED_MSG, ReflexioBase
from reflexio.models.api_schema.retriever_schema import (
    DashboardStats,
    GetDashboardStatsRequest,
    GetDashboardStatsResponse,
    PeriodStats,
    TimeSeriesDataPoint,
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
