"""Session outcome facade."""

import logging
from collections.abc import Callable
from time import time

from reflexio.lib._base import ReflexioBase, _require_storage
from reflexio.models.api_schema.common import sanitise_for_log
from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    GetSessionOutcomesResponse,
    SessionOutcomeFailureReason,
    SetSessionOutcomeRequest,
    SetSessionOutcomeResponse,
)
from reflexio.server.extensions import ServiceKey, get_service

type SessionOutcomeAcceptance = Callable[
    [str, SetSessionOutcomeRequest, int, str], SessionOutcomeFailureReason | None
]

SESSION_OUTCOME_ACCEPTANCE = ServiceKey[SessionOutcomeAcceptance](
    "session_outcome_acceptance"
)

logger = logging.getLogger(__name__)


class SessionOutcomeMixin(ReflexioBase):
    @_require_storage(SetSessionOutcomeResponse)
    def mark_session_outcome(
        self, request: SetSessionOutcomeRequest | dict
    ) -> SetSessionOutcomeResponse:
        if isinstance(request, dict):
            request = SetSessionOutcomeRequest(**request)
        if unknown_fields := request.unknown_field_names():
            shown_fields = [sanitise_for_log(name) for name in unknown_fields[:5]]
            if remaining := len(unknown_fields) - len(shown_fields):
                shown_fields.append(f"+{remaining} more")
            logger.warning(
                "Session outcome payload contained stripped unknown fields: %s",
                ", ".join(shown_fields),
            )
        received_at = int(time())
        storage = self._get_storage()
        if request.occurred_at > received_at + 86400:
            return SetSessionOutcomeResponse(
                success=False,
                reason=SessionOutcomeFailureReason.OCCURRED_IN_FUTURE,
                message="Outcome occurred too far in the future",
            )

        try:
            for _attempt in range(3):
                context = storage.get_session_outcome_context(request.session_id)
                if not context.existing:
                    if context.user_id is None or context.first_request_at is None:
                        return SetSessionOutcomeResponse(
                            success=False,
                            reason=SessionOutcomeFailureReason.UNKNOWN_SESSION,
                            message="Session has no published requests",
                        )
                    if context.user_contract_violation:
                        logger.warning(
                            "Session outcome contract violation: multiple users for session %s",
                            sanitise_for_log(request.session_id),
                        )
                    if context.source_contract_violation:
                        logger.warning(
                            "Session outcome contract violation: multiple sources for session %s",
                            sanitise_for_log(request.session_id),
                        )
                    if request.occurred_at < context.first_request_at:
                        return SetSessionOutcomeResponse(
                            success=False,
                            reason=SessionOutcomeFailureReason.OCCURRED_BEFORE_SESSION,
                            message="Outcome occurred before the session began",
                            user_id=context.user_id,
                            source=context.source,
                        )
                    provider = get_service(SESSION_OUTCOME_ACCEPTANCE)
                    if provider is not None:
                        reason = provider(
                            self.org_id, request, received_at, context.user_id
                        )
                        if reason is not None:
                            return SetSessionOutcomeResponse(
                                success=False,
                                reason=reason,
                                message="Outcome was not accepted",
                                user_id=context.user_id,
                                source=context.source,
                            )
                result = storage.record_session_outcome(
                    request,
                    created_at=received_at,
                    expected_context=context,
                )
                if result.context_changed:
                    continue
                if result.reason is not None:
                    return SetSessionOutcomeResponse(
                        success=False,
                        reason=result.reason,
                        message="Outcome was not recorded",
                        user_id=result.user_id,
                        source=result.source,
                        outcome_id=result.outcome_id,
                        outcome_revision=result.outcome_revision,
                        outcome_contract_digest=result.outcome_contract_digest,
                        finalized_trajectory_digest=result.finalized_trajectory_digest,
                    )
                return SetSessionOutcomeResponse(
                    success=True,
                    recorded=result.recorded,
                    user_id=result.user_id,
                    source=result.source,
                    message=(
                        "Outcome recorded"
                        if result.recorded
                        else "Outcome already exists"
                    ),
                    outcome_id=result.outcome_id,
                    outcome_revision=result.outcome_revision,
                    outcome_contract_digest=result.outcome_contract_digest,
                    finalized_trajectory_digest=result.finalized_trajectory_digest,
                )
        except Exception:
            logger.exception("Failed to record session outcome")
            return SetSessionOutcomeResponse(
                success=False,
                reason=SessionOutcomeFailureReason.STORAGE_ERROR,
                message="Outcome was not recorded",
            )
        return SetSessionOutcomeResponse(
            success=False,
            reason=SessionOutcomeFailureReason.STORAGE_ERROR,
            message="Session changed repeatedly while recording outcome",
        )

    @_require_storage(GetSessionOutcomesResponse)
    def get_session_outcomes(
        self, request: GetSessionOutcomesRequest | dict
    ) -> GetSessionOutcomesResponse:
        if isinstance(request, dict):
            request = GetSessionOutcomesRequest(**request)
        return GetSessionOutcomesResponse(
            success=True,
            session_outcomes=self._get_storage().get_session_outcomes(request),
        )
