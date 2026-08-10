"""Reconstruct playbook-review chronology from persisted interaction provenance."""

from __future__ import annotations

from collections.abc import Sequence

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.services.playbook.components.reviewer import (
    PlaybookCandidateEvidenceError,
)
from reflexio.server.services.storage.storage_base import BaseStorage


class PlaybookReviewWindowError(PlaybookCandidateEvidenceError):
    """Persisted review provenance is incomplete or crosses a user boundary."""


def infer_playbook_review_user_id(
    *,
    storage: BaseStorage,
    source_interaction_ids: Sequence[int],
    subject: str,
) -> str:
    """Infer one legacy run owner from complete persisted source evidence.

    New playbook runs always persist ``user_id``. This compatibility path is
    only for older nullable rows and remains fail-closed unless every cited
    interaction and request exists and names the same non-empty owner.
    """
    source_ids = list(dict.fromkeys(source_interaction_ids))
    if not source_ids:
        raise PlaybookReviewWindowError(
            f"{subject} has no complete generation-window provenance"
        )

    interactions_by_id = {
        interaction.interaction_id: interaction
        for interaction in storage.get_interactions_by_ids(source_ids)
    }
    missing_interaction_ids = [
        interaction_id
        for interaction_id in source_ids
        if interaction_id not in interactions_by_id
    ]
    if missing_interaction_ids:
        raise PlaybookReviewWindowError(
            f"{subject} is missing persisted generation-window interactions: "
            f"{missing_interaction_ids}"
        )

    interaction_owners = {
        interaction.user_id.strip()
        for interaction in interactions_by_id.values()
        if interaction.user_id and interaction.user_id.strip()
    }
    if len(interaction_owners) != 1 or any(
        not interaction.user_id or not interaction.user_id.strip()
        for interaction in interactions_by_id.values()
    ):
        raise PlaybookReviewWindowError(
            f"{subject} does not have one unambiguous interaction owner"
        )
    user_id = next(iter(interaction_owners))

    for request_id in {
        interaction.request_id for interaction in interactions_by_id.values()
    }:
        request = storage.get_request(request_id)
        if request is None:
            raise PlaybookReviewWindowError(
                f"{subject} is missing persisted source request {request_id}"
            )
        if not request.user_id or request.user_id.strip() != user_id:
            raise PlaybookReviewWindowError(
                f"{subject} has source request {request_id} owned by another user"
            )

    return user_id


def reconstruct_playbook_review_window(
    *,
    storage: BaseStorage,
    source_interaction_ids: Sequence[int],
    user_id: str,
    subject: str,
) -> list[RequestInteractionDataModel]:
    """Load and validate one exact persisted interaction window.

    Args:
        storage: Storage containing the persisted requests and interactions.
        source_interaction_ids: Exact interaction IDs recorded by extraction,
            optionally extended with candidate-cited evidence IDs.
        user_id: Owner every interaction and request must match.
        subject: Safe identifier used in fail-closed error messages.

    Returns:
        Request groups ordered by their earliest interaction, with interactions
        inside each request ordered chronologically.

    Raises:
        PlaybookReviewWindowError: If any interaction/request is missing or is
            owned by another user.
    """
    source_ids = list(dict.fromkeys(source_interaction_ids))
    if not source_ids:
        raise PlaybookReviewWindowError(
            f"{subject} has no complete generation-window provenance"
        )

    interactions_by_id = {
        interaction.interaction_id: interaction
        for interaction in storage.get_interactions_by_ids(source_ids)
    }
    missing_interaction_ids = [
        interaction_id
        for interaction_id in source_ids
        if interaction_id not in interactions_by_id
    ]
    if missing_interaction_ids:
        raise PlaybookReviewWindowError(
            f"{subject} is missing persisted generation-window interactions: "
            f"{missing_interaction_ids}"
        )

    interactions_by_request: dict[str, list[Interaction]] = {}
    for interaction in interactions_by_id.values():
        if interaction.user_id != user_id:
            raise PlaybookReviewWindowError(
                f"{subject} has source interaction {interaction.interaction_id} "
                "owned by another user"
            )
        interactions_by_request.setdefault(interaction.request_id, []).append(
            interaction
        )

    review_window: list[RequestInteractionDataModel] = []
    for request_id, interactions in interactions_by_request.items():
        request = storage.get_request(request_id)
        if request is None:
            raise PlaybookReviewWindowError(
                f"{subject} is missing persisted source request {request_id}"
            )
        if request.user_id != user_id:
            raise PlaybookReviewWindowError(
                f"{subject} has source request {request_id} owned by another user"
            )
        review_window.append(
            RequestInteractionDataModel(
                session_id=request.session_id,
                request=request,
                interactions=sorted(
                    interactions,
                    key=lambda item: (item.created_at, item.interaction_id),
                ),
            )
        )

    review_window.sort(
        key=lambda group: (
            group.interactions[0].created_at,
            group.interactions[0].interaction_id,
        )
    )
    return review_window
