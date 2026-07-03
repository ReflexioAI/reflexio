"""Learning-provenance route handler (extracted from api.py, Tier3 A2)."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from reflexio.models.api_schema.retriever_schema import (
    GetLearningProvenanceRequest,
    LearningProvenanceViewResponse,
    SourceUserPlaybookProvenanceView,
)
from reflexio.models.api_schema.ui.converters import (
    to_interaction_view,
    to_user_playbook_view,
)
from reflexio.models.config_schema import (
    DEFAULT_WINDOW_SIZE,
)
from reflexio.server.auth import (
    default_get_org_id,
)
from reflexio.server.cache import reflexio_cache
from reflexio.server.services.extractor_interaction_utils import (
    get_effective_source_filter,
    get_extractor_window_params,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _learning_provenance_error(
    payload: GetLearningProvenanceRequest,
    msg: str,
) -> LearningProvenanceViewResponse:
    return LearningProvenanceViewResponse(
        success=False,
        target_kind=payload.kind,
        target_id=payload.id,
        provenance_status="unavailable",
        msg=msg,
    )


def _parse_learning_target_int(
    payload: GetLearningProvenanceRequest,
) -> int | LearningProvenanceViewResponse:
    try:
        return int(payload.id)
    except ValueError:
        return _learning_provenance_error(
            payload,
            f"{payload.kind} id must be an integer",
        )


def _sort_interactions_by_time(interactions: list[Any]) -> list[Any]:
    return sorted(
        interactions,
        key=lambda i: (
            getattr(i, "created_at", 0) or 0,
            getattr(i, "interaction_id", 0),
        ),
    )


def _interaction_views(interactions: list[Any]) -> list[Any]:
    return [to_interaction_view(i) for i in _sort_interactions_by_time(interactions)]


def _effective_profile_provenance_window(
    reflexio: Any, trigger_source: str | None
) -> tuple[int, list[str] | None]:
    config = reflexio.request_context.configurator.get_config()
    profile_config = getattr(config, "profile_extractor_config", None)
    if profile_config is None:
        return getattr(config, "window_size", DEFAULT_WINDOW_SIZE), None

    window_size, _ = get_extractor_window_params(
        profile_config,
        getattr(config, "window_size", None),
        getattr(config, "stride_size", None),
    )
    should_skip, source_filter = get_effective_source_filter(
        profile_config,
        trigger_source,
    )
    if should_skip:
        return window_size, None
    return window_size, source_filter


def _get_profile_learning_provenance(
    payload: GetLearningProvenanceRequest,
    reflexio: Any,
) -> LearningProvenanceViewResponse:
    storage = reflexio.request_context.storage
    profile = storage.get_profile_by_id(payload.id)
    if profile is None:
        return _learning_provenance_error(payload, "Profile not found")

    if profile.source_interaction_ids:
        interactions = storage.get_interactions_by_ids(profile.source_interaction_ids)
        return LearningProvenanceViewResponse(
            success=True,
            target_kind=payload.kind,
            target_id=payload.id,
            provenance_status="exact",
            trigger_request_id=profile.generated_from_request_id,
            interactions=_interaction_views(interactions),
        )

    if not profile.generated_from_request_id:
        return _learning_provenance_error(
            payload,
            "Profile has no generation request for provenance reconstruction",
        )

    trigger_request = storage.get_request(profile.generated_from_request_id)
    if trigger_request is None:
        return _learning_provenance_error(
            payload,
            "Profile generation request was not found",
        )

    trigger_interactions = storage.get_interactions_by_request_ids(
        [profile.generated_from_request_id]
    )
    anchor_time = (
        max(i.created_at for i in trigger_interactions)
        if trigger_interactions
        else trigger_request.created_at
    )
    window_size, source_filter = _effective_profile_provenance_window(
        reflexio,
        trigger_request.source,
    )
    _, interactions = storage.get_last_k_interactions_grouped(
        user_id=profile.user_id,
        k=window_size,
        sources=source_filter,
        end_time=anchor_time,
    )
    return LearningProvenanceViewResponse(
        success=True,
        target_kind=payload.kind,
        target_id=payload.id,
        provenance_status="best_effort" if interactions else "unavailable",
        trigger_request_id=profile.generated_from_request_id,
        interactions=_interaction_views(interactions),
        msg=None if interactions else "No interactions found for reconstructed window",
    )


def _get_user_playbook_learning_provenance(
    payload: GetLearningProvenanceRequest,
    reflexio: Any,
) -> LearningProvenanceViewResponse:
    parsed_id = _parse_learning_target_int(payload)
    if isinstance(parsed_id, LearningProvenanceViewResponse):
        return parsed_id

    storage = reflexio.request_context.storage
    playbook = storage.get_user_playbook_by_id(parsed_id)
    if playbook is None:
        return _learning_provenance_error(payload, "User playbook not found")

    status = "exact"
    if playbook.source_interaction_ids:
        interactions = storage.get_interactions_by_ids(playbook.source_interaction_ids)
    elif playbook.request_id:
        status = "best_effort"
        interactions = storage.get_interactions_by_request_ids([playbook.request_id])
    else:
        status = "unavailable"
        interactions = []

    return LearningProvenanceViewResponse(
        success=True,
        target_kind=payload.kind,
        target_id=payload.id,
        provenance_status=status if interactions else "unavailable",
        trigger_request_id=playbook.request_id,
        interactions=_interaction_views(interactions),
        msg=None if interactions else "No interactions found for this user playbook",
    )


def _get_agent_playbook_learning_provenance(
    payload: GetLearningProvenanceRequest,
    reflexio: Any,
) -> LearningProvenanceViewResponse:
    parsed_id = _parse_learning_target_int(payload)
    if isinstance(parsed_id, LearningProvenanceViewResponse):
        return parsed_id

    storage = reflexio.request_context.storage
    agent_playbook = storage.get_agent_playbook_by_id(parsed_id)
    if agent_playbook is None:
        return _learning_provenance_error(payload, "Agent playbook not found")

    windows = storage.get_source_windows_for_agent_playbook(parsed_id)
    if not windows:
        return LearningProvenanceViewResponse(
            success=True,
            target_kind=payload.kind,
            target_id=payload.id,
            provenance_status="unavailable",
            msg="No source user playbooks are recorded for this agent playbook",
        )

    user_playbook_ids = [window.user_playbook_id for window in windows]
    user_playbooks = storage.get_user_playbooks_by_ids_any_user(
        user_playbook_ids,
        status_filter=None,
    )
    user_playbooks_by_id = {
        playbook.user_playbook_id: playbook for playbook in user_playbooks
    }

    interaction_ids_to_fetch: set[int] = set()
    request_ids_to_fetch: set[str] = set()
    group_sources: dict[int, tuple[list[int], str | None, bool]] = {}
    for window in windows:
        source_user_playbook = user_playbooks_by_id.get(window.user_playbook_id)
        if source_user_playbook is None:
            continue

        source_ids = list(window.source_interaction_ids)
        request_id: str | None = None
        uses_fallback = False
        if not source_ids and source_user_playbook.source_interaction_ids:
            source_ids = list(source_user_playbook.source_interaction_ids)
        elif not source_ids and source_user_playbook.request_id:
            request_id = source_user_playbook.request_id
            uses_fallback = True
        elif not source_ids:
            uses_fallback = True

        interaction_ids_to_fetch.update(source_ids)
        if request_id:
            request_ids_to_fetch.add(request_id)
        group_sources[window.user_playbook_id] = (source_ids, request_id, uses_fallback)

    interactions_by_id = (
        {
            interaction.interaction_id: interaction
            for interaction in storage.get_interactions_by_ids(
                sorted(interaction_ids_to_fetch)
            )
        }
        if interaction_ids_to_fetch
        else {}
    )
    request_interactions: dict[str, list[Any]] = {}
    if request_ids_to_fetch:
        for interaction in storage.get_interactions_by_request_ids(
            sorted(request_ids_to_fetch)
        ):
            request_interactions.setdefault(interaction.request_id, []).append(
                interaction
            )

    groups: list[SourceUserPlaybookProvenanceView] = []
    used_fallback = False
    found_interactions = False
    for window in windows:
        source_user_playbook = user_playbooks_by_id.get(window.user_playbook_id)
        if source_user_playbook is None:
            used_fallback = True
            continue

        source_ids, request_id, uses_fallback = group_sources.get(
            window.user_playbook_id,
            ([], None, True),
        )
        used_fallback = used_fallback or uses_fallback
        interactions = (
            [interactions_by_id[i] for i in source_ids if i in interactions_by_id]
            if source_ids
            else request_interactions.get(request_id or "", [])
        )

        found_interactions = found_interactions or bool(interactions)
        groups.append(
            SourceUserPlaybookProvenanceView(
                user_playbook=to_user_playbook_view(source_user_playbook),
                interactions=_interaction_views(interactions),
                source_interaction_ids=source_ids,
            )
        )

    if not groups:
        return _learning_provenance_error(
            payload,
            "Recorded source user playbooks were not found",
        )

    return LearningProvenanceViewResponse(
        success=True,
        target_kind=payload.kind,
        target_id=payload.id,
        provenance_status=(
            "exact"
            if found_interactions and not used_fallback
            else "best_effort"
            if found_interactions
            else "unavailable"
        ),
        source_user_playbooks=groups,
        msg=None if found_interactions else "No source interactions were found",
    )


@router.post(
    "/api/get_learning_provenance",
    response_model=LearningProvenanceViewResponse,
    response_model_exclude_none=True,
)
def get_learning_provenance(
    payload: GetLearningProvenanceRequest,
    org_id: str = Depends(default_get_org_id),
) -> LearningProvenanceViewResponse:
    """Return read-only learning provenance for a generated learning row.

    Raises:
        HTTPException: 503 when storage is not configured (matches the sibling
            route convention, e.g. evaluation.py / stall_state_api.py).
    """
    reflexio = reflexio_cache.get_reflexio(org_id=org_id)
    if reflexio.request_context.storage is None:
        raise HTTPException(status_code=503, detail="Storage not configured")
    if payload.kind == "profile":
        return _get_profile_learning_provenance(payload, reflexio)
    if payload.kind == "user_playbook":
        return _get_user_playbook_learning_provenance(payload, reflexio)
    return _get_agent_playbook_learning_provenance(payload, reflexio)
