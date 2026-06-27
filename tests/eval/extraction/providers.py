"""Live extraction provider for the extraction eval harness.

Builds an :class:`~tests.eval.extraction.runner.ExtractionProvider` callable
that runs a golden case's ``sessions`` through the **real** ``PlaybookExtractor``
+ ``ProfileExtractor`` (real prompts + a real ``LiteLLMClient``) and returns the
produced ``(profiles, playbooks)`` pair for the judge to score. This is on-demand
glue only (it hits a paid LLM via the gated real-run smoke); the mocked-seam unit
test exercises the same RIDM + config + extractor construction and call path
without a real API call so default CI catches provider bugs.

Call path (no storage seeding required, but storage IS required):
    Unlike the reflection / consolidation decision seams (which need no
    storage), the classic extractors route through
    ``run_resumable_extraction_agent`` -> ``ResumableExtractionAgent``, which
    *requires* ``request_context.storage`` (it writes an ``_agent_runs`` row).
    ``RequestContext(org_id=..., storage_base_dir=tmp)`` auto-wires a real
    temp SQLite store, so we satisfy that with zero seeding -- the window
    interactions are passed directly as a ``RequestInteractionDataModel``
    (we never call the storage-reading ``_get_interactions`` / ``run``).

    - playbooks = ``PlaybookExtractor.extract_playbook_entries([ridm])``
    - profiles  = ``ProfileExtractor._convert_raw_to_user_profiles(
                       _generate_raw_updates_from_sessions([ridm], existing_profiles=[]),
                       user_id=..., request_id=...)``

Pending tool calls (single-pass guarantee):
    The resumable agent only registers the async ``ask_human`` /
    ``attach_pending_info_request`` tools when ``pending_tool_calls_enabled``
    is true, which requires ``config.pending_tool_call_config.enabled``. That
    field defaults to ``False`` on the auto-wired ``Config``, so by default
    only ``finish_extraction`` is registered and the loop runs a single
    forced-tool pass -- equivalent to the old single-shot structured call. No
    extra turns, no resume chain. We therefore do NOT need to disable anything
    explicitly; the default config already yields a single extraction pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import PlaybookConfig, ProfileExtractorConfig
from reflexio.server.services.playbook.components.extractor import PlaybookExtractor
from reflexio.server.services.playbook.service import (
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.profile.components.extractor import ProfileExtractor
from reflexio.server.services.profile.service import (
    ProfileGenerationServiceConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from tests.eval.extraction.runner import ExtractionCase


_EVAL_REQUEST_ID = "eval-1"
_EVAL_USER_ID = "eval"


def _sessions_to_ridm(
    sessions: list[dict],
    *,
    user_id: str = _EVAL_USER_ID,
    request_id: str = _EVAL_REQUEST_ID,
) -> RequestInteractionDataModel:
    """Wrap a case's ``sessions`` into a single ``RequestInteractionDataModel``.

    Each ``{role, content}`` session entry becomes an ``Interaction`` with a
    1-based ``interaction_id``; all are grouped under one minimal ``Request``.
    Only the fields the extractors / prompt rendering read are populated.

    Args:
        sessions: The golden case's ``sessions`` -- ``{role, content}`` dicts.
        user_id: User id stamped on the request + interactions.
        request_id: Request id stamped on the request + interactions.

    Returns:
        One ``RequestInteractionDataModel`` carrying all interactions.
    """
    interactions = [
        Interaction(
            interaction_id=idx,
            user_id=user_id,
            request_id=request_id,
            content=str(turn.get("content", "")),
            role=str(turn.get("role", "user")),
            created_at=0,
        )
        for idx, turn in enumerate(sessions, start=1)
    ]
    request = Request(
        request_id=request_id,
        user_id=user_id,
        session_id="test_session",
        created_at=0,
        source="api",
    )
    return RequestInteractionDataModel(
        session_id=request_id,
        request=request,
        interactions=interactions,
    )


def _minimal_playbook_extractor_config() -> PlaybookConfig:
    """Minimal ``PlaybookConfig`` -- mirrors the playbook-extractor tests."""
    return PlaybookConfig(
        extractor_name="eval_playbook",
        extraction_definition_prompt="Evaluate agent quality",
    )


def _minimal_profile_extractor_config() -> ProfileExtractorConfig:
    """Minimal ``ProfileExtractorConfig`` -- mirrors the profile-extractor tests."""
    return ProfileExtractorConfig(
        extractor_name="eval_profile",
        extraction_definition_prompt="Extract user preferences",
    )


def _minimal_playbook_service_config() -> PlaybookGenerationServiceConfig:
    """Minimal playbook service config -- mirrors the extractor tests."""
    return PlaybookGenerationServiceConfig(
        agent_version="1.0.0",
        request_id=_EVAL_REQUEST_ID,
        source="api",
    )


def _minimal_profile_service_config() -> ProfileGenerationServiceConfig:
    """Minimal profile service config -- mirrors the extractor tests."""
    return ProfileGenerationServiceConfig(
        user_id=_EVAL_USER_ID,
        request_id=_EVAL_REQUEST_ID,
        source="api",
    )


def make_extraction_provider(
    *,
    llm_client: LiteLLMClient,
    request_context: RequestContext,
    agent_context: str = "",
) -> Callable[[ExtractionCase], tuple[list[Any], list[Any]]]:
    """Build a live extraction provider.

    Returns a closure that, for each case, wraps ``case["sessions"]`` into a
    single ``RequestInteractionDataModel``, constructs a real
    ``PlaybookExtractor`` + ``ProfileExtractor`` (with minimal configs), runs
    both directly over the passed interactions (no storage seeding), and
    returns the produced ``(profiles, playbooks)`` pair.

    Args:
        llm_client: The LLM client the extractors call (real or mocked seam).
        request_context: Supplies ``prompt_manager`` + ``configurator`` + the
            temp SQLite ``storage`` the resumable agent requires.
        agent_context: Agent-context fragment passed to both extractors.

    Returns:
        An ``ExtractionProvider`` callable mapping a case to
        ``(profiles, playbooks)``.
    """

    def provider(case: ExtractionCase) -> tuple[list[Any], list[Any]]:
        ridm = _sessions_to_ridm(case.get("sessions", []))

        playbook_extractor = PlaybookExtractor(
            request_context=request_context,
            llm_client=llm_client,
            extractor_config=_minimal_playbook_extractor_config(),
            service_config=_minimal_playbook_service_config(),
            agent_context=agent_context,
        )
        profile_extractor = ProfileExtractor(
            request_context=request_context,
            llm_client=llm_client,
            extractor_config=_minimal_profile_extractor_config(),
            service_config=_minimal_profile_service_config(),
            agent_context=agent_context,
        )

        playbooks = playbook_extractor.extract_playbook_entries([ridm])
        raw_profiles = profile_extractor._generate_raw_updates_from_sessions(
            request_interaction_data_models=[ridm],
            existing_profiles=[],
        )
        profiles = profile_extractor._convert_raw_to_user_profiles(
            raw_profiles=raw_profiles,
            user_id=_EVAL_USER_ID,
            request_id=_EVAL_REQUEST_ID,
            source_interaction_ids=[
                interaction.interaction_id
                for interaction in ridm.interactions
                if interaction.interaction_id
            ],
        )
        return (profiles, playbooks)

    return provider
