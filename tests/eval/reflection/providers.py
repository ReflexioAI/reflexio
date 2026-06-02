"""Live reflection decision provider for the reflection eval harness.

Builds a :class:`~tests.eval.reflection.runner.DecisionProvider` callable
that runs a :class:`ReflectionEvalCase` through the **real**
``ReflectionExtractor`` + the real ``memory_reflection`` prompt + a real
``LiteLLMClient``. This is on-demand glue only (it hits a paid LLM via the
gated real-run smoke); the mocked-seam unit test exercises the same
construction + call path without a real API call so CI catches provider
bugs.

The provider builds minimal cited entities from ``case.cited_item`` and
returns the first produced ``ReflectionDecision`` (or a no-op
``ReflectionDecision()`` -> ``no_change`` when the LLM proposes nothing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reflexio.models.api_schema.domain.entities import UserPlaybook, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.reflection.reflection_extractor import (
    ReflectionExtractor,
)
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from tests.eval.reflection.case import ReflectionEvalCase


def _cited_entities(
    case: ReflectionEvalCase,
) -> tuple[list[UserProfile], list[UserPlaybook]]:
    """Build the cited entity list for a case from its ``cited_item``.

    A playbook citation yields ``([], [UserPlaybook])``; a profile
    citation yields ``([UserProfile], [])``. Only the fields the reflection
    prompt reads are populated; everything else uses entity defaults.

    Args:
        case: The eval case whose ``cited_item`` is being judged.

    Returns:
        ``(cited_profiles, cited_user_playbooks)`` for ``ReflectionExtractor.run``.
    """
    item = case.cited_item
    if item.kind == "playbook":
        target_id = item.target_id
        playbook = UserPlaybook(
            user_playbook_id=int(target_id) if target_id.isdigit() else 0,
            agent_version="",
            request_id="eval",
            content=item.content,
            trigger=item.trigger,
            rationale="",
        )
        return ([], [playbook])

    ttl = (
        ProfileTimeToLive(item.profile_time_to_live)
        if item.profile_time_to_live
        else ProfileTimeToLive.INFINITY
    )
    profile = UserProfile(
        profile_id=item.target_id,
        user_id="eval",
        content=item.content,
        last_modified_timestamp=0,
        generated_from_request_id="eval",
        profile_time_to_live=ttl,
    )
    return ([profile], [])


def make_reflection_decision_provider(
    *,
    llm_client: LiteLLMClient,
    request_context: RequestContext,
    agent_context: str = "",
) -> Callable[[ReflectionEvalCase], ReflectionDecision]:
    """Build a live reflection decision provider.

    Returns a closure that, for each case, constructs a real
    ``ReflectionExtractor``, builds the cited entity, runs the reflection
    LLM call, and returns the first produced ``ReflectionDecision`` (or a
    no-op ``ReflectionDecision()`` mapping to ``no_change`` when the LLM
    proposes nothing).

    Args:
        llm_client: The LLM client the extractor calls (real or mocked).
        request_context: Supplies ``prompt_manager`` + ``configurator``.
        agent_context: Fallback agent-context fragment; a case's own
            ``agent_context`` wins when set.

    Returns:
        A ``DecisionProvider`` callable.
    """

    def provider(case: ReflectionEvalCase) -> ReflectionDecision:
        extractor = ReflectionExtractor(
            request_context,
            llm_client,
            case.agent_context or agent_context,
            None,
        )
        cited_profiles, cited_user_playbooks = _cited_entities(case)
        output = extractor.run(
            window_interactions=case.window,
            cited_profiles=cited_profiles,
            cited_user_playbooks=cited_user_playbooks,
            horizon_by_key=None,
        )
        return output.decisions[0] if output.decisions else _no_op_decision(case)

    return provider


def _no_op_decision(case: ReflectionEvalCase) -> ReflectionDecision:
    """Build a no-op decision (no revision fields) for a case.

    Maps to ``no_change`` via ``label_for_decision``. ``target_kind`` /
    ``target_id`` are required on ``ReflectionDecision`` so they are
    copied from the cited item.
    """
    item = case.cited_item
    return ReflectionDecision(target_kind=item.kind, target_id=item.target_id)
