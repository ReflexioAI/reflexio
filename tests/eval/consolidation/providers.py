"""Live consolidation decision provider for the consolidation eval harness.

Builds a :class:`~tests.eval.consolidation.runner.DecisionProvider` callable
that runs a :class:`ConsolidationEvalCase` through the **real**
``PlaybookConsolidator`` decision step (``_consolidation_decisions``) + the
real ``playbook_consolidation`` prompt + a real ``LiteLLMClient``. This is
on-demand glue only (it hits a paid LLM via the gated real-run smoke); the
mocked-seam unit test exercises the same entity construction + seam-call path
without a real API call so CI catches provider bugs.

The provider builds minimal ``UserPlaybook`` entities from ``case.existing``
and ``case.candidate``, calls the consolidator's decision seam, and returns
the first produced :class:`ConsolidationDecision` (or an
``IndependentDecision`` no-op default when the LLM proposes nothing).

Note on ``new_id``: ``_format_playbooks_with_prefix`` labels rows by **list
position**, so the single candidate is rendered as ``[NEW-0]`` and the
returned decision's ``new_id`` will be ``"NEW-0"`` -- not
``case.candidate.new_id``. This is harmless for the eval: the harness's
``kind_for_decision`` reads only ``.kind``, never ``.new_id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.server.services.playbook.playbook_consolidator import (
    IndependentDecision,
    PlaybookConsolidator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.playbook.playbook_consolidator import (
        ConsolidationDecision,
    )
    from tests.eval.consolidation.case import ConsolidationEvalCase


def _to_user_playbook(
    *,
    user_playbook_id: int,
    content: str,
    trigger: str | None,
    rationale: str,
) -> UserPlaybook:
    """Build a minimal ``UserPlaybook`` for the consolidation prompt.

    Only the fields the consolidation prompt reads are populated; everything
    else uses entity defaults. Mirrors the reflection provider's minimal
    builder (``user_playbook_id``, ``agent_version=""``, ``request_id``,
    ``content``, ``trigger``, ``rationale``).

    Args:
        user_playbook_id: Stable integer id of the row (EXISTING rows use
            their case id; the candidate uses 0).
        content: Rule content text.
        trigger: Rule trigger (None when unscoped).
        rationale: Rule rationale text.

    Returns:
        A minimally-populated ``UserPlaybook``.
    """
    return UserPlaybook(
        user_playbook_id=user_playbook_id,
        agent_version="",
        request_id="eval",
        content=content,
        trigger=trigger,
        rationale=rationale,
    )


def make_consolidation_decision_provider(
    *,
    llm_client: LiteLLMClient,
    request_context: RequestContext,
) -> Callable[[ConsolidationEvalCase], ConsolidationDecision]:
    """Build a live consolidation decision provider.

    Returns a closure that, for each case, builds ``UserPlaybook`` entities
    for the EXISTING rows and the single NEW candidate, constructs a real
    ``PlaybookConsolidator``, runs the LLM decision seam
    (``_consolidation_decisions``), and returns the first produced
    :class:`ConsolidationDecision`. When the LLM proposes no decisions, the
    closure returns an ``IndependentDecision`` no-op default carrying the
    case's ``candidate.new_id``.

    The returned decision's ``new_id`` will be the prompt's position label
    (``"NEW-0"``) rather than ``case.candidate.new_id`` -- harmless because
    the harness reads only ``.kind`` (see module docstring).

    Args:
        llm_client: The LLM client the consolidator calls (real or mocked).
        request_context: Supplies ``prompt_manager`` (+ ``configurator``);
            the decision seam needs no storage.

    Returns:
        A ``DecisionProvider`` callable.
    """

    def provider(case: ConsolidationEvalCase) -> ConsolidationDecision:
        existing = [
            _to_user_playbook(
                user_playbook_id=e.id,
                content=e.content,
                trigger=e.trigger,
                rationale=e.rationale,
            )
            for e in case.existing
        ]
        candidate = _to_user_playbook(
            user_playbook_id=0,
            content=case.candidate.content,
            trigger=case.candidate.trigger,
            rationale=case.candidate.rationale,
        )
        consolidator = PlaybookConsolidator(request_context, llm_client)
        output = consolidator._consolidation_decisions([candidate], existing)
        if output.decisions:
            return output.decisions[0]
        return IndependentDecision(new_id=case.candidate.new_id)

    return provider
