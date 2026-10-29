"""AI-judge for consolidation decisions.

Wraps a ``LiteLLMClient``-style client (anything exposing
``generate_chat_response(messages,
response_format, model)``) and asks an LLM whether a *produced*
consolidation decision matches the *intended* outcome captured by a case's
``gold_kind`` and source observations, including contradictions across the
complete output memory.

No human gold labels are required at scoring time — the LLM judge is the
oracle, and the harness metric is agreement with the judge. A panel of N
judges (default 1) can be used; the panel ``correct`` verdict is decided by
majority of the ``correct`` votes (ties resolved as incorrect, the
conservative choice for a regression detector). The ``self_contradiction``
flag aggregates differently: a tie resolves as ``True`` (conservatively
flag a possible bad merge for review).

The model is never hardcoded — it comes from the rubric (``judge_model``)
and is forwarded to the client, matching the existing judge.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tests.eval.consolidation.case import ConsolidationEvalCase, kind_for_decision

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.playbook.components.reconciliation import Decisions


class ConsolidationVerdict(BaseModel):
    """A single judge's verdict for one (case, produced_decision) pair.

    Attributes:
        correct: True iff the judge believes the produced decision matches
            the case's intended outcome.
        self_contradiction: Whether the complete proposed memory contains
            contradictory advice for the same situation or collapses
            distinct do/avoid rules.
        reason: One-sentence justification, logged / surfaced in reports.
    """

    correct: bool
    self_contradiction: bool = False
    reason: str = ""


_DEFAULT_PROMPT = (
    "Judge the complete audited playbook consolidation batch against the supplied "
    "source messages and intended memory. Create adds grounded instructions; "
    "Replace can split or combine existing instructions into multiple outputs; "
    "Remove consumes unsupported, obsolete or redundant inputs. Unmentioned "
    "existing rows remain. Do not infer semantic correctness merely from an "
    "action name. Check omissions, unsupported scope, and contradictions across "
    "the entire resulting memory. Distinct do/avoid rules can coexist when their "
    "conditions differ; opposite advice for the same condition is a contradiction.\n"
    "Agent context: {agent_context}\nExisting: {existing}\nCandidate: {candidate}\n"
    "Produced batch: {produced_decision}\nProduced action summary: {produced_kind}\n"
    "Expected action summary: {gold_kind}\nIntended behavior: {notes}\n"
    "Return correct, self_contradiction, and reason."
)


def _render_prompt(
    *,
    case: ConsolidationEvalCase,
    produced_decision: Decisions,
    produced_kind: str,
    template: str,
) -> str:
    """Substitute case + decision context into the rubric template.

    The EXISTING list is serialized as a JSON array (each row dumped via
    its model), and the candidate + produced decision via ``model_dump_json``.
    """
    existing_json = json.dumps(
        [
            {**row.model_dump(mode="json", exclude={"id"}), "id": f"EXISTING-{index}"}
            for index, row in enumerate(case.existing)
        ]
    )
    candidate_json = json.dumps(
        {**case.candidate.model_dump(mode="json", exclude={"new_id"}), "id": "NEW-0"}
    )
    return (
        template.replace("{agent_context}", case.agent_context)
        .replace("{gold_kind}", case.gold_kind)
        .replace("{notes}", case.notes or "")
        .replace("{existing}", existing_json)
        .replace("{candidate}", candidate_json)
        .replace("{produced_decision}", produced_decision.model_dump_json())
        .replace("{produced_kind}", produced_kind)
    )


def judge_consolidation_decision(
    *,
    case: ConsolidationEvalCase,
    produced_decision: Decisions,
    llm_client: LiteLLMClient | Any,
    rubric: dict[str, Any] | None = None,
    panel_size: int = 1,
) -> ConsolidationVerdict:
    """Ask an LLM panel whether ``produced_decision`` matches the gold intent.

    Args:
        case: The consolidation eval case (supplies gold intent + context).
        produced_decision: The decision produced by the consolidation
            service / LLM for this case's candidate.
        llm_client: A ``LiteLLMClient`` (or ``MagicMock`` in tests)
            exposing ``generate_chat_response``.
        rubric: Optional rubric dict with ``prompt`` and ``judge_model``
            keys. Falls back to a built-in prompt and the client's default
            model.
        panel_size: Number of independent judge calls. Default 1. For
            ``N > 1`` the ``correct`` verdict is the majority of ``correct``
            votes with a tie resolved as ``False`` (conservative for a
            regression detector); ``self_contradiction`` is the majority of
            the contradiction votes with a tie resolved as ``True``
            (conservatively flag a possible bad merge for review).

    Returns:
        A :class:`ConsolidationVerdict`. For a panel, ``reason`` aggregates
        the vote tally.

    Raises:
        TypeError: When the client returns something other than a
            structured verdict (misconfigured ``response_format``).
        ValueError: When ``panel_size < 1``.
    """
    if panel_size < 1:
        raise ValueError("panel_size must be >= 1")

    rubric = rubric or {}
    template = rubric.get("prompt", _DEFAULT_PROMPT)
    judge_model = rubric.get("judge_model")
    produced_kind = kind_for_decision(produced_decision)
    prompt = _render_prompt(
        case=case,
        produced_decision=produced_decision,
        produced_kind=produced_kind,
        template=template,
    )

    prompt += "\nOriginal source sessions: " + json.dumps(
        [session.model_dump(mode="json") for session in case.sessions]
    )

    votes: list[ConsolidationVerdict] = []
    for _ in range(panel_size):
        result = llm_client.generate_chat_response(
            messages=[{"role": "user", "content": prompt}],
            response_format=ConsolidationVerdict,
            model=judge_model,
        )
        votes.append(_coerce_verdict(result))

    if panel_size == 1:
        return votes[0]

    yes = sum(1 for v in votes if v.correct)
    no = panel_size - yes
    majority_correct = yes > no  # tie -> False (conservative)

    contra_yes = sum(1 for v in votes if v.self_contradiction)
    contra_no = panel_size - contra_yes
    # tie -> True (conservatively flag a possible bad merge for review).
    majority_contradiction = contra_yes >= contra_no
    return ConsolidationVerdict(
        correct=majority_correct,
        self_contradiction=majority_contradiction,
        reason=f"panel {yes}/{panel_size} correct, "
        f"{contra_yes}/{panel_size} self-contradiction; "
        + "; ".join(v.reason for v in votes if v.reason),
    )


def _coerce_verdict(result: Any) -> ConsolidationVerdict:
    """Coerce a client response into a :class:`ConsolidationVerdict`.

    Raises:
        TypeError: When the response is not a verdict-shaped model.
    """
    if isinstance(result, ConsolidationVerdict):
        return result
    if isinstance(result, BaseModel):
        return ConsolidationVerdict.model_validate(result.model_dump())
    raise TypeError(
        f"judge_consolidation_decision expected ConsolidationVerdict, "
        f"got {type(result).__name__}"
    )
