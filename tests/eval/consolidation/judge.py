"""AI-judge for consolidation decisions.

Mirrors ``tests/eval/reflection/judge.py``: wraps a ``LiteLLMClient``-style
client (anything exposing ``generate_chat_response(messages,
response_format, model)``) and asks an LLM whether a *produced*
consolidation decision matches the *intended* outcome captured by a case's
``gold_kind`` — and, for a ``unify``, whether the merge introduced a
self-contradiction.

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
    from reflexio.server.services.playbook.playbook_consolidator import (
        ConsolidationDecision,
    )


class ConsolidationVerdict(BaseModel):
    """A single judge's verdict for one (case, produced_decision) pair.

    Attributes:
        correct: True iff the judge believes the produced decision matches
            the case's intended outcome.
        self_contradiction: True iff the produced decision is a ``unify``
            whose merged content contradicts itself on the *same* situation
            OR collapses distinct do/avoid rules. For non-``unify``
            decisions this is always False.
        reason: One-sentence justification, logged / surfaced in reports.
    """

    correct: bool
    self_contradiction: bool = False
    reason: str = ""


_DEFAULT_PROMPT = (
    "You are a strict playbook-consolidation judge. A consolidation step "
    "looked at a NEW candidate rule and the EXISTING rules search surfaced, "
    "then produced one decision of exactly one of these four kinds:\n\n"
    "  - unify: collapse the candidate (and 0..N existing rows) into one "
    "row. A unified skill MAY hold mixed-polarity rules (a do-rule and an "
    "avoid-rule) but ONLY when they govern DIFFERENT sub-aspects of the "
    "same task. The compose MUST preserve the distinct do/avoid rules.\n"
    "  - reject_new: the candidate is redundant; an existing row supersedes "
    "it (a storage no-op).\n"
    "  - differentiate: both rules are valid but in distinct contexts; "
    "refine both triggers so they no longer overlap.\n"
    "  - independent: the candidate is unrelated to any existing row; insert "
    "it as-is.\n\n"
    "OPTION-B CONTRACT (critical): a unify MAY hold mixed-polarity rules for "
    "DIFFERENT sub-aspects, but it MUST NOT merge rules that contradict on "
    "the SAME situation (give opposite advice for the same trigger). Two "
    "rules that contradict on the same situation belong in a differentiate "
    "(distinct contexts) or a reject_new (one supersedes), NEVER a unify. A "
    "unify that merges same-situation contradictions, or that collapses two "
    "genuinely distinct do/avoid rules into one muddled rule, is a "
    "SELF-CONTRADICTION.\n\n"
    "Agent context: {agent_context}\n"
    "EXISTING rows (json): {existing}\n"
    "NEW candidate (json): {candidate}\n"
    "Produced decision (json): {produced_decision}\n"
    "Produced decision kind: {produced_kind}\n\n"
    "The INTENDED outcome kind for this case is: {gold_kind}\n"
    "Case notes: {notes}\n\n"
    "Decide two things:\n"
    "  1. correct: does the produced decision's kind and content match the "
    "intended outcome?\n"
    "  2. self_contradiction: ONLY for a unify, did the merge violate the "
    "Option-B contract (merge same-situation contradictions, or collapse "
    "distinct do/avoid rules)? For any non-unify decision return false.\n\n"
    'Respond ONLY with JSON: {"correct": bool, "self_contradiction": bool, '
    '"reason": str}'
)


def _render_prompt(
    *,
    case: ConsolidationEvalCase,
    produced_decision: ConsolidationDecision,
    produced_kind: str,
    template: str,
) -> str:
    """Substitute case + decision context into the rubric template.

    The EXISTING list is serialized as a JSON array (each row dumped via
    its model), and the candidate + produced decision via ``model_dump_json``.
    """
    existing_json = json.dumps([e.model_dump() for e in case.existing])
    return (
        template.replace("{agent_context}", case.agent_context)
        .replace("{gold_kind}", case.gold_kind)
        .replace("{notes}", case.notes or "")
        .replace("{existing}", existing_json)
        .replace("{candidate}", case.candidate.model_dump_json())
        .replace("{produced_decision}", produced_decision.model_dump_json())
        .replace("{produced_kind}", produced_kind)
    )


def judge_consolidation_decision(
    *,
    case: ConsolidationEvalCase,
    produced_decision: ConsolidationDecision,
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
