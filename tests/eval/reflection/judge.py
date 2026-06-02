"""AI-judge for reflection decisions.

Mirrors ``tests/eval/judge.py``: wraps a ``LiteLLMClient``-style client
(anything exposing ``generate_chat_response(messages, response_format,
model)``) and asks an LLM whether a *produced* reflection decision
matches the *intended* outcome captured by a case's ``gold_label``.

No human gold labels are required at scoring time — the LLM judge is the
oracle, and the harness metric is agreement with the judge. A panel of N
judges (default 1) can be used; the panel verdict is decided by majority
of the ``correct`` votes (ties resolved as incorrect, the conservative
choice for a regression detector).

The model is never hardcoded — it comes from the rubric (``judge_model``)
and is forwarded to the client, matching the existing judge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from tests.eval.reflection.case import ReflectionEvalCase, label_for_decision

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.reflection.reflection_service_utils import (
        ReflectionDecision,
    )


class ReflectionVerdict(BaseModel):
    """A single judge's verdict for one (case, produced_decision) pair.

    Attributes:
        correct: True iff the judge believes the produced decision
            matches the case's intended outcome.
        reason: One-sentence justification, logged / surfaced in reports.
    """

    correct: bool
    reason: str = ""


def _render_prompt(
    *,
    case: ReflectionEvalCase,
    produced_decision: ReflectionDecision,
    produced_label: str,
    template: str,
) -> str:
    """Substitute case + decision context into the rubric template."""
    return (
        template.replace("{agent_context}", case.agent_context)
        .replace("{gold_label}", case.gold_label)
        .replace("{gold_new_trigger}", case.gold_new_trigger or "")
        .replace("{notes}", case.notes or "")
        .replace("{cited_item}", case.cited_item.model_dump_json())
        .replace("{produced_decision}", produced_decision.model_dump_json())
        .replace("{produced_label}", produced_label)
    )


_DEFAULT_PROMPT = (
    "You are a strict reflection-decision judge. A reflection step looked at a "
    "cited memory item and produced a revision decision.\n\n"
    "Agent context: {agent_context}\n"
    "Cited item: {cited_item}\n"
    "Produced decision: {produced_decision}\n"
    "Mechanically-derived label for the produced decision: {produced_label}\n\n"
    "The INTENDED outcome for this case is: {gold_label}\n"
    "Expected new trigger (if any): {gold_new_trigger}\n"
    "Case notes: {notes}\n\n"
    "Decide whether the produced decision matches the intended outcome. "
    'Respond ONLY with JSON: {"correct": bool, "reason": str}'
)


def judge_reflection_decision(
    *,
    case: ReflectionEvalCase,
    produced_decision: ReflectionDecision,
    llm_client: LiteLLMClient | Any,
    rubric: dict[str, Any] | None = None,
    panel_size: int = 1,
) -> ReflectionVerdict:
    """Ask an LLM panel whether ``produced_decision`` matches the gold intent.

    Args:
        case: The reflection eval case (supplies gold intent + context).
        produced_decision: The decision produced by the reflection
            service / LLM for this case's cited item.
        llm_client: A ``LiteLLMClient`` (or ``MagicMock`` in tests)
            exposing ``generate_chat_response``.
        rubric: Optional rubric dict with ``prompt`` and ``judge_model``
            keys. Falls back to a built-in prompt and the client's
            default model.
        panel_size: Number of independent judge calls. Default 1. For
            ``N > 1`` the verdict is the majority of ``correct`` votes;
            a tie is resolved as ``correct=False`` (conservative for a
            regression detector).

    Returns:
        A :class:`ReflectionVerdict`. For a panel, ``reason`` aggregates
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
    produced_label = label_for_decision(produced_decision, case.cited_item)
    prompt = _render_prompt(
        case=case,
        produced_decision=produced_decision,
        produced_label=produced_label,
        template=template,
    )

    votes: list[ReflectionVerdict] = []
    for _ in range(panel_size):
        result = llm_client.generate_chat_response(
            messages=[{"role": "user", "content": prompt}],
            response_format=ReflectionVerdict,
            model=judge_model,
        )
        votes.append(_coerce_verdict(result))

    if panel_size == 1:
        return votes[0]

    yes = sum(1 for v in votes if v.correct)
    no = panel_size - yes
    majority_correct = yes > no  # tie -> False (conservative)
    return ReflectionVerdict(
        correct=majority_correct,
        reason=f"panel {yes}/{panel_size} correct; "
        + "; ".join(v.reason for v in votes if v.reason),
    )


def _coerce_verdict(result: Any) -> ReflectionVerdict:
    """Coerce a client response into a :class:`ReflectionVerdict`.

    Raises:
        TypeError: When the response is not a verdict-shaped model.
    """
    if isinstance(result, ReflectionVerdict):
        return result
    if isinstance(result, BaseModel):
        return ReflectionVerdict.model_validate(result.model_dump())
    raise TypeError(
        f"judge_reflection_decision expected ReflectionVerdict, "
        f"got {type(result).__name__}"
    )
