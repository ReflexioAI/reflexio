"""Multi-round memory-scenario runner.

Chains the existing *live* component providers across a scenario's rounds,
judging each produced decision with the existing component judges and
mechanically reflecting that decision into an accumulating in-memory
``book`` via the apply shim (``tests.eval.scenarios.book``). The next round
then reads the state the prior rounds settled — this is what makes the
harness a *multi-round* memory benchmark rather than a bag of independent
single-shot decision evals.

Round shapes
------------
- ``learn`` round: feed the round's ``interactions`` to the live extraction
  provider, then route each produced playbook through the live consolidation
  provider against the *current* book. Each consolidation decision is judged
  (``judge_consolidation_decision``) and applied (``apply_consolidation``).
- ``reflect`` round: build a reflection eval-case over the round's
  interaction window against the cited book rule, run the live reflection
  provider, judge it (``judge_reflection_decision``), and apply it
  (``apply_reflection``).

Gating
------
A ``learn`` round is ``judged_correct`` iff EVERY produced rule's
consolidation verdict is ``correct``. (Extraction quality is *recorded* in
the round detail when an ``extraction_judge`` + ``extraction_signal`` gold
are supplied, but it does NOT gate — gating stays on the consolidation
verdicts, which are the decisions that actually mutate the book.) A
``reflect`` round is ``judged_correct`` iff the reflection verdict is
``correct``. The optional end-state judge scores the final book against the
scenario's ``gold_end_state`` and gates only when supplied.

Position-vs-id contract (lines up with the apply shim)
------------------------------------------------------
For each produced playbook in a learn round we snapshot
``existing_order = list(book)`` *before* building the consolidation case and
pass that exact list to :func:`apply_consolidation` as ``existing_order``.
The consolidation case's ``existing`` is built from the same ``book`` in the
same order, so a ``UnifyDecision``'s ``archive_existing_ids`` (which are
**list positions**, per the shim's contract) index into the same ordering
the consolidator saw. The candidate's fresh id is computed with
``_next_id(book)`` so it does not collide with any existing rule id.

Everything live in this module is INJECTED — providers and judge clients are
parameters. The module itself never constructs a real LLM client; tests pass
mocks and live mode passes a real haiku client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from reflexio.models.api_schema.domain.entities import Interaction
from tests.eval.consolidation.case import (
    CandidatePlaybook,
    ConsolidationEvalCase,
    ExistingPlaybook,
)
from tests.eval.consolidation.judge import judge_consolidation_decision
from tests.eval.reflection.case import CitedItem, ReflectionEvalCase
from tests.eval.reflection.judge import judge_reflection_decision
from tests.eval.scenarios.book import (
    _next_id,
    apply_consolidation,
    apply_reflection,
)
from tests.eval.scenarios.case import BookRule

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.eval.judge import LLMJudge
    from tests.eval.scenarios.case import MemoryScenario, ScenarioRound


@dataclass
class RoundOutcome:
    """The judged result of one scenario round.

    Attributes:
        round_index: 0-based index of the round within the scenario.
        kind: ``"learn"`` or ``"reflect"`` (the round's kind).
        judged_correct: Whether the round's gating verdict(s) passed.
        detail: Human-readable summary (produced kinds + judge reasons).
    """

    round_index: int
    kind: str
    judged_correct: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    """The aggregate result of running a multi-round scenario.

    Attributes:
        scenario_id: The scenario's stable id.
        round_outcomes: One :class:`RoundOutcome` per round, in order.
        end_state_correct: ``True`` / ``False`` from the optional end-state
            judge, or ``None`` when no end-state judge was supplied.
    """

    scenario_id: str
    round_outcomes: list[RoundOutcome] = field(default_factory=list)
    end_state_correct: bool | None = None

    @property
    def scenario_passed(self) -> bool:
        """True iff every round passed and the end-state was not a failure.

        An absent end-state judge (``end_state_correct is None``) does not
        fail the scenario — only an explicit ``False`` does.
        """
        return all(o.judged_correct for o in self.round_outcomes) and (
            self.end_state_correct is not False
        )


def _to_book_rule(produced: Any, *, new_id: int) -> BookRule:
    """Coerce a produced playbook into a :class:`BookRule`.

    Accepts either a ``UserPlaybook`` entity (read via ``getattr``) or a
    plain dict (read via ``.get``). Only ``content`` / ``trigger`` /
    ``rationale`` are carried; the id is the caller-supplied fresh id.

    Args:
        produced: A produced playbook entity or dict.
        new_id: The fresh book id to stamp on the rule.

    Returns:
        A :class:`BookRule` carrying the produced rule's text fields.
    """
    if isinstance(produced, dict):
        content = produced.get("content", "")
        trigger = produced.get("trigger")
        rationale = produced.get("rationale", "")
    else:
        content = getattr(produced, "content", "")
        trigger = getattr(produced, "trigger", None)
        rationale = getattr(produced, "rationale", "")
    return BookRule(
        id=new_id,
        content=content or "",
        trigger=trigger,
        rationale=rationale or "",
    )


def _book_to_existing(book: list[BookRule]) -> list[ExistingPlaybook]:
    """Project the current book into the consolidation case's EXISTING rows.

    Order is preserved — this exact ordering is what the runner also passes
    to :func:`apply_consolidation` as ``existing_order``, so a unify
    decision's list-position ``archive_existing_ids`` line up.
    """
    return [
        ExistingPlaybook(
            id=rule.id,
            content=rule.content,
            trigger=rule.trigger,
            rationale=rule.rationale,
        )
        for rule in book
    ]


def _candidate(rule: BookRule) -> CandidatePlaybook:
    """Project a freshly-extracted :class:`BookRule` into a candidate.

    The candidate ``new_id`` is a string (``"new-<id>"``) mirroring the live
    decision's string ``new_id`` field; the harness reads only the decision
    ``kind``, so the exact value is non-load-bearing.
    """
    return CandidatePlaybook(
        new_id=f"new-{rule.id}",
        content=rule.content,
        trigger=rule.trigger,
        rationale=rule.rationale,
    )


def _interactions(round: ScenarioRound) -> list[Interaction]:
    """Build ``Interaction`` entities from a round's loose interaction dicts.

    Mirrors the fields the reflection provider's window consumes: ``role`` /
    ``content`` (and ``tools_used`` when present). ``user_id`` / ``request_id``
    are required on ``Interaction`` so they are stamped with eval constants.
    """
    interactions: list[Interaction] = []
    for idx, turn in enumerate(round.interactions, start=1):
        kwargs: dict[str, Any] = {
            "interaction_id": idx,
            "user_id": "eval",
            "request_id": "eval",
            "role": str(turn.get("role", "User")),
            "content": str(turn.get("content", "")),
        }
        if turn.get("tools_used") is not None:
            kwargs["tools_used"] = turn["tools_used"]
        interactions.append(Interaction(**kwargs))
    return interactions


def _cited_item(rule: BookRule) -> CitedItem:
    """Build the reflection cited item for a book rule (always a playbook)."""
    return CitedItem(
        kind="playbook",
        target_id=str(rule.id),
        content=rule.content,
        trigger=rule.trigger,
    )


def _run_learn_round(
    *,
    scenario: MemoryScenario,
    idx: int,
    round: ScenarioRound,
    book: list[BookRule],
    extraction_provider: Callable[[dict], tuple[list[Any], list[Any]]],
    consolidation_provider: Callable[[ConsolidationEvalCase], Any],
    consolidation_judge_client: Any,
    extraction_judge: LLMJudge | None,
) -> tuple[list[BookRule], RoundOutcome]:
    """Run one learn round: extract -> consolidate -> judge -> apply.

    Returns the new book and the round outcome. Gates on consolidation
    verdicts only; extraction (when judged) is recorded in the detail.
    """
    _profiles, playbooks = extraction_provider(
        {"id": f"{scenario.id}-{idx}", "sessions": round.interactions}
    )

    all_correct = True
    kinds: list[str] = []
    reasons: list[str] = []
    gold_kind = round.gold.get("consolidation_kind", "independent")

    for p in playbooks:
        existing_order = list(book)
        cand_rule = _to_book_rule(p, new_id=_next_id(book))
        cand_case = ConsolidationEvalCase(
            id=f"{scenario.id}-{idx}-{cand_rule.id}",
            agent_context=scenario.agent_context,
            existing=_book_to_existing(book),
            candidate=_candidate(cand_rule),
            gold_kind=gold_kind,
        )
        decision = consolidation_provider(cand_case)
        verdict = judge_consolidation_decision(
            case=cand_case,
            produced_decision=decision,
            llm_client=consolidation_judge_client,
        )
        all_correct = all_correct and verdict.correct
        kinds.append(getattr(decision, "kind", "?"))
        if verdict.reason:
            reasons.append(verdict.reason)
        book = apply_consolidation(
            book, cand_rule, decision, existing_order=existing_order
        )

    detail_parts = [f"kinds={kinds}", f"gold={gold_kind}"]
    if reasons:
        detail_parts.append("; ".join(reasons))

    extraction_signal = round.gold.get("extraction_signal")
    if extraction_judge is not None and extraction_signal:
        ext_score = extraction_judge.score(
            expected=extraction_signal,
            actual=json.dumps(
                [_to_book_rule(p, new_id=0).model_dump() for p in playbooks]
            ),
        )
        detail_parts.append(f"extraction_signal_f1={ext_score.signal_f1:.2f}")

    return book, RoundOutcome(
        round_index=idx,
        kind=round.kind,
        judged_correct=all_correct,
        detail=" | ".join(detail_parts),
    )


def _run_reflect_round(
    *,
    scenario: MemoryScenario,
    idx: int,
    round: ScenarioRound,
    book: list[BookRule],
    reflection_provider: Callable[[ReflectionEvalCase], Any],
    reflection_judge_client: Any,
) -> tuple[list[BookRule], RoundOutcome]:
    """Run one reflect round: build case -> reflect -> judge -> apply."""
    cited = next((r for r in book if r.id == round.cited), None)
    if cited is None:
        return book, RoundOutcome(
            round_index=idx,
            kind=round.kind,
            judged_correct=False,
            detail=f"cited rule not in book (cited={round.cited})",
        )

    refl_case = ReflectionEvalCase(
        id=f"{scenario.id}-{idx}",
        agent_context=scenario.agent_context,
        window=_interactions(round),
        cited_item=_cited_item(cited),
        gold_label=round.gold.get("reflection", "no_change"),
    )
    decision = reflection_provider(refl_case)
    verdict = judge_reflection_decision(
        case=refl_case,
        produced_decision=decision,
        llm_client=reflection_judge_client,
    )
    book = apply_reflection(book, cited.id, decision)

    return book, RoundOutcome(
        round_index=idx,
        kind=round.kind,
        judged_correct=verdict.correct,
        detail=f"gold={refl_case.gold_label} | {verdict.reason}",
    )


def run_scenario(
    *,
    scenario: MemoryScenario,
    extraction_provider: Callable[[dict], tuple[list[Any], list[Any]]],
    consolidation_provider: Callable[[ConsolidationEvalCase], Any],
    reflection_provider: Callable[[ReflectionEvalCase], Any],
    consolidation_judge_client: Any,
    reflection_judge_client: Any,
    extraction_judge: LLMJudge | None = None,
    end_state_judge: LLMJudge | None = None,
) -> ScenarioResult:
    """Run one multi-round memory scenario end to end.

    Threads an accumulating ``book`` (seeded from ``scenario.seed_book``)
    through the scenario's rounds. Each round runs the relevant live
    provider, judges the produced decision(s), and reflects them into the
    book via the apply shim. After all rounds, an optional end-state judge
    scores the final book against ``scenario.gold_end_state``.

    Args:
        scenario: The multi-round scenario to run.
        extraction_provider: Live extraction provider; called with an
            extraction case dict ``{"id", "sessions"}`` and returns
            ``(profiles, playbooks)``.
        consolidation_provider: Live consolidation decision provider.
        reflection_provider: Live reflection decision provider.
        consolidation_judge_client: LLM client for the consolidation judge.
        reflection_judge_client: LLM client for the reflection judge.
        extraction_judge: Optional judge that scores extraction signal — used
            only for the round detail; it never gates.
        end_state_judge: Optional judge that scores the final book against
            ``scenario.gold_end_state``. When supplied, the end-state gates
            (signal_f1 >= 0.5 -> correct).

    Returns:
        A :class:`ScenarioResult` with per-round outcomes and the optional
        end-state verdict.
    """
    book: list[BookRule] = [r.model_copy() for r in scenario.seed_book]
    outcomes: list[RoundOutcome] = []

    for idx, round in enumerate(scenario.rounds):
        if round.kind == "learn":
            book, outcome = _run_learn_round(
                scenario=scenario,
                idx=idx,
                round=round,
                book=book,
                extraction_provider=extraction_provider,
                consolidation_provider=consolidation_provider,
                consolidation_judge_client=consolidation_judge_client,
                extraction_judge=extraction_judge,
            )
        else:
            book, outcome = _run_reflect_round(
                scenario=scenario,
                idx=idx,
                round=round,
                book=book,
                reflection_provider=reflection_provider,
                reflection_judge_client=reflection_judge_client,
            )
        outcomes.append(outcome)

    end_state_correct: bool | None = None
    if end_state_judge is not None and scenario.gold_end_state:
        score = end_state_judge.score(
            expected=scenario.gold_end_state,
            actual=json.dumps([r.model_dump() for r in book]),
        )
        end_state_correct = score.signal_f1 >= 0.5

    return ScenarioResult(
        scenario_id=scenario.id,
        round_outcomes=outcomes,
        end_state_correct=end_state_correct,
    )
