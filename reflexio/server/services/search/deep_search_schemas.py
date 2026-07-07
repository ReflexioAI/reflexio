"""Structured-output schemas for the deep (agentic) unified search tier.

All models are deliberately flat (no discriminated unions) — the provider
strict-schema path rejects ``oneOf``/``discriminator`` constructs. Time
windows are expressed as relative day offsets rather than ISO dates so the
planner model never has to do calendar arithmetic; the executor converts
them against the real clock.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from reflexio.models.structured_output import StrictStructuredOutput

DeepSearchArm = Literal["profiles", "user_playbooks", "agent_playbooks"]

# Candidate-key prefixes used to address entities across arms in the
# reflect/rerank stage ("P:<profile_id>", "UP:<int>", "AP:<int>").
ARM_KEY_PREFIX: dict[str, str] = {
    "profiles": "P",
    "user_playbooks": "UP",
    "agent_playbooks": "AP",
}


class PlannedSubquery(BaseModel):  # nested in the strict outputs below
    """One targeted retrieval subquery against a single memory arm.

    Args:
        arm (str): Which arm to search (profiles / user_playbooks /
            agent_playbooks).
        query (str): Standalone search text for this subquery.
        search_mode (str): "hybrid" (semantic + keyword) or "fts"
            (keyword-only; use for exact identifiers).
        start_days_ago (float, optional): Older bound of a time window —
            only entities newer than this many days are eligible.
        end_days_ago (float, optional): Newer bound of a time window — only
            entities older than this many days are eligible (e.g. "as of
            last month").
        recency_dominant (bool): True when the query asks for the CURRENT
            or LATEST value — final ordering for this arm is by entity
            timestamp instead of text relevance.
        reason (str): One short sentence on why this subquery helps.
    """

    arm: DeepSearchArm
    query: str
    search_mode: Literal["hybrid", "fts"] = "hybrid"
    start_days_ago: float | None = Field(default=None, ge=0)
    end_days_ago: float | None = Field(default=None, ge=0)
    recency_dominant: bool = False
    reason: str = ""


class SearchPlan(StrictStructuredOutput):
    """Planner output: the set of subqueries to run in parallel.

    Args:
        subqueries (list[PlannedSubquery]): Targeted subqueries; the
            executor runs them concurrently and fuses the results.
        notes (str): Planner's brief reading of the question (kept in the
            trace for observability).
    """

    subqueries: list[PlannedSubquery]
    notes: str = ""


class RerankOutput(StrictStructuredOutput):
    """Listwise reranker output: candidate ordering for one window.

    Args:
        ranked_candidate_ids (list[str]): Candidate keys ordered best-first.
            Unknown keys are dropped by the caller; omitted candidates keep
            their prior order after the ranked ones.
        rationale (str): One short sentence on the ordering decision.
    """

    ranked_candidate_ids: list[str]
    rationale: str = ""


class ReflectVerdict(StrictStructuredOutput):
    """Reflect stage output: sufficiency grade + candidate ordering.

    Args:
        sufficiency (str): Whether the fused candidates can answer the
            query — "sufficient", "partial", or "insufficient".
        ranked_candidate_ids (list[str]): Candidate keys (e.g. "P:abc",
            "UP:3") ordered best-first across all arms. Unknown keys are
            dropped by the caller; omitted candidates keep retrieval order
            after the ranked ones.
        corrective_subqueries (list[PlannedSubquery]): Follow-up subqueries
            to run when sufficiency is not "sufficient" and correction is
            allowed. Empty when the results suffice.
        rationale (str): One short paragraph explaining the grade.
    """

    sufficiency: Literal["sufficient", "partial", "insufficient"]
    ranked_candidate_ids: list[str]
    corrective_subqueries: list[PlannedSubquery] = []
    rationale: str = ""
