"""Evaluate the real audited method on explicitly supplied synthetic evidence.

This is a method-level benchmark, not storage/ownership/publication verification.
Those contracts are exercised through the production adapter's integration tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from reflexio.server.services.playbook.components.reconciliation import (
    Decisions,
    Reconciler,
)
from tests.eval.consolidation.case import ConsolidationEvalCase

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient


def make_consolidation_decision_provider(
    *, llm_client: LiteLLMClient, prompts: Mapping[str, str] | None = None
) -> Callable[[ConsolidationEvalCase], Decisions]:
    model = llm_client.config.model

    def provider(case: ConsolidationEvalCase) -> Decisions:
        if not case.candidate.evidence_refs:
            raise ValueError(
                "Live consolidation evaluation requires candidate source references"
            )
        sessions = [session.model_dump(mode="json") for session in case.sessions]
        coordinates = [
            (session["id"], message["turn"])
            for session in sessions
            for message in session["messages"]
        ]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("Evaluation source coordinates must be unique")
        references = dict.fromkeys(coordinates)

        def item(row, label):
            result = row.model_dump(mode="json", exclude={"id", "new_id"})
            result["id"] = label
            for ref in row.evidence_refs:
                if (ref.session_id, ref.turn) not in references:
                    raise ValueError("Evaluation row cites an absent source message")
            return result

        payload = {
            "new": [item(case.candidate, "NEW-0")],
            "existing": [
                item(row, f"EXISTING-{index}")
                for index, row in enumerate(case.existing)
            ],
            "sessions": sessions,
            "context_is_reference_excerpt": True,
        }
        return Reconciler(llm_client, model, prompts=prompts).run(payload, references)

    return provider
