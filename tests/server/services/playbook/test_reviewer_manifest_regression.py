"""Positive control: must-capture manifest families survive the live reviewer.

Every other reviewer test mocks the LLM, so none of them can detect a prompt
change that makes the model reject good candidates. That gap has bitten: a
reviewer check was written whose disqualifier list textually contradicted the
prompt's own "Useful cores to preserve" section, and every unit test passed --
the contradiction was only visible by reading the two sections side by side.

Window-based measurement did not cover it either, because the production windows
sampled happened to contain no candidate of the affected shape. This test closes
that by driving the reviewer with the repo's own generalization manifest: each
`must_capture` case becomes one candidate, grounded in that case's own turns. A
must-capture family that does not survive is a false rejection.

Costs real API calls, so it is marked `requires_credentials` and excluded from
default runs. Run it when changing the reviewer prompt:

    uv run pytest tests/server/services/playbook/test_reviewer_manifest_regression.py \\
        -m requires_credentials -o 'addopts='
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reflexio.models.api_schema.domain.entities import (
    Interaction,
    Request,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.components.reviewer import (
    PlaybookCandidateReviewer,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    build_playbook_prompt_context,
)

pytestmark = pytest.mark.requires_credentials

_MANIFEST = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "test_data"
    / "playbook_generalization_manifest.json"
)


def _must_capture_cases() -> list[dict]:
    cases = json.loads(_MANIFEST.read_text())["cases"]
    return [case for case in cases if case.get("expectation") == "must_capture"]


def _sessions_for(case: dict) -> list[RequestInteractionDataModel]:
    """Build one interaction per manifest turn, preserving the stated role."""
    sessions = []
    for index, turn in enumerate(case["turns"], start=1):
        request_id = f"req-{case['id']}-{index}"
        sessions.append(
            RequestInteractionDataModel(
                session_id=f"sess-{case['id']}",
                request=Request(
                    request_id=request_id,
                    user_id="u1",
                    source="manifest-regression",
                    agent_version="1.0",
                    session_id=f"sess-{case['id']}",
                    created_at=index,
                ),
                interactions=[
                    Interaction(
                        interaction_id=index,
                        user_id="u1",
                        request_id=request_id,
                        content=turn["content"],
                        role=turn["role"],
                        created_at=index,
                        user_action=UserActionType.NONE,
                        user_action_description="",
                    )
                ],
            )
        )
    return sessions


@pytest.mark.parametrize(
    "case", _must_capture_cases(), ids=lambda case: str(case["id"])
)
def test_must_capture_families_survive_review(case: dict) -> None:
    """A family the manifest says must be captured must not be rejected.

    Asserted per-case rather than as an aggregate rate: which family breaks is
    the diagnostic, and an aggregate would let one regression hide behind
    eleven passes.
    """
    sessions = _sessions_for(case)
    model = resolve_model_name(ModelRole.GENERATION)
    reviewer = PlaybookCandidateReviewer(
        request_context=SimpleNamespace(
            prompt_manager=PromptManager(), org_id="manifest-regression", storage=None
        ),
        llm_client=LiteLLMClient(LiteLLMConfig(model=model)),
    )
    context = build_playbook_prompt_context(sessions, expert=False, label_turns=True)
    candidate = UserPlaybook(
        user_playbook_id=1,
        agent_version="1.0",
        request_id="manifest-regression",
        user_id="u1",
        content=case["expected_rule"],
        trigger=case["earliest_trigger"],
        rationale="Derived from the cited turns.",
        source_interaction_ids=list(range(1, len(case["turns"]) + 1)),
    )

    outcome = reviewer.decide(
        candidates=[candidate],
        request_interaction_data_models=sessions,
        existing_playbooks=[],
        agent_context=context,
        playbook_definition="Reusable user guidance",
        tool_context="",
    )

    decisions = outcome.output.decisions
    assert len(decisions) == 1, f"expected one decision, got {len(decisions)}"
    decision = decisions[0]
    assert decision.decision in ("accept", "revise"), (
        f"{case['id']} ({case['family']}) was rejected as "
        f"{decision.reason_code}: {decision.reason}"
    )
