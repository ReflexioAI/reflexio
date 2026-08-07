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

Costs real API calls. It lives under `e2e_tests/` because that is the only path
`llm_mock._is_e2e_test_run` exempts from the session-wide `litellm.completion`
patch -- from anywhere else the reviewer answers from the mock's profile-shaped
payload, the schema fails to parse, and the repair ladder exhausts into twelve
failures that look like a reviewer regression and are not.

Run it when changing the reviewer prompt:

    set -a && source .env && set +a && \\
    RUN_LOW_PRIORITY=1 uv run pytest \\
      tests/e2e_tests/test_reviewer_manifest_regression_real_llm.py -o 'addopts=' -n 0 -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

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
from reflexio.test_support.llm_credentials import real_generation_provider
from tests.server.test_utils import skip_low_priority

# The model comes from `resolve_model_name(ModelRole.GENERATION)`, so the gate
# has to be provider-agnostic too: keying it on OPENAI/ANTHROPIC skipped the
# whole file on a MiniMax-only machine even though resolution would have picked
# `minimax/MiniMax-M3` and the test would have run.
#
# Restricted to the providers whose models are known to satisfy these semantic
# assertions rather than every generation-capable provider -- a weaker model
# behind some other key would turn silent skips into false regressions.
#
# `real_generation_provider` rather than `os.getenv`, because the credential
# floor pins a placeholder key when none is set; a plain getenv check would let
# this run against a credential that authenticates with nothing.
_REVIEWER_CAPABLE = frozenset({"openai", "anthropic", "minimax"})

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requires_credentials,
    pytest.mark.skipif(
        not real_generation_provider(_REVIEWER_CAPABLE),
        reason=(
            "no real key for a reviewer-capable provider "
            "(openai/anthropic/minimax) is set"
        ),
    ),
]

_MANIFEST = (
    Path(__file__).resolve().parent.parent
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


@skip_low_priority
@pytest.mark.parametrize(
    "case", _must_capture_cases(), ids=lambda case: str(case["id"])
)
def test_must_capture_families_survive_review(case: dict) -> None:
    """A family the manifest says must be captured must not be rejected.

    Asserted per-case rather than as an aggregate rate: which family breaks is
    the diagnostic, and an aggregate would let one regression hide behind
    eleven passes.
    """
    assert os.environ.get("MOCK_LLM_RESPONSE", "").strip().lower() != "true", (
        "this test asserts real reviewer behavior; against the mock every case "
        "fails on schema parse. Invoke it by a path under tests/e2e_tests/."
    )

    sessions = _sessions_for(case)
    model = resolve_model_name(ModelRole.GENERATION)
    request_context = MagicMock()
    request_context.prompt_manager = PromptManager()
    request_context.org_id = "manifest-regression"
    request_context.storage = None
    reviewer = PlaybookCandidateReviewer(
        request_context=request_context,
        llm_client=LiteLLMClient(LiteLLMConfig(model=model)),
    )
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
        # `decide` renders the evidence chronology itself from the sessions
        # above; `agent_context` is the separate "what this agent does" slot
        # that production fills from `configurator.get_agent_context()`.
        agent_context="Test agent",
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
