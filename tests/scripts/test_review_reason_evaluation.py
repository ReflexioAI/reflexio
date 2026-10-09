"""Evaluation accounting must not mistake call errors for rejections."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm.litellm_client import LiteLLMConfig
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.components.reviewer import (
    CandidateReviewDecision,
    PlaybookCandidateReviewOutput,
)
from scripts.evaluate_review_reason_codes import (
    DEFAULT_CASES,
    evaluate,
    load_cases,
    prepare_case,
)


def test_paired_evaluation_uses_real_reviewer_and_counts_errors_separately():
    case = next(
        case for case in load_cases(DEFAULT_CASES) if case.id == "positive-preference"
    )
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    accepted = PlaybookCandidateReviewOutput(
        decisions=[
            CandidateReviewDecision(
                id="C1",
                decision="accept",
                reason_code="grounded_useful",
                evidence_ids=["C1-E1"],
            )
        ]
    )
    client.generate_chat_response.side_effect = [
        accepted,
        RuntimeError("provider failed"),
    ]
    report = evaluate([case], client, 1)
    assert report["summary"]["1.3.0"] == {
        "expected": 1,
        "processed": 1,
        "decision_correct": 1,
        "code_correct": 1,
        "errors": 0,
    }
    assert report["summary"]["1.4.0"] == {
        "expected": 1,
        "processed": 0,
        "decision_correct": 0,
        "code_correct": 0,
        "errors": 1,
    }
    assert report["errors"][0]["error_type"] == "RuntimeError"
    prompts = [
        str(call.args[0]) for call in client.generate_chat_response.call_args_list
    ]
    assert "## Reason-code precedence" not in prompts[0]
    assert "## Reason-code precedence" in prompts[1]


@pytest.mark.parametrize("repeats", [0, -1])
def test_evaluation_rejects_empty_measurement(repeats):
    with pytest.raises(ValueError, match="positive"):
        evaluate(load_cases(DEFAULT_CASES), MagicMock(), repeats)


def test_fixture_validation_rejects_unresolvable_evidence(tmp_path: Path):
    raw = json.loads(DEFAULT_CASES.read_text())
    raw[0]["candidates"][0]["evidence"] = [999]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="evidence index"):
        load_cases(path)


def test_experimental_prompt_preserves_measured_subject_gate_and_default():
    manager = PromptManager()
    assert manager.get_active_version("playbook_candidate_review") == "1.3.0"
    values: dict[str, str] = dict.fromkeys(
        (
            "agent_context_prompt",
            "playbook_definition",
            "tool_context",
            "interaction_context",
            "artifact_availability",
            "candidates",
            "existing_playbooks",
        ),
        "",
    )
    proposed = PromptManager(version_override={"playbook_candidate_review": "1.4.0"})
    old = manager.render_prompt("playbook_candidate_review", values)
    new = proposed.render_prompt("playbook_candidate_review", values)
    assert new.split("## Reason-code precedence")[0] == old.split("## Output rules")[0]
    assert new.split("## Output rules")[1] == old.split("## Output rules")[1]


def test_every_frozen_case_can_be_prepared_before_paid_calls():
    for case in load_cases(DEFAULT_CASES):
        sessions, candidates, existing = prepare_case(case)
        assert len(sessions) == len(case.turns)
        assert len(candidates) == len(case.candidates)
        assert len(existing) == len(case.existing)
        assert all(item.request_id == case.id for item in existing)
