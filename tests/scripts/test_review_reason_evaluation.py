"""Evaluation accounting must not mistake call errors for rejections."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm.litellm_client import LiteLLMConfig
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.components.reviewer import (
    CandidateReviewDecision,
    CandidateRevision,
    PlaybookCandidateReviewOutput,
)
from scripts import evaluate_review_reason_codes as evaluation
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
        "code_expected": 1,
        "errors": 0,
    }
    assert report["summary"]["1.4.0"] == {
        "expected": 1,
        "processed": 0,
        "decision_correct": 0,
        "code_correct": 0,
        "code_expected": 1,
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


def test_evaluation_rejects_empty_corpus_before_calls():
    client = MagicMock()
    with pytest.raises(ValueError, match="nonempty"):
        evaluate([], client, 1)
    client.generate_chat_response.assert_not_called()


def test_survival_oracle_does_not_invent_reason_label_accuracy():
    case = next(
        case for case in load_cases(DEFAULT_CASES) if case.id == "positive-preference"
    )
    case.candidates[0].allowed_decisions = ["accept", "revise"]
    case.candidates[0].reason_code = None
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.return_value = PlaybookCandidateReviewOutput(
        decisions=[
            CandidateReviewDecision(
                id="C1",
                decision="revise",
                reason_code="unsupported_evidence",
                evidence_ids=["C1-E1"],
                revision=CandidateRevision(
                    content="Show totals first in future reports.",
                    trigger="When preparing reports",
                    rationale="The user requested this order.",
                ),
            )
        ]
    )
    report = evaluate([case], client, 1)
    assert all(row["decision_correct"] for row in report["rows"])
    assert all(row["code_correct"] is None for row in report["rows"])
    assert all(arm["code_expected"] == 0 for arm in report["summary"].values())


def test_generalization_holdout_is_disjoint_and_preserves_fatal_revision_controls():
    cases = load_cases(DEFAULT_CASES.with_name("reviewer_generalization_cases.json"))
    development = {case.id for case in cases if case.split == "development"}
    holdout = {case.id for case in cases if case.split == "holdout"}
    assert development and holdout and development.isdisjoint(holdout)
    assert len({case.domain for case in cases}) >= 10
    for split in ("development", "holdout"):
        selected = [case for case in cases if case.split == split]
        assert any(
            candidate.decision == "revise"
            and candidate.reason_code == "unseen_artifact"
            for case in selected
            for candidate in case.candidates
        )
        assert any(
            candidate.decision == "reject"
            and candidate.reason_code == "absence_inference"
            for case in selected
            for candidate in case.candidates
        )


def test_revision_evidence_and_normalized_ids_survive_real_reviewer_and_checkpoint(
    tmp_path: Path,
):
    case = next(
        case for case in load_cases(DEFAULT_CASES) if case.id == "speculative-revisable"
    )
    revision = CandidateRevision(
        content="Show totals first in reports.",
        trigger="When preparing reports",
        rationale="The user explicitly requested this order.",
    )
    output = PlaybookCandidateReviewOutput(
        decisions=[
            CandidateReviewDecision(
                id=" C1 ",
                decision="revise",
                reason_code="speculative",
                evidence_ids=["C1-E1"],
                revision=revision,
            )
        ]
    )
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.return_value = output
    checkpoint = tmp_path / "checkpoint.json"
    report = evaluate([case], client, 1, checkpoint)
    assert not report["errors"]
    for row in report["rows"]:
        assert row["candidate"] == "C1"
        assert row["decision_correct"] and row["code_correct"]
        assert row["revision"] == revision.model_dump()
        assert row["evidence_ids"] == ["C1-E1"]
    assert json.loads(checkpoint.read_text())["rows"] == report["rows"]


@pytest.mark.parametrize("actual_model", ["expected-model", "another-model"])
def test_cli_checks_expected_role_model_before_evaluating(
    monkeypatch, tmp_path: Path, actual_model: str
):
    client = MagicMock()
    client._resolve_primary_model.return_value = actual_model
    constructor = MagicMock(return_value=client)
    run = MagicMock(return_value={"summary": {}, "errors": []})
    monkeypatch.setattr(evaluation, "LiteLLMClient", constructor)
    monkeypatch.setattr(evaluation, "assert_litellm_unpatched", lambda: None)
    monkeypatch.setattr(evaluation, "evaluate", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--expected-model",
            "expected-model",
            "--out",
            str(tmp_path / "out.json"),
        ],
    )
    if actual_model == "expected-model":
        assert evaluation.main() == 0
        run.assert_called_once()
    else:
        with pytest.raises(SystemExit) as error:
            evaluation.main()
        assert error.value.code == 2
        run.assert_not_called()


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


@pytest.mark.parametrize("candidate_version", ["1.4.0", "1.5.0", "1.6.0"])
def test_candidate_version_selection_preserves_baseline_and_report_identity(
    candidate_version,
):
    case = next(
        case for case in load_cases(DEFAULT_CASES) if case.id == "positive-preference"
    )
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.return_value = PlaybookCandidateReviewOutput(
        decisions=[
            CandidateReviewDecision(
                id="C1",
                decision="accept",
                reason_code="grounded_useful",
                evidence_ids=["C1-E1"],
            )
        ]
    )
    report = evaluate([case], client, 1, candidate_version=candidate_version)
    assert set(report["summary"]) == {"1.3.0", candidate_version}
    assert report["candidate_version"] == candidate_version
    assert (
        report["prompt_identities"][candidate_version]["active_version"]
        == candidate_version
    )
    prompts = [
        str(call.args[0]) for call in client.generate_chat_response.call_args_list
    ]
    assert ("**Independent lessons:**" in prompts[1]) == (
        candidate_version in ("1.5.0", "1.6.0")
    )
    changed_case = case.model_copy(deep=True)
    changed_case.turns[0].content += " Changed evidence."
    changed = evaluate([changed_case], client, 1, candidate_version=candidate_version)
    assert report["cases_sha256"] != changed["cases_sha256"]


def test_invalid_candidate_version_fails_before_calls():
    client = MagicMock()
    with pytest.raises(ValueError, match="candidate_version"):
        evaluate(load_cases(DEFAULT_CASES), client, 1, candidate_version="1.3.0")
    client.generate_chat_response.assert_not_called()


@pytest.mark.parametrize("candidate_version", ["1.5.0", "1.6.0"])
def test_clarified_prompt_preserves_fatal_gates_revision_policy_and_active_default(
    candidate_version,
):
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
    old = manager.render_prompt("playbook_candidate_review", values)
    clarified = PromptManager(
        version_override={"playbook_candidate_review": candidate_version}
    ).render_prompt("playbook_candidate_review", values)
    assert (
        clarified.split("## Reason-code precedence")[0]
        == old.split("## Output rules")[0]
    )
    assert clarified.split("## Output rules")[1] == old.split("## Output rules")[1]
