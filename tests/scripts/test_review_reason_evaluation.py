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
    CandidateSupportingEvidence,
    EvidenceFirstReviewOutput,
    PlaybookCandidateReviewOutput,
    SupportingExcerpt,
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
    for call in report["calls"]:
        survivor = call["applied_survivors"][0]
        assert survivor["content"] == revision.content
        assert survivor["trigger"] == revision.trigger
        assert survivor["rationale"] == revision.rationale
        assert survivor["source_interaction_ids"] == [1]
        assert "speculative" in survivor["notes"]
        assert call["public_output"]["decisions"][0]["candidate_id"] == " C1 "


def _accepted_preference():
    return PlaybookCandidateReviewOutput(
        decisions=[
            CandidateReviewDecision(
                id="C1",
                decision="accept",
                reason_code="grounded_useful",
                evidence_ids=["C1-E1"],
            )
        ]
    )


def _observation(monkeypatch, *, complete: bool = True):
    # These tests exercise runner control flow, not provider measurement quality.
    metrics = MagicMock()
    metrics.data = {
        "observation_complete": complete,
        "estimated_cost_complete": complete,
    }
    monkeypatch.setattr(evaluation, "ReviewCallMetrics", lambda *_: metrics)
    return metrics


@pytest.mark.parametrize("failed_call", [1, 2])
def test_strict_run_stops_first_or_second_arm_without_replacement(
    monkeypatch, failed_call
):
    _observation(monkeypatch)
    case = next(c for c in load_cases(DEFAULT_CASES) if c.id == "positive-preference")
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.side_effect = [
        *[_accepted_preference() for _ in range(failed_call - 1)],
        RuntimeError("provider failed"),
    ]
    report = evaluate([case], client, 3, stop_on_incomplete=True)
    assert report["complete"] is False
    assert report["planned_calls"] == 6
    assert report["attempted_calls"] == failed_call
    assert report["unbegun_calls"] == 6 - failed_call
    assert len(report["errors"]) == 1
    assert len(report["rows"]) == failed_call - 1
    assert report["calls"][-1]["applied_survivors"] is None
    assert client.generate_chat_response.call_count == failed_call


@pytest.mark.parametrize("missing", ["observation_complete", "estimated_cost_complete"])
def test_strict_incomplete_observation_keeps_applied_output_without_scoring(
    monkeypatch, missing
):
    metrics = _observation(monkeypatch)
    metrics.data[missing] = False
    case = next(c for c in load_cases(DEFAULT_CASES) if c.id == "positive-preference")
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.return_value = _accepted_preference()
    report = evaluate([case], client, 2, stop_on_incomplete=True)
    assert report["complete"] is False
    assert report["rows"] == []
    assert report["attempted_calls"] == 1
    assert report["unbegun_calls"] == 3
    assert len(report["measurement_errors"]) == 1
    assert report["calls"][0]["public_output"]["decisions"][0]["decision"] == "accept"
    assert (
        report["calls"][0]["applied_survivors"][0]["content"]
        == case.candidates[0].content
    )


def test_strict_application_failure_keeps_public_output(monkeypatch):
    _observation(monkeypatch)
    case = next(c for c in load_cases(DEFAULT_CASES) if c.id == "positive-preference")
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.return_value = _accepted_preference()
    monkeypatch.setattr(
        evaluation.PlaybookCandidateReviewer,
        "apply_decisions",
        MagicMock(side_effect=ValueError("application failed")),
    )
    report = evaluate([case], client, 1, stop_on_incomplete=True)
    assert report["complete"] is False
    assert report["rows"] == []
    assert report["errors"][0]["error_type"] == "ValueError"
    assert report["calls"][0]["public_output"] is not None
    assert report["calls"][0]["applied_survivors"] is None


def test_application_interrupt_cannot_mark_checkpoint_measured(monkeypatch, tmp_path):
    _observation(monkeypatch)
    case = next(c for c in load_cases(DEFAULT_CASES) if c.id == "positive-preference")
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.return_value = _accepted_preference()
    monkeypatch.setattr(
        evaluation.PlaybookCandidateReviewer,
        "apply_decisions",
        MagicMock(side_effect=KeyboardInterrupt),
    )
    checkpoint = tmp_path / "interrupted.json"
    with pytest.raises(KeyboardInterrupt):
        evaluate([case], client, 1, checkpoint, stop_on_incomplete=True)
    saved = json.loads(checkpoint.read_text())
    assert saved["complete"] is False
    assert saved["rows"] == []
    assert saved["calls"][0]["measurement_complete"] is False
    assert saved["calls"][0]["public_output"] is not None
    assert saved["calls"][0]["applied_survivors"] is None
    assert client.generate_chat_response.call_count == 1


def test_semantic_failures_continue_with_global_counterbalance(monkeypatch):
    _observation(monkeypatch)
    cases = [
        c
        for c in load_cases(DEFAULT_CASES)
        if c.id in {"positive-preference", "positive-code-review-preference"}
    ]
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.return_value = PlaybookCandidateReviewOutput(
        decisions=[
            CandidateReviewDecision(
                id="C1", decision="reject", reason_code="generic", evidence_ids=[]
            )
        ]
    )
    report = evaluate(cases, client, 1, stop_on_incomplete=True)
    assert report["complete"] is True
    assert report["attempted_calls"] == 4
    assert report["unbegun_calls"] == 0
    assert [call["version"] for call in report["calls"]] == [
        "1.3.0",
        "1.4.0",
        "1.4.0",
        "1.3.0",
    ]
    assert all(not row["decision_correct"] for row in report["rows"])
    assert all(call["applied_survivors"] == [] for call in report["calls"])


def test_baseline_screen_has_no_candidate_calls_or_denominator(monkeypatch):
    _observation(monkeypatch)
    case = next(c for c in load_cases(DEFAULT_CASES) if c.id == "positive-preference")
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.return_value = _accepted_preference()
    report = evaluate([case], client, 2, baseline_only=True, stop_on_incomplete=True)
    assert report["complete"] is True
    assert report["candidate_version"] is None
    assert report["baseline_only"] is True
    assert set(report["summary"]) == {"1.3.0"}
    assert set(report["prompt_identities"]) == {"1.3.0"}
    assert report["planned_calls"] == report["attempted_calls"] == 2
    assert [call["version"] for call in report["calls"]] == ["1.3.0", "1.3.0"]
    assert all(
        "Reason-code precedence" not in str(call.args[0])
        for call in client.generate_chat_response.call_args_list
    )


def test_paired_inputs_are_isolated_from_reviewer_mutation(monkeypatch):
    case = next(c for c in load_cases(DEFAULT_CASES) if c.id == "positive-preference")
    prepared = prepare_case(case)
    monkeypatch.setattr(evaluation, "prepare_case", lambda _: prepared)
    original = evaluation.PlaybookCandidateReviewer.decide
    seen = []

    def mutate_first_arm(reviewer, **arguments):
        candidate = arguments["candidates"][0]
        seen.append(candidate.notes)
        if len(seen) == 1:
            candidate.notes = "First arm only"
        return original(reviewer, **arguments)

    monkeypatch.setattr(
        evaluation.PlaybookCandidateReviewer, "decide", mutate_first_arm
    )
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    client.generate_chat_response.return_value = _accepted_preference()
    report = evaluate([case], client, 1)
    assert not report["errors"]
    assert seen == [None, None]
    assert prepared[1][0].notes is None
    assert "First arm only" in report["calls"][0]["applied_survivors"][0]["notes"]
    assert "First arm only" not in report["calls"][1]["applied_survivors"][0]["notes"]


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_trial_is_preserved_before_calls(tmp_path, symlink):
    target = tmp_path / "prior.json"
    target.write_text("prior evidence")
    checkpoint = tmp_path / "checkpoint.json" if symlink else target
    if symlink:
        checkpoint.symlink_to(target)
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    with pytest.raises(FileExistsError):
        evaluate(load_cases(DEFAULT_CASES)[:1], client, 1, checkpoint)
    assert target.read_text() == "prior evidence"
    assert checkpoint.is_symlink() is symlink
    client.generate_chat_response.assert_not_called()


@pytest.mark.parametrize("actual_model", ["expected-model", "another-model"])
@pytest.mark.parametrize("baseline_only", [False, True])
def test_cli_checks_expected_role_model_before_evaluating(
    monkeypatch, tmp_path: Path, actual_model: str, baseline_only: bool
):
    client = MagicMock()
    client._resolve_primary_model.return_value = actual_model
    constructor = MagicMock(return_value=client)
    run = MagicMock(
        return_value={"summary": {}, "errors": [], "measurement_errors": []}
    )
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
            *(["--baseline-only"] if baseline_only else []),
        ],
    )
    if actual_model == "expected-model":
        assert evaluation.main() == 0
        run.assert_called_once()
        assert run.call_args.kwargs["stop_on_incomplete"] is True
        assert run.call_args.kwargs["baseline_only"] is baseline_only
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


@pytest.mark.parametrize("source", [None, "AgentGenerated", "chat"])
def test_turn_source_survives_fixture_loading_into_real_reviewer_payload(
    tmp_path: Path, source: str | None
):
    raw = next(
        case
        for case in json.loads(DEFAULT_CASES.read_text())
        if case["id"] == "positive-preference"
    )
    if source is not None:
        raw["turns"][0]["source"] = source
    fixture = tmp_path / "source-case.json"
    fixture.write_text(json.dumps([raw]))
    case = load_cases(fixture)[0]
    decision = CandidateReviewDecision(
        id="C1",
        decision="accept",
        reason_code="grounded_useful",
        evidence_ids=["C1-E1"],
    )
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])

    def respond(*_, **options):
        if options["response_format"] is EvidenceFirstReviewOutput:
            return EvidenceFirstReviewOutput(
                supporting_evidence=[
                    CandidateSupportingEvidence(
                        id="C1",
                        excerpts=[
                            SupportingExcerpt(
                                evidence_id="C1-E1", text=case.turns[0].content
                            )
                        ],
                    )
                ],
                decisions=[decision],
            )
        return PlaybookCandidateReviewOutput(decisions=[decision])

    client.generate_chat_response.side_effect = respond
    report = evaluate([case], client, 1, candidate_version="1.20.0")
    assert report["errors"] == []
    assert all(arm["processed"] == 1 for arm in report["summary"].values())
    messages = client.generate_chat_response.call_args_list[1].args[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    evidence = json.loads(messages[1]["content"])["own_evidence"]["C1"][0]
    assert evidence["role"].casefold() == "user"
    assert evidence["request_source"] == (source or "")


@pytest.mark.parametrize(
    "candidate_version",
    [
        "1.4.0",
        "1.5.0",
        "1.6.0",
        "1.7.0",
        "1.8.0",
        "1.9.0",
        "1.10.0",
        "1.11.0",
        "1.12.0",
        "1.13.0",
        "1.14.0",
        "1.15.0",
        "1.16.0",
        "1.17.0",
        "1.18.0",
        "1.19.0",
        "1.20.0",
    ],
)
def test_candidate_version_selection_preserves_baseline_and_report_identity(
    candidate_version,
):
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

    def respond(*args, **kwargs):
        if kwargs["response_format"] is EvidenceFirstReviewOutput:
            return EvidenceFirstReviewOutput(
                supporting_evidence=[
                    CandidateSupportingEvidence(
                        id="C1",
                        excerpts=[
                            SupportingExcerpt(
                                evidence_id="C1-E1",
                                text=case.turns[0].content[:600],
                            )
                        ],
                    )
                ],
                decisions=accepted.decisions,
            )
        return accepted

    client.generate_chat_response.side_effect = respond
    report = evaluate([case], client, 1, candidate_version=candidate_version)
    assert report["errors"] == []
    assert all(arm["processed"] == 1 for arm in report["summary"].values())
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
        candidate_version
        in (
            "1.5.0",
            "1.6.0",
            "1.7.0",
            "1.8.0",
            "1.9.0",
            "1.10.0",
            "1.11.0",
            "1.12.0",
            "1.13.0",
            "1.14.0",
            "1.15.0",
            "1.16.0",
            "1.17.0",
            "1.18.0",
            "1.19.0",
            "1.20.0",
        )
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


def test_report_writer_restricts_new_and_existing_files(tmp_path):
    import stat

    from scripts.evaluate_review_reason_codes import _write_private_text

    report = tmp_path / "private.json"
    _write_private_text(report, "first")
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    report.chmod(0o644)
    _write_private_text(report, "second")
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert report.read_text() == "second"


@pytest.mark.parametrize("unsupported", ["non_posix", "missing_fchmod"])
@pytest.mark.parametrize("entrypoint", ["writer", "evaluate"])
def test_private_artifacts_reject_unsupported_platform_before_calls_or_truncation(
    tmp_path, monkeypatch, unsupported, entrypoint
):
    from types import SimpleNamespace

    report = tmp_path / "existing.json"
    report.write_bytes(b"preserved private receipt")
    cases = load_cases(DEFAULT_CASES)[:1]
    client = MagicMock()
    client.config = LiteLLMConfig(model="test-model", fallback_models=[])
    platform_os = SimpleNamespace(**vars(evaluation.os))
    if unsupported == "non_posix":
        platform_os.name = "nt"
    else:
        del platform_os.fchmod
    monkeypatch.setattr(evaluation, "os", platform_os)
    with pytest.raises(RuntimeError, match="POSIX.*fchmod"):
        if entrypoint == "writer":
            evaluation._write_private_text(report, "replacement")
        else:
            evaluate(cases, client, 1, checkpoint=report)
    client.generate_chat_response.assert_not_called()
    assert report.read_bytes() == b"preserved private receipt"


def test_private_writer_replaces_symlink_without_touching_victim(tmp_path):
    victim = tmp_path / "victim.json"
    victim.write_text("original victim")
    report = tmp_path / "report.json"
    report.symlink_to(victim)
    evaluation._write_private_text(report, "private replacement")
    assert victim.read_text() == "original victim"
    assert not report.is_symlink()
    assert report.read_text() == "private replacement"
    assert report.stat().st_mode & 0o777 == 0o600


def test_private_writer_leaves_existing_open_descriptor_on_old_inode(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("old public receipt")
    report.chmod(0o666)
    with report.open("r+") as previous:
        evaluation._write_private_text(report, "new private receipt")
        assert previous.read() == "old public receipt"
        previous.seek(0)
        previous.write("old inode modified")
        previous.flush()
        assert report.read_text() == "new private receipt"
    assert report.stat().st_mode & 0o777 == 0o600


def test_private_writer_cleans_temporary_file_if_replace_fails(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    report.write_text("preserved receipt")

    def fail_replace(*_):
        raise OSError("replacement failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        evaluation._write_private_text(report, "new private receipt")
    assert report.read_text() == "preserved receipt"
    assert list(tmp_path.iterdir()) == [report]
