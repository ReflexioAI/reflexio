"""Tests for the playbook ask_human invocation eval."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.eval.playbook_ask_human.case import load_cases
from tests.eval.playbook_ask_human.providers import make_ask_human_prediction_provider
from tests.eval.playbook_ask_human.runner import (
    AskHumanPrediction,
    EvalResults,
    run_eval,
    score_case,
)


def test_dataset_shape_and_balance():
    cases = load_cases()
    positives = [case for case in cases if case.expected_ask_human]
    negatives = [case for case in cases if not case.expected_ask_human]
    no_playbook = [case for case in cases if not case.expected_playbooks_needed]
    extract_without_ask = [
        case
        for case in cases
        if case.expected_playbooks_needed and not case.expected_ask_human
    ]

    assert len(cases) == 36
    assert len(positives) == 12
    assert len(negatives) == 24
    assert len(no_playbook) == 12
    assert len(extract_without_ask) == 12
    assert {
        "customer_support",
        "education",
        "finance",
        "healthcare",
        "general_purpose",
        "digital_employee",
        "sales",
        "legal_compliance",
    } <= {case.vertical for case in cases}

    for case in positives:
        assert case.expected_question_must_include
    for case in cases:
        transcript = "\n".join(turn.content for turn in case.sessions).lower()
        assert "ask_human" not in transcript
        assert "agent builder" not in transcript


def test_runner_perfect_predictions_score_full_precision_and_recall():
    cases = load_cases()
    predictions = [
        AskHumanPrediction(
            case_id=case.id,
            tool_names=["ask_human"] if case.expected_ask_human else [],
            question_texts=[" ".join(case.expected_question_must_include)],
        )
        for case in cases
    ]

    result = run_eval(cases=cases, predictions=predictions)

    assert isinstance(result, EvalResults)
    assert result.metrics.tp == 12
    assert result.metrics.fp == 0
    assert result.metrics.tn == 24
    assert result.metrics.fn == 0
    assert result.metrics.precision == pytest.approx(1.0)
    assert result.metrics.recall == pytest.approx(1.0)
    assert "per-vertical" in result.summary()


def test_runner_records_false_positive_and_false_negative():
    cases = load_cases()
    first_positive = next(case for case in cases if case.expected_ask_human)
    first_negative = next(case for case in cases if not case.expected_ask_human)
    selected = [first_positive, first_negative]
    predictions = [
        AskHumanPrediction(case_id=first_positive.id, tool_names=[]),
        AskHumanPrediction(
            case_id=first_negative.id,
            tool_names=["ask_human"],
            question_texts=["What policy is missing?"],
        ),
    ]

    result = run_eval(cases=selected, predictions=predictions)

    assert result.metrics.tp == 0
    assert result.metrics.fp == 1
    assert result.metrics.tn == 0
    assert result.metrics.fn == 1
    assert result.metrics.precision == pytest.approx(0.0)
    assert result.metrics.recall == pytest.approx(0.0)


def test_score_case_checks_question_fragments():
    case = next(case for case in load_cases() if case.expected_ask_human)

    missing_fragment = score_case(
        case=case,
        prediction=AskHumanPrediction(
            case_id=case.id,
            tool_names=["ask_human"],
            question_texts=["What is the policy?"],
        ),
    )
    matching_fragment = score_case(
        case=case,
        prediction=AskHumanPrediction(
            case_id=case.id,
            tool_names=["ask_human"],
            question_texts=[
                " ".join(case.expected_question_must_include),
            ],
        ),
    )

    assert not missing_fragment.question_matches
    assert matching_fragment.question_matches


def test_run_eval_requires_one_prediction_source():
    cases = load_cases()[:1]

    with pytest.raises(ValueError):
        run_eval(cases=cases)
    with pytest.raises(ValueError):
        run_eval(
            cases=cases,
            predictions=[AskHumanPrediction(case_id=cases[0].id)],
            prediction_provider=lambda case: AskHumanPrediction(case_id=case.id),
        )
    with pytest.raises(ValueError):
        run_eval(cases=cases, predictions=[])


def test_provider_extracts_tool_trace_and_seeds_prior_pending(monkeypatch):
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredPlaybookContent,
        StructuredPlaybookList,
    )
    from tests.eval.playbook_ask_human import providers

    case = next(case for case in load_cases() if case.prior_pending_tool_calls)
    storage = MagicMock()
    request_context = SimpleNamespace(
        org_id="eval-org",
        storage=storage,
        prompt_manager=MagicMock(),
        configurator=SimpleNamespace(),
    )
    fake_result = SimpleNamespace(
        output=StructuredPlaybookList(
            playbooks=[
                StructuredPlaybookContent(
                    trigger="deployment target",
                    content="Attach to the pending deployment target request.",
                    rationale="The pending request matches.",
                )
            ]
        ),
        trace=SimpleNamespace(
            turns=[
                SimpleNamespace(
                    tool_name="attach_pending_info_request",
                    args={"pending_tool_call_id": "ptc_existing_deployment_target"},
                ),
                SimpleNamespace(
                    tool_name="ask_human",
                    args={"question": "What is the canonical deployment target?"},
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        providers,
        "construct_playbook_extraction_messages_from_sessions",
        lambda **_kwargs: [{"role": "system", "content": "prompt"}],
    )
    monkeypatch.setattr(
        providers,
        "run_resumable_extraction_agent",
        lambda **_kwargs: fake_result,
    )

    provider = make_ask_human_prediction_provider(
        llm_client=MagicMock(),
        request_context=request_context,  # type: ignore[arg-type]
    )
    prediction = provider(case)

    assert prediction.case_id == case.id
    assert prediction.tool_names == ["attach_pending_info_request", "ask_human"]
    assert prediction.question_texts == ["What is the canonical deployment target?"]
    assert prediction.playbook_count == 1
    storage.create_pending_tool_call.assert_called_once()
