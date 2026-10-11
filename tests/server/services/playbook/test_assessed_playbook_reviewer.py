"""Original-rule consistency must fail before provider coercion or application."""

from __future__ import annotations

from typing import Any

import pytest

from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.components.reviewer import (
    AssessedReviewDecision,
    AssessedReviewOutput,
)
from tests.server.services.playbook.test_playbook_reviewer import (
    _candidate,
    _interaction_model,
    _reviewer,
)


def _response(
    *,
    basis: str = "supplied_parameter",
    kind: str = "supplied_parameter_use",
    defects: list[str] | None = None,
    decision: str = "revise",
    code: str = "unsupported_causality",
) -> dict[str, Any]:
    return {
        "assessments": [
            {
                "id": "C1",
                "original_rule_kind": kind,
                "original_rule_span": "Use the supplied width",
                "supported_core_basis": basis,
                "defect_codes": ["unsupported_causality"]
                if defects is None
                else defects,
            }
        ],
        "supporting_evidence": []
        if decision == "reject"
        else [
            {
                "id": "C1",
                "excerpts": [{"evidence_id": "C1-E1", "text": "Use width 800."}],
            }
        ],
        "decisions": [
            {
                "id": "C1",
                "decision": decision,
                "reason_code": code,
                "reason": None,
                "evidence_ids": [] if decision == "reject" else ["C1-E1"],
                "revision": {
                    "content": "Use the supplied width if assigned for the current image.",
                    "trigger": "When resizing an image with an assigned width",
                    "rationale": "The user supplied the image width.",
                }
                if decision == "revise"
                else None,
            }
        ],
    }


def _decide(response: AssessedReviewOutput):
    reviewer, client = _reviewer(response)
    reviewer.request_context.prompt_manager = PromptManager(
        version_override={reviewer.PROMPT_ID: "1.26.0"}
    )
    candidates = [
        _candidate(
            101,
            "Use width 800.",
            content="Use the supplied width; never ask again because it causes delays.",
        )
    ]
    result = reviewer.decide(
        candidates=candidates,
        request_interaction_data_models=[_interaction_model(101, "Use width 800.")],
        existing_playbooks=[],
        agent_context="Image assistant",
        playbook_definition="Future image guidance",
        tool_context="",
    )
    return reviewer, client, candidates, result


def test_original_conditional_core_is_preserved_without_persisting_assessment():
    reviewer, client, candidates, outcome = _decide(
        AssessedReviewOutput.model_validate(_response())
    )
    survivors = reviewer.apply_decisions(candidates=candidates, outcome=outcome)
    assert len(survivors) == 1
    assert (
        survivors[0].content
        == "Use the supplied width if assigned for the current image."
    )
    assert "unsupported_causality" in (survivors[0].notes or "")
    assert "assessments" not in survivors[0].model_dump()
    assert len(outcome.assessments) == 1
    assert (
        client.generate_chat_response.call_args.kwargs["response_format"]
        is AssessedReviewOutput
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "standing_value_preference"},
        {"kind": "other"},
        {"decision": "reject"},
        {"basis": "none"},
        {"defects": ["generic"], "code": "generic"},
        {"defects": ["absence_inference"], "code": "absence_inference"},
        {"defects": ["speculative", "unsupported_causality"], "code": "speculative"},
    ],
)
def test_inconsistent_original_assessment_cannot_return_success(changes):
    with pytest.raises(ValueError):
        _decide(AssessedReviewOutput.model_validate(_response(**changes)))


def test_fatal_decision_is_not_silently_normalized_by_experimental_parser():
    raw = _response(defects=["not_agent_decision"], code="not_agent_decision")
    parsed = AssessedReviewOutput.model_validate(raw)
    assert parsed.decisions[0].decision == "revise"
    assert parsed.decisions[0].revision is not None
    with pytest.raises(ValueError):
        _decide(parsed)


def test_duplicate_core_can_reject_with_higher_priority_causal_code():
    _, _, _, outcome = _decide(
        AssessedReviewOutput.model_validate(
            _response(defects=["redundant", "unsupported_causality"], decision="reject")
        )
    )
    assert outcome.output.decisions[0].reason_code == "unsupported_causality"


@pytest.mark.parametrize("reason", ["A historical claim", 1, {}])
def test_optional_historical_reason_is_excluded_in_schema(reason):
    raw = _response()
    raw["decisions"][0]["reason"] = reason
    with pytest.raises(ValueError):
        AssessedReviewOutput.model_validate(raw)
    assert (
        AssessedReviewDecision.model_json_schema()["properties"]["reason"]["type"]
        == "null"
    )


@pytest.mark.parametrize("field", ["content", "trigger", "rationale"])
def test_whitespace_revision_cannot_be_validated(field):
    raw = _response()
    raw["decisions"][0]["revision"][field] = " \n\t"
    with pytest.raises(ValueError):
        _decide(AssessedReviewOutput.model_validate(raw))


def test_outcome_cannot_be_applied_to_mutated_original_candidate():
    reviewer, _, candidates, outcome = _decide(
        AssessedReviewOutput.model_validate(_response())
    )
    candidates[0].content = "A different original rule"
    with pytest.raises(ValueError, match="original candidate batch"):
        reviewer.apply_decisions(candidates=candidates, outcome=outcome)


def _real_reviewer(responses: list[dict[str, Any]], *, fallback: bool = False):
    import json
    from contextlib import contextmanager
    from unittest.mock import patch

    from reflexio.server.llm.litellm_client import LiteLLMConfig
    from tests.server.llm.test_litellm_client_unit import (
        TestStructuredOutputRepair,
        _build_client,
    )

    config = LiteLLMConfig(
        model="primary-model", fallback_models=["fallback-model"] if fallback else []
    )
    client = _build_client(config)
    reviewer, _ = _reviewer(AssessedReviewOutput.model_validate(_response()))
    reviewer.client = client
    reviewer.request_context.prompt_manager = PromptManager(
        version_override={reviewer.PROMPT_ID: "1.26.0"}
    )
    calls: list[dict[str, Any]] = []

    def completion(**kwargs):
        calls.append(kwargs)
        return TestStructuredOutputRepair()._make_mock_response(
            json.dumps(responses[len(calls) - 1]), model=kwargs["model"]
        )

    @contextmanager
    def transport():
        # Role selection uses site/provider defaults rather than config.model.
        # Fix only that routing input; exercise actual parsing/repair/fallback.
        with (
            patch(
                "reflexio.server.llm._litellm_text_generation.resolve_model_name",
                return_value="primary-model",
            ),
            patch("litellm.completion", side_effect=completion),
        ):
            yield

    return reviewer, transport(), calls


def _run_real(reviewer):
    candidate = _candidate(
        101,
        "Use width 800.",
        content="Use the supplied width; never ask again because it causes delays.",
    )
    return reviewer.decide(
        candidates=[candidate],
        request_interaction_data_models=[_interaction_model(101, "Use width 800.")],
        existing_playbooks=[],
        agent_context="UNTRUSTED_CONTEXT_CANARY",
        playbook_definition="Future image guidance",
        tool_context="",
    )


@pytest.mark.parametrize("invalid_reason", [False, True])
def test_real_structured_parser_repairs_without_fatal_coercion(invalid_reason):
    import json

    bad = _response(defects=["not_agent_decision"], code="not_agent_decision")
    if invalid_reason:
        bad["decisions"][0]["reason"] = "A forbidden historical narrative"
    reviewer, transport, calls = _real_reviewer([bad, _response()])
    with transport:
        outcome = _run_real(reviewer)
    assert outcome.output.decisions[0].decision == "revise"
    assert len(calls) == 2
    assert [call["model"] for call in calls] == ["primary-model"] * 2
    assert "UNTRUSTED_CONTEXT_CANARY" not in calls[0]["messages"][0]["content"]
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["candidates"][0]["content"].startswith("Use the supplied width")
    assert calls[0]["messages"][0]["role"] == "system"


def test_existing_fallback_starts_fresh_only_after_invalid_primary_and_repair():
    bad = _response(decision="reject")
    reviewer, transport, calls = _real_reviewer([bad, bad, _response()], fallback=True)
    with transport:
        outcome = _run_real(reviewer)
    assert outcome.output.decisions[0].decision == "revise"
    assert [call["model"] for call in calls] == [
        "primary-model",
        "primary-model",
        "fallback-model",
    ]
    assert len(calls[2]["messages"]) == 2


def test_full_fallback_exhaustion_never_returns_an_outcome(caplog):
    from reflexio.server.llm.litellm_client import StructuredOutputRepairError

    bad = _response()
    bad["assessments"][0]["original_rule_span"] = "PRIVATE_CROSS_CANDIDATE_CANARY"
    reviewer, transport, calls = _real_reviewer([bad] * 4, fallback=True)
    with transport, pytest.raises(StructuredOutputRepairError):
        _run_real(reviewer)
    assert [call["model"] for call in calls] == ["primary-model"] * 2 + [
        "fallback-model"
    ] * 2
    assert "PRIVATE_CROSS_CANDIDATE_CANARY" not in caplog.text


@pytest.mark.parametrize("section", ["assessments", "decisions"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown"])
def test_candidate_accounting_is_exact_before_application(section, change):
    raw = _response()
    if change == "missing":
        raw[section] = []
    elif change == "duplicate":
        raw[section].append(dict(raw[section][0]))
    else:
        raw[section][0]["id"] = "PRIVATE_UNKNOWN_CANDIDATE_CANARY"
    with pytest.raises(ValueError) as exc:
        _decide(AssessedReviewOutput.model_validate(raw))
    assert "PRIVATE_UNKNOWN_CANDIDATE_CANARY" not in str(exc.value)


def test_inference_mutation_cannot_rebind_original_span_or_application():
    reviewer, client = _reviewer(AssessedReviewOutput.model_validate(_response()))
    reviewer.request_context.prompt_manager = PromptManager(
        version_override={reviewer.PROMPT_ID: "1.26.0"}
    )
    candidates = [
        _candidate(
            101,
            "Use width 800.",
            content="Use the supplied width; never ask again because it causes delays.",
        )
    ]

    def mutate_and_respond(*args, **kwargs):
        candidates[0].content = "Different rule during inference"
        result = AssessedReviewOutput.model_validate(_response())
        assert kwargs["structured_output_validator"](result) == ()
        return result

    client.generate_chat_response.side_effect = mutate_and_respond
    outcome = reviewer.decide(
        candidates=candidates,
        request_interaction_data_models=[_interaction_model(101, "Use width 800.")],
        existing_playbooks=[],
        agent_context="",
        playbook_definition="",
        tool_context="",
    )
    with pytest.raises(ValueError, match="original candidate batch"):
        reviewer.apply_decisions(candidates=candidates, outcome=outcome)


def test_defect_priority_covers_entire_existing_vocabulary():
    from typing import get_args

    from reflexio.models.api_schema.domain.enums import PlaybookReviewReasonCode
    from reflexio.server.services.playbook.components.reviewer import OriginalDefectCode

    assert set(get_args(OriginalDefectCode)) == set(
        get_args(PlaybookReviewReasonCode)
    ) - {"grounded_useful"}


@pytest.mark.parametrize("version", ["1.3.0", "1.25.0", "1.26.0"])
@pytest.mark.parametrize("field", ["content", "trigger", "rationale"])
def test_nonblank_revision_guard_applies_to_all_response_contracts(version, field):
    from reflexio.server.services.playbook.components.reviewer import (
        EvidenceFirstReviewOutput,
        PlaybookCandidateReviewOutput,
    )

    raw = _response()
    raw["decisions"][0]["revision"][field] = " \t\n"
    if version == "1.26.0":
        output = AssessedReviewOutput.model_validate(raw)
    elif version == "1.25.0":
        output = EvidenceFirstReviewOutput.model_validate(
            {k: raw[k] for k in ("supporting_evidence", "decisions")}
        )
    else:
        output = PlaybookCandidateReviewOutput.model_validate(
            {"decisions": raw["decisions"]}
        )
    reviewer, _ = _reviewer(output)
    reviewer.request_context.prompt_manager = PromptManager(
        version_override={reviewer.PROMPT_ID: version}
    )
    with pytest.raises(ValueError, match="nonblank"):
        _run_real(reviewer)


@pytest.mark.parametrize(
    "changes,valid",
    [
        (
            {
                "kind": "standing_value_preference",
                "basis": "future_instruction",
                "defects": [],
                "decision": "accept",
                "code": "grounded_useful",
            },
            True,
        ),
        (
            {
                "kind": "standing_value_preference",
                "basis": "none",
                "defects": ["speculative"],
                "decision": "reject",
                "code": "speculative",
            },
            True,
        ),
        (
            {
                "basis": "none",
                "defects": ["absence_inference"],
                "decision": "reject",
                "code": "absence_inference",
            },
            True,
        ),
        (
            {
                "basis": "supplied_parameter",
                "defects": ["absence_inference"],
                "decision": "reject",
                "code": "absence_inference",
            },
            True,
        ),
        ({"defects": ["speculative", "speculative"], "code": "speculative"}, False),
        *[
            (
                {
                    "basis": "none",
                    "defects": [code],
                    "decision": "reject",
                    "code": code,
                },
                False,
            )
            for code in ("late_trigger", "compound", "redundant")
        ],
    ],
)
def test_original_core_consistency_contract(changes, valid):
    response = AssessedReviewOutput.model_validate(_response(**changes))
    if valid:
        _decide(response)
    else:
        with pytest.raises(ValueError):
            _decide(response)


@pytest.mark.integration
@pytest.mark.parametrize("report_only", [True, False])
@pytest.mark.parametrize("version", ["1.3.0", "1.26.0"])
def test_stored_review_uses_real_parser_and_only_projects_approved_fields(
    tmp_path, report_only, version
):
    from unittest.mock import patch

    from reflexio.models.api_schema.domain.enums import Status
    from tests.server.services.playbook.test_playbook_review_service_integration import (
        _playbook,
        _request,
        _seed_source_interactions,
        _service,
        _storage,
    )

    storage = _storage(tmp_path)
    _seed_source_interactions(storage, 3)
    originals = [_playbook(i, created_at=203 - i) for i in (1, 2, 3)]
    storage.save_user_playbooks(originals)
    responses = []
    for i, action, code in [
        (1, "accept", "grounded_useful"),
        (2, "revise", "speculative"),
        (3, "reject", "not_agent_decision"),
    ]:
        raw = _response(
            kind="other",
            basis="none" if action == "reject" else "future_instruction",
            decision=action,
            defects=[] if action == "accept" else [code],
            code=code,
        )
        raw["assessments"][0]["original_rule_span"] = originals[i - 1].content
        if action != "reject":
            raw["supporting_evidence"][0]["excerpts"][0]["text"] = (
                f"Grounded evidence {i}."
            )
        responses.append(
            raw if version == "1.26.0" else {"decisions": raw["decisions"]}
        )
    reviewer, transport, calls = _real_reviewer(responses)
    service = _service(storage)
    service.client = reviewer.client
    service.request_context.prompt_manager = PromptManager(
        version_override={reviewer.PROMPT_ID: version}
    )
    with (
        transport,
        patch(
            "reflexio.server.services.playbook.review_service.PlaybookConsolidator.retrieve_existing_playbooks",
            return_value=[],
        ),
    ):
        response = service.run(
            _request(report_only=report_only), run_id="assessed-service-run"
        )
    assert response.success
    assert [item.decision for item in response.results] == ["accept", "edit", "reject"]
    assert len(calls) == 3
    serialized = response.model_dump(mode="json")
    assert all(
        "assessments" not in item and "supporting_evidence" not in item
        for item in serialized["results"]
    )
    current = storage.get_user_playbooks(status_filter=[None])
    if report_only:
        assert {item.user_playbook_id for item in current} == {
            item.user_playbook_id for item in originals
        }
    else:
        assert len(current) == 2
        assert originals[0].user_playbook_id in {
            item.user_playbook_id for item in current
        }
        assert len(storage.get_user_playbooks(status_filter=[Status.ARCHIVED])) == 1
        assert len(storage.get_user_playbooks(status_filter=[Status.SUPERSEDED])) == 1
        successor = next(
            item
            for item in current
            if item.user_playbook_id != originals[0].user_playbook_id
        )
        assert successor.content == responses[1]["decisions"][0]["revision"]["content"]
        assert "assessments" not in successor.model_dump()
        assert "Grounded evidence 2." not in (successor.notes or "")


@pytest.mark.integration
def test_stored_review_exhaustion_preserves_prior_commit_and_failed_original(
    tmp_path, caplog
):
    from unittest.mock import patch

    from tests.server.services.playbook.test_playbook_review_service_integration import (
        _playbook,
        _request,
        _seed_source_interactions,
        _service,
        _storage,
    )

    storage = _storage(tmp_path)
    _seed_source_interactions(storage, 2)
    originals = [_playbook(1, created_at=202), _playbook(2, created_at=201)]
    storage.save_user_playbooks(originals)
    first = _response(kind="other", basis="future_instruction")
    first["assessments"][0]["original_rule_span"] = originals[0].content
    first["supporting_evidence"][0]["excerpts"][0]["text"] = "Grounded evidence 1."
    bad = _response()
    reviewer, transport, calls = _real_reviewer([first, bad, bad])
    service = _service(storage)
    service.client = reviewer.client
    service.request_context.prompt_manager = reviewer.request_context.prompt_manager
    with (
        transport,
        patch(
            "reflexio.server.services.playbook.review_service.PlaybookConsolidator.retrieve_existing_playbooks",
            return_value=[],
        ),
    ):
        response = service.run(_request(report_only=False), run_id="partial-review-run")
    assert response.success is False
    assert len(response.results) == 1 and response.results[0].applied
    assert len(calls) == 3
    current = storage.get_user_playbooks(status_filter=[None])
    failed = next(
        item
        for item in current
        if item.user_playbook_id == originals[1].user_playbook_id
    )
    assert failed.content == originals[1].content
    assert "event=playbook_review_failed run_id=partial-review-run" in caplog.text


@pytest.mark.integration
def test_normal_service_cannot_drop_blank_reviewed_core_and_return_success(
    tmp_path, monkeypatch
):
    from reflexio.models.config_schema import PlaybookConfig
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.llm.litellm_client import StructuredOutputRepairError
    from reflexio.server.services.playbook.service import (
        PlaybookGenerationService,
        PlaybookGenerationServiceConfig,
    )
    from tests.server.services.playbook.test_playbook_generation_service import (
        _playbook_run,
        _storage,
    )

    raw = _response()
    raw["decisions"][0]["revision"]["content"] = " \t\n"
    reviewer, transport, calls = _real_reviewer([raw, raw])
    service = PlaybookGenerationService(
        llm_client=reviewer.client,
        request_context=RequestContext(org_id="0", storage_base_dir=str(tmp_path)),
    )
    service.service_config = PlaybookGenerationServiceConfig(
        request_id="test_request_id",
        agent_version="1.0",
        user_id="user-1",
        source="test",
        auto_run=False,
    )
    service.configurator.set_config_by_name(
        "user_playbook_extractor_config",
        PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt="Review grounded lessons",
        ),
    )
    service.request_context.prompt_manager = reviewer.request_context.prompt_manager
    source = _interaction_model(101, "Use width 800.")
    _storage(service).add_request(source.request)
    _storage(service).add_user_interaction("user-1", source.interactions[0])
    candidate = _candidate(
        101,
        "Use width 800.",
        content="Use the supplied width; never ask again because it causes delays.",
    )
    service._review_run = _playbook_run(user_id="user-1", source_interaction_ids=[101])
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    with transport, pytest.raises(StructuredOutputRepairError):
        service._resolve_write_plan([[candidate]])
    assert len(calls) == 2
    assert _storage(service).get_user_playbooks(status_filter=[None]) == []


def _mixed_batch():
    import copy

    candidates = [
        _candidate(
            101,
            "For future plots, use metric units.",
            content="Use metric units in future plots.",
        ),
        _candidate(
            102,
            "Use width 400.",
            content="Use the supplied width; never ask again because it causes delays.",
        ),
        _candidate(
            103,
            "The external system returned ready.",
            content="Say ready whenever the external system outputs ready.",
        ),
    ]
    response = {"assessments": [], "supporting_evidence": [], "decisions": []}
    for index, (decision, code, basis) in enumerate(
        [
            ("accept", "grounded_useful", "future_instruction"),
            ("revise", "unsupported_causality", "supplied_parameter"),
            ("reject", "not_agent_decision", "none"),
        ],
        1,
    ):
        packet = _response(
            kind="supplied_parameter_use" if index == 2 else "other",
            basis=basis,
            decision=decision,
            defects=[] if index == 1 else [code],
            code=code,
        )
        packet["assessments"][0]["id"] = f"C{index}"
        packet["assessments"][0]["original_rule_span"] = candidates[index - 1].content
        packet["decisions"][0]["id"] = f"C{index}"
        if index != 3:
            packet["decisions"][0]["evidence_ids"] = [f"C{index}-E1"]
            packet["supporting_evidence"][0]["id"] = f"C{index}"
            packet["supporting_evidence"][0]["excerpts"] = [
                {
                    "evidence_id": f"C{index}-E1",
                    "text": candidates[index - 1].source_span,
                }
            ]
        for key in response:
            response[key].extend(copy.deepcopy(packet[key]))
    return candidates, response


@pytest.mark.parametrize("damage", [None, "span", "quote"])
def test_real_mixed_batch_cannot_borrow_original_rule_or_evidence(damage):
    from reflexio.server.llm.litellm_client import StructuredOutputRepairError

    candidates, raw = _mixed_batch()
    if damage == "span":
        raw["assessments"][1]["original_rule_span"] = candidates[0].content
    if damage == "quote":
        raw["supporting_evidence"][1]["excerpts"][0]["text"] = candidates[0].source_span
    reviewer, transport, calls = _real_reviewer([raw, raw])
    interactions = []
    for i, candidate in enumerate(candidates):
        assert candidate.source_span is not None
        interactions.append(_interaction_model(101 + i, candidate.source_span))
    with transport:
        if damage:
            with pytest.raises(StructuredOutputRepairError):
                reviewer.decide(
                    candidates=candidates,
                    request_interaction_data_models=interactions,
                    existing_playbooks=[],
                    agent_context="",
                    playbook_definition="",
                    tool_context="",
                )
            assert len(calls) == 2
        else:
            outcome = reviewer.decide(
                candidates=candidates,
                request_interaction_data_models=interactions,
                existing_playbooks=[],
                agent_context="",
                playbook_definition="",
                tool_context="",
            )
            survivors = reviewer.apply_decisions(candidates=candidates, outcome=outcome)
            assert [item.source_interaction_ids for item in survivors] == [[101], [102]]
            assert (
                len(outcome.assessments) == 3 and len(outcome.supporting_evidence) == 2
            )
            assert survivors[0].content == candidates[0].content
            assert survivors[1].content == raw["decisions"][1]["revision"]["content"]
            with pytest.raises(ValueError, match="original candidate batch"):
                reviewer.apply_decisions(
                    candidates=list(reversed(candidates)), outcome=outcome
                )
