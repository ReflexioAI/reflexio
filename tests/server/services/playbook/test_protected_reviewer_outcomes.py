"""Protected review outcomes retain their actual child records and batch binding."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.input_preparation import (
    prepare_extraction_input,
)
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowCoordinator,
    current_overflow_coordinator,
)
from reflexio.server.services.extraction.protected_record import (
    ProtectedEvidence,
    ProtectedRecord,
    ProtectedRecordUpdate,
)
from reflexio.server.services.playbook.components.reviewer import (
    PlaybookCandidateReviewOutput,
)
from reflexio.server.services.storage.storage_base._extraction_stream import Window
from tests.server.services.playbook.test_playbook_reviewer import (
    _candidate,
    _interaction_model,
    _reviewer,
)
from tests.server.services.playbook.test_primary_assessed_playbook_reviewer import (
    _primary_response,
)


@pytest.mark.parametrize("version", ["1.3.0", "1.47.0", "1.48.0", "1.49.0"])
def test_protected_combined_review_binds_originals_and_preserves_children(version):
    groups = [_interaction_model(i, "Use width 800.") for i in (101, 102)]
    prepared = prepare_extraction_input(groups, user_id="user-1")
    spans = tuple(span for chunk in prepared.chunks for span in chunk.spans)
    candidates = [
        _candidate(i, "Use width 800.", content="Use the supplied width.")
        for i in (101, 102)
    ]
    record = ProtectedRecord().merge(
        ProtectedRecordUpdate(
            evidence=[
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(
                        next(s for s in spans if s.interaction_id == i).evidence_ref,
                    ),
                    quote="Use width 800.",
                    role="user",
                )
                for i in (101, 102)
            ]
        ),
        visible_spans=spans,
    )
    coordinator = OverflowCoordinator(
        MagicMock(),
        Window("w", "user-1", "playbook", "", 0, 2, [], {}, False, False),
        "inert",
        OverflowCoordinator.initial_manifest(prepared.source_digest),
        source_digest=prepared.source_digest,
        context_limits={},
        prepared=prepared,
        agent_context="Image resize jobs",
    )
    coordinator.record = record
    for candidate in candidates:
        coordinator.candidate_records[coordinator.playbook_key(candidate)] = record
    baseline = {
        "decisions": [
            {
                "id": "C1",
                "decision": "accept",
                "reason_code": "grounded_useful",
                "reason": None,
                "evidence_ids": ["C1-E1"],
                "revision": None,
            }
        ]
    }
    raw = baseline if version == "1.3.0" else _primary_response()
    coordinator.completed_step = Mock(
        return_value={
            "output": raw,
            "model_provenance": {"model_name": "cached-review-model"},
        }
    )
    coordinator.complete_step = Mock()
    reviewer, client = _reviewer(PlaybookCandidateReviewOutput.model_validate(baseline))
    reviewer.request_context.prompt_manager = PromptManager(
        version_override={reviewer.PROMPT_ID: version}
    )
    actual_children = []
    decide = reviewer.decide

    def capture_child(**kwargs):
        outcome = decide(**kwargs)
        if len(kwargs["candidates"]) == 1:
            actual_children.append(outcome)
        return outcome

    reviewer.decide = capture_child
    token = current_overflow_coordinator.set(coordinator)
    try:
        outcome = reviewer.decide(
            candidates=candidates,
            request_interaction_data_models=groups,
            existing_playbooks=[],
            agent_context="Image resize jobs",
            playbook_definition="Future image guidance",
            tool_context="",
        )
        assert outcome.original_candidates_sha256 is not None
        assert len(outcome.child_outcomes) == len(actual_children) == 2
        assert all(
            a is b for a, b in zip(outcome.child_outcomes, actual_children, strict=True)
        )
        assert [d.candidate_id for d in outcome.output.decisions] == ["C1", "C2"]
        assert [d.evidence_ids for d in outcome.output.decisions] == [
            ["C1-E1"],
            ["C2-E1"],
        ]
        assert (
            len(reviewer.apply_decisions(candidates=candidates, outcome=outcome)) == 2
        )
        mutated = deepcopy(candidates)
        mutated[0].notes = "Changed after review"
        for changed in (mutated, list(reversed(candidates))):
            with pytest.raises(ValueError):
                reviewer.apply_decisions(candidates=changed, outcome=outcome)
        assert not client.generate_chat_response.called
        assert not client.generate_chat_response_with_provenance.called
        if version != "1.3.0":
            assert outcome.primary_output is None
            for child in outcome.child_outcomes:
                assert child.primary_output is not None
                assert child.primary_output.assessments[0].candidate_id == "C1"
                assert len(child.assessments) == 1
                assert (
                    child.primary_output.supporting_evidence[0].excerpts[0].evidence_id
                    == "C1-E1"
                )
            assert outcome.supporting_evidence[1].candidate_id == "C2"
            assert outcome.supporting_evidence[1].excerpts[0].evidence_id == "C2-E1"
    finally:
        current_overflow_coordinator.reset(token)


@pytest.mark.parametrize("version", ["1.3.0", "1.47.0", "1.48.0", "1.49.0"])
def test_protected_review_cache_identity_records_only_declared_override(version):
    reviewer, client = _reviewer(PlaybookCandidateReviewOutput(decisions=[]))
    coordinator = MagicMock(spec=OverflowCoordinator)
    coordinator.completed_step.return_value = None
    client.generate_chat_response_with_provenance = Mock(
        return_value=SimpleNamespace(
            value=PlaybookCandidateReviewOutput(decisions=[]),
            provenance=ModelProvenance(model_name="offline"),
        )
    )
    messages = [{"role": "user", "content": "Offline cache identity control"}]
    reviewer._generate_review(
        messages=messages,
        response_format=PlaybookCandidateReviewOutput,
        validator=lambda *_: (),
        units_by_candidate={},
        version=version,
        coordinator=coordinator,
    )
    identity = {
        "messages": messages,
        "schema": PlaybookCandidateReviewOutput.model_json_schema(),
        "max_tokens": reviewer.max_tokens,
        "reasoning_effort": reviewer.reasoning_effort,
    }
    options = client.generate_chat_response_with_provenance.call_args.kwargs
    if version in ("1.48.0", "1.49.0"):
        identity["temperature_override"] = 0.0
        assert options["temperature"] == 0.0
    else:
        assert "temperature" not in options
    coordinator.step_key.assert_called_once_with("review-v1", identity)
