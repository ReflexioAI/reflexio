"""Native protected reviews keep source identity, actual raw children and seals."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import litellm
import pytest

from reflexio.models.api_schema.domain import entities
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    ProviderRequestGuardError,
)
from reflexio.server.llm.model_defaults import provider_env_vars
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
from reflexio.server.services.playbook.components.overflow_evidence import (
    EVIDENCE_SCOPE_POLICY,
)
from reflexio.server.services.playbook.components.reviewer import (
    PlaybookCandidateReviewer,
)
from reflexio.server.services.playbook.components.reviewer_native import (
    SurvivorReviewOutput,
)
from reflexio.server.services.storage.storage_base._extraction_stream import Window
from tests.server.services.playbook.test_playbook_reviewer import (
    _candidate,
    _interaction_model,
)

_ACTIONS = (
    "Use short paragraphs for future reports.",
    "List requested checks before taking action.",
)
_BAD_BENEFIT = "This eliminates network failures."
_OWNER = "private-owner-751-not-model-evidence"
_CANDIDATE_REQUEST = "private-generation-request-751-not-model-evidence"
_PRIVATE_NOTES = "Private governance annotation 751; not review evidence."
_EMBEDDING_VALUE = 98765.4321751


def _response(action, *, revise=False):
    """One child-local DTO; source entailment is deliberately uncomplicated."""
    return {
        "decisions": [
            {
                "id": "C1",
                "decision": "revise" if revise else "accept",
                "reason_code": "unsupported_causality" if revise else "grounded_useful",
                "defect_anchor": {"field": "content", "text": _BAD_BENEFIT}
                if revise
                else None,
                "survivor_witnesses": [
                    {
                        "kind": "user_learning",
                        "original_spans": [{"field": "content", "text": action}],
                        "excerpts": [{"evidence_id": "C1-E1", "text": action}],
                        "input_facts": None,
                    }
                ],
                "stored_match": None,
                "evidence_ids": ["C1-E1"],
                "revision": {
                    "content": action,
                    "trigger": "When the relevant future condition occurs",
                    "premise": {
                        "kind": "user_instruction",
                        "witness_index": 0,
                        "source": {"evidence_id": "C1-E1", "text": action},
                        "original_span": {"field": "content", "text": action},
                    },
                }
                if revise
                else None,
                "reason": None,
            }
        ]
    }


@pytest.fixture
def protected_native(monkeypatch):
    """Real coordinator/client/parser; only provider transport and storage are fake."""
    for name in provider_env_vars():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "unit-test-placeholder")
    groups = [_interaction_model(i, action) for i, action in enumerate(_ACTIONS, 101)]
    for group in groups:
        group.request.user_id = _OWNER
        for interaction in group.interactions:
            interaction.user_id = _OWNER
    prepared = prepare_extraction_input(
        groups, user_id=_OWNER, include_request_source=True
    )
    spans = tuple(span for chunk in prepared.chunks for span in chunk.spans)
    record = ProtectedRecord().merge(
        ProtectedRecordUpdate(
            evidence=[
                ProtectedEvidence(
                    kind="instruction",
                    evidence_refs=(span.evidence_ref,),
                    quote=span.text,
                )
                for span in spans
            ]
        ),
        visible_spans=spans,
    )
    storage = MagicMock()
    coordinator = OverflowCoordinator(
        storage,
        Window("native-protected", _OWNER, "playbook", "", 0, 2, [], {}, False, False),
        "test-owner",
        OverflowCoordinator.initial_manifest(prepared.source_digest),
        source_digest=prepared.source_digest,
        context_limits={},
        prepared=prepared,
        agent_context="Produce future reports and perform requested checks.",
    )
    coordinator.record = record

    def checkpoint(_window, _token, *, expected_revision, source_digest, **change):
        assert source_digest == coordinator.source_digest
        manifest = deepcopy(coordinator._manifest)
        assert expected_revision == manifest["revision"]
        manifest.update(deepcopy(change))
        manifest["revision"] += 1
        return manifest

    storage.checkpoint_extraction_overflow.side_effect = checkpoint
    client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_retries=0, fallback_models=[])
    )
    context = MagicMock()
    context.prompt_manager = PromptManager(
        version_override={"playbook_candidate_review": "1.51.0"}
    )
    reviewer = PlaybookCandidateReviewer(request_context=context, llm_client=client)
    wire = [_response(action) for action in _ACTIONS]
    requests = []

    def completion(**kwargs):
        requests.append(deepcopy(kwargs))
        payload = wire.pop(0)
        return litellm.ModelResponse(
            model="MiniMax-M3",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(payload)},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180},
        )

    monkeypatch.setattr(litellm, "completion", completion)
    parsed = []
    parse = client._maybe_parse_structured_output

    def observe_parse(*args, **kwargs):
        value = parse(*args, **kwargs)
        if isinstance(value, SurvivorReviewOutput):
            parsed.append(value)
        return value

    monkeypatch.setattr(client, "_maybe_parse_structured_output", observe_parse)

    def decide(*, batch=False, revise=False, existing=None):
        candidates = [
            _candidate(
                index,
                action,
                content=action + (" " + _BAD_BENEFIT if revise else ""),
            )
            for index, action in enumerate(_ACTIONS[: 2 if batch else 1], 101)
        ]
        for candidate in candidates:
            candidate.user_id = _OWNER
            candidate.request_id = _CANDIDATE_REQUEST
            candidate.user_playbook_id = 876543210
            candidate.notes = _PRIVATE_NOTES
            candidate.embedding = [_EMBEDDING_VALUE]
            coordinator.candidate_records[coordinator.playbook_key(candidate)] = record
        outcome = reviewer.decide(
            candidates=candidates,
            request_interaction_data_models=groups,
            existing_playbooks=existing or [],
            agent_context="Caller context is replaced by the protected owned context.",
            playbook_definition="Reusable user guidance",
            tool_context="",
        )
        return candidates, outcome

    token = current_overflow_coordinator.set(coordinator)
    try:
        yield SimpleNamespace(
            coordinator=coordinator,
            storage=storage,
            client=client,
            reviewer=reviewer,
            groups=groups,
            wire=wire,
            requests=requests,
            parsed=parsed,
            decide=decide,
        )
    finally:
        current_overflow_coordinator.reset(token)


@pytest.mark.parametrize("revise", [False, True])
def test_protected_native_single_preserves_actual_raw_and_publication_policy(
    protected_native, revise
):
    fixture = protected_native
    fixture.wire[:] = [_response(_ACTIONS[0], revise=revise)]
    candidates, outcome = fixture.decide(revise=revise)
    assert outcome.native_output is fixture.parsed[-1]
    assert outcome.original_candidates_sha256
    assert outcome.validated_projection_sha256
    original = candidates[0].model_dump(mode="json")
    survivors = fixture.reviewer.apply_decisions(candidates=candidates, outcome=outcome)
    assert candidates[0].model_dump(mode="json") == original
    assert len(survivors) == (0 if revise else 1)
    assert outcome.native_output.decisions[0].decision == (
        "revise" if revise else "accept"
    )
    messages = fixture.requests[0]["messages"]
    assert [item["role"] for item in messages] == ["system", "system", "user"]
    assert messages[1]["content"] == EVIDENCE_SCOPE_POLICY
    assert original["rationale"] in messages[-1]["content"]
    assert "protected_source_origins" in messages[-1]["content"]
    assert fixture.requests[0]["temperature"] == 0
    if survivors:
        assert survivors[0].content == original["content"]
        assert survivors[0].rationale == original["rationale"]
        assert survivors[0].source_interaction_ids == [101]
        assert survivors[0].source_span == _ACTIONS[0]


def test_protected_batch_retains_actual_child_local_ids_and_merges_only_public_aliases(
    protected_native,
):
    fixture = protected_native
    candidates, outcome = fixture.decide(batch=True)
    assert len(fixture.parsed) == len(outcome.child_outcomes) == 2
    assert outcome.native_output is None
    for child, parsed in zip(outcome.child_outcomes, fixture.parsed, strict=True):
        assert child.native_output is parsed
        assert parsed.decisions[0].id == "C1"
        assert parsed.decisions[0].evidence_ids == ("C1-E1",)
    assert [decision.candidate_id for decision in outcome.output.decisions] == [
        "C1",
        "C2",
    ]
    assert [decision.evidence_ids for decision in outcome.output.decisions] == [
        ["C1-E1"],
        ["C2-E1"],
    ]
    assert [unit.interaction_id for unit in outcome.units_by_candidate["C2"]] == [102]
    assert (
        len(fixture.reviewer.apply_decisions(candidates=candidates, outcome=outcome))
        == 2
    )


def test_native_dispatch_keeps_private_binding_fields_out_of_model_payload(
    protected_native,
):
    fixture = protected_native
    stored = _candidate(999, "Stored source.", content="A distinct stored procedure.")
    stored.user_id = _OWNER
    stored.user_playbook_id = 900
    stored.request_id = _CANDIDATE_REQUEST
    stored.notes = _PRIVATE_NOTES
    stored.embedding = [_EMBEDDING_VALUE]
    fixture.decide(existing=[stored])
    text = fixture.requests[0]["messages"][-1]["content"]
    payload = json.loads(text)
    assert set(payload["candidates"][0]) == {
        "id",
        "content",
        "trigger",
        "rationale",
        "reader_angle",
        "own_source_units",
    }
    assert set(payload["candidates"][0]["own_source_units"][0]) == {
        "evidence_id",
        "turn_ref",
        "role",
        "request_source",
        "source_span",
    }
    assert set(payload["existing"][0]) == {"id", "content", "trigger", "rationale"}
    assert "source_sha256" not in text
    for private in (
        _OWNER,
        _CANDIDATE_REQUEST,
        _PRIVATE_NOTES,
        str(_EMBEDDING_VALUE),
        "876543210",
    ):
        assert private not in text
    # Legacy protected serialization keeps its owned source-origin references.
    assert "protected_record" in text
    assert "protected_source_origins" in text


def test_native_cache_hit_reparses_and_revalidates_foreign_source(
    protected_native, monkeypatch
):
    fixture = protected_native
    clock = Mock()
    clock.now.return_value = datetime.fromtimestamp(1_700_000_000, UTC)
    monkeypatch.setattr(entities, "datetime", clock)
    first_candidates, first = fixture.decide()
    clock.now.return_value = datetime.fromtimestamp(1_700_000_002, UTC)
    cached_candidates, cached = fixture.decide()
    assert first_candidates[0].created_at == cached_candidates[0].created_at
    assert len(fixture.requests) == 1
    assert cached.native_output == first.native_output
    assert cached.native_output is not first.native_output
    steps = fixture.coordinator._manifest["checkpoint"]["completed_steps"]
    assert len(steps) == 1
    saved = next(iter(steps.values()))["output"]
    saved["decisions"][0]["survivor_witnesses"][0]["excerpts"][0]["text"] = (
        "A foreign source never supplied this quotation."
    )
    writes = fixture.storage.checkpoint_extraction_overflow.call_count
    with pytest.raises(ValueError):
        fixture.decide()
    assert len(fixture.requests) == 1
    assert fixture.storage.checkpoint_extraction_overflow.call_count == writes


@pytest.mark.parametrize("changed", ["role", "catalog", "profile"])
def test_native_cache_identity_refuses_stale_source_catalog_or_profile(
    protected_native, changed
):
    fixture = protected_native
    fixture.decide()
    fixture.wire[:] = [_response(_ACTIONS[0])]
    existing = []
    if changed == "role":
        fixture.groups[0].interactions[0].role = "assistant"
    elif changed == "catalog":
        existing = [_candidate(999, "Stored distinct source.", content="Stored rule.")]
        existing[0].user_playbook_id = 900
        existing[0].user_id = _OWNER
    else:
        fixture.client.config.timeout += 1
    try:
        fixture.decide(existing=existing)
    except ProviderRequestGuardError:
        assert changed == "role"
    else:
        assert len(fixture.requests) == 2


def test_native_admission_binding_drift_stops_before_fallback(protected_native):
    fixture = protected_native
    fixture.client.config.fallback_models = ["zai/glm-5.2"]
    admissions = []

    def mutate_source(params, *_args):
        admissions.append(params["model"])
        fixture.groups[0].interactions[0].content = "Changed after owned preparation."

    fixture.reviewer.provider_request_guard = mutate_source
    with pytest.raises(ProviderRequestGuardError):
        fixture.decide()
    assert len(admissions) == 1
    assert fixture.requests == []
    assert fixture.coordinator.checkpoint.get("completed_steps", {}) == {}


@pytest.mark.parametrize(
    "mutation",
    ["public", "unit", "raw", "children", "child_public", "missing_seal", "original"],
)
def test_native_seal_rejects_corruption_before_apply_or_coordinator_writes(
    protected_native, monkeypatch, mutation
):
    fixture = protected_native
    candidates, outcome = fixture.decide(batch=True)
    if mutation == "public":
        outcome.output.decisions[0].reason = "Unvalidated replacement explanation."
    elif mutation == "unit":
        unit = outcome.units_by_candidate["C1"][0]
        outcome.units_by_candidate["C1"][0] = replace(unit, role="assistant")
    elif mutation == "raw":
        raw = outcome.child_outcomes[0].native_output
        assert raw is not None
        raw.__dict__["decisions"] = ()
    elif mutation == "children":
        outcome = replace(
            outcome, child_outcomes=tuple(reversed(outcome.child_outcomes))
        )
    elif mutation == "child_public":
        outcome.child_outcomes[0].output.decisions[
            0
        ].reason = "Changed child projection."
    elif mutation == "missing_seal":
        outcome = replace(outcome, validated_projection_sha256=None)
    else:
        candidates.reverse()
    apply = Mock(wraps=fixture.reviewer._apply_decisions)
    monkeypatch.setattr(fixture.reviewer, "_apply_decisions", apply)
    before = deepcopy(fixture.coordinator.checkpoint)
    writes = fixture.storage.checkpoint_extraction_overflow.call_count
    with pytest.raises(ValueError):
        fixture.reviewer.apply_decisions(candidates=candidates, outcome=outcome)
    apply.assert_not_called()
    assert fixture.coordinator.checkpoint == before
    assert fixture.storage.checkpoint_extraction_overflow.call_count == writes
