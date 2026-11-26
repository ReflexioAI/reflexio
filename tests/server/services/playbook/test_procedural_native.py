"""The reviewed final plan survives retirement, replay and publication fencing."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.input_budget import ExtractionInputIntegrityError
from reflexio.server.services.base_generation_service import PreparedGenerationRun
from reflexio.server.services.deferred_learning_plan import GenerationComputePlan
from reflexio.server.services.extraction.input_preparation import (
    prepare_extraction_input,
)
from reflexio.server.services.extraction.resumable_agent import encode_committed_output
from reflexio.server.services.playbook.procedural import native, workflow
from reflexio.server.services.playbook.procedural.models import (
    Input,
    Observation,
    Source,
)
from reflexio.server.services.write_plan_codec import encode_write_plan
from tests.server.services.playbook.test_source_quoted_overflow import sources


@pytest.mark.parametrize(
    "field", ["interacted_image_url", "image_encoding", "expert_content"]
)
def test_visual_and_expert_inputs_keep_their_independent_capability(field):
    from reflexio.server.services.playbook.procedural.input_scope import (
        is_ordinary_text,
    )

    groups = sources()
    setattr(groups[0].interactions[0], field, "independent input")
    assert not is_ordinary_text(groups)


@pytest.mark.parametrize(
    "field,value",
    [
        ("shadow_content", "Alternative textual response"),
        ("user_action_description", "Clicked approve"),
        ("retrieved_learnings", [{"kind": "user_playbook", "learning_id": "7"}]),
        ("citations", [{"kind": "user_playbook", "real_id": "7"}]),
        (
            "tools_used",
            [{"tool_name": "inspect", "tool_data": {"result": "source text"}}],
        ),
    ],
)
def test_textual_evidence_and_retrieval_metadata_do_not_select_an_old_engine(
    field, value
):
    from reflexio.models.api_schema.domain.entities import Interaction
    from reflexio.server.services.playbook.procedural.input_scope import (
        is_ordinary_text,
    )

    groups = sources()
    groups[0].interactions[0] = Interaction.model_validate(
        {**groups[0].interactions[0].model_dump(), field: value}
    )
    assert is_ordinary_text(groups)


def test_catalog_field_context_does_not_rewrite_source_witnesses():
    from reflexio.server.services.playbook.procedural.catalog import catalog

    pieces = ["Require approval.", "click", "An alternative proposed response"]
    value = Input(
        owner="u",
        sources=[
            Source(
                interaction_id=1,
                request_id="r",
                owner="u",
                role="user",
                request_source="test",
                text="\n\n".join(pieces),
                piece_fields=["content", "user_action", "shadow_content"],
            )
        ],
        existing=[],
        agent_context="",
        definition="Learn procedures",
        tool_context="",
    )
    body, quotes, _ = catalog(value, {1: pieces})
    segments = body["sources"][0]["segments"]
    assert "field" not in segments[0]
    assert segments[1]["field"] == "user_action"
    assert segments[2]["field"] == "shadow_content"
    assert [quote.text for quote in quotes.values()] == pieces
    assert "source_field_context" in body
    assert "piece_fields" not in value.model_dump(mode="json")["sources"][0]


def test_pure_content_catalog_is_identical_without_or_with_explicit_content_label():
    from reflexio.server.services.playbook.procedural.catalog import catalog

    value = Input(
        owner="u",
        sources=[
            Source(
                interaction_id=1,
                request_id="r",
                owner="u",
                role="user",
                request_source="test",
                text="Require approval.",
            )
        ],
        existing=[],
        agent_context="",
        definition="Learn procedures",
        tool_context="",
    )
    original = catalog(value, {1: ["Require approval."]})
    value.sources[0].piece_fields = ["content"]
    assert catalog(value, {1: ["Require approval."]}) == original
    assert "source_field_context" not in original[0]


def receipt(*, retire=False):
    records = (
        {}
        if retire
        else {
            "W0": {
                "content": "Require approval before deployment.",
                "trigger": "Before deploying a service.",
                "expanded_terms": None,
                "source_interaction_ids": [1],
                "source_span": "Require approval.",
                "replaces": [7],
            }
        }
    )
    retired = [7] if retire else []
    return {
        "method_version": workflow.VERSION,
        "rendered": records,
        "accepted": list(records),
        "plan": {"selected": list(records), "keep_existing": []},
        "retired_ids": retired,
        "reviewed_sha256": workflow.digest(
            {"records": records, "keep_existing": [], "retired_ids": retired}
        ),
        "old_state": {"7": "original-version"},
        "outcome": "accept",
    }


def plan(value):
    return native.receipt_to_plan(
        value,
        user_id="u",
        agent_version="v",
        request_id="r",
        source="test",
        output_pending_status=False,
        skip_aggregation=True,
    )


def test_retirement_only_is_a_real_zero_learning_native_plan():
    value = plan(receipt(retire=True))
    assert value is not None
    assert value.new_playbooks == []
    assert value.superseded_ids == [7]
    assert value.merge_groups == []


def test_native_plan_has_exact_reviewed_rows_and_lineage():
    value = plan(receipt())
    assert value is not None
    assert value.merge_groups == [(0, [7])]
    assert value.lineage_contexts[0].source_ids == ["7"]
    assert value.new_playbooks[0].user_id == "u"
    assert value.new_playbooks[0].expanded_terms is None
    assert value.new_playbooks[0].source_interaction_ids == [1]


def test_a_changed_reviewed_result_cannot_become_a_native_plan():
    value = receipt()
    value["rendered"]["W0"]["content"] = "Deploy without approval."
    with pytest.raises(ExtractionInputIntegrityError, match="result changed"):
        plan(value)


def test_retirement_target_must_have_been_an_original_historical_input():
    value = receipt(retire=True)
    value["old_state"] = {"8": "other"}
    with pytest.raises(ExtractionInputIntegrityError, match="original history"):
        plan(value)


@pytest.mark.parametrize("changed", [None, "scope", "method", "schema", "version"])
def test_v5_requires_the_exact_procedural_parent_contract(changed):
    run = SimpleNamespace(
        id="window:w",
        project_id="project",
        binding=SimpleNamespace(
            org_id="org",
            user_id="u",
            extractor_kind="playbook",
            request_id="r",
            agent_version="v",
        ),
        generation_request_snapshot={
            "output_schema_name": native.ProceduralOutput.__name__
        },
    )
    manifest = {
        "preparation_version": native.PREPARATION_VERSION,
        "checkpoint": {native.KEY: native.initial_contract(run)},
    }
    if changed == "scope":
        manifest["checkpoint"][native.KEY]["scope"]["project_id"] = "other-project"
    elif changed == "method":
        manifest["checkpoint"][native.KEY]["method"] = "unqualified-method"
    elif changed == "schema":
        run.generation_request_snapshot["output_schema_name"] = "OtherOutput"
    elif changed == "version":
        manifest["preparation_version"] = 3
    if changed is None:
        native.require_preparation_contract(run, manifest)
    else:
        with pytest.raises(ExtractionInputIntegrityError, match="contract changed"):
            native.require_preparation_contract(run, manifest)


def observation_models() -> dict[str, type[BaseModel]]:
    return {"record_observation": Observation}


def journal_fixture() -> tuple[Any, Any, Input, dict[str, Any]]:
    # This fake is deliberately confined to the coordinator-protocol unit seam;
    # separate HTTP/SQLite tests exercise the real ownership and publication.
    data = Input(
        owner="u",
        sources=[
            Source(
                interaction_id=1,
                request_id="r",
                owner="u",
                role="user",
                request_source="test",
                text="Require approval.",
            )
        ],
        existing=[],
        agent_context="",
        definition="Learn procedures",
        tool_context="",
    )
    saved, checkpoint = {}, {native.KEY: {"reader_steps": {}}}
    coordinator = SimpleNamespace(
        step_key=lambda stage, inputs: workflow.digest(
            {"stage": stage, "inputs": inputs}
        ),
        validate_current_parent=Mock(),
        completed_step=lambda key: deepcopy(saved.get(key)),
        complete_step=lambda key, output, provenance: saved.update(
            {
                key: {
                    "output": output,
                    "model_provenance": {
                        "model_name": provenance.model_name,
                        "provider": provenance.provider,
                    },
                }
            }
        ),
        checkpoint=checkpoint,
        candidate_key=workflow.digest,
        require_prepared=lambda: SimpleNamespace(
            chunks=[SimpleNamespace(text="original-full-chunk")]
        ),
        save_checkpoint=lambda value: checkpoint.update(deepcopy(value)),
    )
    delegate = SimpleNamespace(
        profiles={"source": {"model": "test-model"}},
        model_ladders={"source": ["test-model"]},
        latest_provenance=ModelProvenance(model_name="test-model", provider="test"),
        ask=Mock(
            return_value=[
                Observation(
                    topic="Deployment",
                    evidence="EXCERPT_1_1_1",
                    observation="Approval is required.",
                )
            ]
        ),
    )
    return coordinator, delegate, data, saved


def test_a_completed_source_stage_replays_without_a_model_call():
    coordinator, delegate, data, _ = journal_fixture()
    journal = native.JournalClient(
        coordinator, delegate, data, {1: ["Require approval."]}
    )
    options = (
        "source",
        "reader prompt",
        {"sources": ["Require approval."]},
        observation_models(),
        lambda batch: batch,
        lambda _result: None,
    )
    first = journal.ask(*options, evidence_choices=["EXCERPT_1_1_1"])
    second = journal.ask(*options, evidence_choices=["EXCERPT_1_1_1"])
    assert first == second
    assert delegate.ask.call_count == 1
    assert len(coordinator.checkpoint[native.KEY]["reader_steps"]) == 1
    assert journal.latest_provenance is not None
    assert journal.latest_provenance.model_name == "test-model"


def test_recovered_reader_output_still_requires_available_original_evidence():
    coordinator, delegate, data, saved = journal_fixture()
    journal = native.JournalClient(
        coordinator, delegate, data, {1: ["Require approval."]}
    )
    options = (
        "source",
        "reader prompt",
        {},
        observation_models(),
        lambda batch: batch,
        lambda _result: None,
    )
    journal.ask(*options)
    next(iter(saved.values()))["output"][0]["evidence"] = "EXCERPT_missing"
    with pytest.raises(ValueError, match="available witness"):
        journal.ask(*options)
    assert delegate.ask.call_count == 1


def test_a_terminal_method_failure_is_not_redrawn_on_resume():
    from reflexio.server.services.playbook.procedural.transport import (
        IncompleteBatchError,
    )

    coordinator, delegate, data, _ = journal_fixture()
    delegate.ask.side_effect = IncompleteBatchError("Cannot finish this batch")
    journal = native.JournalClient(
        coordinator, delegate, data, {1: ["Require approval."]}
    )
    options = (
        "source",
        "reader prompt",
        {},
        observation_models(),
        lambda batch: batch,
        lambda _value: None,
    )
    with pytest.raises(IncompleteBatchError):
        journal.ask(*options)
    with pytest.raises(
        IncompleteBatchError, match="Retained procedural method failure"
    ):
        journal.ask(*options)
    assert delegate.ask.call_count == 1


@pytest.mark.parametrize("retire", [False, True])
def test_publication_requires_the_exact_retained_plan_and_full_reader_coverage(retire):
    prepared = prepare_extraction_input(sources(), user_id="u")
    coverage = [workflow.digest(chunk.text) for chunk in prepared.chunks]
    value = plan(receipt(retire=retire))
    assert value is not None
    encoded = encode_write_plan(value)
    output = native.ProceduralOutput(
        source_digest=prepared.source_digest,
        source_coverage=coverage,
        historical_playbooks={"7": "original-state"},
        receipt=receipt(retire=retire),
        write_plan=encoded,
        plan_sha256=workflow.digest(encoded),
    )
    checkpoint = {
        native.KEY: {
            "output": output.model_dump(mode="json"),
            "reader_steps": dict.fromkeys(coverage, "reader-key"),
        },
        "historical_playbooks": output.historical_playbooks,
    }
    run = SimpleNamespace(
        id="run", committed_output=encode_committed_output(output, None)
    )
    coordinator: Any = SimpleNamespace(
        checkpoint=checkpoint,
        source_digest=prepared.source_digest,
        require_prepared=lambda: prepared,
        candidate_key=workflow.digest,
        completed_step=lambda key: {"output": []} if key == "reader-key" else None,
        saved_run=run,
        window=SimpleNamespace(user_id="u"),
    )
    service = SimpleNamespace(
        service_config=SimpleNamespace(request_id="r", agent_version="v"),
        output_pending_status=False,
        skip_aggregation=True,
    )
    count = len(value.new_playbooks)
    computed = GenerationComputePlan(
        prepared=PreparedGenerationRun({}, "playbook", "u"),
        generated_count=count,
        billable_count=count,
        write_plan=value,
        bookmark_advance=None,
        generation_start=0,
        extraction_run_ids=["run"],
        token_totals=None,
    )
    native.validate_window_plan(coordinator, service, computed)
    if retire:
        value.superseded_ids.append(8)
    else:
        value.new_playbooks[0].content = "Changed after review"
    with pytest.raises(ExtractionInputIntegrityError, match="after review"):
        native.validate_window_plan(coordinator, service, computed)
    computed.write_plan = native.decode_write_plan(encoded, "playbook")
    checkpoint[native.KEY]["reader_steps"] = {}
    with pytest.raises(ExtractionInputIntegrityError, match="source/state binding"):
        native.validate_window_plan(coordinator, service, computed)
