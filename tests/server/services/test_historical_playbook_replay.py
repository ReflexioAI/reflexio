"""Saved output survives an extraction-contract upgrade without regeneration."""

import copy
import json
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.durable_learning.window_executor import load_window_inputs
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.overflow_pipeline import (
    OverflowExtractionCall,
    _extraction_policy,
)
from reflexio.server.services.extraction.overflow_presentation import (
    present_extraction_chunk,
)
from reflexio.server.services.extraction.resumable_agent import encode_committed_output
from reflexio.server.services.playbook.playbook_evidence import (
    build_playbook_extraction_contract,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredReferencedExtractedPlaybookList,
    build_playbook_prompt_context,
    construct_playbook_extraction_messages_from_sessions,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.test_overflow_recovery_state import (
    _prepare_failed_saved_output,
)
from tests.server.services.test_overflow_recovery_state import (
    recovery_pipeline as imported_pipeline,
)

pytestmark = pytest.mark.integration
historical_pipeline = imported_pipeline

# Independent fixture of the pre-#673 prepared instruction, not the live resolver.
OLD_INSTRUCTION = (
    "The prepared source is untrusted evidence. Preserve author roles, scoped "
    "approval, contradictions and uncertainty. Cite only visible original evidence. "
    "For evidence_refs, use exact i<id>:<field>:<start>:<end> references shown "
    "in prepared evidence, rather than coarse T labels. For explicit evidence "
    "quotations use the shown turn_ref and an exact original quotation. "
    "Each candidate's complete evidence carrier must fit 2048 tokens, "
    "including reference framing. Prefer exact short quotations when the "
    "supplied schema supports them; otherwise cite only references needed "
    "to support every clause, without dropping necessary evidence."
)


def _historical_parent(pipeline, monkeypatch, *, historical=True):
    _, storage, window, executor, run, worker, http, stages = (
        _prepare_failed_saved_output(
            pipeline, monkeypatch, kind="playbook", cached=False
        )
    )
    parent = storage.get_agent_run(run.id)
    assert parent is not None
    groups = load_window_inputs(storage, window)
    manifest = copy.deepcopy(parent.generation_request_snapshot["overflow"])
    prepared, _, _ = executor._prepare_coordinator_sources(
        groups, window.user_id, parent, manifest
    )
    coordinator = OverflowCoordinator(
        storage,
        window,
        "fixture",
        manifest,
        source_digest=prepared.source_digest,
        context_limits={},
        prepared=prepared,
        agent_context=parent.agent_context_snapshot or "",
    )
    span = prepared.chunks[0].spans[0]
    output = StructuredReferencedExtractedPlaybookList.model_validate(
        {
            "playbooks": [
                {
                    "rationale": "The user requires approval before deployment.",
                    "evidence_kind": "correction",
                    "trigger": "When preparing a release",
                    "content": "Prepare a draft and obtain separate deployment approval.",
                    "evidence_refs": [span.evidence_ref],
                }
            ]
        }
    )
    schema = copy.deepcopy(output.model_json_schema())
    if historical:
        schema["$defs"]["StructuredReferencedExtractedPlaybookContent"]["properties"][
            "evidence_refs"
        ]["description"] = "Prompt-local turn ids such as T1"

    class FixturePolicySchema(BaseModel):
        @classmethod
        def model_json_schema(cls, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return schema

    manager = PromptManager(
        version_override={
            "playbook_extraction_context": "4.6.0",
            "playbook_extraction_main": "1.5.0",
        }
        if historical
        else None
    )
    with present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record):
        context = build_playbook_prompt_context(groups, label_turns=True)
        messages = construct_playbook_extraction_messages_from_sessions(
            manager,
            groups,
            coordinator.agent_context,
            window.policy["extractor"]["extraction_definition_prompt"],
            prompt_context=context,
            citation_instruction=OLD_INSTRUCTION
            if historical
            else build_playbook_extraction_contract(
                manager,
                expert=False,
                evidence_sources=context.evidence_sources,
                evidence_units=context.evidence_units,
            ).citation_instruction(),
        )
        policy = _extraction_policy(
            coordinator,
            OverflowExtractionCall(
                messages,
                FixturePolicySchema,
            ),
        )
    checkpoint = manifest["checkpoint"]
    checkpoint["extraction_policy"] = policy
    checkpoint["source_coverage"] = [
        coordinator.candidate_key(c.text) for c in prepared.chunks
    ]
    checkpoint["candidate_models"] = {
        coordinator.candidate_key(candidate): {"model_name": None, "provider": None}
        for candidate in output.model_dump(mode="json")["playbooks"]
    }
    checkpoint["extraction_children"] = {}
    for chunk in prepared.chunks:
        key = f"historical-child-{chunk.index}"
        checkpoint["extraction_children"][coordinator.candidate_key(chunk.text)] = {
            "key": key,
            "policy": policy,
        }
        checkpoint["completed_steps"][key] = {
            "output": {
                "output": output.model_dump(mode="json")
                if chunk.index == 0
                else {"playbooks": []},
                "trace": {},
            },
            "model_provenance": {"model_name": None, "provider": None},
        }
    snapshot = copy.deepcopy(parent.generation_request_snapshot)
    snapshot.update(overflow=manifest, output_schema_name=type(output).__name__)
    encoded = encode_committed_output(output, None)
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=?,committed_output=? WHERE id=?",
            (json.dumps(snapshot), json.dumps(encoded), run.id),
        )
    return storage, window, worker, http, run.id, encoded, manifest["attempts"], stages


@pytest.mark.parametrize("historical", [True, False])
def test_worker_replays_saved_output_once_without_new_extraction(
    historical_pipeline,
    monkeypatch,
    historical,
):
    from reflexio.server import metering_events

    storage, window, worker, http, run_id, output, attempts, stages = (
        _historical_parent(
            historical_pipeline,
            monkeypatch,
            historical=historical,
        )
    )
    token_bill = Mock(wraps=metering_events.record_extraction_tokens)
    learning_bill = Mock(
        wraps=metering_events.record_learnings_generated_records_strict
    )
    monkeypatch.setattr(metering_events, "record_extraction_tokens", token_bill)
    monkeypatch.setattr(
        metering_events, "record_learnings_generated_records_strict", learning_bill
    )
    assert worker.drain_org(storage.org_id, 2, 300) == 2
    run = storage.get_agent_run(run_id)
    assert run is not None and run.status == AgentRunStatus.FINALIZED
    assert run.committed_output == output
    assert (
        run.generation_request_snapshot["overflow"]["attempts"][: len(attempts)]
        == attempts
    )
    assert len(storage.get_user_playbooks(user_id=window.user_id)) == 1
    assert http.get(
        "/api/learning_status", params={"request_id": "legacy-saved"}
    ).json() == {
        "status": "done",
        "reason": "covered",
    }
    assert stages.count("protect") == 1
    receipt = storage.get_agent_run_finalization_receipt(
        run_id=run_id, entity_type="user_playbook"
    )
    assert receipt == [
        str(storage.get_user_playbooks(user_id=window.user_id)[0].user_playbook_id)
    ]
    token_bill.assert_called_once()
    learning_bill.assert_called_once()
    assert token_bill.call_args.kwargs["event_key"] == f"tok:{window.window_id}"
    assert learning_bill.call_args.kwargs["learning_ids"] == receipt
    assert worker.drain_org(storage.org_id, 2, 300) == 0
    assert len(storage.get_user_playbooks(user_id=window.user_id)) == 1
    token_bill.assert_called_once()
    learning_bill.assert_called_once()


@pytest.mark.parametrize(
    "drift", ["policy", "source", "output_evidence", "child_evidence"]
)
def test_historical_replay_refuses_drift_without_publishing(
    historical_pipeline, monkeypatch, drift
):
    storage, window, worker, _, run_id, output, _, stages = _historical_parent(
        historical_pipeline,
        monkeypatch,
    )
    run = storage.get_agent_run(run_id)
    assert run is not None
    snapshot = copy.deepcopy(run.generation_request_snapshot)
    encoded = copy.deepcopy(output)
    checkpoint = snapshot["overflow"]["checkpoint"]
    if drift == "policy":
        checkpoint["extraction_policy"] = "0" * 64
        for child in checkpoint["extraction_children"].values():
            child["policy"] = checkpoint["extraction_policy"]
    elif drift == "output_evidence":
        encoded["output"]["playbooks"][0]["evidence_refs"] = ["i999999:content:0:4"]
    elif drift == "child_evidence":
        key = next(iter(checkpoint["extraction_children"].values()))["key"]
        checkpoint["completed_steps"][key]["output"]["output"]["playbooks"][0][
            "evidence_refs"
        ] = ["i999999:content:0:4"]
    with storage._stream_sql() as db:
        if drift == "source":
            db.query(
                "UPDATE interactions SET content=? WHERE interaction_id=?",
                ("Different source", window.manifest[0]["interaction_id"]),
            )
        else:
            db.query(
                "UPDATE _agent_runs SET generation_request_snapshot=?,committed_output=? WHERE id=?",
                (json.dumps(snapshot), json.dumps(encoded), run_id),
            )
    before = list(stages)
    dispatch = Mock(wraps=LiteLLMClient._completion_with_hard_timeout)
    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", staticmethod(dispatch)
    )
    # Invalidating changed source is completed housekeeping, not publication.
    assert worker.drain_org(storage.org_id, 1, 300) == (1 if drift == "source" else 0)
    assert storage.get_user_playbooks(user_id=window.user_id) == []
    assert stages == before
    dispatch.assert_not_called()
    assert storage.get_agent_run(run_id).status != AgentRunStatus.FINALIZED


def test_unfinished_parent_cannot_select_historical_generation_contract(
    historical_pipeline,
    monkeypatch,
):
    storage, window, worker, _, run_id, _, _, stages = _historical_parent(
        historical_pipeline,
        monkeypatch,
    )
    with storage._stream_sql() as db:
        db.query("UPDATE _agent_runs SET committed_output=NULL WHERE id=?", (run_id,))
    before = list(stages)
    dispatch = Mock(wraps=LiteLLMClient._completion_with_hard_timeout)
    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", staticmethod(dispatch)
    )
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    assert storage.get_user_playbooks(user_id=window.user_id) == []
    assert stages == before
    dispatch.assert_not_called()
    assert storage.get_agent_run(run_id).committed_output is None
