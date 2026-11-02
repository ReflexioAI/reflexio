"""Real owner/storage source proof; provider output alone is scripted."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.durable_learning.user_lease import (
    adopt_agent_run_for_execution,
    agent_run_user_work,
    owned_agent_run_write,
)
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.extraction.history_inputs import HistoricalInputs
from reflexio.server.services.storage.storage_base import LeaseLostError
from tests.server.services.extraction import (
    test_resume_publication_integration as cases,
)

storage = cases.storage
request_context = cases.request_context
explicit_test_billing_exemption = cases.explicit_test_billing_exemption
no_background_tagging = cases.no_background_tagging
publication = cases.publication
shared_worker = cases.shared_worker
pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "change",
    [
        "same",
        "request",
        "interaction",
        "deleted",
        "user",
        "derived",
        "fallback",
        "wrong_output",
        "missing_evidence",
        "missing_source",
        "extra_source",
        "wrong_project",
        "malformed_digest",
    ],
)
def test_source_proof_validates_full_captured_input_and_rolls_back(
    shared_worker: DurableLearningWorker, publication, monkeypatch, change
):
    storage, worker, kind, _, _ = publication
    getter = storage.get_request
    if change == "fallback":
        monkeypatch.setattr(storage, "get_request", lambda _id: None)

    def complete(**kwargs):
        if change == "fallback":
            # A real request now has different qualifiers. Validation must use
            # the actual pre-model fallback, not switch to this later row.
            storage._execute(
                "UPDATE requests SET session_id='later-real-session' WHERE request_id='request_1'"
            )
            monkeypatch.setattr(storage, "get_request", getter)
        return cases._loop_result(kwargs, kind, storage, nonempty=False)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    saved = storage.get_agent_run("run_1")
    assert saved is not None and saved.committed_output is not None
    if change == "request":
        storage._execute(
            "UPDATE requests SET session_id='replaced-session' WHERE request_id='request_1'"
        )
    elif change == "interaction":
        storage._execute(
            "UPDATE interactions SET content='replaced-source' WHERE interaction_id=2"
        )
    elif change == "deleted":
        storage._execute("DELETE FROM interactions WHERE interaction_id=2")
    elif change == "user":
        storage._execute(
            "UPDATE interactions SET user_id='foreign-user' WHERE interaction_id=2"
        )
    elif change == "derived":
        storage._execute(
            "UPDATE interactions SET token_count=23,embedding=? WHERE interaction_id=2",
            (json.dumps([0.5] * 512),),
        )
    elif change in {
        "missing_evidence",
        "missing_source",
        "extra_source",
        "wrong_project",
        "malformed_digest",
    }:
        output = deepcopy(saved.committed_output)
        frame = output["publication"]["source_inputs"]
        if change == "missing_evidence":
            del output["publication"]["source_inputs"]
        elif change == "missing_source":
            del frame["sources"]["2"]
        elif change == "extra_source":
            frame["sources"]["9999"] = deepcopy(frame["sources"]["1"])
        elif change == "wrong_project":
            frame["project_id"] = "foreign-project"
        else:
            frame["sources"]["2"]["digest"] = "not-a-sha256"
        storage._execute(
            "UPDATE _agent_runs SET committed_output=? WHERE id='run_1'",
            (json.dumps(output),),
        )
        saved = storage.get_agent_run("run_1")
        assert saved is not None
    with agent_run_user_work(storage, saved):
        current = adopt_agent_run_for_execution(storage, saved)
        before = storage.get_agent_run("run_1")
        if change == "wrong_output":
            assert current.committed_output is not None
            output = deepcopy(current.committed_output)
            output["output"] = {"forged": True}
            prepared = replace(current, committed_output=output)
        else:
            prepared = current
        if change in {"same", "derived", "fallback"}:
            storage.validate_agent_output_sources(prepared)
        else:
            error = (
                LeaseLostError
                if change == "wrong_output"
                else InputBudgetConfigurationError
            )
            with pytest.raises(error), owned_agent_run_write(storage, current):
                storage._execute(
                    "UPDATE _agent_runs SET last_error='must-rollback' WHERE id='run_1'"
                )
                storage.validate_agent_output_sources(prepared)
        assert storage.get_agent_run("run_1") == before


@pytest.mark.parametrize("entry", ["start", "resume"])
def test_missing_source_proof_refuses_before_provider_or_attempt_charge(
    publication, monkeypatch, entry
):
    from unittest.mock import MagicMock

    from reflexio.server.services.extraction.resumable_agent import (
        ResumableExtractionAgent,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )
    from reflexio.server.services.storage.storage_base import AgentRunStatus

    storage, _, _, _, _ = publication
    original = storage.get_agent_run("run_1")
    assert original is not None
    model = MagicMock(side_effect=AssertionError("Unwitnessed source reached provider"))
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    before_model = MagicMock(side_effect=lambda run: run)
    agent = ResumableExtractionAgent(client=MagicMock(), storage=storage)
    with pytest.raises(InputBudgetConfigurationError, match="source proof"):
        if entry == "resume":
            agent.resume(
                run=original,
                messages=[],
                history_inputs=HistoricalInputs(),
                output_schema=StructuredProfilesOutput,
                resolved_tool_calls=[],
                before_model=before_model,
            )
        else:
            agent.start(
                run=replace(
                    original,
                    id="unwitnessed_new",
                    status=AgentRunStatus.RUNNING,
                    committed_output=None,
                ),
                messages=[],
                history_inputs=HistoricalInputs(),
                output_schema=StructuredProfilesOutput,
            )
    before_model.assert_not_called()
    model.assert_not_called()
    assert storage.get_agent_run("run_1").resume_attempts == original.resume_attempts
    assert storage.get_agent_run("unwitnessed_new") is None
