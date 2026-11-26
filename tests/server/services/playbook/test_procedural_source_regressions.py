"""Source-only windows need no model; mixed evidence remains byte-exact."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.server.llm.input_budget import ExtractionInputIntegrityError
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.extraction.agent_run_records import (
    source_interaction_digest,
)
from reflexio.server.services.extraction.input_preparation import (
    prepare_extraction_input,
)
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.server.services.playbook.procedural import native, workflow
from tests.server.services.playbook import test_procedural_window_integration as windows
from tests.server.services.playbook.test_source_quoted_overflow import sources

pipeline = windows.pipeline
pytestmark = pytest.mark.integration


def test_whitespace_only_native_window_retains_noop_without_retrieval_or_model(
    pipeline, monkeypatch
):
    engine, storage, window, token, client, calls, existing = windows.setup(
        pipeline, monkeypatch, retire=True
    )
    original = " \t\n " * 200
    storage._execute(
        "UPDATE interactions SET content=? WHERE user_id=?", (original, "u")
    )
    retrieval = Mock(side_effect=AssertionError("Blank evidence must not retrieve"))
    monkeypatch.setattr(PlaybookConsolidator, "retrieve_existing_playbooks", retrieval)
    executor = WindowExecutor(engine.request_context, client)
    service, _, _ = executor.service(window)
    expected_witnesses = {
        str(row.interaction_id): source_interaction_digest(row)
        for group in service.service_config.window_interactions
        for row in group.interactions
    }
    executor.execute(window, token)
    retrieval.assert_not_called()
    assert calls == []
    run = storage.get_agent_run(f"window:{window.window_id}")
    assert run is not None and run.committed_output is not None
    assert run.status.value == "finalized"
    assert run.generation_request_snapshot["overflow"]["preparation_version"] == 5
    assert run.committed_output["output"]["receipt"]["source_projection"] == (
        "no_visible_native_evidence"
    )
    assert run.generation_request_snapshot["source_interaction_digests"] == (
        expected_witnesses
    )
    assert [
        row.user_playbook_id for row in storage.get_user_playbooks(user_id="u")
    ] == [existing[0].user_playbook_id]
    with storage._stream_sql(read_only=True) as db:
        rows = db.query("SELECT content FROM interactions WHERE user_id=?", ("u",))
    assert rows and all(row["content"] == original for row in rows)


def test_blank_projection_marker_cannot_bypass_reader_coverage_for_real_text():
    prepared = prepare_extraction_input(sources(), user_id="u")
    output = native.ProceduralOutput(
        source_digest=prepared.source_digest,
        source_coverage=[workflow.digest(chunk.text) for chunk in prepared.chunks],
        historical_playbooks={},
        receipt={"source_projection": "no_visible_native_evidence"},
        write_plan=None,
        plan_sha256=workflow.digest(None),
    )
    coordinator: Any = SimpleNamespace(
        source_digest=prepared.source_digest,
        attempt_count=0,
        require_prepared=lambda: prepared,
        candidate_key=workflow.digest,
        completed_step=Mock(side_effect=AssertionError("There is no model receipt")),
        checkpoint={
            native.KEY: {"output": output.model_dump(mode="json"), "reader_steps": {}},
            "historical_playbooks": {},
        },
    )
    with pytest.raises(ExtractionInputIntegrityError, match="source/state binding"):
        native._retained(coordinator)
