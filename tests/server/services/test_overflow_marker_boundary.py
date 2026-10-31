"""Damaged protocol markers cannot become legacy publication or fresh admission."""

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowCheckpointConflictError,
)
from reflexio.test_support.skip_decorators import skip_in_precommit
from tests.server.services.test_durable_window_pipeline import _saved_legacy_window
from tests.server.services.test_durable_window_pipeline import (
    pipeline as imported_pipeline,
)

pytestmark = pytest.mark.integration
marker_pipeline = imported_pipeline


def _replace_marker(storage, run, marker):
    snapshot = {**run.generation_request_snapshot, "overflow": marker}
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), run.id),
        )
    return snapshot


@pytest.mark.parametrize(
    ("marker", "enabled", "mode"),
    [
        (None, False, "native"),
        (None, True, "cached"),
        (None, True, "mock"),
        (None, False, "image"),
        ({}, False, "native"),
        ("invalid", True, "native"),
        (
            {
                "version": 1,
                "source_digest": "saved-source",
                "attempt_limit": 128,
                "revision": "invalid",
                "attempts": [{"id": "already-consumed"}],
            },
            True,
            "native",
        ),
    ],
)
def test_marker_rejection_precedes_every_admission_bypass(
    marker_pipeline, monkeypatch, marker, enabled, mode
):
    (_, storage, window, token, executor, service, request, config, groups, run) = (
        _saved_legacy_window(marker_pipeline, monkeypatch, kind="profile")
    )
    _replace_marker(storage, run, marker)
    before = storage.get_agent_run(run.id)
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", str(enabled).lower())
    monkeypatch.setenv("MOCK_LLM_RESPONSE", str(mode == "mock").lower())
    if mode == "cached":
        window = replace(window, outcome={"must_not_decode": True})
    if mode == "image":
        groups[0].interactions[
            0
        ].interacted_image_url = "https://example.test/image.png"
    monkeypatch.setattr(executor, "service", lambda _: (service, request, config))
    compute = Mock(side_effect=AssertionError("Legacy compute must not run"))
    persist = Mock(side_effect=AssertionError("No learning may publish"))
    dispatch = Mock(side_effect=AssertionError("No provider may run"))
    decode = Mock(side_effect=AssertionError("Cached output must not decode"))
    monkeypatch.setattr(service, "compute_generation", compute)
    monkeypatch.setattr(service, "persist_generation", persist)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.window_executor.decode_plan", decode
    )
    with pytest.raises(InputBudgetConfigurationError, match="overflow marker"):
        executor.execute(window, token)
    assert storage.get_agent_run(run.id) == before
    assert storage.count_all_profiles() == 0
    with storage._stream_sql(read_only=True) as db:
        row = db.query(
            "SELECT completed,outcome,effects FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        )[0]
    assert not row["completed"] and row["outcome"] is None and row["effects"] is None
    for operation in (compute, persist, dispatch, decode):
        operation.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@skip_in_precommit
def test_worker_reports_null_marker_without_publication_or_journal_replacement(
    marker_pipeline, monkeypatch, enabled
):
    (engine, storage, window, _, _, _, _, _, _, run) = _saved_legacy_window(
        marker_pipeline, monkeypatch, kind="profile"
    )
    snapshot = _replace_marker(storage, run, None)
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", str(enabled).lower())
    dispatch = Mock(side_effect=AssertionError("No provider may run"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    # Simulate the previous process losing its lease after saving the parent.
    with storage._stream_sql() as db:
        db.query("UPDATE learning_work SET lease_until=0,due_at=0 WHERE user_id='u'")
    _, worker, http = marker_pipeline
    assert worker.drain_org("stream-test", 1, 300) == 0
    assert http.get(
        "/api/learning_status", params={"request_id": "legacy-saved"}
    ).json() == {"status": "failed", "reason": "input_budget_configuration_error"}
    after = storage.get_agent_run(run.id)
    assert after is not None
    assert after.generation_request_snapshot == snapshot
    assert after.committed_output == run.committed_output
    assert engine.get_storage().count_all_profiles() == 0
    assert len(storage.get_user_interaction("u")) == 10
    with storage._stream_sql(read_only=True) as db:
        row = db.query(
            "SELECT completed,invalidated,outcome,effects FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        )[0]
    assert not row["completed"] and not row["invalidated"]
    assert row["outcome"] is None and row["effects"] is None
    dispatch.assert_not_called()


@pytest.mark.parametrize("marker", [None, "invalid", []])
def test_direct_adoption_cannot_replace_present_marker(
    marker_pipeline, monkeypatch, marker
):
    (_, storage, window, token, _, _, _, _, _, run) = _saved_legacy_window(
        marker_pipeline, monkeypatch, kind="profile"
    )
    _replace_marker(storage, run, marker)
    before = storage.get_agent_run(run.id)
    with (
        pytest.raises(OverflowCheckpointConflictError, match="overflow marker"),
        storage.commit_scope(),
    ):
        storage.adopt_extraction_overflow(
            window,
            token,
            source_digest="replacement-source",
            source_digests=run.generation_request_snapshot[
                "source_interaction_digests"
            ],
            manifest={
                "version": 1,
                "source_digest": "replacement-source",
                "revision": 0,
                "attempt_limit": 128,
                "attempts": [],
            },
        )
    assert storage.get_agent_run(run.id) == before
    assert storage.count_all_profiles() == 0
