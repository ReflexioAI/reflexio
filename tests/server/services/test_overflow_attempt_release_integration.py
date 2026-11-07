"""The overflow journal's one refund: an attempt the provider never answered.

``checkpoint_extraction_overflow(released_attempt_id=...)`` is how an outage
probe returns its attempt (``OverflowCoordinator._settle_failure``). It must
give the quota back and nothing else: an answered attempt, an unknown one, or
a release mixed into another journal change is refused, and the references
the reservation created leave with it.
"""

from __future__ import annotations

import pytest

from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowAttemptBudgetExceededError,
    OverflowCheckpointConflictError,
)
from tests.server.services.test_durable_window_pipeline import (
    _overflow_coordinator,
)
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline

_SOURCE = "captured-source"


def _unanswered(identity: str) -> dict:
    return {"id": identity, "usage": None, "model_provenance": None}


def test_released_attempt_returns_journal_quota(pipeline):
    full = [_unanswered(f"attempt-{index}") for index in range(128)]
    storage, window, token = _overflow_coordinator(pipeline, attempts=full)
    with pytest.raises(OverflowAttemptBudgetExceededError):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=0,
            source_digest=_SOURCE,
            reserve_attempt_id="attempt-next",
        )

    manifest = storage.checkpoint_extraction_overflow(
        window,
        token,
        expected_revision=0,
        source_digest=_SOURCE,
        released_attempt_id="attempt-127",
    )
    assert [attempt["id"] for attempt in manifest["attempts"]] == [
        f"attempt-{index}" for index in range(127)
    ]
    manifest = storage.checkpoint_extraction_overflow(
        window,
        token,
        expected_revision=1,
        source_digest=_SOURCE,
        reserve_attempt_id="attempt-next",
    )
    assert len(manifest["attempts"]) == 128
    saved = storage.get_agent_run(f"window:{window.window_id}")
    assert saved.generation_request_snapshot["overflow"] == manifest


def test_only_an_unanswered_attempt_is_released_on_its_own(pipeline):
    storage, window, token = _overflow_coordinator(
        pipeline,
        attempts=[
            {
                "id": "answered",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_write_input_tokens": 0,
                },
                "model_provenance": {"model_name": "fixture", "provider": None},
            },
            _unanswered("unanswered"),
        ],
    )
    refused = (
        {"released_attempt_id": "answered"},
        {"released_attempt_id": "unknown"},
        {"released_attempt_id": "unanswered", "reserve_attempt_id": "another"},
        {
            "released_attempt_id": "unanswered",
            "checkpoint": {"stage": "prepare", "completed_steps": {}},
        },
    )
    for change in refused:
        with pytest.raises(OverflowCheckpointConflictError):
            storage.checkpoint_extraction_overflow(
                window, token, expected_revision=0, source_digest=_SOURCE, **change
            )
    saved = storage.get_agent_run(f"window:{window.window_id}")
    overflow = saved.generation_request_snapshot["overflow"]
    assert overflow["revision"] == 0
    assert [attempt["id"] for attempt in overflow["attempts"]] == [
        "answered",
        "unanswered",
    ]


def test_release_returns_the_reconciliation_allowance(pipeline):
    storage, window, token = _overflow_coordinator(
        pipeline, attempts=[_unanswered("kept"), _unanswered("probe")]
    )
    operation = "a" * 64
    manifest = storage.checkpoint_extraction_overflow(
        window,
        token,
        expected_revision=0,
        source_digest=_SOURCE,
        checkpoint={
            "stage": "prepare",
            "completed_steps": {},
            "reconciliation_context": {
                "version": 1,
                "operation_key": operation,
                "limit": 2,
                "plan": None,
                "attempt_ids": ["kept", "probe"],
                "nodes": {},
            },
            "audited_consolidation_budget": {"key": operation, "calls": 2},
        },
    )
    manifest = storage.checkpoint_extraction_overflow(
        window,
        token,
        expected_revision=manifest["revision"],
        source_digest=_SOURCE,
        released_attempt_id="probe",
    )
    checkpoint = manifest["checkpoint"]
    assert [attempt["id"] for attempt in manifest["attempts"]] == ["kept"]
    assert checkpoint["reconciliation_context"]["attempt_ids"] == ["kept"]
    assert checkpoint["audited_consolidation_budget"] == {
        "key": operation,
        "calls": 1,
    }
