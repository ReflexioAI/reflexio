"""No extraction window failure is retried forever.

Invariant: a window that keeps failing is attempted at most
``MAX_EXTRACTION_WINDOW_ATTEMPTS`` times. The failure that reaches the ceiling
parks the cursor (``retry_at=inf``) with the real error locator -- the root
cause's site, not the line that wrapped it -- logs one ERROR, reports
``failed/retry_exhausted``, and an operator can re-arm it through the existing
recovery receipt flow.

Before this, ``retry_extraction`` re-armed every failure with backoff capped at
300 s and no ceiling, so a deterministic failure (34 masked ``TypeError``
abstentions in 72 h of production) retried and stalled the user's stream
forever.
"""

from __future__ import annotations

import logging

import pytest

from reflexio.server.services.durable_learning import worker as worker_module
from reflexio.server.services.extraction.input_recovery import (
    apply_input_budget_recovery,
    plan_input_budget_recovery,
)
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.storage_base._extraction_stream import (
    MAX_EXTRACTION_WINDOW_ATTEMPTS,
    RETRY_EXHAUSTED_PREFIX,
)
from tests.server.services.test_durable_window_pipeline import (
    payload,
)
from tests.server.services.test_durable_window_pipeline import (
    pipeline as pipeline,
)

SENTINEL = "SENTINEL_CUSTOMER_TEXT"


def _raise_root() -> None:
    raise TypeError(SENTINEL)  # ROOT-CAUSE-SITE


def _wrapped_failure(*_args: object) -> bool:
    try:
        _raise_root()
    except TypeError as exc:
        raise RuntimeError("wrapped") from exc  # WRAPPER-SITE
    return False


def _line_of(marker: str) -> int:
    with open(__file__) as source:
        for number, line in enumerate(source, start=1):
            if marker in line and "def " not in line and "_line_of" not in line:
                return number
    raise AssertionError(marker)


def _cursor(storage) -> dict:
    with storage._stream_sql() as db:
        return db.query(
            "SELECT window_id,attempts,retry_at,last_error FROM extraction_cursors "
            "WHERE user_id='u' AND kind='profile'"
        )[0]


def _drain_once(worker, storage) -> None:
    worker.drain_org("stream-test", 1, 300)
    with storage._stream_sql() as db:
        # Skip backoff, never a park: a parked cursor must stay parked.
        db.query("UPDATE extraction_cursors SET retry_at=0 WHERE retry_at<1e300")
        db.query("UPDATE learning_work SET due_at=0 WHERE user_id='u'")


@pytest.mark.parametrize("after_rearm", ["succeeds", "fails_again"])
def test_window_failures_park_at_the_ceiling_with_locator_and_rearm(
    pipeline, monkeypatch, caplog, after_rearm
):
    engine, worker, client = pipeline
    assert client.post("/api/publish_interaction", json=payload("ceiling")).json()[
        "success"
    ]
    storage = engine.get_storage()
    calls: list[int] = []

    def gate(*args: object) -> bool:
        calls.append(1)
        return _wrapped_failure(*args)

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)

    with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
        for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 3):
            _drain_once(worker, storage)

    # Attempted exactly up to the ceiling, then held.
    assert len(calls) == MAX_EXTRACTION_WINDOW_ATTEMPTS
    cursor = _cursor(storage)
    assert cursor["attempts"] == MAX_EXTRACTION_WINDOW_ATTEMPTS
    assert cursor["retry_at"] == float("inf")
    last_error = cursor["last_error"]
    assert last_error.startswith(f"{RETRY_EXHAUSTED_PREFIX}RuntimeError@")
    # The real site: where the TypeError was raised, not the wrapping line.
    assert f":{_line_of('# ROOT-CAUSE-SITE')}" in last_error, last_error
    assert f":{_line_of('# WRAPPER-SITE')}" not in last_error, last_error
    assert SENTINEL not in last_error
    assert storage.extraction_status("u", "ceiling") == {
        "status": "failed",
        "reason": "retry_exhausted",
    }

    parked = [r for r in caplog.records if "Extraction window parked" in r.getMessage()]
    assert len(parked) == 1
    assert parked[0].levelno == logging.ERROR
    assert "cause=TypeError" in parked[0].getMessage()
    assert SENTINEL not in caplog.text

    # Operator re-arm through the existing receipt flow.
    receipt = plan_input_budget_recovery(storage, cursor["window_id"])
    assert receipt["recoverable"], receipt
    assert receipt["hold"] == "retry_exhausted"
    apply_input_budget_recovery(storage, receipt)
    assert _cursor(storage)["retry_at"] != float("inf")
    with pytest.raises(Exception, match="not eligible"):
        apply_input_budget_recovery(storage, receipt)

    if after_rearm == "succeeds":
        monkeypatch.setattr(
            ProfileGenerationService, "_should_run_before_extraction", lambda *_: False
        )
        _drain_once(worker, storage)
        assert storage.extraction_status("u", "ceiling")["status"] == "done"
        assert _cursor(storage)["attempts"] == 0
    else:
        # The count is kept: one more failure re-parks immediately.
        _drain_once(worker, storage)
        _drain_once(worker, storage)
        assert len(calls) == MAX_EXTRACTION_WINDOW_ATTEMPTS + 1
        again = _cursor(storage)
        assert again["retry_at"] == float("inf")
        assert again["attempts"] == MAX_EXTRACTION_WINDOW_ATTEMPTS + 1
        assert again["last_error"].startswith(RETRY_EXHAUSTED_PREFIX)


def test_configuration_hold_keeps_other_users_progressing_after_worker_restart(
    pipeline, monkeypatch
):
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError

    engine, worker, client = pipeline
    storage = engine.get_storage()

    def gate(service, _config):
        if service.service_config.user_id == "u":
            raise InputBudgetConfigurationError("test checkpoint mismatch")
        return False

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    assert client.post("/api/publish_interaction", json=payload("bad-policy")).json()[
        "success"
    ]
    worker.drain_org(storage.org_id, 1, 300)
    assert client.post(
        "/api/publish_interaction", json=payload("other-user", user_id="v")
    ).json()["success"]
    worker.drain_org(storage.org_id, 1, 300)
    assert storage.extraction_status("v", "other-user")["status"] == "done"
    for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 3):
        _drain_once(worker, storage)
    assert storage.extraction_status("u", "bad-policy") == {
        "status": "failed",
        "reason": "retry_exhausted",
    }
    restarted = worker_module.DurableLearningWorker(lambda _: engine.request_context)
    for _ in range(3):
        _drain_once(restarted, storage)
    assert _cursor(storage)["attempts"] == MAX_EXTRACTION_WINDOW_ATTEMPTS
    assert _cursor(storage)["retry_at"] == float("inf")
    assert storage.count_all_profiles() == 0
    # A pre-parent configuration error must remain recoverable after correction.
    receipt = plan_input_budget_recovery(storage, _cursor(storage)["window_id"])
    assert receipt["recoverable"] and receipt["hold"] == "retry_exhausted"
    apply_input_budget_recovery(storage, receipt)
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", lambda *_: False
    )
    _drain_once(restarted, storage)
    assert storage.extraction_status("u", "bad-policy")["status"] == "done"
