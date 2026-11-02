"""Selected previews follow actual user ownership rather than logical claim age."""

from unittest.mock import MagicMock

import pytest

from reflexio.server.services.durable_learning.user_lease import (
    adopt_agent_run_for_execution,
    user_extraction_lease,
)
from reflexio.server.services.extraction.recovery import (
    inspect_run,
    payload_fingerprint,
    selection_from_report,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.extraction import test_selected_recovery_integration as base

context = base.context
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("retired_ttl", ["1", "600", "86400"])
def test_recent_released_parent_can_recover_without_a_claim_age_delay(
    context, monkeypatch, retired_ttl
):
    monkeypatch.setenv("REFLEXIO_RESUME_CLAIM_TTL_SECONDS", retired_ttl)
    run = base.seed(context)
    with user_extraction_lease(context.storage, "user", max_wait_seconds=0):
        current = adopt_agent_run_for_execution(context.storage, run)
    assert current.claimed_by and current.claimed_at is not None
    before = payload_fingerprint(current)
    report = inspect_run(context, current)
    assert report["eligible"], report["blockers"]
    assert report["claim_expires_at"] is None
    client = MagicMock()
    worker = ExtractionResumeWorker(request_context=context, llm_client=client)
    recovered = worker.recover_selected(
        selection_from_report(report), operation_id="recent"
    )
    assert recovered.status == AgentRunStatus.FINALIZED
    assert payload_fingerprint(recovered) == before
    assert recovered.recovery_operation_id == "recent"
    client.generate.assert_not_called()


def test_old_logical_claim_does_not_hide_a_live_actual_owner(context):
    run = base.seed(context)
    token = context.storage.claim_user_extraction("user", "actual-live-owner", 300)
    assert token is not None
    try:
        before = context.storage.get_agent_run(run.id)
        report = inspect_run(context, before)
        assert not report["eligible"]
        assert "live_user_work_owner" in report["blockers"]
        assert report["claim_expires_at"] is not None
        assert context.storage.get_agent_run(run.id) == before
        context.storage.fence_user_extraction("user", token)
    finally:
        context.storage.release_user_extraction("user", token)


def test_same_writer_clock_cannot_reuse_a_superseded_selected_revision(
    context, monkeypatch
):
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    original = base.seed(context)
    with user_extraction_lease(context.storage, "user", max_wait_seconds=0):
        current = adopt_agent_run_for_execution(context.storage, original)
        assert current.updated_at is not None
        fixed = current.updated_at.timestamp()
        monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: fixed)
        first = context.storage.claim_agent_run_for_recovery(
            expected=current, worker_id="first", operation_id="first-operation"
        )
        assert first is not None and first.updated_at == current.updated_at
        assert (
            context.storage.claim_agent_run_for_recovery(
                expected=current, worker_id="second", operation_id="second-operation"
            )
            is None
        )
        assert context.storage.get_agent_run(first.id) == first
