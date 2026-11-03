"""The shared admitted worker must select resume-only users without window work."""

import pytest

from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.extraction import (
    test_initial_publication_integration as initial,
)

storage = initial.storage
request_context = initial.request_context
publication = initial.publication
explicit_test_billing_exemption = initial.explicit_test_billing_exemption
no_background_tagging = initial.no_background_tagging
pytestmark = pytest.mark.integration


def test_resume_only_user_does_not_require_window_due(publication, monkeypatch):
    storage, resume, kind, _, events = publication
    service, request, calls = initial._initial(publication, monkeypatch)
    create = storage._create_agent_output_receipt

    def fail_first_receipt(*args, **kwargs):
        raise RuntimeError("synthetic publication rollback before receipt")

    with monkeypatch.context() as failed:
        failed.setattr(storage, "_create_agent_output_receipt", fail_first_receipt)
        initial._run_initial(service, request)
    first = service._last_extraction_run
    assert first is not None and first.status == AgentRunStatus.FINALIZATION_FAILED
    assert first.committed_output is not None and calls == [1]
    assert not initial.base._learnings(storage, kind)
    # Only the real business retry delay is made due in this test. No lease,
    # status or owner is forged; initial execution has released its user token.
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (first.id,)
    )
    ordinary = storage.claim_extraction("window-only-probe", 300)
    assert ordinary is not None and ordinary[0] == first.binding.user_id
    assert storage.prepare_extraction(*ordinary) is None
    assert storage.claim_extraction("empty-window-probe", 300) is None
    assert storage._create_agent_output_receipt == create
    worker = DurableLearningWorker(lambda _org: resume.request_context)
    worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    recovered = storage.get_agent_run(first.id)
    assert recovered is not None and recovered.status == AgentRunStatus.FINALIZED, (
        "Shared admitted user turns must service resume-only finalization without a due window"
    )
    assert recovered.committed_output == first.committed_output and calls == [1]
    assert len(initial.base._learnings(storage, kind)) == 1
    assert (
        len([event for event in events if event.event_name == "learnings_generated"])
        == 1
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_continuation_drains_accepted_answer_and_controls_only_new_tools(
    publication, monkeypatch, enabled
):
    from threading import BoundedSemaphore

    from reflexio.server.services.durable_learning import worker as workers
    from reflexio.server.services.durable_learning.user_lease import (
        _require_user_work_owner,
    )

    storage, resume, kind, _, events = publication
    initial._initial(publication, monkeypatch)
    context = resume.request_context
    context.configurator.get_config().pending_tool_call_config.pending_ttl_seconds = (
        86_400 if enabled else 0
    )
    before = storage.get_agent_run("run_1")
    assert before is not None and before.status == AgentRunStatus.RESUME_READY
    # The historical paused-run fixture has no ordinary work row at all.
    # Canonical resume acquisition must create its owner without a due window.
    assert storage.claim_extraction("window-empty", 300) is None
    calls = []
    monkeypatch.setattr(workers, "_budget", BoundedSemaphore(1))

    def model(**kwargs):
        tools = kwargs["registry"].openai_specs()
        assert (
            "attach_pending_info_request"
            in {tool["function"]["name"] for tool in tools}
        ) is enabled
        owner = _require_user_work_owner(storage)
        owner.fence(storage)
        assert owner.user_id == before.binding.user_id
        assert not workers._reserve(), (
            "Physical execution must retain its original process permit"
        )
        calls.append(1)
        return initial.base._loop_result(kwargs, kind, storage, nonempty=True)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    worker = DurableLearningWorker(lambda _org: context)
    worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    current = storage.get_agent_run(before.id)
    assert current is not None
    dependencies = storage.list_run_tool_dependencies(before.id)
    assert current.status == AgentRunStatus.FINALIZED
    assert current.resume_attempts == before.resume_attempts + 1
    assert (
        calls == [1]
        and dependencies
        and all(row.consumed_at is not None for row in dependencies)
    )
    assert len(initial.base._learnings(storage, kind)) == 1
    assert (
        len([event for event in events if event.event_name == "learnings_generated"])
        == 1
    )
    assert workers._reserve(), (
        "Completed or refused turns must return the original permit"
    )
    workers._release()


@pytest.mark.parametrize("wrapped_conversion", [False, True])
def test_source_reload_failure_after_adoption_logs_no_customer_values(
    publication, monkeypatch, caplog, wrapped_conversion
):
    from reflexio.server.services.durable_learning.user_lease import (
        _require_user_work_owner,
    )

    storage, resume, kind, _, events = publication
    service, request, calls = initial._initial(publication, monkeypatch)

    def rollback(*_a, **_kw):
        raise RuntimeError("Synthetic initial publication rollback")

    with monkeypatch.context() as failed:
        failed.setattr(storage, "_create_agent_output_receipt", rollback)
        initial._run_initial(service, request)
    saved = service._last_extraction_run
    assert saved is not None and saved.status == AgentRunStatus.FINALIZATION_FAILED
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (saved.id,)
    )
    reloaded = []

    def private_reload(_ids):
        owner = _require_user_work_owner(storage)
        owner.fence(storage)
        current = storage.get_agent_run(saved.id)
        assert current is not None and current.status == AgentRunStatus.FINALIZING
        assert current.claimed_by != saved.claimed_by
        reloaded.append(current.id)
        raise ValueError("SENTINEL_PRIVATE_STORED_CUSTOMER_OUTPUT_AND_CREDENTIAL")

    if wrapped_conversion:
        from reflexio.server.services.storage.sqlite_storage import _extras

        monkeypatch.setattr(_extras, "_row_to_interaction", private_reload)
    else:
        monkeypatch.setattr(storage, "get_interactions_by_ids", private_reload)
    worker = DurableLearningWorker(lambda _org: resume.request_context)
    worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    assert reloaded == [saved.id] and calls == [1]
    assert not initial.base._learnings(storage, kind)
    failures = [
        record
        for record in caplog.records
        if "extraction_turn_failed" in record.getMessage()
    ]
    assert len(failures) == 1
    assert "SENTINEL_PRIVATE_STORED_CUSTOMER_OUTPUT_AND_CREDENTIAL" not in caplog.text
    assert failures[0].exc_info is None
    expected_error = "StorageError" if wrapped_conversion else "ValueError"
    assert expected_error in failures[0].getMessage()
    assert all(record.exc_info is None for record in caplog.records)
    if wrapped_conversion:
        assert "Error in get_interactions_by_ids: ValueError" in caplog.text


def test_default_discovery_finds_real_resume_only_work(publication, monkeypatch):
    from reflexio.server.services.durable_learning import local
    from reflexio.server.services.durable_learning import scheduler as scheduling

    storage, resume, kind, _, _ = publication
    service, request, calls = initial._initial(publication, monkeypatch)

    def rollback(*_args, **_kwargs):
        raise RuntimeError("Synthetic initial publication rollback")

    with monkeypatch.context() as failed:
        failed.setattr(storage, "_create_agent_output_receipt", rollback)
        initial._run_initial(service, request)
    saved = service._last_extraction_run
    assert saved is not None and saved.status == AgentRunStatus.FINALIZATION_FAILED
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (saved.id,)
    )
    ordinary = storage.claim_extraction("park-empty-window", 300)
    assert ordinary is not None and storage.prepare_extraction(*ordinary) is None
    assert storage.list_extraction_orgs() == []
    before_discovery = storage.get_agent_run(saved.id)
    monkeypatch.setattr(
        scheduling.DurableLearningScheduler, "start", lambda _self: None
    )
    monkeypatch.setattr(local, "adopt_server_scheduler", lambda _scheduler: None)
    scheduler = scheduling.maybe_start_durable_learning(
        lambda _org: resume.request_context, bootstrap_org_id=storage.org_id
    )
    assert list(scheduler._org_ids_provider()) == [storage.org_id]
    assert storage.get_agent_run(saved.id) == before_discovery
    assert calls == [1] and not initial.base._learnings(storage, kind)
