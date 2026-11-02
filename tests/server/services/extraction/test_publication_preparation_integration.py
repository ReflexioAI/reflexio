"""A publication retry reuses the original resolved and embedded business plan."""

import pytest

from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.extraction import test_resume_publication_integration as base

storage = base.storage
request_context = base.request_context
explicit_test_billing_exemption = base.explicit_test_billing_exemption
no_background_tagging = base.no_background_tagging
publication = base.publication
shared_worker = base.shared_worker
pytestmark = pytest.mark.integration


def test_failed_writer_reuses_completed_preparation_without_recomputing(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    storage, worker, kind, dispatches, events = publication
    service_type = (
        ProfileGenerationService if kind == "profile" else PlaybookGenerationService
    )
    resolve = service_type._resolve_write_plan
    calls = []
    embedding_calls = []
    embedding_method = (
        "precompute_profile_embeddings"
        if kind == "profile"
        else "precompute_user_playbook_embeddings"
    )
    embed = getattr(storage, embedding_method)

    def resolve_once(self, groups):
        calls.append(1)
        return resolve(self, groups)

    def embed_once(rows):
        embedding_calls.append(1)
        return embed(rows)

    monkeypatch.setattr(service_type, "_resolve_write_plan", resolve_once)
    monkeypatch.setattr(storage, embedding_method, embed_once)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    create = storage._create_agent_output_receipt
    fail = [True]

    def fail_after_learning(*args, **kwargs):
        if fail[0]:
            fail[0] = False
            raise RuntimeError("Controlled publication rollback after preparation")
        return create(*args, **kwargs)

    monkeypatch.setattr(storage, "_create_agent_output_receipt", fail_after_learning)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    first = storage.get_agent_run("run_1")
    assert first is not None and first.status == AgentRunStatus.FINALIZATION_FAILED
    assert not base._learnings(storage, kind)
    assert calls == [1] and embedding_calls == [1]
    assert not dispatches
    assert not [event for event in events if event.event_name == "learnings_generated"]
    original_output = first.committed_output
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (first.id,)
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    second = storage.get_agent_run("run_1")
    assert second is not None and second.status == AgentRunStatus.FINALIZED
    assert second.committed_output == original_output
    assert len(base._learnings(storage, kind)) == 1
    assert calls == [1], (
        "The original resolved plan must survive a publication rollback"
    )
    assert embedding_calls == [1], (
        "A publication retry must not repeat completed embedding work"
    )
    assert (
        len([event for event in events if event.event_name == "learnings_generated"])
        == 1
    )
    assert dispatches.count("tagging") == 1


@pytest.mark.parametrize("change", ["source", "answer", "cache"])
def test_cached_preparation_cannot_backfill_changed_or_corrupt_inputs(
    shared_worker: DurableLearningWorker, publication, monkeypatch, change
):
    import json

    storage, worker, kind, dispatches, events = publication
    service_type = (
        ProfileGenerationService if kind == "profile" else PlaybookGenerationService
    )
    original_resolve = service_type._resolve_write_plan
    calls = []

    def resolve(self, groups):
        calls.append(1)
        return original_resolve(self, groups)

    monkeypatch.setattr(service_type, "_resolve_write_plan", resolve)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    create = storage._create_agent_output_receipt
    failed = []

    def fail_once(*args, **kwargs):
        if not failed:
            failed.append(1)
            raise RuntimeError("Controlled rollback before first receipt")
        return create(*args, **kwargs)

    monkeypatch.setattr(storage, "_create_agent_output_receipt", fail_once)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    first = storage.get_agent_run("run_1")
    assert first is not None and first.status == AgentRunStatus.FINALIZATION_FAILED
    original_output = first.committed_output
    if change == "source":
        storage._execute(
            "UPDATE interactions SET content='Changed original source' WHERE interaction_id=1",
            (),
        )
    elif change == "answer":
        storage._execute(
            "UPDATE _pending_tool_calls SET result=? WHERE id='ptc_1'",
            (json.dumps({"answer": "Changed original answer"}),),
        )
    else:
        payload = json.loads(
            storage.conn.execute(
                "SELECT preparation FROM _agent_output_preparations WHERE run_id=?",
                (first.id,),
            ).fetchone()[0]
        )
        payload["output_digest"] = "0" * 64
        storage._execute(
            "UPDATE _agent_output_preparations SET preparation=? WHERE run_id=?",
            (json.dumps(payload), first.id),
        )
    before = storage.conn.execute(
        "SELECT preparation FROM _agent_output_preparations WHERE run_id=?", (first.id,)
    ).fetchone()[0]
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (first.id,)
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    second = storage.get_agent_run("run_1")
    assert second is not None and second.status == AgentRunStatus.FINALIZATION_FAILED
    assert second.committed_output == original_output
    assert calls == [1]
    assert not base._learnings(storage, kind)
    assert not dispatches
    assert not [event for event in events if event.event_name == "learnings_generated"]
    assert all(
        d.consumed_at is None for d in storage.list_run_tool_dependencies(first.id)
    )
    assert (
        storage.conn.execute(
            "SELECT preparation FROM _agent_output_preparations WHERE run_id=?",
            (first.id,),
        ).fetchone()[0]
        == before
    )


@pytest.mark.parametrize("publication", ["playbook"], indirect=True)
def test_retry_checks_saved_consolidation_history_not_just_candidate_history(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    import json

    from reflexio.server.services.playbook.components.overflow_evidence import (
        historical_playbook_digest,
    )
    from tests.server.services.extraction import (
        test_publication_history_integration as history,
    )

    storage, worker, kind, dispatches, events = publication
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    original = history._seed_history(storage, kind)
    expected = {str(original.user_playbook_id): historical_playbook_digest(original)}
    monkeypatch.setattr(
        "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator",
        PlaybookConsolidator,
    )
    monkeypatch.setattr(
        PlaybookConsolidator,
        "retrieve_existing_playbooks",
        lambda *_a, **_kw: storage.get_user_playbooks(
            user_id="user_1", agent_version="v1"
        ),
    )
    decisions = []

    def reconcile(_consolidator, new_rows, existing_rows, **_kwargs):
        assert (
            len(existing_rows) == 1
            and existing_rows[0].content == "Historical approval policy."
        )
        decisions.append(existing_rows[0].content)
        return new_rows, [], []

    monkeypatch.setattr(
        "reflexio.server.services.playbook.components.reconciliation.reconcile_playbooks",
        reconcile,
    )
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    create = storage._create_agent_output_receipt
    failed = []

    def fail_once(*args, **kwargs):
        if not failed:
            failed.append(1)
            raise RuntimeError("Controlled rollback before receipt")
        return create(*args, **kwargs)

    monkeypatch.setattr(storage, "_create_agent_output_receipt", fail_once)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    first = storage.get_agent_run("run_1")
    assert first is not None and first.status == AgentRunStatus.FINALIZATION_FAILED
    assert first.committed_output["publication"]["history_inputs"] == {
        "profiles": {},
        "playbooks": {},
    }
    raw = storage.conn.execute(
        "SELECT preparation FROM _agent_output_preparations WHERE run_id=?", (first.id,)
    ).fetchone()[0]
    assert json.loads(raw)["history_inputs"] == {"profiles": {}, "playbooks": expected}
    storage._execute(
        "UPDATE user_playbooks SET content='Changed after saved consolidation' WHERE user_playbook_id=?",
        (original.user_playbook_id,),
    )
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (first.id,)
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    second = storage.get_agent_run("run_1")
    assert second is not None and second.status == AgentRunStatus.FINALIZATION_FAILED
    assert decisions == ["Historical approval policy."]
    assert len(base._learnings(storage, kind)) == 1
    assert not dispatches
    assert not [event for event in events if event.event_name == "learnings_generated"]
    assert all(
        d.consumed_at is None for d in storage.list_run_tool_dependencies(first.id)
    )
    assert (
        storage.conn.execute(
            "SELECT preparation FROM _agent_output_preparations WHERE run_id=?",
            (first.id,),
        ).fetchone()[0]
        == raw
    )


def test_current_preparation_schema_check_is_read_only(storage):
    from reflexio.server.services.storage.sqlite_storage._publication_preparations import (
        migrate_publication_preparations,
    )

    statements = []
    storage.conn.set_trace_callback(statements.append)
    try:
        migrate_publication_preparations(storage.conn)
    finally:
        storage.conn.set_trace_callback(None)
    assert statements and all(
        statement.lstrip().upper().startswith("SELECT ") for statement in statements
    )


def test_new_output_replaces_preparation_only_after_previous_receipt(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    import json
    from datetime import UTC, datetime

    from reflexio.server.services.storage.storage_base import RunToolDependencyRecord

    storage, worker, kind, _, events = publication
    base._question(storage)
    models = []

    def complete(**kwargs):
        models.append(1)
        if len(models) == 1:
            storage.attach_run_tool_dependency(
                RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_2")
            )
            storage.resolve_pending_tool_call(
                "ptc_2",
                result={"answer": "A second approval requirement"},
                resolved_at=datetime.now(UTC),
                valid_for_seconds=3600,
            )
        result = base._loop_result(kwargs, kind, storage, nonempty=True)
        if len(models) == 2:
            rows = getattr(
                result.structured_output,
                "profiles" if kind == "profile" else "playbooks",
            )
            rows[0].content = "Require approval before deployment."
        return result

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", complete
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    first = storage.get_agent_run("run_1")
    assert first is not None and first.status == AgentRunStatus.RESUME_READY
    first_id = first.committed_output["publication"]["id"]
    first_preparation = json.loads(
        storage.conn.execute(
            "SELECT preparation FROM _agent_output_preparations WHERE run_id=?",
            (first.id,),
        ).fetchone()[0]
    )
    assert first_preparation["output_id"] == first_id
    storage._execute(
        "UPDATE _agent_runs SET next_resume_at=NULL WHERE id=?", (first.id,)
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    second = storage.get_agent_run("run_1")
    assert second is not None and second.status == AgentRunStatus.FINALIZED
    second_id = second.committed_output["publication"]["id"]
    assert first_id != second_id
    second_preparation = json.loads(
        storage.conn.execute(
            "SELECT preparation FROM _agent_output_preparations WHERE run_id=?",
            (second.id,),
        ).fetchone()[0]
    )
    assert second_preparation["output_id"] == second_id
    assert models == [1, 1] and len(base._learnings(storage, kind)) == 2
    assert (
        len([event for event in events if event.event_name == "learnings_generated"])
        == 2
    )
    with storage._stream_sql(read_only=True) as db:
        for identifier in (first_id, second_id):
            assert (
                storage._get_output_publication_receipt(db, second.id, identifier)
                is not None
            )
    assert all(
        d.consumed_at is not None for d in storage.list_run_tool_dependencies(second.id)
    )


def test_unpublished_preparation_cannot_be_replaced_by_another_output(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    import json
    import uuid
    from copy import deepcopy

    storage, worker, kind, dispatches, events = publication
    service_type = (
        ProfileGenerationService if kind == "profile" else PlaybookGenerationService
    )
    resolve = service_type._resolve_write_plan
    calls = []

    def resolve_once(self, groups):
        calls.append(1)
        return resolve(self, groups)

    monkeypatch.setattr(service_type, "_resolve_write_plan", resolve_once)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )

    def fail_receipt(*_args, **_kwargs):
        raise RuntimeError("Controlled publication rollback")

    monkeypatch.setattr(storage, "_create_agent_output_receipt", fail_receipt)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    first = storage.get_agent_run("run_1")
    assert first is not None and first.status == AgentRunStatus.FINALIZATION_FAILED
    before = storage.conn.execute(
        "SELECT preparation FROM _agent_output_preparations WHERE run_id=?", (first.id,)
    ).fetchone()[0]
    # Deliberately simulate an invalid raw-output replacement without publishing
    # the prior output. The durable store must refuse to hide its preparation.
    replacement = deepcopy(first.committed_output)
    replacement["publication"]["id"] = uuid.uuid4().hex
    storage._execute(
        "UPDATE _agent_runs SET committed_output=?,next_resume_at=NULL WHERE id=?",
        (json.dumps(replacement), first.id),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=6, lease_seconds=300) == 1
    second = storage.get_agent_run("run_1")
    assert second is not None and second.status == AgentRunStatus.FINALIZATION_FAILED
    assert second.committed_output == replacement
    assert calls == [1]
    assert not base._learnings(storage, kind) and not dispatches
    assert not [event for event in events if event.event_name == "learnings_generated"]
    assert (
        storage.conn.execute(
            "SELECT preparation FROM _agent_output_preparations WHERE run_id=?",
            (first.id,),
        ).fetchone()[0]
        == before
    )
