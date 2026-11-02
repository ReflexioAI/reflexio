"""Real claim/publish actors replace obsolete run-wide finalizer race tests."""

from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import replace
from threading import BoundedSemaphore, Event

import pytest

from reflexio.server.services.durable_learning import worker as admission
from reflexio.server.services.durable_learning.user_lease import (
    adopt_agent_run_for_execution,
    agent_run_user_work,
)
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredReferencedExtractedPlaybookList,
)
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.extraction import test_resume_publication_integration as base
from tests.server.services.extraction._publication_fixtures import (
    write_completed_output,
)

storage = base.storage
request_context = base.request_context
explicit_test_billing_exemption = base.explicit_test_billing_exemption
no_background_tagging = base.no_background_tagging
publication = base.publication
shared_worker = base.shared_worker
pytestmark = pytest.mark.integration
_REAL_PLAYBOOK_DISPATCH = PlaybookGenerationService._dispatch_playbook_schedulers
_CONTENTS = ("Require approval before deployment.", "Review the deployment approval.")


def _two_outputs(kwargs, kind, storage):
    result = base._loop_result(kwargs, kind, storage, nonempty=True)
    field = "profiles" if kind == "profile" else "playbooks"
    output = result.structured_output
    assert output is not None
    row = getattr(output, field)[0]
    return result.model_copy(
        update={
            "structured_output": output.model_copy(
                update={
                    field: [
                        row.model_copy(update={"content": text}) for text in _CONTENTS
                    ]
                }
            )
        }
    )


@pytest.mark.parametrize("held_stage", ["model", "preparation"])
def test_two_actual_workers_publish_one_ordered_output(
    shared_worker: DurableLearningWorker, publication, monkeypatch, held_stage
):
    storage, worker, kind, dispatches, events = publication
    # Both actors enter the REAL budget. A full one-slot budget would return
    # zero before touching storage and cannot qualify ownership exclusion.
    monkeypatch.setattr(admission, "_budget", BoundedSemaphore(2))
    entered, release = Event(), Event()
    model_calls, preparations = [], []

    def model(**kwargs):
        model_calls.append(1)
        if held_stage == "model":
            assert not storage.in_commit_scope
            entered.set()
            assert release.wait(5)
        return _two_outputs(kwargs, kind, storage)

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop", model
    )
    service_type = (
        ProfileGenerationService if kind == "profile" else PlaybookGenerationService
    )
    resolve = service_type._resolve_write_plan

    def prepare(self, groups):
        preparations.append(1)
        if held_stage == "preparation":
            assert not self.storage.in_commit_scope
            entered.set()
            assert release.wait(5)
        return resolve(self, groups)

    monkeypatch.setattr(service_type, "_resolve_write_plan", prepare)
    target_run_id = "run_1"
    if held_stage == "preparation":
        # A real writer saved work and released its actual user owner.
        # No logical age or execution token is fabricated.
        original = storage.get_agent_run("run_1")
        assert original is not None
        storage._execute("DELETE FROM _agent_runs WHERE id='run_1'")
        shape = (
            StructuredProfilesOutput
            if kind == "profile"
            else StructuredReferencedExtractedPlaybookList
        )
        pending = replace(
            original,
            id="two-actor-output",
            status=AgentRunStatus.RUNNING,
            claimed_by=None,
            claimed_at=None,
            updated_at=None,
            committed_output=None,
            pending_tool_call_ids=[],
            generation_request_snapshot={
                "request_id": "request_1",
                "output_schema_name": shape.__name__,
            },
        )
        result = _two_outputs({"response_format": shape, "ctx": None}, kind, storage)
        assert result.structured_output is not None
        with monkeypatch.context() as patch:
            patch.setattr(
                "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
                lambda **_kwargs: result,
            )
            completed = write_completed_output(
                storage, pending, result.structured_output
            )
            target_run_id = completed.id

    peer = SQLiteStorage(org_id=storage.org_id, db_path=storage.db_path)
    context = copy(worker.request_context)
    context.storage = peer
    other = DurableLearningWorker(lambda _org: context)
    attempts = []
    claim = peer.claim_extraction

    def observe_claim(*args, **kwargs):
        result = claim(*args, **kwargs)
        attempts.append(result)
        return result

    monkeypatch.setattr(peer, "claim_extraction", observe_claim)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            active = pool.submit(shared_worker.drain_org, storage.org_id, 1, 300)
            try:
                assert entered.wait(5)
                assert (
                    pool.submit(other.drain_org, peer.org_id, 1, 300).result(timeout=5)
                    == 0
                )
                assert attempts == [None], (
                    "Peer must reach real storage admission and be refused"
                )
                assert (
                    peer.claim_user_extraction("user_1", "ownership-control", 300)
                    is None
                )
                assert not base._learnings(peer, kind)
            finally:
                release.set()
            assert active.result(timeout=5) == 1
        published = peer.get_agent_run(target_run_id)
        assert published is not None and published.status == AgentRunStatus.FINALIZED
        assert model_calls == ([1] if held_stage == "model" else [])
        assert preparations == [1]
        rows = base._learnings(peer, kind)
        by_id = {
            str(row.profile_id if kind == "profile" else row.user_playbook_id): row
            for row in rows
        }
        ids = peer.get_agent_run_finalization_receipt(
            run_id=published.id,
            entity_type="profile" if kind == "profile" else "user_playbook",
        )
        assert ids is not None and len(ids) == len(rows) == 2
        assert set(ids) == set(by_id)
        assert [by_id[identifier].content for identifier in ids] == list(_CONTENTS)
        with peer._stream_sql(read_only=True) as db:
            assert (
                len(
                    db.query(
                        "SELECT output_id FROM _agent_output_receipts WHERE org_id=? AND run_id=? LIMIT 2",
                        (peer.org_id, published.id),
                    )
                )
                == 1
            )
        billed = [e for e in events if e.event_name == "learnings_generated"]
        assert [e.entity_id for e in billed] == ids
        assert len({e.event_key for e in billed}) == 2
        assert dispatches.count("tagging") == 1
        assert dispatches.count("playbook") == (1 if kind == "playbook" else 0)
    finally:
        with peer._lock:
            peer.conn.close()


@pytest.mark.parametrize("resolution", ["drop", "keep_last", "replace"])
def test_current_billing_follows_actual_persisted_survivors(
    shared_worker: DurableLearningWorker, publication, monkeypatch, resolution
):
    storage, worker, kind, _, events = publication
    input_ids = []
    path = f"reflexio.server.services.{kind}.components.consolidator"
    module = __import__(
        path,
        fromlist=[
            "ProfileConsolidator" if kind == "profile" else "PlaybookConsolidator"
        ],
    )
    factory = getattr(
        module, "ProfileConsolidator" if kind == "profile" else "PlaybookConsolidator"
    )

    def deduplicate(groups, *_args, **_kwargs):
        rows = (
            groups if kind == "profile" else [row for group in groups for row in group]
        )
        input_ids.extend(
            str(row.profile_id if kind == "profile" else row.user_playbook_id)
            for row in rows
        )
        selected = [] if resolution == "drop" else [rows[-1]]
        if resolution == "replace":
            selected = [
                rows[-1].model_copy(
                    update={
                        "content": "Consolidated deployment approval policy.",
                        **(
                            {"profile_id": "consolidated-profile"}
                            if kind == "profile"
                            else {"user_playbook_id": 0}
                        ),
                    }
                )
            ]
        return selected, [], []

    monkeypatch.setattr(factory.return_value.deduplicate, "side_effect", deduplicate)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kwargs: _two_outputs(kwargs, kind, storage),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    published = storage.get_agent_run("run_1")
    assert published is not None and published.status == AgentRunStatus.FINALIZED
    rows = base._learnings(storage, kind)
    ids = [
        str(row.profile_id if kind == "profile" else row.user_playbook_id)
        for row in rows
    ]
    assert len(rows) == (0 if resolution == "drop" else 1)
    billed = [e for e in events if e.event_name == "learnings_generated"]
    assert [e.entity_id for e in billed] == ids
    assert (
        storage.get_agent_run_finalization_receipt(
            run_id=published.id,
            entity_type="profile" if kind == "profile" else "user_playbook",
        )
        == ids
    )
    if resolution == "replace":
        assert ids[0] not in input_ids
    if resolution == "keep_last":
        assert rows[0].content == _CONTENTS[-1]


@pytest.mark.parametrize("publication", ["playbook"], indirect=True)
@pytest.mark.parametrize("failed_scheduler", ["optimizer", "aggregation"])
def test_current_scheduler_failure_and_replay_preserve_other_obligations(
    shared_worker: DurableLearningWorker, publication, monkeypatch, failed_scheduler
):
    storage, worker, kind, dispatches, events = publication
    schedulers, resolves = [], []
    monkeypatch.setattr(
        PlaybookGenerationService,
        "_dispatch_playbook_schedulers",
        _REAL_PLAYBOOK_DISPATCH,
    )

    def dispatch(name):
        schedulers.append(name)
        if name == failed_scheduler:
            raise RuntimeError("Synthetic scheduler outage")

    monkeypatch.setattr(
        PlaybookGenerationService,
        "_enqueue_user_playbook_optimization",
        lambda _self, _rows: dispatch("optimizer"),
    )
    monkeypatch.setattr(
        PlaybookGenerationService,
        "_trigger_playbook_aggregation",
        lambda _self: dispatch("aggregation"),
    )
    resolve = PlaybookGenerationService._resolve_write_plan

    def prepare(self, groups):
        resolves.append(1)
        return resolve(self, groups)

    monkeypatch.setattr(PlaybookGenerationService, "_resolve_write_plan", prepare)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kwargs: _two_outputs(kwargs, kind, storage),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    first = storage.get_agent_run("run_1")
    assert first is not None and first.status == AgentRunStatus.FINALIZED
    with agent_run_user_work(storage, first):
        current = adopt_agent_run_for_execution(storage, first)
        from reflexio.server.services.extraction.publication import (
            publish_resumed_output,
        )

        def no_replay(*_a, **_kw):
            raise AssertionError("Accepted output cannot reprepare or redispatch")

        second = publish_resumed_output(
            storage, current, prepare=no_replay, on_published=no_replay
        )
    assert second.status == AgentRunStatus.FINALIZED
    assert resolves == [1] and schedulers == ["optimizer", "aggregation"]
    ids = storage.get_agent_run_finalization_receipt(
        run_id=second.id, entity_type="user_playbook"
    )
    assert ids is not None and len(ids) == 2
    assert set(ids) == {
        str(row.user_playbook_id) for row in base._learnings(storage, kind)
    }
    assert [e.entity_id for e in events if e.event_name == "learnings_generated"] == ids
    assert dispatches.count("tagging") == 1


def test_external_acceptance_before_failed_ack_replays_exact_keys(
    shared_worker: DurableLearningWorker, publication, monkeypatch
):
    from reflexio.server.services.extraction.publication import _deliver
    from reflexio.server.usage_metrics import (
        UsageEventDeliveryStatus,
        configure_usage_event_recorder,
    )

    storage, worker, kind, _, _ = publication
    attempts, accepted = [], {}

    def append(event):
        if event.event_name != "learnings_generated":
            return UsageEventDeliveryStatus.EXEMPT
        attempts.append(event)
        if event.event_key in accepted:
            assert accepted[event.event_key] == event
            return UsageEventDeliveryStatus.DUPLICATE
        accepted[event.event_key] = event
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(append)
    acknowledge = storage._ack_publication_receipt
    first_ack = []

    def lose_first_ack(receipt):
        if not first_ack:
            first_ack.append(1)
            return False
        return acknowledge(receipt)

    monkeypatch.setattr(storage, "_ack_publication_receipt", lose_first_ack)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kwargs: _two_outputs(kwargs, kind, storage),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    published = storage.get_agent_run("run_1")
    assert published is not None and published.status == AgentRunStatus.FINALIZED
    assert (
        published.finalization_attempts == 0 and published.committed_output is not None
    )
    output_id = published.committed_output["publication"]["id"]
    with storage._stream_sql(read_only=True) as db:
        receipt = storage._get_output_publication_receipt(db, published.id, output_id)
    assert receipt is not None and receipt.delivery_state == "pending"
    ids = list(receipt.learning_ids)
    assert len(ids) == len(accepted) == len(base._learnings(storage, kind)) == 2
    _deliver(storage, receipt)
    assert [event.entity_id for event in attempts] == [*ids, *ids]
    assert len(accepted) == 2
    with storage._stream_sql(read_only=True) as db:
        current = storage._get_output_publication_receipt(db, published.id, output_id)
    assert current is not None and current.delivery_state == "accepted"
    assert current.financial_facts is None
    assert len(base._learnings(storage, kind)) == 2
