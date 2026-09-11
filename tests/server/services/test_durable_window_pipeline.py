"""Real HTTP/storage/executor boundaries with controlled external model calls."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api import create_app
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.profile.service import ProfileGenerationService

pytestmark = pytest.mark.integration


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    # Drive turns explicitly so crashes and overlapping publishes are reproducible.
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )
    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_scheduler.schedule_tagging",
        lambda **_: None,
    )
    config = DefaultConfigurator(org_id="stream-test", base_dir=str(tmp_path))
    config.set_config(
        Config(
            storage_config=StorageConfigSQLite(db_path=str(tmp_path / "stream.db")),
            window_size=10,
            stride_size=5,
            profile_extractor_config=ProfileExtractorConfig(
                extraction_definition_prompt="Extract user preferences"
            ),
            user_playbook_extractor_config=None,
            agent_success_config=None,
        )
    )
    engine = Reflexio(
        org_id="stream-test", storage_base_dir=str(tmp_path), configurator=config
    )
    monkeypatch.setattr(
        "reflexio.server.cache.reflexio_cache.get_reflexio", lambda **_: engine
    )
    monkeypatch.setattr(
        "reflexio.server.api_endpoints.publisher_api.get_reflexio", lambda **_: engine
    )
    worker = DurableLearningWorker(lambda _: engine.request_context)
    client = TestClient(create_app(get_org_id=lambda: "stream-test"))
    return engine, worker, client


def payload(request_id, count=10, user_id="u", force=False):
    return {
        "user_id": user_id,
        "request_id": request_id,
        "session_id": "s",
        "source": "test",
        "force_extraction": force,
        "interaction_data_list": [
            {
                "role": "User",
                "content": f"I prefer concise answers with details {i}",
                "created_at": 100 - i,
            }
            for i in range(count)
        ],
    }


def test_http_ack_is_durable_and_backlog_does_not_skip(pipeline, monkeypatch):
    engine, worker, client = pipeline
    entered, released = threading.Event(), threading.Event()
    seen = []

    def gate(service, _):
        seen.append(
            [
                i.interaction_id
                for group in service.service_config.window_interactions
                for i in group.interactions
            ]
        )
        entered.set()
        assert released.wait(5)
        return False

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    response = client.post("/api/publish_interaction", json=payload("first"))
    assert response.status_code == 200 and response.json()["success"]
    assert response.json()["learning_status"] == "deferred"
    assert engine.get_storage().get_request("first") is not None
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(worker.drain_org, "stream-test", 1, 3)
        assert entered.wait(3)
        for n in range(3):
            response = client.post(
                "/api/publish_interaction", json=payload(f"later{n}", 5)
            )
            assert response.json()["success"]
        assert len(seen) == 1
        released.set()
        assert active.result(timeout=5) == 1
    worker.drain_org("stream-test", 10, 3)
    assert [x[-1] for x in seen] == [10, 15, 20, 25]
    assert (
        client.get("/api/learning_status", params={"request_id": "later2"}).json()[
            "status"
        ]
        == "done"
    )


def test_ingestion_failure_not_acknowledged(pipeline, monkeypatch):
    engine, _, client = pipeline

    def fail(*_, **__):
        raise RuntimeError("write failed")

    monkeypatch.setattr(engine.get_storage(), "admit_extraction", fail)
    response = client.post("/api/publish_interaction", json=payload("failed"))
    assert not response.json()["success"]
    assert engine.get_storage().get_request("failed") is None
    assert engine.get_storage().get_user_interaction("u") == []


def test_sync_request_completes_while_later_tail_remains(pipeline, monkeypatch):
    engine, worker, client = pipeline
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", lambda *_: False
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(
            client.post,
            "/api/publish_interaction?wait_for_response=true",
            json=payload("sync"),
        )
        deadline = time.monotonic() + 3
        while (
            engine.get_storage().get_request("sync") is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert not waiting.done()
        assert client.post("/api/publish_interaction", json=payload("tail", 1)).json()[
            "success"
        ]
        worker.drain_org("stream-test", 1, 3)
        assert waiting.result(timeout=3).json()["learning_status"] == "done"
        assert (
            engine.get_storage().extraction_status("u", "tail")["status"] == "pending"
        )


def test_real_extractor_persist_and_outcome_replay(pipeline):
    engine, worker, client = pipeline
    response = client.post(
        "/api/publish_interaction", json=payload("extract", 10, force=True)
    )
    assert response.json()["success"]
    # The suite's external LLM double still exercises actual extraction,
    # deduplication, writes and receipt-backed finalization.
    worker.drain_org("stream-test", 1, 300)
    storage = engine.get_storage()
    assert storage.extraction_status("u", "extract")["status"] == "done"
    assert storage.count_all_profiles() > 0
    before = storage.count_all_profiles()
    with patch.object(
        ProfileGenerationService,
        "compute_generation",
        side_effect=AssertionError("must not extract twice"),
    ):
        worker.drain_org("stream-test", 2, 300)
    assert storage.count_all_profiles() == before


def test_computed_outcome_survives_failed_commit_without_model_replay(
    pipeline, monkeypatch
):
    engine, worker, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("recover", force=True)
    ).json()["success"]
    original = ProfileGenerationService.persist_generation
    monkeypatch.setattr(
        ProfileGenerationService,
        "persist_generation",
        lambda *_: (_ for _ in ()).throw(RuntimeError("commit unavailable")),
    )
    worker.drain_org("stream-test", 1, 300)
    storage = engine.get_storage()
    assert storage.count_all_profiles() == 0
    assert storage.extraction_status("u", "recover")["reason"] == "retrying"
    monkeypatch.setattr(ProfileGenerationService, "persist_generation", original)
    monkeypatch.setattr(
        ProfileGenerationService,
        "compute_generation",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("computed output must be reused")
        ),
    )
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_cursors SET retry_at=0")
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert storage.extraction_status("u", "recover")["status"] == "done"
    assert storage.count_all_profiles() > 0


def test_billing_retry_does_not_repeat_extraction(pipeline, monkeypatch):
    engine, worker, client = pipeline
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor

    assert client.post(
        "/api/publish_interaction", json=payload("billing", force=True)
    ).json()["success"]
    original = WindowExecutor._bill
    monkeypatch.setattr(
        WindowExecutor,
        "_bill",
        lambda *_: (_ for _ in ()).throw(RuntimeError("billing unavailable")),
    )
    worker.drain_org("stream-test", 2, 300)
    storage = engine.get_storage()
    assert storage.extraction_status("u", "billing")["status"] == "done"
    before = storage.count_all_profiles()
    assert before > 0
    monkeypatch.setattr(WindowExecutor, "_bill", original)
    monkeypatch.setattr(
        ProfileGenerationService,
        "compute_generation",
        lambda *_: (_ for _ in ()).throw(AssertionError("no model replay for billing")),
    )
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_windows SET effects_retry_at=0")
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert storage.pending_extraction_effects() == []
    assert storage.count_all_profiles() == before


def test_transient_failures_retry_beyond_three_attempts(pipeline, monkeypatch):
    engine, worker, client = pipeline
    assert client.post("/api/publish_interaction", json=payload("retry")).json()[
        "success"
    ]
    attempts = []

    def gate(*_):
        attempts.append(1)
        if len(attempts) <= 4:
            raise RuntimeError("provider temporarily unavailable")
        return False

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    storage = engine.get_storage()
    for _ in range(5):
        worker.drain_org("stream-test", 1, 300)
        with storage._stream_sql() as db:
            db.query("UPDATE extraction_cursors SET retry_at=0")
    assert len(attempts) == 5
    assert storage.extraction_status("u", "retry")["status"] == "done"


def test_worker_heartbeat_keeps_long_window_exclusive(pipeline, monkeypatch):
    engine, worker, client = pipeline
    entered, release = threading.Event(), threading.Event()

    def gate(*_):
        entered.set()
        assert release.wait(8)
        return False

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    assert client.post("/api/publish_interaction", json=payload("long")).json()[
        "success"
    ]
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(worker.drain_org, "stream-test", 1, 3)
        assert entered.wait(3)
        time.sleep(3.2)
        assert engine.get_storage().claim_extraction("racing-worker", 3) is None
        release.set()
        assert running.result(timeout=3) == 1


def test_compute_has_no_learning_writes_and_commit_has_no_model_calls(
    pipeline, monkeypatch
):
    engine, _, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("pure", force=True)
    ).json()["success"]
    from reflexio.lib._base import create_generation_litellm_client
    from reflexio.server.services.durable_learning.window_codec import encode_plan
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor

    storage = engine.get_storage()
    user, token = storage.claim_extraction("test", 300)
    window = storage.prepare_extraction(user, token)
    executor = WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    service, request, _ = executor.service(window)
    with monkeypatch.context() as patches:
        for name in (
            "add_user_profile",
            "supersede_profiles_by_ids",
            "delete_profiles_by_ids",
            "save_user_playbooks",
            "supersede_user_playbooks_by_ids",
            "delete_user_playbooks_by_ids",
            "merge_records",
            "supersede_record",
            "update_operation_state",
            "upsert_operation_state",
        ):
            patches.setattr(
                storage,
                name,
                lambda *_, **__: pytest.fail("compute wrote a learning or bookmark"),
            )
        plan = service.compute_generation(request)
    assert plan is not None
    storage.save_extraction_outcome(window, token, encode_plan(plan, service))
    replay = storage.prepare_extraction(user, token)
    with monkeypatch.context() as patches:
        patches.setattr(
            ProfileGenerationService,
            "compute_generation",
            lambda *_: pytest.fail("must reuse compute"),
        )
        for name in ("_get_embedding", "_expand_document"):
            patches.setattr(
                storage,
                name,
                lambda *_, **__: pytest.fail("network work during commit"),
            )
        patches.setattr(
            "litellm.completion", lambda *_, **__: pytest.fail("model during commit")
        )
        executor.execute(replay, token)
    assert storage.count_all_profiles() > 0
    assert storage.extraction_status(user, "pure")["status"] == "done"


def test_superseded_worker_rolls_back_artifacts_cursor_and_billing(
    pipeline, monkeypatch
):
    engine, worker, client = pipeline
    storage = engine.get_storage()
    assert client.post(
        "/api/publish_interaction", json=payload("fenced", force=True)
    ).json()["success"]
    original = storage.complete_extraction

    def superseded(window, token, effects):
        return original(window, "superseded", effects)

    monkeypatch.setattr(storage, "complete_extraction", superseded)
    worker.drain_org("stream-test", 1, 300)
    assert storage.count_all_profiles() == 0
    assert storage.extraction_status("u", "fenced")["status"] != "done"
    assert storage.pending_extraction_effects() == []


def test_best_effort_tagging_failure_does_not_replay_winner_dispatch(
    pipeline, monkeypatch
):
    engine, worker, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("tag", force=True)
    ).json()["success"]
    calls = []

    def tagging(**kwargs):
        calls.append(kwargs["user_id"])
        if len(calls) == 1:
            raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_scheduler.schedule_tagging", tagging
    )
    worker.drain_org("stream-test", 2, 300)
    assert calls == ["u"]
    storage = engine.get_storage()
    monkeypatch.setattr(
        ProfileGenerationService,
        "compute_generation",
        lambda *_: pytest.fail("no extraction replay"),
    )
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_windows SET effects_retry_at=0")
    worker.drain_org("stream-test", 1, 300)
    assert calls == ["u"]
    assert storage.pending_extraction_effects() == []


def test_occupied_budget_leaves_durable_work_unclaimed(pipeline, monkeypatch):
    from reflexio.server.services.durable_learning import worker as module

    engine, worker, client = pipeline
    assert client.post("/api/publish_interaction", json=payload("capacity")).json()[
        "success"
    ]
    budget = threading.BoundedSemaphore(1)
    monkeypatch.setattr(module, "_budget", budget)
    assert budget.acquire(False)
    assert not worker.start_org("stream-test", 300)
    storage = engine.get_storage()
    with storage._stream_sql() as db:
        assert (
            db.query("SELECT lease_token FROM learning_work")[0]["lease_token"] is None
        )
    budget.release()
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", lambda *_: False
    )
    assert worker.drain_org("stream-test", 1, 300) == 1


def test_two_users_compute_concurrently_and_same_user_waits(pipeline, monkeypatch):
    _, worker, client = pipeline
    from reflexio.server.services.durable_learning import worker as module

    monkeypatch.setattr(module, "_budget", threading.BoundedSemaphore(2))
    entered = threading.Barrier(2)
    release = threading.Event()

    def gate(*_):
        entered.wait(timeout=3)
        assert release.wait(3)
        return False

    monkeypatch.setattr(ProfileGenerationService, "_should_run_before_extraction", gate)
    for user in ["u", "v"]:
        assert client.post(
            "/api/publish_interaction", json=payload(user, user_id=user)
        ).json()["success"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = [pool.submit(worker.drain_org, "stream-test", 1, 300) for _ in range(2)]
        deadline = time.monotonic() + 3
        while entered.n_waiting != 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        release.set()
        assert [run.result(timeout=5) for run in runs] == [1, 1]


def test_http_budget_includes_ingestion_and_never_acknowledges_uncommitted_work(
    pipeline, monkeypatch
):
    engine, _, client = pipeline
    from reflexio.server.routes import interactions as routes

    monkeypatch.setattr(routes, "PUBLISH_REQUEST_TIMEOUT_SECONDS", 0.1)
    released = threading.Event()

    def slow_embedding(*_):
        assert released.wait(2)

    monkeypatch.setattr(
        engine.get_storage(), "prepare_interaction_embeddings", slow_embedding
    )
    import asyncio

    import httpx

    async def invoke():
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=client.app), base_url="http://test"
            ) as session:
                response = await session.post(
                    "/api/publish_interaction", json=payload("slow")
                )
            assert response.status_code == 504
            assert response.json()["detail"]["request_id"] == "slow"
            assert time.monotonic() - started < 1
            assert engine.get_storage().get_request("slow") is None
        finally:
            released.set()

    asyncio.run(invoke())


def test_sync_tail_timeout_keeps_durable_pending_work(pipeline, monkeypatch):
    """A sync tail that runs out of clock must not discard the durable work.

    Publishes a FULL window (10, matching the fixture's ``window_size``) so the
    cursor can select one and the status is ``queued`` -- work the wait could
    legitimately sit through. Nothing drains it here, so the deadline is what
    ends the wait, which is the path this test exists to cover. An earlier
    version published 1 interaction, which now returns early on
    ``waiting_for_window`` and so no longer reaches the timeout branch at all.
    """
    engine, _, client = pipeline
    monkeypatch.setattr(
        "reflexio.server.routes.interactions.PUBLISH_REQUEST_TIMEOUT_SECONDS", 0.1
    )
    response = client.post(
        "/api/publish_interaction?wait_for_response=true", json=payload("tailwait")
    )
    assert response.status_code == 200 and response.json()["success"]
    assert response.json()["learning_reason"] == "wait_timeout"
    assert (
        engine.get_storage().extraction_status("u", "tailwait")["status"] == "pending"
    )


def test_sync_tail_returns_early_on_incomplete_window(pipeline):
    """A partial window ends the wait at once, and keeps the work pending.

    The fixture's ``window_size`` is 10, so a single interaction cannot close a
    window and no amount of waiting will change that. The full
    ``PUBLISH_REQUEST_TIMEOUT_SECONDS`` (240s) is left in place deliberately:
    the point is that the route does not consume it. The reason must stay
    distinguishable from a real timeout -- ``wait_timeout`` on a healthy stream
    would send an operator looking for a stall that is not there.
    """
    engine, _, client = pipeline
    started = time.monotonic()
    response = client.post(
        "/api/publish_interaction?wait_for_response=true", json=payload("partial", 1)
    )
    elapsed = time.monotonic() - started

    assert response.status_code == 200 and response.json()["success"]
    assert response.json()["learning_status"] == "deferred"
    assert response.json()["learning_reason"] == "waiting_for_window"
    assert elapsed < 30, f"route consumed {elapsed:.1f}s of the publish deadline"
    # The input is admitted and still owed extraction -- returning early is not
    # the same as dropping the work.
    assert engine.get_storage().extraction_status("u", "partial")["status"] == "pending"


def test_waiter_capacity_does_not_block_admission(pipeline):
    engine, _, client = pipeline
    from reflexio.server.services.durable_learning.waiting import (
        acquire_waiter,
        release_waiter,
    )

    for _ in range(8):
        assert acquire_waiter("stream-test")
    try:
        response = client.post(
            "/api/publish_interaction?wait_for_response=true", json=payload("waiters")
        )
        assert response.json()["success"]
        assert response.json()["learning_reason"] == "waiter_capacity"
        assert engine.get_storage().get_request("waiters") is not None
    finally:
        for _ in range(8):
            release_waiter("stream-test")


def test_outbox_ack_failure_does_not_repeat_derived_dispatch(pipeline, monkeypatch):
    engine, worker, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("ack", force=True)
    ).json()["success"]
    dispatches = []
    monkeypatch.setattr(
        ProfileGenerationService,
        "emit_generation_side_effects",
        lambda *_: dispatches.append("profile"),
    )
    storage = engine.get_storage()
    ack = storage.ack_extraction_effects
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("ack unavailable")
        return ack(*args, **kwargs)

    monkeypatch.setattr(storage, "ack_extraction_effects", fail_first)
    worker.drain_org("stream-test", 2, 300)
    assert dispatches == ["profile"]
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_windows SET effects_retry_at=0")
        db.query("UPDATE learning_work SET due_at=0")
    worker.drain_org("stream-test", 2, 300)
    assert dispatches == ["profile"]
    assert storage.pending_extraction_effects() == []


def test_erasure_cancels_uncommitted_window_run_instead_of_resuming_it(
    pipeline, monkeypatch
):
    from datetime import UTC, datetime, timedelta

    from reflexio.server.services.storage.storage_base import (
        AgentBinding,
        AgentRunRecord,
        AgentRunStatus,
    )

    engine, worker, client = pipeline
    assert client.post(
        "/api/publish_interaction", json=payload("erased", force=True)
    ).json()["success"]
    monkeypatch.setattr(
        ProfileGenerationService,
        "persist_generation",
        lambda *_: (_ for _ in ()).throw(RuntimeError("commit unavailable")),
    )
    worker.drain_org("stream-test", 1, 300)
    storage = engine.get_storage()
    with storage._stream_sql() as db:
        window_id = db.query("SELECT window_id FROM extraction_windows")[0]["window_id"]
    # The external model double skips the resumable loop, so seed the same
    # durable agent-completed state that a crash after a real model call leaves.
    storage.create_agent_run(
        AgentRunRecord(
            id=f"window:{window_id}",
            binding=AgentBinding(
                org_id="stream-test",
                user_id="u",
                extractor_kind="profile",
                request_id="erased",
                agent_version=None,
                source="test",
            ),
            status=AgentRunStatus.AGENT_COMPLETED,
            generation_request_snapshot={},
            committed_output={"profiles": []},
        )
    )
    # Even an old completed model call belongs to the window engine until its
    # learning writes and cursor commit together.
    assert (
        storage.claim_finalization_failed_agent_run(
            org_id="stream-test",
            worker_id="resume",
            now=datetime.now(UTC) + timedelta(hours=1),
        )
        is None
    )
    storage.delete_all_interactions()
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_cursors SET retry_at=0")
        db.query("UPDATE learning_work SET due_at=0")
    worker.drain_org("stream-test", 1, 300)
    assert (
        storage.get_agent_run(f"window:{window_id}").status == AgentRunStatus.CANCELLED
    )
    assert storage.count_all_profiles() == 0
