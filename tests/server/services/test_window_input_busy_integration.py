"""Input contention yields a turn without consuming its window failure budget."""

from unittest.mock import Mock

import pytest

from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.profile.service import ProfileGenerationService
from reflexio.server.services.storage.error import ExecutionInputBusyError, StorageError
from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
)
from reflexio.server.services.storage.storage_base import LeaseLostError
from reflexio.server.services.storage.storage_base._extraction_stream import (
    MAX_EXTRACTION_WINDOW_ATTEMPTS,
)
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline
from tests.server.services.test_window_readiness_integration import _state

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("wrapped", [False, True])
def test_busy_persist_yields_without_timer_or_retry_spending_and_reuses_saved_compute(
    pipeline, monkeypatch, wrapped
):
    engine, worker, http = pipeline
    assert http.post(
        "/api/publish_interaction", json=payload("input-busy", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    with storage._stream_sql(read_only=True) as db:
        clock = [db.now()]
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: clock[0])
    ready = False
    persist = ProfileGenerationService.persist_generation
    compute = Mock(wraps=ProfileGenerationService.compute_generation)
    monkeypatch.setattr(
        ProfileGenerationService,
        "compute_generation",
        lambda self, *args, **kw: compute(self, *args, **kw),
    )

    def write(service, plan):
        if not ready:
            if wrapped:
                try:
                    raise ExecutionInputBusyError("synthetic row is held")
                except ExecutionInputBusyError as error:
                    raise StorageError("explicit wrapper") from error
            raise ExecutionInputBusyError("synthetic row is held")
        return persist(service, plan)

    monkeypatch.setattr(ProfileGenerationService, "persist_generation", write)
    saved = None
    for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS + 1):
        assert worker.drain_org("stream-test", 1, 300) == 0
        cursor, windows, work = _state(storage)
        assert cursor["attempts"] == 0 and cursor["last_error"] is None
        assert cursor["retry_at"] != float("inf")
        assert len(windows) == 1 and not windows[0]["completed"]
        saved = windows[0]["outcome"] if saved is None else saved
        assert saved is not None and windows[0]["outcome"] == saved
        assert compute.call_count == 1, "A busy writer must retain its saved compute"
        assert work == {"due_at": clock[0], "lease_token": None}
        assert storage.count_all_profiles() == 0
        clock[0] += 1
    ready = True
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert compute.call_count == 1
    cursor, windows, _ = _state(storage)
    assert cursor["attempts"] == 0 and windows[0]["completed"]
    count = storage.count_all_profiles()
    assert count > 0
    assert worker.drain_org("stream-test", 1, 300) == 1
    assert _state(storage)[1][0]["effects_done"]
    assert worker.drain_org("stream-test", 1, 300) == 0
    assert storage.count_all_profiles() == count


@pytest.mark.parametrize("withdrawal", ["replacement", "erasure"])
def test_busy_yield_requires_the_actual_live_owner(pipeline, withdrawal):
    engine, _, http = pipeline
    assert http.post(
        "/api/publish_interaction", json=payload("stale-busy", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("old", 300)
    window = storage.prepare_extraction(user, token)
    assert window is not None
    if withdrawal == "replacement":
        with storage._stream_sql() as db:
            db.query("UPDATE learning_work SET lease_until=0 WHERE user_id=?", (user,))
        replacement = storage.claim_user_extraction(user, "new", 300)
        assert replacement is not None and replacement != token
        before = _state(storage)
    else:
        storage.clear_user_data(user)
    with pytest.raises(LeaseLostError):
        DurableLearningWorker._handle_window_failure(
            storage, window, token, ExecutionInputBusyError("busy")
        )
    storage.release_user_extraction(user, token)
    if withdrawal == "replacement":
        assert _state(storage) == before
        storage.release_user_extraction(user, replacement)
    else:
        with storage._stream_sql(read_only=True) as db:
            assert (
                db.query("SELECT user_id FROM learning_work WHERE user_id=?", (user,))
                == []
            )


def test_unrelated_failure_while_handling_busy_keeps_its_own_retry_classification(
    pipeline, monkeypatch
):
    engine, worker, http = pipeline
    assert http.post(
        "/api/publish_interaction", json=payload("not-busy", force=True)
    ).json()["success"]

    def unrelated(_service, *_args, **_kwargs):
        try:
            raise ExecutionInputBusyError("earlier handling context")
        except ExecutionInputBusyError:
            raise RuntimeError("different failure") from None

    monkeypatch.setattr(ProfileGenerationService, "compute_generation", unrelated)
    assert worker.drain_org("stream-test", 1, 300) == 0
    cursor, _, _ = _state(engine.get_storage())
    assert cursor["attempts"] == 1
    assert "RuntimeError" in cursor["last_error"]


def test_paid_compatibility_tool_contention_reaches_existing_window_ceiling(
    pipeline, monkeypatch
):
    import json
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from reflexio.server.llm._litellm_types import ModelProvenance
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor
    from reflexio.server.services.storage.storage_base import (
        PendingToolCallRecord,
        PendingToolCallStatus,
        build_pending_tool_call_dedup_key,
        build_scope_hash,
        human_feedback_scope,
    )
    from tests.server.services.test_window_retry_ceiling_integration import _drain_once

    engine, worker, http = pipeline
    storage = engine.get_storage()
    config = engine.request_context.configurator.get_config()
    config.pending_tool_call_config.enabled = True
    engine.request_context.configurator.set_config(config)
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    # Select the established historical compatibility adapter; all ownership,
    # actual tool dispatch and retry accounting remain production code.
    monkeypatch.setattr(
        WindowExecutor,
        "_requires_compatibility_execution",
        staticmethod(lambda *_a: True),
    )
    monkeypatch.setattr(
        ProfileGenerationService,
        "_should_run_before_extraction",
        lambda *_a, **_kw: True,
    )
    scope = human_feedback_scope(storage.org_id)
    question = "Which synthetic compatibility approval?"
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="held-window-question",
            org_id=storage.org_id,
            user_id="u",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            question_text=question,
            args={"question": question},
            status=PendingToolCallStatus.PENDING,
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            cache_until=datetime.now(UTC) + timedelta(minutes=10),
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human", question_text=question
            ),
        )
    )
    calls = []

    def completion(*_a, **_kw):
        calls.append(1)
        return SimpleNamespace(
            value=SimpleNamespace(
                content=None,
                tool_calls=[
                    {
                        "id": "compatibility-tool",
                        "type": "function",
                        "function": {
                            "name": "attach_pending_info_request",
                            "arguments": json.dumps(
                                {"pending_tool_call_id": "held-window-question"}
                            ),
                        },
                    }
                ],
            ),
            provenance=ModelProvenance("gpt-4o-mini", "test"),
        )

    monkeypatch.setattr(
        LiteLLMClient, "get_completion_model_ladder", lambda *_a, **_kw: ["gpt-4o-mini"]
    )
    monkeypatch.setattr(
        LiteLLMClient, "generate_chat_response_with_provenance", completion
    )
    refusals = []

    def busy(*_a, **_kw):
        refusals.append(1)
        raise ExecutionInputBusyError("synthetic input contention after model response")

    monkeypatch.setattr(storage, "_lock_pending_tool_execution_inputs", busy)
    assert http.post(
        "/api/publish_interaction", json=payload("paid-busy", force=True)
    ).json()["success"]
    for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS * 3 + 3):
        _drain_once(worker, storage)
    cursor, _, _ = _state(storage)
    assert calls and len(calls) == len(refusals), (
        "Exercise actual model/tool execution, not a gate failure"
    )
    assert len(calls) <= MAX_EXTRACTION_WINDOW_ATTEMPTS
    assert cursor["attempts"] == MAX_EXTRACTION_WINDOW_ATTEMPTS
    assert cursor["retry_at"] == float("inf")
    assert storage.count_all_profiles() == 0


@pytest.mark.parametrize("held_refusals", [1, 2])
@pytest.mark.parametrize("save_boundary", ["model_result", "window_outcome"])
def test_image_compatibility_completed_compute_save_contention_is_bounded(
    pipeline, monkeypatch, save_boundary, held_refusals
):
    import json
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    from threading import Event
    from types import SimpleNamespace

    import litellm

    from reflexio.server.llm.litellm_client import (
        LiteLLMClient,
        LiteLLMConfig,
    )
    from tests.server.services.test_window_retry_ceiling_integration import _drain_once

    engine, worker, http = pipeline
    storage = engine.get_storage()
    config = engine.request_context.configurator.get_config()
    config.pending_tool_call_config.enabled = True
    engine.request_context.configurator.set_config(config)
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "cloud")
    monkeypatch.setattr(storage, "embedding_model_name", "text-embedding-3-small")
    monkeypatch.setattr(storage, "_should_expand_documents", lambda: False)
    monkeypatch.setattr(
        storage, "llm_client", LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    )
    monkeypatch.setattr(
        litellm,
        "embedding",
        lambda **_kw: SimpleNamespace(
            data=[{"index": 0, "embedding": [0.0] * storage.embedding_dimensions}]
        ),
    )
    monkeypatch.setattr(
        ProfileGenerationService,
        "_should_run_before_extraction",
        lambda *_a, **_kw: True,
    )
    storage.conn.execute("PRAGMA busy_timeout=50")
    busy, held = Event(), Event()
    refusals, models, computes, holders = [], [], [], []
    scope = storage.commit_scope
    compute = ProfileGenerationService.compute_generation

    @contextmanager
    def observed_writer():
        try:
            with scope():
                yield
        except Exception as error:
            cause = error
            while cause is not None:
                if (
                    isinstance(cause, sqlite3.OperationalError)
                    and getattr(cause, "sqlite_errorcode", 0) & 255
                    == sqlite3.SQLITE_BUSY
                ):
                    refusals.append(1)
                    if len(refusals) % held_refusals == 0:
                        busy.set()
                    break
                cause = cause.__cause__
            raise

    def hold_writer():
        with sqlite3.connect(storage.db_path) as other:
            other.execute("BEGIN IMMEDIATE")
            other.execute("UPDATE interactions SET created_at=created_at")
            held.set()
            assert busy.wait(5), "Must observe an actual SQLite admission timeout"
            other.rollback()

    with ThreadPoolExecutor(max_workers=1) as pool:

        def block_next_save():
            busy.clear()
            held.clear()
            holders.append(pool.submit(hold_writer))
            assert held.wait(5)

        def completion(_client, params, _timeout):
            if "ProfileDeduplication" in str(params.get("response_format")):
                output = {
                    "duplicate_groups": [],
                    "unique_ids": ["NEW-0"],
                    "deletions": [],
                }
            else:
                models.append(1)
                output = {
                    "profiles": [
                        {
                            "content": "Prefer concise answers.",
                            "time_to_live": "infinity",
                        }
                    ]
                }
                if save_boundary == "model_result":
                    block_next_save()
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(output), tool_calls=None, refusal=None
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
                model="gpt-4o-mini",
                _hidden_params={"custom_llm_provider": "openai"},
            )

        def compute_then_save(service, request):
            plan = compute(service, request)
            computes.append(1)
            if save_boundary == "window_outcome":
                block_next_save()
            return plan

        monkeypatch.setattr(storage, "commit_scope", observed_writer)
        monkeypatch.setattr(
            ProfileGenerationService, "compute_generation", compute_then_save
        )
        monkeypatch.setattr(
            LiteLLMClient,
            "get_completion_model_ladder",
            lambda *_a, **_kw: ["gpt-4o-mini"],
        )
        monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", completion)
        request = payload("completed-save-busy", force=True)
        # Actual image evidence selects compatibility execution, without a
        # mocked adapter decision or any image/network/provider service.
        request["interaction_data_list"][0]["interacted_image_url"] = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+kRasAAAAASUVORK5CYII="
        )
        assert http.post("/api/publish_interaction", json=request).json()["success"]
        for _ in range(MAX_EXTRACTION_WINDOW_ATTEMPTS * 3 + 3):
            _drain_once(worker, storage)
        for holder in holders:
            holder.result(timeout=5)
    cursor, windows, _ = _state(storage)
    assert models and refusals
    if save_boundary == "model_result":
        assert len(models) * held_refusals == len(refusals)
        assert len(models) <= MAX_EXTRACTION_WINDOW_ATTEMPTS
    else:
        assert computes and len(computes) * held_refusals == len(refusals)
        assert len(computes) <= MAX_EXTRACTION_WINDOW_ATTEMPTS
    assert cursor["attempts"] == MAX_EXTRACTION_WINDOW_ATTEMPTS
    assert cursor["retry_at"] == float("inf")
    assert windows[0]["outcome"] is None and not windows[0]["completed"]
    assert storage.count_all_profiles() == 0
