"""Integration test: regen worker bounds concurrent in-flight LLM judge
calls at ``config.eval_concurrency_limit`` via a ``ThreadPoolExecutor``.

Uses a real SQLite storage (in a temp dir) so candidate discovery runs
exactly as it would in production. The LLM judge inside
``run_group_evaluation`` is replaced by a stub that records the
high-water mark of concurrently-executing calls. We assert both that the
cap is never exceeded *and* that genuine overlap occurred — without the
ThreadPoolExecutor wiring, the sequential loop would peak at 1.
"""

from __future__ import annotations

import random
import tempfile
import threading
import time
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentSuccessEvaluationResult,
    Request,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.agent_success_evaluation import regen_jobs
from reflexio.server.services.agent_success_evaluation.regen_jobs import (
    RegenJob,
    run_regen,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage() -> Generator[SQLiteStorage]:
    """Fresh SQLite store in a temp dir with embedding stubbed."""
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(
            org_id="regen_concurrency_test",
            db_path=f"{tmp_dir}/reflexio.db",
        )


def _build_request_context(storage: SQLiteStorage, config: Config) -> SimpleNamespace:
    """Build a minimal request_context exposing ``.storage`` and
    ``.configurator.get_config()`` — the only attributes ``run_regen``
    reads."""
    configurator = SimpleNamespace(get_config=lambda: config)
    return SimpleNamespace(storage=storage, configurator=configurator)


def _seed(storage: SQLiteStorage, session_id: str, ts: int) -> None:
    """Seed a session with one Request and one eval result so it appears
    as a candidate in ``get_session_ids_in_window``."""
    storage.add_request(
        Request(
            request_id=f"req-{session_id}",
            user_id="u1",
            created_at=ts,
            source="test",
            agent_version="v1",
            session_id=session_id,
            metadata={"reflexio_retrieval_enabled": True},
        )
    )
    storage.save_agent_success_evaluation_results(
        [
            AgentSuccessEvaluationResult(
                agent_version="v1",
                session_id=session_id,
                is_success=True,
                evaluation_name="overall",
                created_at=ts,
            )
        ]
    )


def test_run_regen_bounds_concurrent_inflight(
    monkeypatch: pytest.MonkeyPatch, storage: SQLiteStorage
) -> None:
    """``run_group_evaluation`` stub records the in-flight high-water
    mark; it must never exceed ``config.eval_concurrency_limit`` and
    must exceed 1 (confirming real overlap)."""
    ts = 1_700_000_000
    for i in range(50):
        _seed(storage, f"s{i}", ts)

    inflight = 0
    high_water = 0
    lock = threading.Lock()

    def fake_run_group_evaluation(**_kwargs: object) -> None:
        nonlocal inflight, high_water
        with lock:
            inflight += 1
            high_water = max(high_water, inflight)
        # Hold the slot long enough for the executor to enqueue more
        # workers; 20ms is comfortably above thread-scheduling jitter.
        time.sleep(0.02)
        with lock:
            inflight -= 1

    monkeypatch.setattr(regen_jobs, "run_group_evaluation", fake_run_group_evaluation)

    config = Config(
        storage_config=StorageConfigSQLite(),
        eval_sample_n_per_stratum=200,
        eval_concurrency_limit=5,
    )
    rc = _build_request_context(storage, config)
    job = RegenJob(
        job_id="j1",
        org_id="0",
        evaluation_name="overall",
        from_ts=ts - 1,
        to_ts=ts + 1,
        status="running",
        total=0,
    )

    run_regen(
        job=job,
        request_context=rc,  # type: ignore[arg-type]  # SimpleNamespace stand-in
        llm_client=None,  # type: ignore[arg-type]
        rng=random.Random(0),  # noqa: S311 — sampling, not crypto
    )

    # Concurrency bound never exceeded.
    assert high_water <= 5, (
        f"high_water={high_water} exceeded eval_concurrency_limit=5; "
        "ThreadPoolExecutor cap is not enforced."
    )
    # Concurrency actually happened — without ThreadPoolExecutor wiring,
    # the sequential loop releases each slot before the next iteration
    # begins, so high_water peaks at 1.
    assert high_water >= 2, (
        f"Expected concurrent execution (high_water >= 2), got {high_water}. "
        "Either the ThreadPoolExecutor wiring is missing or the sleep is too "
        "short for overlap."
    )
    # All 50 sessions dispatched.
    assert job.completed == 50
    assert job.failed == 0
    # Counter populated.
    assert job.concurrency_limit == 5
    assert job.status == "completed"
