"""Integration test: regen pipeline samples candidates per-stratum before
dispatching to ``run_group_evaluation``.

Uses a real SQLite storage (in a temp dir) so the candidate-discovery
path and per-session first-request metadata lookup run exactly as they
would in production. The LLM judge inside ``run_group_evaluation`` is
patched out — the test asserts dispatch counts and job counter math,
not the judge's output.
"""

from __future__ import annotations

import random
import tempfile
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
            org_id="regen_sampling_test",
            db_path=f"{tmp_dir}/reflexio.db",
        )


def _build_request_context(storage: SQLiteStorage, config: Config) -> SimpleNamespace:
    """Build a minimal request_context with the fields ``run_regen`` reads.

    Only ``.storage`` and ``.configurator.get_config()`` are exercised by
    the regen worker, so a plain SimpleNamespace is sufficient and far
    simpler than spinning up a full BaseConfigurator stack.
    """
    configurator = SimpleNamespace(get_config=lambda: config)
    return SimpleNamespace(storage=storage, configurator=configurator)


def _seed(
    storage: SQLiteStorage,
    session_id: str,
    ts: int,
    metadata: dict,
    user_id: str = "u1",
) -> None:
    """Seed a session with one Request and one eval result so it appears
    as a candidate in ``get_session_ids_in_window``."""
    storage.add_request(
        Request(
            request_id=f"req-{session_id}",
            user_id=user_id,
            created_at=ts,
            source="test",
            agent_version="v1",
            session_id=session_id,
            metadata=metadata,
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


def test_run_regen_samples_when_stratum_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch, storage: SQLiteStorage
) -> None:
    """500 candidates (250 treatment, 250 control) on a single day with
    N=50/stratum -> at most 100 dispatches; counters reflect the sample."""
    ts = 1_700_000_000
    for i in range(250):
        _seed(storage, f"t{i}", ts, {"reflexio_retrieval_enabled": True})
    for i in range(250):
        _seed(storage, f"c{i}", ts, {"reflexio_retrieval_enabled": False})

    call_log: list[str] = []

    def fake_run(**kwargs: object) -> None:
        call_log.append(str(kwargs["session_id"]))

    monkeypatch.setattr(regen_jobs, "run_group_evaluation", fake_run)

    config = Config(
        storage_config=StorageConfigSQLite(),
        eval_sample_n_per_stratum=50,
        eval_concurrency_limit=2,
    )
    rc = _build_request_context(storage, config)
    job = RegenJob(
        job_id="j1",
        org_id="0",
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

    # Two strata (one day x {treatment, control}) x 50 cap = at most 100.
    assert len(call_log) <= 100
    assert job.total_candidates == 500
    assert job.sampled_count == len(call_log)
    assert job.completed + job.failed == job.sampled_count
    assert job.concurrency_limit == 2
    assert job.status == "completed"


def test_run_regen_no_sampling_when_below_cap(
    monkeypatch: pytest.MonkeyPatch, storage: SQLiteStorage
) -> None:
    """Small windows keep every candidate; ``sampled_count == total_candidates``."""
    ts = 1_700_000_000
    for i in range(10):
        _seed(storage, f"t{i}", ts, {"reflexio_retrieval_enabled": True})

    calls: list[str] = []

    def fake_run(**kwargs: object) -> None:
        calls.append(str(kwargs["session_id"]))

    monkeypatch.setattr(regen_jobs, "run_group_evaluation", fake_run)

    config = Config(
        storage_config=StorageConfigSQLite(),
        eval_sample_n_per_stratum=200,
        eval_concurrency_limit=2,
    )
    rc = _build_request_context(storage, config)
    job = RegenJob(
        job_id="j2",
        org_id="0",
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

    assert len(calls) == 10
    assert job.total_candidates == 10
    assert job.sampled_count == 10
    assert job.completed == 10
    assert job.status == "completed"


def test_run_regen_continues_when_one_session_storage_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, storage: SQLiteStorage
) -> None:
    """A storage glitch on one session's metadata lookup must not abort
    the whole job — the session is sampled with fallback metadata and the
    per-session loop reports its own failure (or success) for it.
    """
    ts = 1_700_000_000
    for i in range(5):
        _seed(storage, f"ok-{i}", ts, {"reflexio_retrieval_enabled": True})

    real_get = storage.get_requests_by_session
    poison_session = "ok-2"

    def flaky_get_requests_by_session(user_id: str, session_id: str) -> list[Request]:
        if session_id == poison_session:
            raise RuntimeError("simulated transient DB error")
        return real_get(user_id, session_id)

    monkeypatch.setattr(
        storage, "get_requests_by_session", flaky_get_requests_by_session
    )

    calls: list[str] = []

    def fake_run(**kwargs: object) -> None:
        calls.append(str(kwargs["session_id"]))

    monkeypatch.setattr(regen_jobs, "run_group_evaluation", fake_run)

    config = Config(
        storage_config=StorageConfigSQLite(),
        eval_sample_n_per_stratum=200,
        eval_concurrency_limit=2,
    )
    rc = _build_request_context(storage, config)
    job = RegenJob(
        job_id="j-flaky",
        org_id="0",
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

    # All 5 sessions still dispatched to run_group_evaluation; the job
    # didn't error out at the candidate-discovery stage.
    assert len(calls) == 5
    assert job.status != "error"
