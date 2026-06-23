"""Integration tests for source-set comparison in EvaluationOverviewService.

Uses a real SQLite storage in a temp dir (no mocks) so the join between eval
results and the requests table is exercised end-to-end. The ``_get_embedding``
call on the SQLite backend is patched out — without it,
``save_agent_success_evaluation_results`` would try to hit a real LLM endpoint
just to embed the (empty) failure text.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentSuccessEvaluationResult,
    Request,
)
from reflexio.models.api_schema.eval_overview_schema import (
    EvaluationSourceSetRequest,
    GetEvaluationOverviewRequest,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.evaluation_overview.service import (
    EvaluationOverviewService,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage() -> Generator[SQLiteStorage]:
    """Yield a fresh SQLite store in a temp dir with embedding stubbed."""
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(
            org_id="eval_overview_grouping_test",
            db_path=f"{tmp_dir}/reflexio.db",
        )


def _seed_session(
    storage: SQLiteStorage,
    session_id: str,
    ts: int,
    is_success: bool,
    user_id: str = "u1",
    source: str = "test",
    evaluation_only: bool = False,
) -> None:
    """Add one Request and one eval result for the session."""
    storage.add_request(
        Request(
            request_id=f"req-{session_id}",
            user_id=user_id,
            created_at=ts,
            source=source,
            agent_version="v1",
            session_id=session_id,
            evaluation_only=evaluation_only,
        )
    )
    storage.save_agent_success_evaluation_results(
        [
            AgentSuccessEvaluationResult(
                user_id=user_id,
                agent_version="v1",
                session_id=session_id,
                is_success=is_success,
                evaluation_name="overall",
                created_at=ts,
            )
        ]
    )


def test_run_exposes_available_sources_from_first_session_request(
    storage: SQLiteStorage,
) -> None:
    base_ts = 1_700_000_000
    _seed_session(storage, "s1", base_ts, True, source="baseline")
    _seed_session(storage, "s2", base_ts, False, source="")

    service = EvaluationOverviewService(
        storage=storage, config=Config(storage_config=StorageConfigSQLite())
    )
    response = service.run(
        GetEvaluationOverviewRequest(from_ts=base_ts - 1, to_ts=base_ts + 1)
    )

    assert response.source_set_comparison.available_sources == ["", "baseline"]
    assert response.source_set_comparison.sets == []
    assert response.source_set_comparison.unmatched_session_count == 0


def test_run_computes_source_set_metrics_by_first_request_source(
    storage: SQLiteStorage,
) -> None:
    base_ts = 1_700_000_000
    _seed_session(storage, "s1", base_ts, True, source="baseline")
    _seed_session(storage, "s2", base_ts, False, source="baseline")
    _seed_session(storage, "s3", base_ts, True, source="candidate")
    _seed_session(storage, "s4", base_ts, True, source="other")
    storage.add_request(
        Request(
            request_id="req-s3-later",
            user_id="u1",
            created_at=base_ts + 1,
            source="baseline",
            agent_version="v1",
            session_id="s3",
        )
    )

    service = EvaluationOverviewService(
        storage=storage, config=Config(storage_config=StorageConfigSQLite())
    )
    response = service.run(
        GetEvaluationOverviewRequest(
            from_ts=base_ts - 1,
            to_ts=base_ts + 2,
            source_sets=[
                EvaluationSourceSetRequest(label="Baseline", sources=["baseline"]),
                EvaluationSourceSetRequest(label="Candidate", sources=["candidate"]),
            ],
        )
    )

    comparison = response.source_set_comparison
    assert comparison.available_sources == ["baseline", "candidate", "other"]
    assert comparison.unmatched_session_count == 1

    by_label = {row.label: row for row in comparison.sets}
    assert by_label["Baseline"].session_count == 2
    assert by_label["Baseline"].session_ids == ["s1", "s2"]
    assert by_label["Baseline"].success_rate_pp == 50.0
    assert by_label["Baseline"].context_tiles.success.current == 50.0

    assert by_label["Candidate"].session_count == 1
    assert by_label["Candidate"].session_ids == ["s3"]
    assert by_label["Candidate"].success_rate_pp == 100.0


def test_run_source_set_includes_evaluation_only_requests(
    storage: SQLiteStorage,
) -> None:
    base_ts = 1_700_000_000
    _seed_session(
        storage,
        "s1",
        base_ts,
        True,
        source="eval_only_source",
        evaluation_only=True,
    )

    service = EvaluationOverviewService(
        storage=storage, config=Config(storage_config=StorageConfigSQLite())
    )
    response = service.run(
        GetEvaluationOverviewRequest(
            from_ts=base_ts - 1,
            to_ts=base_ts + 1,
            source_sets=[
                EvaluationSourceSetRequest(
                    label="Eval only", sources=["eval_only_source"]
                )
            ],
        )
    )

    row = response.source_set_comparison.sets[0]
    assert row.session_count == 1
    assert row.session_ids == ["s1"]
    assert row.success_rate_pp == 100.0
