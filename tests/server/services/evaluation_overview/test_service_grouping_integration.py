"""Integration tests for the F2 group-by aggregation wired into
EvaluationOverviewService.run().

The service must:
- Look up each session's first Request and read its metadata.
- Bucket session outcomes by the metadata's reflexio_retrieval_enabled value.
- Populate success_rate_trend_by_group on the response.

Uses a real SQLite storage in a temp dir (no mocks) so the join between
eval results and the requests table is exercised end-to-end. The
``_get_embedding`` call on the SQLite backend is patched out — without
it, ``save_agent_success_evaluation_results`` would try to hit a real
LLM endpoint just to embed the (empty) failure text.
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
    metadata: dict,
    user_id: str = "u1",
) -> None:
    """Add one Request (carrying metadata) and one eval result for the session."""
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
                is_success=is_success,
                evaluation_name="overall",
                created_at=ts,
            )
        ]
    )


def test_run_produces_three_curves_with_correct_n_and_rate(
    storage: SQLiteStorage,
) -> None:
    base_ts = 1_700_000_000
    _seed_session(storage, "s1", base_ts, True, {"reflexio_retrieval_enabled": True})
    _seed_session(storage, "s2", base_ts, True, {"reflexio_retrieval_enabled": True})
    _seed_session(storage, "s3", base_ts, False, {"reflexio_retrieval_enabled": True})
    _seed_session(storage, "s4", base_ts, True, {"reflexio_retrieval_enabled": False})
    _seed_session(storage, "s5", base_ts, False, {"reflexio_retrieval_enabled": False})
    _seed_session(storage, "s6", base_ts, True, {})

    service = EvaluationOverviewService(
        storage=storage, config=Config(storage_config=StorageConfigSQLite())
    )
    response = service.run(
        GetEvaluationOverviewRequest(from_ts=base_ts - 1, to_ts=base_ts + 1)
    )

    g = response.success_rate_trend_by_group
    assert len(g.treatment) == 1
    assert g.treatment[0].n == 3
    assert abs(g.treatment[0].rate - (2 / 3)) < 1e-9

    assert len(g.control) == 1
    assert g.control[0].n == 2
    assert g.control[0].rate == 0.5

    assert len(g.untagged) == 1
    assert g.untagged[0].n == 1
    assert g.untagged[0].rate == 1.0


def test_run_untagged_only_falls_back_to_empty_treatment_control(
    storage: SQLiteStorage,
) -> None:
    base_ts = 1_700_000_000
    _seed_session(storage, "s1", base_ts, True, {})
    _seed_session(storage, "s2", base_ts, False, {})

    service = EvaluationOverviewService(
        storage=storage, config=Config(storage_config=StorageConfigSQLite())
    )
    response = service.run(
        GetEvaluationOverviewRequest(from_ts=base_ts - 1, to_ts=base_ts + 1)
    )

    g = response.success_rate_trend_by_group
    assert g.treatment == []
    assert g.control == []
    assert len(g.untagged) == 1
    assert g.untagged[0].n == 2


def test_run_no_sessions_returns_empty_group_trend(
    storage: SQLiteStorage,
) -> None:
    service = EvaluationOverviewService(
        storage=storage, config=Config(storage_config=StorageConfigSQLite())
    )
    response = service.run(GetEvaluationOverviewRequest(from_ts=0, to_ts=1))
    g = response.success_rate_trend_by_group
    assert g.treatment == []
    assert g.control == []
    assert g.untagged == []
