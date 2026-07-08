"""Regression tests for group evaluation no longer dispatching shadow comparison."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import Interaction, Request
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.agent_success_evaluation.runner import (
    run_group_evaluation,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage() -> Generator[SQLiteStorage]:
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(
            org_id="f1_runner_test",
            db_path=f"{tmp_dir}/reflexio.db",
        )


def _seed_session_with_shadow_interaction(storage: SQLiteStorage) -> None:
    ts = 1_700_000_000
    storage.add_request(
        Request(
            request_id="req-1",
            user_id="u1",
            session_id="sess-1",
            created_at=ts,
            source="test",
            agent_version="v1",
        )
    )
    storage.add_user_interaction(
        "u1",
        Interaction(
            user_id="u1",
            request_id="req-1",
            created_at=ts + 1,
            role="assistant",
            content="REGULAR",
            shadow_content="SHADOW",
        ),
    )


def _make_request_context(storage: SQLiteStorage) -> SimpleNamespace:
    return SimpleNamespace(
        storage=storage,
        configurator=SimpleNamespace(
            get_config=lambda: Config(storage_config=StorageConfigSQLite())
        ),
        prompt_manager=MagicMock(),
    )


def test_run_group_evaluation_no_longer_writes_shadow_verdicts(
    monkeypatch: pytest.MonkeyPatch,
    storage: SQLiteStorage,
) -> None:
    """Regen/session evaluation must not duplicate publish-time shadow verdicts."""
    _seed_session_with_shadow_interaction(storage)
    fake_service = MagicMock()
    fake_service.has_run_failures.return_value = False
    fake_service.last_run_saved_result_count = 1
    fake_service.last_run_save_failed = False
    monkeypatch.setattr(
        "reflexio.server.services.agent_success_evaluation."
        "runner.AgentSuccessEvaluationService",
        MagicMock(return_value=fake_service),
    )
    judge_cls = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.dispatcher.ShadowComparisonJudge",
        judge_cls,
    )

    run_group_evaluation(
        org_id="0",
        user_id="u1",
        session_id="sess-1",
        agent_version="v1",
        source=None,
        request_context=_make_request_context(storage),  # type: ignore[arg-type]
        llm_client=MagicMock(),
        force_regenerate=True,
    )

    judge_cls.assert_not_called()
    assert (
        storage.get_shadow_comparison_verdicts(
            from_ts=0,
            to_ts=2_000_000_000,
            judge_prompt_version="v1.1.0",
        )
        == []
    )
