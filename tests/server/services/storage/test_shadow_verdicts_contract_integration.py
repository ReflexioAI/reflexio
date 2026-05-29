"""Contract: ShadowComparisonVerdict CRUD across every locally-testable backend.

Parametrized over sqlite only after the disk backend was retired. Supabase
has its own integration test because it requires a live Postgres instance.

This file defines its own parametrized ``storage`` fixture (shadowing the
conftest one) so the new shadow-verdict CRUD surface is exercised in
isolation without enrolling pre-existing contract tests against
under-developed backends.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage() -> Generator[BaseStorage]:
    """Yield a fresh, isolated SQLite storage instance."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(
            org_id="contract_test_shadow_verdicts",
            db_path=f"{temp_dir}/reflexio.db",
        )


def _make_verdict(**overrides) -> ShadowComparisonVerdict:
    base: dict = {
        "verdict_id": 0,  # autoincrement
        "interaction_id": "cv-i1",
        "session_id": "cv-s1",
        "agent_version": "v1",
        "reflexio_is_request_1": True,
        "output": ShadowComparisonOutput(
            better_request="1",
            is_significantly_better=True,
            comparison_reason="r1 was direct",
        ),
        "judge_prompt_version": "v1.0.0",
        "created_at": datetime.now(UTC),
    }
    base.update(overrides)
    return ShadowComparisonVerdict(**base)


def test_verdict_roundtrips(storage: BaseStorage) -> None:
    v = storage.save_shadow_comparison_verdict(_make_verdict())
    assert v.verdict_id > 0
    got = storage.get_shadow_comparison_verdict(v.verdict_id)
    assert got is not None
    assert got.interaction_id == "cv-i1"
    assert got.output.better_request == "1"
    assert got.output.is_significantly_better is True
    assert got.judge_prompt_version == "v1.0.0"
    assert got.reflexio_is_request_1 is True


def test_get_verdict_returns_none_for_missing_id(storage: BaseStorage) -> None:
    """Cross-backend invariant: missing id returns None, never raises."""
    assert storage.get_shadow_comparison_verdict(999_999) is None


def test_get_in_window_excludes_prompt_version_mismatch(
    storage: BaseStorage,
) -> None:
    base_ts = datetime.now(UTC)
    storage.save_shadow_comparison_verdict(
        _make_verdict(
            interaction_id="cv-i-v1",
            created_at=base_ts,
            judge_prompt_version="v1.0.0",
        )
    )
    storage.save_shadow_comparison_verdict(
        _make_verdict(
            interaction_id="cv-i-v2",
            created_at=base_ts,
            judge_prompt_version="v2.0.0",
        )
    )
    result = storage.get_shadow_comparison_verdicts(
        from_ts=int(base_ts.timestamp()) - 10,
        to_ts=int(base_ts.timestamp()) + 10,
        judge_prompt_version="v1.0.0",
    )
    assert [r.interaction_id for r in result] == ["cv-i-v1"]


def test_get_in_window_orders_ascending_by_created_at(
    storage: BaseStorage,
) -> None:
    earlier_ts = datetime.now(UTC).timestamp() - 100
    later_ts = datetime.now(UTC).timestamp()
    storage.save_shadow_comparison_verdict(
        _make_verdict(
            interaction_id="cv-i-late",
            created_at=datetime.fromtimestamp(later_ts, tz=UTC),
        )
    )
    storage.save_shadow_comparison_verdict(
        _make_verdict(
            interaction_id="cv-i-early",
            created_at=datetime.fromtimestamp(earlier_ts, tz=UTC),
        )
    )
    result = storage.get_shadow_comparison_verdicts(
        from_ts=int(earlier_ts) - 10,
        to_ts=int(later_ts) + 10,
        judge_prompt_version="v1.0.0",
    )
    assert [r.interaction_id for r in result] == ["cv-i-early", "cv-i-late"]


def test_delete_by_session_returns_count(storage: BaseStorage) -> None:
    base_ts = datetime.now(UTC)
    storage.save_shadow_comparison_verdict(
        _make_verdict(interaction_id="cv-i-1", session_id="cv-s-a", created_at=base_ts)
    )
    storage.save_shadow_comparison_verdict(
        _make_verdict(interaction_id="cv-i-2", session_id="cv-s-a", created_at=base_ts)
    )
    storage.save_shadow_comparison_verdict(
        _make_verdict(interaction_id="cv-i-3", session_id="cv-s-b", created_at=base_ts)
    )
    deleted = storage.delete_shadow_comparison_verdicts_by_session("cv-s-a")
    assert deleted == 2

    remaining = storage.get_shadow_comparison_verdicts(
        from_ts=0,
        to_ts=int(base_ts.timestamp()) + 10,
        judge_prompt_version="v1.0.0",
    )
    assert {r.interaction_id for r in remaining} == {"cv-i-3"}


def test_delete_by_session_unknown_returns_zero(storage: BaseStorage) -> None:
    assert storage.delete_shadow_comparison_verdicts_by_session("never-existed") == 0
