"""SQLite CRUD for shadow_comparison_verdicts (F1)."""

from datetime import UTC, datetime

import pytest

from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(org_id="0", db_path=str(tmp_path / "shadow.db"))


def _make_verdict(**overrides) -> ShadowComparisonVerdict:
    base = {
        "verdict_id": 0,  # autoincrement
        "interaction_id": "int-1",
        "session_id": "sess-1",
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


def test_save_and_get_verdict_by_id(storage):
    v = _make_verdict()
    saved = storage.save_shadow_comparison_verdict(v)
    assert saved.verdict_id > 0
    got = storage.get_shadow_comparison_verdict(saved.verdict_id)
    assert got is not None
    assert got.interaction_id == "int-1"
    assert got.output.better_request == "1"
    assert got.judge_prompt_version == "v1.0.0"


def test_get_verdicts_in_window(storage):
    base_ts = datetime.now(UTC)
    storage.save_shadow_comparison_verdict(
        _make_verdict(interaction_id="i1", created_at=base_ts)
    )
    storage.save_shadow_comparison_verdict(
        _make_verdict(interaction_id="i2", created_at=base_ts)
    )
    verdicts = storage.get_shadow_comparison_verdicts(
        from_ts=int(base_ts.timestamp()) - 10,
        to_ts=int(base_ts.timestamp()) + 10,
        judge_prompt_version="v1.0.0",
    )
    assert {v.interaction_id for v in verdicts} == {"i1", "i2"}


def test_get_verdicts_filters_by_prompt_version(storage):
    base_ts = datetime.now(UTC)
    storage.save_shadow_comparison_verdict(
        _make_verdict(
            interaction_id="i_v1",
            created_at=base_ts,
            judge_prompt_version="v1.0.0",
        )
    )
    storage.save_shadow_comparison_verdict(
        _make_verdict(
            interaction_id="i_v2",
            created_at=base_ts,
            judge_prompt_version="v2.0.0",
        )
    )
    v1_only = storage.get_shadow_comparison_verdicts(
        from_ts=int(base_ts.timestamp()) - 10,
        to_ts=int(base_ts.timestamp()) + 10,
        judge_prompt_version="v1.0.0",
    )
    assert {v.interaction_id for v in v1_only} == {"i_v1"}


def test_delete_verdicts_by_session(storage):
    base_ts = datetime.now(UTC)
    storage.save_shadow_comparison_verdict(
        _make_verdict(interaction_id="i1", session_id="s1", created_at=base_ts)
    )
    storage.save_shadow_comparison_verdict(
        _make_verdict(interaction_id="i2", session_id="s2", created_at=base_ts)
    )
    deleted = storage.delete_shadow_comparison_verdicts_by_session(session_id="s1")
    assert deleted == 1
    remaining = storage.get_shadow_comparison_verdicts(
        from_ts=0,
        to_ts=int(base_ts.timestamp()) + 10,
        judge_prompt_version="v1.0.0",
    )
    assert {v.interaction_id for v in remaining} == {"i2"}
