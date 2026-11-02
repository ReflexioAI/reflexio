"""Historical digests are scoped immutable inputs, never execution authority."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace

import pytest

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.extraction import history_inputs as history
from reflexio.server.services.extraction.profile_package_state import profile_witness
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
)
from tests.server.services.extraction import test_resume_worker as base

storage = base.storage


def _run(storage):
    return AgentRunRecord(
        id="captured",
        binding=AgentBinding(
            org_id=storage.org_id,
            extractor_kind="profile",
            user_id="user_1",
            request_id="request",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.FINALIZING,
        generation_request_snapshot={},
    )


def _profile():
    return UserProfile(
        profile_id="old-profile",
        user_id="user_1",
        content="Original historical input",
        last_modified_timestamp=1700000000,
        generated_from_request_id="old-request",
    )


def test_history_finish_seals_and_detaches_without_overwriting_changed_input(storage):
    original = _profile()
    changed = original.model_copy(update={"content": "Changed input"})
    with history.historical_input_scope(storage, _run(storage)) as capture:
        history.record_historical_profiles(storage, [original])
        with pytest.raises(InputBudgetConfigurationError, match="changed during"):
            history.record_historical_profiles(storage, [changed])
        copied = copy_context()
        inputs = capture.finish()
        assert inputs.profiles == {original.profile_id: profile_witness(original)}
        inputs.profiles.clear()
        with pytest.raises(InputBudgetConfigurationError):
            copied.run(history.record_historical_profiles, storage, [original])
        with pytest.raises(InputBudgetConfigurationError, match="ended"):
            capture.finish()
    with pytest.raises(InputBudgetConfigurationError):
        copied.run(history.record_historical_profiles, storage, [original])


def test_history_scope_rejects_other_storage_user_project_and_nested_preparation(
    storage, tmp_path, monkeypatch
):
    other = SQLiteStorage(org_id=storage.org_id, db_path=str(tmp_path / "other.db"))
    original = _profile()
    with history.historical_input_scope(storage, _run(storage)) as capture:
        with pytest.raises(InputBudgetConfigurationError, match="scope changed"):
            history.record_historical_profiles(other, [original])
        with pytest.raises(InputBudgetConfigurationError, match="scope changed"):
            history.record_historical_profiles(
                storage, [original.model_copy(update={"user_id": "other-user"})]
            )
        monkeypatch.setattr(history, "current_project_id", lambda: "foreign-project")
        with pytest.raises(InputBudgetConfigurationError, match="scope changed"):
            history.record_historical_profiles(storage, [original])
        with (
            pytest.raises(InputBudgetConfigurationError, match="already active"),
            history.historical_input_scope(
                storage, replace(_run(storage), id="nested")
            ),
        ):
            pass
        assert capture.finish().profiles == {}
    other.conn.close()


def test_concurrent_history_cannot_overwrite_a_conflicting_input(storage):
    original = _profile()
    changed = original.model_copy(update={"content": "Concurrent input"})

    def record(row):
        try:
            history.record_historical_profiles(storage, [row])
            return row
        except InputBudgetConfigurationError:
            return None

    with history.historical_input_scope(storage, _run(storage)) as capture:
        first, second = copy_context(), copy_context()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(first.run, record, original),
                executor.submit(second.run, record, changed),
            ]
            results = [future.result(timeout=5) for future in futures]
        winners = [row for row in results if row is not None]
        assert len(winners) == 1
        assert capture.finish().profiles == {
            original.profile_id: profile_witness(winners[0])
        }


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"publication": {}},
        {
            "publication": {
                "history_inputs": {"profiles": {}, "playbooks": {}, "extra": {}}
            }
        },
        {
            "publication": {
                "history_inputs": {"profiles": {"p": "invalid"}, "playbooks": {}}
            }
        },
        {
            "publication": {
                "history_inputs": {"profiles": {}, "playbooks": {"01": "a" * 64}}
            }
        },
    ],
)
def test_saved_history_requires_explicit_valid_original_maps(payload):
    with pytest.raises(InputBudgetConfigurationError):
        history.history_from_output(payload)


def test_stage_history_union_preserves_original_inputs_and_refuses_conflict():
    original = history.HistoricalInputs(profiles={"p": "a" * 64})
    prepared = history.HistoricalInputs(
        profiles={"p": "a" * 64, "other": "b" * 64}, playbooks={"12": "c" * 64}
    )
    merged = history.merge_historical_inputs(original, prepared)
    prepared.profiles.clear()
    original.profiles.clear()
    assert merged.profiles == {"p": "a" * 64, "other": "b" * 64}
    assert merged.playbooks == {"12": "c" * 64}
    with pytest.raises(InputBudgetConfigurationError, match="changed between"):
        history.merge_historical_inputs(
            merged, history.HistoricalInputs(profiles={"p": "d" * 64})
        )
