# ruff: noqa: S608 -- Test SQL identifiers are fixed profile/playbook field registries.
"""Ordinary historical inputs through actual consolidation and publication."""

from types import SimpleNamespace

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook, UserProfile
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.server.services.profile.components.consolidator import (
    ProfileConsolidator,
    ProfileDeduplicationOutput,
    ProfileDuplicateGroup,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.extraction import test_resume_publication_integration as base

storage = base.storage
request_context = base.request_context
explicit_test_billing_exemption = base.explicit_test_billing_exemption
no_background_tagging = base.no_background_tagging
publication = base.publication
shared_worker = base.shared_worker
pytestmark = pytest.mark.integration


def _seed_history(storage, kind):
    if kind == "profile":
        row = UserProfile(
            profile_id="historical-profile",
            user_id="user_1",
            content="Historical approval policy.",
            last_modified_timestamp=1700000000,
            generated_from_request_id="request_1",
        )
        storage.add_user_profile("user_1", [row], skip_embedding=True)
        return storage.get_user_profile("user_1")[0]
    row = UserPlaybook(
        user_id="user_1",
        agent_version="v1",
        request_id="request_1",
        content="Historical approval policy.",
        trigger="before deployment",
        rationale="Earlier agreement",
    )
    storage.save_user_playbooks([row])
    return storage.get_user_playbooks(user_id="user_1", agent_version="v1")[0]


@pytest.mark.parametrize("change", [None, "content", "delete", "status", "owner"])
def test_ordinary_publication_checks_actual_historical_decision_inputs(
    shared_worker: DurableLearningWorker, publication, monkeypatch, change
):
    storage, worker, kind, dispatches, events = publication
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    original = _seed_history(storage, kind)
    observed = []
    historical_id = (
        original.profile_id if kind == "profile" else original.user_playbook_id
    )
    table, key = (
        ("profiles", "profile_id")
        if kind == "profile"
        else ("user_playbooks", "user_playbook_id")
    )

    def decide(existing):
        assert (
            len(existing) == 1 and existing[0].content == "Historical approval policy."
        )
        observed.append(existing[0].content)
        if change == "content":
            storage._execute(
                f"UPDATE {table} SET content=? WHERE {key}=?",
                ("Changed approval policy.", historical_id),
            )
        elif change == "delete":
            storage._execute(f"DELETE FROM {table} WHERE {key}=?", (historical_id,))
        elif change == "status":
            storage._execute(
                f"UPDATE {table} SET status='pending' WHERE {key}=?", (historical_id,)
            )
        elif change == "owner":
            storage._execute(
                f"UPDATE {table} SET user_id='other-user' WHERE {key}=?",
                (historical_id,),
            )

    if kind == "profile":
        monkeypatch.setattr(
            "reflexio.server.services.profile.components.consolidator.ProfileConsolidator",
            ProfileConsolidator,
        )
        # Search selection is deterministic; the rows are read from the actual
        # store and pass through the real prompt producer and decision path.
        monkeypatch.setattr(
            ProfileConsolidator,
            "_retrieve_existing_profiles",
            lambda *_a, **_kw: storage.get_user_profile("user_1"),
        )

        def profile_model(_self, **kwargs):
            assert "Historical approval policy." in str(kwargs["messages"])
            decide(storage.get_user_profile("user_1"))
            value = ProfileDeduplicationOutput(
                duplicate_groups=[], unique_ids=["NEW-0"], deletions=[]
            )
            assert not kwargs["validator"](value)
            return SimpleNamespace(value=value, provenance=None)

        monkeypatch.setattr(ProfileConsolidator, "_call_native_dedup", profile_model)
    else:
        monkeypatch.setattr(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator",
            PlaybookConsolidator,
        )
        monkeypatch.setattr(
            PlaybookConsolidator,
            "retrieve_existing_playbooks",
            lambda *_a, **_kw: storage.get_user_playbooks(
                user_id="user_1", agent_version="v1"
            ),
        )

        def playbook_model(_consolidator, new_rows, existing_rows, **_kwargs):
            decide(existing_rows)
            return new_rows, [], []

        monkeypatch.setattr(
            "reflexio.server.services.playbook.components.reconciliation.reconcile_playbooks",
            playbook_model,
        )

    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert observed == ["Historical approval policy."], (
        "The real producer must supply the historical row to its decision"
    )
    assert result is not None and result.committed_output is not None
    if change is None:
        assert result.status == AgentRunStatus.FINALIZED
        assert len(base._learnings(storage, kind)) == 2
        assert dispatches
    else:
        assert result.status == AgentRunStatus.FINALIZATION_FAILED
        rows = storage.conn.execute(
            f"SELECT {key} FROM {table} WHERE {key}<>?", (historical_id,)
        ).fetchall()
        assert rows == [], "Stale historical inputs may not produce new learning"
        assert not dispatches
        assert not [
            event for event in events if event.event_name == "learnings_generated"
        ]
        assert all(
            d.consumed_at is None for d in storage.list_run_tool_dependencies(result.id)
        )
        with storage._stream_sql(read_only=True) as db:
            assert (
                storage._get_output_publication_receipt(
                    db, result.id, result.committed_output["publication"]["id"]
                )
                is None
            )


@pytest.mark.parametrize("publication", ["profile"], indirect=True)
@pytest.mark.parametrize("merge_history", [False, True])
@pytest.mark.parametrize("expire_history", [False, True])
def test_historical_profile_lifetime_after_own_writes_and_final_status(
    shared_worker: DurableLearningWorker,
    publication,
    monkeypatch,
    merge_history,
    expire_history,
):
    from datetime import UTC, datetime

    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    storage, worker, kind, dispatches, events = publication
    assert kind == "profile"
    old = _seed_history(storage, kind)
    deadline = int(datetime.now(UTC).timestamp()) + 10
    storage._execute(
        "UPDATE profiles SET expiration_timestamp=? WHERE profile_id=?",
        (deadline, old.profile_id),
    )
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setattr(
        "reflexio.server.services.profile.components.consolidator.ProfileConsolidator",
        ProfileConsolidator,
    )
    monkeypatch.setattr(
        ProfileConsolidator,
        "_retrieve_existing_profiles",
        lambda *_a, **_kw: storage.get_user_profile("user_1"),
    )
    supplied = []

    def model(_self, **kwargs):
        assert "Historical approval policy." in str(kwargs["messages"])
        supplied.append(1)
        return SimpleNamespace(
            value=ProfileDeduplicationOutput(
                duplicate_groups=[
                    ProfileDuplicateGroup(
                        item_ids=["NEW-0", "EXISTING-0"],
                        merged_content="Merged approval policy.",
                    )
                ]
                if merge_history
                else [],
                unique_ids=[] if merge_history else ["NEW-0"],
                deletions=[],
            ),
            provenance=None,
        )

    monkeypatch.setattr(ProfileConsolidator, "_call_native_dedup", model)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
        lambda **kw: base._loop_result(kw, kind, storage, nonempty=True),
    )
    write = storage.update_agent_run_status
    boundaries = []

    def expire_after_status(run_id, status, **kwargs):
        result = write(run_id, status, **kwargs)
        if status == AgentRunStatus.FINALIZED:
            boundaries.append(status)
            if expire_history:
                monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _db: deadline + 1)
        return result

    monkeypatch.setattr(storage, "update_agent_run_status", expire_after_status)
    assert shared_worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300) == 1
    result = storage.get_agent_run("run_1")
    assert supplied == [1] and boundaries == [AgentRunStatus.FINALIZED]
    assert result is not None and result.committed_output is not None
    if not expire_history:
        assert result.status == AgentRunStatus.FINALIZED
        rows = storage.get_user_profile("user_1")
        assert len(rows) == (1 if merge_history else 2)
        if merge_history:
            assert rows[0].content == "Merged approval policy."
        old_status = storage.conn.execute(
            "SELECT status FROM profiles WHERE profile_id=?", (old.profile_id,)
        ).fetchone()[0]
        assert old_status == ("superseded" if merge_history else None)
        assert dispatches
        assert (
            len(
                [event for event in events if event.event_name == "learnings_generated"]
            )
            == 1
        )
        assert all(
            d.consumed_at is not None
            for d in storage.list_run_tool_dependencies(result.id)
        )
        return
    assert result.status == AgentRunStatus.FINALIZATION_FAILED
    assert (
        storage.conn.execute("SELECT profile_id FROM profiles").fetchall()[0][0]
        == old.profile_id
    )
    assert storage.conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 1
    assert storage.conn.execute(
        "SELECT status,retired_at FROM profiles WHERE profile_id=?", (old.profile_id,)
    ).fetchone()[:] == (None, None)
    assert not dispatches and not [
        event for event in events if event.event_name == "learnings_generated"
    ]
    assert all(
        d.consumed_at is None for d in storage.list_run_tool_dependencies(result.id)
    )
    with storage._stream_sql(read_only=True) as db:
        assert (
            storage._get_output_publication_receipt(
                db, result.id, result.committed_output["publication"]["id"]
            )
            is None
        )


@pytest.mark.parametrize("publication", ["profile"], indirect=True)
@pytest.mark.parametrize("history_kind,count", [("profile", 1001), ("playbook", 129)])
def test_ordinary_history_proof_pages_without_a_new_input_cap(
    publication, history_kind, count
):
    import json
    import sqlite3
    from datetime import UTC, datetime

    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        user_extraction_lease,
    )
    from reflexio.server.services.extraction.profile_package_state import (
        PROFILE_WITNESS_FIELDS,
        profile_witness,
    )
    from reflexio.server.services.playbook.components.overflow_evidence import (
        HISTORICAL_PLAYBOOK_FIELDS,
        historical_playbook_digest,
    )
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )
    from reflexio.server.services.storage.storage_base import (
        AgentBinding,
        AgentRunRecord,
    )
    from tests.server.services.extraction._publication_fixtures import (
        write_completed_output,
    )

    storage, _worker, _kind, _, _ = publication
    run = write_completed_output(
        storage,
        AgentRunRecord(
            id="historical-page-probe",
            binding=AgentBinding(
                org_id=storage.org_id,
                extractor_kind="profile",
                user_id="user_1",
                request_id="request_1",
                agent_version="v1",
                source="api",
                source_interaction_ids=[1, 2, 3],
            ),
            status=AgentRunStatus.RUNNING,
            generation_request_snapshot={
                "output_schema_name": "StructuredProfilesOutput"
            },
        ),
        StructuredProfilesOutput(profiles=[]),
    )
    if history_kind == "profile":
        table, key, fields = "profiles", "profile_id", PROFILE_WITNESS_FIELDS
        rows = [
            UserProfile(
                profile_id=f"history-{index:04d}",
                user_id="user_1",
                content=f"Historical input {index}",
                last_modified_timestamp=1700000000,
                generated_from_request_id="request_1",
            )
            for index in range(count)
        ]
    else:
        table, key, fields = (
            "user_playbooks",
            "user_playbook_id",
            HISTORICAL_PLAYBOOK_FIELDS,
        )
        rows = [
            UserPlaybook(
                user_playbook_id=index + 1,
                user_id="user_1",
                agent_version="v1",
                request_id="request_1",
                created_at=1700000000,
                content=f"Historical input {index}",
                trigger="before deployment",
                rationale="Earlier agreement",
            )
            for index in range(count)
        ]
    values = []
    witnesses = {}
    for row in rows:
        payload = row.model_dump(mode="json")
        witnesses[str(payload[key])] = (
            profile_witness(row)
            if isinstance(row, UserProfile)
            else historical_playbook_digest(row)
        )
        if isinstance(row, UserPlaybook):
            payload["created_at"] = datetime.fromtimestamp(
                row.created_at, UTC
            ).isoformat()
        for name in (
            "custom_features",
            "extractor_names",
            "source_interaction_ids",
            "blocking_issue",
            "tags",
        ):
            if name in fields and payload[name] is not None:
                payload[name] = json.dumps(payload[name])
        values.append(tuple(payload[name] for name in fields))
    with storage.commit_scope():
        storage.conn.executemany(
            f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
            values,
        )
    previous_limit = storage.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        with user_extraction_lease(storage, "user_1", max_wait_seconds=0):
            retained = adopt_agent_run_for_execution(storage, run)
            profiles, playbooks = (
                (witnesses, {}) if history_kind == "profile" else ({}, witnesses)
            )
            deadline = storage.validate_agent_history_inputs(
                retained, profiles, playbooks
            )
            first, last = rows[0], rows[-1]
            assert deadline == (
                first.expiration_timestamp if isinstance(first, UserProfile) else None
            )
            last_id = (
                last.profile_id
                if isinstance(last, UserProfile)
                else last.user_playbook_id
            )
            with storage.commit_scope():
                storage.conn.execute(
                    f"UPDATE {table} SET content='Changed final historical row' WHERE {key}=?",
                    (last_id,),
                )
            with pytest.raises(InputBudgetConfigurationError, match="Historical"):
                storage.validate_agent_history_inputs(retained, profiles, playbooks)
            assert storage.get_agent_run(run.id) == retained
    finally:
        storage.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)
