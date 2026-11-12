"""Real SQLite retention authority, publication, index and rollback regressions."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.api_schema.service_schemas import Status, UserProfile
from reflexio.models.config_schema import LLMConfig
from reflexio.server.services.deferred_learning_plan import ProfileWritePlan
from reflexio.server.services.profile.learning.explicit_retention import (
    ANCESTORS_KEY,
    KEYS,
    ancestor_bindings,
    due_epoch,
    persist_with_deadlines,
    profile_version,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


RetentionStorage = SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    store = RetentionStorage(
        org_id="explicit-retention",
        db_path=str(tmp_path / "rows.db"),
        llm_config=LLMConfig(generation_model_name="openai/gpt-4o-mini"),
    )
    yield store
    store.conn.close()


def features(
    deadline=100, boundary="stop_at", *, basis="explicit_retention", key=KEYS[0]
):
    return {key: {"basis": basis, "logical_deadline": deadline, "boundary": boundary}}


def profile(identity, *, owner="owner", deadline=100, boundary="stop_at", status=None):
    return UserProfile(
        profile_id=identity,
        user_id=owner,
        content="User paints with ultramarine.",
        last_modified_timestamp=10,
        expiration_timestamp=deadline - (boundary == "stop_at"),
        custom_features=features(deadline, boundary),
        embedding=[0.0] * 512,
        generated_from_request_id="create-" + identity,
        status=status,
    )


def save(storage, *profiles):
    for item in profiles:
        storage.add_user_profile(item.user_id, [item], skip_embedding=True)


def physical_ids(storage):
    return [
        row[0]
        for row in storage.conn.execute(
            "SELECT profile_id FROM profiles ORDER BY profile_id"
        )
    ]


def hard_deletes(storage):
    return [
        event
        for event in storage.get_lineage_events(entity_type="profile")
        if event.op == "hard_delete"
    ]


@pytest.mark.parametrize("key", KEYS)
def test_exact_authority_and_boundary(key):
    assert due_epoch(features(100, "stop_at", key=key)) == 100
    assert due_epoch(features(100, "through", key=key)) == 101
    for basis in ("system_estimate", "current_reaffirmation", "unknown"):
        assert due_epoch(features(100, basis=basis, key=key)) is None


@pytest.mark.parametrize("deadline", [True, 1.2, "100", None])
def test_malformed_authority_never_coerces(deadline):
    with pytest.raises(ValueError, match="exact deadline"):
        due_epoch(features(deadline))


def test_conflicting_legacy_metadata_and_unowned_ancestry_fail():
    with pytest.raises(ValueError, match="Conflicting"):
        due_epoch({**features(100), **features(200, key=KEYS[1])})
    fresh = profile("new")
    assert fresh.custom_features is not None
    fresh.custom_features[ANCESTORS_KEY] = [{"profile_id": "old", "version": "a" * 64}]
    with pytest.raises(ValueError, match="actual retired"):
        ancestor_bindings([fresh], [], user_id="owner")
    with pytest.raises(ValueError, match="foreign owner"):
        ancestor_bindings([fresh], ["old"], user_id="foreign")
    fresh.custom_features[KEYS[0]]["basis"] = "system_estimate"
    with pytest.raises(ValueError, match="actual explicit"):
        ancestor_bindings([fresh], ["old"], user_id="owner")


def test_idle_tick_and_restart_delete_only_due_owned_explicit_copies(storage):
    save(
        storage,
        profile("a-stop"),
        profile("b-through", boundary="through"),
        profile("c-inactive", status=Status.SUPERSEDED),
        profile("d-foreign", owner="other"),
    )
    estimated = profile("e-estimated")
    estimated.custom_features = features(100, basis="system_estimate")
    save(storage, estimated)
    # A fresh storage object has no timer/job memory and reads durable authority.
    peer = RetentionStorage(
        org_id=storage.org_id,
        db_path=storage.db_path,
        llm_config=LLMConfig(generation_model_name="openai/gpt-4o-mini"),
    )
    try:
        audit = peer.purge_profile_deadlines(100, user_id="owner", batch_size=1)
        assert {item["profile_id"] for item in audit} == {"a-stop", "c-inactive"}
        assert all(item["status"] == "hard_deleted" for item in audit)
        assert physical_ids(peer) == ["b-through", "d-foreign", "e-estimated"]
        peer.purge_profile_deadlines(101, user_id="owner")
        assert physical_ids(peer) == ["d-foreign", "e-estimated"]
        assert peer.purge_profile_deadlines(200, user_id="owner") == []
        assert len(hard_deletes(peer)) == 3
        assert all(
            event.reason == "explicit_retention_deadline"
            for event in hard_deletes(peer)
        )
        indexed = [
            row[0] for row in peer.conn.execute("SELECT profile_id FROM profiles_fts")
        ]
        assert sorted(indexed) == ["d-foreign", "e-estimated"]
    finally:
        peer.conn.close()


def test_stale_job_cannot_delete_renewal_or_foreign_owner(storage):
    old = profile("renewal")
    save(storage, old)
    job = storage.list_profile_deadline_jobs(user_id="owner")[0]
    renewed = old.model_copy(deep=True)
    renewed.custom_features = features(200)
    renewed.expiration_timestamp = 199
    save(storage, renewed)
    assert storage.delete_profile_deadline(job, 100)["status"] == "stale_or_not_due"
    with pytest.raises(ValueError, match="foreign owner"):
        storage.delete_profile_deadline(job, 300, user_id="foreign")
    with pytest.raises(ValueError, match="foreign owner or project"):
        storage.delete_profile_deadline(replace(job, project_id="foreign"), 300)
    assert physical_ids(storage) == ["renewal"]
    current = storage.list_profile_deadline_jobs(user_id="owner")[0]
    assert storage.delete_profile_deadline(current, 200)["status"] == "hard_deleted"
    assert len(hard_deletes(storage)) == 1


def test_deletion_and_indexes_rollback_with_lineage_failure(storage, monkeypatch):
    save(storage, profile("rollback"))
    job = storage.list_profile_deadline_jobs()[0]

    def fail(*args, **kwargs):
        raise RuntimeError("lineage failure")

    monkeypatch.setattr(
        "reflexio.server.services.storage.sqlite_storage.profiles._explicit_retention._append_event_stmt",
        fail,
    )
    with pytest.raises(RuntimeError, match="lineage failure"):
        storage.delete_profile_deadline(job, 100)
    assert physical_ids(storage) == ["rollback"]
    assert storage.conn.execute("SELECT count(*) FROM profiles_fts").fetchone()[0] == 1
    assert hard_deletes(storage) == []


def test_ancestor_constraint_and_publication_rollback_together(storage):
    old = profile("ancestor", deadline=400)
    new = profile("successor", deadline=100)
    save(storage, old)
    assert new.custom_features is not None
    new.custom_features[ANCESTORS_KEY] = [
        {
            "profile_id": old.profile_id,
            "version": profile_version(old.model_dump(mode="json")),
        }
    ]
    plan = ProfileWritePlan("owner", "publish", [new], [old.profile_id])

    def fail_after_write():
        save(storage, new)
        raise RuntimeError("publication failure")

    with pytest.raises(RuntimeError, match="publication failure"):
        persist_with_deadlines(storage, plan, fail_after_write, 100)
    assert physical_ids(storage) == [old.profile_id]
    assert storage.list_profile_deadline_jobs()[0].due == 400
    assert hard_deletes(storage) == []
    assert storage.conn.execute("SELECT count(*) FROM profiles_fts").fetchone()[0] == 1


def test_cached_plan_overdue_publish_has_no_primary_or_search_resurrection(storage):
    old = profile("ancestor", deadline=100)
    fresh = profile("successor", deadline=200)
    residual = profile("residual", deadline=400)
    residual.custom_features = features(400, basis="system_estimate")
    save(storage, old, residual)
    assert fresh.custom_features is not None
    fresh.custom_features[ANCESTORS_KEY] = [
        {
            "profile_id": old.profile_id,
            "version": profile_version(old.model_dump(mode="json")),
        }
    ]
    # Reconstruct from a serialized native write plan: no live compiler state.
    plan = ProfileWritePlan("owner", "cached", [fresh], [old.profile_id])
    encoded = json.loads(
        json.dumps(asdict(plan), default=lambda value: value.model_dump(mode="json"))
    )
    restored = ProfileWritePlan(
        **{
            **encoded,
            "new_profiles": [
                UserProfile.model_validate(row) for row in encoded["new_profiles"]
            ],
        }
    )
    result, audit = persist_with_deadlines(
        storage, restored, lambda: save(storage, *restored.new_profiles), 210
    )
    assert result is None
    assert any(item["status"] == "earlier_constraint_kept" for item in audit)
    assert physical_ids(storage) == ["residual"]
    assert [
        row[0] for row in storage.conn.execute("SELECT profile_id FROM profiles_fts")
    ] == ["residual"]
    assert {event.entity_id for event in hard_deletes(storage)} == {
        "ancestor",
        "successor",
    }
    if storage._has_sqlite_vec:
        assert (
            storage.conn.execute("SELECT count(*) FROM profiles_vec").fetchone()[0] == 1
        )


def test_changed_ancestor_refuses_whole_publication_and_independent_rows_survive(
    storage,
):
    old = profile("old", deadline=400)
    new = profile("new", deadline=100)
    save(storage, old)
    assert new.custom_features is not None
    new.custom_features[ANCESTORS_KEY] = [
        {
            "profile_id": old.profile_id,
            "version": profile_version(old.model_dump(mode="json")),
        }
    ]
    changed = old.model_copy(update={"content": "Changed after compilation"})
    save(storage, changed)
    with pytest.raises(ValueError, match="changed since"):
        persist_with_deadlines(
            storage,
            ProfileWritePlan("owner", "publish", [new], [old.profile_id]),
            lambda: save(storage, new),
            100,
        )
    assert physical_ids(storage) == ["old"]
    assert hard_deletes(storage) == []


def test_paging_ignores_estimated_rows_and_finds_late_due_jobs(storage, monkeypatch):
    for index in range(5):
        item = profile(f"{index}-estimated")
        item.custom_features = features(100, basis="system_estimate")
        save(storage, item)
    save(storage, profile("z-due"))
    original = storage._profile_deadline_rows

    def short_page(**kwargs):
        return original(**{**kwargs, "limit": 1})

    monkeypatch.setattr(storage, "_profile_deadline_rows", short_page)
    assert [job.profile_id for job in storage.list_profile_deadline_jobs(limit=2)] == [
        "z-due"
    ]
    assert len(storage.purge_profile_deadlines(100, batch_size=2)) == 1
    assert "z-due" not in physical_ids(storage)


def test_next_deadline_uses_indexed_minimum_and_preserves_through(storage, monkeypatch):
    save(
        storage,
        profile("a-later", deadline=400),
        profile("z-earliest", deadline=100, boundary="through"),
    )
    save(storage, profile("foreign-earlier", owner="other", deadline=1))
    original = storage._profile_deadline_rows
    seen = []

    def short_page(**kwargs):
        rows = original(**{**kwargs, "limit": 1})
        seen.extend(rows)
        return rows

    monkeypatch.setattr(storage, "_profile_deadline_rows", short_page)
    assert storage.next_profile_deadline(user_id="owner") == 101
    assert len(seen) == 1 and seen[0]["profile_id"] == "z-earliest"
    assert all("content" not in row for row in seen)
    assert storage.next_profile_deadline() == 1


def test_clock_and_budget_validation_do_not_start_writes(storage):
    with pytest.raises(ValueError, match="whole seconds"):
        persist_with_deadlines(storage, None, lambda: None, True)
    with pytest.raises(ValueError, match="between 1 and 1000"):
        storage.list_profile_deadline_jobs(limit=0)
    with pytest.raises(ValueError, match="positive"):
        storage.purge_profile_deadlines(100, limit=0)
    # A null native write plan touches no profiles; catch-up stays background-owned.
    save(storage, profile("due"))
    result, audit = persist_with_deadlines(storage, None, lambda: "empty", 100)
    assert result == "empty" and audit == []
    assert physical_ids(storage) == ["due"]
    assert len(storage.purge_profile_deadlines(100, user_id="owner")) == 1
    assert physical_ids(storage) == []


@pytest.mark.parametrize("outer_rollback", [False, True])
def test_native_final_resolved_write_plan_owns_deadline_publication(
    storage, monkeypatch, outer_rollback
):
    from reflexio.server.services.profile import service as service_module
    from reflexio.server.services.profile.service import ProfileGenerationService

    old = profile("old-copy", deadline=400)
    fresh = profile("new-copy", deadline=100)
    assert fresh.custom_features is not None
    fresh.custom_features[ANCESTORS_KEY] = [
        {
            "profile_id": old.profile_id,
            "version": profile_version(old.model_dump(mode="json")),
        }
    ]
    save(storage, old)
    # The actual native service seam also receives cached/restored plans and
    # plans materialized by inline finalization, after the outer owner fences.
    service = object.__new__(ProfileGenerationService)
    service.storage = storage
    plan = ProfileWritePlan(
        "owner",
        "publication",
        [fresh],
        [old.profile_id],
        lineage_contexts=[
            LineageContext(
                op_kind="create",
                actor="consolidator",
                source_ids=[old.profile_id],
                request_id="publication",
            )
        ],
    )
    monkeypatch.setattr(service_module.time, "time", lambda: 100)
    if outer_rollback:
        with (
            pytest.raises(RuntimeError, match="outer fence rollback"),
            storage.commit_scope(),
        ):
            service._persist_write_plan(plan)
            raise RuntimeError("outer fence rollback")
        assert physical_ids(storage) == [old.profile_id]
        assert storage.list_profile_deadline_jobs()[0].due == 400
        assert hard_deletes(storage) == []
    else:
        service._persist_write_plan(plan)
        assert physical_ids(storage) == []
        assert {event.entity_id for event in hard_deletes(storage)} == {
            old.profile_id,
            fresh.profile_id,
        }


@pytest.mark.parametrize("overdue", [False, True])
def test_publication_enforces_only_touched_ids_without_discovery(
    storage, monkeypatch, overdue
):
    unrelated = profile("unrelated")
    foreign = profile("foreign", owner="other")
    fresh = profile("fresh")
    if not overdue:
        fresh.custom_features = None
        fresh.expiration_timestamp = 999
    save(storage, unrelated, foreign)

    def unavailable(**kwargs):
        raise AssertionError("Publication must not use indexed discovery")

    monkeypatch.setattr(storage, "list_profile_deadline_jobs", unavailable)
    result, audit = persist_with_deadlines(
        storage,
        ProfileWritePlan("owner", "publish", [fresh], []),
        lambda: (save(storage, fresh), "published")[1],
        100,
    )
    assert result == "published"
    assert physical_ids(storage) == (
        ["foreign", "unrelated"] if overdue else ["foreign", "fresh", "unrelated"]
    )
    assert {event.entity_id for event in hard_deletes(storage)} == (
        {"fresh"} if overdue else set()
    )
    assert len(audit) == int(overdue)
    # Foreign identities cannot grant deletion, even if named directly.
    assert (
        storage.purge_profile_deadlines(
            100, user_id="owner", profile_ids=[foreign.profile_id]
        )
        == []
    )


def test_empty_publication_touched_set_performs_no_deadline_reads(storage, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("Empty touched set must not query")

    monkeypatch.setattr(storage, "_profile_deadline_rows", forbidden)
    assert storage.purge_profile_deadlines(100, profile_ids=[]) == []
    assert persist_with_deadlines(storage, None, lambda: "empty", 100) == ("empty", [])
    assert persist_with_deadlines(
        storage, ProfileWritePlan("owner", "empty", [], []), lambda: "empty", 100
    ) == ("empty", [])


def test_touched_authority_failure_rolls_back_prior_deletion_and_publication(storage):
    due, invalid, fresh = profile("a-due"), profile("z-invalid"), profile("fresh")
    invalid.custom_features = features(True)
    save(storage, due, invalid)
    with pytest.raises(ValueError, match="exact deadline"):
        persist_with_deadlines(
            storage,
            ProfileWritePlan(
                "owner", "publish", [fresh], [due.profile_id, invalid.profile_id]
            ),
            lambda: save(storage, fresh),
            100,
        )
    assert physical_ids(storage) == ["a-due", "z-invalid"]
    assert hard_deletes(storage) == []
    assert sorted(
        row[0] for row in storage.conn.execute("SELECT profile_id FROM profiles_fts")
    ) == ["a-due", "z-invalid"]
