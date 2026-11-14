"""Real SQLite publication: permission, payload CAS and the outer commit fence."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from reflexio.models.api_schema.domain.entities import LineageContext, UserProfile
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.deferred_learning_plan import (
    ProfilePayloadErasure,
    ProfileWritePlan,
)
from reflexio.server.services.profile.lifecycle import (
    KEY,
    ExplicitStorageDeadline,
    attention_epoch,
    expiration_for,
    merge_features,
    metadata,
    new_features,
)
from reflexio.server.services.profile.lifecycle_deadlines import (
    NativeLifecycleDeadlineStore,
    job_for,
    profile_version,
    publish_with_lifecycle,
)
from reflexio.server.services.write_plan_codec import (
    decode_write_plan,
    encode_write_plan,
)

pytestmark = pytest.mark.integration
NOW = 2_000_000_000
DEADLINE = ExplicitStorageDeadline(
    deadline_utc="2033-05-18T03:33:20Z", source_quote="Forget at that instant"
)
assert DEADLINE.epoch == NOW


def _profile(identity="old", *, user="owner", deadline=None, legacy=False, **kwargs):
    features = (
        None
        if legacy
        else new_features(
            None,
            now=NOW - 100,
            deadline=deadline,
            source_texts=[DEADLINE.source_quote],
        )
    )
    return UserProfile(
        profile_id=identity,
        user_id=user,
        content="Stored source fact",
        last_modified_timestamp=NOW - 100,
        generated_from_request_id="prior",
        custom_features=features,
        expiration_timestamp=expiration_for(features) if features else NOW + 100,
        source=None,
        embedding=[0.0] * 512,
        source_span="Source fact",
        **kwargs,
    )


def _save(storage, *profiles):
    storage.add_user_profile("owner", list(profiles), skip_embedding=True)


def _instruction(profile):
    return ProfilePayloadErasure(
        profile.profile_id,
        profile.user_id,
        "forget-request",
        profile_version(profile.model_dump(mode="json")),
        "new-forget-source",
        "Forget my stored fact",
    )


def _plan(old, new=()):
    return ProfileWritePlan(
        "owner",
        "forget-request",
        list(new),
        [old.profile_id],
        erasure_instructions=[_instruction(old)],
        lifecycle_epoch=NOW,
    )


def _persist(storage, plan):
    if plan.new_profiles:
        _save(storage, *plan.new_profiles)
    if plan.superseded_ids:
        storage.supersede_profiles_by_ids(
            user_id=plan.user_id,
            profile_ids=plan.superseded_ids,
            request_id=plan.request_id,
        )


@pytest.mark.parametrize("fail", [False, True])
def test_exact_forget_and_replacement_share_outer_transaction(storage, fail):
    old, new = _profile(), _profile("new")
    _save(storage, old)
    native = storage.get_profile_by_id("old")
    plan = _plan(native, [new])
    store = NativeLifecycleDeadlineStore(storage, "owner")

    def publish():
        with storage.commit_scope():
            publish_with_lifecycle(store, plan, lambda p: _persist(storage, p), NOW)
            if fail:
                raise RuntimeError("Lost publication fence")

    if fail:
        with pytest.raises(RuntimeError, match="Lost publication fence"):
            publish()
    else:
        publish()
    assert (storage.get_profile_by_id("old") is not None) == fail
    assert (storage.get_profile_by_id("new") is not None) != fail
    events = storage.get_lineage_events(entity_type="profile", entity_id="old")
    erased = [event for event in events if event.op == "hard_delete"]
    assert len(erased) == (0 if fail else 1)
    if erased:
        assert erased[0].reason == "source_authorized_explicit_forget"
    count = storage.conn.execute(
        "SELECT count(*) FROM profiles_fts WHERE profile_id='old'"
    ).fetchone()[0]
    assert count == (1 if fail else 0)
    if storage._has_sqlite_vec:
        vector_count = storage.conn.execute(
            "SELECT count(*) FROM profiles_vec"
        ).fetchone()[0]
        assert vector_count == 1


@pytest.mark.parametrize("execution_epoch", [NOW - 1, NOW])
def test_exact_forget_precedes_its_own_due_deadline(storage, execution_epoch):
    old = _profile(deadline=DEADLINE)
    history = _profile("deadline-history", deadline=DEADLINE, status=Status.SUPERSEDED)
    _save(storage, old, history)
    plan = replace(
        _plan(storage.get_profile_by_id("old"), [_profile("replacement")]),
        lifecycle_epoch=NOW - 10,
    )
    _, audit = publish_with_lifecycle(
        NativeLifecycleDeadlineStore(storage, "owner"),
        plan,
        lambda projected: _persist(storage, projected),
        execution_epoch,
    )
    assert storage.get_profile_by_id("old") is None
    assert storage.get_profile_by_id("replacement") is not None
    assert (
        storage.get_profile_by_id("deadline-history", include_tombstones=True)
        is not None
    ) == (execution_epoch < NOW)
    assert [entry["profile_id"] for entry in audit].count("old") == 1
    assert audit[0]["status"] == "erasure_staged"
    erased = [
        event
        for event in storage.get_lineage_events(entity_id="old")
        if event.op == "hard_delete"
    ]
    assert len(erased) == 1
    assert erased[0].reason == "source_authorized_explicit_forget"


@pytest.mark.parametrize("execution_epoch", [NOW - 1, NOW])
def test_due_exact_forget_still_refuses_stale_payload_and_rolls_back(
    storage, execution_epoch
):
    _save(
        storage,
        _profile(deadline=DEADLINE),
        _profile("first-target", deadline=DEADLINE),
        _profile("deadline-history", deadline=DEADLINE),
    )
    plan = _plan(storage.get_profile_by_id("old"), [_profile("replacement")])
    plan.superseded_ids.append("first-target")
    plan.erasure_instructions.append(
        _instruction(storage.get_profile_by_id("first-target"))
    )
    with storage.commit_scope():
        storage.conn.execute(
            "UPDATE profiles SET notes='Changed at the same timestamp' WHERE profile_id='old'"
        )
    with pytest.raises(ValueError, match="changed since"):
        publish_with_lifecycle(
            NativeLifecycleDeadlineStore(storage, "owner"),
            plan,
            lambda projected: _persist(storage, projected),
            execution_epoch,
        )
    assert storage.get_profile_by_id("old").notes == "Changed at the same timestamp"
    assert storage.get_profile_by_id("deadline-history") is not None
    assert storage.get_profile_by_id("first-target") is not None
    assert storage.get_profile_by_id("replacement") is None
    assert not any(event.op == "hard_delete" for event in storage.get_lineage_events())


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "Changed fact"),
        ("notes", "New provenance"),
        ("tags", '["changed"]'),
    ],
)
def test_same_timestamp_payload_change_refuses_erasure(storage, field, value):
    _save(storage, _profile())
    plan = _plan(storage.get_profile_by_id("old"), [_profile("new")])
    # Fixed test columns; leave modification timestamp unchanged deliberately.
    with storage.commit_scope():
        storage.conn.execute(
            f"UPDATE profiles SET {field}=? WHERE profile_id='old'",  # noqa: S608
            (value,),
        )
    with pytest.raises(ValueError, match="changed since"):
        publish_with_lifecycle(
            NativeLifecycleDeadlineStore(storage, "owner"),
            plan,
            lambda p: _persist(storage, p),
            NOW,
        )
    assert storage.get_profile_by_id("old") is not None
    assert storage.get_profile_by_id("new") is None


def test_locked_null_source_matches_native_empty_source(storage):
    _save(storage, _profile())
    assert storage.conn.execute("SELECT source FROM profiles").fetchone()[0] is None
    native = storage.get_profile_by_id("old")
    assert native.source == ""
    store = NativeLifecycleDeadlineStore(storage, "owner")
    publish_with_lifecycle(store, _plan(native), lambda p: _persist(storage, p), NOW)
    assert storage.get_profile_by_id("old") is None


def test_postpublication_due_erasure_does_not_flush_erased_index_payload(storage):
    due = _profile("new-already-due", deadline=DEADLINE)
    plan = ProfileWritePlan("owner", "new-request", [due], [], lifecycle_epoch=NOW)
    publish_with_lifecycle(
        NativeLifecycleDeadlineStore(storage, "owner"),
        plan,
        lambda p: _persist(storage, p),
        NOW,
    )
    assert storage.get_profile_by_id(due.profile_id) is None
    assert storage.conn.execute("SELECT count(*) FROM profiles_fts").fetchone()[0] == 0
    if storage._has_sqlite_vec:
        assert (
            storage.conn.execute("SELECT count(*) FROM profiles_vec").fetchone()[0] == 0
        )


def test_ordinary_correction_remains_soft_and_legacy_has_no_deadline(storage):
    old = _profile(legacy=True)
    _save(storage, old)
    plan = replace(_plan(old, [_profile("new")]), erasure_instructions=[])
    publish_with_lifecycle(
        NativeLifecycleDeadlineStore(storage, "owner"),
        plan,
        lambda p: _persist(storage, p),
        NOW,
    )
    historical = storage.get_profile_by_id("old", include_tombstones=True)
    assert historical.status is Status.SUPERSEDED
    assert historical.expiration_timestamp == old.expiration_timestamp
    assert job_for(historical.model_dump()) is None
    assert not any(event.op == "hard_delete" for event in storage.get_lineage_events())


def test_due_census_pages_all_statuses_and_does_not_use_legacy_expiry(storage):
    rows = [
        _profile(f"due-{i:04}", deadline=DEADLINE, status=Status.SUPERSEDED)
        for i in range(258)
    ]
    ordinary = _profile("legacy", legacy=True)
    ordinary.expiration_timestamp = NOW - 1
    foreign = _profile("foreign", user="other", deadline=DEADLINE)
    _save(storage, *rows, ordinary, foreign)
    store = NativeLifecycleDeadlineStore(storage, "owner")
    assert store.purge_due(NOW - 1) == []
    assert len(store.purge_due(NOW)) == 258
    assert storage.get_profile_by_id("legacy") is not None
    assert storage.get_profile_by_id("foreign") is not None
    assert store.purge_due(NOW) == []


def test_storage_owner_rejects_foreign_payload_and_unlocked_delete(storage):
    _save(storage, _profile())
    with pytest.raises(ValueError, match="publication scope"):
        storage.delete_profile_lifecycle_payload("owner", "", "old", reason="erasure")
    with storage.commit_scope():
        assert (
            storage.delete_profile_lifecycle_payload(
                "foreign", "", "old", reason="erasure"
            )
            == 0
        )
        with pytest.raises(ValueError, match="no project"):
            storage.profile_lifecycle_rows("owner", "foreign-project")
    assert storage.get_profile_by_id("old") is not None


def test_legacy_attention_and_mixed_merge_preserve_unknown_origin_and_bound():
    legacy, qualified = _profile(legacy=True), _profile("qualified")
    legacy.expiration_timestamp = NOW + 5
    assert attention_epoch(legacy, now=NOW) == NOW
    mixed = merge_features(
        [qualified, legacy], inherited_features=None, existing_profiles=[legacy]
    )
    merged = qualified.model_copy(update={"custom_features": mixed})
    assert metadata(merged)["created_at"] is None
    assert attention_epoch(merged, now=NOW + 1) == NOW + 1
    assert expiration_for(mixed) == NOW + 5
    assert job_for(merged.model_dump()) is None
    again = merge_features(
        [merged, _profile("later")], inherited_features=None, existing_profiles=[merged]
    )
    assert again[KEY]["created_at"] is None
    assert expiration_for(again) == NOW + 5


def test_mixed_legacy_bound_cannot_override_earlier_explicit_permission():
    legacy, qualified = _profile(legacy=True), _profile("qualified", deadline=DEADLINE)
    features = merge_features(
        [legacy, qualified], inherited_features=None, existing_profiles=[legacy]
    )
    assert expiration_for(features) == NOW - 1
    merged = qualified.model_copy(update={"custom_features": features})
    job = job_for(merged.model_dump())
    assert job is not None
    assert job.due == NOW


def test_inherited_deadline_proof_survives_sqlite_without_forgotten_source_payload(
    storage,
):
    private = "10 Secret Street"
    quote = f"Keep my address {private} only until 2033-05-18T03:33:20Z."
    deadline = ExplicitStorageDeadline(
        deadline_utc=DEADLINE.deadline_utc, source_quote=quote
    )
    old = _profile("private-parent").model_copy(
        update={
            "content": f"Address is {private}; enjoys hiking",
            "source_span": f"{quote} I enjoy hiking.",
            "custom_features": new_features(
                None, now=NOW - 100, deadline=deadline, source_texts=[quote]
            ),
            "expiration_timestamp": NOW - 1,
        }
    )
    _save(storage, old)
    native_old = storage.get_profile_by_id("private-parent")
    features = merge_features(
        [native_old, _profile("forget-source")],
        inherited_features={},
        existing_profiles=[native_old],
        erase_replaced_payloads=True,
    )
    safe = _profile("safe-replacement").model_copy(
        update={
            "content": "Enjoys hiking",
            "source_span": "I enjoy hiking.",
            "custom_features": features,
            "expiration_timestamp": expiration_for(features),
        }
    )
    store = NativeLifecycleDeadlineStore(storage, "owner")
    plan = replace(_plan(native_old, [safe]), lifecycle_epoch=NOW - 10)
    publish_with_lifecycle(
        store, plan, lambda projected: _persist(storage, projected), NOW - 1
    )
    assert storage.get_profile_by_id("private-parent") is None
    persisted = storage.get_profile_by_id("safe-replacement")
    assert persisted is not None
    assert private not in persisted.model_dump_json()
    proof = metadata(persisted)["explicit_storage_deadline"]
    assert "source_quote" not in proof
    assert len(proof["source_quote_sha256"]) == 64
    assert proof["deadline_utc"] == deadline.deadline_utc
    native_rows = store._rows(profile_id="safe-replacement")
    assert private not in str(native_rows)
    job = job_for(native_rows[0])
    assert job is not None and job.due == NOW
    assert store.purge_due(NOW - 1) == []
    assert [entry["profile_id"] for entry in store.purge_due(NOW)] == [
        "safe-replacement"
    ]
    assert storage.get_profile_by_id("safe-replacement") is None
    assert storage.conn.execute("SELECT count(*) FROM profiles_fts").fetchone()[0] == 0
    if storage._has_sqlite_vec:
        assert (
            storage.conn.execute("SELECT count(*) FROM profiles_vec").fetchone()[0] == 0
        )


@pytest.mark.parametrize("bad", [{}, None, {"created_at": None}])
def test_malformed_qualified_metadata_refuses_instead_of_becoming_legacy(bad):
    profile = _profile()
    profile.custom_features = {KEY: bad}
    with pytest.raises(ValueError):
        metadata(profile)
    with pytest.raises(ValueError):
        job_for(profile.model_dump())


def test_codec_preserves_original_publication_epoch_and_full_permission():
    plan = _plan(_profile(), [_profile("new")])
    restored = decode_write_plan(encode_write_plan(plan), "profile")
    assert isinstance(restored, ProfileWritePlan)
    assert restored == plan
    assert restored.lifecycle_epoch == NOW
    assert (
        restored.erasure_instructions[0].expected_version
        == plan.erasure_instructions[0].expected_version
    )


def test_old_saved_plan_recovers_original_payload_without_sweeping_current_authority(
    storage,
):
    from reflexio.server.services.profile.service import ProfileGenerationService

    due = _profile("due-existing", deadline=DEADLINE)
    _save(storage, due)
    legacy = _profile("legacy-result", legacy=True)
    plan = ProfileWritePlan(
        "owner",
        "old-request",
        [legacy],
        [],
        lineage_contexts=[
            LineageContext(
                op_kind="create", actor="extractor", request_id="old-request"
            )
        ],
    )
    raw = encode_write_plan(plan)
    assert raw is not None
    raw.pop("lifecycle_epoch")
    raw.pop("erasure_instructions")
    recovered = decode_write_plan(raw, "profile")
    assert isinstance(recovered, ProfileWritePlan)
    service = object.__new__(ProfileGenerationService)
    service.storage = storage
    service._persist_write_plan(recovered)
    persisted = storage.get_profile_by_id("legacy-result")
    assert persisted.content == legacy.content
    assert persisted.expiration_timestamp == legacy.expiration_timestamp
    assert KEY not in (persisted.custom_features or {})
    assert storage.get_profile_by_id("due-existing") is not None


@pytest.mark.parametrize("authority", ["metadata", "erasure"])
@pytest.mark.parametrize("execution_epoch", [None, NOW])
def test_missing_epoch_cannot_publish_new_authority(
    storage, authority, execution_epoch
):
    from reflexio.server.services.profile.service import ProfileGenerationService

    old = _profile()
    _save(storage, old)
    plan = replace(_plan(old, [_profile("new")]), lifecycle_epoch=None)
    if authority == "metadata":
        plan.erasure_instructions = []
    else:
        plan.new_profiles = []
    service = object.__new__(ProfileGenerationService)
    service.storage = storage
    with pytest.raises(ValueError, match="declared epoch"):
        service._persist_write_plan(plan, now=execution_epoch)
    assert storage.get_profile_by_id("old") is not None
    assert storage.get_profile_by_id("new") is None


def _prepared_deadline_plan():
    return ProfileWritePlan(
        "owner",
        "prepared-request",
        [_profile("delayed-result", deadline=DEADLINE)],
        [],
        lineage_contexts=[
            LineageContext(
                op_kind="create", actor="extractor", request_id="prepared-request"
            )
        ],
        lifecycle_epoch=NOW - 10,
    )


def _execution_service(storage, monkeypatch, execution_epoch):
    from reflexio.server.services.profile.service import ProfileGenerationService
    from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
        SQLiteStreamSQL,
    )

    def observed_now(db):
        assert storage.in_commit_scope
        assert db.conn is storage.conn
        return float(execution_epoch)

    clock = Mock(side_effect=observed_now)
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda db: clock(db))
    actor = Mock(
        side_effect=AssertionError("Recovery must not recompute the prepared plan")
    )
    monkeypatch.setattr(ProfileGenerationService, "_resolve_write_plan", actor)
    service = object.__new__(ProfileGenerationService)
    service.storage = storage
    return service, clock, actor


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("execution_epoch", [NOW - 1, NOW])
def test_delayed_publication_uses_execution_clock_without_reauthoring_plan(
    storage, monkeypatch, recovered, execution_epoch
):
    plan = _prepared_deadline_plan()
    frozen = encode_write_plan(plan)
    if recovered:
        plan = decode_write_plan(frozen, "profile")
    assert isinstance(plan, ProfileWritePlan)
    service, clock, actor = _execution_service(storage, monkeypatch, execution_epoch)
    service._persist_write_plan(plan)
    assert encode_write_plan(plan) == frozen
    assert plan.lifecycle_epoch == NOW - 10
    assert metadata(plan.new_profiles[0])["created_at"] == NOW - 100
    assert (storage.get_profile_by_id("delayed-result") is not None) == (
        execution_epoch < NOW
    )
    expected = 1 if execution_epoch < NOW else 0
    assert (
        storage.conn.execute("SELECT count(*) FROM profiles_fts").fetchone()[0]
        == expected
    )
    if storage._has_sqlite_vec:
        assert (
            storage.conn.execute("SELECT count(*) FROM profiles_vec").fetchone()[0]
            == expected
        )
    clock.assert_called_once()
    actor.assert_not_called()


def test_recovered_delayed_publication_rolls_back_and_retries_same_frozen_plan(
    storage, monkeypatch
):
    _save(
        storage, _profile("already-stored", deadline=DEADLINE, status=Status.SUPERSEDED)
    )
    frozen = encode_write_plan(_prepared_deadline_plan())
    plan = decode_write_plan(frozen, "profile")
    assert isinstance(plan, ProfileWritePlan)
    service, clock, actor = _execution_service(storage, monkeypatch, NOW)
    with (
        pytest.raises(RuntimeError, match="Lost receipt fence"),
        storage.commit_scope(),
    ):
        service._persist_write_plan(plan)
        raise RuntimeError("Lost receipt fence")
    assert (
        storage.get_profile_by_id("already-stored", include_tombstones=True) is not None
    )
    assert storage.get_profile_by_id("delayed-result") is None
    assert not any(event.op == "hard_delete" for event in storage.get_lineage_events())
    assert storage.conn.execute("SELECT count(*) FROM profiles_fts").fetchone()[0] == 1
    service._persist_write_plan(plan)
    assert storage.get_profile_by_id("already-stored", include_tombstones=True) is None
    assert storage.get_profile_by_id("delayed-result") is None
    assert encode_write_plan(plan) == frozen
    assert storage.conn.execute("SELECT count(*) FROM profiles_fts").fetchone()[0] == 0
    assert clock.call_count == 2
    actor.assert_not_called()


def test_explicit_execution_clock_does_not_read_or_change_preparation_clock(
    storage, monkeypatch
):
    plan = _prepared_deadline_plan()
    frozen = encode_write_plan(plan)
    service, clock, actor = _execution_service(storage, monkeypatch, NOW - 1)
    service._persist_write_plan(plan, now=NOW)
    assert storage.get_profile_by_id("delayed-result") is None
    assert encode_write_plan(plan) == frozen
    clock.assert_not_called()
    actor.assert_not_called()
