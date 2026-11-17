"""Idle source-authorized profile deadlines use the existing maintenance owner."""

from concurrent.futures import CancelledError
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import Status, UserProfile
from reflexio.models.config_schema import LineageGCConfig
from reflexio.server.services.lineage import gc_scheduler
from reflexio.server.services.profile.lifecycle import (
    ExplicitStorageDeadline,
    expiration_for,
    new_features,
)
from reflexio.server.services.profile.maintenance import purge_due_profile_payloads
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

NOW = 2_000_000_000


def profile(identity, owner="owner", *, due=True, status=None):
    deadline = ExplicitStorageDeadline(
        deadline_utc=datetime.fromtimestamp(NOW if due else NOW + 100, UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        source_quote="Forget then",
    )
    features = new_features(
        None, now=NOW - 100, deadline=deadline, source_texts=["Forget then"]
    )
    return UserProfile(
        profile_id=identity,
        user_id=owner,
        content="Source fact",
        last_modified_timestamp=NOW - 100,
        generated_from_request_id="source",
        custom_features=features,
        expiration_timestamp=expiration_for(features),
        status=status,
        embedding=[1.0] + [0.0] * 511,
    )


@pytest.fixture
def storage(tmp_path):
    with patch.object(
        SQLiteStorage, "_get_embedding", return_value=[1.0] + [0.0] * 511
    ):
        store = SQLiteStorage(
            org_id="maintenance", db_path=str(tmp_path / "profiles.db")
        )
        yield store
        store.conn.close()


def test_idle_deadline_payload_is_maintained_with_all_optional_gc_off(
    storage, monkeypatch
):
    due = profile("due", status=Status.SUPERSEDED)
    legacy = profile("legacy").model_copy(update={"custom_features": None})
    future = profile("future", due=False)
    storage.add_user_profile("owner", [due, legacy, future])
    assert storage.profile_lifecycle_due_users(now=NOW) == ["owner"]
    assert storage.profile_lifecycle_due_users(now=NOW - 1) == []
    monkeypatch.setattr(gc_scheduler.time, "time", lambda: NOW)
    monkeypatch.setattr(
        gc_scheduler, "sweep_retention_caps", lambda *_: SimpleNamespace(failed=False)
    )
    cfg = SimpleNamespace(
        lineage_gc=LineageGCConfig(tombstone_grace_window_days=0),
        expiry_reclamation=SimpleNamespace(enabled=False),
    )
    owner = gc_scheduler.LineageGCScheduler(
        request_context_factory=MagicMock(), bootstrap_org_id="maintenance"
    )
    # No actor, request, result or new profile publication occurs after the
    # deadline. The pre-existing owner alone performs the physical maintenance.
    owner._sweep_project_data("maintenance", storage, cfg)
    assert storage.get_profile_by_id("due", include_tombstones=True) is None
    assert storage.get_profile_by_id("legacy") is not None
    assert storage.get_profile_by_id("future") is not None
    assert storage.profile_lifecycle_due_users(now=NOW) == []
    assert owner._fanout_owner is None


def test_due_subject_discovery_keysets_full_pages_without_legacy_ttl_authority(storage):
    for index in range(260):
        owner = f"user-{index:03}"
        storage.add_user_profile(owner, [profile(f"p-{index}", owner)])
    first = storage.profile_lifecycle_due_users(now=NOW)
    second = storage.profile_lifecycle_due_users(now=NOW, after_user_id=first[-1])
    assert len(first) == 256 and len(second) == 4
    assert first + second == [f"user-{index:03}" for index in range(260)]
    assert storage.profile_lifecycle_due_users(now=NOW, after_user_id=second[-1]) == []


def test_maintenance_joins_existing_rollback_owner(storage):
    storage.add_user_profile("owner", [profile("due")])
    with pytest.raises(RuntimeError, match="rollback"), storage.commit_scope():
        purge_due_profile_payloads(storage, NOW)
        raise RuntimeError("rollback")
    assert storage.get_profile_by_id("due", include_tombstones=True) is not None


def test_failed_subject_does_not_skip_sibling_and_failure_is_not_silenced():
    storage = MagicMock()
    storage.profile_lifecycle_due_users.side_effect = [["a", "b"], []]
    bad, good = MagicMock(), MagicMock()
    bad.purge_due.side_effect = ValueError("source authority malformed")
    good.purge_due.return_value = [{"status": "hard_deleted"}]
    with (
        patch(
            "reflexio.server.services.profile.maintenance.NativeLifecycleDeadlineStore",
            side_effect=[bad, good],
        ),
        pytest.raises(ValueError, match="source authority malformed"),
    ):
        purge_due_profile_payloads(storage, NOW, "project")
    good.purge_due.assert_called_once_with(NOW, checkpoint=None)
    assert storage.profile_lifecycle_due_users.call_args.kwargs["after_user_id"] == "b"


def test_required_project_error_keeps_existing_failure_observer_and_sibling_project(
    monkeypatch,
):
    calls = []

    def sweep(storage, now, project_id, *, checkpoint=None):
        calls.append(project_id)
        if project_id == "bad":
            raise ValueError("source authority malformed")
        return 0

    monkeypatch.setattr(
        gc_scheduler, "_REQUIRED_PROJECT_SWEEPS", (("profile_deadlines", sweep),)
    )
    scheduler = gc_scheduler.LineageGCScheduler(
        request_context_factory=MagicMock(), bootstrap_org_id="org"
    )
    with (
        patch.object(gc_scheduler, "report_background_failure") as failure,
        patch.object(gc_scheduler, "report_background_success") as success,
    ):
        scheduler._sweep_required_project_data("org", MagicMock(), "bad")
        scheduler._sweep_required_project_data("org", MagicMock(), "good")
    assert calls == ["bad", "good"]
    assert scheduler._tick_had_failure
    assert failure.call_args.kwargs["scope"] == "lineage-profile_deadlines:org:bad"
    success.assert_called_once_with("lineage-profile_deadlines:org:good")


def test_nonadvancing_subject_page_fails_without_an_unbounded_retry_loop():
    storage = MagicMock()
    # A nonconforming adapter returning a truthy but empty iterable must not
    # keep the shared maintenance worker spinning on the same keyset forever.
    storage.profile_lifecycle_due_users.return_value = MagicMock()
    with pytest.raises(ValueError, match="subject page did not advance"):
        purge_due_profile_payloads(storage, NOW)
    storage.profile_lifecycle_due_users.assert_called_once()


def test_subject_keyset_follows_storage_collation_for_opaque_user_ids():
    storage = MagicMock()
    storage.profile_lifecycle_due_users.side_effect = [["a", "B"], []]
    per_user = MagicMock()
    per_user.purge_due.return_value = []
    with patch(
        "reflexio.server.services.profile.maintenance.NativeLifecycleDeadlineStore",
        return_value=per_user,
    ):
        assert purge_due_profile_payloads(storage, NOW) == 0
    assert storage.profile_lifecycle_due_users.call_args.kwargs["after_user_id"] == "B"
    assert per_user.purge_due.call_count == 2


def test_stopping_owner_does_not_start_a_subject_page_or_report_success():
    storage = MagicMock()
    scheduler = gc_scheduler.LineageGCScheduler(
        request_context_factory=MagicMock(), bootstrap_org_id="org"
    )
    scheduler._stop_event.set()
    with (
        patch.object(gc_scheduler, "report_background_failure") as failure,
        patch.object(gc_scheduler, "report_background_success") as success,
    ):
        assert not scheduler._sweep_required_project_data("org", storage, None)
    storage.profile_lifecycle_due_users.assert_not_called()
    failure.assert_not_called()
    success.assert_not_called()
    assert not scheduler._tick_had_failure


def test_cancelled_subject_is_not_absorbed_as_a_local_failure():
    storage = MagicMock()
    storage.profile_lifecycle_due_users.return_value = ["a", "b"]
    per_user = MagicMock()
    per_user.purge_due.side_effect = CancelledError("owner stopping")
    with (
        patch(
            "reflexio.server.services.profile.maintenance.NativeLifecycleDeadlineStore",
            return_value=per_user,
        ) as constructor,
        pytest.raises(CancelledError),
    ):
        purge_due_profile_payloads(storage, NOW)
    assert constructor.call_count == 1


def test_owner_stop_during_sqlite_purge_prevents_next_profile_page_and_user(
    storage, monkeypatch
):
    storage.add_user_profile("a", [profile("a-profile", "a")])
    storage.add_user_profile("b", [profile("b-profile", "b")])
    scheduler = gc_scheduler.LineageGCScheduler(
        request_context_factory=MagicMock(), bootstrap_org_id="maintenance"
    )
    native_delete = storage.delete_profile_lifecycle_payload

    def delete_then_stop(*args, **kwargs):
        deleted = native_delete(*args, **kwargs)
        scheduler._stop_event.set()
        return deleted

    monkeypatch.setattr(gc_scheduler.time, "time", lambda: NOW)
    with (
        patch.object(
            storage, "delete_profile_lifecycle_payload", side_effect=delete_then_stop
        ),
        patch.object(
            storage, "profile_lifecycle_rows", wraps=storage.profile_lifecycle_rows
        ) as reads,
        patch.object(gc_scheduler, "report_background_failure") as failure,
        patch.object(gc_scheduler, "report_background_success") as success,
    ):
        assert not scheduler._sweep_required_project_data("maintenance", storage, None)
    assert storage.get_profile_by_id("a-profile", include_tombstones=True) is None
    assert storage.get_profile_by_id("b-profile", include_tombstones=True) is not None
    page_reads = [
        call for call in reads.call_args_list if not call.kwargs.get("lock", False)
    ]
    assert len(page_reads) == 1
    failure.assert_not_called()
    success.assert_not_called()
    assert not scheduler._tick_had_failure
