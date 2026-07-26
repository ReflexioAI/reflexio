from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import reflexio.server.services.storage.sqlite_storage.playbook._user as playbook_mod
import reflexio.server.services.storage.sqlite_storage.profiles._profile_store as profile_mod
from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.api_schema.service_schemas import UserPlaybook, UserProfile
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _local_governance_secret(monkeypatch) -> None:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "test-governance-secret")


def _storage(tmp_path) -> SQLiteStorage:
    storage = SQLiteStorage(org_id="org-create", db_path=str(tmp_path / "create.db"))
    storage.migrate()
    return storage


def _context() -> LineageContext:
    return LineageContext(
        op_kind="create",
        actor="extractor",
        request_id="req-create",
        model_name="claude-sonnet-4-5-20250929",
        provider="anthropic",
    )


def _profile() -> UserProfile:
    return UserProfile(
        profile_id="p-create",
        user_id="u1",
        content="likes concise answers",
        last_modified_timestamp=int(time.time()),
        generated_from_request_id="req-create",
    )


def _playbook() -> UserPlaybook:
    return UserPlaybook(
        user_id="u1",
        agent_version="v1",
        request_id="req-create",
        content="Answer concisely.",
    )


def test_profile_create_event_is_atomic_and_provenance_aware(tmp_path) -> None:
    storage = _storage(tmp_path)
    storage.add_user_profile(
        "u1", [_profile()], skip_embedding=True, lineage_contexts=[_context()]
    )

    event = storage.get_lineage_events(entity_type="profile", entity_id="p-create")[0]
    assert event.op == "create"
    assert event.actor == "extractor"
    assert event.model_name == "claude-sonnet-4-5-20250929"


def test_profile_create_without_context_emits_lineage_with_unknown_model(
    tmp_path,
) -> None:
    storage = _storage(tmp_path)
    storage.add_user_profile("u1", [_profile()], skip_embedding=True)

    event = storage.get_lineage_events(entity_type="profile", entity_id="p-create")[0]
    assert event.op == "create"
    assert event.request_id == "req-create"
    assert event.model_name is None
    assert event.provider is None


def test_profile_replace_does_not_emit_a_second_create(tmp_path) -> None:
    storage = _storage(tmp_path)
    profile = _profile()
    storage.add_user_profile(
        "u1", [profile], skip_embedding=True, lineage_contexts=[_context()]
    )
    profile.content = "updated in place"
    storage.add_user_profile(
        "u1", [profile], skip_embedding=True, lineage_contexts=[_context()]
    )

    events = storage.get_lineage_events(entity_type="profile", entity_id="p-create")
    assert [event.op for event in events] == ["create"]


def test_user_playbook_create_event_is_atomic_and_provenance_aware(tmp_path) -> None:
    storage = _storage(tmp_path)
    playbook = _playbook()
    storage.save_user_playbooks(
        [playbook], skip_embedding=True, lineage_contexts=[_context()]
    )

    event = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(playbook.user_playbook_id)
    )[0]
    assert event.op == "create"
    assert event.provider == "anthropic"


def test_user_playbook_without_context_emits_lineage_with_unknown_model(
    tmp_path,
) -> None:
    storage = _storage(tmp_path)
    playbook = _playbook()

    storage.save_user_playbooks([playbook], skip_embedding=True)

    event = storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(playbook.user_playbook_id)
    )[0]
    assert event.op == "create"
    assert event.request_id == "req-create"
    assert event.model_name is None
    assert event.provider is None


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_context_length_is_validated_before_db_work(tmp_path, kind: str) -> None:
    storage = _storage(tmp_path)
    with pytest.raises(StorageError, match="lineage_contexts must match"):
        if kind == "profile":
            storage.add_user_profile(
                "u1", [_profile()], skip_embedding=True, lineage_contexts=[]
            )
        else:
            storage.save_user_playbooks(
                [_playbook()], skip_embedding=True, lineage_contexts=[]
            )
    assert storage.get_lineage_events(org_id="org-create") == []


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_create_context_rejects_other_operation_kinds(tmp_path, kind: str) -> None:
    storage = _storage(tmp_path)
    context = _context().model_copy(update={"op_kind": "revise"})

    with pytest.raises(StorageError, match="must use op_kind='create'"):
        if kind == "profile":
            storage.add_user_profile(
                "u1", [_profile()], skip_embedding=True, lineage_contexts=[context]
            )
        else:
            storage.save_user_playbooks(
                [_playbook()], skip_embedding=True, lineage_contexts=[context]
            )
    assert storage.get_lineage_events(org_id="org-create") == []


def test_profile_event_failure_rolls_back_insert(tmp_path) -> None:
    storage = _storage(tmp_path)
    with (
        patch.object(
            profile_mod, "_append_event_stmt", side_effect=RuntimeError("boom")
        ),
        pytest.raises(StorageError, match="boom"),
    ):
        storage.add_user_profile(
            "u1", [_profile()], skip_embedding=True, lineage_contexts=[_context()]
        )
    assert storage.get_profile_by_id("p-create") is None


def test_playbook_event_failure_rolls_back_insert(tmp_path) -> None:
    storage = _storage(tmp_path)
    playbook = _playbook()
    with (
        patch.object(
            playbook_mod, "_append_event_stmt", side_effect=RuntimeError("boom")
        ),
        pytest.raises(StorageError, match="boom"),
    ):
        storage.save_user_playbooks(
            [playbook], skip_embedding=True, lineage_contexts=[_context()]
        )
    assert storage.get_user_playbooks(user_id="u1") == []
