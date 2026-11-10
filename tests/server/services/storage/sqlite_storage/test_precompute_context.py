"""Storage preparation preserves caller context in independently entered workers."""

from contextlib import nullcontext
from contextvars import ContextVar
from importlib import import_module
from threading import Barrier
from time import monotonic
from types import SimpleNamespace

import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.llm.call_deadline import model_call_deadline, model_deadline
from reflexio.server.services.storage.error import StorageError

_SCOPE: ContextVar[str | None] = ContextVar("precompute_test_scope", default=None)
_PATHS = {
    "user": (
        "reflexio.server.services.storage.sqlite_storage.playbook._user",
        "UserPlaybookStoreMixin",
        "precompute_user_playbook_embeddings",
    ),
    "profile": (
        "reflexio.server.services.storage.sqlite_storage.profiles._profile_store",
        "ProfileStoreMixin",
        "precompute_profile_embeddings",
    ),
    "agent": (
        "reflexio.server.services.storage.sqlite_storage.playbook._agent",
        "AgentPlaybookStoreMixin",
        "prepare_agent_playbooks_for_save",
    ),
}


def _row(kind):
    if kind == "profile":
        return UserProfile(
            profile_id="profile",
            user_id="synthetic",
            content="content",
            last_modified_timestamp=0,
            generated_from_request_id="request",
        )
    if kind == "agent":
        return AgentPlaybook(
            agent_version="version", content="content", trigger="trigger"
        )
    return UserPlaybook(
        agent_version="version",
        request_id="request",
        content="content",
        trigger="trigger",
    )


@pytest.mark.parametrize("kind", _PATHS)
@pytest.mark.parametrize(
    "bounded", [False, True], ids=["no-deadline", "existing-deadline"]
)
@pytest.mark.parametrize("failure", [None, "embedding", "expansion"])
def test_preparation_workers_preserve_and_isolate_caller_context(
    kind, bounded, failure
):
    module, cls, method = _PATHS[kind]
    prepare = getattr(getattr(import_module(module), cls), method)
    # Sequential distinct callers detect context retained from a previous call.
    for caller in ("scope-a", "scope-b"):
        seen = {}
        entered = Barrier(2, timeout=5)
        deadline = monotonic() + 60 if bounded else None

        def operation(stage, *, observations=seen, barrier=entered):
            observations[stage] = (_SCOPE.get(), model_call_deadline.get())
            # Mutating one worker's Context must not affect its sibling/caller.
            _SCOPE.set(stage)
            model_call_deadline.set(-1)
            # Both tasks must enter concurrently: reusing one Context raises
            # RuntimeError, while serial execution breaks this barrier.
            barrier.wait()
            if failure == stage:
                raise ValueError("synthetic preparation failure")
            return [0.1] * 512 if stage == "embedding" else "expanded terms"

        storage = SimpleNamespace(
            _get_embedding=lambda _text: operation("embedding"),
            _expand_document=lambda _text: operation("expansion"),
            _should_expand_documents=lambda: True,
            _prepared_agent_playbooks=lambda: {},
            llm_client=SimpleNamespace(
                config=SimpleNamespace(fallback_models=[]),
            ),
        )
        row = _row(kind)
        token = _SCOPE.set(caller)
        try:
            with model_deadline(deadline) if deadline is not None else nullcontext():
                if failure is not None:
                    with pytest.raises(
                        (ValueError, StorageError),
                        match="synthetic preparation failure",
                    ):
                        prepare(storage, [row])
                else:
                    prepare(storage, [row])
                    assert row.embedding == [0.1] * 512
                    assert row.expanded_terms == (None if failure else "expanded terms")
                assert _SCOPE.get() == caller
                assert model_call_deadline.get() == deadline
        finally:
            _SCOPE.reset(token)
        assert seen == {
            "embedding": (caller, deadline),
            "expansion": (caller, deadline),
        }


@pytest.mark.integration
@pytest.mark.parametrize("kind", _PATHS)
def test_context_preserving_preparation_persists_without_repeating_models(
    kind, tmp_path, monkeypatch
):
    import json

    from reflexio.models.api_schema.domain.entities import LineageContext
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    vector = [0.1] * 512
    monkeypatch.setattr(SQLiteStorage, "_get_embedding", lambda *_a: vector)
    storage = SQLiteStorage(org_id="context-test", db_path=str(tmp_path / "context.db"))
    calls = []
    deadline = monotonic() + 60

    def operation(stage):
        assert _SCOPE.get() == "persist-scope"
        assert model_call_deadline.get() == deadline
        calls.append(stage)
        return vector if stage == "embedding" else "expanded terms"

    monkeypatch.setattr(storage, "_get_embedding", lambda _text: operation("embedding"))
    monkeypatch.setattr(
        storage, "_expand_document", lambda _text: operation("expansion")
    )
    monkeypatch.setattr(storage, "_should_expand_documents", lambda: True)
    row = _row(kind)
    if kind == "user":
        assert isinstance(row, UserPlaybook)
        row.user_id = "synthetic"
    _, _, method = _PATHS[kind]
    token = _SCOPE.set("persist-scope")
    try:
        with model_deadline(deadline):
            getattr(storage, method)([row])
            if kind == "profile":
                assert isinstance(row, UserProfile)
                storage.add_user_profile("synthetic", [row], skip_embedding=True)
            elif kind == "user":
                assert isinstance(row, UserPlaybook)
                storage.save_user_playbooks([row], skip_embedding=True)
            else:
                assert isinstance(row, AgentPlaybook)
                storage.save_prepared_agent_playbooks(
                    [row], lineage_contexts=[LineageContext(op_kind="create")]
                )
        query = {
            "profile": "SELECT embedding, expanded_terms FROM profiles LIMIT 1",
            "user": "SELECT embedding, expanded_terms FROM user_playbooks LIMIT 1",
            "agent": "SELECT embedding, expanded_terms FROM agent_playbooks LIMIT 1",
        }[kind]
        saved = storage.conn.execute(query).fetchone()
        assert saved is not None
        assert json.loads(saved["embedding"]) == vector
        assert saved["expanded_terms"] == "expanded terms"
        assert sorted(calls) == ["embedding", "expansion"]
    finally:
        _SCOPE.reset(token)
        storage.conn.close()
