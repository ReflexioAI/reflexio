"""A legacy ``loaded`` key on retrieved learnings is accepted, then discarded.

``RetrievedLearning`` once declared an optional ``loaded`` marker. It was
removed; clients built against it may still send the key, and rows published
while it existed may still carry it inside the ``retrieved_learnings`` JSON.
Neither may fail: the publish must not 422, and the key must never be stored
again or returned from a read.

Real SQLite storage, mocked LLM.
"""

from __future__ import annotations

import json

import pytest

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
    RetrievedLearning,
)
from reflexio.models.api_schema.ui.converters import to_interaction_view
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_REFS = [
    {"kind": "user_playbook", "learning_id": "1"},
    {"kind": "profile", "learning_id": "p2"},
    {"kind": "agent_playbook", "learning_id": "3"},
]


@pytest.fixture(autouse=True)
def _no_embeddings_or_scheduler(monkeypatch):
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )


def _service(tmp_path, worker_id: str) -> GenerationService:
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=RequestContext(
            org_id=f"legacy_loaded_{worker_id}", storage_base_dir=str(tmp_path)
        ),
    )


def _request() -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest.model_validate(
        {
            "request_id": "legacy-loaded-request",
            "user_id": "u1",
            "session_id": "s1",
            "interaction_data_list": [
                {"role": "User", "content": "hello", "created_at": 1_700_000_000},
                {
                    "role": "Agent",
                    "content": "hi",
                    "created_at": 1_700_000_001,
                    "retrieved_learnings": [
                        {"kind": "user_playbook", "learning_id": "1", "loaded": True},
                        {"kind": "profile", "learning_id": "p2", "loaded": False},
                        {"kind": "agent_playbook", "learning_id": "3"},
                    ],
                },
            ],
        }
    )


def _stored_refs(store: SQLiteStorage) -> list[dict]:
    (raw,) = [
        row[0]
        for row in store.conn.execute(
            "SELECT retrieved_learnings FROM interactions WHERE role = 'Agent'"
        ).fetchall()
    ]
    return json.loads(raw)


def _agent_turn_view(store: SQLiteStorage) -> dict:
    agent_turn = next(i for i in store.get_user_interaction("u1") if i.role == "Agent")
    return to_interaction_view(agent_turn).model_dump(mode="json")


def test_publish_with_legacy_loaded_succeeds_and_drops_it(tmp_path, worker_id):
    svc = _service(tmp_path, worker_id)
    store = svc.storage
    assert isinstance(store, SQLiteStorage)
    request = _request()
    # Accepted with a warning naming the ignored key, not rejected.
    warnings = request.payload_warnings()
    assert len(warnings) == 1
    assert "loaded" in warnings[0]

    first = svc.run(request, defer_learning=True)
    assert first.interaction_ids

    assert _stored_refs(store) == _REFS
    assert _agent_turn_view(store)["retrieved_learnings"] == _REFS

    # An identical retry from the same old client is still an exact replay.
    replay = svc.run(_request(), defer_learning=True)
    assert replay.replayed and replay.interaction_ids == first.interaction_ids


def test_stale_stored_loaded_key_is_ignored_on_read(tmp_path, worker_id):
    svc = _service(tmp_path, worker_id)
    store = svc.storage
    assert isinstance(store, SQLiteStorage)
    first = svc.run(_request(), defer_learning=True)
    # Simulate a row written while ``loaded`` was a declared field.
    legacy_rows = [{**ref, "loaded": True} for ref in _REFS]
    with store.conn:
        store.conn.execute(
            "UPDATE interactions SET retrieved_learnings = ? WHERE role = 'Agent'",
            (json.dumps(legacy_rows),),
        )

    assert _agent_turn_view(store)["retrieved_learnings"] == _REFS
    replay = svc.run(_request(), defer_learning=True)
    assert replay.replayed and replay.interaction_ids == first.interaction_ids


def test_loaded_is_reported_as_an_unknown_field():
    data = InteractionData.model_validate(
        {
            "role": "Agent",
            "content": "hi",
            "retrieved_learnings": [
                {"kind": "profile", "learning_id": "p1", "loaded": True}
            ],
        }
    )
    assert data.reportable_unknown_fields() == ["retrieved_learnings[0].loaded"]
    assert "loaded" not in data.retrieved_learnings[0].model_dump()
    hydrated = RetrievedLearning.from_stored_payload(
        {"kind": "profile", "learning_id": "p1", "loaded": True}
    )
    assert "loaded" not in hydrated.persistence_payload()
