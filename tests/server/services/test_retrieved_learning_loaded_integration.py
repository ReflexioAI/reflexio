"""``RetrievedLearning.loaded`` round-trips through publish, storage and reads.

Real SQLite storage, mocked LLM. ``loaded`` records that the agent opened a
learning's full text after it was injected; it is optional, so an unset value
must stay absent on the wire and in the stored JSON (byte-compatible with
payloads written before the field existed).
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
            org_id=f"loaded_{worker_id}", storage_base_dir=str(tmp_path)
        ),
    )


def _request() -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest.model_validate(
        {
            "request_id": "loaded-request",
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


def test_loaded_round_trips_through_publish_storage_and_read(tmp_path, worker_id):
    svc = _service(tmp_path, worker_id)
    store = svc.storage
    assert isinstance(store, SQLiteStorage)
    request = _request()
    assert request.payload_warnings() == []

    first = svc.run(request, defer_learning=True)

    (raw,) = [
        row[0]
        for row in store.conn.execute(
            "SELECT retrieved_learnings FROM interactions WHERE role = 'Agent'"
        ).fetchall()
    ]
    assert json.loads(raw) == [
        {"kind": "user_playbook", "learning_id": "1", "loaded": True},
        {"kind": "profile", "learning_id": "p2", "loaded": False},
        {"kind": "agent_playbook", "learning_id": "3"},
    ]

    agent_turn = next(i for i in store.get_user_interaction("u1") if i.role == "Agent")
    assert [r.loaded for r in agent_turn.retrieved_learnings] == [True, False, None]

    view = to_interaction_view(agent_turn).model_dump(mode="json")
    assert view["retrieved_learnings"] == json.loads(raw)

    # The identical retry is still recognised as a replay with the new field.
    replay = svc.run(_request(), defer_learning=True)
    assert replay.replayed and replay.interaction_ids == first.interaction_ids


def test_unset_loaded_is_omitted_from_payloads():
    reference = RetrievedLearning(kind="profile", learning_id="p1")
    assert reference.loaded is None
    assert "loaded" not in reference.model_dump()
    assert "loaded" not in reference.persistence_payload()

    hydrated = RetrievedLearning.from_stored_payload(
        {"kind": "profile", "learning_id": "p1"}
    )
    assert hydrated.loaded is None


def test_interaction_data_accepts_loaded_without_unknown_field_warning():
    data = InteractionData.model_validate(
        {
            "role": "Agent",
            "content": "hi",
            "retrieved_learnings": [
                {"kind": "profile", "learning_id": "p1", "loaded": True}
            ],
        }
    )
    assert data.retrieved_learnings[0].loaded is True
    assert data.reportable_unknown_fields() == []
