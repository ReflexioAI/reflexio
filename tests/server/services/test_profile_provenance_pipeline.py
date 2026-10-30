"""Canonical profile evidence survives HTTP extraction, consolidation and reopen."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.test_support.skip_decorators import skip_in_precommit
from tests.server.services.test_durable_window_pipeline import (
    payload,
)
from tests.server.services.test_durable_window_pipeline import (
    pipeline as imported_pipeline,
)

pytestmark = pytest.mark.integration
provenance_pipeline = imported_pipeline


@skip_in_precommit
def test_http_profile_provenance_survives_mixed_merge_and_storage_reopen(
    provenance_pipeline, monkeypatch
):
    engine, worker, client = provenance_pipeline
    storage = engine.get_storage()
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    quotes = ["I prefer short answers.", "Keep my answers brief, including examples."]
    notes = ["Initial preference caveat", "Follow-up clarification"]
    phase = {"index": 0}

    def respond(params, _hard_timeout):
        prompt = "\n".join(message.get("content", "") for message in params["messages"])
        deduplication = "You are a profile deduplication assistant." in prompt
        extraction = "You are a user personalization learning assistant." in prompt
        assert deduplication != extraction, "Unexpected or ambiguous provider stage"
        index = phase["index"]
        if deduplication:
            assert "[NEW Profiles]" in prompt and "[EXISTING Profiles]" in prompt
            result = (
                {
                    "duplicate_groups": [
                        {
                            "item_ids": ["NEW-0", "EXISTING-0"],
                            "merged_content": "The user prefers concise answers with examples.",
                            "merged_time_to_live": "infinity",
                        }
                    ],
                    "unique_ids": [],
                    "deletions": [],
                }
                if index
                else {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
            )
        else:
            # MiniMax's transport is json_object; the real policy and immutable
            # source presentation identify extraction without a schema-name hint.
            assert '"original_chunk"' in prompt and quotes[index] in prompt
            result = {
                "profiles": [
                    {
                        "content": "The user prefers concise answers.",
                        "time_to_live": "infinity",
                        "source_span": quotes[index],
                        "notes": notes[index],
                        "reader_angle": "preferences",
                        "topic": "answer-style",
                    }
                ]
            }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(result), tool_calls=None, refusal=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="minimax/MiniMax-M3",
            _hidden_params={"custom_llm_provider": "minimax"},
        )

    dispatch = Mock(side_effect=respond)
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    request = payload("profile-first", force=True)
    for interaction in request["interaction_data_list"]:
        interaction["content"] = quotes[0]
    assert client.post("/api/publish_interaction", json=request).json()["success"]
    worker.drain_org("stream-test", 3, 300)
    original = storage.get_user_profile("u")
    assert len(original) == 1
    old = original[0]
    assert (old.source_span, old.notes, old.reader_angle) == (
        quotes[0],
        notes[0],
        "preferences",
    )
    assert old.custom_features == {"topic": "answer-style"}
    phase["index"] = 1
    request = payload("profile-second", count=5, force=True)
    for interaction in request["interaction_data_list"]:
        interaction["content"] = quotes[1]
    assert client.post("/api/publish_interaction", json=request).json()["success"]
    worker.drain_org("stream-test", 3, 300)
    assert (
        client.get(
            "/api/learning_status", params={"request_id": "profile-second"}
        ).json()["status"]
        == "done"
    )
    current = storage.get_user_profile("u")
    assert len(current) == 1
    survivor = current[0]
    assert survivor.profile_id != old.profile_id
    assert survivor.source_span is not None and survivor.notes is not None
    assert survivor.source_span.startswith(
        "Original source evidence values (separate excerpts):"
    )
    assert all(quote in survivor.source_span for quote in quotes)
    assert all(note in survivor.notes for note in notes)
    assert survivor.reader_angle == "preferences"
    assert survivor.custom_features == {"topic": "answer-style"}
    assert len(survivor.source_interaction_ids) == 15
    assert set(survivor.source_interaction_ids) == set(range(1, 16))
    lineage = storage.get_lineage_events(
        entity_type="profile", entity_id=survivor.profile_id
    )
    assert any(old.profile_id in event.source_ids for event in lineage)
    with storage._stream_sql() as db:
        assert (
            db.query(
                "SELECT status,superseded_by FROM profiles WHERE profile_id=?",
                (old.profile_id,),
            )[0]["status"]
            == "superseded"
        )
    snapshot = survivor.model_dump(mode="json")
    db_path = storage.db_path
    storage.conn.close()
    reopened = SQLiteStorage(org_id=storage.org_id, db_path=db_path)
    try:
        persisted = reopened.get_user_profile("u")
        assert len(persisted) == 1
        assert persisted[0].model_dump(mode="json") == snapshot
        assert any(
            old.profile_id in event.source_ids
            for event in reopened.get_lineage_events(
                entity_type="profile", entity_id=survivor.profile_id
            )
        )
    finally:
        reopened.conn.close()
