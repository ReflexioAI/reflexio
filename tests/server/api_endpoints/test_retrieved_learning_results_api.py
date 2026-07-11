"""Route tests for POST /api/get_retrieved_learning_evaluation_results.

Uses a per-test isolated SQLite directory (via ``LOCAL_STORAGE_PATH``) instead
of the developer's shared default database: these tests seed rows with fixed
identities and assert on unfiltered ordering, which must not observe — or
collide with — rows from other suites or earlier runs.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from reflexio.models.api_schema.domain import RetrievedLearningEvaluationResult
from reflexio.server.api import create_app
from reflexio.server.cache.reflexio_cache import (
    get_reflexio,
    invalidate_reflexio_cache,
)

ENDPOINT = "/api/get_retrieved_learning_evaluation_results"


@pytest.fixture
def isolated_client(tmp_path, monkeypatch) -> Generator[tuple[TestClient, str]]:
    monkeypatch.setattr("reflexio.server.LOCAL_STORAGE_PATH", str(tmp_path))
    org_id = f"test-rle-api-{uuid.uuid4().hex[:12]}"
    app = create_app(get_org_id=lambda: org_id)
    client = TestClient(app, raise_server_exceptions=False)
    get_reflexio(org_id=org_id)
    try:
        yield client, org_id
    finally:
        invalidate_reflexio_cache(org_id=org_id)


def _seed_rows(org_id: str) -> None:
    storage = get_reflexio(org_id=org_id).request_context.storage
    assert storage is not None
    conn = storage.conn  # type: ignore[attr-defined]
    rows = [
        ("u1", "sess-1", "profile", "prof-1", 100),
        ("u1", "sess-1", "user_playbook", "42", 100),
        ("u2", "sess-2", "agent_playbook", "7", 200),
    ]
    for user_id, session_id, kind, learning_id, created_at in rows:
        conn.execute(
            """INSERT INTO retrieved_learning_evaluation
               (user_id, session_id, agent_version, kind, learning_id,
                is_relevant, relevance_reason, impact, impact_reason, created_at)
               VALUES (?, ?, 'v1', ?, ?, 1, 'fits', 'positive', 'helped', ?)""",
            (user_id, session_id, kind, learning_id, created_at),
        )
    conn.commit()


def test_returns_rows_newest_first_with_filters(isolated_client) -> None:
    client, org_id = isolated_client
    _seed_rows(org_id)

    resp = client.post(ENDPOINT, json={})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert [r["session_id"] for r in body["results"]] == ["sess-2", "sess-1", "sess-1"]
    first = body["results"][0]
    assert first["kind"] == "agent_playbook"
    assert first["learning_id"] == "7"
    assert first["is_relevant"] is True
    assert first["impact"] == "positive"
    assert "title" not in first
    assert "real_id" not in first

    by_session = client.post(ENDPOINT, json={"session_id": "sess-1"}).json()
    assert len(by_session["results"]) == 2
    by_user = client.post(ENDPOINT, json={"user_id": "u2"}).json()
    assert len(by_user["results"]) == 1
    limited = client.post(ENDPOINT, json={"limit": 1}).json()
    assert len(limited["results"]) == 1


def test_empty_result_is_success(isolated_client) -> None:
    client, _org_id = isolated_client
    resp = client.post(ENDPOINT, json={"session_id": "no-such-session"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["results"] == []


def test_limit_bounds_are_validated(isolated_client) -> None:
    client, _org_id = isolated_client
    assert client.post(ENDPOINT, json={"limit": 0}).status_code == 422
    assert client.post(ENDPOINT, json={"limit": 1001}).status_code == 422


def test_result_model_round_trips_none_verdicts(isolated_client) -> None:
    client, org_id = isolated_client
    storage = get_reflexio(org_id=org_id).request_context.storage
    assert storage is not None
    conn = storage.conn  # type: ignore[attr-defined]
    conn.execute(
        """INSERT INTO retrieved_learning_evaluation
           (user_id, session_id, agent_version, kind, learning_id,
            is_relevant, relevance_reason, impact, impact_reason, created_at)
           VALUES ('u3', 'sess-3', 'v1', 'profile', 'p9', NULL, '', NULL, '', 5)""",
    )
    conn.commit()
    body = client.post(ENDPOINT, json={"session_id": "sess-3"}).json()
    row = body["results"][0]
    assert row["is_relevant"] is None
    assert row["impact"] is None
    # The wire shape parses back into the public entity.
    parsed = RetrievedLearningEvaluationResult(**row)
    assert parsed.learning_id == "p9"
