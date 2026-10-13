"""OSS persistence strips receipts even when no enterprise issuer is installed."""

import json
import sqlite3
from hashlib import sha256

from reflexio.models.api_schema.domain import Interaction, Request, RetrievedLearning
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


def test_sqlite_strips_token_without_mutating_publish_input(tmp_path):
    path = str(tmp_path / "receipts.db")
    storage = SQLiteStorage(org_id="receipt-oss", db_path=path)
    token = "opaque-served-version-receipt"
    reference = RetrievedLearning(
        kind="user_playbook",
        learning_id="7",
        exposure_receipt=token,
        exposure_receipt_digest="0" * 64,
    )
    interaction = Interaction(
        user_id="alice",
        request_id="request",
        role="Assistant",
        content="Answer",
        retrieved_learnings=[reference],
    )
    storage.add_request(
        Request(request_id="request", user_id="alice", session_id="session")
    )
    storage.add_user_interactions_bulk("alice", [interaction], embeddings_prepared=True)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT retrieved_learnings FROM interactions LIMIT 1"
        ).fetchone()
    stored = json.loads(row[0])
    assert stored[0]["exposure_receipt_digest"] == sha256(token.encode()).hexdigest()
    assert "exposure_receipt" not in stored[0]
    assert reference.exposure_receipt == token

    hydrated = storage.get_user_interaction("alice")[0]
    hydrated.content = "Edited answer"
    storage.add_user_interactions_bulk("alice", [hydrated], embeddings_prepared=True)
    rewritten = storage.get_user_interaction("alice")[0].retrieved_learnings[0]
    assert rewritten.exposure_receipt_digest == sha256(token.encode()).hexdigest()
    # A caller cannot confer database provenance through a JSON round trip.
    submitted = RetrievedLearning.model_validate(rewritten.model_dump())
    assert "exposure_receipt_digest" not in submitted.persistence_payload()
    hydrated.retrieved_learnings[0].learning_id = "8"
    assert (
        "exposure_receipt_digest"
        not in hydrated.retrieved_learnings[0].persistence_payload()
    )
