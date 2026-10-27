"""A saved old policy reaches real SQLite FTS and the current floor scorer."""

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from reflexio.config_file import LocalConfigFile
from reflexio.models.api_schema.domain.entities import (
    NEVER_EXPIRES_TIMESTAMP,
    UserProfile,
)
from reflexio.models.api_schema.retriever_schema import UnifiedSearchRequest
from reflexio.models.config_schema import (
    SearchMode,
    StorageConfigSQLite,
    validate_stored_config,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.unified_search_service import run_unified_search


@pytest.mark.parametrize("enabled", [False, True])
def test_saved_retrieval_policy_survives_upgrade_and_real_search(
    tmp_path, monkeypatch, request, enabled
):
    database = tmp_path / "owned.sqlite"
    path = tmp_path / "config.json"
    original = json.dumps(
        {
            "storage_config": {"type": "sqlite", "db_path": str(database)},
            "retrieval_floor": {
                "enabled": enabled,
                "pool_size": 7,
                "profile_floor": 0.0,
            },
        }
    ).encode()
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    assert owner.prepare() is True
    config = validate_stored_config(owner.read_document())
    assert isinstance(config.storage_config, StorageConfigSQLite)
    assert config.storage_config.db_path == str(database)
    storage = SQLiteStorage(
        org_id="floor-upgrade", db_path=config.storage_config.db_path
    )
    request.addfinalizer(storage.conn.close)
    storage.migrate()
    # Embedding transport is outside this FTS test; all database reads/writes,
    # config preparation and search phases remain real.
    monkeypatch.setattr(
        storage, "_get_embedding", lambda *_args, **_kwargs: [0.0] * 512
    )
    storage.add_user_profile(
        "u",
        [
            UserProfile(
                user_id="u",
                profile_id=identity,
                content=content,
                last_modified_timestamp=1_700_000_000,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                generated_from_request_id="owned-request",
            )
            for identity, content in [
                ("good", "pasta preference"),
                ("junk", "pasta unrelated"),
            ]
        ],
    )
    monkeypatch.setenv("REFLEXIO_RERANK_ENABLED", "true")
    calls = []

    def score(query, documents):
        calls.append((query, list(documents)))
        return "cross-encoder/ms-marco-MiniLM-L-6-v2", [
            1.0 if content == "pasta preference" else -1.0 for content in documents
        ]

    monkeypatch.setattr(
        "reflexio.server.services.retrieval.relevance_floor.score_pairs_with_model",
        score,
    )
    response = run_unified_search(
        request=UnifiedSearchRequest(
            query="pasta",
            user_id="u",
            top_k=2,
            search_mode=SearchMode.FTS,
            enable_reformulation=False,
            entity_types=["profiles"],
        ),
        org_id="floor-upgrade",
        storage=storage,
        llm_client=MagicMock(),
        prompt_manager=MagicMock(),
        retrieval_floor=config.retrieval_floor,
    )
    assert response.success
    assert {profile.profile_id for profile in response.profiles} == (
        {"good"} if enabled else {"good", "junk"}
    )
    assert len(calls) == int(enabled)
    assert (
        tmp_path.joinpath(
            ".config-migrations",
            path.name,
            hashlib.sha256(original).hexdigest() + ".json",
        ).read_bytes()
        == original
    )
    assert owner.prepare() is False
