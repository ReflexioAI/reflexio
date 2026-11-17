"""SQLite semantic indexes omit lifecycle authority without changing storage."""

from copy import deepcopy
from hashlib import sha256
from json import loads
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.retriever_schema import SearchUserProfileRequest
from reflexio.models.api_schema.service_schemas import (
    ProfileTimeToLive,
    UserProfile,
)
from reflexio.models.config_schema import SearchMode
from reflexio.server.services.profile.lifecycle import KEY
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage


@pytest.mark.parametrize(
    "ordinary", [None, {"theme": "dark", "nested": {KEY: "literal"}}]
)
@pytest.mark.parametrize(
    "operation", ["precompute", "expanded_precompute", "add", "update"]
)
def test_sqlite_profile_vectors_ignore_only_lifecycle_bookkeeping(
    storage: BaseStorage, operation: str, ordinary: dict | None
) -> None:
    assert isinstance(storage, SQLiteStorage)
    plain = UserProfile(
        user_id="semantic-owner",
        profile_id="plain",
        content=f"User works on {KEY} diagnostics.",
        last_modified_timestamp=1_800_000_000,
        generated_from_request_id="semantic-request",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        custom_features=deepcopy(ordinary),
    )
    controlled = plain.model_copy(deep=True)
    controlled.profile_id = "controlled"
    controlled.custom_features = {
        **(ordinary or {}),
        KEY: {
            "created_at": 1_800_000_000,
            "ancestor_ids": ["retired-parent"],
            "explicit_storage_deadline": {
                "deadline_utc": "2030-01-01T00:00:00Z",
                "source_quote": "Keep until January 2030",
            },
        },
    }
    before = deepcopy(controlled.custom_features)
    expected = plain.content + (f"\n{ordinary}" if ordinary else "")
    inputs: list[str] = []

    def deterministic_embedding(text: str) -> list[float]:
        inputs.append(text)
        return [byte / 255 for byte in sha256(text.encode()).digest()] * 16

    if operation == "update":
        storage.add_user_profile(
            plain.user_id, [plain, controlled], skip_embedding=True
        )

    with (
        patch.object(storage, "_get_embedding", side_effect=deterministic_embedding),
        patch.object(
            storage,
            "_should_expand_documents",
            return_value=operation == "expanded_precompute",
        ),
        patch.object(storage, "_expand_document", return_value="expanded") as expand,
    ):
        if operation in {"precompute", "expanded_precompute"}:
            storage.precompute_profile_embeddings([plain, controlled])
        elif operation == "add":
            storage.add_user_profile(plain.user_id, [plain, controlled])
        else:
            for profile in [plain, controlled]:
                storage.update_user_profile_by_id(
                    profile.user_id, profile.profile_id, profile
                )
        assert inputs == [expected, expected]
        assert plain.embedding == controlled.embedding
        if operation == "expanded_precompute":
            assert [call.args for call in expand.call_args_list] == [
                (plain.content,),
                (plain.content,),
            ]
        else:
            expand.assert_not_called()

        # Ordinary custom features remain semantic input, including nested names
        # that merely resemble the reserved top-level lifecycle field.
        changed = plain.model_copy(deep=True)
        changed.custom_features = {**(ordinary or {}), "theme": "light"}
        storage.precompute_profile_embeddings([changed])
        assert inputs[-1] == changed.content + "\n" + str(changed.custom_features)
        assert changed.embedding != plain.embedding

    assert controlled.custom_features == before
    if operation in {"add", "update"}:
        stored = {
            row.profile_id: row for row in storage.get_user_profile(plain.user_id)
        }
        assert stored[controlled.profile_id].custom_features == before
        vectors = [
            loads(row["embedding"])
            for row in storage.conn.execute(
                "SELECT embedding FROM profiles WHERE user_id = ? ORDER BY profile_id",
                (plain.user_id,),
            ).fetchall()
        ]
        assert vectors == [plain.embedding, controlled.embedding]


@pytest.mark.parametrize("operation", ["add", "update"])
def test_sqlite_profile_fts_excludes_operational_metadata_but_keeps_facts(
    storage: BaseStorage, operation: str
) -> None:
    assert isinstance(storage, SQLiteStorage)
    profile = UserProfile(
        user_id="lexical-owner",
        profile_id="lexical-profile",
        content="factuniquetoken",
        last_modified_timestamp=1_800_000_000,
        generated_from_request_id="lexical-request",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        custom_features={
            "ordinaryuniquekey": "ordinaryuniquetoken",
            "nested": {KEY: "nestedordinarytoken"},
            KEY: {
                "created_at": 1_893_456_789,
                "ancestor_ids": ["ancestoruniquetoken"],
                "explicit_storage_deadline": {
                    "deadline_utc": "2030-01-01T00:00:00Z",
                    "source_quote": "permissionuniquetoken",
                },
            },
        },
        expanded_terms="expansionuniquetoken",
    )
    before = deepcopy(profile.custom_features)
    if operation == "add":
        storage.add_user_profile(profile.user_id, [profile], skip_embedding=True)
    else:
        original = profile.model_copy(
            update={"content": "oldfactuniquetoken", "custom_features": None}
        )
        storage.add_user_profile(profile.user_id, [original], skip_embedding=True)
        storage.update_user_profile_by_id(profile.user_id, profile.profile_id, profile)

    def matches(token: str) -> list[str]:
        return [
            row.profile_id
            for row in storage.search_user_profile(
                SearchUserProfileRequest(
                    user_id=profile.user_id, query=token, search_mode=SearchMode.FTS
                )
            )
        ]

    for token in [
        "factuniquetoken",
        "ordinaryuniquetoken",
        "nestedordinarytoken",
        "expansionuniquetoken",
    ]:
        assert matches(token) == [profile.profile_id]
    for token in [
        "1893456789",
        "ancestoruniquetoken",
        "permissionuniquetoken",
        "created_at",
        "oldfactuniquetoken",
        "ordinaryuniquekey",
    ]:
        assert matches(token) == []
    assert storage.get_user_profile(profile.user_id)[0].custom_features == before

    # Literal fact text is still searchable even when it repeats an operational
    # value: exclude metadata at its source rather than scrubbing matching words.
    profile.content = "factuniquetoken permissionuniquetoken"
    storage.update_user_profile_by_id(profile.user_id, profile.profile_id, profile)
    assert matches("permissionuniquetoken") == [profile.profile_id]
