"""Frozen OLD profile checks through the real shared SQL infrastructure."""

import time
from dataclasses import replace

import pytest

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.extraction.profile_package_state import profile_witness
from tests.server.services.storage.test_storage_contract_extraction_stream import (
    publish,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "mutation",
    [
        "content",
        "expiry",
        "status",
        "merge",
        "supersede",
        "retired",
        "user",
        "delete",
        "source",
        "notes",
        "reader",
        "source_ids",
        "malformed",
    ],
)
def test_changed_old_profile_blocks_publication(storage, mutation):
    publish(storage, 1, width=1, stride=1)
    user, token = storage.claim_extraction("profile-witness", 300)
    window = storage.prepare_extraction(user, token)
    row = UserProfile(
        profile_id="old",
        user_id=user,
        content="A condition matters",
        last_modified_timestamp=int(time.time()),
        generated_from_request_id="r",
        custom_features={"condition": "only at work"},
        source_interaction_ids=[1],
    )
    storage.add_user_profile(user, [row], skip_embedding=True)
    witnesses = {row.profile_id: profile_witness(row)}
    with storage.commit_scope():
        storage.fence_user_extraction(user, token)
        storage.validate_extraction_historical_profiles(window, witnesses)
    changes = {
        "content": ("content", "A changed rule"),
        "expiry": ("expiration_timestamp", 1),
        "status": ("status", "pending"),
        "merge": ("merged_into", "other"),
        "supersede": ("superseded_by", "other"),
        "retired": ("retired_at", 1),
        "user": ("user_id", "other"),
        "source": ("source", "other"),
        "notes": ("notes", "Changed condition"),
        "reader": ("reader_angle", "changed angle"),
        "source_ids": ("source_interaction_ids", "[2]"),
        "malformed": ("custom_features", "[1]"),
    }
    with storage._stream_sql() as db:
        if mutation == "delete":
            db.query("DELETE FROM profiles WHERE profile_id=?", (row.profile_id,))
        else:
            column, value = changes[mutation]
            db.query(
                f"UPDATE profiles SET {column}=? WHERE profile_id=?",  # noqa: S608 -- fixed test mapping
                (value, row.profile_id),
            )
    with (
        storage.commit_scope(),
        pytest.raises(
            InputBudgetConfigurationError,
            match="Historical profile is malformed"
            if mutation == "malformed"
            else "Historical profile changed",
        ),
    ):
        storage.validate_extraction_historical_profiles(window, witnesses)


def test_profile_witness_read_normalization_and_project_scope(storage):
    publish(storage, 1, width=1, stride=1)
    user, token = storage.claim_extraction("profile-witness", 300)
    window = storage.prepare_extraction(user, token)
    row = UserProfile(
        profile_id="old",
        user_id=user,
        content="Keep the qualifier",
        last_modified_timestamp=100,
        generated_from_request_id="r",
        source=None,
        custom_features={"a": [1, "value"]},
    )
    storage.add_user_profile(user, [row], skip_embedding=True)
    witnesses = {row.profile_id: profile_witness(row)}
    storage.validate_extraction_historical_profiles(window, witnesses)
    assert profile_witness(
        row.model_copy(update={"source": "", "tags": ["derived"]})
    ) == profile_witness(row)
    with pytest.raises(InputBudgetConfigurationError, match="scope changed"):
        storage.validate_extraction_historical_profiles(
            replace(window, project_id="other"), witnesses
        )
