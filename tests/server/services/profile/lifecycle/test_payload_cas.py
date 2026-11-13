"""Exact source-payload projection and native representation normalization."""

import json

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.server.services.profile.lifecycle_deadlines import (
    DERIVED_FIELDS,
    PAYLOAD_FIELDS,
    profile_version,
)

from .helpers import old_profile


def test_payload_projection_covers_native_schema_without_search_sidecars():
    assert set(PAYLOAD_FIELDS) == set(UserProfile.model_fields) - DERIVED_FIELDS
    assert {"embedding", "expanded_terms"} == DERIVED_FIELDS
    assert len(PAYLOAD_FIELDS) == len(set(PAYLOAD_FIELDS))


def test_row_json_enum_and_nullable_array_normalization_matches_native_model():
    model = old_profile().model_copy(
        update={
            "extractor_names": ["extractor"],
            "tags": ["tag"],
            "source_interaction_ids": [],
        }
    )
    row = model.model_dump(mode="json")
    for field in ["custom_features", "extractor_names", "tags"]:
        row[field] = json.dumps(row[field])
    row["source_interaction_ids"] = None
    assert profile_version(row) == profile_version(model.model_dump(mode="python"))
    assert profile_version(row) == profile_version(model.model_dump(mode="json"))
