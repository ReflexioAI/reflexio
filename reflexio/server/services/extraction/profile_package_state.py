"""Exact current-profile witnesses for frozen source-package decisions.

Embedding vectors and derived search/tag data are not decision inputs. Preserve
the content, lifetime, ownership and provenance fields the consolidator uses.
The caller owns authorization and holds the shared publication transaction.
"""

import hashlib
import json
from typing import Any

from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server.services.storage.storage_base import canonical_json

PROFILE_WITNESS_FIELDS = (
    "profile_id",
    "user_id",
    "content",
    "last_modified_timestamp",
    "generated_from_request_id",
    "profile_time_to_live",
    "expiration_timestamp",
    "custom_features",
    "source",
    "status",
    "extractor_names",
    "source_interaction_ids",
    "source_span",
    "notes",
    "reader_angle",
    "merged_into",
    "superseded_by",
)


def profile_witness(profile: UserProfile) -> str:
    """Use the same semantic representation for SQL and existing read models."""
    values = profile.model_dump(mode="json", include=set(PROFILE_WITNESS_FIELDS))
    # SQLite reads normalize NULL source to empty text; neither denotes a producer.
    values["source"] = values["source"] or ""
    return hashlib.sha256(canonical_json(values).encode()).hexdigest()


def profile_row_witness(row: dict[str, Any]) -> str:
    """Normalize SQLite JSON text and PostgreSQL JSON values without clipping."""
    values = dict(row)
    for field in ("custom_features", "extractor_names", "source_interaction_ids"):
        if isinstance(values[field], str):
            values[field] = json.loads(values[field])
    values["source_interaction_ids"] = values["source_interaction_ids"] or []
    return profile_witness(UserProfile.model_validate(values))
