"""Profile inputs shared by transferred domain regression cases."""

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.server.services.profile.components.consolidator import ProfileConsolidator
from reflexio.server.services.profile.components.extractor import ProfileExtractor
from reflexio.server.services.profile.lifecycle import new_features

NOW = 1_700_000_000
FORGET = "Forget my address, but keep that I enjoy hiking."
SURVIVOR = "I enjoy hiking."


def old_profile():
    return UserProfile(
        profile_id="old",
        user_id="u",
        content="Address is 10 Secret Street; enjoys hiking",
        source_span="My address is 10 Secret Street. I enjoy hiking.",
        last_modified_timestamp=NOW - 1,
        generated_from_request_id="earlier",
        custom_features={
            **new_features(None, now=NOW - 1, deadline=None),
            "sensitive": "10 Secret Street",
        },
        notes="10 Secret Street",
        reader_angle="old view",
    )


def new_request(quote: str | None = FORGET):
    return ProfileExtractor._convert_raw_to_user_profiles(
        [
            {
                "content": "Requested removal of address",
                "source_span": FORGET,
                "explicit_forget_quote": quote,
            }
        ],
        "u",
        "r",
        source_texts=[FORGET],
        user_source_texts=[FORGET],
    )[0]


def consolidator():
    instance = object.__new__(ProfileConsolidator)
    instance.output_pending_status = False
    instance.consolidated_output_indices = set()
    instance.lineage_sources_by_profile_id = {}
    return instance
