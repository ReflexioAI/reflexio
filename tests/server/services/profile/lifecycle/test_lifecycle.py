"""Pure native policy tests: no services, model calls or storage connections."""

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.common import NEVER_EXPIRES_TIMESTAMP
from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.server.services.profile import lifecycle as policy
from reflexio.server.services.profile.components.consolidator import (
    ProfileConsolidator,
    ProfileDeduplicationOutput,
    ProfileDuplicateGroup,
)
from reflexio.server.services.profile.components.extractor import ProfileExtractor
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileAddItem,
    StructuredProfilesOutput,
)
from reflexio.server.services.retrieval.recency import (
    RecencyConfig,
    ScoredItem,
    decay_for_item,
)

DAY = 86400
T0 = 1700000000
QUOTE = "Stop storing this memory at 2024-01-01T00:00:00Z."
DEADLINE = {"deadline_utc": "2024-01-01T00:00:00Z", "source_quote": QUOTE}


def profile(identity="p", created=T0, deadline=None, owner="u"):
    features = policy.new_features(
        None,
        now=created,
        deadline=policy.ExplicitStorageDeadline.model_validate(deadline)
        if deadline
        else None,
        source_texts=[QUOTE],
    )
    return UserProfile(
        profile_id=identity,
        user_id=owner,
        content="likes Python",
        last_modified_timestamp=created,
        generated_from_request_id="r",
        custom_features=features,
        expiration_timestamp=policy.expiration_for(features),
    )


def merge(old, new):
    features = policy.merge_features(
        [new, old], inherited_features=new.custom_features, existing_profiles=[old]
    )
    return new.model_copy(
        update={
            "profile_id": "merged",
            "custom_features": features,
            "expiration_timestamp": policy.expiration_for(features),
        }
    )


def test_model_schemas_remove_guessed_lifetime_and_decline_legacy_fields():
    assert "time_to_live" not in ProfileAddItem.model_json_schema()["properties"]
    assert (
        "merged_time_to_live"
        not in ProfileDuplicateGroup.model_json_schema()["properties"]
    )
    assert (
        "explicit_storage_deadline" in ProfileAddItem.model_json_schema()["properties"]
    )
    with pytest.raises(ValidationError, match="estimated lifetime"):
        ProfileAddItem.model_validate(
            {"content": "ordinary fact", "time_to_live": "infinity"}
        )
    with pytest.raises(ValidationError, match="estimated lifetime"):
        ProfileDuplicateGroup.model_validate(
            {
                "item_ids": ["NEW-0"],
                "merged_content": "fact",
                "merged_time_to_live": "infinity",
            }
        )
    output = StructuredProfilesOutput(
        profiles=[ProfileAddItem(content="ordinary fact")]
    )
    assert output.profiles is not None
    assert output.profiles[0].explicit_storage_deadline is None


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-01",
        "2024-01-01T00:00:00+00:00",
        "2024-01-01T00:00:00.1Z",
        "2024-02-31T00:00:00Z",
        "2024-1-1T00:00:00Z",
    ],
)
def test_deadlines_require_exact_unambiguous_utc(value):
    with pytest.raises(ValidationError):
        policy.ExplicitStorageDeadline(deadline_utc=value, source_quote=QUOTE)


def test_quote_membership_and_native_builder_cannot_invent_permission(monkeypatch):
    deadline = policy.ExplicitStorageDeadline.model_validate(DEADLINE)
    with pytest.raises(ValueError, match="absent from original source"):
        policy.new_features(
            None, now=T0, deadline=deadline, source_texts=["Project due January 1"]
        )

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(T0, tz or UTC)

    monkeypatch.setattr(
        "reflexio.server.services.profile.components.extractor.datetime", Clock
    )
    result = ProfileExtractor._convert_raw_to_user_profiles(
        [{"content": "fact", "explicit_storage_deadline": DEADLINE}],
        "u",
        "r",
        source_texts=[QUOTE],
    )[0]
    assert policy.metadata(result)["created_at"] == T0
    assert result.profile_time_to_live == ProfileTimeToLive.INFINITY
    assert result.expiration_timestamp == deadline.epoch - 1
    with pytest.raises(ValueError, match="absent from original source"):
        ProfileExtractor._convert_raw_to_user_profiles(
            [{"content": "fact", "explicit_storage_deadline": DEADLINE}], "u", "r"
        )
    ordinary = ProfileExtractor._convert_raw_to_user_profiles(
        [{"content": "event due January 1"}], "u", "r"
    )[0]
    assert ordinary.expiration_timestamp == NEVER_EXPIRES_TIMESTAMP


def test_model_cannot_author_its_own_creation_clock():
    with pytest.raises(ValueError, match="cannot author lifecycle clocks"):
        policy.new_features({policy.KEY: {"created_at": 0}}, now=T0, deadline=None)


def test_merge_preserves_attention_and_earliest_permission_through_multiple_generations():
    old = profile("old", deadline=DEADLINE)
    newer = profile("new", T0 + DAY)
    first = merge(old, newer)
    second = merge(first, profile("fresh", T0 + 2 * DAY))
    assert policy.metadata(second)["created_at"] == T0
    assert policy.metadata(second)["ancestor_ids"] == ["merged", "old"]
    assert second.expiration_timestamp == old.expiration_timestamp
    assert policy.metadata(second)["explicit_storage_deadline"] == DEADLINE
    assert decay_for_item(
        second, entity_type="profiles", now=T0 + 30 * DAY
    ) == pytest.approx(0.5)
    # Reading an ancestor survives replacement without modifying source truth.
    with policy.attention_context({"old": T0 + 20 * DAY}):
        assert policy.attention_epoch(second, now=T0 + 30 * DAY) == T0 + 20 * DAY


def test_later_explicit_deadline_does_not_extend_existing_permission():
    old = profile("old", deadline=DEADLINE)
    later = {**DEADLINE, "deadline_utc": "2025-01-01T00:00:00Z"}
    new = profile("new", T0 + DAY, later)
    assert merge(old, new).expiration_timestamp == old.expiration_timestamp


def test_native_consolidator_builder_preserves_lifecycle_without_ttl_authoring():
    consolidator = object.__new__(ProfileConsolidator)
    consolidator.consolidated_output_indices = set()
    consolidator.lineage_sources_by_profile_id = {}
    old = profile("old", deadline=DEADLINE)
    new = profile("new", T0 + DAY)
    output = ProfileDeduplicationOutput(
        duplicate_groups=[
            ProfileDuplicateGroup(
                item_ids=["NEW-0", "EXISTING-0"],
                merged_content="likes Python; timezone UTC",
            )
        ]
    )
    rows, retired, _ = consolidator._build_deduplicated_results(
        [new], [old], output, "u", "r"
    )
    assert len(rows) == 1 and retired == ["old"]
    assert policy.metadata(rows[0])["created_at"] == T0
    assert rows[0].expiration_timestamp == old.expiration_timestamp
    rendered = consolidator._format_profiles_with_prefix([old], "EXISTING")
    assert "TTL" not in rendered and "infinity" not in rendered


@pytest.mark.parametrize(
    "offset,expected", [(-1, []), (0, ["merged"]), (1, ["merged"])]
)
def test_deadline_equality_and_historical_identity_planning(offset, expected):
    item = merge(profile("old", deadline=DEADLINE), profile("new", T0 + DAY))
    due = policy.ExplicitStorageDeadline.model_validate(DEADLINE).epoch
    assert policy.due_profile_ids([item], now=due + offset) == expected
    # Native inclusive expiry filters must exclude equality for this permission.
    assert (item.expiration_timestamp >= due + offset) == (offset < 0)


def test_access_is_separate_per_arm_and_audits_do_not_mutate_it():
    item = profile()
    before = item.model_dump()
    supplied = {"p": T0 + 29 * DAY}
    with policy.attention_context(supplied):
        supplied["p"] = T0  # snapshot is immutable from the caller's perspective
        assert decay_for_item(item, entity_type="profiles", now=T0 + 30 * DAY) > 0.97
        with policy.attention_context({}):
            assert decay_for_item(
                item, entity_type="profiles", now=T0 + 30 * DAY
            ) == pytest.approx(0.5)
        assert decay_for_item(item, entity_type="profiles", now=T0 + 30 * DAY) > 0.97
    assert item.model_dump() == before
    assert decay_for_item(
        item, entity_type="profiles", now=T0 + 30 * DAY
    ) == pytest.approx(0.5)


def test_missing_or_future_creation_cannot_fall_back_to_modification():
    item = profile().model_copy(
        update={"custom_features": None, "last_modified_timestamp": T0}
    )
    # Legacy compatibility is neutral; modification does not invent an origin.
    assert decay_for_item(item, entity_type="profiles", now=T0) == 1.0
    assert policy.metadata(item)["created_at"] is None
    with pytest.raises(ValueError, match="after the probe"):
        decay_for_item(profile(created=T0 + 1), entity_type="profiles", now=T0)
    with (
        policy.attention_context({"p": T0 + 1}),
        pytest.raises(ValueError, match="after the probe"),
    ):
        decay_for_item(profile(), entity_type="profiles", now=T0)


def test_storage_permission_and_modification_do_not_change_attention_half_life():
    original = profile()
    mutated = original.model_copy(
        update={
            "expiration_timestamp": T0 + DAY,
            "last_modified_timestamp": T0 + 29 * DAY,
        }
    )
    assert (
        decay_for_item(original, entity_type="profiles", now=T0 + 30 * DAY)
        == decay_for_item(mutated, entity_type="profiles", now=T0 + 30 * DAY)
        == pytest.approx(0.5)
    )


def test_real_native_bounded_score_seam_uses_attention_before_top_k(monkeypatch):
    from reflexio.server.services import unified_search_service as unified

    old = profile("old")
    fresh = profile("fresh", T0 + 120 * DAY)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(T0 + 120 * DAY, tz or UTC)

    monkeypatch.setattr(unified, "datetime", Clock)
    scores = [ScoredItem(old, 1.0), ScoredItem(fresh, 0.95)]
    cfg = RecencyConfig()
    assert unified._apply_combined_score_recency(
        scores, entity_type="profiles", top_k=1, cfg=cfg
    ) == [fresh]
    with policy.attention_context({"old": T0 + 120 * DAY}):
        assert unified._apply_combined_score_recency(
            scores, entity_type="profiles", top_k=1, cfg=cfg
        ) == [old]


def test_active_prompts_have_no_guessed_lifetime_or_rendered_ttl():
    bank = Path(policy.__file__).resolve().parents[2] / "prompt/prompt_bank"
    for relative in [
        "profile_update_instruction_start/v1.4.0.prompt.md",
        "profile_deduplication/v1.2.0.prompt.md",
    ]:
        text = (bank / relative).read_text()
        assert "active: true" in text
        assert "time_to_live" not in text and "TTL" not in text
    extraction = (
        bank / "profile_update_instruction_start/v1.4.0.prompt.md"
    ).read_text()
    assert "Requested removal of" in extraction
    assert "explicit_storage_deadline" in extraction


def test_cross_owner_lifecycle_merge_and_census_are_refused():
    a, b = profile(), profile("b", owner="someone_else")
    with pytest.raises(ValueError, match="crossed profile owners"):
        policy.merge_features([a, b], inherited_features=None, existing_profiles=[a])
    with pytest.raises(ValueError, match="crossed profile owners"):
        policy.due_profile_ids([a, b], now=T0)


def test_attention_ancestry_is_not_deletion_authority():
    from reflexio.server.services.profile.lifecycle_deadlines import job_for

    old = profile("old")
    replacement = merge(old, profile("new", T0 + DAY, DEADLINE))
    due = policy.ExplicitStorageDeadline.model_validate(DEADLINE).epoch
    assert policy.due_profile_ids([old, replacement], now=due) == ["merged"]
    assert job_for(old.model_dump()) is None


def test_partial_forget_does_not_copy_forgotten_payload_in_inherited_permission():
    from reflexio.server.services.profile.lifecycle_deadlines import job_for

    from .helpers import SURVIVOR, consolidator, new_request, old_profile

    secret = "10 Secret Street"
    quote = f"Stop storing my address {secret} at 2033-05-18T03:33:20Z."
    deadline = policy.ExplicitStorageDeadline(
        deadline_utc="2033-05-18T03:33:20Z", source_quote=quote
    )
    old = old_profile()
    old.custom_features = policy.new_features(
        None,
        now=T0 - 1,
        deadline=deadline,
        source_texts=[quote],
    )
    old.expiration_timestamp = policy.expiration_for(old.custom_features)
    output = ProfileDeduplicationOutput(
        duplicate_groups=[
            ProfileDuplicateGroup(
                item_ids=["NEW-0", "EXISTING-0"],
                merged_content="Enjoys hiking",
                erase_replaced_payloads=True,
                retained_source_quote=SURVIVOR,
            )
        ]
    )
    rows, retired, _ = consolidator()._build_deduplicated_results(
        [new_request()], [old], output, "u", "r"
    )
    replacement = rows[0]
    assert retired == [old.profile_id]
    assert replacement.source_span == SURVIVOR
    assert secret not in replacement.model_dump_json()
    permission = policy.metadata(replacement)["explicit_storage_deadline"]
    assert permission == {
        "deadline_utc": deadline.deadline_utc,
        "source_quote_sha256": sha256(quote.encode("utf-8")).hexdigest(),
    }
    assert replacement.expiration_timestamp == deadline.epoch - 1
    job = job_for(replacement.model_dump())
    assert job is not None and job.due == deadline.epoch

    next_generation = merge(replacement, profile("ordinary", T0 + DAY))
    assert secret not in next_generation.model_dump_json()
    assert policy.metadata(next_generation)["explicit_storage_deadline"] == permission
    assert next_generation.expiration_timestamp == replacement.expiration_timestamp
    next_job = job_for(next_generation.model_dump())
    assert next_job is not None and next_job.due == deadline.epoch

    # A stored proof reference cannot be authored as fresh source permission.
    with pytest.raises(ValidationError):
        ProfileAddItem.model_validate(
            {"content": "Invented permission", "explicit_storage_deadline": permission}
        )
