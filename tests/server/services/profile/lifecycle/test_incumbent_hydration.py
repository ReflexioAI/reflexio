"""Selected native hybrid candidates gain authoritative witnesses before model duty."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.profile.components.consolidator import (
    ProfileDeduplicationOutput,
    ProfileDuplicateGroup,
)
from reflexio.server.services.profile.lifecycle_deadlines import profile_version

from .helpers import NOW, SURVIVOR, consolidator, new_request, old_profile

# Actual frozen hybrid_match_profiles return payload, excluding score sidecars.
HYBRID_PAYLOAD = {
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
    "source_span",
    "notes",
    "reader_angle",
    "tags",
}


def hybrid_projection(profile):
    raw = profile.model_dump(mode="json", include=HYBRID_PAYLOAD)
    # Native converter supplies these defaults for omitted SQL return columns.
    raw["source_interaction_ids"] = []
    raw["merged_into"] = None
    raw["superseded_by"] = None
    return UserProfile.model_validate(raw)


class ByIDStorage:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def get_profiles_by_ids(self, user_id, ids, *, status_filter):
        self.calls.append((user_id, list(ids), status_filter))
        return list(self.rows)


def method_for(storage, *, pending=False):
    method = cast(Any, consolidator())
    method.request_context = SimpleNamespace(storage=storage, org_id="org")
    method.output_pending_status = pending
    return method


def test_actual_partial_hybrid_shape_hydrates_full_native_source_ids_and_lineage_in_order():
    first = old_profile().model_copy(
        update={"source_interaction_ids": [1], "merged_into": "old-parent"}
    )
    second = old_profile().model_copy(
        update={"profile_id": "second", "source_interaction_ids": [2]}
    )
    partial = [hybrid_projection(second), hybrid_projection(first)]
    assert partial[0].source_interaction_ids == partial[1].source_interaction_ids == []
    storage = ByIDStorage([first, second])  # SQL row order need not match ranking.
    hydrated = method_for(storage)._hydrate_selected_profiles(partial, "u")
    assert [p.profile_id for p in hydrated] == ["second", "old"]
    assert [p.source_interaction_ids for p in hydrated] == [[2], [1]]
    assert hydrated[1].merged_into == "old-parent"
    assert storage.calls == [("u", ["second", "old"], [None])]
    assert partial[0].source_interaction_ids == []  # no mutation of saved search rows


def test_empty_search_does_not_broaden_retrieval_or_issue_a_read():
    storage = ByIDStorage([old_profile()])
    assert method_for(storage)._hydrate_selected_profiles([], "u") == []
    assert storage.calls == []


@pytest.mark.parametrize(
    "bad", ["missing", "duplicate", "extra", "foreign", "noncurrent"]
)
def test_bad_authoritative_rows_refuse_instead_of_replacing_selected_scope(bad):
    old = old_profile()
    rows = [old]
    if bad == "missing":
        rows = []
    if bad == "duplicate":
        rows = [old, old]
    if bad == "extra":
        rows = [old, old.model_copy(update={"profile_id": "unseen"})]
    if bad == "foreign":
        rows = [old.model_copy(update={"user_id": "other"})]
    if bad == "noncurrent":
        rows = [old.model_copy(update={"status": Status.SUPERSEDED})]
    with pytest.raises(ValueError, match="missing|identity, owner or status"):
        method_for(ByIDStorage(rows))._hydrate_selected_profiles(
            [hybrid_projection(old)], "u"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "changed meaning"),
        ("notes", "changed source notes"),
        ("reader_angle", "new interpretation"),
        ("last_modified_timestamp", NOW + 1),
        ("generated_from_request_id", "later"),
    ],
)
def test_search_observed_meaning_and_version_cannot_change_during_hydration(
    field, value
):
    old = old_profile()
    changed = old.model_copy(update={field: value})
    with pytest.raises(ValueError, match="meaning or version changed"):
        method_for(ByIDStorage([changed]))._hydrate_selected_profiles(
            [hybrid_projection(old)], "u"
        )


def test_nondefault_source_ids_from_fuller_search_are_observations_too():
    observed = old_profile().model_copy(update={"source_interaction_ids": [7]})
    full = observed.model_copy(update={"source_interaction_ids": [8]})
    with pytest.raises(ValueError, match="meaning or version changed"):
        method_for(ByIDStorage([full]))._hydrate_selected_profiles([observed], "u")


def test_preview_hydration_keeps_existing_pending_scope_without_touching_current_rows():
    pending = old_profile().model_copy(update={"status": Status.PENDING})
    storage = ByIDStorage([pending])
    assert method_for(storage, pending=True)._hydrate_selected_profiles(
        [pending], "u"
    ) == [pending]
    assert storage.calls == [("u", ["old"], [Status.PENDING])]


def test_native_deduplicate_renders_and_compiles_full_snapshot_before_partial_erasure(
    monkeypatch,
):
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)
    old = old_profile().model_copy(update={"source_interaction_ids": [1]})
    partial = hybrid_projection(old)
    assert profile_version(partial.model_dump(mode="json")) != profile_version(
        old.model_dump(mode="json")
    )
    storage = ByIDStorage([old])
    method = method_for(storage)
    captured = []
    monkeypatch.setattr(method, "_retrieve_existing_profiles", lambda *_args: [partial])

    def render(_new, existing):
        captured.append(existing)
        assert existing[0].source_interaction_ids == [1]
        return "pure scripted consolidation duty"

    monkeypatch.setattr(method, "_render_dedup_prompt", render)
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
    monkeypatch.setattr(
        method,
        "_call_native_dedup",
        lambda **_kwargs: SimpleNamespace(value=output, provenance=None),
    )
    rows, retired, _ = method.deduplicate([new_request()], "u", "r")
    assert captured == [[old]]
    assert method.erasure_instructions[0].expected_version == profile_version(
        old.model_dump(mode="json")
    )
    assert storage.calls == [("u", ["old"], [None])]


@pytest.mark.parametrize("bad", ["duplicate", "foreign", "noncurrent", "empty_id"])
def test_invalid_selected_scope_refuses_before_extra_read(bad):
    old = old_profile()
    selected = [old]
    if bad == "duplicate":
        selected = [old, old]
    if bad == "foreign":
        selected = [old.model_copy(update={"user_id": "other"})]
    if bad == "noncurrent":
        selected = [old.model_copy(update={"status": Status.SUPERSEDED})]
    if bad == "empty_id":
        selected = [old.model_copy(update={"profile_id": ""})]
    storage = ByIDStorage([old])
    with pytest.raises(ValueError, match="identity, owner or status"):
        method_for(storage)._hydrate_selected_profiles(selected, "u")
    assert storage.calls == []


def test_hydration_refusal_precedes_native_model_duty(monkeypatch):
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)
    old = old_profile()
    storage = ByIDStorage([old.model_copy(update={"notes": "changed since search"})])
    method = method_for(storage)
    monkeypatch.setattr(
        method, "_retrieve_existing_profiles", lambda *_args: [hybrid_projection(old)]
    )

    def forbidden_model_call(**_kwargs):
        pytest.fail("Hydration mismatch must fail before model dispatch")

    monkeypatch.setattr(method, "_call_native_dedup", forbidden_model_call)
    with pytest.raises(ValueError, match="meaning or version changed"):
        method.deduplicate([new_request()], "u", "r")
