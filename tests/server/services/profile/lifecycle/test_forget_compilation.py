"""Forgetting compilation preserves preview and exact-target publication duties."""

import pytest

from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.profile.components.consolidator import (
    ProfileDeduplicationOutput,
    ProfileDeletionDirective,
    ProfileDuplicateGroup,
)

from .helpers import SURVIVOR, consolidator, new_request, old_profile


@pytest.mark.parametrize("partial", [False, True])
def test_pending_preview_retains_survivors_without_compiling_current_erasure(partial):
    instance = consolidator()
    instance.output_pending_status = True
    old = old_profile()
    old.status = Status.PENDING
    output = ProfileDeduplicationOutput(
        duplicate_groups=[
            ProfileDuplicateGroup(
                item_ids=["NEW-0", "EXISTING-0"],
                merged_content="Enjoys hiking",
                erase_replaced_payloads=True,
                retained_source_quote=SURVIVOR,
            )
        ]
        if partial
        else [],
        deletions=[]
        if partial
        else [
            ProfileDeletionDirective(
                new_id="NEW-0",
                existing_ids=["EXISTING-0"],
                reasoning="Forget whole profile",
            )
        ],
    )
    rows, retired, _ = instance._build_deduplicated_results(
        [new_request()], [old], output, "u", "r"
    )
    assert retired == [old.profile_id]
    assert instance.erasure_instructions == []
    assert [p.content for p in rows] == (["Enjoys hiking"] if partial else [])


def test_two_source_directives_for_same_exact_target_compile_one_erasure():
    instance = consolidator()
    instance.output_pending_status = False
    first, second = new_request(), new_request()
    second.profile_id = "another-source"
    output = ProfileDeduplicationOutput(
        deletions=[
            ProfileDeletionDirective(
                new_id=f"NEW-{i}",
                existing_ids=["EXISTING-0"],
                reasoning="Same exact requested target",
            )
            for i in range(2)
        ]
    )
    rows, retired, _ = instance._build_deduplicated_results(
        [first, second], [old_profile()], output, "u", "r"
    )
    assert rows == [] and retired == ["old"]
    assert len(instance.erasure_instructions) == 1
    assert instance.erasure_instructions[0].source_profile_id == first.profile_id


def test_different_versions_of_same_target_remain_a_conflict():
    instance = consolidator()
    instance.output_pending_status = False
    instance.erasure_instructions = []
    old, authority = old_profile(), new_request()
    instance._authorize_payload_erasure(old, authority, "u", "r")
    old.content = "Concurrent replacement"
    with pytest.raises(ValueError, match="version"):
        instance._authorize_payload_erasure(old, authority, "u", "r")
    assert len(instance.erasure_instructions) == 1
