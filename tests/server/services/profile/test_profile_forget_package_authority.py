"""Recovery rechecks User authority without altering prepared package bytes."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.deferred_learning_plan import (
    ProfilePayloadErasure,
    ProfileWritePlan,
)
from reflexio.server.services.durable_learning.profile_package_executor import (
    ProfilePackageWindowExecutor,
)
from reflexio.server.services.extraction.input_preparation import SourceSpan
from reflexio.server.services.profile.components.extractor import ProfileExtractor
from reflexio.server.services.write_plan_codec import encode_write_plan

QUOTE = "Forget my address."


def candidate(quote: str | None = QUOTE):
    return ProfileExtractor._convert_raw_to_user_profiles(
        [{"content": "Removal request", "explicit_forget_quote": quote}],
        "u",
        "r",
        user_source_texts=[QUOTE],
    )[0]


def result(role="User", field="content"):
    spans = (SourceSpan(1, "r", role, field, 0, len(QUOTE), QUOTE),)
    return SimpleNamespace(
        gate_decision=True,
        package=SimpleNamespace(presentation=SimpleNamespace(visible_spans=spans)),
    )


@pytest.mark.parametrize("role", ["User", "Assistant", "Tool"])
@pytest.mark.parametrize(
    "field",
    [
        "content",
        "user_action_description",
        "tools_used[0]",
        "expert_content",
        "shadow_content",
    ],
)
def test_cached_converted_candidate_requires_original_user_instruction(role, field):
    profiles = [candidate()]
    before = copy.deepcopy(profiles)
    if role == "User" and field in {"content", "user_action_description"}:
        ProfilePackageWindowExecutor._validate_candidate_authority(
            profiles, result(role, field)
        )
    else:
        with pytest.raises(ValueError, match="User instruction"):
            ProfilePackageWindowExecutor._validate_candidate_authority(
                profiles, result(role, field)
            )
    assert profiles == before


@pytest.mark.parametrize("role", ["User", "Assistant", "Tool"])
@pytest.mark.parametrize(
    "defect", ["none", "missing-candidate", "different-candidate", "new-profile"]
)
def test_saved_plan_user_binding_revalidation_preserves_frozen_bytes(role, defect):
    authority = candidate()
    erasure = ProfilePayloadErasure(
        "old", "u", "r", "version", authority.profile_id, QUOTE
    )
    plan = ProfileWritePlan(
        "u",
        "r",
        [authority] if defect == "new-profile" else [],
        [],
        erasure_instructions=[erasure],
        lifecycle_epoch=1,
    )
    candidates = (
        []
        if defect in {"missing-candidate", "new-profile"}
        else [candidate(None) if defect == "different-candidate" else authority]
    )
    checkpoint = {
        "profile_package_candidates": [
            row.model_dump(mode="json") for row in candidates
        ],
        "profile_package_plan": {"write_plan": encode_write_plan(plan)},
        "profile_package_plan_digest": "frozen",
    }
    before = copy.deepcopy(checkpoint)
    executor = object.__new__(ProfilePackageWindowExecutor)
    executor._plan_digest = Mock(return_value="frozen")
    if role == "User" and defect == "none":
        executor._validate_host_binding(
            SimpleNamespace(checkpoint=checkpoint), result(role)
        )
    else:
        with pytest.raises((ValueError, InputBudgetConfigurationError), match="User"):
            executor._validate_host_binding(
                SimpleNamespace(checkpoint=checkpoint), result(role)
            )
    assert checkpoint == before


def test_legacy_frozen_plan_without_forget_authority_is_unchanged():
    row = candidate(None)
    row.custom_features = {}  # Authentic legacy rows carry no G1 metadata.
    plan = ProfileWritePlan("u", "r", [row], [])
    checkpoint = {
        "profile_package_candidates": [row.model_dump(mode="json")],
        "profile_package_plan": {"write_plan": encode_write_plan(plan)},
        "profile_package_plan_digest": "frozen",
    }
    before = copy.deepcopy(checkpoint)
    executor = object.__new__(ProfilePackageWindowExecutor)
    executor._plan_digest = Mock(return_value="frozen")
    executor._validate_host_binding(
        SimpleNamespace(checkpoint=checkpoint), result("Assistant")
    )
    assert checkpoint == before
