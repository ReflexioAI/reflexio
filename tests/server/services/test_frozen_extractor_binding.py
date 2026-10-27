"""Frozen-policy compatibility must never authorize changed work or reset history."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.models.config_schema import (
    DeduplicationConfig,
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
)
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
    matches_frozen_extractor_binding,
    matches_frozen_window_binding,
    snapshot_extractor_config,
)
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    build_scope_hash,
)


def _records(
    kind: str = "playbook",
) -> tuple[AgentRunRecord, AgentRunRecord, dict[str, Any]]:
    config_type = (
        ProfileExtractorConfig if kind == "profile" else UserPlaybookExtractorConfig
    )
    current = config_type(
        extraction_definition_prompt="Preserve explicit safeguards"
    ).model_dump(mode="json")
    if kind == "playbook":
        current["deduplication_config"] = DeduplicationConfig().model_dump(mode="json")
    frozen = {**deepcopy(current), "extractor_name": "historical-label"}
    if kind == "playbook":
        frozen["deduplication_config"]["max_unified_content_chars"] = 37
    binding = AgentBinding(
        org_id="org-1",
        extractor_kind=kind,
        user_id="user-1",
        request_id="request-1",
        agent_version="version-1",
        source="published",
        source_interaction_ids=[12, 7, 20],
        window_start_interaction_id=7,
        window_end_interaction_id=20,
        extractor_config_hash=build_scope_hash(frozen),
    )
    prior = AgentRunRecord(
        id="window:original",
        binding=binding,
        status=AgentRunStatus.FAILED,
        generation_request_snapshot={
            "extractor_config": deepcopy(frozen),
            "overflow": {
                "revision": 23,
                "attempts": [
                    {"id": f"paid-{i}", "usage": None, "model_provenance": None}
                    for i in range(5)
                ],
            },
        },
        committed_output={"saved": "original draft"},
        resume_attempts=7,
    )
    expected = replace(
        prior,
        binding=replace(binding, extractor_config_hash=build_scope_hash(current)),
        generation_request_snapshot={"extractor_config": current},
    )
    return prior, expected, frozen


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_retired_label_is_compatible_without_mutating_paid_history(kind: str) -> None:
    prior, expected, frozen = _records(kind)
    original = deepcopy((prior, expected, frozen))

    assert matches_frozen_extractor_binding(prior, expected, frozen)
    assert (prior, expected, frozen) == original
    assert prior.generation_request_snapshot["overflow"]["revision"] == 23
    assert len(prior.generation_request_snapshot["overflow"]["attempts"]) == 5


def test_exact_binding_remains_accepted_without_legacy_snapshot_witness() -> None:
    prior, _, _ = _records()
    exact = replace(prior, generation_request_snapshot={})
    assert matches_frozen_extractor_binding(exact, exact, {})


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("org_id", "org-2"),
        ("extractor_kind", "profile"),
        ("user_id", "user-2"),
        ("request_id", "request-2"),
        ("agent_version", "version-2"),
        ("source", "other-source"),
        ("source_interaction_ids", [7, 12, 20]),
        ("source_interaction_ids", [12, 7, 21]),
        ("window_start_interaction_id", 6),
        ("window_end_interaction_id", 21),
        ("extractor_config_hash", "unverified-current-hash"),
    ],
)
def test_compatibility_keeps_every_other_binding_field_strict(
    field: str, changed: Any
) -> None:
    prior, expected, frozen = _records()
    expected = replace(expected, binding=replace(expected.binding, **{field: changed}))
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize("tampered", ["saved", "window", "original_hash"])
def test_original_hash_requires_both_frozen_and_saved_witnesses(tampered: str) -> None:
    prior, expected, frozen = _records()
    if tampered == "saved":
        prior.generation_request_snapshot["extractor_config"]["extractor_name"] = (
            "other"
        )
    elif tampered == "window":
        frozen["extractor_name"] = "other"
    else:
        prior = replace(
            prior, binding=replace(prior.binding, extractor_config_hash="wrong")
        )
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize("change", ["default", "retired_other_field", "changed_prompt"])
def test_retired_label_exception_cannot_authorize_other_normalization(
    change: str,
) -> None:
    prior, expected, frozen = _records()
    current = expected.generation_request_snapshot["extractor_config"]
    if change == "default":
        current["new_runtime_default"] = 1
    elif change == "retired_other_field":
        frozen["another_retired_field"] = "old"
        prior.generation_request_snapshot["extractor_config"] = deepcopy(frozen)
        prior = replace(
            prior,
            binding=replace(
                prior.binding, extractor_config_hash=build_scope_hash(frozen)
            ),
        )
    else:
        current["extraction_definition_prompt"] = "Different extraction policy"
    expected = replace(
        expected,
        binding=replace(
            expected.binding, extractor_config_hash=build_scope_hash(current)
        ),
    )
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize(("old", "new"), [(True, 1), (1, 1.0)])
def test_canonical_comparison_does_not_confuse_equal_python_numeric_values(
    old: Any, new: Any
) -> None:
    prior, expected, frozen = _records()
    frozen["opaque_value"] = old
    prior.generation_request_snapshot["extractor_config"] = deepcopy(frozen)
    prior = replace(
        prior,
        binding=replace(prior.binding, extractor_config_hash=build_scope_hash(frozen)),
    )
    current = expected.generation_request_snapshot["extractor_config"]
    current["opaque_value"] = new
    expected = replace(
        expected,
        binding=replace(
            expected.binding, extractor_config_hash=build_scope_hash(current)
        ),
    )
    assert old == new  # Python equality alone cannot guard this boundary.
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize("remove_nested", [False, True])
def test_only_top_level_label_may_be_removed(remove_nested: bool) -> None:
    prior, expected, frozen = _records()
    frozen["opaque_metadata"] = {"extractor_name": "meaningful nested data"}
    prior.generation_request_snapshot["extractor_config"] = deepcopy(frozen)
    prior = replace(
        prior,
        binding=replace(prior.binding, extractor_config_hash=build_scope_hash(frozen)),
    )
    current = expected.generation_request_snapshot["extractor_config"]
    current["opaque_metadata"] = (
        {} if remove_nested else deepcopy(frozen["opaque_metadata"])
    )
    expected = replace(
        expected,
        binding=replace(
            expected.binding, extractor_config_hash=build_scope_hash(current)
        ),
    )
    assert (
        matches_frozen_extractor_binding(prior, expected, frozen) is not remove_nested
    )


@pytest.mark.parametrize("kind", ["aggregator", "offline_tuner"])
def test_no_compatibility_exception_for_other_components(kind: str) -> None:
    prior, expected, frozen = _records()
    prior = replace(prior, binding=replace(prior.binding, extractor_kind=kind))
    expected = replace(expected, binding=replace(expected.binding, extractor_kind=kind))
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_label",
        "missing_saved",
        "missing_current",
        "non_dict_saved",
        "non_dict_current",
    ],
)
def test_incomplete_witness_does_not_enable_compatibility(invalid: str) -> None:
    prior, expected, frozen = _records()
    if invalid == "missing_label":
        frozen.pop("extractor_name")
        prior.generation_request_snapshot["extractor_config"] = deepcopy(frozen)
        prior = replace(
            prior,
            binding=replace(
                prior.binding, extractor_config_hash=build_scope_hash(frozen)
            ),
        )
        expected = replace(
            expected,
            binding=replace(expected.binding, extractor_config_hash="different"),
        )
    elif invalid == "missing_saved":
        prior.generation_request_snapshot.pop("extractor_config")
    elif invalid == "missing_current":
        expected.generation_request_snapshot.pop("extractor_config")
    elif invalid == "non_dict_saved":
        prior.generation_request_snapshot["extractor_config"] = []
    else:
        expected.generation_request_snapshot["extractor_config"] = []
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize("retired_label", [False, True])
def test_missing_dedup_strategy_means_only_legacy(retired_label: bool) -> None:
    prior, expected, frozen = _records()
    frozen["deduplication_config"].pop("strategy")
    if not retired_label:
        frozen.pop("extractor_name")
    prior = replace(
        prior,
        binding=replace(prior.binding, extractor_config_hash=build_scope_hash(frozen)),
        generation_request_snapshot={"extractor_config": deepcopy(frozen)},
    )
    original = deepcopy((prior, expected, frozen))
    assert matches_frozen_extractor_binding(prior, expected, frozen)
    assert (prior, expected, frozen) == original

    current = expected.generation_request_snapshot["extractor_config"]
    current["deduplication_config"]["strategy"] = "audited"
    expected = replace(
        expected,
        binding=replace(
            expected.binding, extractor_config_hash=build_scope_hash(current)
        ),
    )
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize(
    "change", ["unknown", "threshold", "null_strategy", "removed_strategy"]
)
def test_nested_dedup_policy_changes_are_not_schema_compatibility(change: str) -> None:
    prior, expected, frozen = _records()
    if change == "unknown":
        frozen["deduplication_config"]["retired_unknown"] = "must retain"
    elif change == "threshold":
        frozen["deduplication_config"]["search_threshold"] = 0.9
    elif change == "null_strategy":
        frozen["deduplication_config"]["strategy"] = None
    else:
        expected.generation_request_snapshot["extractor_config"][
            "deduplication_config"
        ].pop("strategy")
        expected = replace(
            expected,
            binding=replace(
                expected.binding,
                extractor_config_hash=build_scope_hash(
                    expected.generation_request_snapshot["extractor_config"]
                ),
            ),
        )
    prior = replace(
        prior,
        binding=replace(prior.binding, extractor_config_hash=build_scope_hash(frozen)),
        generation_request_snapshot={"extractor_config": deepcopy(frozen)},
    )
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize("with_label", [False, True])
@pytest.mark.parametrize("with_strategy", [False, True])
@pytest.mark.parametrize("with_cap", [False, True])
def test_length_warning_retirement_composes_only_with_known_transitions(
    with_label, with_strategy, with_cap
):
    prior, expected, frozen = _records()
    if not with_label:
        frozen.pop("extractor_name")
    if not with_strategy:
        frozen["deduplication_config"].pop("strategy")
    if not with_cap:
        frozen["deduplication_config"].pop("max_unified_content_chars")
    saved = deepcopy(prior.generation_request_snapshot)
    saved["extractor_config"] = deepcopy(frozen)
    prior = replace(
        prior,
        binding=replace(prior.binding, extractor_config_hash=build_scope_hash(frozen)),
        generation_request_snapshot=saved,
    )
    before = deepcopy((prior, expected, frozen))
    assert matches_frozen_extractor_binding(prior, expected, frozen)
    assert (prior, expected, frozen) == before


def test_retired_warning_value_still_requires_original_witness_hash():
    prior, expected, frozen = _records()
    frozen["deduplication_config"]["max_unified_content_chars"] += 1
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize("retired_context", [None, "Historical playbook instructions"])
def test_retired_playbook_context_keeps_frozen_binding(
    retired_context: str | None,
) -> None:
    prior, expected, frozen = _records("playbook")
    frozen["context_prompt"] = retired_context
    prior = replace(
        prior,
        binding=replace(prior.binding, extractor_config_hash=build_scope_hash(frozen)),
        generation_request_snapshot={
            **prior.generation_request_snapshot,
            "extractor_config": deepcopy(frozen),
        },
    )
    before = deepcopy((prior, expected, frozen))
    assert matches_frozen_extractor_binding(prior, expected, frozen)
    assert (prior, expected, frozen) == before


def test_live_profile_context_still_invalidates_frozen_binding() -> None:
    prior, expected, frozen = _records("profile")
    frozen["context_prompt"] = "Customer profile policy must remain strict"
    prior = replace(
        prior,
        binding=replace(prior.binding, extractor_config_hash=build_scope_hash(frozen)),
        generation_request_snapshot={"extractor_config": deepcopy(frozen)},
    )
    assert not matches_frozen_extractor_binding(prior, expected, frozen)


@pytest.mark.parametrize("witness", ["snapshot", "window"])
@pytest.mark.parametrize("changed", [None, "malformed", {"context_prompt": "tampered"}])
def test_equal_parent_binding_authenticates_identity_witnesses(
    witness, changed
) -> None:
    from reflexio.server.services.extraction.resumable_agent import (
        _require_overflow_parent,
    )

    prior, _, frozen = _records()
    storage = Mock()
    storage.get_agent_run.return_value = prior
    coordinator = Mock()
    coordinator.window = SimpleNamespace(
        window_id="original", project_id="", policy={"extractor": deepcopy(frozen)}
    )
    assert _require_overflow_parent(storage, coordinator, prior) == prior
    if witness == "snapshot":
        prior.generation_request_snapshot["extractor_config"] = changed
    else:
        coordinator.window.policy["extractor"] = changed
    with pytest.raises(RuntimeError, match="child parent binding changed"):
        _require_overflow_parent(storage, coordinator, prior)


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_first_parent_created_from_retired_label_window_keeps_its_binding(kind):
    from reflexio.server.services.extraction.resumable_agent import (
        _require_overflow_parent,
    )

    config_type = (
        ProfileExtractorConfig if kind == "profile" else UserPlaybookExtractorConfig
    )
    frozen = snapshot_extractor_config(
        config_type(extraction_definition_prompt="Preserve explicit safeguards")
    )
    frozen["extractor_name"] = "historical-label"
    current = config_type.model_validate(frozen)
    parent = build_extractor_agent_run_record(
        org_id="org-1",
        extractor_kind=kind,
        user_id="user-1",
        request_id="request-1",
        agent_version=None,
        source="published",
        request_interaction_data_models=[],
        extractor_config=current,
        service_config=SimpleNamespace(frozen_extractor_config=frozen),
        agent_context="",
    )
    parent = replace(parent, id="window:original")
    assert (
        "extractor_name" not in parent.generation_request_snapshot["extractor_config"]
    )
    storage = Mock()
    storage.get_agent_run.return_value = parent
    coordinator = Mock()
    coordinator.window = SimpleNamespace(
        window_id="original", project_id="", policy={"extractor": frozen}
    )
    before = deepcopy((parent, frozen))
    assert _require_overflow_parent(storage, coordinator, parent) == parent
    assert (parent, frozen) == before


@pytest.mark.parametrize(
    "change",
    [
        "unknown_window_field",
        "behavior",
        "saved_hash",
        "saved_missing",
        "saved_label",
        "missing_window",
        "kind",
        "org",
        "user",
        "request",
        "version",
        "source",
        "source_order",
        "project",
    ],
)
def test_normalized_parent_window_compatibility_refuses_other_changes(change):
    prior, _, _ = _records("profile")
    saved = deepcopy(prior.generation_request_snapshot["extractor_config"])
    saved.pop("extractor_name")
    prior = replace(
        prior,
        binding=replace(prior.binding, extractor_config_hash=build_scope_hash(saved)),
        generation_request_snapshot={"extractor_config": saved},
        project_id="project-1",
    )
    expected = deepcopy(prior)
    frozen = {**deepcopy(saved), "extractor_name": "historical-label"}
    project_id = "project-1"
    if change == "unknown_window_field":
        frozen["unexpected"] = True
    elif change == "behavior":
        frozen["extraction_definition_prompt"] = "Changed policy"
    elif change == "saved_hash":
        prior = replace(
            prior, binding=replace(prior.binding, extractor_config_hash="corrupt")
        )
    elif change == "saved_missing":
        prior = replace(prior, generation_request_snapshot={})
    elif change == "saved_label":
        saved["extractor_name"] = "other-label"
        prior = replace(
            prior,
            binding=replace(
                prior.binding, extractor_config_hash=build_scope_hash(saved)
            ),
        )
    elif change == "missing_window":
        frozen = None
    elif change == "project":
        project_id = "project-2"
    else:
        field, value = {
            "kind": ("extractor_kind", "playbook"),
            "org": ("org_id", "org-2"),
            "user": ("user_id", "user-2"),
            "request": ("request_id", "request-2"),
            "version": ("agent_version", "version-2"),
            "source": ("source", "other"),
            "source_order": ("source_interaction_ids", [7, 12, 20]),
        }[change]
        expected = replace(
            expected, binding=replace(expected.binding, **{field: value})
        )
    assert not matches_frozen_window_binding(
        prior, expected, frozen, project_id=project_id
    )
