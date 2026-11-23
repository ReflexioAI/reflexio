"""Current geometry selectors cannot rewrite frozen v1 job/hash identities."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from reflexio.models.config_schema import (
    ProfileExtractorConfig,
    UserPlaybookExtractorConfig,
)
from reflexio.server.services.extraction.agent_run_records import (
    effective_frozen_extractor_config,
    frozen_extractor_config_identity,
    matches_frozen_extractor_config,
    snapshot_extractor_config,
)
from reflexio.server.services.extraction.resume_worker import _log_config_hash_drift
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    build_scope_hash,
)

_PROFILE_V16 = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures/profile_execution_v16_contract.json"
    ).read_text()
)


@pytest.mark.unit
@pytest.mark.parametrize("case", _PROFILE_V16["cases"], ids=lambda case: case["id"])
def test_retired_profile_admission_retains_actual_predecessor_snapshot_and_lookup(case):
    assert _PROFILE_V16["source_commit"] == "d6b0091d27332d81d3edbeeaf7a314fb235edb81"
    frozen = deepcopy(case["snapshot"])
    before = deepcopy(frozen)
    original_hash = case["original_sha256"]
    assert build_scope_hash(frozen) == original_hash
    parsed = ProfileExtractorConfig.model_validate(
        effective_frozen_extractor_config(frozen, extractor_kind="profile")
    )
    continued = snapshot_extractor_config(parsed, frozen=frozen)
    assert continued == before == frozen
    assert matches_frozen_extractor_config(
        frozen, continued, extractor_kind="profile", original_hash=original_hash
    )
    current = {
        key: value for key, value in continued.items() if key != "manual_trigger"
    }
    assert matches_frozen_extractor_config(
        frozen, current, extractor_kind="profile", original_hash=original_hash
    )
    assert (
        frozen_extractor_config_identity(parsed, frozen)
        == case["prior_answer_identity"]
    )
    assert "manual_trigger" not in snapshot_extractor_config(parsed)
    manual = frozen["manual_trigger"]
    assert not matches_frozen_extractor_config(
        frozen,
        {**continued, "manual_trigger": not manual},
        extractor_kind="profile",
        original_hash=original_hash,
    )
    assert not matches_frozen_extractor_config(
        {**frozen, "manual_trigger": not manual},
        continued,
        extractor_kind="profile",
        original_hash=original_hash,
    )


@pytest.mark.unit
@pytest.mark.parametrize("manual", [None, "false", 0, {}, []])
def test_historical_profile_mode_projection_refuses_unwitnessed_shape(manual):
    current = ProfileExtractorConfig(extraction_definition_prompt="Synthetic policy")
    frozen = {**snapshot_extractor_config(current), "manual_trigger": manual}
    with pytest.raises(ValueError, match="manual_trigger"):
        effective_frozen_extractor_config(frozen, extractor_kind="profile")
    with pytest.raises(ValueError, match="manual_trigger"):
        snapshot_extractor_config(current, frozen=frozen)


@pytest.mark.unit
def test_profile_mode_retirement_does_not_hide_unknown_frozen_policy():
    current = ProfileExtractorConfig(extraction_definition_prompt="Synthetic policy")
    frozen = {
        **snapshot_extractor_config(current),
        "manual_trigger": False,
        "unknown": "retain",
    }
    assert (
        effective_frozen_extractor_config(frozen, extractor_kind="profile")["unknown"]
        == "retain"
    )
    assert not matches_frozen_extractor_config(
        frozen,
        snapshot_extractor_config(current),
        extractor_kind="profile",
        original_hash=build_scope_hash(frozen),
    )


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("width,stride", [(None, None), (16, 3)])
def test_historical_snapshot_hash_and_lookup_repr_survive_current_parsing(
    kind, width, stride
):
    model = ProfileExtractorConfig if kind == "profile" else UserPlaybookExtractorConfig
    current = model(extraction_definition_prompt="Synthetic extraction policy")
    frozen = snapshot_extractor_config(current)
    frozen.update(window_size_override=width, stride_size_override=stride)
    before = deepcopy(frozen)
    parsed = model.model_validate(
        effective_frozen_extractor_config(frozen, extractor_kind=kind)
    )
    continued = snapshot_extractor_config(parsed, frozen=frozen)
    assert continued == frozen == before
    assert build_scope_hash(continued) == build_scope_hash(frozen)
    assert matches_frozen_extractor_config(
        frozen, continued, extractor_kind=kind, original_hash=build_scope_hash(frozen)
    )
    identity = frozen_extractor_config_identity(parsed, frozen)
    assert identity.endswith(
        f"window_size_override={width!r} stride_size_override={stride!r}"
    )
    assert "window_size_override" not in parsed.model_dump()
    assert "stride_size_override" not in parsed.model_dump()
    changed = {**continued, "window_size_override": 999}
    assert not matches_frozen_extractor_config(
        frozen, changed, extractor_kind=kind, original_hash=build_scope_hash(frozen)
    )
    assert frozen == before


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_geometry_projection_never_authenticates_a_mutated_original(kind):
    model = ProfileExtractorConfig if kind == "profile" else UserPlaybookExtractorConfig
    config = model(extraction_definition_prompt="Synthetic extraction policy")
    original = snapshot_extractor_config(config)
    changed = {**original, "window_size_override": 99}
    assert not matches_frozen_extractor_config(
        changed, original, extractor_kind=kind, original_hash=build_scope_hash(original)
    )


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_new_wire_slots_are_inert_and_do_not_create_false_resume_drift(kind, caplog):
    model = ProfileExtractorConfig if kind == "profile" else UserPlaybookExtractorConfig
    config = model(extraction_definition_prompt="Synthetic extraction policy")
    snapshot = snapshot_extractor_config(config)
    assert snapshot["window_size_override"] is None
    assert snapshot["stride_size_override"] is None
    run = AgentRunRecord(
        id="owned-run",
        binding=AgentBinding(
            org_id="owned-org",
            user_id="owned-user",
            extractor_kind=kind,
            request_id="owned-request",
            agent_version="v1",
            source="api",
            source_interaction_ids=[],
            window_start_interaction_id=None,
            window_end_interaction_id=None,
            extractor_config_hash=build_scope_hash(snapshot),
        ),
        status=AgentRunStatus.FAILED,
        generation_request_snapshot={"extractor_config": snapshot},
    )
    with caplog.at_level("INFO"):
        _log_config_hash_drift(run, config)
    assert "resumable_extraction_config_hash_drift" not in caplog.text
    changed = config.model_copy(
        update={"extraction_definition_prompt": "Changed policy"}
    )
    with caplog.at_level("INFO"):
        _log_config_hash_drift(replace(run), changed)
    assert "resumable_extraction_config_hash_drift" in caplog.text
