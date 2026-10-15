"""CLI must not promote or claim success for incomplete generation."""

import json

import pytest

from reflexio.models.api_schema.service_schemas import (
    RerunPlaybookGenerationResponse,
    RerunProfileGenerationResponse,
    UpgradeProfilesResponse,
    UpgradeUserPlaybooksResponse,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ["profiles", "playbooks"])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize(
    "failure", ["generation", "generation_zero", "promotion", "promotion_exception"]
)
def test_regeneration_failure_exits_nonzero(
    runner, app, mock_client, kind, json_mode, failure
):
    if kind == "profiles":
        command = ["user-profiles", "regenerate", "--user-id", "alice", "--wait"]
        generation = mock_client.rerun_profile_generation
        promotion = mock_client.upgrade_profiles
    else:
        command = ["user-playbooks", "regenerate", "--agent-version", "v1", "--wait"]
        generation = mock_client.rerun_playbook_generation
        promotion = mock_client.upgrade_user_playbooks
    if kind == "profiles":
        generation.return_value = RerunProfileGenerationResponse(
            success=not failure.startswith("generation"),
            msg="Generation incomplete",
            profiles_generated=0 if failure == "generation_zero" else 1,
        )
        promotion.return_value = UpgradeProfilesResponse(
            success=False, message="Promotion rejected"
        )
    else:
        generation.return_value = RerunPlaybookGenerationResponse(
            success=not failure.startswith("generation"),
            msg="Generation incomplete",
            playbooks_generated=0 if failure == "generation_zero" else 1,
        )
        promotion.return_value = UpgradeUserPlaybooksResponse(
            success=False, message="Promotion rejected"
        )
    if failure == "promotion_exception":
        promotion.side_effect = RuntimeError("promotion unavailable")
    result = runner.invoke(app, (["--json"] if json_mode else []) + command)
    assert result.exit_code == 1, result.output
    assert "complete (" not in result.output
    if failure.startswith("generation"):
        promotion.assert_not_called()
    else:
        promotion.assert_called_once()
    if json_mode:
        assert json.loads(result.output)["ok"] is False


@pytest.mark.parametrize("kind", ["profiles", "playbooks"])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("generated", [0, 2, None])
def test_successful_regeneration_promotes_only_new_output(
    runner, app, mock_client, kind, json_mode, generated
):
    promoted = 2 if generated is None else generated
    if kind == "profiles":
        generation = mock_client.rerun_profile_generation
        promotion = mock_client.upgrade_profiles
        generation.return_value = RerunProfileGenerationResponse(
            success=True, operation_id="attempt-profiles", profiles_generated=generated
        )
        promotion.return_value = UpgradeProfilesResponse(
            success=True, profiles_promoted=promoted
        )
        command = ["user-profiles", "regenerate", "--user-id", "alice"]
    else:
        generation = mock_client.rerun_playbook_generation
        promotion = mock_client.upgrade_user_playbooks
        generation.return_value = RerunPlaybookGenerationResponse(
            success=True,
            operation_id="attempt-playbooks",
            playbooks_generated=generated,
        )
        promotion.return_value = UpgradeUserPlaybooksResponse(
            success=True, user_playbooks_promoted=promoted
        )
        command = ["user-playbooks", "regenerate", "--agent-version", "v1"]
    result = runner.invoke(
        app, (["--json"] if json_mode else []) + command + ["--wait"]
    )
    assert result.exit_code == 0, result.output
    if generated != 0:
        promotion.assert_called_once()
    else:
        promotion.assert_not_called()
    if json_mode:
        result_data = json.loads(result.output)
        assert result_data["ok"] is True
        assert result_data["data"]["operation_id"] == f"attempt-{kind}"
    else:
        assert f"complete ({promoted}" in result.output
