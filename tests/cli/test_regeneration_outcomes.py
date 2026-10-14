"""CLI must not promote or claim success for incomplete generation."""

import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("kind", ["profiles", "playbooks"])
@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("failure", ["generation", "promotion", "promotion_exception"])
def test_regeneration_failure_exits_nonzero(
    runner, app, mock_client, kind, json_mode, failure
):
    if kind == "profiles":
        command = ["user-profiles", "regenerate", "--user-id", "alice", "--wait"]
        generation = mock_client.rerun_profile_generation
        promotion = mock_client.upgrade_profiles
        count = "profiles_promoted"
    else:
        command = ["user-playbooks", "regenerate", "--agent-version", "v1", "--wait"]
        generation = mock_client.rerun_playbook_generation
        promotion = mock_client.upgrade_user_playbooks
        count = "user_playbooks_promoted"
    generation.return_value = SimpleNamespace(
        success=failure != "generation", msg="Generation incomplete"
    )
    promotion.return_value = SimpleNamespace(
        success=False, msg="Promotion rejected", **{count: 0}
    )
    if failure == "promotion_exception":
        promotion.side_effect = RuntimeError("promotion unavailable")
    result = runner.invoke(app, (["--json"] if json_mode else []) + command)
    assert result.exit_code == 1, result.output
    assert "complete (" not in result.output
    if failure == "generation":
        promotion.assert_not_called()
    else:
        promotion.assert_called_once()
    if json_mode:
        assert json.loads(result.output)["ok"] is False


@pytest.mark.parametrize("kind", ["profiles", "playbooks"])
def test_successful_zero_output_can_be_promoted(runner, app, mock_client, kind):
    generation = getattr(
        mock_client,
        f"rerun_{'profile' if kind == 'profiles' else 'playbook'}_generation",
    )
    generation.return_value = SimpleNamespace(success=True)
    if kind == "profiles":
        mock_client.upgrade_profiles.return_value = SimpleNamespace(
            success=True, profiles_promoted=0
        )
        command = ["user-profiles", "regenerate", "--user-id", "alice"]
    else:
        mock_client.upgrade_user_playbooks.return_value = SimpleNamespace(
            success=True, user_playbooks_promoted=0
        )
        command = ["user-playbooks", "regenerate", "--agent-version", "v1"]
    result = runner.invoke(app, command + ["--wait"])
    assert result.exit_code == 0, result.output
    assert "complete (0" in result.output
