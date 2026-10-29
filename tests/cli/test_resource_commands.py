"""Behaviour of the resource CLI commands (profiles, playbooks, interactions).

Each test drives the real Typer app through ``CliRunner`` with the client
mocked at the HTTP boundary (``get_client``) and real response models, and
checks what a user observes: exit code, stdout/stderr text, the JSON
envelope, and which client call (if any) was made with which arguments.
"""

from __future__ import annotations

import json

import pytest
import requests

from reflexio.cli.errors import EXIT_GENERAL, EXIT_VALIDATION
from reflexio.models.api_schema.domain import (
    AddAgentPlaybookResponse,
    AddUserPlaybookResponse,
    AddUserProfileResponse,
    BulkDeleteResponse,
    DeleteAgentPlaybookResponse,
    DeleteUserInteractionResponse,
    DeleteUserPlaybookResponse,
    DeleteUserProfileResponse,
    PublishUserInteractionResponse,
)
from reflexio.models.api_schema.retriever_schema import (
    GetAgentPlaybooksViewResponse,
    GetInteractionsViewResponse,
    GetProfilesViewResponse,
    GetUserPlaybooksViewResponse,
    SearchAgentPlaybooksViewResponse,
    SearchInteractionsViewResponse,
    SearchProfilesViewResponse,
    SearchUserPlaybooksViewResponse,
    UpdateAgentPlaybookResponse,
    UpdatePlaybookStatusResponse,
    UpdateUserPlaybookResponse,
)
from reflexio.models.api_schema.service_schemas import PlaybookStatus, Status
from reflexio.models.api_schema.ui.entities import (
    AgentPlaybookView,
    InteractionView,
    ProfileView,
    UserPlaybookView,
)


@pytest.fixture(autouse=True)
def _isolated_identity_env(monkeypatch):
    """Commands read user/agent identity from env; keep the host's out."""
    monkeypatch.delenv("REFLEXIO_USER_ID", raising=False)
    monkeypatch.delenv("REFLEXIO_AGENT_VERSION", raising=False)


def _envelope(result) -> dict:
    """Parse the JSON envelope a ``--json`` command writes to stdout."""
    return json.loads(result.stdout)


def _profile(content: str = "Prefers terse answers") -> ProfileView:
    return ProfileView(
        profile_id="p-1",
        user_id="cli-user",
        content=content,
        last_modified_timestamp=1_700_000_000,
        generated_from_request_id="req-1",
    )


def _user_playbook(content: str = "Ask before deleting") -> UserPlaybookView:
    return UserPlaybookView(
        user_playbook_id=7,
        agent_version="v1",
        request_id="req-1",
        playbook_name="agent_corrections",
        content=content,
    )


def _agent_playbook(content: str = "Cite the source") -> AgentPlaybookView:
    return AgentPlaybookView(agent_playbook_id=3, agent_version="v1", content=content)


def _interaction(content: str = "hello there") -> InteractionView:
    return InteractionView(
        interaction_id=11,
        user_id="cli-user",
        request_id="req-1",
        role="user",
        content=content,
    )


# ---------------------------------------------------------------------------
# user-profiles
# ---------------------------------------------------------------------------


class TestProfilesCommands:
    def test_list_rejects_unknown_status_before_calling_server(
        self, runner, app, mock_client
    ) -> None:
        result = runner.invoke(app, ["user-profiles", "list", "--status", "stale"])
        assert result.exit_code == EXIT_VALIDATION
        assert "Invalid status: stale" in result.output
        mock_client.get_profiles.assert_not_called()
        mock_client.get_all_profiles.assert_not_called()

    def test_list_without_user_queries_every_user_with_status(
        self, runner, app, mock_client
    ) -> None:
        mock_client.get_all_profiles.return_value = GetProfilesViewResponse(
            success=True, user_profiles=[_profile()]
        )
        result = runner.invoke(
            app,
            ["--json", "user-profiles", "list", "--status", "archived", "--limit", "1"],
        )
        assert result.exit_code == 0, result.output
        mock_client.get_all_profiles.assert_called_once_with(
            limit=1, status_filter="archived"
        )
        mock_client.get_profiles.assert_not_called()
        envelope = _envelope(result)
        assert envelope["ok"] is True
        assert envelope["data"][0]["content"] == "Prefers terse answers"
        assert envelope["meta"]["limit"] == 1

    def test_list_for_one_user_passes_status_as_list(
        self, runner, app, mock_client
    ) -> None:
        mock_client.get_profiles.return_value = GetProfilesViewResponse(
            success=True, user_profiles=[]
        )
        result = runner.invoke(
            app,
            ["user-profiles", "list", "--user-id", "cli-user", "--status", "pending"],
        )
        assert result.exit_code == 0, result.output
        mock_client.get_profiles.assert_called_once_with(
            user_id="cli-user", top_k=20, status_filter=["pending"]
        )
        assert "Found 0 profile(s)" in result.output

    def test_search_requires_a_user_instead_of_defaulting(
        self, runner, app, mock_client
    ) -> None:
        result = runner.invoke(app, ["user-profiles", "search-user-profiles", "terse"])
        assert result.exit_code == EXIT_VALIDATION
        mock_client.search_user_profiles.assert_not_called()

    def test_search_uses_env_user_and_forwards_threshold(
        self, runner, app, mock_client, monkeypatch
    ) -> None:
        monkeypatch.setenv("REFLEXIO_USER_ID", "env-user")
        mock_client.search_user_profiles.return_value = SearchProfilesViewResponse(
            success=True, user_profiles=[_profile()]
        )
        result = runner.invoke(
            app,
            [
                "user-profiles",
                "search-user-profiles",
                "terse",
                "--threshold",
                "0.25",
                "--limit",
                "3",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.search_user_profiles.assert_called_once_with(
            query="terse", user_id="env-user", top_k=3, threshold=0.25
        )
        assert "Found 1 profile(s)" in result.output
        assert "Prefers terse answers" in result.output

    def test_search_without_threshold_omits_it_and_renders_json(
        self, runner, app, mock_client
    ) -> None:
        mock_client.search_user_profiles.return_value = SearchProfilesViewResponse(
            success=True, user_profiles=[]
        )
        result = runner.invoke(
            app,
            [
                "--json",
                "user-profiles",
                "search-user-profiles",
                "terse",
                "--user-id",
                "cli-user",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.search_user_profiles.assert_called_once_with(
            query="terse", user_id="cli-user", top_k=10
        )
        assert _envelope(result)["data"] == []

    def test_delete_reports_the_deleted_profile(self, runner, app, mock_client) -> None:
        mock_client.delete_profile.return_value = DeleteUserProfileResponse(
            success=True
        )
        result = runner.invoke(
            app,
            ["user-profiles", "delete", "--user-id", "cli-user", "--profile-id", "p-1"],
        )
        assert result.exit_code == 0, result.output
        mock_client.delete_profile.assert_called_once_with(
            user_id="cli-user", profile_id="p-1", wait_for_response=True
        )
        assert "Deleted profile p-1 for user cli-user" in result.output

    def test_delete_failure_exits_nonzero_with_server_message(
        self, runner, app, mock_client
    ) -> None:
        mock_client.delete_profile.return_value = DeleteUserProfileResponse(
            success=False, message="Profile id or search query is required"
        )
        result = runner.invoke(
            app, ["user-profiles", "delete", "--user-id", "cli-user"]
        )
        assert result.exit_code == EXIT_GENERAL
        assert "Profile id or search query is required" in result.output
        assert "Deleted profile" not in result.output

    def test_delete_json_mode_renders_server_envelope(
        self, runner, app, mock_client
    ) -> None:
        mock_client.delete_profile.return_value = DeleteUserProfileResponse(
            success=True, message="deleted"
        )
        result = runner.invoke(
            app,
            [
                "--json",
                "user-profiles",
                "delete",
                "--user-id",
                "cli-user",
                "--profile-id",
                "p-1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert _envelope(result)["data"] == {"success": True, "message": "deleted"}

    def test_delete_all_declined_prompt_never_calls_server(
        self, runner, app, mock_client
    ) -> None:
        result = runner.invoke(app, ["user-profiles", "delete-all"], input="n\n")
        assert result.exit_code != 0
        assert "Aborted" in result.output
        mock_client.delete_all_profiles.assert_not_called()

    def test_delete_all_confirmed_prompt_deletes(
        self, runner, app, mock_client
    ) -> None:
        mock_client.delete_all_profiles.return_value = BulkDeleteResponse(
            success=True, deleted_count=4
        )
        result = runner.invoke(app, ["user-profiles", "delete-all"], input="y\n")
        assert result.exit_code == 0, result.output
        mock_client.delete_all_profiles.assert_called_once_with()
        assert "All profiles deleted" in result.output

    def test_delete_all_json_reports_count(self, runner, app, mock_client) -> None:
        mock_client.delete_all_profiles.return_value = BulkDeleteResponse(
            success=True, deleted_count=4
        )
        result = runner.invoke(app, ["--json", "user-profiles", "delete-all", "-y"])
        assert result.exit_code == 0, result.output
        assert _envelope(result)["data"]["deleted_count"] == 4

    def test_add_sends_manual_profile_and_renders_json(
        self, runner, app, mock_client
    ) -> None:
        mock_client.add_user_profile.return_value = AddUserProfileResponse(
            success=True, added_count=1
        )
        result = runner.invoke(
            app,
            [
                "--json",
                "user-profiles",
                "add",
                "--user-id",
                "cli-user",
                "--content",
                "Uses vim",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.add_user_profile.assert_called_once_with(
            [{"user_id": "cli-user", "content": "Uses vim", "source": "cli-manual"}]
        )
        assert _envelope(result)["data"]["added_count"] == 1

    def test_add_failure_exits_nonzero(self, runner, app, mock_client) -> None:
        mock_client.add_user_profile.return_value = AddUserProfileResponse(
            success=False
        )
        result = runner.invoke(
            app,
            ["user-profiles", "add", "--user-id", "cli-user", "--content", "Uses vim"],
        )
        assert result.exit_code == EXIT_GENERAL
        assert "Failed to add user profile" in result.output
        assert "User profile added" not in result.output


# ---------------------------------------------------------------------------
# user-playbooks
# ---------------------------------------------------------------------------


class TestUserPlaybooksCommands:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("current", [Status.CURRENT]),
            ("pending", [Status.PENDING]),
            ("archived", [Status.ARCHIVED]),
            (None, None),
        ],
    )
    def test_list_maps_status_to_server_filter(
        self, runner, app, mock_client, status, expected
    ) -> None:
        mock_client.get_user_playbooks.return_value = GetUserPlaybooksViewResponse(
            success=True, user_playbooks=[_user_playbook()]
        )
        args = ["user-playbooks", "list", "--user-id", "cli-user"]
        if status:
            args += ["--status", status]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        mock_client.get_user_playbooks.assert_called_once_with(
            limit=20,
            user_id="cli-user",
            playbook_name=None,
            agent_version=None,
            status_filter=expected,
        )
        assert "Found 1 user playbook(s)" in result.output
        assert "Ask before deleting" in result.output

    def test_list_rejects_unknown_status(self, runner, app, mock_client) -> None:
        result = runner.invoke(app, ["user-playbooks", "list", "--status", "approved"])
        assert result.exit_code == EXIT_VALIDATION
        mock_client.get_user_playbooks.assert_not_called()

    def test_list_json_mode_includes_pagination_meta(
        self, runner, app, mock_client
    ) -> None:
        mock_client.get_user_playbooks.return_value = GetUserPlaybooksViewResponse(
            success=True, user_playbooks=[_user_playbook()]
        )
        result = runner.invoke(
            app, ["--json", "user-playbooks", "list", "--limit", "1"]
        )
        assert result.exit_code == 0, result.output
        envelope = _envelope(result)
        assert envelope["data"][0]["user_playbook_id"] == 7
        assert envelope["meta"]["limit"] == 1

    def test_search_human_and_json(self, runner, app, mock_client) -> None:
        mock_client.search_user_playbooks.return_value = (
            SearchUserPlaybooksViewResponse(
                success=True, user_playbooks=[_user_playbook()]
            )
        )
        human = runner.invoke(
            app, ["user-playbooks", "search", "deleting", "--agent-version", "v1"]
        )
        assert human.exit_code == 0, human.output
        assert "Found 1 user playbook(s)" in human.output
        assert "Ask before deleting" in human.output
        mock_client.search_user_playbooks.assert_called_with(
            query="deleting", agent_version="v1", top_k=10
        )
        as_json = runner.invoke(app, ["--json", "user-playbooks", "search", "x"])
        assert as_json.exit_code == 0, as_json.output
        assert _envelope(as_json)["data"][0]["content"] == "Ask before deleting"

    def test_search_with_no_results_prints_only_count(
        self, runner, app, mock_client
    ) -> None:
        mock_client.search_user_playbooks.return_value = (
            SearchUserPlaybooksViewResponse(success=True, user_playbooks=[])
        )
        result = runner.invoke(app, ["user-playbooks", "search", "nothing"])
        assert result.exit_code == 0, result.output
        assert "Found 0 user playbook(s)" in result.output

    def test_add_defaults_agent_version_to_cli(self, runner, app, mock_client) -> None:
        mock_client.add_user_playbook.return_value = AddUserPlaybookResponse(
            success=True, added_count=1
        )
        result = runner.invoke(
            app,
            [
                "user-playbooks",
                "add",
                "--content",
                "Ask first",
                "--rationale",
                "safety",
            ],
        )
        assert result.exit_code == 0, result.output
        (playbooks,), _ = mock_client.add_user_playbook.call_args
        assert playbooks[0].agent_version == "cli"
        assert playbooks[0].rationale == "safety"
        assert playbooks[0].request_id == "cli-manual"
        assert "User playbook added: Ask first" in result.output

    def test_add_explicit_version_json_and_failure(
        self, runner, app, mock_client
    ) -> None:
        mock_client.add_user_playbook.return_value = AddUserPlaybookResponse(
            success=True, added_count=1
        )
        result = runner.invoke(
            app,
            [
                "--json",
                "user-playbooks",
                "add",
                "--content",
                "Ask first",
                "--agent-version",
                "openclaw-agent",
            ],
        )
        assert result.exit_code == 0, result.output
        (playbooks,), _ = mock_client.add_user_playbook.call_args
        assert playbooks[0].agent_version == "openclaw-agent"
        assert _envelope(result)["data"]["added_count"] == 1

        mock_client.add_user_playbook.return_value = AddUserPlaybookResponse(
            success=False, message="storage unavailable"
        )
        failed = runner.invoke(app, ["user-playbooks", "add", "--content", "x"])
        assert failed.exit_code == EXIT_GENERAL
        assert "storage unavailable" in failed.output

    def test_update_requires_a_field(self, runner, app, mock_client) -> None:
        result = runner.invoke(app, ["user-playbooks", "update", "--playbook-id", "7"])
        assert result.exit_code == EXIT_VALIDATION
        assert "--content/--playbook-name" in result.output
        mock_client.update_user_playbook.assert_not_called()

    def test_update_success_failure_and_json(self, runner, app, mock_client) -> None:
        mock_client.update_user_playbook.return_value = UpdateUserPlaybookResponse(
            success=True
        )
        ok = runner.invoke(
            app,
            ["user-playbooks", "update", "--playbook-id", "7", "--content", "New"],
        )
        assert ok.exit_code == 0, ok.output
        mock_client.update_user_playbook.assert_called_with(
            user_playbook_id=7, content="New", playbook_name=None
        )
        assert "User playbook 7 updated" in ok.output

        as_json = runner.invoke(
            app,
            [
                "--json",
                "user-playbooks",
                "update",
                "--playbook-id",
                "7",
                "--playbook-name",
                "renamed",
            ],
        )
        assert as_json.exit_code == 0, as_json.output
        assert _envelope(as_json)["data"]["success"] is True

        mock_client.update_user_playbook.return_value = UpdateUserPlaybookResponse(
            success=False, msg="not found"
        )
        failed = runner.invoke(
            app,
            ["user-playbooks", "update", "--playbook-id", "8", "--content", "New"],
        )
        assert failed.exit_code == EXIT_GENERAL
        assert "not found" in failed.output

    def test_delete_rejects_non_integer_id(self, runner, app, mock_client) -> None:
        result = runner.invoke(
            app, ["user-playbooks", "delete", "--playbook-id", "seven"]
        )
        assert result.exit_code == EXIT_VALIDATION
        mock_client.delete_user_playbook.assert_not_called()

    def test_delete_human_and_json(self, runner, app, mock_client) -> None:
        mock_client.delete_user_playbook.return_value = DeleteUserPlaybookResponse(
            success=True
        )
        human = runner.invoke(app, ["user-playbooks", "delete", "--playbook-id", "7"])
        assert human.exit_code == 0, human.output
        mock_client.delete_user_playbook.assert_called_with(
            user_playbook_id=7, wait_for_response=True
        )
        assert "Deleted user playbook 7" in human.output
        as_json = runner.invoke(
            app, ["--json", "user-playbooks", "delete", "--playbook-id", "7"]
        )
        assert _envelope(as_json)["data"]["success"] is True

    def test_delete_all_prompt_and_json(self, runner, app, mock_client) -> None:
        mock_client.delete_all_user_playbooks.return_value = BulkDeleteResponse(
            success=True, deleted_count=2
        )
        declined = runner.invoke(app, ["user-playbooks", "delete-all"], input="n\n")
        assert declined.exit_code != 0
        mock_client.delete_all_user_playbooks.assert_not_called()

        accepted = runner.invoke(app, ["user-playbooks", "delete-all"], input="y\n")
        assert accepted.exit_code == 0, accepted.output
        assert "All user playbooks deleted" in accepted.output

        as_json = runner.invoke(app, ["--json", "user-playbooks", "delete-all", "-y"])
        assert _envelope(as_json)["data"]["deleted_count"] == 2
        assert mock_client.delete_all_user_playbooks.call_count == 2

    def test_regenerate_without_wait_only_starts(
        self, runner, app, mock_client
    ) -> None:
        mock_client.rerun_playbook_generation.return_value = None
        result = runner.invoke(
            app, ["user-playbooks", "regenerate", "--agent-version", "v1"]
        )
        assert result.exit_code == 0, result.output
        mock_client.rerun_playbook_generation.assert_called_once_with(
            agent_version="v1", wait_for_response=False
        )
        mock_client.upgrade_user_playbooks.assert_not_called()
        assert "Playbook regeneration started" in result.output


# ---------------------------------------------------------------------------
# agent-playbooks
# ---------------------------------------------------------------------------


class TestAgentPlaybooksCommands:
    def test_list_json_mode(self, runner, app, mock_client) -> None:
        mock_client.get_agent_playbooks.return_value = GetAgentPlaybooksViewResponse(
            success=True, agent_playbooks=[_agent_playbook()]
        )
        result = runner.invoke(
            app,
            ["--json", "agent-playbooks", "list", "--playbook-status", "approved"],
        )
        assert result.exit_code == 0, result.output
        _, kwargs = mock_client.get_agent_playbooks.call_args
        assert kwargs["playbook_status_filter"] == PlaybookStatus.APPROVED
        assert _envelope(result)["data"][0]["content"] == "Cite the source"

    def test_search_validates_status_and_renders(
        self, runner, app, mock_client
    ) -> None:
        bad = runner.invoke(
            app, ["agent-playbooks", "search", "cite", "--playbook-status", "maybe"]
        )
        assert bad.exit_code == EXIT_VALIDATION
        mock_client.search_agent_playbooks.assert_not_called()

        mock_client.search_agent_playbooks.return_value = (
            SearchAgentPlaybooksViewResponse(
                success=True, agent_playbooks=[_agent_playbook()]
            )
        )
        human = runner.invoke(
            app,
            [
                "agent-playbooks",
                "search",
                "cite",
                "--playbook-status",
                "rejected",
                "--limit",
                "2",
            ],
        )
        assert human.exit_code == 0, human.output
        mock_client.search_agent_playbooks.assert_called_with(
            query="cite",
            agent_version=None,
            playbook_status_filter=PlaybookStatus.REJECTED,
            top_k=2,
        )
        assert "Found 1 agent playbook(s)" in human.output
        assert "Cite the source" in human.output

        as_json = runner.invoke(app, ["--json", "agent-playbooks", "search", "cite"])
        assert _envelope(as_json)["meta"]["limit"] == 10

    def test_search_without_results(self, runner, app, mock_client) -> None:
        mock_client.search_agent_playbooks.return_value = (
            SearchAgentPlaybooksViewResponse(success=True, agent_playbooks=[])
        )
        result = runner.invoke(app, ["agent-playbooks", "search", "none"])
        assert result.exit_code == 0, result.output
        assert "Found 0 agent playbook(s)" in result.output

    def test_add_json_carries_manual_source(self, runner, app, mock_client) -> None:
        mock_client.add_agent_playbooks.return_value = AddAgentPlaybookResponse(
            success=True, added_count=1
        )
        result = runner.invoke(
            app,
            [
                "--json",
                "agent-playbooks",
                "add",
                "--content",
                "Cite",
                "--agent-version",
                "v1",
                "--playbook-status",
                "approved",
            ],
        )
        assert result.exit_code == 0, result.output
        (playbooks,), _ = mock_client.add_agent_playbooks.call_args
        assert playbooks[0].playbook_status == PlaybookStatus.APPROVED
        assert json.loads(playbooks[0].playbook_metadata) == {"source": "cli-manual"}
        assert _envelope(result)["data"]["added_count"] == 1

    def test_add_rejects_unknown_status(self, runner, app, mock_client) -> None:
        result = runner.invoke(
            app,
            [
                "agent-playbooks",
                "add",
                "--content",
                "Cite",
                "--agent-version",
                "v1",
                "--playbook-status",
                "current",
            ],
        )
        assert result.exit_code == EXIT_VALIDATION
        mock_client.add_agent_playbooks.assert_not_called()

    def test_update_requires_field_and_renders_json(
        self, runner, app, mock_client
    ) -> None:
        missing = runner.invoke(
            app, ["agent-playbooks", "update", "--playbook-id", "3"]
        )
        assert missing.exit_code == EXIT_VALIDATION
        mock_client.update_agent_playbook.assert_not_called()

        mock_client.update_agent_playbook.return_value = UpdateAgentPlaybookResponse(
            success=True
        )
        result = runner.invoke(
            app,
            [
                "--json",
                "agent-playbooks",
                "update",
                "--playbook-id",
                "3",
                "--content",
                "Cite twice",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.update_agent_playbook.assert_called_once_with(
            agent_playbook_id=3, content="Cite twice", playbook_name=None
        )
        assert _envelope(result)["data"]["success"] is True

    def test_update_status_rejects_empty_status(self, runner, app, mock_client) -> None:
        result = runner.invoke(
            app,
            ["agent-playbooks", "update-status", "--playbook-id", "3", "--status", ""],
        )
        assert result.exit_code == EXIT_VALIDATION
        assert "Invalid status" in result.output
        mock_client.update_agent_playbook_status.assert_not_called()

    def test_update_status_json(self, runner, app, mock_client) -> None:
        mock_client.update_agent_playbook_status.return_value = (
            UpdatePlaybookStatusResponse(success=True)
        )
        result = runner.invoke(
            app,
            [
                "--json",
                "agent-playbooks",
                "update-status",
                "--playbook-id",
                "3",
                "--status",
                "approved",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.update_agent_playbook_status.assert_called_once_with(
            agent_playbook_id=3, playbook_status=PlaybookStatus.APPROVED
        )
        assert _envelope(result)["data"]["success"] is True

    def test_delete_human_json_and_bad_id(self, runner, app, mock_client) -> None:
        mock_client.delete_agent_playbook.return_value = DeleteAgentPlaybookResponse(
            success=True
        )
        human = runner.invoke(app, ["agent-playbooks", "delete", "--playbook-id", "3"])
        assert human.exit_code == 0, human.output
        mock_client.delete_agent_playbook.assert_called_with(
            agent_playbook_id=3, wait_for_response=True
        )
        assert "Deleted agent playbook 3" in human.output

        as_json = runner.invoke(
            app, ["--json", "agent-playbooks", "delete", "--playbook-id", "3"]
        )
        assert _envelope(as_json)["data"]["success"] is True

        bad = runner.invoke(app, ["agent-playbooks", "delete", "--playbook-id", "x"])
        assert bad.exit_code == EXIT_VALIDATION
        assert mock_client.delete_agent_playbook.call_count == 2

    def test_delete_all_prompt_and_json(self, runner, app, mock_client) -> None:
        mock_client.delete_all_agent_playbooks.return_value = BulkDeleteResponse(
            success=True, deleted_count=5
        )
        declined = runner.invoke(app, ["agent-playbooks", "delete-all"], input="n\n")
        assert declined.exit_code != 0
        mock_client.delete_all_agent_playbooks.assert_not_called()

        accepted = runner.invoke(app, ["agent-playbooks", "delete-all"], input="y\n")
        assert accepted.exit_code == 0, accepted.output
        assert "All agent playbooks deleted" in accepted.output

        as_json = runner.invoke(app, ["--json", "agent-playbooks", "delete-all", "-y"])
        assert _envelope(as_json)["data"]["deleted_count"] == 5

    def test_aggregate_client_error_is_not_reframed_as_uncertain(
        self, runner, app, mock_client
    ) -> None:
        """A definite 4xx rejection is reported as such, not as 'uncertain'."""
        response = requests.Response()
        response.status_code = 422
        response.url = "http://reflexio.test/api/aggregation_operations"
        mock_client.submit_playbook_aggregation.side_effect = requests.HTTPError(
            "422 Client Error", response=response
        )
        result = runner.invoke(
            app, ["agent-playbooks", "aggregate", "--agent-version", "v1"]
        )
        assert result.exit_code != 0
        assert "admission is uncertain" not in result.output
        mock_client.submit_playbook_aggregation.assert_called_once()
        mock_client.get_playbook_aggregation_operation.assert_not_called()

    @pytest.mark.parametrize("json_mode", [False, True])
    def test_aggregate_with_no_user_playbooks_submits_nothing(
        self, runner, app, mock_client, json_mode
    ) -> None:
        mock_client.get_user_playbooks.return_value = GetUserPlaybooksViewResponse(
            success=True, user_playbooks=[]
        )
        args = ["--json"] if json_mode else []
        result = runner.invoke(app, [*args, "agent-playbooks", "aggregate"])
        assert result.exit_code == 0, result.output
        mock_client.submit_playbook_aggregation.assert_not_called()
        if json_mode:
            assert _envelope(result)["data"] == {
                "message": "No user playbooks found to aggregate"
            }
        else:
            assert "No user playbooks found to aggregate" in result.output


# ---------------------------------------------------------------------------
# interactions
# ---------------------------------------------------------------------------


def _published(**changes) -> PublishUserInteractionResponse:
    values: dict = {"success": True, "message": "Interaction published successfully"}
    values.update(changes)
    return PublishUserInteractionResponse(**values)


_PAYLOAD = {
    "user_id": "payload-user",
    "session_id": "s-1",
    "interactions": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ],
}


class TestInteractionsPublish:
    def test_single_turn_requires_agent_response(
        self, runner, app, mock_client
    ) -> None:
        result = runner.invoke(
            app, ["interactions", "publish", "--user-message", "hi", "--user-id", "u"]
        )
        assert result.exit_code == EXIT_VALIDATION
        assert "--agent-response is required" in result.output
        mock_client.publish_interaction.assert_not_called()

    def test_single_turn_requires_a_user(self, runner, app, mock_client) -> None:
        result = runner.invoke(
            app,
            [
                "interactions",
                "publish",
                "--user-message",
                "hi",
                "--agent-response",
                "hello",
                "--session-id",
                "s-1",
            ],
        )
        assert result.exit_code == EXIT_VALIDATION
        assert "--user-id is required" in result.output
        mock_client.publish_interaction.assert_not_called()

    def test_single_turn_rejects_blank_session(self, runner, app, mock_client) -> None:
        result = runner.invoke(
            app,
            [
                "interactions",
                "publish",
                "--user-message",
                "hi",
                "--agent-response",
                "hello",
                "--user-id",
                "u",
                "--session-id",
                "   ",
            ],
        )
        assert result.exit_code == EXIT_VALIDATION
        assert "session_id is required" in result.output
        mock_client.publish_interaction.assert_not_called()

    def test_single_turn_uses_env_user_and_reports_diagnostics(
        self, runner, app, mock_client, monkeypatch
    ) -> None:
        monkeypatch.setenv("REFLEXIO_USER_ID", "env-user")
        long_warning = "line one\nline two " + "x" * 400
        mock_client.base_url = "http://reflexio.test"
        mock_client.publish_interaction.return_value = _published(
            message="Published with notes",
            storage_type="sqlite",
            storage_label="~/.reflexio/data.db",
            profiles_added=1,
            playbooks_added=0,
            request_id="req-9",
            warnings=[long_warning],
        )
        result = runner.invoke(
            app,
            [
                "interactions",
                "publish",
                "--user-message",
                "hi",
                "--agent-response",
                "hello",
                "--session-id",
                " s-1 ",
                "--wait",
            ],
        )
        assert result.exit_code == 0, result.output
        _, kwargs = mock_client.publish_interaction.call_args
        assert kwargs["user_id"] == "env-user"
        assert kwargs["session_id"] == "s-1"
        assert kwargs["wait_for_response"] is True
        out = result.output
        assert "Publishing to http://reflexio.test (server-sync)" in out
        assert "Published (env-user): Published with notes" in out
        assert "Storage: sqlite → ~/.reflexio/data.db" in out
        assert "Extraction: 1 profile(s) added, 0 playbook(s) added" in out
        assert "Request ID: req-9" in out
        warning_line = next(line for line in out.splitlines() if "Warning:" in line)
        # Collapsed to one line and truncated with an ellipsis.
        assert "line one line two" in warning_line
        assert warning_line.endswith("…")
        assert len(warning_line) < 300

    def test_storage_label_without_type_and_count_without_playbooks(
        self, runner, app, mock_client
    ) -> None:
        mock_client.publish_interaction.return_value = _published(
            storage_label="remote", playbooks_added=2
        )
        result = runner.invoke(
            app, ["interactions", "publish", "--data", json.dumps(_PAYLOAD)]
        )
        assert result.exit_code == 0, result.output
        assert "Storage: ? → remote" in result.output
        assert "Extraction: 2 playbook(s) added" in result.output
        assert "profile(s) added" not in result.output
        assert "Published (payload-user)" in result.output

    def test_data_from_at_file_and_payload_overrides(
        self, runner, app, mock_client, tmp_path
    ) -> None:
        payload = {
            **_PAYLOAD,
            "source": "import",
            "agent_version": "v9",
            "evaluation_only": True,
        }
        path = tmp_path / "payload.json"
        path.write_text(json.dumps(payload))
        mock_client.publish_interaction.return_value = _published()
        result = runner.invoke(
            app, ["--json", "interactions", "publish", "--data", f"@{path}"]
        )
        assert result.exit_code == 0, result.output
        _, kwargs = mock_client.publish_interaction.call_args
        assert kwargs["user_id"] == "payload-user"
        assert kwargs["source"] == "import"
        assert kwargs["agent_version"] == "v9"
        assert kwargs["evaluation_only"] is True
        assert [i.content for i in kwargs["interactions"]] == ["hi", "hello"]
        assert _envelope(result)["data"]["success"] is True

    def test_data_at_missing_file_is_validation_error(
        self, runner, app, mock_client, tmp_path
    ) -> None:
        missing = tmp_path / "absent.json"
        result = runner.invoke(
            app, ["interactions", "publish", "--data", f"@{missing}"]
        )
        assert result.exit_code == EXIT_VALIDATION
        assert "File not found" in result.output
        mock_client.publish_interaction.assert_not_called()

    def test_file_option_missing_and_present(
        self, runner, app, mock_client, tmp_path
    ) -> None:
        missing = runner.invoke(
            app, ["interactions", "publish", "--file", str(tmp_path / "none.jsonl")]
        )
        assert missing.exit_code == EXIT_VALIDATION
        assert "File not found" in missing.output

        second = {**_PAYLOAD, "user_id": "other-user"}
        path = tmp_path / "batch.jsonl"
        path.write_text(json.dumps(_PAYLOAD) + "\n\n" + json.dumps(second) + "\n")
        mock_client.publish_interaction.return_value = _published()
        result = runner.invoke(app, ["interactions", "publish", "--file", str(path)])
        assert result.exit_code == 0, result.output
        assert "Processing 2 payload(s)" in result.output
        users = [
            c.kwargs["user_id"] for c in mock_client.publish_interaction.call_args_list
        ]
        assert users == ["payload-user", "other-user"]

    def test_stdin_json_array(self, runner, app, mock_client) -> None:
        mock_client.publish_interaction.return_value = _published()
        result = runner.invoke(
            app,
            ["interactions", "publish", "--stdin"],
            input=json.dumps([_PAYLOAD]),
        )
        assert result.exit_code == 0, result.output
        mock_client.publish_interaction.assert_called_once()

    def test_no_input_mode_is_validation_error(self, runner, app, mock_client) -> None:
        result = runner.invoke(app, ["interactions", "publish"])
        assert result.exit_code == EXIT_VALIDATION
        assert "Specify --user-message/--agent-response" in result.output

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ("   ", "Empty input"),
            ("[1, 2]", "JSON array elements must be objects"),
            ('{"interactions": []}\n{not json', "Invalid JSON on line 2"),
            ('{"session_id": "s"}', "Payload missing 'interactions' list"),
            (
                json.dumps({"session_id": "s", "interactions": [{"role": 5}]}),
                "Invalid interaction data",
            ),
        ],
    )
    def test_malformed_payloads_are_validation_errors(
        self, runner, app, mock_client, raw, message
    ) -> None:
        result = runner.invoke(
            app, ["interactions", "publish", "--user-id", "u", "--data", raw]
        )
        assert result.exit_code == EXIT_VALIDATION, result.output
        assert message in result.output
        mock_client.publish_interaction.assert_not_called()

    def test_payload_without_any_user_is_rejected(
        self, runner, app, mock_client
    ) -> None:
        payload = {k: v for k, v in _PAYLOAD.items() if k != "user_id"}
        result = runner.invoke(
            app, ["interactions", "publish", "--data", json.dumps(payload)]
        )
        assert result.exit_code == EXIT_VALIDATION
        assert "--user-id is required" in result.output
        mock_client.publish_interaction.assert_not_called()


class TestInteractionsReadAndDelete:
    def test_list_requires_user(self, runner, app, mock_client) -> None:
        result = runner.invoke(app, ["interactions", "list"])
        assert result.exit_code == EXIT_VALIDATION
        mock_client.get_interactions.assert_not_called()

    def test_list_empty(self, runner, app, mock_client) -> None:
        mock_client.get_interactions.return_value = GetInteractionsViewResponse(
            success=True, interactions=[]
        )
        result = runner.invoke(app, ["interactions", "list", "--user-id", "cli-user"])
        assert result.exit_code == 0, result.output
        mock_client.get_interactions.assert_called_once_with(
            user_id="cli-user", top_k=20
        )
        assert "Found 0 interaction(s)" in result.output

    def test_search_human_json_and_empty(self, runner, app, mock_client) -> None:
        mock_client.search_interactions.return_value = SearchInteractionsViewResponse(
            success=True, interactions=[_interaction()]
        )
        human = runner.invoke(
            app,
            [
                "interactions",
                "search",
                "--query",
                "hello",
                "--user-id",
                "cli-user",
                "--limit",
                "2",
            ],
        )
        assert human.exit_code == 0, human.output
        mock_client.search_interactions.assert_called_with(
            query="hello", user_id="cli-user", top_k=2
        )
        assert "Found 1 interaction(s)" in human.output
        assert "hello there" in human.output

        as_json = runner.invoke(
            app, ["--json", "interactions", "search", "--query", "hello"]
        )
        envelope = _envelope(as_json)
        assert envelope["data"][0]["interaction_id"] == 11
        assert envelope["meta"]["limit"] == 10

        mock_client.search_interactions.return_value = SearchInteractionsViewResponse(
            success=True, interactions=[]
        )
        empty = runner.invoke(app, ["interactions", "search", "--query", "zzz"])
        assert "Found 0 interaction(s)" in empty.output

    def test_delete_human_and_json(self, runner, app, mock_client) -> None:
        mock_client.delete_interaction.return_value = DeleteUserInteractionResponse(
            success=True
        )
        human = runner.invoke(
            app,
            ["interactions", "delete", "--interaction-id", "11", "--user-id", "u"],
        )
        assert human.exit_code == 0, human.output
        mock_client.delete_interaction.assert_called_with(
            user_id="u", interaction_id=11, wait_for_response=True
        )
        assert "Deleted interaction 11" in human.output

        as_json = runner.invoke(
            app,
            [
                "--json",
                "interactions",
                "delete",
                "--interaction-id",
                "11",
                "--user-id",
                "u",
            ],
        )
        assert _envelope(as_json)["data"]["success"] is True

    def test_delete_all_prompt_and_json(self, runner, app, mock_client) -> None:
        mock_client.delete_all_interactions.return_value = BulkDeleteResponse(
            success=True, deleted_count=9
        )
        declined = runner.invoke(app, ["interactions", "delete-all"], input="n\n")
        assert declined.exit_code != 0
        mock_client.delete_all_interactions.assert_not_called()

        accepted = runner.invoke(app, ["interactions", "delete-all"], input="y\n")
        assert accepted.exit_code == 0, accepted.output
        assert "All interactions deleted" in accepted.output

        as_json = runner.invoke(app, ["--json", "interactions", "delete-all", "--yes"])
        assert _envelope(as_json)["data"]["deleted_count"] == 9
