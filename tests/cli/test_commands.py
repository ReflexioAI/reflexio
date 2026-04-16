"""Command integration tests using Typer CliRunner with a mocked client.

Each test invokes a CLI command through the runner and checks exit_code
and expected output content.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


class TestInteractionsList:
    """Tests for 'interactions list'."""

    def test_list_human_output(self, runner, app, mock_client) -> None:
        mock_client.get_interactions.return_value = MagicMock(
            interactions=[
                MagicMock(
                    interaction_id=1,
                    user_id="test",
                    request_id="req-1",
                    created_at=1000000000,
                    role="user",
                    content="Hello",
                ),
                MagicMock(
                    interaction_id=2,
                    user_id="test",
                    request_id="req-1",
                    created_at=1000000001,
                    role="assistant",
                    content="Hi",
                ),
            ]
        )
        result = runner.invoke(app, ["interactions", "list", "--user-id", "test"])
        assert result.exit_code == 0, result.output
        assert "Hello" in result.output
        assert "Hi" in result.output

    def test_list_json_output(self, runner, app, mock_client) -> None:
        ix = MagicMock()
        ix.model_dump.return_value = {
            "interaction_id": 1,
            "role": "user",
            "content": "Hi",
        }
        mock_client.get_interactions.return_value = MagicMock(interactions=[ix])
        result = runner.invoke(
            app, ["--json", "interactions", "list", "--user-id", "test"]
        )
        assert result.exit_code == 0, result.output
        envelope = json.loads(result.output)
        assert envelope["ok"] is True
        assert "data" in envelope
        assert "meta" in envelope


class TestPlaybooksList:
    """Tests for 'agent-playbooks list'."""

    def test_list_playbooks(self, runner, app, mock_client) -> None:
        fb = MagicMock(spec=[])
        fb.content = "Be concise"
        fb.playbook_status = "pending"
        fb.status = None
        mock_client.get_agent_playbooks.return_value = MagicMock(agent_playbooks=[fb])
        result = runner.invoke(app, ["agent-playbooks", "list"])
        assert result.exit_code == 0, result.output
        assert "Be concise" in result.output

    def test_list_playbooks_with_status_filter(self, runner, app, mock_client) -> None:
        fb = MagicMock(spec=[])
        fb.content = "Approved rule"
        fb.playbook_status = "approved"
        fb.status = None
        mock_client.get_agent_playbooks.return_value = MagicMock(agent_playbooks=[fb])
        result = runner.invoke(
            app, ["agent-playbooks", "list", "--playbook-status", "approved"]
        )
        assert result.exit_code == 0, result.output
        assert "Approved rule" in result.output

    def test_list_playbooks_invalid_status(self, runner, app, mock_client) -> None:
        result = runner.invoke(
            app, ["agent-playbooks", "list", "--playbook-status", "invalid"]
        )
        assert result.exit_code != 0


class TestUserPlaybooks:
    """Tests for 'user-playbooks' commands."""

    def test_list_user_playbooks(self, runner, app, mock_client) -> None:
        fb = MagicMock(spec=[])
        fb.content = "Raw note"
        fb.playbook_status = None
        fb.status = None
        mock_client.get_user_playbooks.return_value = MagicMock(user_playbooks=[fb])
        result = runner.invoke(app, ["user-playbooks", "list"])
        assert result.exit_code == 0, result.output
        assert "Raw note" in result.output

    def test_add_user_playbook(self, runner, app, mock_client) -> None:
        mock_client.add_user_playbook.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "user-playbooks",
                "add",
                "--content",
                "test playbook",
                "--trigger",
                "when X",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.add_user_playbook.assert_called_once()


class TestProfilesList:
    """Tests for 'profiles list'."""

    def test_list_profiles(self, runner, app, mock_client) -> None:
        p = MagicMock(spec=[])
        p.content = "Likes Python"
        p.status = None
        mock_client.get_profiles.return_value = MagicMock(user_profiles=[p])
        result = runner.invoke(app, ["user-profiles", "list", "--user-id", "test"])
        assert result.exit_code == 0, result.output
        assert "Likes Python" in result.output


class TestProfilesAdd:
    """Tests for the new 'user-profiles add' command."""

    def test_add_profile(self, runner, app, mock_client) -> None:
        mock_client.add_user_profile.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "user-profiles",
                "add",
                "--user-id",
                "alice",
                "--content",
                "Prefers concise responses",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.add_user_profile.assert_called_once()
        # The CLI passes a single-element list of dicts; the client
        # wrapper is responsible for filling in the required UserProfile
        # fields (profile_id, last_modified_timestamp, etc) before
        # serializing to the request body.
        args, _ = mock_client.add_user_profile.call_args
        profiles = args[0]
        assert len(profiles) == 1
        entry = profiles[0]
        # entry is a dict — access via key
        assert entry["user_id"] == "alice"
        assert entry["content"] == "Prefers concise responses"
        assert entry["source"] == "cli-manual"

    def test_add_profile_requires_user_id(self, runner, app, mock_client) -> None:
        result = runner.invoke(app, ["user-profiles", "add", "--content", "no user"])
        assert result.exit_code != 0


class TestProfilesRegenerate:
    """Tests for the renamed 'user-profiles regenerate' command."""

    def test_regenerate_profile(self, runner, app, mock_client) -> None:
        mock_client.rerun_profile_generation.return_value = MagicMock()
        result = runner.invoke(
            app, ["user-profiles", "regenerate", "--user-id", "alice"]
        )
        assert result.exit_code == 0, result.output
        mock_client.rerun_profile_generation.assert_called_once_with(
            user_id="alice", wait_for_response=False
        )

    def test_old_generate_name_removed(self, runner, app, mock_client) -> None:
        """The old 'generate' verb should no longer exist (no alias)."""
        result = runner.invoke(app, ["user-profiles", "generate", "--user-id", "alice"])
        assert result.exit_code != 0


class TestUserPlaybooksUpdate:
    """Tests for the new 'user-playbooks update' command."""

    def test_update_user_playbook_content(self, runner, app, mock_client) -> None:
        mock_client.update_user_playbook.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "user-playbooks",
                "update",
                "--playbook-id",
                "42",
                "--content",
                "new content",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.update_user_playbook.assert_called_once_with(
            user_playbook_id=42,
            content="new content",
            playbook_name=None,
        )

    def test_update_user_playbook_no_fields_errors(
        self, runner, app, mock_client
    ) -> None:
        """Calling update with no editable fields should fail with a validation error."""
        result = runner.invoke(app, ["user-playbooks", "update", "--playbook-id", "42"])
        assert result.exit_code != 0
        mock_client.update_user_playbook.assert_not_called()


class TestAgentPlaybooksAdd:
    """Tests for the new 'agent-playbooks add' command."""

    def test_add_agent_playbook(self, runner, app, mock_client) -> None:
        mock_client.add_agent_playbooks.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "agent-playbooks",
                "add",
                "--content",
                "Always greet by name",
                "--agent-version",
                "v1.0",
                "--rationale",
                "Personalization",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.add_agent_playbooks.assert_called_once()
        args, _ = mock_client.add_agent_playbooks.call_args
        playbooks = args[0]
        assert len(playbooks) == 1
        assert playbooks[0].content == "Always greet by name"
        assert playbooks[0].agent_version == "v1.0"
        assert playbooks[0].rationale == "Personalization"


class TestAgentPlaybooksUpdate:
    """Tests for the new 'agent-playbooks update' command."""

    def test_update_agent_playbook(self, runner, app, mock_client) -> None:
        mock_client.update_agent_playbook.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "agent-playbooks",
                "update",
                "--playbook-id",
                "7",
                "--playbook-name",
                "new_category",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.update_agent_playbook.assert_called_once_with(
            agent_playbook_id=7,
            content=None,
            playbook_name="new_category",
        )

    def test_update_agent_playbook_no_fields_errors(
        self, runner, app, mock_client
    ) -> None:
        result = runner.invoke(app, ["agent-playbooks", "update", "--playbook-id", "7"])
        assert result.exit_code != 0
        mock_client.update_agent_playbook.assert_not_called()


class TestAgentPlaybooksUpdateStatus:
    """Tests for the new 'agent-playbooks update-status' command."""

    def test_update_status_approved(self, runner, app, mock_client) -> None:
        mock_client.update_agent_playbook_status.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "agent-playbooks",
                "update-status",
                "--playbook-id",
                "5",
                "--status",
                "approved",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.update_agent_playbook_status.assert_called_once()
        _, kwargs = mock_client.update_agent_playbook_status.call_args
        assert kwargs["agent_playbook_id"] == 5
        # playbook_status is the PlaybookStatus enum
        assert kwargs["playbook_status"].value == "approved"

    def test_update_status_pending(self, runner, app, mock_client) -> None:
        mock_client.update_agent_playbook_status.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "agent-playbooks",
                "update-status",
                "--playbook-id",
                "5",
                "--status",
                "pending",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.update_agent_playbook_status.assert_called_once()
        _, kwargs = mock_client.update_agent_playbook_status.call_args
        assert kwargs["agent_playbook_id"] == 5
        assert kwargs["playbook_status"].value == "pending"

    def test_update_status_rejected(self, runner, app, mock_client) -> None:
        mock_client.update_agent_playbook_status.return_value = MagicMock()
        result = runner.invoke(
            app,
            [
                "agent-playbooks",
                "update-status",
                "--playbook-id",
                "5",
                "--status",
                "rejected",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_client.update_agent_playbook_status.assert_called_once()
        _, kwargs = mock_client.update_agent_playbook_status.call_args
        assert kwargs["agent_playbook_id"] == 5
        assert kwargs["playbook_status"].value == "rejected"

    def test_update_status_invalid(self, runner, app, mock_client) -> None:
        result = runner.invoke(
            app,
            [
                "agent-playbooks",
                "update-status",
                "--playbook-id",
                "5",
                "--status",
                "garbage",
            ],
        )
        assert result.exit_code != 0
        mock_client.update_agent_playbook_status.assert_not_called()


class TestConfigShow:
    """Tests for 'config show'."""

    def test_show_config(self, runner, app, mock_client) -> None:
        config_mock = MagicMock()
        config_mock.model_dump.return_value = {"key": "value"}
        mock_client.get_config.return_value = config_mock
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0, result.output
        assert "key" in result.output


class TestPublishUserIdResolution:
    """Tests for 'interactions publish' user_id precedence: payload > flag > env > error."""

    @staticmethod
    def _write_payload(tmp_path, include_user_id: bool = False, **extras) -> str:
        """Write a minimal publish payload JSON and return its path."""
        payload: dict[str, object] = {
            "interactions": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        if include_user_id:
            payload["user_id"] = extras.pop("payload_user_id", "payload-user")
        payload.update(extras)
        path = tmp_path / "payload.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def test_env_user_id_used_when_flag_missing(
        self, runner, app, mock_client, tmp_path, monkeypatch
    ) -> None:
        """No flag, no payload user_id — REFLEXIO_USER_ID env var is used."""
        monkeypatch.setenv("REFLEXIO_USER_ID", "env-user")
        mock_client.publish_interaction.return_value = MagicMock(
            counts={"interactions": 2}, user_id="env-user"
        )
        payload_file = self._write_payload(tmp_path)

        result = runner.invoke(app, ["interactions", "publish", "--file", payload_file])

        assert result.exit_code == 0, result.output
        mock_client.publish_interaction.assert_called_once()
        assert mock_client.publish_interaction.call_args.kwargs["user_id"] == "env-user"

    def test_payload_user_id_beats_env(
        self, runner, app, mock_client, tmp_path, monkeypatch
    ) -> None:
        """Payload user_id takes precedence over env var."""
        monkeypatch.setenv("REFLEXIO_USER_ID", "env-user")
        mock_client.publish_interaction.return_value = MagicMock(
            counts={"interactions": 2}, user_id="payload-user"
        )
        payload_file = self._write_payload(tmp_path, include_user_id=True)

        result = runner.invoke(app, ["interactions", "publish", "--file", payload_file])

        assert result.exit_code == 0, result.output
        assert (
            mock_client.publish_interaction.call_args.kwargs["user_id"]
            == "payload-user"
        )

    def test_flag_beats_env(
        self, runner, app, mock_client, tmp_path, monkeypatch
    ) -> None:
        """--user-id flag wins over env var when payload lacks user_id."""
        monkeypatch.setenv("REFLEXIO_USER_ID", "env-user")
        mock_client.publish_interaction.return_value = MagicMock(
            counts={"interactions": 2}, user_id="flag-user"
        )
        payload_file = self._write_payload(tmp_path)

        result = runner.invoke(
            app,
            [
                "interactions",
                "publish",
                "--user-id",
                "flag-user",
                "--file",
                payload_file,
            ],
        )

        assert result.exit_code == 0, result.output
        assert (
            mock_client.publish_interaction.call_args.kwargs["user_id"] == "flag-user"
        )

    def test_error_when_all_sources_missing(
        self, runner, app, mock_client, tmp_path, monkeypatch
    ) -> None:
        """No flag, no payload user_id, no env var → validation error."""
        monkeypatch.delenv("REFLEXIO_USER_ID", raising=False)
        payload_file = self._write_payload(tmp_path)

        result = runner.invoke(app, ["interactions", "publish", "--file", payload_file])

        assert result.exit_code != 0
        assert "REFLEXIO_USER_ID" in result.output or "user-id" in result.output.lower()
        mock_client.publish_interaction.assert_not_called()

    def test_single_turn_env_fallback(
        self, runner, app, mock_client, monkeypatch
    ) -> None:
        """Single-turn mode also falls back to REFLEXIO_USER_ID when --user-id omitted."""
        monkeypatch.setenv("REFLEXIO_USER_ID", "env-user")
        mock_client.publish_interaction.return_value = MagicMock(
            counts={"interactions": 2}, user_id="env-user"
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
            ],
        )

        assert result.exit_code == 0, result.output
        assert mock_client.publish_interaction.call_args.kwargs["user_id"] == "env-user"
