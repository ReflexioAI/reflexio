"""Unit tests for ``reflexio config storage``.

These don't spin up the server — they patch ``client.get_my_config()``
to return fixed ``MyConfigResponse`` values and assert the CLI renders
the expected output. The goal is to pin the contract of:

- masking behaviour by default (storage)
- the --reveal confirmation prompt
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from reflexio.cli.app import create_app
from reflexio.models.api_schema.service_schemas import MyConfigResponse


@pytest.fixture
def runner() -> CliRunner:
    # Combine stderr into result.output — print_info goes to stderr via
    # plain print(), which CliRunner still captures when mix_stderr=True.
    return CliRunner()


@pytest.fixture
def cli_app():
    return create_app()


def _make_supabase_response() -> MyConfigResponse:
    return MyConfigResponse(
        success=True,
        storage_type="supabase",
        storage_config={
            "url": "https://jpkjckbyxrdefzomiyse.supabase.co",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.verysecrettoken",
            "db_url": "postgresql://postgres.abc:pw@host.supabase.com:6543/postgres",
        },
    )


class TestConfigStorage:
    def test_masks_by_default(self, runner: CliRunner, cli_app) -> None:
        mock_client = MagicMock()
        mock_client.get_my_config.return_value = _make_supabase_response()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(cli_app, ["config", "storage"])
        assert result.exit_code == 0, result.output
        # The full key must never appear in any captured output
        assert "verysecrettoken" not in result.output
        # Masked form appears
        assert "supabase" in result.output

    def test_reveal_requires_confirmation(self, runner: CliRunner, cli_app) -> None:
        mock_client = MagicMock()
        mock_client.get_my_config.return_value = _make_supabase_response()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            # Decline the confirmation prompt → raises typer.Abort
            result = runner.invoke(
                cli_app, ["config", "storage", "--reveal"], input="n\n"
            )
        # Abort exit code is 1
        assert result.exit_code != 0

    def test_reveal_confirmed_prints_raw(self, runner: CliRunner, cli_app) -> None:
        mock_client = MagicMock()
        mock_client.get_my_config.return_value = _make_supabase_response()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app, ["config", "storage", "--reveal"], input="y\n"
            )
        assert result.exit_code == 0
        assert "verysecrettoken" in result.output


class TestConfigLocal:
    """Tests for ``reflexio config local`` — reads local config, no server needed."""

    _PATCH_LOAD = "reflexio.cli.bootstrap_config.load_storage_from_config"
    _PATCH_RESOLVE = "reflexio.cli.bootstrap_config.resolve_storage"

    def test_human_readable_output(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value="sqlite"),
            patch(self._PATCH_RESOLVE, return_value="sqlite"),
        ):
            result = runner.invoke(cli_app, ["config", "local"])
        assert result.exit_code == 0, result.output
        assert "Persisted storage: sqlite" in result.output
        assert "Resolved storage:  sqlite" in result.output
        assert "mode: local" in result.output

    def test_human_readable_no_persisted(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value=None),
            patch(self._PATCH_RESOLVE, return_value="sqlite"),
        ):
            result = runner.invoke(cli_app, ["config", "local"])
        assert result.exit_code == 0, result.output
        assert "(not set)" in result.output

    def test_json_mode(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value="supabase"),
            patch(self._PATCH_RESOLVE, return_value="supabase"),
        ):
            result = runner.invoke(cli_app, ["--json", "config", "local"])
        assert result.exit_code == 0, result.output
        import json

        envelope = json.loads(result.output)
        assert envelope["ok"] is True
        data = envelope["data"]
        assert data["persisted_storage"] == "supabase"
        assert data["resolved_storage"] == "supabase"
        assert data["resolved_mode"] == "cloud"
        assert "config_file" in data

    def test_cloud_mode_for_supabase(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value="supabase"),
            patch(self._PATCH_RESOLVE, return_value="supabase"),
        ):
            result = runner.invoke(cli_app, ["config", "local"])
        assert result.exit_code == 0, result.output
        assert "mode: cloud" in result.output

    def test_local_mode_for_sqlite(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value="sqlite"),
            patch(self._PATCH_RESOLVE, return_value="sqlite"),
        ):
            result = runner.invoke(cli_app, ["config", "local"])
        assert result.exit_code == 0, result.output
        assert "mode: local" in result.output


class TestConfigUpdate:
    """Tests for ``reflexio config update`` (PATCH-style partial update).

    The command builds a ``partial`` dict from one of three input modes
    (``--data``, ``--file``, or repeated ``--field``) and forwards it to
    ``client.update_config``. We mock the client and assert the dict.
    """

    @staticmethod
    def _mock_client_with_success() -> MagicMock:
        mock_client = MagicMock()
        mock_client.update_config.return_value = {
            "success": True,
            "msg": "Configuration set successfully",
        }
        return mock_client

    def test_data_partial(self, runner: CliRunner, cli_app) -> None:
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                ["config", "update", "--data", '{"window_size":8}'],
            )
        assert result.exit_code == 0, result.output
        mock_client.update_config.assert_called_once_with({"window_size": 8})

    def test_file_partial(self, runner: CliRunner, cli_app, tmp_path) -> None:
        cfg = tmp_path / "partial.json"
        cfg.write_text('{"window_size": 8}')
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(cli_app, ["config", "update", "--file", str(cfg)])
        assert result.exit_code == 0, result.output
        mock_client.update_config.assert_called_once_with({"window_size": 8})

    def test_field_pairs(self, runner: CliRunner, cli_app) -> None:
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                [
                    "config",
                    "update",
                    "--field",
                    "window_size=:json:8",
                    "--field",
                    "stride_size=:json:4",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_client.update_config.assert_called_once_with(
            {"window_size": 8, "stride_size": 4}
        )

    def test_field_json_value_int(self, runner: CliRunner, cli_app) -> None:
        """Numeric / bool / null literals come through ``:json:`` prefix."""
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                ["config", "update", "--field", "window_size=:json:10"],
            )
        assert result.exit_code == 0, result.output
        mock_client.update_config.assert_called_once_with({"window_size": 10})
        # The value must be a real int, not the string "10".
        sent = mock_client.update_config.call_args.args[0]
        assert isinstance(sent["window_size"], int)

    def test_field_json_value_bool(self, runner: CliRunner, cli_app) -> None:
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                ["config", "update", "--field", "skip_should_run_check=:json:true"],
            )
        assert result.exit_code == 0, result.output
        mock_client.update_config.assert_called_once_with(
            {"skip_should_run_check": True}
        )

    def test_field_dotted_path_one_level(self, runner: CliRunner, cli_app) -> None:
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                [
                    "config",
                    "update",
                    "--field",
                    "llm_config.embedding_model_name=local/minilm-l6-v2",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_client.update_config.assert_called_once_with(
            {"llm_config": {"embedding_model_name": "local/minilm-l6-v2"}}
        )

    def test_field_dotted_paths_merge_under_same_top(
        self, runner: CliRunner, cli_app
    ) -> None:
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                [
                    "config",
                    "update",
                    "--field",
                    "llm_config.embedding_model_name=local/minilm-l6-v2",
                    "--field",
                    "llm_config.extraction_agent=anthropic/claude-3-5-sonnet",
                ],
            )
        assert result.exit_code == 0, result.output
        mock_client.update_config.assert_called_once_with(
            {
                "llm_config": {
                    "embedding_model_name": "local/minilm-l6-v2",
                    "extraction_agent": "anthropic/claude-3-5-sonnet",
                }
            }
        )

    def test_field_double_dot_rejected(self, runner: CliRunner, cli_app) -> None:
        """``a.b.c`` is rejected — only one level of nesting via --field."""
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                ["config", "update", "--field", "a.b.c=value"],
            )
        assert result.exit_code != 0
        mock_client.update_config.assert_not_called()

    def test_field_missing_equals_rejected(self, runner: CliRunner, cli_app) -> None:
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app, ["config", "update", "--field", "no_equals_here"]
            )
        assert result.exit_code != 0
        mock_client.update_config.assert_not_called()

    def test_no_args_rejected(self, runner: CliRunner, cli_app) -> None:
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(cli_app, ["config", "update"])
        assert result.exit_code != 0
        mock_client.update_config.assert_not_called()

    def test_mixed_sources_rejected(self, runner: CliRunner, cli_app) -> None:
        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                [
                    "config",
                    "update",
                    "--data",
                    '{"a":1}',
                    "--field",
                    "b=2",
                ],
            )
        assert result.exit_code != 0
        mock_client.update_config.assert_not_called()

    def test_json_mode_renders_envelope(self, runner: CliRunner, cli_app) -> None:
        import json as _json

        mock_client = self._mock_client_with_success()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                [
                    "--json",
                    "config",
                    "update",
                    "--field",
                    "window_size=:json:8",
                ],
            )
        assert result.exit_code == 0, result.output
        envelope = _json.loads(result.output)
        assert envelope["ok"] is True


_GET_CLIENT = "reflexio.cli.commands.config_cmd.get_client"


class TestConfigSet:
    """``reflexio config set`` input modes and their validation."""

    @staticmethod
    def _invoke(runner: CliRunner, cli_app, args: list[str]):
        mock_client = MagicMock()
        mock_client.set_config.return_value = {"success": True}
        with patch(_GET_CLIENT, return_value=mock_client):
            result = runner.invoke(cli_app, ["config", "set", *args])
        return result, mock_client

    def test_requires_a_source(self, runner: CliRunner, cli_app) -> None:
        result, client = self._invoke(runner, cli_app, [])
        assert result.exit_code == 2
        assert "Must provide either --data or --file" in result.output
        client.set_config.assert_not_called()

    def test_rejects_both_sources(self, runner: CliRunner, cli_app, tmp_path) -> None:
        cfg = tmp_path / "c.json"
        cfg.write_text("{}")
        result, client = self._invoke(
            runner, cli_app, ["--data", "{}", "--file", str(cfg)]
        )
        assert result.exit_code == 2
        assert "Cannot provide both" in result.output
        client.set_config.assert_not_called()

    def test_inline_data(self, runner: CliRunner, cli_app) -> None:
        result, client = self._invoke(runner, cli_app, ["--data", '{"window_size":4}'])
        assert result.exit_code == 0, result.output
        client.set_config.assert_called_once_with({"window_size": 4})
        assert "Configuration updated" in result.output

    def test_at_file_data(self, runner: CliRunner, cli_app, tmp_path) -> None:
        cfg = tmp_path / "c.json"
        cfg.write_text('{"window_size": 6}')
        result, client = self._invoke(runner, cli_app, ["--data", f"@{cfg}"])
        assert result.exit_code == 0, result.output
        client.set_config.assert_called_once_with({"window_size": 6})

    def test_file_option_json_mode(self, runner: CliRunner, cli_app, tmp_path) -> None:
        import json as _json

        cfg = tmp_path / "c.json"
        cfg.write_text('{"window_size": 2}')
        mock_client = MagicMock()
        mock_client.set_config.return_value = {"success": True}
        with patch(_GET_CLIENT, return_value=mock_client):
            result = runner.invoke(
                cli_app, ["--json", "config", "set", "--file", str(cfg)]
            )
        assert result.exit_code == 0, result.output
        mock_client.set_config.assert_called_once_with({"window_size": 2})
        assert _json.loads(result.stdout)["data"] == {"success": True}

    @pytest.mark.parametrize("bad", ["{not json", "@/nonexistent/reflexio-cfg.json"])
    def test_unparseable_data_is_validation_error(
        self, runner: CliRunner, cli_app, bad: str
    ) -> None:
        result, client = self._invoke(runner, cli_app, ["--data", bad])
        assert result.exit_code == 2
        assert "Failed to parse config data" in result.output
        client.set_config.assert_not_called()


class TestConfigUpdateValidation:
    """Malformed ``config update`` inputs never reach the server."""

    @staticmethod
    def _invoke(runner: CliRunner, cli_app, args: list[str]):
        mock_client = MagicMock()
        with patch(_GET_CLIENT, return_value=mock_client):
            result = runner.invoke(cli_app, ["config", "update", *args])
        return result, mock_client

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            (["window_size=:json:{bad"], "invalid JSON value"),
            (
                ["storage_config=x", "storage_config.type=sqlite"],
                "conflicts with prior",
            ),
            (
                ["storage_config.type=sqlite", "storage_config=x"],
                "conflicts with prior",
            ),
        ],
    )
    def test_bad_fields(
        self, runner: CliRunner, cli_app, fields: list[str], message: str
    ) -> None:
        args: list[str] = []
        for spec in fields:
            args += ["--field", spec]
        result, client = self._invoke(runner, cli_app, args)
        assert result.exit_code == 2
        assert message in result.output
        client.update_config.assert_not_called()

    def test_missing_file(self, runner: CliRunner, cli_app, tmp_path) -> None:
        result, client = self._invoke(
            runner, cli_app, ["--file", str(tmp_path / "absent.json")]
        )
        assert result.exit_code == 2
        assert "Failed to parse config data" in result.output
        client.update_config.assert_not_called()

    def test_non_object_partial(self, runner: CliRunner, cli_app) -> None:
        result, client = self._invoke(runner, cli_app, ["--data", "[1, 2]"])
        assert result.exit_code == 2
        assert "Partial config must be a JSON object" in result.output
        client.update_config.assert_not_called()


class TestConfigStorageEdges:
    """``config storage`` transport failures, empty configs and JSON masking."""

    @staticmethod
    def _invoke(runner: CliRunner, cli_app, client: MagicMock, *args: str):
        with patch(_GET_CLIENT, return_value=client):
            return runner.invoke(cli_app, [*args, "config", "storage"])

    @staticmethod
    def _http_error(status: int):
        import requests

        response = requests.Response()
        response.status_code = status
        return requests.HTTPError(f"{status} error", response=response)

    def test_missing_endpoint_is_actionable(self, runner: CliRunner, cli_app) -> None:
        client = MagicMock(base_url="http://old.reflexio.test")
        client.get_my_config.side_effect = self._http_error(404)
        result = self._invoke(runner, cli_app, client)
        assert result.exit_code == 4
        assert "doesn't expose this endpoint" in result.output
        assert "http://old.reflexio.test/api/my_config" in result.output

    def test_other_http_error_is_network_error(
        self, runner: CliRunner, cli_app
    ) -> None:
        client = MagicMock(base_url="http://reflexio.test")
        client.get_my_config.side_effect = self._http_error(503)
        result = self._invoke(runner, cli_app, client)
        assert result.exit_code == 4
        assert "Failed to reach http://reflexio.test/api/my_config" in result.output

    def test_connection_error_is_network_error(
        self, runner: CliRunner, cli_app
    ) -> None:
        import requests

        client = MagicMock(base_url="http://reflexio.test")
        client.get_my_config.side_effect = requests.ConnectionError("refused")
        result = self._invoke(runner, cli_app, client)
        assert result.exit_code == 4
        assert "Failed to reach http://reflexio.test/api/my_config" in result.output

    @pytest.mark.parametrize("json_mode", [False, True])
    def test_no_storage_configured(
        self, runner: CliRunner, cli_app, json_mode: bool
    ) -> None:
        import json as _json

        client = MagicMock()
        client.get_my_config.return_value = MyConfigResponse(
            success=False, message="Org has no storage"
        )
        args = ("--json",) if json_mode else ()
        result = self._invoke(runner, cli_app, client, *args)
        assert result.exit_code == 0, result.output
        if json_mode:
            data = _json.loads(result.stdout)["data"]
            assert data["success"] is False
            assert data["message"] == "Org has no storage"
        else:
            assert "Error: Org has no storage" in result.output

    def test_json_mode_masks_and_keeps_non_string_values(
        self, runner: CliRunner, cli_app
    ) -> None:
        import json as _json

        secret = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.verysecrettoken"
        client = MagicMock()
        client.get_my_config.return_value = MyConfigResponse(
            success=True,
            storage_type="supabase",
            storage_config={
                "url": "https://abc.supabase.co",
                "key": secret,
                "pool_size": 5,
                "schema": None,
            },
        )
        # --reveal is ignored in JSON mode: no prompt, still masked.
        with patch(_GET_CLIENT, return_value=client):
            result = runner.invoke(cli_app, ["--json", "config", "storage", "--reveal"])
        assert result.exit_code == 0, result.output
        assert "verysecrettoken" not in result.stdout
        data = _json.loads(result.stdout)["data"]
        assert data["storage_type"] == "supabase"
        assert data["storage_config"]["pool_size"] == 5
        assert data["storage_config"]["schema"] is None
        assert data["storage_config"]["key"] != secret
