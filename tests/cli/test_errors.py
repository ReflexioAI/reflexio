"""Error handling unit tests for reflexio.cli.errors."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import requests

from reflexio.cli.errors import (
    EXIT_AUTH,
    EXIT_GENERAL,
    EXIT_NETWORK,
    EXIT_VALIDATION,
    CliError,
    _classify_http_error,
    handle_errors,
    render_error,
)


class TestCliError:
    """Tests for CliError construction."""

    def test_construction_defaults(self) -> None:
        err = CliError()
        assert err.error_type == "general"
        assert err.message == "An error occurred"
        assert err.hint is None
        assert err.exit_code == EXIT_GENERAL

    def test_construction_custom(self) -> None:
        err = CliError(
            error_type="auth",
            message="Bad token",
            hint="Re-authenticate",
            exit_code=EXIT_AUTH,
        )
        assert err.error_type == "auth"
        assert err.message == "Bad token"
        assert err.hint == "Re-authenticate"
        assert err.exit_code == EXIT_AUTH

    def test_str_is_message(self) -> None:
        err = CliError(message="Test msg")
        assert str(err) == "Test msg"


class TestRenderError:
    """Tests for render_error() output."""

    def test_json_mode(self, capsys) -> None:
        err = CliError(
            error_type="validation",
            message="bad input",
            hint="fix it",
            exit_code=EXIT_VALIDATION,
        )
        render_error(err, json_mode=True)
        captured = capsys.readouterr()
        envelope = json.loads(captured.err)
        assert envelope["ok"] is False
        assert envelope["error"]["type"] == "validation"
        assert envelope["error"]["message"] == "bad input"
        assert envelope["error"]["hint"] == "fix it"

    def test_human_mode(self, capsys) -> None:
        err = CliError(
            error_type="network",
            message="Connection refused",
            hint="Start the server",
        )
        render_error(err, json_mode=False)
        captured = capsys.readouterr()
        assert "Error: Connection refused" in captured.err
        assert "Hint: Start the server" in captured.err

    def test_human_mode_no_hint(self, capsys) -> None:
        err = CliError(message="Something broke")
        render_error(err, json_mode=False)
        captured = capsys.readouterr()
        assert "Error: Something broke" in captured.err
        assert "Hint:" not in captured.err


class TestHandleErrors:
    """Tests for the handle_errors decorator."""

    def test_catches_cli_error(self) -> None:
        @handle_errors
        def boom():
            raise CliError(message="cli boom", exit_code=EXIT_VALIDATION)

        try:
            boom()
            raised = False
        except SystemExit as exc:
            raised = True
            assert exc.code == EXIT_VALIDATION
        assert raised, "Expected SystemExit"

    def test_catches_connection_error(self) -> None:
        @handle_errors
        def boom():
            raise requests.ConnectionError("refused")

        try:
            boom()
            raised = False
        except SystemExit as exc:
            raised = True
            assert exc.code == EXIT_NETWORK
        assert raised, "Expected SystemExit"


class TestClassifyHttpError:
    """Tests for _classify_http_error()."""

    def _make_http_error(self, status_code: int) -> requests.HTTPError:
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = {"detail": "test detail"}
        return requests.HTTPError(response=response)

    def test_401_maps_to_auth(self) -> None:
        err = _classify_http_error(self._make_http_error(401))
        assert err.error_type == "auth"
        assert err.exit_code == EXIT_AUTH

    def test_403_maps_to_auth(self) -> None:
        err = _classify_http_error(self._make_http_error(403))
        assert err.error_type == "auth"
        assert err.exit_code == EXIT_AUTH

    def test_422_maps_to_validation(self) -> None:
        err = _classify_http_error(self._make_http_error(422))
        assert err.error_type == "validation"
        assert err.exit_code == EXIT_VALIDATION

    def test_404_maps_to_general(self) -> None:
        err = _classify_http_error(self._make_http_error(404))
        assert err.error_type == "general"
        assert err.exit_code == EXIT_GENERAL

    def test_429_maps_to_rate_limit(self) -> None:
        err = _classify_http_error(self._make_http_error(429))
        assert err.error_type == "rate_limit"
        assert err.exit_code == EXIT_GENERAL
