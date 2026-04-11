"""Tests for shared CLI service builder functions."""

from __future__ import annotations

import argparse
import os
from unittest.mock import patch

import pytest
import typer

from reflexio.cli.commands.services import validate_storage_backend
from reflexio.cli.run_services import (
    build_backend_service,
    build_nextjs_service,
    parse_only_flag,
    resolve_ports,
)
from reflexio.cli.stop_services import build_stop_targets

# ---------------------------------------------------------------------------
# resolve_ports
# ---------------------------------------------------------------------------


class TestResolvePorts:
    """Tests for resolve_ports()."""

    def test_cli_arg_wins_over_default(self) -> None:
        args = argparse.Namespace(backend_port=9090)
        result = resolve_ports(args, {"backend": 8081})
        assert result["backend"] == 9090

    def test_cli_arg_wins_over_env(self) -> None:
        args = argparse.Namespace(backend_port=9090)
        with patch.dict(os.environ, {"BACKEND_PORT": "7777"}):
            result = resolve_ports(args, {"backend": 8081})
        assert result["backend"] == 9090

    def test_none_arg_falls_to_env(self) -> None:
        args = argparse.Namespace(backend_port=None)
        with patch.dict(os.environ, {"BACKEND_PORT": "7777"}):
            result = resolve_ports(args, {"backend": 8081})
        assert result["backend"] == 7777

    def test_none_arg_no_env_falls_to_default(self) -> None:
        args = argparse.Namespace(backend_port=None)
        with patch.dict(os.environ, {}, clear=False):
            # Ensure BACKEND_PORT is not in the environment
            env_copy = {k: v for k, v in os.environ.items() if k != "BACKEND_PORT"}
            with patch.dict(os.environ, env_copy, clear=True):
                result = resolve_ports(args, {"backend": 8081})
        assert result["backend"] == 8081

    def test_env_var_naming_convention(self) -> None:
        """Environment variable name is {NAME_UPPER}_PORT."""
        args = argparse.Namespace(docs_port=None)
        with patch.dict(os.environ, {"DOCS_PORT": "4000"}, clear=False):
            result = resolve_ports(args, {"docs": 3000})
        assert result["docs"] == 4000

    def test_multiple_services(self) -> None:
        args = argparse.Namespace(backend_port=9090, docs_port=None)
        with patch.dict(os.environ, {"DOCS_PORT": "4000"}, clear=False):
            result = resolve_ports(args, {"backend": 8081, "docs": 3000})
        assert result["backend"] == 9090
        assert result["docs"] == 4000

    def test_missing_attribute_falls_to_env_or_default(self) -> None:
        """When the Namespace has no matching attribute, getattr returns None."""
        args = argparse.Namespace()  # no backend_port attribute
        env = {k: v for k, v in os.environ.items() if k != "BACKEND_PORT"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_ports(args, {"backend": 8081})
        assert result["backend"] == 8081


# ---------------------------------------------------------------------------
# parse_only_flag
# ---------------------------------------------------------------------------


class TestParseOnlyFlag:
    """Tests for parse_only_flag()."""

    def test_none_returns_defaults(self) -> None:
        defaults = {"backend", "docs"}
        assert parse_only_flag(None, defaults) == defaults

    def test_comma_separated_parsed_correctly(self) -> None:
        result = parse_only_flag("backend,docs", {"backend", "docs"})
        assert result == {"backend", "docs"}

    def test_single_service(self) -> None:
        result = parse_only_flag("backend", {"backend", "docs"})
        assert result == {"backend"}

    def test_whitespace_trimmed(self) -> None:
        result = parse_only_flag("  backend , docs  ", {"backend", "docs"})
        assert result == {"backend", "docs"}

    def test_empty_string_returns_empty_set(self) -> None:
        """An empty string is falsy, so falls through to defaults."""
        defaults = {"backend", "docs"}
        result = parse_only_flag("", defaults)
        assert result == defaults


# ---------------------------------------------------------------------------
# build_backend_service
# ---------------------------------------------------------------------------


class TestBuildBackendService:
    """Tests for build_backend_service()."""

    def test_reload_true_adds_reload_flag(self) -> None:
        svc = build_backend_service({"backend": 8081}, reload=True)
        assert "--reload" in svc.command

    def test_reload_false_omits_reload_flag(self) -> None:
        svc = build_backend_service({"backend": 8081}, reload=False)
        assert "--reload" not in svc.command

    def test_reload_includes_appended(self) -> None:
        svc = build_backend_service(
            {"backend": 8081},
            reload=True,
            reload_includes=["*.json", "*.yaml"],
        )
        assert "--reload-include" in svc.command
        json_idx = svc.command.index("--reload-include")
        assert svc.command[json_idx + 1] == "*.json"
        # Second pattern
        remaining = svc.command[json_idx + 2 :]
        assert "--reload-include" in remaining
        yaml_idx = remaining.index("--reload-include")
        assert remaining[yaml_idx + 1] == "*.yaml"

    def test_reload_includes_without_reload_not_added(self) -> None:
        svc = build_backend_service(
            {"backend": 8081},
            reload=False,
            reload_includes=["*.json"],
        )
        assert "--reload-include" not in svc.command

    def test_custom_app_module(self) -> None:
        svc = build_backend_service(
            {"backend": 8081},
            app_module="myapp:app",
            reload=False,
        )
        assert "myapp:app" in svc.command

    def test_port_from_dict(self) -> None:
        svc = build_backend_service({"backend": 9999}, reload=False)
        port_idx = svc.command.index("--port")
        assert svc.command[port_idx + 1] == "9999"

    def test_service_name(self) -> None:
        svc = build_backend_service({"backend": 8081}, reload=False)
        assert svc.name == "backend"

    def test_default_app_module(self) -> None:
        svc = build_backend_service({"backend": 8081}, reload=False)
        assert "reflexio.server.api:app" in svc.command

    def test_host_is_all_interfaces(self) -> None:
        svc = build_backend_service({"backend": 8081}, reload=False)
        host_idx = svc.command.index("--host")
        assert svc.command[host_idx + 1] == "0.0.0.0"  # noqa: S104


# ---------------------------------------------------------------------------
# build_nextjs_service
# ---------------------------------------------------------------------------


class TestBuildNextjsService:
    """Tests for build_nextjs_service()."""

    def test_correct_command(self) -> None:
        svc = build_nextjs_service("docs", {"docs": 3000}, cwd="public_docs")
        assert svc.command == ["npx", "next", "dev", "-p", "3000"]

    def test_correct_cwd(self) -> None:
        svc = build_nextjs_service("docs", {"docs": 3000}, cwd="public_docs")
        assert svc.cwd == "public_docs"

    def test_service_name(self) -> None:
        svc = build_nextjs_service("frontend", {"frontend": 8080}, cwd="website")
        assert svc.name == "frontend"

    def test_port_from_dict(self) -> None:
        svc = build_nextjs_service("docs", {"docs": 4567}, cwd="docs")
        assert svc.command[-1] == "4567"


# ---------------------------------------------------------------------------
# validate_storage_backend
# ---------------------------------------------------------------------------


class TestValidateStorageBackend:
    """Tests for validate_storage_backend()."""

    def test_none_is_noop(self) -> None:
        env_before = os.environ.get("REFLEXIO_STORAGE")
        validate_storage_backend(None)
        assert os.environ.get("REFLEXIO_STORAGE") == env_before

    def test_sqlite_sets_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            validate_storage_backend("sqlite")
            assert os.environ["REFLEXIO_STORAGE"] == "sqlite"

    def test_supabase_sets_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            validate_storage_backend("supabase")
            assert os.environ["REFLEXIO_STORAGE"] == "supabase"

    def test_case_insensitive(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            validate_storage_backend("SQLite")
            assert os.environ["REFLEXIO_STORAGE"] == "sqlite"

    def test_invalid_raises_bad_parameter(self) -> None:
        with pytest.raises(typer.BadParameter, match="Invalid storage backend"):
            validate_storage_backend("postgres")


# ---------------------------------------------------------------------------
# build_stop_targets
# ---------------------------------------------------------------------------


class TestBuildStopTargets:
    """Tests for build_stop_targets()."""

    def test_backend_only(self) -> None:
        ports = {"backend": 8081, "docs": 3000}
        port_map, patterns = build_stop_targets({"backend"}, ports)
        assert port_map == {"backend": 8081}
        assert "backend" in patterns
        assert "docs" not in port_map
        assert "docs" not in patterns

    def test_docs_only(self) -> None:
        ports = {"backend": 8081, "docs": 3000}
        port_map, patterns = build_stop_targets({"docs"}, ports)
        assert port_map == {"docs": 3000}
        assert "docs" in patterns
        assert "3000" in patterns["docs"]
        assert "backend" not in port_map

    def test_both_services(self) -> None:
        ports = {"backend": 8081, "docs": 3000}
        port_map, patterns = build_stop_targets({"backend", "docs"}, ports)
        assert port_map == {"backend": 8081, "docs": 3000}
        assert "backend" in patterns
        assert "docs" in patterns

    def test_custom_backend_pattern(self) -> None:
        ports = {"backend": 8081, "docs": 3000}
        port_map, patterns = build_stop_targets(
            {"backend"},
            ports,
            backend_pattern="uvicorn myapp:app",
        )
        assert patterns["backend"] == "uvicorn myapp:app"

    def test_docs_pattern_includes_port(self) -> None:
        ports = {"backend": 8081, "docs": 5555}
        _, patterns = build_stop_targets({"docs"}, ports)
        assert "5555" in patterns["docs"]

    def test_empty_only_returns_empty(self) -> None:
        ports = {"backend": 8081, "docs": 3000}
        port_map, patterns = build_stop_targets(set(), ports)
        assert port_map == {}
        assert patterns == {}
