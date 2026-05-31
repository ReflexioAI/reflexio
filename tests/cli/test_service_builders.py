"""Tests for shared CLI service builder functions."""

from __future__ import annotations

import argparse
import os
import sys
from unittest.mock import patch

import pytest
import typer

from reflexio.cli.commands.services import validate_storage_backend
from reflexio.cli.run_services import (
    _ensure_nextjs_dependencies,
    build_backend_service,
    build_embedding_service,
    build_nextjs_service,
    parse_only_flag,
    resolve_ports,
    should_start_local_embedding_service,
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
        svc = build_backend_service({"backend": 8081}, reload=True, workers=1)
        assert "--reload" in svc.command

    def test_build_embedding_service_uses_embedding_daemon(self) -> None:
        svc = build_embedding_service({"embedding": 8072})

        assert svc.name == "embedding"
        assert "reflexio.server.llm.embedding_service:app" in svc.command
        assert svc.env == {"REFLEXIO_EMBEDDING_DAEMON": "1"}

    def test_should_start_local_embedding_service_for_claude_smart(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")

        assert should_start_local_embedding_service() is True

    def test_should_not_start_local_embedding_service_for_true_string(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "true")

        assert should_start_local_embedding_service() is False

    def test_should_not_start_local_embedding_service_for_internal(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "internal_service")
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")

        assert should_start_local_embedding_service() is False

    def test_reload_false_omits_reload_flag(self) -> None:
        svc = build_backend_service({"backend": 8081}, reload=False)
        assert "--reload" not in svc.command

    def test_reload_includes_appended(self) -> None:
        svc = build_backend_service(
            {"backend": 8081},
            reload=True,
            workers=1,
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
        app_idx = svc.command.index("--app")
        assert svc.command[app_idx + 1] == "myapp:app"

    def test_port_from_dict(self) -> None:
        svc = build_backend_service({"backend": 9999}, reload=False)
        port_idx = svc.command.index("--port")
        assert svc.command[port_idx + 1] == "9999"

    def test_service_name(self) -> None:
        svc = build_backend_service({"backend": 8081}, reload=False)
        assert svc.name == "backend"

    def test_default_app_module(self) -> None:
        svc = build_backend_service({"backend": 8081}, reload=False)
        app_idx = svc.command.index("--app")
        assert svc.command[app_idx + 1] == "reflexio.server.api:app"

    def test_host_is_all_interfaces(self) -> None:
        svc = build_backend_service({"backend": 8081}, reload=False)
        host_idx = svc.command.index("--host")
        assert svc.command[host_idx + 1] == "0.0.0.0"  # noqa: S104

    def test_uses_python_module_entrypoint(self) -> None:
        """Backend launches via ``python -m reflexio.server`` so the uvicorn
        log config in :mod:`reflexio.server.uvicorn_logging` is applied."""
        svc = build_backend_service({"backend": 8081}, reload=False)
        assert svc.command[0] == sys.executable
        assert svc.command[1:3] == ["-m", "reflexio.server"]


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

    def test_ensure_nextjs_dependencies_skips_existing_node_modules(
        self, tmp_path, monkeypatch
    ) -> None:
        project_dir = tmp_path / "docs"
        (project_dir / "node_modules").mkdir(parents=True)
        run_calls = []
        monkeypatch.setattr(
            "reflexio.cli.run_services.subprocess.run",
            lambda *args, **kwargs: run_calls.append((args, kwargs)),
        )

        assert _ensure_nextjs_dependencies(project_dir) is True
        assert run_calls == []

    def test_ensure_nextjs_dependencies_runs_npm_install(
        self, tmp_path, monkeypatch
    ) -> None:
        project_dir = tmp_path / "docs"
        project_dir.mkdir()
        run_calls = []

        def fake_run(*args, **kwargs):
            run_calls.append((args, kwargs))
            return argparse.Namespace(returncode=0)

        monkeypatch.setattr("reflexio.cli.run_services.subprocess.run", fake_run)

        assert _ensure_nextjs_dependencies(project_dir) is True
        assert run_calls == [
            ((["npm", "install"],), {"cwd": str(project_dir), "check": False})
        ]

    def test_ensure_nextjs_dependencies_reports_install_failure(
        self, tmp_path, monkeypatch
    ) -> None:
        project_dir = tmp_path / "docs"
        project_dir.mkdir()

        def fake_run(*_args, **_kwargs):
            return argparse.Namespace(returncode=1)

        monkeypatch.setattr("reflexio.cli.run_services.subprocess.run", fake_run)

        assert _ensure_nextjs_dependencies(project_dir) is False


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
            validate_storage_backend("not-a-real-backend")


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
