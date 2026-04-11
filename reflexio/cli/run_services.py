"""Run reflexio services (backend + docs)."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import reflexio
from reflexio.cli.env_loader import load_reflexio_env
from reflexio.cli.utils import ServiceConfig, get_env_port, run_services

_PACKAGE_DIR = Path(reflexio.__file__).resolve().parent
# Editable installs lay out as `…/open_source/reflexio/reflexio/__init__.py`,
# so the repo root is one level above the package dir. PyPI/wheel installs
# resolve to `…/site-packages/`, where DOCS_DIR will not exist and the
# docs service is silently skipped (see execute()).
_EDITABLE_REPO_ROOT = _PACKAGE_DIR.parent
DOCS_DIR = _EDITABLE_REPO_ROOT / "docs"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add run-services arguments to the parser."""
    parser.add_argument(
        "--backend-port",
        type=int,
        default=None,
        help="Backend port (default: 8081, env: BACKEND_PORT)",
    )
    parser.add_argument(
        "--docs-port",
        type=int,
        default=None,
        help="Docs port (default: 8082, env: DOCS_PORT)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of services to start: backend,docs",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable uvicorn auto-reload",
    )


def resolve_ports(
    args: argparse.Namespace,
    defaults: dict[str, int],
) -> dict[str, int]:
    """Resolve service ports from CLI args, env vars, or defaults.

    Priority: CLI arg > env var > default value.

    Args:
        args: Parsed CLI arguments (looks for `{name}_port` attributes)
        defaults: Map of service name to default port number

    Returns:
        dict[str, int]: Resolved ports keyed by service name
    """
    ports = {}
    for name, default in defaults.items():
        arg_val = getattr(args, f"{name}_port", None)
        ports[name] = (
            arg_val
            if arg_val is not None
            else get_env_port(f"{name.upper()}_PORT", default)
        )
    return ports


def build_backend_service(
    ports: dict[str, int],
    *,
    app_module: str = "reflexio.server.api:app",
    reload: bool = True,
    reload_includes: list[str] | None = None,
) -> ServiceConfig:
    """Build a backend ServiceConfig.

    Args:
        ports: Resolved port map (must contain "backend" key)
        app_module: Uvicorn app module path
        reload: Whether to enable auto-reload
        reload_includes: Additional glob patterns for reload watching

    Returns:
        ServiceConfig: Backend service configuration
    """
    cmd = [
        "uvicorn",
        app_module,
        "--host",
        "0.0.0.0",
        "--port",
        str(ports["backend"]),
    ]
    if reload:
        includes = reload_includes or []
        for pattern in includes:
            cmd.extend(["--reload-include", pattern])
        cmd.append("--reload")
    return ServiceConfig(name="backend", command=cmd)


def build_nextjs_service(
    name: str,
    ports: dict[str, int],
    *,
    cwd: str,
) -> ServiceConfig:
    """Build a Next.js dev server ServiceConfig.

    Args:
        name: Service name (e.g., "docs", "frontend")
        ports: Resolved port map (must contain key matching name)
        cwd: Working directory for the Next.js project

    Returns:
        ServiceConfig: Next.js service configuration
    """
    return ServiceConfig(
        name=name,
        command=["npx", "next", "dev", "-p", str(ports[name])],
        cwd=cwd,
    )


def parse_only_flag(only: str | None, default_services: set[str]) -> set[str]:
    """Parse the --only flag into a set of service names.

    Args:
        only: Comma-separated service names, or None for all defaults
        default_services: Set of services to start when --only is not specified

    Returns:
        set[str]: Services to start
    """
    if only:
        return {s.strip() for s in only.split(",")}
    return default_services


def execute(args: argparse.Namespace) -> None:
    """Execute the run-services command."""
    load_reflexio_env()

    # Banner so any captured stdout (e.g. ~/.reflexio/logs/server.log via the
    # Claude Code session-start hook) clearly marks the start of a new server
    # lifetime instead of blurring into output from a previous run.
    bar = "=" * 64
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\n{bar}\n=== NEW REFLEXIO SERVER START — {ts} ===\n{bar}\n")

    ports = resolve_ports(args, defaults={"backend": 8081, "docs": 8082})
    os.environ["API_BACKEND_URL"] = os.environ.get(
        "API_BACKEND_URL", f"http://localhost:{ports['backend']}"
    )

    only = parse_only_flag(args.only, {"backend", "docs"})
    docs_explicit = args.only is not None and "docs" in only
    services: list[ServiceConfig] = []

    if "backend" in only:
        services.append(
            build_backend_service(
                ports,
                reload=not args.no_reload,
                reload_includes=["reflexio/server/site_var/site_var_sources/*.json"],
            )
        )

    if "docs" in only:
        if DOCS_DIR.is_dir():
            services.append(
                build_nextjs_service("docs", ports, cwd=str(DOCS_DIR))
            )
        elif docs_explicit:
            print(
                f"Cannot start docs: {DOCS_DIR} not found. "
                "The docs site is only available in source/editable installs."
            )
        else:
            print(
                f"Skipping docs: {DOCS_DIR} not found "
                "(docs site is not shipped with the PyPI package)."
            )

    if not services:
        print("No services selected. Available: backend, docs")
        return

    started_ports = {s.name: ports[s.name] for s in services}
    print(f"Starting services: {', '.join(s.name for s in services)}")
    run_services(services, started_ports)
