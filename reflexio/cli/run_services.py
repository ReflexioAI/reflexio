"""Run reflexio services (backend + docs)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import reflexio
from reflexio.cli.env_loader import load_reflexio_env
from reflexio.cli.utils import ServiceConfig, get_env_port, run_services
from reflexio.server.llm.providers.embedding_service_provider import (
    embedding_service_url,
)

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(reflexio.__file__).resolve().parent
# Editable installs lay out as `…/open_source/reflexio/reflexio/__init__.py`,
# so the repo root is one level above the package dir. PyPI/wheel installs
# resolve to `…/site-packages/`, where DOCS_DIR will not exist and the
# docs service is silently skipped (see execute()).
_EDITABLE_REPO_ROOT = _PACKAGE_DIR.parent
DOCS_DIR = _EDITABLE_REPO_ROOT / "docs"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add run-services arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): The parser to populate with
            run-services flags. Mutated in-place.
    """
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
        "--embedding-port",
        type=int,
        default=None,
        help="Embedding service port (default: 8072, env: EMBEDDING_PORT)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of services to start: backend,docs,embedding",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable uvicorn auto-reload",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Number of backend worker processes (daemon mode only). "
            "Default 2 enables zero-downtime worker recycling. "
            "Forced to 1 when --reload is on."
        ),
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=10000,
        help="Worker recycles after this many requests. 0 disables. Default 10000.",
    )
    parser.add_argument(
        "--max-requests-jitter",
        type=int,
        default=1000,
        help="Random 0..jitter per worker. Default 1000.",
    )
    parser.add_argument(
        "--graceful-shutdown-sec",
        type=int,
        default=30,
        help="Drain window on shutdown. Default 30.",
    )


def _build_run_services_parser() -> argparse.ArgumentParser:
    """Build the standalone run_services CLI parser.

    The parent ``reflexio`` CLI normally builds the parser and calls
    :func:`add_arguments` to populate it. This helper exists so tests (and
    any ad-hoc ``python -m reflexio.cli.run_services``-style invocation)
    can construct a self-contained parser with all run-services flags.

    Returns:
        argparse.ArgumentParser: Parser populated with all run_services
        flags via :func:`add_arguments`.
    """
    parser = argparse.ArgumentParser(prog="reflexio.cli.run_services")
    add_arguments(parser)
    return parser


def _warn_if_sqlite_multi_worker(*, storage_backend: str, workers: int) -> None:
    """Emit a warning when SQLite is paired with multi-worker mode.

    SQLite supports concurrent reads but serializes writes (even in WAL
    mode). Multi-worker deployments will see increased write latency and
    occasional "database is locked" errors under contention. Operators are
    not blocked — just notified — so they can switch to Postgres/Supabase
    if they hit issues.

    Args:
        storage_backend (str): One of "sqlite", "postgres", "supabase".
        workers (int): Configured worker count.
    """
    if storage_backend == "sqlite" and workers > 1:
        logger.warning(
            "SQLite has limited concurrent write throughput; consider "
            "--workers 1 or switching to Postgres/Supabase. (workers=%d)",
            workers,
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
    workers: int = 2,
    max_requests: int = 10000,
    max_requests_jitter: int = 1000,
    graceful_shutdown_sec: int = 30,
) -> ServiceConfig:
    """Build a ServiceConfig describing how to spawn the backend.

    Args:
        ports (dict[str, int]): Resolved port map (must contain "backend").
        app_module (str): ASGI app path; defaults to the production app
            factory.
        reload (bool): When True, enables uvicorn autoreload (single
            worker forced). When False, daemon mode with multi-worker
            request-count recycling is used.
        reload_includes (list[str] | None): Additional glob patterns for
            reload watching. Only used when ``reload`` is True.
        workers (int): Number of uvicorn worker processes. Must be 1
            when ``reload`` is True (validated). Default 2.
        max_requests (int): Daemon-mode worker recycles after this many
            requests. Set to 0 to disable. Default 10000.
        max_requests_jitter (int): Random 0..jitter added to
            ``max_requests`` per worker. Default 1000.
        graceful_shutdown_sec (int): Drain window on shutdown. Default 30.

    Returns:
        ServiceConfig: Backend service configuration ready to spawn.

    Raises:
        ValueError: When ``reload=True`` and ``workers > 1``.
    """
    if reload and workers > 1:
        raise ValueError(
            "--workers N (N>1) is incompatible with --reload; "
            "pass --no-reload or set --workers 1"
        )
    # Launch via our own ``python -m reflexio.server`` entrypoint rather
    # than the ``uvicorn`` CLI so the log config in
    # :mod:`reflexio.server.uvicorn_logging` is applied upfront via
    # uvicorn's native ``log_config`` parameter.
    cmd = [
        sys.executable,
        "-m",
        "reflexio.server",
        "--app",
        app_module,
        "--host",
        "0.0.0.0",  # noqa: S104
        "--port",
        str(ports["backend"]),
    ]
    if reload:
        for pattern in reload_includes or []:
            cmd.extend(["--reload-include", pattern])
        cmd.append("--reload")
        # Dev mode is always single-worker; reload + multi-worker is
        # rejected above.
        cmd.extend(["--workers", "1"])
    else:
        cmd.extend(["--workers", str(workers)])
        cmd.extend(["--max-requests", str(max_requests)])
        cmd.extend(["--max-requests-jitter", str(max_requests_jitter)])
        cmd.extend(["--graceful-shutdown-sec", str(graceful_shutdown_sec)])
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


def build_embedding_service(ports: dict[str, int]) -> ServiceConfig:
    """Build the local embedding daemon service configuration."""
    return ServiceConfig(
        name="embedding",
        command=[
            sys.executable,
            "-m",
            "uvicorn",
            "reflexio.server.llm.embedding_service:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(ports["embedding"]),
        ],
        env={"REFLEXIO_EMBEDDING_DAEMON": "1"},
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


def should_start_local_embedding_service() -> bool:
    """Return True when backend startup depends on the local embedding daemon."""
    provider = os.environ.get("REFLEXIO_EMBEDDING_PROVIDER", "").strip().lower()
    if provider == "local_service":
        return True
    if provider in {"cloud", "internal_service", "inprocess", "off"}:
        return False
    return os.environ.get("CLAUDE_SMART_USE_LOCAL_EMBEDDING") == "1"


def execute(args: argparse.Namespace) -> None:
    """Execute the run-services command."""
    load_reflexio_env()

    # Banner so any captured stdout (e.g. ~/.reflexio/logs/server.log via the
    # Claude Code session-start hook) clearly marks the start of a new server
    # lifetime instead of blurring into output from a previous run.
    bar = "=" * 64
    ts = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\n{bar}\n=== NEW REFLEXIO SERVER START — {ts} ===\n{bar}\n")

    ports = resolve_ports(
        args, defaults={"backend": 8081, "docs": 8082, "embedding": 8072}
    )
    os.environ["API_BACKEND_URL"] = os.environ.get(
        "API_BACKEND_URL", f"http://localhost:{ports['backend']}"
    )
    os.environ["EMBEDDING_PORT"] = str(ports["embedding"])

    only = parse_only_flag(args.only, {"backend", "docs"})
    if "backend" in only and should_start_local_embedding_service():
        only.add("embedding")
        os.environ["REFLEXIO_EMBEDDING_PROVIDER"] = os.environ.get(
            "REFLEXIO_EMBEDDING_PROVIDER", "local_service"
        )
        os.environ["REFLEXIO_EMBEDDING_SERVICE_URL"] = os.environ.get(
            "REFLEXIO_EMBEDDING_SERVICE_URL",
            embedding_service_url("local_service"),
        )
    docs_explicit = args.only is not None and "docs" in only
    services: list[ServiceConfig] = []

    if "embedding" in only:
        services.append(build_embedding_service(ports))

    if "backend" in only:
        reload = not args.no_reload
        # Dev mode (--reload) is always single-worker; coerce silently so
        # users running with default flags (reload on, workers default 2)
        # don't hit the reload+multi-worker rejection in build_backend_service.
        workers = 1 if reload else getattr(args, "workers", 2)
        max_requests = getattr(args, "max_requests", 10000)
        max_requests_jitter = getattr(args, "max_requests_jitter", 1000)
        graceful_shutdown_sec = getattr(args, "graceful_shutdown_sec", 30)
        storage_backend = os.environ.get("REFLEXIO_STORAGE", "sqlite").lower()
        _warn_if_sqlite_multi_worker(storage_backend=storage_backend, workers=workers)
        services.append(
            build_backend_service(
                ports,
                reload=reload,
                reload_includes=["reflexio/server/site_var/site_var_sources/*.json"],
                workers=workers,
                max_requests=max_requests,
                max_requests_jitter=max_requests_jitter,
                graceful_shutdown_sec=graceful_shutdown_sec,
            )
        )

    if "docs" in only:
        if DOCS_DIR.is_dir():
            services.append(build_nextjs_service("docs", ports, cwd=str(DOCS_DIR)))
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
        print("No services selected. Available: backend, docs, embedding")
        return

    started_ports = {s.name: ports[s.name] for s in services}
    print(f"Starting services: {', '.join(s.name for s in services)}")
    run_services(services, started_ports)
