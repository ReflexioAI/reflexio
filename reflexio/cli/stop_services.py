"""Stop reflexio services."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from reflexio.cli.run_services import parse_only_flag, resolve_ports
from reflexio.cli.utils import stop_services


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add stop-services arguments to the parser."""
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
        help="Comma-separated list of services to stop: backend,docs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="SIGKILL immediately (skip graceful shutdown)",
    )


def build_stop_targets(
    only: set[str],
    ports: dict[str, int],
    *,
    backend_pattern: str = "uvicorn reflexio.server.api:app",
) -> tuple[dict[str, int], dict[str, str]]:
    """Build port_map and process_patterns for stopping services.

    Args:
        only: Set of service names to stop
        ports: Resolved port map
        backend_pattern: Process pattern for the backend service

    Returns:
        Tuple of (port_map, process_patterns)
    """
    port_map: dict[str, int] = {}
    process_patterns: dict[str, str] = {}

    if "backend" in only:
        port_map["backend"] = ports["backend"]
        process_patterns["backend"] = backend_pattern

    if "docs" in only:
        port_map["docs"] = ports["docs"]
        process_patterns["docs"] = f"next dev.*-p {ports['docs']}"

    return port_map, process_patterns


def execute(args: argparse.Namespace) -> None:
    """Execute the stop-services command."""
    load_dotenv()

    ports = resolve_ports(args, defaults={"backend": 8081, "docs": 8082})
    only = parse_only_flag(args.only, {"backend", "docs"})
    port_map, process_patterns = build_stop_targets(only, ports)

    if not port_map:
        print("No services selected. Available: backend, docs")
        return

    print("Stopping services...")
    stop_services(port_map, process_patterns, force=args.force)
    print("All services stopped.")
