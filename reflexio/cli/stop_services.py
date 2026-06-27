"""Stop reflexio services."""

from __future__ import annotations

import argparse

from reflexio.cli.env_loader import load_reflexio_env
from reflexio.cli.run_services import (
    DEFAULT_OSS_PORTS,
    parse_only_flag,
    resolve_ports,
)
from reflexio.cli.utils import stop_services


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add stop-services arguments to the parser."""
    parser.add_argument(
        "--backend-port",
        type=int,
        default=None,
        help="Backend port (default: 8061, env: BACKEND_PORT)",
    )
    parser.add_argument(
        "--docs-port",
        type=int,
        default=None,
        help="Docs port (default: 8062, env: DOCS_PORT)",
    )
    parser.add_argument(
        "--embedding-port",
        type=int,
        default=None,
        help="Embedding service port (default: 8069, env: EMBEDDING_PORT)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of services to stop: backend,docs,embedding",
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

    if "embedding" in only:
        port_map["embedding"] = ports["embedding"]
        process_patterns["embedding"] = "reflexio.server.llm.embedding_service:app"

    return port_map, process_patterns


def execute(args: argparse.Namespace) -> None:
    """Execute the stop-services command."""
    # Use reflexio's scoped loader (./.env + ~/.reflexio/.env), not a bare
    # ``dotenv.load_dotenv()`` which walks up to a parent ``.env``.
    load_reflexio_env()

    ports = resolve_ports(args, defaults=DEFAULT_OSS_PORTS)
    only = parse_only_flag(args.only, {"backend", "docs", "embedding"})
    port_map, process_patterns = build_stop_targets(only, ports)

    if not port_map:
        print("No services selected. Available: backend, docs, embedding")
        return

    print("Stopping services...")
    stop_services(port_map, process_patterns, force=args.force)
    print("All services stopped.")
