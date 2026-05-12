"""Backend server entrypoint.

Run the FastAPI backend with Reflexio's uvicorn log config applied upfront::

    # Dev mode (autoreload, single process):
    python -m reflexio.server --port 8081 --reload

    # Daemon mode (multi-worker with recycling):
    python -m reflexio.server --port 8081 --workers 2 --max-requests 10000

Flags mirror the subset of ``uvicorn`` CLI options that
:func:`reflexio.cli.run_services.build_backend_service` uses.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from reflexio.server.uvicorn_logging import UVICORN_LOG_CONFIG


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser for ``reflexio.server``.
    """
    parser = argparse.ArgumentParser(prog="reflexio.server")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--reload-include",
        action="append",
        default=[],
        help="Glob pattern to watch for reload (repeatable).",
    )
    parser.add_argument(
        "--app",
        default="reflexio.server.api:app",
        help="ASGI app module path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of uvicorn worker processes. Default 1. "
            "Must be 1 when --reload is set (incompatible with autoreload)."
        ),
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=10000,
        help=(
            "Worker exits after this many requests (daemon mode only). "
            "Set to 0 to disable recycling. Default 10000."
        ),
    )
    parser.add_argument(
        "--max-requests-jitter",
        type=int,
        default=1000,
        help="Random 0..jitter added to --max-requests per worker. Default 1000.",
    )
    parser.add_argument(
        "--graceful-shutdown-sec",
        type=int,
        default=30,
        help="Seconds to drain in-flight requests on shutdown. Default 30.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse args and hand off to ``uvicorn.run``.

    Args:
        argv (list[str] | None): Optional argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: When ``--reload`` is combined with ``--workers > 1``, or when
            ``--workers < 1`` is specified.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    if args.reload and args.workers > 1:
        print(
            "error: --workers N (N>1) is incompatible with --reload; "
            "pass --no-reload or remove --workers",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.reload:
        uvicorn.run(
            args.app,
            host=args.host,
            port=args.port,
            reload=True,
            reload_includes=args.reload_include or None,
            log_config=UVICORN_LOG_CONFIG,
        )
        return

    # uvicorn treats limit_max_requests=0 as "recycle after 0 requests" (i.e. the
    # worker exits on its first served request). Translate the operator-facing
    # "0 disables recycling" semantics into uvicorn's None to actually disable.
    limit_max_requests = args.max_requests if args.max_requests > 0 else None
    uvicorn.run(
        args.app,
        host=args.host,
        port=args.port,
        reload=False,
        workers=args.workers,
        limit_max_requests=limit_max_requests,
        limit_max_requests_jitter=args.max_requests_jitter,
        timeout_graceful_shutdown=args.graceful_shutdown_sec,
        log_config=UVICORN_LOG_CONFIG,
    )


if __name__ == "__main__":
    main()
