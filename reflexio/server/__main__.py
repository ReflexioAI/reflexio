"""Backend server entrypoint.

Run the FastAPI backend with Reflexio's uvicorn log config applied
upfront::

    python -m reflexio.server --port 8081 --reload

Flags mirror the subset of ``uvicorn`` CLI options that
:func:`reflexio.cli.run_services.build_backend_service` uses.
"""

from __future__ import annotations

import argparse

import uvicorn

from reflexio.server.uvicorn_logging import UVICORN_LOG_CONFIG


def main(argv: list[str] | None = None) -> None:
    """Parse args and hand off to ``uvicorn.run``.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).
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
    args = parser.parse_args(argv)

    uvicorn.run(
        args.app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_includes=args.reload_include or None,
        log_config=UVICORN_LOG_CONFIG,
    )


if __name__ == "__main__":
    main()
