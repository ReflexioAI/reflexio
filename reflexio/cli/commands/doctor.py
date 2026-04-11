"""Diagnostic health check command."""

from __future__ import annotations

import sys
from typing import Any

import requests
import typer

from reflexio.cli.env_loader import get_env_path
from reflexio.cli.errors import handle_errors
from reflexio.cli.output import print_doctor_checks, render
from reflexio.cli.state import resolve_api_key, resolve_url

app = typer.Typer(help="Diagnose Reflexio setup.")


@app.command("check")
@handle_errors
def doctor(ctx: typer.Context) -> None:
    """Run diagnostic checks on Reflexio configuration and connectivity."""
    json_mode = ctx.obj.json_mode
    state = ctx.obj
    checks: list[dict[str, Any]] = []

    # 1. Env file
    env_path = get_env_path()
    env_exists = env_path.exists()
    checks.append(
        {
            "name": "env_file",
            "status": "pass" if env_exists else "warn",
            "message": f"Env file at {env_path}"
            if env_exists
            else f"No env file at {env_path}",
            "hint": None
            if env_exists
            else "Run: reflexio auth login --api-key KEY --server-url URL",
        }
    )

    # 2. Resolve connection info
    url = resolve_url(state.server_url)
    api_key = resolve_api_key(state.api_key)

    has_key = bool(api_key)
    checks.append(
        {
            "name": "api_key",
            "status": "pass" if has_key else "warn",
            "message": f"API key configured (****{api_key[-4:]})"
            if has_key
            else "No API key configured",
            "hint": None
            if has_key
            else "Set REFLEXIO_API_KEY env var or run: reflexio auth login --api-key KEY",
        }
    )

    # 3. Server connectivity
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(f"{url.rstrip('/')}/health", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")
        checks.append(
            {
                "name": "server_health",
                "status": "pass" if status == "healthy" else "warn",
                "message": f"Server at {url} is {status}",
                "hint": None,
            }
        )
    except requests.ConnectionError:
        is_remote = url and "localhost" not in url and "127.0.0.1" not in url
        hint = (
            "Check your API key and network connection, or verify the server URL"
            if is_remote
            else "Start the server: reflexio services start"
        )
        checks.append(
            {
                "name": "server_health",
                "status": "fail",
                "message": f"Cannot connect to {url}",
                "hint": hint,
            }
        )
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else 0
        checks.append(
            {
                "name": "server_health",
                "status": "fail",
                "message": f"Server returned HTTP {code}",
                "hint": "Check server logs for details",
            }
        )

    # 4. Python version
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    checks.append(
        {
            "name": "python_version",
            "status": "pass" if sys.version_info >= (3, 11) else "warn",
            "message": f"Python {py_version}",
            "hint": None if sys.version_info >= (3, 11) else "Python 3.11+ recommended",
        }
    )

    # Render output
    if json_mode:
        all_pass = all(c["status"] == "pass" for c in checks)
        render(checks, json_mode=True, meta={"all_pass": all_pass})
    else:
        print_doctor_checks(checks)

        fails = sum(1 for c in checks if c["status"] == "fail")
        if fails:
            raise SystemExit(1)
