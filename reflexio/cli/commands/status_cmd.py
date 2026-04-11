"""Server health and status check."""

from __future__ import annotations

import json
import sys

import requests
import typer

from reflexio.cli.errors import EXIT_NETWORK, CliError, handle_errors, render_error
from reflexio.cli.output import print_whoami_summary, render
from reflexio.cli.state import CliState, get_client, resolve_api_key, resolve_url

app = typer.Typer(help="Server health and status.")


@app.command()
@handle_errors
def check(
    ctx: typer.Context,
) -> None:
    """Check server health status.

    Makes a GET request to the /health endpoint and reports whether
    the server is reachable and healthy.

    Args:
        ctx: Typer context with CliState in ctx.obj
    """
    state: CliState = ctx.obj
    json_mode: bool = state.json_mode
    url = resolve_url(state.server_url)
    api_key = resolve_api_key(state.api_key)

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(f"{url}/health", headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        err = CliError(
            error_type="network",
            message=f"Cannot reach server at {url}",
            hint="Is the Reflexio server running? Try: reflexio services start",
            exit_code=EXIT_NETWORK,
        )
        render_error(err, json_mode=json_mode)
        raise SystemExit(1) from exc
    except requests.HTTPError as exc:
        if json_mode:
            envelope = {
                "ok": False,
                "error": {"type": "unhealthy", "message": str(exc), "url": url},
            }
            print(json.dumps(envelope, indent=2), file=sys.stderr)
        else:
            print(
                f"Error: Server at {url} returned {exc.response.status_code}",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc

    if json_mode:
        envelope = {"ok": True, "data": {"status": "healthy", "url": url}}
        print(json.dumps(envelope, indent=2))
    else:
        print(f"Connected to {url} (healthy)")


@app.command()
@handle_errors
def whoami(
    ctx: typer.Context,
) -> None:
    """Show who you are on the server + where your data lands.

    Calls ``GET /api/whoami`` to report the org ID, resolved storage
    type, and masked storage label for the current API key. Useful
    for sanity-checking whether you're pointed at the right backend
    and whether your org has storage configured.

    Args:
        ctx: Typer context with CliState in ctx.obj
    """
    state: CliState = ctx.obj
    json_mode: bool = state.json_mode
    client = get_client(ctx)

    try:
        resp = client.whoami()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            # The server is up and reachable — it just doesn't ship the
            # /api/whoami endpoint yet. Don't tell the user to check
            # that their server is running; tell them the server may
            # be on a version that predates this endpoint.
            err = CliError(
                error_type="api",
                message=(
                    f"{client.base_url}/api/whoami returned 404 — the "
                    "server is reachable but doesn't expose this endpoint."
                ),
                hint=(
                    "The backend may be running a version that "
                    "predates '/api/whoami'. Ask the server operator to "
                    "upgrade, or point REFLEXIO_URL at a deployment "
                    "that exposes this endpoint."
                ),
                exit_code=EXIT_NETWORK,
            )
            render_error(err, json_mode=json_mode)
            raise SystemExit(1) from exc
        raise  # let handle_errors classify other HTTP errors (401/403/etc.)
    except requests.ConnectionError as exc:
        err = CliError(
            error_type="network",
            message=f"Failed to reach {client.base_url}/api/whoami: {exc}",
            hint=(
                "Check that REFLEXIO_URL points at a running server and "
                "that REFLEXIO_API_KEY is valid for that backend."
            ),
            exit_code=EXIT_NETWORK,
        )
        render_error(err, json_mode=json_mode)
        raise SystemExit(1) from exc

    if json_mode:
        render(resp, json_mode=True)
        return

    api_key = resolve_api_key(state.api_key)
    print_whoami_summary(
        endpoint=client.base_url,
        api_key=api_key,
        org_id=resp.org_id,
        storage_type=resp.storage_type,
        storage_label=resp.storage_label,
        storage_configured=bool(resp.storage_configured),
        message=resp.message,
    )
