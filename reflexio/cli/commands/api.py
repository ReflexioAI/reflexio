"""Raw API command for calling any Reflexio endpoint directly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import requests
import typer

from reflexio.cli.errors import EXIT_VALIDATION, CliError, handle_errors
from reflexio.cli.output import render
from reflexio.cli.state import resolve_api_key, resolve_url

app = typer.Typer(help="Call any Reflexio API endpoint directly.")

_VALID_METHODS = {"GET", "POST", "DELETE"}

_METHOD_DISPATCH = {
    "GET": requests.get,
    "POST": requests.post,
    "DELETE": requests.delete,
}


@app.callback(invoke_without_command=True)
@handle_errors
def call(
    ctx: typer.Context,
    method: Annotated[
        str,
        typer.Argument(help="HTTP method (GET, POST, DELETE)"),
    ],
    path: Annotated[
        str,
        typer.Argument(help="API path (e.g. /api/get_agent_playbooks)"),
    ],
    data: Annotated[
        str | None,
        typer.Option("--data", help="JSON body string or @filepath"),
    ] = None,
) -> None:
    """Make a raw HTTP request to a Reflexio API endpoint.

    Args:
        ctx: Typer context with CliState in ctx.obj
        method: HTTP method (GET, POST, DELETE)
        path: API endpoint path
        data: Optional JSON body as string or @filepath
    """
    method_upper = method.upper()
    if method_upper not in _VALID_METHODS:
        raise CliError(
            error_type="validation",
            message=f"Invalid method: {method}. Must be one of GET, POST, DELETE.",
            exit_code=EXIT_VALIDATION,
        )

    state = ctx.obj

    url = resolve_url(state.server_url)
    api_key = resolve_api_key(state.api_key)
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = _parse_data(data) if data else None

    request_fn = _METHOD_DISPATCH[method_upper]
    kwargs: dict[str, object] = {"headers": headers}
    if body is not None:
        kwargs["json"] = body

    resp = request_fn(f"{url}{path}", **kwargs)  # type: ignore[arg-type]
    resp.raise_for_status()

    json_mode: bool = state.json_mode
    render(resp.json(), json_mode=json_mode)


def _parse_data(data: str) -> object:
    """Parse a --data value as JSON string or @filepath.

    Args:
        data: JSON string or @filepath reference

    Returns:
        object: Parsed JSON data

    Raises:
        CliError: If the JSON is invalid or file cannot be read
    """
    if data.startswith("@"):
        file_path = Path(data[1:])
        try:
            raw = file_path.read_text()
        except OSError as exc:
            raise CliError(
                error_type="validation",
                message=f"Cannot read data file: {exc}",
                exit_code=EXIT_VALIDATION,
            ) from exc
    else:
        raw = data

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError(
            error_type="validation",
            message=f"Invalid JSON in --data: {exc}",
            exit_code=EXIT_VALIDATION,
        ) from exc
