"""Configuration management commands (show, set, storage, pull)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from reflexio.cli.errors import EXIT_NETWORK, EXIT_VALIDATION, CliError, handle_errors
from reflexio.cli.output import (
    print_error,
    print_info,
    print_storage_credentials,
    render,
)
from reflexio.cli.state import get_client
from reflexio.lib._storage_labels import mask_secret, mask_url
from reflexio.models.api_schema.service_schemas import MyConfigResponse

if TYPE_CHECKING:
    from reflexio.client.client import ReflexioClient

app = typer.Typer(help="View and update server configuration.")


def _resolve_data(data: str) -> dict:
    """Resolve a JSON data string, supporting @filepath syntax.

    If the string starts with '@', reads the file at the given path
    and parses it as JSON. Otherwise, parses the string directly.

    Args:
        data: JSON string or @filepath reference

    Returns:
        dict: Parsed configuration data
    """
    if data.startswith("@"):
        return json.loads(Path(data[1:]).read_text())
    return json.loads(data)


@app.command()
@handle_errors
def show(
    ctx: typer.Context,
    show_all: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Show all fields including unset optional settings with defaults",
        ),
    ] = False,
) -> None:
    """Show current server configuration.

    Args:
        ctx: Typer context with CliState in ctx.obj
        show_all: If True, include all fields (even None/default) in output
    """
    from reflexio.cli.bootstrap_config import default_config_path

    client = get_client(ctx)
    resp = client.get_config()

    config_path = default_config_path()
    config_exists = config_path.exists()
    # `MyConfigResponse` owns the `data` envelope field, so local-file
    # state ships under `meta` for this command — unlike `config local`
    # and `auth status`, where the file path IS the primary data.
    local_config_meta = {
        "path": str(config_path),
        "exists": config_exists,
        "using_defaults": not config_exists,
    }

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(
            resp,
            json_mode=True,
            exclude_none=not show_all,
            meta={"local_config": local_config_meta},
        )
    else:
        config_data = (
            resp.model_dump(mode="json", exclude_none=not show_all)
            if hasattr(resp, "model_dump")
            else resp
        )
        suffix = "" if config_exists else " (not found — showing defaults)"
        print_info(f"Local config file: {config_path}{suffix}")
        print_info("Server configuration:")
        print(json.dumps(config_data, indent=2, default=str))


@app.command(name="local")
@handle_errors
def show_local(ctx: typer.Context) -> None:
    """Show locally persisted settings (no server required).

    Reads the local config file and resolves the effective storage backend
    using the priority chain: CLI flag > env var > config file > default.

    Args:
        ctx: Typer context with CliState in ctx.obj
    """
    from reflexio.cli.bootstrap_config import (
        default_config_path,
        load_storage_from_config,
        resolve_storage,
    )

    persisted = load_storage_from_config()
    resolved = resolve_storage(None)  # full resolution without CLI flag
    config_path = default_config_path()
    config_exists = config_path.exists()
    resolved_mode = "local" if resolved in ("sqlite", "disk") else "cloud"

    json_mode: bool = ctx.obj.json_mode

    data = {
        "config_file": str(config_path),
        "config_file_exists": config_exists,
        "persisted_storage": persisted,
        "resolved_storage": resolved,
        "resolved_mode": resolved_mode,
    }

    if json_mode:
        render(data, json_mode=True)
    else:
        path_suffix = "" if config_exists else " (not found — showing defaults)"
        persisted_label = persisted or "(not set)"
        resolved_suffix = " (default)" if persisted is None else ""
        print_info(f"Config file: {config_path}{path_suffix}")
        print_info(f"Persisted storage: {persisted_label}")
        print_info(
            f"Resolved storage:  {resolved} (mode: {resolved_mode}){resolved_suffix}"
        )


@app.command(name="set")
@handle_errors
def set_config(
    ctx: typer.Context,
    data: Annotated[
        str | None,
        typer.Option("--data", help="JSON string or @filepath with config data"),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option("--file", help="Path to JSON config file"),
    ] = None,
) -> None:
    """Update server configuration.

    Provide configuration data via --data (inline JSON or @filepath) or --file.

    Args:
        ctx: Typer context with CliState in ctx.obj
        data: JSON string or @filepath with configuration data
        file: Path to a JSON configuration file
    """
    if not data and not file:
        raise CliError(
            error_type="validation",
            message="Must provide either --data or --file",
            hint="Use --data '{...}' or --data @path/to/config.json or --file path/to/config.json",
            exit_code=EXIT_VALIDATION,
        )

    if data and file:
        raise CliError(
            error_type="validation",
            message="Cannot provide both --data and --file",
            exit_code=EXIT_VALIDATION,
        )

    try:
        if data:
            config_data = _resolve_data(data)
        else:
            assert file is not None  # guaranteed by guard above  # noqa: S101
            config_data = json.loads(file.read_text())
    except (json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        raise CliError(
            error_type="validation",
            message=f"Failed to parse config data: {exc}",
            exit_code=EXIT_VALIDATION,
        ) from exc

    client = get_client(ctx)
    resp = client.set_config(config_data)

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True)
    else:
        print_info("Configuration updated")


# ---------------------------------------------------------------------------
# Storage credential inspection / pull (backed by GET /api/my_config)
# ---------------------------------------------------------------------------


def _mask_storage_config(storage_config: dict) -> dict:
    """Return a masked copy of a serialized StorageConfig.

    Keeps field names + structure intact so users can see *which* fields
    are set without exposing the secret material. URL-like values go
    through :func:`mask_url`, everything else through :func:`mask_secret`.
    """
    masked: dict = {}
    for key, value in storage_config.items():
        if value is None:
            masked[key] = None
        elif not isinstance(value, str):
            masked[key] = value
        elif "url" in key.lower() or "://" in value:
            masked[key] = mask_url(value)
        else:
            masked[key] = mask_secret(value)
    return masked


def _fetch_my_config(client: ReflexioClient) -> MyConfigResponse:
    """Call ``client.get_my_config()`` and wrap any failure in a CliError.

    ``reflexio config storage`` hits the ``GET /api/my_config`` endpoint and
    needs uniform error framing on transport failures, so we centralise the
    try/except here.

    Args:
        client: A configured ``ReflexioClient`` instance.

    Returns:
        MyConfigResponse: The server's response on success.

    Raises:
        CliError: When the underlying HTTP call raises. 404 is framed
            as "the server doesn't expose this endpoint yet";
            everything else is a generic network error.
    """
    import requests

    try:
        return client.get_my_config()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise CliError(
                error_type="api",
                message=(
                    f"{client.base_url}/api/my_config returned 404 — the "
                    "server is reachable but doesn't expose this endpoint."
                ),
                hint=(
                    "The backend may be running a version that "
                    "predates '/api/my_config'. Ask the server "
                    "operator to upgrade, or point REFLEXIO_URL at a "
                    "deployment that exposes this endpoint."
                ),
                exit_code=EXIT_NETWORK,
            ) from exc
        raise CliError(
            error_type="network",
            message=f"Failed to reach {client.base_url}/api/my_config: {exc}",
            hint="Confirm REFLEXIO_URL + REFLEXIO_API_KEY, then try again.",
            exit_code=EXIT_NETWORK,
        ) from exc
    except requests.ConnectionError as exc:
        raise CliError(
            error_type="network",
            message=f"Failed to reach {client.base_url}/api/my_config: {exc}",
            hint="Confirm REFLEXIO_URL + REFLEXIO_API_KEY, then try again.",
            exit_code=EXIT_NETWORK,
        ) from exc


@app.command()
@handle_errors
def storage(
    ctx: typer.Context,
    reveal: Annotated[
        bool,
        typer.Option(
            "--reveal",
            help="Print the raw credentials instead of a masked summary",
        ),
    ] = False,
) -> None:
    """Show the storage credentials the server has on file for your org.

    Calls ``GET /api/my_config``. The default output masks credentials
    so it's safe to paste into bug reports; pass ``--reveal`` to print
    the unmasked values when copying to a new machine.

    Args:
        ctx: Typer context with CliState in ctx.obj
        reveal: When True, print unmasked credentials after confirmation.
    """
    client = get_client(ctx)
    resp = _fetch_my_config(client)

    json_mode: bool = ctx.obj.json_mode

    if not resp.success or not resp.storage_config:
        if json_mode:
            render(resp, json_mode=True)
        else:
            print_error(resp.message or "No storage configured for this org")
        return

    payload: dict = dict(resp.storage_config)

    if reveal and not json_mode:
        if not typer.confirm(
            "This will print your raw storage credentials. Continue?",
            default=False,
        ):
            raise typer.Abort()
        display = payload
    else:
        display = _mask_storage_config(payload)

    if json_mode:
        render(
            {"storage_type": resp.storage_type, "storage_config": display},
            json_mode=True,
        )
        return

    print_storage_credentials(
        resp.storage_type,
        display,
        revealed=reveal,
    )


