"""Configuration management commands (show, set, storage, pull)."""

from __future__ import annotations

import json
import re
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

# Mapping from StorageConfigSupabase field names to the env var names the
# rest of the codebase reads. Keep these in sync with .env.example and the
# setup wizard so pulled creds actually take effect without a rename pass.
_SUPABASE_FIELD_TO_ENV = {
    "url": "SUPABASE_URL",
    "key": "SUPABASE_KEY",
    "db_url": "SUPABASE_DB_URL",
}


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
    client = get_client(ctx)
    resp = client.get_config()

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(resp, json_mode=True, exclude_none=not show_all)
    else:
        config_data = (
            resp.model_dump(mode="json", exclude_none=not show_all)
            if hasattr(resp, "model_dump")
            else resp
        )
        print_info("Server configuration:")
        print(json.dumps(config_data, indent=2, default=str))


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


def _format_env_assignment(key: str, value: str) -> str:
    """Format a key=value pair for a .env file, quoting and escaping.

    Mirrors the quoting used by :func:`reflexio.cli.commands.setup_cmd._set_env_var`
    so lines written via ``config pull`` are indistinguishable from lines
    written by the setup wizard.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{key}="{escaped}"'


def _fetch_my_config(client: ReflexioClient) -> MyConfigResponse:
    """Call ``client.get_my_config()`` and wrap any failure in a CliError.

    Both ``reflexio config storage`` and ``reflexio config pull`` hit the
    same endpoint and want identical error framing on transport failures,
    so we centralise the try/except here instead of duplicating it.

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


def _upsert_env_line(lines: list[str], key: str, value: str) -> None:
    """Replace an existing ``KEY=`` line in-place or append a new one.

    Prefers updating an active (uncommented) line; falls back to a
    commented-out line if present; appends a new line otherwise.
    """
    pattern = re.compile(rf"^#?\s*{re.escape(key)}=")
    active_idx: int | None = None
    commented_idx: int | None = None
    for i, line in enumerate(lines):
        if not pattern.match(line):
            continue
        if line.lstrip().startswith("#"):
            if commented_idx is None:
                commented_idx = i
        else:
            active_idx = i
            break
    replacement = _format_env_assignment(key, value)
    target = active_idx if active_idx is not None else commented_idx
    if target is not None:
        lines[target] = replacement
    else:
        lines.append(replacement)


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


@app.command()
@handle_errors
def pull(
    ctx: typer.Context,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Overwrite existing credential lines in the target file"
        ),
    ] = False,
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            help="Target .env file (default: ~/.reflexio/.env)",
        ),
    ] = None,
) -> None:
    """Pull the server-side storage credentials down to your local .env.

    Calls ``GET /api/my_config`` and writes the returned Supabase (or
    other) storage fields into the target .env so a fresh machine can
    talk to your own Supabase directly via the local lib.

    Refuses to overwrite existing active credential lines unless
    ``--force`` is passed, so a stale pull can't silently clobber a
    working configuration.

    Args:
        ctx: Typer context with CliState in ctx.obj
        force: When True, overwrite existing credential lines in the target file.
        env_file: Explicit target path. Defaults to ``~/.reflexio/.env``.
    """
    client = get_client(ctx)
    resp = _fetch_my_config(client)

    if not resp.success or not resp.storage_config:
        raise CliError(
            error_type="validation",
            message=resp.message or "No storage configured for this org",
            hint="Configure storage at https://reflexio.ai/settings first.",
            exit_code=EXIT_VALIDATION,
        )

    if resp.storage_type != "supabase":
        raise CliError(
            error_type="validation",
            message=(
                f"Pulling {resp.storage_type} storage is not supported. "
                "Only Supabase creds can be pulled to a local .env."
            ),
            exit_code=EXIT_VALIDATION,
        )

    target = env_file or Path.home() / ".reflexio" / ".env"
    # ``mode`` on mkdir only applies when the directory is freshly
    # created, so this is a no-op for an existing ``~/.reflexio`` but
    # still tightens permissions for first-run callers.
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lines = target.read_text().splitlines() if target.exists() else []

    # Detect existing credential lines so we can refuse clobbers. We
    # include the REFLEXIO_* identity pair because ``pull`` overwrites
    # them unconditionally further down — without the guard a stale
    # pull could silently rewrite a user's API key. The leading ``\s*``
    # mirrors ``_upsert_env_line`` so commented/indented lines trip the
    # guard consistently.
    guarded_vars = [
        *_SUPABASE_FIELD_TO_ENV.values(),
        "REFLEXIO_URL",
        "REFLEXIO_API_KEY",
    ]
    existing_active = {
        env_name
        for env_name in guarded_vars
        if any(re.match(rf"^\s*{re.escape(env_name)}=", line) for line in lines)
    }
    if existing_active and not force:
        raise CliError(
            error_type="validation",
            message=(
                f"{target} already has {', '.join(sorted(existing_active))} set. "
                "Pass --force to overwrite."
            ),
            exit_code=EXIT_VALIDATION,
        )

    written: list[str] = []
    for field, env_name in _SUPABASE_FIELD_TO_ENV.items():
        value = resp.storage_config.get(field)
        if not value:
            continue
        _upsert_env_line(lines, env_name, str(value))
        written.append(env_name)

    # Always also record REFLEXIO_URL/API key so the local lib knows how
    # to identify itself next time.
    _upsert_env_line(lines, "REFLEXIO_URL", client.base_url)
    written.append("REFLEXIO_URL")
    if client.api_key:
        _upsert_env_line(lines, "REFLEXIO_API_KEY", client.api_key)
        written.append("REFLEXIO_API_KEY")

    # Create the file with restricted permissions *before* writing the
    # credentials so there's no brief window where the raw Supabase key
    # lands on disk as world-readable. The explicit ``chmod`` afterwards
    # stays as a belt-and-suspenders guard in case the file pre-existed
    # with looser permissions.
    target.touch(mode=0o600, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    target.chmod(0o600)

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(
            {
                "path": str(target),
                "written": written,
                "storage_type": resp.storage_type,
            },
            json_mode=True,
        )
    else:
        print_info(f"Wrote {len(written)} key(s) to {target}")
        for env_name in written:
            print_info(f"  {env_name}=<set>")
