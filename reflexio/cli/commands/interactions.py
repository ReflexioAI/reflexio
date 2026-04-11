"""Manage interactions (conversations) via the Reflexio CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from reflexio.cli.errors import EXIT_VALIDATION, CliError, handle_errors
from reflexio.cli.output import (
    pagination_meta,
    print_error,
    print_info,
    print_interactions,
    render,
)
from reflexio.cli.state import get_client, resolve_agent_version
from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionResponse,
)

app = typer.Typer(help="Manage interactions (conversations).")


# ---------------------------------------------------------------------------
# Helpers (ported from capture.py)
# ---------------------------------------------------------------------------


def _parse_json_payload(raw: str) -> list[dict]:
    """Parse JSON or JSONL text into a list of conversation payloads.

    Args:
        raw (str): Raw JSON or JSONL string.

    Returns:
        list[dict]: List of parsed payload dicts, each containing an
            ``interactions`` key at minimum.

    Raises:
        CliError: If the input is empty or contains invalid JSON.
    """
    stripped = raw.strip()
    if not stripped:
        raise CliError(
            error_type="validation",
            message="Empty input",
            exit_code=EXIT_VALIDATION,
        )

    # Try single JSON object/array first
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            if not all(isinstance(item, dict) for item in data):
                raise CliError(
                    error_type="validation",
                    message="JSON array elements must be objects, not primitives",
                    exit_code=EXIT_VALIDATION,
                )
            return data
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL (one JSON object per line)
    payloads: list[dict] = []
    for lineno, line in enumerate(stripped.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CliError(
                error_type="validation",
                message=f"Invalid JSON on line {lineno}: {exc}",
                exit_code=EXIT_VALIDATION,
            ) from exc
    return payloads


def _interactions_from_payload(payload: dict) -> list[InteractionData]:
    """Convert a payload dict into a list of InteractionData objects.

    Forwards ``tools_used`` (and any other InteractionData field) from each
    item so structured tool-call metadata reaches the server renderer that
    emits ``[used tool: ...]`` markers for playbook extraction.

    Args:
        payload (dict): Payload with an ``interactions`` list of dicts.
            Each item may include ``role``, ``content``, ``tools_used``,
            and other ``InteractionData`` fields.

    Returns:
        list[InteractionData]: Parsed interaction data objects.

    Raises:
        CliError: If the payload is missing the ``interactions`` list or
            any item fails ``InteractionData`` validation.
    """
    raw_interactions = payload.get("interactions", [])
    if not raw_interactions:
        raise CliError(
            error_type="validation",
            message="Payload missing 'interactions' list",
            exit_code=EXIT_VALIDATION,
        )
    try:
        return [InteractionData(**item) for item in raw_interactions]
    except (TypeError, ValueError) as exc:
        raise CliError(
            error_type="validation",
            message=f"Invalid interaction data: {exc}",
            exit_code=EXIT_VALIDATION,
        ) from exc


def _read_data_arg(data: str) -> str:
    """Read inline JSON or from a file path prefixed with ``@``.

    Args:
        data (str): JSON string, or ``@filepath`` to read from disk.

    Returns:
        str: Raw JSON content.

    Raises:
        CliError: If the referenced file does not exist.
    """
    if data.startswith("@"):
        path = Path(data[1:])
        if not path.exists():
            raise CliError(
                error_type="validation",
                message=f"File not found: {path}",
                exit_code=EXIT_VALIDATION,
            )
        return path.read_text()
    return data


_MAX_WARNING_LEN = 240


def _oneline(text: str, max_len: int = _MAX_WARNING_LEN) -> str:
    """Collapse newlines and truncate so a pathological server message
    can't dump a multi-line stack trace into the CLI output.

    The server's generation service appends free-form exception
    strings into ``warnings``, and historically some backends
    embedded tracebacks in ``StorageError.message``. Even after we
    fix those on the server side, this helper is a belt-and-
    suspenders guard so a future regression can't turn ``reflexio
    publish`` into a terminal dump.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 1] + "…"


def _print_publish_result(
    result: PublishUserInteractionResponse, user_id: str
) -> None:
    """Print human-readable publish result with diagnostics.

    Surfaces storage routing + extraction counts so users can tell *where*
    their publish landed and *what* the extraction produced, not just that
    the call returned 200. Degrades gracefully when the server hasn't
    populated the diagnostic fields (server-async mode, which only returns a
    bare acknowledgement instead of counts + request_id + storage routing).
    """
    message = _oneline(result.message or "")
    warnings = result.warnings or []

    if not result.success:
        print_error(f"Publish failed for {user_id}: {message}")
        for warning in warnings:
            print_error(f"  Warning: {_oneline(warning)}")
        return

    header = f"Published ({user_id})"
    if message and message != "Interaction published successfully":
        header = f"{header}: {message}"
    print_info(header)

    storage_type = getattr(result, "storage_type", None)
    storage_label = getattr(result, "storage_label", None)
    if storage_type or storage_label:
        label = storage_label or "<unknown>"
        kind = storage_type or "?"
        print_info(f"  Storage: {kind} → {label}")

    profiles_added = getattr(result, "profiles_added", None)
    playbooks_added = getattr(result, "playbooks_added", None)
    if profiles_added is not None or playbooks_added is not None:
        parts: list[str] = []
        if profiles_added is not None:
            parts.append(f"{profiles_added} profile(s) added")
        if playbooks_added is not None:
            parts.append(f"{playbooks_added} playbook(s) added")
        print_info(f"  Extraction: {', '.join(parts)}")

    request_id = getattr(result, "request_id", None)
    if request_id:
        print_info(f"  Request ID: {request_id}")

    for warning in warnings:
        print_error(f"  Warning: {_oneline(warning)}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
@handle_errors
def publish(
    ctx: typer.Context,
    user_id: Annotated[
        str | None,
        typer.Option(
            help="User identifier (optional if provided in --file/--data payload)"
        ),
    ] = None,
    session_id: Annotated[
        str | None, typer.Option(help="Session ID for grouping")
    ] = None,
    source: Annotated[str, typer.Option(help="Source tag")] = "cli",
    agent_version: Annotated[
        str | None, typer.Option(help="Agent version string")
    ] = None,
    wait: Annotated[
        bool, typer.Option(help="Block until processing completes")
    ] = False,
    data: Annotated[str | None, typer.Option(help="JSON string or @filepath")] = None,
    file: Annotated[Path | None, typer.Option(help="Path to JSON/JSONL file")] = None,
    stdin: Annotated[bool, typer.Option(help="Read JSON data from stdin")] = False,
    user_message: Annotated[
        str | None, typer.Option(help="Single-turn user message")
    ] = None,
    agent_response: Annotated[
        str | None, typer.Option(help="Single-turn agent response")
    ] = None,
    skip_aggregation: Annotated[
        bool,
        typer.Option(
            "--skip-aggregation",
            help="Extract profiles/playbooks but skip aggregation to agent playbooks",
        ),
    ] = False,
    force_extraction: Annotated[
        bool,
        typer.Option(
            "--force-extraction",
            help="Bypass batch_interval checks and always run extractors",
        ),
    ] = False,
) -> None:
    """Publish interaction data for a user.

    Supports three input modes:
    1. ``--user-message`` + ``--agent-response`` for single-turn conversations.
    2. ``--file`` or ``--stdin`` for JSON/JSONL payloads.
    3. ``--data`` for inline JSON (prefix with ``@`` for a file path).
    """
    client = get_client(ctx)
    json_mode: bool = ctx.obj.json_mode

    # Announce the endpoint + wait mode up-front (stderr, non-JSON only) so
    # users can tell which server they're hitting before anything happens.
    if not json_mode:
        # The label refers to the *server* extraction mode: "sync" waits
        # for extraction to finish before returning real counts; "async"
        # queues extraction as a server BackgroundTask and returns
        # immediately. The client always blocks on the HTTP round-trip
        # in both cases.
        mode_label = "server-sync" if wait else "server-async"
        print_info(f"Publishing to {client.base_url} ({mode_label})")

    # Mode 1: single-turn
    if user_message is not None:
        if agent_response is None:
            raise CliError(
                error_type="validation",
                message="--agent-response is required with --user-message",
                exit_code=EXIT_VALIDATION,
            )
        if user_id is None:
            raise CliError(
                error_type="validation",
                message="--user-id is required with --user-message",
                exit_code=EXIT_VALIDATION,
            )
        interactions: list[InteractionData | dict] = [
            InteractionData(role="user", content=user_message),
            InteractionData(role="assistant", content=agent_response),
        ]
        result = client.publish_interaction(
            user_id=user_id,
            interactions=interactions,
            source=source,
            agent_version=resolve_agent_version(agent_version),
            session_id=session_id,
            wait_for_response=wait,
            skip_aggregation=skip_aggregation,
            force_extraction=force_extraction,
        )
        if json_mode:
            render(result, json_mode=True)
        else:
            _print_publish_result(result, user_id)
        return

    # Determine raw JSON from --data, --file, or --stdin
    raw: str | None = None
    if data is not None:
        raw = _read_data_arg(data)
    elif file is not None:
        if not file.exists():
            raise CliError(
                error_type="validation",
                message=f"File not found: {file}",
                exit_code=EXIT_VALIDATION,
            )
        raw = file.read_text()
    elif stdin:
        raw = sys.stdin.read()

    if raw is None:
        raise CliError(
            error_type="validation",
            message="Specify --user-message/--agent-response, --file, --stdin, or --data",
            exit_code=EXIT_VALIDATION,
        )

    payloads = _parse_json_payload(raw)
    print_info(f"Processing {len(payloads)} payload(s)...")

    resolved_version = resolve_agent_version(agent_version)
    for payload in payloads:
        interaction_items = _interactions_from_payload(payload)
        resolved_user_id = payload.get("user_id") or user_id
        if not resolved_user_id:
            raise CliError(
                error_type="validation",
                message="--user-id is required (or include 'user_id' in the JSON payload)",
                exit_code=EXIT_VALIDATION,
            )
        result = client.publish_interaction(
            user_id=resolved_user_id,
            interactions=interaction_items,  # type: ignore[arg-type]
            source=payload.get("source", source),
            agent_version=payload.get("agent_version", resolved_version),
            session_id=payload.get("session_id", session_id),
            wait_for_response=wait,
            skip_aggregation=payload.get("skip_aggregation", skip_aggregation),
            force_extraction=payload.get("force_extraction", force_extraction),
        )
        if json_mode:
            render(result, json_mode=True)
        else:
            uid = payload.get("user_id", user_id)
            _print_publish_result(result, uid)


@app.command(name="list")
@handle_errors
def list_interactions(
    ctx: typer.Context,
    user_id: Annotated[str | None, typer.Option(help="Filter by user ID")] = None,
    limit: Annotated[
        int, typer.Option(help="Maximum number of interactions to return")
    ] = 20,
) -> None:
    """List interactions, optionally filtered by user."""
    client = get_client(ctx)
    result = client.get_interactions(user_id=user_id, top_k=limit)
    interactions = result.interactions or []

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(interactions, json_mode=True, meta=pagination_meta(interactions, limit))
    else:
        print_info(f"Found {len(interactions)} interaction(s)")
        if interactions:
            print_interactions(interactions)


@app.command()
@handle_errors
def search(
    ctx: typer.Context,
    query: Annotated[str, typer.Option(help="Search query string")],
    user_id: Annotated[str | None, typer.Option(help="Filter by user ID")] = None,
    limit: Annotated[int, typer.Option(help="Maximum number of results")] = 10,
) -> None:
    """Search interactions by semantic query."""
    client = get_client(ctx)
    result = client.search_interactions(query=query, user_id=user_id, top_k=limit)
    interactions = result.interactions or []

    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(interactions, json_mode=True, meta=pagination_meta(interactions, limit))
    else:
        print_info(f"Found {len(interactions)} interaction(s)")
        if interactions:
            print_interactions(interactions)


@app.command()
@handle_errors
def delete(
    ctx: typer.Context,
    interaction_id: Annotated[str, typer.Option(help="Interaction ID to delete")],
    user_id: Annotated[str, typer.Option(help="User ID that owns the interaction")],
) -> None:
    """Delete a single interaction by ID."""
    client = get_client(ctx)
    result = client.delete_interaction(
        user_id=user_id,
        interaction_id=interaction_id,
        wait_for_response=True,
    )
    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(result, json_mode=True)
    else:
        print_info(f"Deleted interaction {interaction_id}")


@app.command(name="delete-all")
@handle_errors
def delete_all(
    ctx: typer.Context,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")
    ] = False,
) -> None:
    """Delete ALL interactions. This cannot be undone."""
    if not yes:
        typer.confirm("Delete all interactions? This cannot be undone", abort=True)
    client = get_client(ctx)
    result = client.delete_all_interactions()
    json_mode: bool = ctx.obj.json_mode
    if json_mode:
        render(result, json_mode=True)
    else:
        print_info("All interactions deleted")
