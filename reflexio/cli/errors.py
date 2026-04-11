"""Structured error handling for Reflexio CLI."""

from __future__ import annotations

import contextlib
import functools
import json
import sys
from typing import Any

import requests
from pydantic import ValidationError

# Semantic exit codes (modeled on Lark CLI)
EXIT_SUCCESS = 0
EXIT_GENERAL = 1
EXIT_VALIDATION = 2
EXIT_AUTH = 3
EXIT_NETWORK = 4


class CliError(Exception):
    """Structured CLI error with type, message, hint, and exit code.

    Args:
        error_type: Category of error (general, validation, auth, network, rate_limit)
        message: Human-readable error description
        hint: Actionable suggestion for fixing the error
        exit_code: Process exit code (0-4)
    """

    def __init__(
        self,
        error_type: str = "general",
        message: str = "An error occurred",
        hint: str | None = None,
        exit_code: int = EXIT_GENERAL,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.hint = hint
        self.exit_code = exit_code


def _classify_http_error(exc: requests.HTTPError) -> CliError:
    """Map HTTP status codes to structured CliError instances.

    Args:
        exc: The HTTP error to classify

    Returns:
        CliError: Classified error with appropriate type, hint, and exit code
    """
    status = exc.response.status_code if exc.response is not None else 0

    if status in (401, 403):
        return CliError(
            error_type="auth",
            message=f"Authentication failed (HTTP {status})",
            hint="Run: reflexio auth login --api-key <key> --url <url>",
            exit_code=EXIT_AUTH,
        )
    if status == 422:
        detail = ""
        with contextlib.suppress(ValueError, AttributeError):
            detail = exc.response.json().get("detail", "")
        return CliError(
            error_type="validation",
            message=f"Invalid request: {detail or exc}",
            exit_code=EXIT_VALIDATION,
        )
    if status == 429:
        return CliError(
            error_type="rate_limit",
            message="Rate limit exceeded",
            hint="Wait a moment and retry",
            exit_code=EXIT_GENERAL,
        )
    return CliError(
        error_type="general",
        message=f"HTTP error: {exc}",
        exit_code=EXIT_GENERAL,
    )


def raise_if_failed(resp: object, default: str = "operation failed") -> None:
    """Raise a ``CliError`` if the API response envelope reports failure.

    Intended for CLI mutation commands (``add`` / ``update`` /
    ``update-status``) so that ``success=False`` envelopes do not get
    reported as a success. The server's response schemas use either
    ``message`` or ``msg`` for the human-readable error string; both
    are checked. Call only in non-JSON mode — JSON-mode callers
    introspect the response envelope themselves.

    Args:
        resp: Response object returned by the client.
        default: Fallback message when neither ``message`` nor ``msg`` is set.

    Raises:
        CliError: If ``resp.success`` is falsy. The wrapped
            ``handle_errors`` decorator will render the error and
            exit with ``EXIT_GENERAL``.
    """
    if getattr(resp, "success", True):
        return
    message = getattr(resp, "message", None) or getattr(resp, "msg", None) or default
    raise CliError(
        error_type="api",
        message=message,
        exit_code=EXIT_GENERAL,
    )


def render_error(error: CliError, json_mode: bool = False) -> None:
    """Render an error to stderr in structured or human-readable format.

    Args:
        error: The CLI error to render
        json_mode: If True, output JSON envelope; otherwise plain text
    """
    if json_mode:
        envelope: dict[str, Any] = {
            "ok": False,
            "error": {
                "type": error.error_type,
                "message": error.message,
            },
        }
        if error.hint:
            envelope["error"]["hint"] = error.hint
        print(json.dumps(envelope, indent=2), file=sys.stderr)
    else:
        print(f"Error: {error.message}", file=sys.stderr)
        if error.hint:
            print(f"Hint: {error.hint}", file=sys.stderr)


def handle_errors(fn):  # noqa: ANN001, ANN201
    """Decorator that catches common exceptions and renders structured errors.

    Catches ``CliError``, ``ConnectionError``, ``HTTPError``, and the
    client's ``ReflexioAPIError`` (for "2xx with non-JSON body"
    failures, typically a misconfigured ``REFLEXIO_URL`` pointing at a
    marketing site or misrouting proxy), rendering them through the
    structured error system before exiting with the appropriate code.
    """
    # Imported lazily to avoid a circular import: client.py → cli.output
    # → cli.errors. The import only fires when the decorator is built.
    from reflexio.client.client import ReflexioAPIError

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Determine json_mode from context if available
        import typer

        ctx = None
        for arg in args:
            if isinstance(arg, typer.Context):
                ctx = arg
                break

        json_mode = False
        if ctx and ctx.obj:
            json_mode = getattr(ctx.obj, "json_mode", False)

        try:
            return fn(*args, **kwargs)
        except CliError as exc:
            # CliError must be caught first so its subclasses don't
            # get swallowed by the generic ValueError branch below.
            render_error(exc, json_mode=json_mode)
            raise SystemExit(exc.exit_code) from None
        except ValidationError as exc:
            err = CliError(
                error_type="validation",
                message=str(exc),
                exit_code=EXIT_VALIDATION,
            )
            render_error(err, json_mode=json_mode)
            raise SystemExit(err.exit_code) from None
        except ValueError as exc:
            err = CliError(
                error_type="validation",
                message=str(exc),
                exit_code=EXIT_VALIDATION,
            )
            render_error(err, json_mode=json_mode)
            raise SystemExit(err.exit_code) from None
        except requests.ConnectionError as exc:
            err = CliError(
                error_type="network",
                message=f"Connection failed: {exc}",
                hint="Is the Reflexio server running? Try: reflexio services start",
                exit_code=EXIT_NETWORK,
            )
            render_error(err, json_mode=json_mode)
            raise SystemExit(err.exit_code) from None
        except requests.HTTPError as exc:
            err = _classify_http_error(exc)
            render_error(err, json_mode=json_mode)
            raise SystemExit(err.exit_code) from None
        except ReflexioAPIError as exc:
            err = CliError(
                error_type="api",
                message=str(exc),
                hint=(
                    "Double-check REFLEXIO_URL — it should point at your "
                    "Reflexio API host, not a marketing site or proxy."
                ),
                exit_code=EXIT_GENERAL,
            )
            render_error(err, json_mode=json_mode)
            raise SystemExit(err.exit_code) from None

    return wrapper
