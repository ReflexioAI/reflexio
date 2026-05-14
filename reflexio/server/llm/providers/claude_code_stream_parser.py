"""Parse the NDJSON stream from ``claude -p --output-format stream-json``
and classify whether the call ended in a credit/auth stall.

Public surface:
    - parse_stream_json(stdout, exit_code, stderr_text) -> ParseResult
    - classify_stall(result) -> "billing_error" | "auth_error" | None
    - parse_reset_estimate(text) -> datetime | None
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone

from reflexio.models.api_schema.stall_state_schema import StallReason

_LOGGER = logging.getLogger(__name__)

_BILLING_CATEGORIES = {"billing_error"}
_AUTH_CATEGORIES = {"authentication_failed", "oauth_org_not_allowed"}

_TEXT_PATTERNS_BILLING = (
    "hit your weekly limit",
    "hit your session limit",
    "credit balance is too low",
    "billing_error",
)
_TEXT_PATTERNS_AUTH = (
    "please run /login",
    "oauth token revoked",
    "oauth token has expired",
    "not logged in",
)

_RESET_RE = re.compile(
    r"resets\s+(?:(?P<weekday>mon|tue|wed|thu|fri|sat|sun)\w*\s+)?"
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>am|pm)",
    re.IGNORECASE,
)


@dataclass
class ParseResult:
    """Aggregated state of one ``claude -p`` invocation.

    Attributes:
        success (bool): True iff exit_code==0 AND a terminal result event
            with non-empty text appeared.
        terminal_text (str): The ``result`` field from the terminal event.
        retry_errors (list[str]): All ``error`` strings observed in
            ``api_retry`` events, in order.
        stderr_text (str): Raw stderr text from the subprocess.
        raw_lines_parsed (int): NDJSON lines parsed successfully.
        raw_lines_failed (int): NDJSON lines that failed to parse.
    """

    success: bool
    terminal_text: str
    retry_errors: list[str] = field(default_factory=list)
    stderr_text: str = ""
    raw_lines_parsed: int = 0
    raw_lines_failed: int = 0

    @property
    def stall_candidate(self) -> str | None:
        """The last retry error string in a stall category, or None."""
        for err in reversed(self.retry_errors):
            if err in _BILLING_CATEGORIES or err in _AUTH_CATEGORIES:
                return err
        return None


def parse_stream_json(
    stdout: str,
    *,
    exit_code: int,
    stderr_text: str = "",
) -> ParseResult:
    """Parse the full NDJSON stream from claude -p.

    Args:
        stdout (str): Raw subprocess stdout — newline-delimited JSON events.
        exit_code (int): Subprocess exit code.
        stderr_text (str): Raw stderr — used by the text-fallback path.

    Returns:
        ParseResult: Aggregated state. ``success`` requires both a clean
            exit and a parseable terminal ``result`` event.
    """
    retry_errors: list[str] = []
    terminal_text = ""
    parsed = 0
    failed = 0
    saw_terminal = False
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            failed += 1
            continue
        parsed += 1
        if not isinstance(event, dict):
            continue
        match event.get("type"), event.get("subtype"):
            case ("system", "api_retry"):
                err = event.get("error")
                if isinstance(err, str):
                    retry_errors.append(err)
            case ("result", _) | (_, "result"):
                text = event.get("result")
                if isinstance(text, str):
                    terminal_text = text
                    saw_terminal = True
    return ParseResult(
        success=(exit_code == 0 and saw_terminal and bool(terminal_text)),
        terminal_text=terminal_text,
        retry_errors=retry_errors,
        stderr_text=stderr_text,
        raw_lines_parsed=parsed,
        raw_lines_failed=failed,
    )


def classify_stall(result: ParseResult) -> StallReason | None:
    """Decide whether a finished ParseResult represents a stall.

    Args:
        result (ParseResult): Output of :func:`parse_stream_json`.

    Returns:
        StallReason | None: ``"billing_error"`` or ``"auth_error"`` only
            when the call terminated unsuccessfully AND a stall-class
            signal was observed (event or text fallback).
    """
    if result.success:
        return None
    candidate = result.stall_candidate
    if candidate in _BILLING_CATEGORIES:
        return "billing_error"
    if candidate in _AUTH_CATEGORIES:
        return "auth_error"
    return _classify_from_text(result.stderr_text + " " + result.terminal_text)


def _classify_from_text(text: str) -> StallReason | None:
    """Belt-and-suspenders: regex the raw error text after NDJSON has been parsed."""
    lower = text.lower()
    if any(p in lower for p in _TEXT_PATTERNS_BILLING):
        _LOGGER.warning("Stall classified via text fallback (billing): %r", text[:200])
        return "billing_error"
    if any(p in lower for p in _TEXT_PATTERNS_AUTH):
        _LOGGER.warning("Stall classified via text fallback (auth): %r", text[:200])
        return "auth_error"
    return None


def parse_reset_estimate(text: str) -> datetime | None:
    """Best-effort parse of a reset time from Claude Code error text.

    The hour/minute extracted from the message text is treated as UTC, then
    advanced one day if it has already passed. Claude Code's error messages
    typically express reset times in the user's local timezone, so the
    returned datetime may be off by the user's UTC offset (up to ±24h).
    Callers should treat this as an approximation, not an authoritative
    reset moment.

    Args:
        text (str): Error text — typically the terminal event message or stderr.

    Returns:
        datetime | None: A UTC-naive-in-spirit but tz-aware datetime that
            best approximates the next reset, or None when no recognizable
            pattern is found.
    """
    match = _RESET_RE.search(text or "")
    if not match:
        return None
    hour_raw = int(match.group("hour"))
    minute = int(match.group("minute"))
    # The regex permits 1–2 digit hours and minutes; reject out-of-range
    # values so "13:00pm" or "10:75am" parse as None instead of producing
    # silently-wrong times via modulo arithmetic.
    if not 1 <= hour_raw <= 12 or not 0 <= minute <= 59:
        return None
    hour = hour_raw % 12
    if match.group("ampm").lower() == "pm":
        hour += 12
    today = datetime.now(timezone.utc).date()
    candidate = datetime.combine(today, time(hour, minute), tzinfo=timezone.utc)
    if candidate <= datetime.now(timezone.utc):
        candidate += timedelta(days=1)
    return candidate
