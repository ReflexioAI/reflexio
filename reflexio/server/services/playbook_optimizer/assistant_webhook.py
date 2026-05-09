"""Assistant backends for the playbook optimizer.

The playbook optimizer runs paired rollouts (incumbent vs candidate playbook)
against a "real" assistant. This module hosts the two pluggable backends that
produce the assistant's reply for a given turn:

- ``WebhookAssistant``      — POSTs JSON to a configured URL.
- ``LocalScriptAssistant``  — spawns a subprocess and exchanges JSON via
  stdin/stdout.

Both satisfy ``AssistantCallable``, so ``MultiTurnRollout`` and the GEPA
adapter never need to know which one is in use. Selection happens once in
``PlaybookOptimizer._create_assistant`` based on config.

Failures from either backend raise ``AssistantFailedError`` (or a subclass);
the GEPA adapter catches that single base class and turns it into an
``aborted`` evaluation row instead of crashing the search loop.

The file is named ``assistant_webhook.py`` for historical reasons — the
local-script backend was added later. New code should still import from this
module.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Protocol

import requests

from reflexio.models.api_schema.domain import AgentPlaybook

from .models import ChatMessage


class AssistantCallable(Protocol):
    """Shape every assistant backend must satisfy.

    A backend takes the conversation so far plus the playbook(s) to inject
    and returns the assistant's next reply as a plain string. Errors are
    signalled by raising ``AssistantFailedError`` (or a subclass).
    """

    def __call__(
        self, messages: list[ChatMessage], playbooks: list[AgentPlaybook]
    ) -> str: ...


class AssistantFailedError(Exception):
    """Base class for any assistant-backend failure.

    The GEPA adapter catches this single type to absorb backend faults
    (network errors, subprocess crashes, malformed responses) and record
    them as ``verdict='aborted'`` evaluations rather than aborting the
    optimizer run.
    """


class WebhookFailedError(AssistantFailedError):
    """Raised when ``WebhookAssistant`` exhausts its retries."""


class LocalScriptFailedError(AssistantFailedError):
    """Raised when ``LocalScriptAssistant`` exhausts its retries."""


def _build_payload(
    messages: list[ChatMessage], playbooks: list[AgentPlaybook]
) -> dict[str, Any]:
    """Build the wire payload shared by both backends.

    Centralising this guarantees that the webhook body and the script's
    stdin contain the *same* fields in the *same* shape — the only thing
    that differs between the two transports is how the bytes are delivered.
    """
    return {
        "messages": [message.model_dump() for message in messages],
        "playbooks": [
            {
                "id": pb.agent_playbook_id,
                "content": pb.content,
                "trigger": pb.trigger,
            }
            for pb in playbooks
        ],
    }


class WebhookAssistant:
    """HTTP assistant backend.

    POSTs the rollout payload to ``url`` and expects ``{"content": str}`` in
    the response body. Retries any exception (network error, non-2xx,
    malformed JSON, missing field) with exponential backoff up to
    ``max_retries`` additional attempts before raising
    ``WebhookFailedError``.

    Args:
        url: Destination URL. The optimizer does not validate the scheme,
            so operators are responsible for ensuring it points at a host
            that satisfies their data-residency requirements.
        auth_header: Sent verbatim as the ``Authorization`` header. Never
            logged.
        timeout_s: Per-request timeout (passed to ``requests.post``).
        max_retries: Number of *additional* attempts after the first try.
            ``max_retries=0`` means a single attempt.
        backoff_base_s: Base for the exponential delay
            ``backoff_base_s * 2**attempt`` between retries.
    """

    def __init__(
        self,
        url: str,
        auth_header: str | None,
        timeout_s: int,
        max_retries: int,
        backoff_base_s: float,
    ) -> None:
        self.url = url
        self.auth_header = auth_header
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

    def __call__(
        self, messages: list[ChatMessage], playbooks: list[AgentPlaybook]
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        payload = _build_payload(messages, playbooks)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("content")
                if not isinstance(content, str):
                    raise WebhookFailedError("assistant webhook returned no content")
                return content
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_base_s * (2**attempt))
        # Wrap-and-rethrow ensures the adapter sees a single AssistantFailedError
        # subclass regardless of which underlying exception triggered the failure.
        raise WebhookFailedError(str(last_error)) from last_error


class LocalScriptAssistant:
    """Subprocess assistant backend.

    Spawns ``[script_path, *script_args]`` per turn and exchanges JSON over
    stdin/stdout. The script must:

    1. Read a single JSON object from stdin matching ``_build_payload``'s
       shape (``{"messages": [...], "playbooks": [...]}``).
    2. Print a single JSON object to stdout containing at least
       ``{"content": "<assistant reply>"}``.
    3. Exit with code 0 on success. Any non-zero exit, malformed JSON,
       missing/non-string ``content``, or timeout is treated as a failure
       and counted against ``max_retries``.

    The command is built as a list and passed to ``subprocess.run`` *without*
    ``shell=True``, so config values cannot inject shell metacharacters.
    Retry/timeout/backoff semantics mirror ``WebhookAssistant``.
    """

    def __init__(
        self,
        script_path: str,
        script_args: list[str],
        timeout_s: int,
        max_retries: int,
        backoff_base_s: float,
    ) -> None:
        self.command = [script_path, *script_args]
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s

    def __call__(
        self, messages: list[ChatMessage], playbooks: list[AgentPlaybook]
    ) -> str:
        payload = _build_payload(messages, playbooks)
        stdin = json.dumps(payload, ensure_ascii=False)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                # shell=False (the default) is intentional — see class docstring.
                result = subprocess.run(  # noqa: S603
                    self.command,
                    input=stdin,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_s,
                    check=False,
                )
                if result.returncode != 0:
                    # Truncate stderr to keep error messages bounded — long
                    # tracebacks would otherwise blow up DB rows and logs.
                    raise LocalScriptFailedError(
                        "assistant script exited with code "
                        f"{result.returncode}: {_truncate(result.stderr)}"
                    )
                try:
                    data = json.loads(result.stdout)
                except json.JSONDecodeError as exc:
                    raise LocalScriptFailedError(
                        f"assistant script returned invalid JSON: {_truncate(result.stdout)}"
                    ) from exc
                content = data.get("content") if isinstance(data, dict) else None
                if not isinstance(content, str):
                    raise LocalScriptFailedError("assistant script returned no content")
                return content
            except subprocess.TimeoutExpired:
                # Timeouts come up via a different exception class than the
                # validation errors above, but feed into the same retry budget.
                last_error = LocalScriptFailedError(
                    f"assistant script timed out after {self.timeout_s}s"
                )
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_base_s * (2**attempt))
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_base_s * (2**attempt))
        raise LocalScriptFailedError(str(last_error)) from last_error


def _truncate(value: str, limit: int = 1000) -> str:
    """Cap a string for inclusion in error messages and stored rows."""
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."
