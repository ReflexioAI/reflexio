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
import os
import subprocess
import time
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from reflexio.http_tls import tls_verify_from_env
from reflexio.models.api_schema.domain import AgentPlaybook
from reflexio.server.llm._litellm_types import LLMHardTimeoutError
from reflexio.server.llm._native_cli import request_timeout_seconds, run_native_cli

from .assistant_native import (
    NativeAssistantError,
    NativeAssistantRequest,
    remaining_assistant_time,
    run_native_assistant,
)
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
    """Raised when ``WebhookAssistant`` cannot complete an invocation."""


class LocalScriptFailedError(AssistantFailedError):
    """Raised when ``LocalScriptAssistant`` cannot complete an invocation."""


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
    """Send one POST per invocation; the integration owns any safe replay.

    Failures, redirects and malformed output abort the current evaluation.
    A later evaluator call is distinct work, not an exactly-once guarantee.
    ``timeout_s`` bounds each phase (connect, write, read, pool), as the
    former transport bounded connect and each read.
    """

    def __init__(self, url: str, auth_header: str | None, timeout_s: int) -> None:
        self.url = url
        self.auth_header = auth_header
        self.timeout_s = timeout_s

    def __call__(
        self, messages: list[ChatMessage], playbooks: list[AgentPlaybook]
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        try:
            response = httpx.post(
                self.url,
                json=_build_payload(messages, playbooks),
                headers=headers,
                timeout=self.timeout_s,
                follow_redirects=False,
                verify=tls_verify_from_env(),
            )
            if not 200 <= response.status_code < 300:
                raise WebhookFailedError(
                    f"assistant webhook returned HTTP {response.status_code}"
                )
            try:
                data = response.json()
            except ValueError:
                raise WebhookFailedError(
                    "assistant webhook returned invalid JSON"
                ) from None
            content = data.get("content") if isinstance(data, dict) else None
            if not isinstance(content, str):
                raise WebhookFailedError("assistant webhook returned no content")
            return content
        except (httpx.HTTPError, httpx.InvalidURL):
            # The server may have executed. Do not replay or expose URL/body/auth.
            raise WebhookFailedError("assistant webhook transport failed") from None


class LocalScriptAssistant:
    """Compile one script response and execute its selected native call once.

    The command is a list without a shell. Failure/invalid output/timeout aborts
    this evaluation; scripts own any retries they can perform safely. The
    native result form lets the caller launch the actual CLI after compilation.
    Retirement covers each immediate child, not arbitrary wrapper descendants.
    """

    def __init__(
        self, script_path: str, script_args: list[str], timeout_s: int
    ) -> None:
        self.command = [script_path, *script_args]
        self.timeout_s = timeout_s

    def __call__(
        self, messages: list[ChatMessage], playbooks: list[AgentPlaybook]
    ) -> str:
        deadline = time.monotonic() + request_timeout_seconds(self.timeout_s)
        env = {
            **os.environ,
            "_REFLEXIO_ASSISTANT_INVOCATION": json.dumps(
                {"protocol": "native-cli/v1", "deadline_monotonic": deadline}
            ),
        }
        try:
            result = run_native_cli(
                self.command,
                input=json.dumps(
                    _build_payload(messages, playbooks), ensure_ascii=False
                ),
                text=True,
                capture_output=True,
                timeout=remaining_assistant_time(deadline),
                check=False,
                env=env,
            )
        except (subprocess.TimeoutExpired, LLMHardTimeoutError, NativeAssistantError):
            raise LocalScriptFailedError(
                f"assistant script timed out after {self.timeout_s}s"
            ) from None
        except (OSError, ValueError):
            raise LocalScriptFailedError("assistant script could not start") from None
        if result.returncode != 0:
            raise LocalScriptFailedError(
                f"assistant script exited with code {result.returncode}"
            )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            raise LocalScriptFailedError(
                "assistant script returned invalid JSON"
            ) from None
        if isinstance(data, dict) and "native_cli_v1" in data:
            if set(data) != {"native_cli_v1"}:
                raise LocalScriptFailedError(
                    "assistant script returned ambiguous output"
                )
            try:
                request = NativeAssistantRequest.model_validate(data["native_cli_v1"])
            except ValidationError:
                raise LocalScriptFailedError(
                    "assistant script returned an invalid native request"
                ) from None
            try:
                return run_native_assistant(request, deadline)
            except NativeAssistantError as exc:
                raise LocalScriptFailedError(str(exc)) from None
        content = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, str):
            raise LocalScriptFailedError("assistant script returned no content")
        return content
