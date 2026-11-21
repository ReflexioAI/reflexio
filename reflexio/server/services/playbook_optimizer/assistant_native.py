"""Execute a compile-only assistant result under the optimizer caller's budget."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from reflexio.server.llm._litellm_types import LLMHardTimeoutError
from reflexio.server.llm._native_cli import remaining_native_time, run_native_cli
from reflexio.server.llm._native_model_ref import (
    CLI_TIMEOUT_ENV,
    NativeCLISelectionError,
)

NativeAssistantProtocol = Literal[
    "claude-json", "codex", "opencode-json", "opencode-compat-json"
]
NATIVE_ASSISTANT_TRANSPORTS = get_args(NativeAssistantProtocol)


class NativeAssistantError(Exception):
    """A metadata-only native assistant failure, with no replay authorization."""


class NativeAssistantRequest(BaseModel):
    """A selected invocation, without caller-controlled permissions or budgets."""

    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: NativeAssistantProtocol
    executable: str = Field(min_length=1)
    model: str | None = None
    system_prompt: str
    dialogue: str


def remaining_assistant_time(deadline: float) -> float:
    """Spend one caller deadline across compilation, parsing and execution."""
    try:
        return remaining_native_time(deadline)
    except LLMHardTimeoutError:
        raise NativeAssistantError("native assistant deadline expired") from None


def _compat_deadline(deadline: float) -> float:
    # Preserve an explicit cap supported by the old bridge. Without one,
    # the original request owner supplies the full remaining allowance.
    try:
        cap = float(os.environ.get(CLI_TIMEOUT_ENV, ""))
    except ValueError:
        return deadline
    return (
        min(deadline, time.monotonic() + cap)
        if math.isfinite(cap) and cap > 0
        else deadline
    )


def run_native_assistant(request: NativeAssistantRequest, deadline: float) -> str:
    """Own the immediate CLI child; arbitrary wrappers retain descendant limits."""
    from reflexio.server.llm.providers._opencode_cli import (
        OpenCodeCLIError,
        run_opencode,
    )
    from reflexio.server.llm.providers.native_cli_provider import (
        ClaudeCodeCLIError,
        ClaudeCodeCLITimeoutError,
        _run_codex_stream,
    )

    try:
        remaining_assistant_time(deadline)
        if request.protocol == "codex":
            result = _run_codex_stream(
                codex_path=request.executable,
                system_prompt=request.system_prompt,
                dialogue=request.dialogue,
                model=request.model,
                optimizer=True,
                deadline=deadline,
            )
            if not result.success:
                raise NativeAssistantError("native assistant execution failed")
            content = result.terminal_text
        elif request.protocol in {"opencode-json", "opencode-compat-json"}:
            legacy_wire = request.protocol == "opencode-compat-json"
            if legacy_wire:
                deadline = _compat_deadline(deadline)
            result = run_opencode(
                cli_path=request.executable,
                system_prompt=request.system_prompt,
                dialogue=request.dialogue,
                model=request.model,
                deadline=deadline,
                wire="optimizer" if legacy_wire else "native",
            )
            if not result.success:
                raise NativeAssistantError("native assistant execution failed")
            content = result.terminal_text
        else:
            content = _run_claude_json(request, deadline)
        remaining_assistant_time(deadline)
        return content
    except (subprocess.TimeoutExpired, LLMHardTimeoutError, ClaudeCodeCLITimeoutError):
        raise NativeAssistantError("native assistant timed out") from None
    except (ClaudeCodeCLIError, OpenCodeCLIError, NativeCLISelectionError):
        raise NativeAssistantError("native assistant execution failed") from None
    except (OSError, ValueError):
        raise NativeAssistantError("native assistant execution failed") from None


def _run_claude_json(request: NativeAssistantRequest, deadline: float) -> str:
    # Preserve the optimizer's custom Claude JSON executable contract, rather
    # than substituting the generation provider's stream-json wire format.
    argv = [request.executable, "-p", "--output-format", "json"]
    if request.model:
        argv += ["--model", request.model]
    argv += [
        "--permission-mode",
        "plan",
        "--tools",
        "Read,Grep,Glob,LS",
        "--disallowedTools",
        "Bash,Edit,Write,MultiEdit,NotebookEdit",
        "--no-session-persistence",
        "--mcp-config",
        '{"mcpServers": {}}',
        "--strict-mcp-config",
    ]
    if request.system_prompt:
        argv += ["--append-system-prompt", request.system_prompt]
    env = {
        **os.environ,
        "CLAUDE_SMART_INTERNAL": "1",
        "CLAUDE_CODE_ENTRYPOINT": "optimizer",
    }
    result = run_native_cli(
        argv,
        input=request.dialogue,
        capture_output=True,
        text=True,
        timeout=remaining_assistant_time(deadline),
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise NativeAssistantError("native assistant execution failed")
    try:
        response = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise NativeAssistantError("native assistant returned invalid JSON") from None
    content = response.get("result") if isinstance(response, dict) else None
    if not isinstance(content, str) and isinstance(response, dict):
        content = response.get("response")
    if not isinstance(content, str):
        raise NativeAssistantError("native assistant returned no content")
    return content
