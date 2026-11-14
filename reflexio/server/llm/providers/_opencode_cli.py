"""OpenCode's native wire format under the shared immediate-process owner."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from reflexio.server.llm._litellm_types import LLMHardTimeoutError
from reflexio.server.llm._native_cli import (
    native_cli_directory,
    request_timeout_seconds,
    run_native_cli,
)
from reflexio.server.llm.providers.claude_code_stream_parser import (
    ParseResult,
    classify_stall,
    parse_reset_estimate,
)

_AGENT = "claude-smart-extractor"


class OpenCodeCLIError(RuntimeError):
    """Metadata-only transport failure; never includes captured model/error text."""


def _error_message(error: Any) -> str:
    if isinstance(error, str):
        return error
    if not isinstance(error, dict):
        return ""
    data = error.get("data")
    message = (
        error.get("message")
        or (data.get("message") if isinstance(data, dict) else None)
        or error.get("name")
    )
    return message if isinstance(message, str) else ""


def parse_output(stdout: str, returncode: int, stderr: str) -> ParseResult:
    """An explicit error defeats partial text, even when the CLI exits zero."""
    chunks: list[str] = []
    errors = [stderr]
    error_count = parsed = failed = 0
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
        if event.get("type") == "error":
            error_count += 1
            errors.append(_error_message(event.get("error")))
            continue
        part = event.get("part")
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
        elif event.get("type") == "text":
            text = event.get("text")
        elif event.get("type") == "result":
            text = event.get("result")
        else:
            continue
        if isinstance(text, str):
            chunks.append(text)
    content = "".join(chunks).strip()
    fence = re.fullmatch(r"```[^\r\n]*\r?\n([\s\S]*?)\r?\n```[ \t]*", content)
    if fence:
        content = fence.group(1).strip()
    success = returncode == 0 and error_count == 0 and bool(content)
    # Classify only failure-channel evidence. Generated text is not an error
    # signal, and no raw evidence may escape through the provider's sinks.
    error_text = " ".join(errors)
    reason = (
        classify_stall(
            ParseResult(success=False, terminal_text="", stderr_text=error_text)
        )
        if not success
        else None
    )
    category = "authentication_failed" if reason == "auth_error" else reason
    return ParseResult(
        success=success,
        terminal_text=content if success else "",
        retry_errors=[category] if category else [],
        stderr_text=(
            f"OpenCode failure: exit={returncode}; error_events={error_count}"
            if not success
            else ""
        ),
        raw_lines_parsed=parsed,
        raw_lines_failed=failed,
        cli_binary="opencode",
        reset_estimate=parse_reset_estimate(error_text) if reason else None,
    )


def run_opencode(
    *,
    cli_path: str,
    system_prompt: str,
    dialogue: str,
    model: str | None,
    timeout_seconds: float,
) -> ParseResult:
    """Launch the actual native executable; retain delegated scratch with its owner."""
    deadline = time.monotonic() + request_timeout_seconds(timeout_seconds)
    owner = native_cli_directory()
    directory: Path | None = None
    try:
        selected = Path(cli_path)
        if not selected.is_file():
            found = shutil.which(cli_path)
            if not found:
                raise OpenCodeCLIError("OpenCode native executable not found")
            selected = Path(found)
        executable = selected.absolute()
        if os.name == "nt" and executable.suffix.lower() != ".exe":
            raise OpenCodeCLIError(
                "Native OpenCode requires an .exe; select it with CLAUDE_SMART_CLI_PATH. "
                "Batch shims require the existing compatibility bridge."
            )
        directory = Path(tempfile.mkdtemp(prefix="opencode-", dir=owner)).resolve()
        config_dir = directory / ".opencode"
        config_dir.mkdir()
        (config_dir / "opencode.json").write_text(
            json.dumps(
                {
                    "agent": {
                        _AGENT: {
                            "description": "Internal claude-smart learning extraction",
                            "prompt": "You are running as an internal claude-smart extraction subprocess. "
                            "Return only the requested text or JSON. Do not use tools, edit files, "
                            "search the workspace, or ask follow-up questions.",
                            "steps": 2,
                            "permission": "deny",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        argv = [str(executable), "run", "--format", "json", "--agent", _AGENT]
        if model:
            argv += ["--model", model]
        env = {
            **os.environ,
            "PWD": str(directory),
            "OPENCODE_PURE": "1",
            "CLAUDE_SMART_HOST": "opencode",
            "CLAUDE_SMART_INTERNAL": "1",
            "CLAUDE_CODE_ENTRYPOINT": "optimizer",
        }
        prompt = (
            f"{system_prompt}\n\n## Task\n{dialogue}" if system_prompt else dialogue
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OpenCodeCLIError("OpenCode request deadline expired before launch")
        try:
            result = run_native_cli(
                argv,
                input=prompt,
                cwd=str(directory),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace" if os.name == "nt" else "strict",
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise OpenCodeCLIError("OpenCode request timed out") from None
        except UnicodeError:
            raise OpenCodeCLIError(
                "OpenCode native output could not be decoded"
            ) from None
        except LLMHardTimeoutError:
            raise
        except OSError:
            raise OpenCodeCLIError(
                "OpenCode native executable could not start"
            ) from None
        return parse_output(result.stdout, result.returncode, result.stderr)
    except LLMHardTimeoutError:
        raise
    except OSError:
        raise OpenCodeCLIError("OpenCode private workdir setup failed") from None
    finally:
        # Worker death cannot delete a running delegated CLI's cwd. The parent
        # owns the whole directory until its native retirement receipt.
        if owner is None and directory is not None:
            try:
                shutil.rmtree(directory)
            except OSError:
                raise OpenCodeCLIError(
                    "OpenCode private workdir cleanup failed"
                ) from None
