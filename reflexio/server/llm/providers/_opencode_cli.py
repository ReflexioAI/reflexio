"""OpenCode's native wire format under the shared immediate-process owner."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from reflexio.opencode_executable import OpenCodeCLIError, resolve_executable
from reflexio.server.llm._litellm_types import LLMHardTimeoutError
from reflexio.server.llm._native_cli import (
    native_cli_directory,
    remaining_native_time,
    run_native_cli,
)
from reflexio.server.llm._native_model_ref import (
    reject_native_selection,
)
from reflexio.server.llm.providers.claude_code_stream_parser import (
    ParseResult,
    classify_stall,
    parse_reset_estimate,
)

_AGENT = "claude-smart-extractor"


def _declared_run_options(help_text: str) -> set[str]:
    """Recognize declared CLI capabilities, never mentions in diagnostic prose."""
    return set(
        re.findall(
            r"(?m)^\s*(?:-[A-Za-z0-9],\s*)?(--[A-Za-z0-9][A-Za-z0-9-]*)(?=[,\s=]|$)",
            help_text,
        )
    )


def _run_options(help_text: str) -> set[str]:
    options = _declared_run_options(help_text)
    if not {"--model", "--agent", "--format", "--title"} <= options:
        raise OpenCodeCLIError("OpenCode CLI capabilities are unsupported")
    if "--standalone" in options or {"--attach", "--variant"} <= options:
        return options
    raise OpenCodeCLIError("OpenCode CLI capabilities are unsupported")


def _wire_argument(option: str, value: str, *, legacy_wire: bool) -> list[str]:
    # Older wrappers consume separate tokens. Bind a flag-shaped value inline
    # so it can never become another option on either command line.
    if legacy_wire and not value.startswith("-"):
        return [option, value]
    return [f"{option}={value}"]


def _model_arguments(
    model: str | None,
    options: set[str],
    *,
    wire: Literal["native", "plugin", "optimizer"],
) -> list[str]:
    if not model:
        return []
    legacy_wire = wire != "native"
    selected, separator, variant = model.partition("#")
    if separator and ("--standalone" not in options or not selected):
        if "--variant" not in options:
            reject_native_selection(
                "Selected OpenCode CLI does not support a variant without a model"
            )
        flags = (
            _wire_argument("--model", selected, legacy_wire=legacy_wire)
            if selected
            else []
        )
        return flags + _wire_argument(
            "--variant", variant, legacy_wire=wire == "optimizer"
        )
    return _wire_argument("--model", model, legacy_wire=legacy_wire)


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
    deadline: float,
    wire: Literal["native", "plugin", "optimizer"] = "native",
) -> ParseResult:
    """Launch the actual native executable; retain delegated scratch with its owner."""
    legacy_wire = wire != "native"
    selected, separator, variant = (model or "").partition("#")
    if separator and ((selected and not selected.strip()) or not variant.strip()):
        reject_native_selection("OpenCode model variant reference is invalid")
    owner = native_cli_directory()
    directory: Path | None = None
    try:
        executable = resolve_executable(cli_path)
        if os.name == "nt" and executable.suffix.lower() != ".exe":
            raise OpenCodeCLIError(
                "Native OpenCode requires an .exe; select it with CLAUDE_SMART_CLI_PATH. "
                "Use an npm package with a native .exe or keep the compatibility bridge for older and custom launchers."
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
        capture = {
            "cwd": str(directory),
            "env": env,
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace" if os.name == "nt" else "strict",
            "check": False,
        }
        try:
            options: set[str] = set()
            # The plugin bridge only needed discovery for variant references.
            # In particular, a CLI wrapper may implement generation without help.
            if not legacy_wire or separator:
                discovery = run_native_cli(
                    [str(executable), "run", "--help"],
                    input="",
                    timeout=remaining_native_time(deadline),
                    **capture,
                )
                if discovery.returncode != 0:
                    raise OpenCodeCLIError("OpenCode CLI capability discovery failed")
                help_text = discovery.stdout + "\n" + discovery.stderr
                options = (
                    _declared_run_options(help_text)
                    if legacy_wire
                    else _run_options(help_text)
                )
            standalone = "--standalone" in options
            argv = [str(executable), "run", "--format", "json", "--agent", _AGENT]
            if not legacy_wire:
                argv.extend(["--title", _AGENT])
                if standalone:
                    argv.append("--standalone")
            argv.extend(_model_arguments(model, options, wire=wire))
            result = run_native_cli(
                argv,
                input=prompt,
                timeout=remaining_native_time(deadline),
                **capture,
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
        parsed = parse_output(result.stdout, result.returncode, result.stderr)
        remaining_native_time(deadline)
        return parsed
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
