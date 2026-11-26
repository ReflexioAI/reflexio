"""Find the selected OpenCode program without initializing the server or LLM providers."""

import os
import shutil
from pathlib import Path


class OpenCodeCLIError(RuntimeError):
    """Metadata-only transport failure; never includes captured model/error text."""


def _npm_opencode_executable(shim: Path) -> Path | None:
    """Recognize npm's forwarding-only launcher without running batch code."""
    if shim.name.lower() != "opencode.cmd" or not shim.is_file():
        return None
    local = (
        shim.parent.name.lower() == ".bin"
        and shim.parent.parent.name.lower() == "node_modules"
    )
    prefix = ("..",) if local else ("node_modules",)
    suffix = ("opencode-ai", "bin", "opencode.exe")
    target = "\\".join((*prefix, *suffix))
    try:
        with shim.open("rb") as source:
            raw = source.read(2049)
        if len(raw) > 2048:
            return None
        commands = (
            raw.decode("ascii").replace("\r\n", "\n").lower().rstrip("\n").split("\n")
        )
        if commands[:-1] != [
            "@echo off",
            "goto start",
            ":find_dp0",
            "set dp0=%~dp0",
            "exit /b",
            ":start",
            "setlocal",
            "call :find_dp0",
        ] or commands[-1].split() != [f'"%dp0%\\{target}"', "%*"]:
            return None
        parent = shim.parent.parent if local else shim.parent / "node_modules"
        executable = parent.joinpath(*suffix)
        if executable.is_file():
            with executable.open("rb") as binary:
                if binary.read(2) == b"MZ":
                    return executable
    except (OSError, UnicodeError):
        return None
    return None


def resolve_executable(cli_path: str) -> Path:
    """Resolve only the selected path, preserving a bridge symlink's basename."""
    path = Path(cli_path)
    candidates = (
        # An empty PATH entry (e.g. a trailing separator) would anchor to the
        # working directory; a bare name is searched on real entries only.
        (
            Path(directory) / path
            for directory in os.get_exec_path()
            if directory.strip()
        )
        if path.name == cli_path
        else (path,)
    )
    for candidate in candidates:
        # An absolute candidate prevents Windows' implicit current-directory
        # search; which still supplies platform PATHEXT handling.
        absolute = candidate.absolute()
        # An exact executable path is independent of PATHEXT. In particular,
        # a package binary returned by npm shim resolution can be used again.
        if os.name == "nt" and absolute.suffix.lower() == ".exe" and absolute.is_file():
            return absolute
        found = shutil.which(str(absolute), path="")
        if found:
            executable = Path(found).absolute()
            if os.name == "nt":
                return _npm_opencode_executable(executable) or executable
            return executable
    raise OpenCodeCLIError("OpenCode selected executable is unavailable")
