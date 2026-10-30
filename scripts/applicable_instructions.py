#!/usr/bin/env python3
"""Print the project instruction files that apply to changed files.

For each changed file, walk from its directory up to the checkout root and
collect ``AGENTS.override.md`` and ``AGENTS.md``, nearest directory first and,
within one directory, ``AGENTS.override.md`` before ``AGENTS.md``. Output is
grouped by instruction file (deduplicated) with the changed files it covers
and its content with line numbers, so a reviewer can cite a rule and its line
range without searching directories.

Submodules: the checkout root is the OUTERMOST superproject's working tree.
A file inside a submodule therefore gets the submodule's own instruction
files first, then the superproject's (lower precedence), because work in a
submodule checkout is still governed by the superproject's instructions.
A standalone clone has no superproject, so its own root is the boundary.

Usage:
    scripts/applicable_instructions.py [files...]
    scripts/applicable_instructions.py --base origin/main --paths-only

With no files, the changed files are ``git diff --name-only`` against the
merge base of HEAD and ``--base`` (default ``origin/main``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Within one directory, earlier names take precedence.
INSTRUCTION_FILENAMES: tuple[str, ...] = ("AGENTS.override.md", "AGENTS.md")


class InstructionError(Exception):
    """A user-facing failure (not a git repo, bad base, path outside root)."""


@dataclass(frozen=True)
class Applicable:
    """One instruction file and the changed files it applies to."""

    path: Path
    changed: tuple[Path, ...]


def _git(cwd: Path, *args: str) -> str:
    try:
        result = subprocess.run(  # noqa: S603 - fixed git argv, no shell
            ["git", *args],  # noqa: S607 - git resolved from PATH by design
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise InstructionError("git is not installed or not on PATH") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise InstructionError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def repo_toplevel(cwd: Path) -> Path:
    """Return the top of the git work tree containing ``cwd``."""
    try:
        return Path(_git(cwd, "rev-parse", "--show-toplevel").strip()).resolve()
    except InstructionError as exc:
        raise InstructionError(f"not inside a git repository: {cwd}") from exc


def checkout_root(cwd: Path) -> Path:
    """Return the outermost superproject work tree (see module docstring)."""
    root = repo_toplevel(cwd)
    while True:
        parent = _git(root, "rev-parse", "--show-superproject-working-tree").strip()
        if not parent:
            return root
        root = Path(parent).resolve()


def changed_files(cwd: Path, base: str) -> list[Path]:
    """Return absolute paths changed versus the merge base of HEAD and base."""
    top = repo_toplevel(cwd)
    merge_base = _git(top, "merge-base", "HEAD", base).strip()
    # -z keeps paths exact; line output C-quotes non-ASCII, tabs and newlines.
    names = _git(top, "diff", "--name-only", "-z", merge_base).split("\0")
    return [top / name for name in names if name]


def instruction_chain(changed: Path, root: Path) -> list[Path]:
    """Instruction files for one changed path, highest precedence first."""
    # A changed symlink is a file entry in Git even when it points at a
    # directory, so its own directory decides the scope.
    is_dir = changed.is_dir() and not changed.is_symlink()
    start = changed if is_dir else changed.parent
    try:
        start.relative_to(root)
    except ValueError as exc:
        raise InstructionError(f"{changed} is outside the checkout {root}") from exc
    chain: list[Path] = []
    directory = start
    while True:
        chain.extend(
            directory / name
            for name in INSTRUCTION_FILENAMES
            if (directory / name).is_file()
        )
        if directory == root:
            return chain
        directory = directory.parent


def collect(changed: list[Path], root: Path) -> list[Applicable]:
    """Group instruction files across changed files, deduplicated.

    Groups keep first-seen order over each file's precedence chain, so the
    nearest files of the first changed path lead and shared root files trail.
    """
    covers: dict[Path, list[Path]] = {}
    for path in changed:
        for instruction in instruction_chain(path, root):
            covered = covers.setdefault(instruction, [])
            if path not in covered:
                covered.append(path)
    return [Applicable(path, tuple(files)) for path, files in covers.items()]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() or "."


def render(applicable: list[Applicable], root: Path, *, paths_only: bool) -> str:
    """Format the grouped result for printing."""
    if paths_only:
        return "".join(f"{_relative(a.path, root)}\n" for a in applicable)
    blocks: list[str] = []
    for item in applicable:
        lines = item.path.read_text(encoding="utf-8").splitlines()
        width = max(3, len(str(len(lines))))
        header = [f"== {_relative(item.path, root)}", "applies to:"]
        header += [f"  {_relative(path, root)}" for path in item.changed]
        body = [
            f"{n:>{width}}: {text}" if text else f"{n:>{width}}:"
            for n, text in enumerate(lines, start=1)
        ]
        blocks.append("\n".join([*header, "--", *body]) + "\n")
    return "\n".join(blocks)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", help="changed files (default: git diff)")
    parser.add_argument(
        "--base",
        default="origin/main",
        help="ref whose merge base with HEAD defines the diff (default: origin/main)",
    )
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="print only the applicable instruction file paths",
    )
    return parser


def main(argv: list[str] | None = None, cwd: Path | None = None) -> int:
    """CLI entry point. Returns 0 on success, 2 on a reported error."""
    args = _parser().parse_args(argv)
    here = (cwd or Path.cwd()).resolve()
    try:
        root = checkout_root(here)
        if args.files:
            # Normalize lexically: resolve() would replace a changed symlink
            # with its target and lose the path Git reports.
            changed = [Path(os.path.normpath(here / name)) for name in args.files]
        else:
            changed = changed_files(here, args.base)
        applicable = collect(changed, root)
    except InstructionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not changed:
        if not args.paths_only:
            print("No changed files.")
        return 0
    sys.stdout.write(render(applicable, root, paths_only=args.paths_only))
    return 0


if __name__ == "__main__":
    sys.exit(main())
