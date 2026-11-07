"""Helpers for ``.github/workflows/upstream-canary.yml``.

The canary watches the runtime dependencies reflexio-ai keeps a ceiling on.
For each it needs three facts this script answers from the repository:

* ``locked <package>``: the version ``uv.lock`` resolves;
* ``admits <package> <version>``: whether ``pyproject.toml``'s requirement
  admits a version (``true``/``false``);
* ``tests <package>``: the contract tests that qualify the package.

Run it with ``uv run --no-project --with packaging python``; it needs no
project environment.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent

#: Tests that exercise each watched package, relative to the repository root.
CONTRACT_TESTS: dict[str, tuple[str, ...]] = {
    # LiteLLMClient: request/response, tool-call, reasoning and mock contracts.
    "litellm": (
        "tests/server/llm/test_litellm_client.py",
        "tests/server/llm/test_litellm_client_surface.py",
        "tests/server/llm/test_litellm_client_tool_calls.py",
        "tests/server/llm/test_litellm_client_unit.py",
        "tests/server/llm/test_litellm_subprocess_fork_reset.py",
        "tests/server/llm/test_completion_request_contract.py",
        "tests/server/llm/test_completion_response_transport.py",
        "tests/server/llm/test_completion_io.py",
        "tests/server/llm/test_reasoning_exhausted_thinking_retry.py",
        "tests/server/llm/test_input_budget.py",
        "tests/server/llm/test_model_defaults.py",
        "tests/server/llm/test_llm_mock_collection.py",
        "tests/server/llm/test_llm_mock_patching.py",
        "tests/server/llm/test_provider_outage_classification_guard.py",
    ),
    # The GEPA adapter, optimizer loop and winner publication.
    "gepa": ("tests/server/services/playbook_optimizer",),
}


def locked(package: str, lock: Path = ROOT / "uv.lock") -> str:
    entries = tomllib.loads(lock.read_text())["package"]
    versions = [entry["version"] for entry in entries if entry["name"] == package]
    if len(versions) != 1:
        raise SystemExit(f"uv.lock holds {len(versions)} entries for {package}")
    return versions[0]


def admits(package: str, version: str, project: Path = ROOT / "pyproject.toml") -> bool:
    dependencies = tomllib.loads(project.read_text())["project"]["dependencies"]
    matches = [
        requirement
        for requirement in map(Requirement, dependencies)
        if requirement.name == package
    ]
    if len(matches) != 1:
        raise SystemExit(f"pyproject.toml declares {package} {len(matches)} times")
    return matches[0].specifier.contains(Version(version), prereleases=True)


def tests(package: str) -> tuple[str, ...]:
    try:
        return CONTRACT_TESTS[package]
    except KeyError:
        raise SystemExit(f"No contract tests recorded for {package}") from None


def main(arguments: list[str]) -> str:
    match arguments:
        case ["locked", package]:
            return locked(package)
        case ["admits", package, version]:
            return str(admits(package, version)).lower()
        case ["tests", package]:
            return " ".join(tests(package))
    raise SystemExit(f"usage: {Path(__file__).name} locked|admits|tests ...")


if __name__ == "__main__":
    print(main(sys.argv[1:]))
