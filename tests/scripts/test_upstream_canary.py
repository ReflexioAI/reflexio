"""The upstream canary watches the ceilings it claims to, with least privilege."""

from pathlib import Path

import pytest
import yaml

from scripts import upstream_canary as canary

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load(
    (ROOT / ".github/workflows/upstream-canary.yml").read_text(encoding="utf-8")
)
# PyYAML reads the ``on`` key as boolean True.
TRIGGERS = WORKFLOW[True]
WATCHED = {"litellm", "gepa"}


def test_every_watched_package_has_existing_contract_tests() -> None:
    assert set(canary.CONTRACT_TESTS) == WATCHED
    for package in WATCHED:
        paths = canary.tests(package)
        assert paths
        for path in paths:
            assert (ROOT / path).exists(), f"{package}: {path} is gone"


@pytest.mark.parametrize("package", sorted(WATCHED))
def test_locked_version_is_admitted_and_the_next_major_is_not(package: str) -> None:
    version = canary.locked(package)
    assert canary.admits(package, version)
    major, minor, *_ = (int(part) for part in version.split("."))
    # litellm is 1.x (ceiling <2); gepa is 0.x (ceiling <0.2).
    beyond = f"{major + 1}.0.0" if major else f"0.{minor + 1}.0"
    assert not canary.admits(package, beyond)


def test_main_reports_lowercase_booleans() -> None:
    assert canary.main(["admits", "litellm", canary.locked("litellm")]) == "true"
    assert canary.main(["admits", "litellm", "2.0.0"]) == "false"
    # An excluded known-bad release inside the ceiling is not admitted either.
    assert canary.main(["admits", "litellm", "1.108.2"]) == "false"
    with pytest.raises(SystemExit):
        canary.main(["tests", "requests"])


def test_canary_never_gates_pull_requests() -> None:
    assert set(TRIGGERS) == {"schedule", "workflow_dispatch"}
    default = TRIGGERS["workflow_dispatch"]["inputs"]["packages"]["default"]
    assert set(default.split()) == WATCHED


def test_only_the_writing_jobs_hold_write_scopes() -> None:
    assert WORKFLOW["permissions"] == {"contents": "read"}
    jobs = WORKFLOW["jobs"]
    # Jobs that install and run upstream releases inherit the read-only token.
    for name in ("plan", "canary"):
        assert "permissions" not in jobs[name]
    checkout = next(
        step for step in jobs["canary"]["steps"] if "checkout" in step.get("uses", "")
    )
    assert checkout["with"]["persist-credentials"] is False
    assert jobs["propose"]["permissions"] == {
        "contents": "write",
        "pull-requests": "write",
    }
    assert jobs["report"]["permissions"] == {"issues": "write"}
    # The writing job resolves the lock but must never run a build backend.
    for step in jobs["propose"]["steps"]:
        if "uv lock" in step.get("run", ""):
            assert step["env"]["UV_NO_BUILD"] == "1"
    assert set(jobs) == {"plan", "canary", "propose", "report"}
