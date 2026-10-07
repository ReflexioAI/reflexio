"""Run pinned skill variants in disposable applications and grade their edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
VERSIONS = {
    "original": "7ce76a843eef201f53029a027cba21a914a3ed29",
    "updated": "91956d6c59865746a1591062dfa0ff2202285b90",
}
PROMPT = """Prepare this existing Reflexio integration for managed-cloud playbook tuning.
Use the supplied integrate-reflexio skill, implement the required changes, test them,
and report what you verified. Work offline using API.md and the preinstalled SDK;
live verification is not authorized. Keep the application entrypoint and existing
behavior. Do not change protected support/dependency files or existing tests.
"""
PROTECTED = ("support.py", "pyproject.toml", "test_app.py", "AGENTS.md", "API.md")


def run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    stdout: Path,
    stderr: Path,
    prompt: str | None = None,
) -> dict:
    started = time.monotonic()
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if prompt else subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.communicate(prompt.encode() if prompt else None, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def parse_events(path: Path) -> dict:
    usage = None
    completed = False
    errors = []
    test_commands = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            completed = True
            usage = event.get("usage")
        if event.get("type") in ("error", "turn.failed"):
            errors.append(event.get("message") or event.get("error"))
        item = event.get("item", {})
        if (
            event.get("type") == "item.completed"
            and item.get("type") == "command_execution"
            and "pytest" in item.get("command", "")
        ):
            test_commands.append(
                {"command": item["command"], "exit_code": item.get("exit_code")}
            )
    return {
        "completed": completed,
        "usage": usage,
        "errors": errors,
        "observed_test_commands": test_commands,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(workspace: Path, kind: str, sha: str, python: Path) -> dict[str, str]:
    shutil.copytree(ROOT / "fixtures" / kind, workspace)
    for name in ("support.py", "API.md"):
        shutil.copy2(ROOT / "fixtures" / name, workspace / name)
    (workspace / "AGENTS.md").write_text(
        "This is the customer's application. Read API.md and the integration skill.\n"
        "Preserve support.py, pyproject.toml, existing tests, and this file.\n"
        "Use .venv/bin/python; dependencies are installed. No live requests.\n"
    )
    required = "3.12" if kind == "python_app" else "3.11"
    (workspace / "pyproject.toml").write_text(
        f'[project]\nname="skill-benchmark-app"\nversion="0.0.0"\n'
        f'requires-python=">={required}"\ndependencies=["reflexio-client==0.2.16"]\n'
    )
    shutil.copytree(python.parent.parent, workspace / ".venv", symlinks=True)
    skill = workspace / ".agents" / "skills" / "integrate-reflexio"
    files = subprocess.check_output(  # noqa: S603
        ["git", "ls-tree", "-r", "--name-only", sha, "skills/integrate-reflexio"],
        cwd=REPOSITORY,
        text=True,
    ).splitlines()
    if not files:
        raise ValueError(f"no skill files at {sha}")
    for file in files:
        target = skill / Path(file).relative_to("skills/integrate-reflexio")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            subprocess.check_output(  # noqa: S603
                ["git", "show", f"{sha}:{file}"], cwd=REPOSITORY
            )
        )
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)  # noqa: S603
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)  # noqa: S603
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark",
            "-c",
            "user.email=benchmark@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=workspace,
        check=True,
    )  # noqa: S603
    names = [
        *PROTECTED,
        *(
            str(file.relative_to(workspace))
            for file in skill.rglob("*")
            if file.is_file()
        ),
    ]
    return {name: digest(workspace / name) for name in names}


def write_config(
    home: Path, workspace: Path, python: Path, model: str, *, grading: bool = False
) -> None:
    q = json.dumps
    codex_root = Path(shutil.which("codex") or "codex").resolve().parent.parent
    protected = [
        *(workspace / name for name in PROTECTED),
        workspace / ".agents",
        workspace / ".git",
        workspace / ".venv",
    ]
    access = "read" if grading else "write"
    grader_rule = f'{q(str(ROOT))}="read"' if grading else ""
    filesystem = "\n".join(f'{q(str(path))}="read"' for path in protected)
    (home / "config.toml").write_text(f"""
model={q(model)}
model_reasoning_effort="medium"
approval_policy="never"
web_search="disabled"
default_permissions="benchmark"
[shell_environment_policy]
inherit="none"
[shell_environment_policy.set]
PATH={q(str(python.parent) + ":/usr/local/bin:/usr/bin:/bin")}
PYTHON_DOTENV_DISABLED="1"
[permissions.benchmark.filesystem]
":minimal"="read"
{q(str(python.parent.parent))}="read"
{q(str(python.resolve().parent.parent.parent))}="read"
{q(str(codex_root))}="read"
{q(str(workspace))}={q(access)}
{grader_rule}
{q(str(home))}="deny"
{filesystem}
[permissions.benchmark.network]
enabled=false
""")


def sandbox_probe(
    codex: str, home: Path, workspace: Path, env: dict, python: Path
) -> None:
    code = """import pathlib, socket
p = pathlib.Path("sandbox-write-probe")
p.write_text("ok")
p.unlink()
for name in ("support.py", "test_app.py"):
    try:
        with open(name, "a") as f: f.write("BAD")
    except OSError as exc:
        if exc.errno not in (1, 13, 30): raise
    else: raise AssertionError("protected file writable")
try:
    socket.socket()
except PermissionError: pass
else: raise AssertionError("network socket allowed")
print("sandbox verified")
"""
    result = subprocess.run(  # noqa: S603
        [
            codex,
            "sandbox",
            "-P",
            "benchmark",
            "-C",
            str(workspace),
            str(python),
            "-c",
            code,
        ],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode or "sandbox verified" not in result.stdout:
        raise RuntimeError(f"sandbox preflight failed: {result.stderr}")
    assert (home / "config.toml").exists()


def evaluate(
    output: Path,
    kind: str,
    version: str,
    repetition: int,
    python: Path,
    codex: str,
    model: str,
    timeout: int,
) -> dict:
    directory = output / f"{kind}-{version}-{repetition}"
    directory.mkdir()
    workspace = directory / "application"
    hashes = prepare(workspace, kind, VERSIONS[version], python)
    env = {
        "PATH": str(python.parent) + ":/usr/local/bin:/usr/bin:/bin",
        "PYTHON_DOTENV_DISABLED": "1",
    }
    auth_cache = Path.home() / ".cache" / "reflexio-skill-benchmark"
    auth_cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auth-", dir=auth_cache) as temp:
        home = Path(temp)
        home.chmod(0o700)
        auth = (
            Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
            / "auth.json"
        )
        shutil.copy2(auth, home / "auth.json")
        (home / "auth.json").chmod(0o600)
        write_config(home, workspace, python, model)
        env["CODEX_HOME"] = str(home)
        sandbox_probe(codex, home, workspace, env, python)
        command = [
            codex,
            "exec",
            "--ephemeral",
            "--json",
            "--ignore-rules",
            "-C",
            str(workspace),
            "-o",
            str(workspace / "agent-report.md"),
            "-",
        ]
        execution = run_process(
            command,
            cwd=workspace,
            env=env,
            timeout=timeout,
            stdout=directory / "events.jsonl",
            stderr=directory / "stderr.log",
            prompt=PROMPT,
        )
    env.pop("CODEX_HOME")
    events = parse_events(directory / "events.jsonl")
    violations = [
        name
        for name, before in hashes.items()
        if not (workspace / name).is_file() or digest(workspace / name) != before
    ]
    with tempfile.TemporaryDirectory(prefix="grader-", dir=auth_cache) as temp:
        home = Path(temp)
        write_config(home, workspace, python, model, grading=True)
        env["CODEX_HOME"] = str(home)
        grading = run_process(
            [
                codex,
                "sandbox",
                "-P",
                "benchmark",
                "-C",
                str(workspace),
                str(python),
                str(ROOT / "grader.py"),
                str(workspace),
                kind,
            ],
            cwd=workspace,
            env=env,
            timeout=45,
            stdout=directory / "grade.json",
            stderr=directory / "grade-stderr.log",
        )
    env.pop("CODEX_HOME")
    try:
        grade = json.loads((directory / "grade.json").read_text())
    except json.JSONDecodeError:
        grade = {"passed": False, "error": "grader did not produce JSON"}
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=workspace, text=True)  # noqa: S603
    (directory / "changes.diff").write_text(diff)
    changed = subprocess.check_output(  # noqa: S603
        ["git", "status", "--porcelain"], cwd=workspace, text=True
    ).splitlines()
    status = (
        "pass"
        if (
            execution["exit_code"] == 0
            and events["completed"]
            and grade["passed"]
            and grading["exit_code"] == 0
            and not violations
        )
        else "fail"
    )
    if not events["completed"] and not execution["timed_out"]:
        status = "infrastructure_error"
    result = {
        "fixture": kind,
        "version": version,
        "skill_sha": VERSIONS[version],
        "repetition": repetition,
        "model": model,
        "status": status,
        **execution,
        **events,
        "grade": grade,
        "protected_violations": violations,
        "changed_files": changed,
        "artifacts": str(directory),
        "verification_claim_review": "pending manual transcript review",
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    return result


def summarize(results: list[dict]) -> dict:
    summary = {}
    for version in VERSIONS:
        rows = [row for row in results if row["version"] == version]
        summary[version] = {
            "counts": dict(Counter(row["status"] for row in rows)),
            "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 3),
            "reported_input_tokens": sum(
                (row["usage"] or {}).get("input_tokens", 0) for row in rows
            ),
            "reported_output_tokens": sum(
                (row["usage"] or {}).get("output_tokens", 0) for row in rows
            ),
            "usage_missing_runs": sum(row["usage"] is None for row in rows),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        required=True,
        help="Python in a preinstalled lightweight SDK + pytest venv",
    )
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.repetitions < 1 or args.timeout < 1:
        parser.error("repetitions and timeout must be positive")
    codex = shutil.which("codex")
    if not codex:
        parser.error("codex is required")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "versions": VERSIONS,
        "model": args.model,
        "reasoning": "medium",
        "prompt": PROMPT,
        "timeout": args.timeout,
        "codex_version": subprocess.check_output(  # noqa: S603
            [codex, "--version"], text=True
        ).strip(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = []
    for kind in ("python_app", "http_app"):
        for repetition in range(1, args.repetitions + 1):
            order = (
                ("original", "updated") if repetition % 2 else ("updated", "original")
            )
            for version in order:
                print(f"Starting {kind} {version} repetition {repetition}", flush=True)
                result = evaluate(
                    args.output,
                    kind,
                    version,
                    repetition,
                    args.python.absolute(),
                    codex,
                    args.model,
                    args.timeout,
                )
                results.append(result)
                report = {"summary": summarize(results), "runs": results}
                (args.output / "results.json").write_text(json.dumps(report, indent=2))
                print(
                    f"Completed: {result['status']} ({result['elapsed_seconds']}s)",
                    flush=True,
                )
    lines = [
        "# Integration skill benchmark",
        "",
        "| Version | Results | Seconds |",
        "| --- | --- | --- |",
    ]
    for version, item in summarize(results).items():
        lines.append(f"| {version} | {item['counts']} | {item['elapsed_seconds']} |")
    lines.extend(
        [
            "",
            "See results.json for checks, reported usage, and per-run artifacts.",
            "Small local sample; no live cloud or tuning verification.",
            "Verification claims require manual transcript review.",
        ]
    )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
