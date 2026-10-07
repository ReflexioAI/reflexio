import contextlib
import copy
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from benchmark.integration_skill.grader import check_record, grade
from benchmark.integration_skill.runner import (
    ROOT,
    parse_events,
    run_process,
    summarize,
)


def repaired_source(kind):
    source = (ROOT / "fixtures" / kind / "app.py").read_text()
    if kind == "python_app":
        construction = """    references = []
    if context:
        for kind, key, identity in (
            ("profile", "profiles", "profile_id"),
            ("user_playbook", "user_playbooks", "user_playbook_id"),
            ("agent_playbook", "agent_playbooks", "agent_playbook_id"),
        ):
            references.extend(
                {"kind": kind, "learning_id": str(getattr(item, identity))}
                for item in getattr(results, key)
                if not item.content.startswith("[discard]")
            )
"""
        source = source.replace(
            'InteractionData(role="Agent", content=response)',
            'InteractionData(role="Agent", content=response, retrieved_learnings=references)',
        )
    else:
        construction = """    references = []
    if context:
        for kind, key, identity in (
            ("profile", "profiles", "profile_id"),
            ("user_playbook", "user_playbooks", "user_playbook_id"),
            ("agent_playbook", "agent_playbooks", "agent_playbook_id"),
        ):
            references.extend(
                {"kind": kind, "learning_id": str(item[identity])}
                for item in results[key]
                if not item["content"].startswith("[discard]")
            )
"""
        source = source.replace(
            '{"role": "Agent", "content": response}',
            '{"role": "Agent", "content": response, "retrieved_learnings": references}',
        )
    return source.replace(
        "    response = model.generate", construction + "    response = model.generate"
    )


@pytest.mark.parametrize("kind", ["python_app", "http_app"])
def test_broken_fixtures_fail_and_real_repairs_pass(tmp_path, kind):
    app = tmp_path / "app.py"
    app.write_text((ROOT / "fixtures" / kind / "app.py").read_text())
    broken = grade(tmp_path, kind)
    assert not broken["passed"]
    assert "references:" in broken["checks"]["populated"]["error"]
    app.write_text(repaired_source(kind))
    assert grade(tmp_path, kind)["passed"]


@pytest.mark.parametrize("kind", ["python_app", "http_app"])
@pytest.mark.parametrize(
    "defect",
    ["kind", "id", "discarded", "shared", "duplicate", "foreign_context", "identities"],
)
def test_grader_rejects_repaired_application_regressions(tmp_path, kind, defect):
    source = repaired_source(kind)
    if defect == "kind":
        source = source.replace(
            '("user_playbook", "user_playbooks",',
            '("agent_playbook", "user_playbooks",',
        )
    elif defect == "id":
        source = source.replace('"learning_id": str(', '"learning_id": "wrong-" + str(')
    elif defect == "discarded":
        source = source.replace(
            'if not item.content.startswith("[discard]")', "if True"
        )
        source = source.replace(
            'if not item["content"].startswith("[discard]")', "if True"
        )
    elif defect == "shared":
        source = "shared = []\n" + source
        source = source.replace("    references = []", "    references = shared")
    elif defect == "foreign_context":
        source = source.replace(
            '"\\n".join(context)', '"\\n".join(context) + " user-content-999"'
        )
    elif defect == "identities":
        anchor = "    session =" if kind == "python_app" else "    identities ="
        source = source.replace(anchor, '    user_id = "wrong-owner"\n' + anchor)
    elif kind == "python_app":
        source = source.replace(
            "    result = session.publish_interaction(",
            "    session.publish_interaction(user_id=user_id, source=source, agent_version=agent_version, "
            'interactions=[InteractionData(role="User", content=user_message), '
            'InteractionData(role="Agent", content=response, retrieved_learnings=references)])\n'
            "    result = session.publish_interaction(",
        )
    else:
        source = source.replace(
            "    result = client.post(",
            '    client.post("/api/publish_interaction", json={**identities, "interaction_data_list": '
            '[{"role":"User", "content":user_message}, {"role":"Agent", "content":response, '
            '"retrieved_learnings":references}]}, timeout=5)\n    result = client.post(',
        )
    (tmp_path / "app.py").write_text(source)
    assert not grade(tmp_path, kind)["passed"]


def valid_record():
    identities = {
        "user_id": "u",
        "session_id": "s",
        "source": "support",
        "agent_version": "support@1",
    }
    return {
        "seed": 117,
        "state": "populated",
        "identities": identities.copy(),
        "events": ["search", "model", "publish"],
        "search": [identities],
        "model": [
            ("message-117", "profile-content-117 user-content-117 agent-content-117")
        ],
        "publish": [
            {
                **identities,
                "interaction_data_list": [
                    {"role": "User", "content": "message-117"},
                    {
                        "role": "Agent",
                        "content": "response:message-117",
                        "retrieved_learnings": [
                            {"kind": "profile", "learning_id": "profile-117"},
                            {"kind": "user_playbook", "learning_id": "117"},
                            {"kind": "agent_playbook", "learning_id": "117"},
                        ],
                    },
                ],
            }
        ],
    }


def test_numeric_ids_and_identity_mismatch_are_rejected():
    record = valid_record()
    check_record(record)
    bad = copy.deepcopy(record)
    bad["publish"][0]["interaction_data_list"][1]["retrieved_learnings"][1][
        "learning_id"
    ] = 117
    with pytest.raises((AssertionError, ValueError)):
        check_record(bad)
    bad = copy.deepcopy(record)
    bad["publish"][0]["session_id"] = "different"
    with pytest.raises(AssertionError, match="identity mismatch"):
        check_record(bad)


def test_event_parser_tracks_completed_usage_and_test_exit_status(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "python -m pytest",
                "exit_code": 1,
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 120, "output_tokens": 15}},
    ]
    path.write_text("diagnostic\n" + "\n".join(json.dumps(e) for e in events))
    parsed = parse_events(path)
    assert parsed["completed"] and parsed["usage"]["input_tokens"] == 120
    assert parsed["observed_test_commands"][0]["exit_code"] == 1
    path.write_text('{"type":"turn.failed","error":"auth unavailable"}\n')
    assert parse_events(path)["usage"] is None
    assert parse_events(path)["errors"] == ["auth unavailable"]


def test_timeout_terminates_agent_process(tmp_path):
    result = run_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        timeout=1,
        stdout=tmp_path / "stdout",
        stderr=tmp_path / "stderr",
    )
    assert result["timed_out"] and result["exit_code"] != 0
    assert result["elapsed_seconds"] < 10


def test_summary_preserves_infrastructure_errors_and_missing_usage():
    rows = [
        {
            "version": "original",
            "status": "pass",
            "elapsed_seconds": 2.0,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        },
        {
            "version": "updated",
            "status": "infrastructure_error",
            "elapsed_seconds": 1.0,
            "usage": None,
        },
    ]
    result = summarize(rows)
    assert result["original"]["counts"] == {"pass": 1}
    assert result["updated"]["counts"] == {"infrastructure_error": 1}
    assert result["updated"]["usage_missing_runs"] == 1


def test_grader_rejects_foreign_context_and_mutually_wrong_identities():
    record = valid_record()
    bad = copy.deepcopy(record)
    message, context = bad["model"][0]
    bad["model"][0] = (message, context + " user-content-218")
    with pytest.raises(AssertionError, match="foreign learning"):
        check_record(bad)
    for key in ("user_id", "session_id", "source", "agent_version"):
        bad = copy.deepcopy(record)
        bad["search"][0][key] = "wrong"
        bad["publish"][0][key] = "wrong"
        with pytest.raises(AssertionError, match="identity mismatch"):
            check_record(bad)


def test_timeout_kills_term_ignoring_descendant(tmp_path):
    pid_file = tmp_path / "child.pid"
    child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    parent = (
        "import subprocess,sys,time,pathlib; "
        f"p=subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    result = run_process(
        [sys.executable, "-c", parent],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        timeout=1,
        stdout=tmp_path / "stdout",
        stderr=tmp_path / "stderr",
    )
    assert result["timed_out"]
    pid = int(pid_file.read_text())
    try:
        for _ in range(50):
            status = Path(f"/proc/{pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("descendant survived process-group cleanup")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize("failure", ["auth", "preflight", "grader_exit", "grader_json"])
def test_evaluate_persists_infrastructure_failures(tmp_path, monkeypatch, failure):
    from benchmark.integration_skill import runner

    def prepare(workspace, *args):
        workspace.mkdir()
        return {}

    def copy_auth(source, destination):
        if failure == "auth":
            raise OSError("synthetic missing authentication")
        Path(destination).write_text("{}")

    def probe(*args):
        if failure == "preflight":
            raise RuntimeError("synthetic sandbox unavailable")

    def process(command, *, stdout, stderr, **kwargs):
        stderr.write_text("")
        if stdout.name == "events.jsonl":
            stdout.write_text('{"type":"turn.completed","usage":{"input_tokens":1}}\n')
            code = 0
        else:
            stdout.write_text(
                "invalid" if failure == "grader_json" else '{"passed":true}'
            )
            code = 1 if failure == "grader_exit" else 0
        return {"exit_code": code, "timed_out": False, "elapsed_seconds": 0.1}

    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner.shutil, "copy2", copy_auth)
    monkeypatch.setattr(runner, "sandbox_probe", probe)
    monkeypatch.setattr(runner, "run_process", process)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "")
    result = runner.evaluate(
        tmp_path,
        "python_app",
        "original",
        1,
        Path(sys.executable),
        "/bin/true",
        "fixture-model",
        5,
    )
    assert result["status"] == "infrastructure_error"
    stored = json.loads((tmp_path / "python_app-original-1/result.json").read_text())
    assert stored["status"] == result["status"]
    assert summarize([result])["original"]["counts"] == {"infrastructure_error": 1}


def test_provenance_records_sources_and_rejects_wrong_sdk(monkeypatch):
    from benchmark.integration_skill import runner

    identity = {
        "python_version": "3.12",
        "sdk_version": "0.2.16",
        "sdk_sha256": "fixture-hash",
    }
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *args, **kwargs: json.dumps(identity)
    )
    result = runner.provenance(Path(sys.executable))
    assert result["sdk_sha256"] == "fixture-hash"
    assert "grader.py" in result["source_sha256"]
    assert "fixtures/python_app/app.py" in result["source_sha256"]
    assert result["source_sha256"]["grader.py"] == runner.digest(
        runner.ROOT / "grader.py"
    )
    identity["sdk_version"] = "0.0.0"
    with pytest.raises(ValueError, match="pinned"):
        runner.provenance(Path(sys.executable))


def test_rescore_runs_saved_application_and_preserves_initial_grade(
    tmp_path, monkeypatch
):
    from benchmark.integration_skill import runner

    directory = tmp_path / "python_app-original-1"
    workspace = directory / "application"
    workspace.mkdir(parents=True)
    (workspace / "app.py").write_text(repaired_source("python_app"))
    result = {
        "fixture": "python_app",
        "version": "original",
        "repetition": 1,
        "status": "fail",
        "grade": {"passed": False},
        "exit_code": 0,
        "completed": True,
        "protected_violations": [],
        "usage": None,
        "elapsed_seconds": 1.0,
    }
    (directory / "result.json").write_text(json.dumps(result))
    monkeypatch.setattr(
        runner,
        "provenance",
        lambda python: {"source_sha256": {"grader.py": "verified-grader"}},
    )

    def process(command, *, stdout, stderr, **kwargs):
        stdout.write_text(json.dumps(grade(workspace, "python_app")))
        stderr.write_text("")
        return {"exit_code": 0, "timed_out": False, "elapsed_seconds": 0.1}

    monkeypatch.setattr(runner, "run_process", process)
    report = runner.rescore(
        tmp_path, Path(sys.executable), "/bin/true", "fixture-model"
    )
    assert report["runs"][0]["status"] == "pass"
    assert report["runs"][0]["initial_grade"] == {"passed": False}
    assert report["runs"][0]["final_grader_sha256"] == "verified-grader"
    assert "1/1 passed" in (tmp_path / "report.md").read_text()
