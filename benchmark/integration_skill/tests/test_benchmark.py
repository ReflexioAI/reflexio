import copy
import json
import os
import sys

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
@pytest.mark.parametrize("defect", ["kind", "id", "discarded", "shared", "duplicate"])
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
