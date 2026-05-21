"""Tests for openclaw_smart.cli."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

import pytest
from openclaw_smart import cli


@pytest.fixture(autouse=True)
def isolate_state_dir(monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(sessions))
    return sessions


def test_cmd_show_fetches_all_entities(capsys):
    with patch("openclaw_smart.cli.Adapter") as Ad, patch(
        "openclaw_smart.cli.ids.resolve_project_id", return_value="proj-x"
    ):
        Ad.return_value.fetch_all.return_value = ([], [], [])
        rc = cli.cmd_show(Namespace(project=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert "proj-x" in out


def test_cmd_show_renders_markdown(capsys):
    with patch("openclaw_smart.cli.Adapter") as Ad, patch(
        "openclaw_smart.cli.ids.resolve_project_id", return_value="proj-x"
    ):
        Ad.return_value.fetch_all.return_value = (
            [{"user_playbook_id": "abc", "content": "Test rule"}],
            [],
            [],
        )
        cli.cmd_show(Namespace(project=None))
    out = capsys.readouterr().out
    assert "Test rule" in out
    assert "[oc:" in out


def test_cmd_show_honors_project_override(capsys):
    with patch("openclaw_smart.cli.Adapter") as Ad, patch(
        "openclaw_smart.cli.ids.resolve_project_id", return_value="ignored"
    ):
        Ad.return_value.fetch_all.return_value = ([], [], [])
        cli.cmd_show(Namespace(project="explicit-proj"))
    Ad.return_value.fetch_all.assert_called_once()
    kwargs = Ad.return_value.fetch_all.call_args[1]
    assert kwargs["project_id"] == "explicit-proj"


def test_cmd_learn_no_session_returns_zero(isolate_state_dir):
    rc = cli.cmd_learn(Namespace(session=None, project=None, note=None))
    assert rc == 0


def test_cmd_learn_force_extracts(isolate_state_dir):
    # Seed a session JSONL so _latest_session_id picks it up.
    from openclaw_smart import state

    state.append("sess-1", {"role": "User", "content": "x"})
    with patch("openclaw_smart.cli.publish") as pub, patch(
        "openclaw_smart.cli.ids.resolve_project_id", return_value="proj-x"
    ):
        pub.publish_unpublished.return_value = ("ok", 1)
        rc = cli.cmd_learn(Namespace(session=None, project=None, note=None))
    assert rc == 0
    kwargs = pub.publish_unpublished.call_args[1]
    assert kwargs["force_extraction"] is True
    assert kwargs["session_id"] == "sess-1"


def test_cmd_learn_appends_note(isolate_state_dir):
    from openclaw_smart import state

    state.append("sess-2", {"role": "User", "content": "earlier"})
    with patch("openclaw_smart.cli.publish") as pub, patch(
        "openclaw_smart.cli.ids.resolve_project_id", return_value="proj-x"
    ):
        pub.publish_unpublished.return_value = ("ok", 2)
        cli.cmd_learn(Namespace(session="sess-2", project=None, note="key insight"))
    records = state.read_all("sess-2")
    contents = [r.get("content") for r in records]
    assert "key insight" in contents


def test_cmd_learn_handles_unreachable_backend(isolate_state_dir, capsys):
    from openclaw_smart import state

    state.append("sess-3", {"role": "User", "content": "x"})
    with patch("openclaw_smart.cli.publish") as pub, patch(
        "openclaw_smart.cli.ids.resolve_project_id", return_value="proj-x"
    ):
        pub.publish_unpublished.return_value = ("failed", 0)
        rc = cli.cmd_learn(Namespace(session=None, project=None, note=None))
    assert rc == 1
    assert "Failed to reach reflexio" in capsys.readouterr().out


def test_cmd_clear_all_requires_yes(capsys):
    with patch(
        "openclaw_smart.cli._resolve_clear_all_targets", return_value=[]
    ):
        rc = cli.cmd_clear_all(Namespace(yes=False))
    assert rc != 0
    assert "--yes" in capsys.readouterr().out


def test_cmd_clear_all_with_yes_proceeds(monkeypatch, tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "old.jsonl").write_text('{"role":"User"}\n')
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(sessions))

    with patch(
        "openclaw_smart.cli._resolve_clear_all_targets", return_value=[]
    ), patch("openclaw_smart.cli._service_status", return_value="not running"), patch(
        "openclaw_smart.cli._run_service", return_value=0
    ):
        rc = cli.cmd_clear_all(Namespace(yes=True))
    assert rc == 0
    assert not (sessions / "old.jsonl").exists()


def test_cmd_clear_all_restarts_backend_after_delete_failure(tmp_path):
    target = cli._ClearAllTarget(
        path=tmp_path / "reflexio-openclaw-test",
        kind="dir",
        label="test target",
    )
    service_calls: list[str] = []

    def fake_run_service(_script, command) -> int:  # noqa: ANN001
        service_calls.append(command)
        return 0

    with patch(
        "openclaw_smart.cli._resolve_clear_all_targets", return_value=[target]
    ), patch("openclaw_smart.cli._service_status", return_value="running on 8071"), patch(
        "openclaw_smart.cli._remove_clear_all_target",
        side_effect=cli._ClearAllError("boom"),
    ), patch(
        "openclaw_smart.cli._run_service", side_effect=fake_run_service
    ):
        rc = cli.cmd_clear_all(Namespace(yes=True))

    assert rc == 1
    assert service_calls == ["stop", "start"]


def test_build_parser_accepts_show():
    parser = cli._build_parser()
    args = parser.parse_args(["show"])
    assert args.command == "show"
    assert args.project is None


def test_build_parser_accepts_learn_with_note():
    parser = cli._build_parser()
    args = parser.parse_args(["learn", "--note", "hello"])
    assert args.command == "learn"
    assert args.note == "hello"


def test_build_parser_accepts_clear_all_yes():
    parser = cli._build_parser()
    args = parser.parse_args(["clear-all", "--yes"])
    assert args.yes is True


def test_build_parser_accepts_restart_flags():
    parser = cli._build_parser()
    args = parser.parse_args(["restart", "--skip-backend", "--no-rebuild"])
    assert args.skip_backend is True
    assert args.no_rebuild is True
