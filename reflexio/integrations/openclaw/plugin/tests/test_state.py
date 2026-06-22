"""Tests for openclaw_smart.state."""

from __future__ import annotations

import json

import pytest
from openclaw_smart import state


@pytest.fixture(autouse=True)
def isolate_state_dir(monkeypatch, tmp_path):
    """Each test gets its own state dir via OPENCLAW_SMART_STATE_DIR."""
    sessions = tmp_path / "sessions"
    monkeypatch.setenv("OPENCLAW_SMART_STATE_DIR", str(sessions))
    return sessions


def test_state_dir_honours_env(isolate_state_dir):
    assert state.state_dir() == isolate_state_dir


def test_append_creates_file(isolate_state_dir):
    state.append("sess1", {"role": "User", "content": "hi"})
    path = isolate_state_dir / "sess1.jsonl"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["content"] == "hi"


def test_read_all_returns_records():
    state.append("sess2", {"role": "User", "content": "a"})
    state.append("sess2", {"role": "Assistant", "content": "b"})
    records = state.read_all("sess2")
    assert len(records) == 2
    assert records[0]["content"] == "a"
    assert records[1]["content"] == "b"


def test_read_all_missing_file_returns_empty():
    assert state.read_all("never-existed") == []


def test_read_all_skips_malformed_lines(isolate_state_dir):
    isolate_state_dir.mkdir(parents=True, exist_ok=True)
    path = isolate_state_dir / "broken.jsonl"
    path.write_text(
        '{"role": "User", "content": "good"}\nnot json\n{"role": "Assistant", "content": "also good"}\n'
    )
    records = state.read_all("broken")
    assert len(records) == 2
    assert records[0]["content"] == "good"
    assert records[1]["content"] == "also good"


def test_unpublished_slice_returns_watermark_and_turns():
    state.append("s3", {"role": "User", "content": "1"})
    state.append("s3", {"role": "Assistant", "content": "2"})
    state.append("s3", {"published_up_to": 2})
    state.append("s3", {"role": "User", "content": "3"})
    state.append("s3", {"role": "Assistant", "content": "4"})
    records = state.read_all("s3")
    watermark, turns = state.unpublished_slice(records)
    assert watermark == 2
    contents = [t["content"] for t in turns]
    assert contents == ["3", "4"]


def test_unpublished_slice_folds_tool_records():
    records = [
        {"role": "User", "content": "run ls"},
        {
            "role": "Assistant_tool",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_output": "a.txt",
            "status": "success",
        },
        {"role": "Assistant", "content": "Done."},
    ]
    _, turns = state.unpublished_slice(records)
    assert len(turns) == 2
    assert turns[0]["role"] == "User"
    assistant_turn = turns[1]
    assert assistant_turn["role"] == "Assistant"
    assert assistant_turn["tools_used"] == [
        {
            "tool_name": "Bash",
            "status": "success",
            "tool_data": {
                "input": {"command": "ls"},
                "output": "a.txt",
            },
        }
    ]


def test_unpublished_slice_truncates_long_tool_fields():
    long_value = "x" * 500
    records = [
        {
            "role": "Assistant_tool",
            "tool_name": "Edit",
            "tool_input": {"new_string": long_value},
            "tool_output": "",
            "status": "success",
        },
        {"role": "Assistant", "content": "ok"},
    ]
    _, turns = state.unpublished_slice(records)
    truncated = turns[0]["tools_used"][0]["tool_data"]["input"]["new_string"]
    assert len(truncated) == 256
    assert truncated == "x" * 256


def test_unpublished_slice_ignores_malformed_watermark_and_tool_input():
    records = [
        {"published_up_to": "not-an-int"},
        {
            "role": "Assistant_tool",
            "tool_name": "Bash",
            "tool_input": ["not", "a", "mapping"],
            "tool_output": "ok",
        },
        {"role": "Assistant", "content": "Done."},
    ]

    watermark, turns = state.unpublished_slice(records)

    assert watermark == 0
    assert turns == [
        {
            "role": "Assistant",
            "content": "Done.",
            "tools_used": [
                {
                    "tool_name": "Bash",
                    "status": "success",
                    "tool_data": {"output": "ok"},
                }
            ],
        }
    ]


def test_unpublished_slice_ignores_out_of_range_watermark():
    # A marker larger than the records seen before it (e.g. a corrupt/tampered
    # buffer) must be ignored, not applied — otherwise the ``idx < published``
    # gate would skip every later turn and silently drop unpublished records.
    records = [
        {"published_up_to": 99},
        {"role": "Assistant", "content": "Hello."},
    ]

    watermark, turns = state.unpublished_slice(records)

    assert watermark == 0
    assert turns == [{"role": "Assistant", "content": "Hello."}]


def test_append_injected_writes_registry():
    state.append_injected(
        "s4",
        [
            {"id": "s1-abcd", "kind": "skill", "title": "Test skill", "real_id": "abc"},
            {
                "id": "p1-efgh",
                "kind": "preference",
                "title": "Test pref",
                "real_id": "def",
            },
        ],
    )
    registry = state.read_injected("s4")
    assert set(registry) == {"s1-abcd", "p1-efgh"}
    assert registry["s1-abcd"]["title"] == "Test skill"


def test_append_injected_noop_when_empty():
    state.append_injected("s5", [])
    assert state.read_injected("s5") == {}


def test_read_injected_later_wins():
    state.append_injected("s6", [{"id": "s1-abcd", "title": "old"}])
    state.append_injected("s6", [{"id": "s1-abcd", "title": "new"}])
    registry = state.read_injected("s6")
    assert registry["s1-abcd"]["title"] == "new"


def test_path_traversal_session_id_rejected(isolate_state_dir, tmp_path):
    """A crafted session id with path separators must not escape state_dir()."""
    escape_attempts = [
        "../escape",
        "../../etc/passwd",
        "a/b/c",
        "sub/sess",
        "..",
        "",
        "a" * 200,  # over 128 char cap
    ]
    for sid in escape_attempts:
        assert state.session_path(sid) is None, f"unsafe session_path accepted: {sid!r}"
        assert state.injected_path(sid) is None
        # Append + read must silently no-op rather than writing anywhere.
        state.append(sid, {"role": "User", "content": "x"})
        state.append_injected(sid, [{"id": "s1-abcd", "title": "x"}])
        assert state.read_all(sid) == []
        assert state.read_injected(sid) == {}

    # The state dir itself should remain empty — nothing escaped.
    if isolate_state_dir.exists():
        assert not any(isolate_state_dir.glob("**/*.jsonl"))
    # And nowhere outside it either.
    assert not (tmp_path / "escape.jsonl").exists()


def test_safe_session_ids_accepted():
    """The safe-id charset (alphanumeric + dot/underscore/hyphen/colon) is allowed."""
    # Colons appear in real openClaw sessionKeys (``agent:main:main``); they're
    # not POSIX path separators so they're safe inside filenames.
    for sid in (
        "sess1",
        "a.b.c",
        "a_b",
        "a-b",
        "01234",
        "S.1_2-3",
        "agent:main:main",
        "agent:work:abc123",
    ):
        path = state.session_path(sid)
        assert path is not None, f"safe id {sid!r} should have been accepted"
        assert path.name == f"{sid}.jsonl"
