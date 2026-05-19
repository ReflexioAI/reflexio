"""Tests for openclaw_smart.query_compose."""

from __future__ import annotations

from openclaw_smart import query_compose


def test_edit_tool_query_uses_basename_and_snippet():
    q = query_compose.from_tool_call(
        "Edit", {"file_path": "src/auth.py", "new_string": "import oauth2"}
    )
    assert "auth.py" in q
    assert "import oauth2" in q
    # Full directory path is dropped — only basename survives.
    assert "src/" not in q


def test_write_tool_uses_content_fallback():
    q = query_compose.from_tool_call(
        "Write", {"file_path": "notes.md", "content": "release plan"}
    )
    assert "notes.md" in q
    assert "release plan" in q


def test_bash_tool_query_uses_first_line():
    q = query_compose.from_tool_call(
        "Bash", {"command": "pytest tests/\nexit 0"}
    )
    assert "pytest tests/" in q
    assert "exit 0" not in q


def test_empty_input_returns_empty_for_known_tool():
    assert query_compose.from_tool_call("Edit", {}) == ""
    assert query_compose.from_tool_call("Bash", {}) == ""


def test_unknown_tool_returns_empty():
    assert query_compose.from_tool_call("UnknownTool", {"foo": "bar"}) == ""


def test_long_snippet_is_truncated():
    long_text = "a" * 1000
    q = query_compose.from_tool_call(
        "Edit", {"file_path": "x.py", "new_string": long_text}
    )
    # Basename + space + snippet truncated to 400 chars
    assert "a" * 400 in q
    assert len(q) <= len("x.py ") + 400
