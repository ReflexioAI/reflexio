"""Tests for interaction helper functions (_parse_json_payload, _interactions_from_payload)."""

from __future__ import annotations

import pytest

from reflexio.cli.commands.interactions import (
    _interactions_from_payload,
    _parse_json_payload,
)
from reflexio.cli.errors import CliError


class TestParseJsonPayload:
    """Tests for _parse_json_payload()."""

    def test_parse_json_object(self) -> None:
        """Single JSON object wraps in a one-element list."""
        result = _parse_json_payload(
            '{"interactions": [{"role": "user", "content": "hi"}]}'
        )
        assert len(result) == 1
        assert result[0]["interactions"][0]["role"] == "user"

    def test_parse_json_array(self) -> None:
        """JSON array of objects returns the list directly."""
        raw = '[{"interactions": [{"role": "user", "content": "a"}]}, {"interactions": [{"role": "user", "content": "b"}]}]'
        result = _parse_json_payload(raw)
        assert len(result) == 2

    def test_parse_json_array_non_objects(self) -> None:
        """JSON array of non-objects should raise CliError."""
        with pytest.raises(CliError, match="array elements must be objects"):
            _parse_json_payload("[1, 2]")

    def test_parse_jsonl(self) -> None:
        """Multi-line JSONL with one object per line."""
        raw = '{"a": 1}\n{"b": 2}'
        result = _parse_json_payload(raw)
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}

    def test_parse_empty_input(self) -> None:
        """Empty or whitespace-only input should raise CliError."""
        with pytest.raises(CliError, match="Empty input"):
            _parse_json_payload("")

    def test_parse_whitespace_only(self) -> None:
        with pytest.raises(CliError, match="Empty input"):
            _parse_json_payload("   \n  ")


class TestInteractionsFromPayload:
    """Tests for _interactions_from_payload()."""

    def test_valid_payload(self) -> None:
        payload = {
            "interactions": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        result = _interactions_from_payload(payload)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].content == "Hello"
        assert result[1].role == "assistant"
        assert result[1].content == "Hi there"

    def test_missing_interactions_key(self) -> None:
        """Payload without 'interactions' key should raise CliError."""
        with pytest.raises(CliError, match="missing 'interactions' list"):
            _interactions_from_payload({"other": "data"})

    def test_empty_interactions_list(self) -> None:
        """Payload with empty interactions list should raise CliError."""
        with pytest.raises(CliError, match="missing 'interactions' list"):
            _interactions_from_payload({"interactions": []})

    def test_default_role(self) -> None:
        """Items without a role key default to 'User'."""
        payload = {"interactions": [{"content": "no role"}]}
        result = _interactions_from_payload(payload)
        assert result[0].role == "User"

    def test_tools_used_preserved(self) -> None:
        """Structured tools_used metadata survives the CLI adapter so the
        server renderer can emit `[used tool: ...]` markers for playbook
        extraction."""
        payload = {
            "interactions": [
                {"role": "user", "content": "run query"},
                {
                    "role": "assistant",
                    "content": "switched tools after first failed",
                    "tools_used": [
                        {
                            "tool_name": "run_snowflake_query",
                            "tool_data": {"statement": "SELECT ...", "status": "failed"},
                        },
                        {
                            "tool_name": "run_snowflake_query",
                            "tool_data": {"statement": "SELECT * LIMIT 1", "status": "ok"},
                        },
                    ],
                },
            ]
        }
        result = _interactions_from_payload(payload)
        assert len(result[1].tools_used) == 2
        assert result[1].tools_used[0].tool_name == "run_snowflake_query"
        assert result[1].tools_used[0].tool_data["status"] == "failed"
        assert result[1].tools_used[1].tool_data["status"] == "ok"

    def test_tools_used_defaults_empty(self) -> None:
        """Interactions without tools_used default to an empty list."""
        payload = {"interactions": [{"role": "user", "content": "hi"}]}
        result = _interactions_from_payload(payload)
        assert result[0].tools_used == []

    def test_invalid_tools_used_raises(self) -> None:
        """Malformed tools_used data raises CliError at publish time
        instead of silently dropping."""
        payload = {
            "interactions": [
                {
                    "role": "assistant",
                    "content": "bad",
                    "tools_used": [{"not_a_tool_field": 123}],
                }
            ]
        }
        with pytest.raises(CliError, match="Invalid interaction data"):
            _interactions_from_payload(payload)
