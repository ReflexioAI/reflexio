"""Formatter unit tests for reflexio.cli.output.

All tests use MagicMock objects to simulate Pydantic models,
avoiding any dependency on storage or network.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from reflexio.cli.output import (
    format_agent_playbooks,
    format_interactions,
    format_profiles,
    format_user_playbooks,
    mask_api_key,
    pagination_meta,
    render,
)


def _make_playbook(**kwargs):
    """Build a playbook-like mock with safe defaults for status attributes.

    MagicMock auto-creates attributes on access, which confuses
    _lifecycle_tag(). We use spec=[] to prevent that, then set
    only the attributes we need.
    """
    fb = MagicMock(spec=[])
    fb.content = kwargs.get("content", "")
    fb.structured_data = kwargs.get("structured_data")
    fb.playbook_status = kwargs.get("playbook_status")
    fb.status = kwargs.get("status")
    return fb


# ---------------------------------------------------------------------------
# format_interactions
# ---------------------------------------------------------------------------


class TestFormatInteractions:
    """Tests for format_interactions()."""

    def test_empty_list(self) -> None:
        assert format_interactions([]) == ""

    def test_groups_by_request_id(self) -> None:
        ix1 = MagicMock(request_id="req-1", role="user", content="Hi", created_at=1000)
        ix2 = MagicMock(
            request_id="req-1", role="assistant", content="Hello", created_at=1001
        )
        output = format_interactions([ix1, ix2])
        # Both should appear in a single block headed by req-1's short id
        assert "req-1" in output[:20]
        assert "Hi" in output
        assert "Hello" in output
        # Only one header line (one group)
        header_count = output.count("──")
        # The header format is: ── {short_id} ({dt} UTC) ──
        # So each group has 2 occurrences of "──"
        assert header_count == 2

    def test_sorted_by_timestamp(self) -> None:
        """Earlier request_id groups should appear first."""
        ix_early = MagicMock(
            request_id="aaa-early", role="user", content="First", created_at=100
        )
        ix_late = MagicMock(
            request_id="bbb-later", role="user", content="Second", created_at=999
        )
        output = format_interactions([ix_late, ix_early])
        # The early group should come before the late group
        assert output.index("First") < output.index("Second")

    def test_shows_role_and_content(self) -> None:
        ix = MagicMock(
            request_id="req-1", role="user", content="What is 2+2?", created_at=5000
        )
        ix2 = MagicMock(
            request_id="req-1",
            role="assistant",
            content="4",
            created_at=5001,
        )
        output = format_interactions([ix, ix2])
        assert "User:" in output
        assert "Assistant:" in output

    def test_invalid_created_at(self) -> None:
        """Non-int created_at should fall back to 'unknown'."""
        ix = MagicMock(
            request_id="req-x", role="user", content="test", created_at="not-a-number"
        )
        output = format_interactions([ix])
        assert "unknown" in output


# ---------------------------------------------------------------------------
# format_agent_playbooks
# ---------------------------------------------------------------------------


class TestFormatAgentPlaybooks:
    """Tests for format_agent_playbooks()."""

    def test_empty_list(self) -> None:
        assert format_agent_playbooks([]) == ""

    def test_shows_playbook_content(self) -> None:
        fb = _make_playbook(content="Use formal tone")
        output = format_agent_playbooks([fb])
        assert "Use formal tone" in output

    def test_shows_structured_data_labels(self) -> None:
        sd = MagicMock(spec=[])
        sd.instruction = "Address as Mr/Ms"
        sd.pitfall = "Don't use casual language"
        sd.trigger = "enterprise user detected"
        sd.rationale = "Compliance requirement"
        fb = _make_playbook(
            content="Tone rule",
            structured_data=sd,
        )
        output = format_agent_playbooks([fb])
        assert "Instruction:" in output
        assert "Pitfall:" in output
        assert "Trigger:" in output
        assert "Rationale:" in output

    def test_shows_approval_status_tag(self) -> None:
        fb = _make_playbook(
            content="rule",
            playbook_status="approved",
        )
        output = format_agent_playbooks([fb])
        assert "[APPROVED]" in output

    def test_no_crash_on_missing_playbook_status(self) -> None:
        """Works when playbook_status is None."""
        fb = _make_playbook(content="safe")
        output = format_agent_playbooks([fb])
        assert "safe" in output
        assert "[" not in output  # No status tag

    def test_no_crash_on_playbook_status_attr_missing(self) -> None:
        """Works when the playbook_status attribute does not exist at all."""
        fb = MagicMock(spec=[])
        fb.content = "no attr"
        fb.structured_data = None
        fb.status = None
        # No playbook_status attribute set -- getattr returns None
        output = format_agent_playbooks([fb])
        assert "no attr" in output


# ---------------------------------------------------------------------------
# format_user_playbooks
# ---------------------------------------------------------------------------


class TestFormatUserPlaybooks:
    """Tests for format_user_playbooks()."""

    def test_empty_list(self) -> None:
        assert format_user_playbooks([]) == ""

    def test_shows_playbook_content(self) -> None:
        fb = MagicMock(spec=[])
        fb.content = "Raw note"
        fb.structured_data = None
        fb.status = None
        fb.source = "cli"
        fb.request_id = "req-abc"
        output = format_user_playbooks([fb])
        assert "Raw note" in output

    def test_shows_source_metadata(self) -> None:
        fb = MagicMock(spec=[])
        fb.content = "entry"
        fb.structured_data = None
        fb.status = None
        fb.source = "api"
        fb.request_id = "req-xyz"
        output = format_user_playbooks([fb])
        assert "Source:" in output
        assert "req-xyz" in output

    def test_no_approval_tag(self) -> None:
        """User playbooks should not show [APPROVED]/[PENDING]/[REJECTED]."""
        fb = MagicMock(spec=[])
        fb.content = "raw item"
        fb.structured_data = None
        fb.status = None
        fb.source = "cli"
        fb.request_id = "req-1"
        output = format_user_playbooks([fb])
        assert "[APPROVED]" not in output
        assert "[PENDING]" not in output
        assert "[REJECTED]" not in output


# ---------------------------------------------------------------------------
# format_profiles
# ---------------------------------------------------------------------------


class TestFormatProfiles:
    """Tests for format_profiles()."""

    def test_empty_list(self) -> None:
        assert format_profiles([]) == ""

    def test_shows_profile_content(self) -> None:
        p1 = MagicMock(spec=[])
        p1.content = "Likes Python"
        p1.status = None
        p2 = MagicMock(spec=[])
        p2.content = "Prefers dark mode"
        p2.status = None
        output = format_profiles([p1, p2])
        assert "- Likes Python" in output
        assert "- Prefers dark mode" in output

    def test_shows_status_tag(self) -> None:
        """Profiles with 'pending' status should show a [PENDING] tag."""
        p = MagicMock(spec=[])
        p.content = "Preference item"
        p.status = "pending"
        output = format_profiles([p])
        assert "[PENDING]" in output


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


class TestRender:
    """Tests for render()."""

    def test_json_mode_envelope(self, capsys) -> None:
        render({"key": "value"}, json_mode=True, meta={"count": 1})
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["ok"] is True
        assert envelope["data"] == {"key": "value"}
        assert envelope["meta"] == {"count": 1}

    def test_human_mode_string(self, capsys) -> None:
        render("Hello, world!")
        captured = capsys.readouterr()
        assert captured.out.strip() == "Hello, world!"

    def test_human_mode_dict(self, capsys) -> None:
        render({"a": 1})
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == {"a": 1}

    def test_pydantic_safety_net(self, capsys) -> None:
        """A model with model_dump() should be serialized to dict."""
        model = MagicMock()
        model.model_dump.return_value = {"field": "val"}
        render(model)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == {"field": "val"}


# ---------------------------------------------------------------------------
# pagination_meta
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rich helpers — captured via Console(record=True) and exported as plain text
# ---------------------------------------------------------------------------


def _capture_rich(callable_, *args, **kwargs) -> str:
    """Run a print_* helper with a recording console patched in.

    Each ``print_*`` helper instantiates ``rich.console.Console()``
    locally inside the function (so they degrade to plain text when
    stdout is piped). To capture output for assertions, we monkey-patch
    ``rich.console.Console`` to a recording variant for the duration of
    the call, then export plain text via ``console.export_text()``.

    Args:
        callable_: The print_* helper under test.
        *args: Positional args forwarded to the helper.
        **kwargs: Keyword args forwarded to the helper.

    Returns:
        str: The captured output with ANSI codes stripped.
    """
    import rich.console as rc

    captured: list[rc.Console] = []
    original = rc.Console

    def _make_recording_console(*c_args, **c_kwargs):  # type: ignore[no-untyped-def]
        c_kwargs.setdefault("record", True)
        c_kwargs.setdefault("width", 120)
        c_kwargs.setdefault("force_terminal", False)
        console = original(*c_args, **c_kwargs)
        captured.append(console)
        return console

    rc.Console = _make_recording_console  # type: ignore[misc]
    try:
        callable_(*args, **kwargs)
    finally:
        rc.Console = original  # type: ignore[misc]

    return "".join(c.export_text() for c in captured)


class TestMaskApiKey:
    """Tests for mask_api_key()."""

    def test_empty(self) -> None:
        assert mask_api_key("") == "<unset>"

    def test_short_key(self) -> None:
        # 8 chars or fewer become a row of stars
        assert mask_api_key("abc") == "***"
        assert mask_api_key("12345678") == "********"

    def test_long_key_with_dash_prefix(self) -> None:
        # rflx-XXXXXXX...XXXX -> first 5 chars + stars + last 4
        masked = mask_api_key("rflx-abcdefghij1234")
        assert masked.startswith("rflx-")
        assert masked.endswith("1234")
        assert "abcdefghij" not in masked

    def test_long_key_no_dash(self) -> None:
        # No dash in first 8 -> first 4 chars + stars + last 4
        masked = mask_api_key("abcdefghijklmnop")
        assert masked.startswith("abcd")
        assert masked.endswith("mnop")
        assert "efghijkl" not in masked


class TestPrintAgentPlaybooks:
    """Tests for print_agent_playbooks()."""

    def test_renders_approval_badge(self) -> None:
        from reflexio.cli.output import print_agent_playbooks

        pb = _make_playbook(content="Always greet politely", playbook_status="approved")
        output = _capture_rich(print_agent_playbooks, [pb])
        assert "Always greet politely" in output
        assert "[APPROVED]" in output

    def test_pending_badge(self) -> None:
        from reflexio.cli.output import print_agent_playbooks

        pb = _make_playbook(content="Refer user by name", playbook_status="pending")
        output = _capture_rich(print_agent_playbooks, [pb])
        assert "[PENDING]" in output

    def test_renders_structured_grid(self) -> None:
        from reflexio.cli.output import print_agent_playbooks

        sd = MagicMock(spec=[])
        sd.trigger = "user mentions price"
        sd.instruction = "quote the latest figure"
        sd.pitfall = "do not invent prices"
        sd.rationale = "accuracy matters"
        pb = _make_playbook(content="Pricing rule", structured_data=sd)
        output = _capture_rich(print_agent_playbooks, [pb])
        assert "Trigger" in output
        assert "Instruction" in output
        assert "Pitfall" in output
        assert "Rationale" in output
        assert "user mentions price" in output

    def test_empty_list_no_output(self) -> None:
        from reflexio.cli.output import print_agent_playbooks

        assert _capture_rich(print_agent_playbooks, []) == ""


class TestPrintInteractions:
    """Tests for print_interactions()."""

    def test_groups_by_request_id(self) -> None:
        from reflexio.cli.output import print_interactions

        ix1 = MagicMock(
            request_id="req-aaa", role="user", content="Hello", created_at=1000
        )
        ix2 = MagicMock(
            request_id="req-aaa",
            role="assistant",
            content="World",
            created_at=1001,
        )
        ix3 = MagicMock(
            request_id="req-bbb", role="user", content="Foo", created_at=2000
        )
        output = _capture_rich(print_interactions, [ix1, ix2, ix3])
        assert "Hello" in output
        assert "World" in output
        assert "Foo" in output
        assert "User" in output
        assert "Assistant" in output
        # Two distinct groups => two short ids
        assert "req-aaa"[:8] in output
        assert "req-bbb"[:8] in output

    def test_shows_turn_count(self) -> None:
        from reflexio.cli.output import print_interactions

        ix = MagicMock(request_id="req-1", role="user", content="hi", created_at=100)
        output = _capture_rich(print_interactions, [ix])
        assert "1 turn" in output

        ix2 = MagicMock(
            request_id="req-1", role="assistant", content="hello", created_at=101
        )
        output_two = _capture_rich(print_interactions, [ix, ix2])
        assert "2 turns" in output_two

    def test_empty_list_no_output(self) -> None:
        from reflexio.cli.output import print_interactions

        assert _capture_rich(print_interactions, []) == ""


class TestPrintStorageCredentials:
    """Tests for print_storage_credentials()."""

    def test_masked_shows_reveal_hint(self) -> None:
        from reflexio.cli.output import print_storage_credentials

        output = _capture_rich(
            print_storage_credentials,
            "supabase",
            {"url": "https://*****.supabase.co", "key": "ey***"},
            revealed=False,
        )
        assert "supabase" in output
        assert "url" in output
        assert "--reveal" in output

    def test_revealed_no_hint(self) -> None:
        from reflexio.cli.output import print_storage_credentials

        output = _capture_rich(
            print_storage_credentials,
            "sqlite",
            {"path": "/var/data/reflexio.db"},
            revealed=True,
        )
        assert "/var/data/reflexio.db" in output
        assert "--reveal" not in output

    def test_unknown_storage_type(self) -> None:
        from reflexio.cli.output import print_storage_credentials

        output = _capture_rich(
            print_storage_credentials, None, {"foo": "bar"}, revealed=True
        )
        assert "unknown" in output


class TestPrintWhoamiSummary:
    """Tests for print_whoami_summary()."""

    def test_all_rows_present(self) -> None:
        from reflexio.cli.output import print_whoami_summary

        output = _capture_rich(
            print_whoami_summary,
            endpoint="https://api.reflexio.ai",
            api_key="rflx-abcdefghijklmn1234",
            org_id="acme-corp",
            storage_type="supabase",
            storage_label="db.reflexio.ai/****",
            storage_configured=True,
            message=None,
        )
        assert "Endpoint" in output
        assert "https://api.reflexio.ai" in output
        assert "rflx-" in output
        assert "1234" in output
        assert "acme-corp" in output
        assert "supabase" in output
        assert "[configured]" in output

    def test_unconfigured_marker(self) -> None:
        from reflexio.cli.output import print_whoami_summary

        output = _capture_rich(
            print_whoami_summary,
            endpoint="https://api.reflexio.ai",
            api_key="",
            org_id=None,
            storage_type=None,
            storage_label="db.reflexio.ai/****",
            storage_configured=False,
            message="upgrade your plan",
        )
        assert "[unconfigured]" in output
        assert "upgrade your plan" in output
        assert "<unset>" in output  # masked empty api key
        assert "<none>" in output  # missing org id

    def test_no_storage_label_no_config(self) -> None:
        from reflexio.cli.output import print_whoami_summary

        output = _capture_rich(
            print_whoami_summary,
            endpoint="http://localhost:8081",
            api_key="rflx-shortish-key",
            org_id="self-host",
            storage_type=None,
            storage_label=None,
            storage_configured=False,
            message=None,
        )
        assert "<not configured>" in output


class TestPrintDoctorChecks:
    """Tests for print_doctor_checks()."""

    def test_renders_pass_warn_fail(self) -> None:
        from reflexio.cli.output import print_doctor_checks

        checks = [
            {
                "name": "env_file",
                "status": "pass",
                "message": "Env file at /tmp/.env",
                "hint": None,
            },
            {
                "name": "api_key",
                "status": "warn",
                "message": "No API key configured",
                "hint": "Run: reflexio auth login",
            },
            {
                "name": "server_health",
                "status": "fail",
                "message": "Cannot connect to http://localhost:8081",
                "hint": "Start the server: reflexio services start",
            },
        ]
        output = _capture_rich(print_doctor_checks, checks)
        assert "env_file" in output
        assert "api_key" in output
        assert "server_health" in output
        assert "Env file at /tmp/.env" in output
        assert "No API key configured" in output
        assert "Cannot connect to" in output
        assert "Run: reflexio auth login" in output  # hint text
        assert "Start the server" in output

    def test_empty_list_no_output(self) -> None:
        from reflexio.cli.output import print_doctor_checks

        assert _capture_rich(print_doctor_checks, []) == ""


class TestPrintAuthStatus:
    """Tests for print_auth_status()."""

    def test_all_rows(self) -> None:
        from reflexio.cli.output import print_auth_status

        output = _capture_rich(
            print_auth_status,
            url="https://api.reflexio.ai",
            api_key="rflx-abcdefghijklmn1234",
            env_path="/home/me/.reflexio/.env",
        )
        assert "URL" in output
        assert "https://api.reflexio.ai" in output
        assert "API Key" in output
        assert "rflx-" in output
        assert "1234" in output
        assert "Env file" in output
        assert "/home/me/.reflexio/.env" in output

    def test_empty_url_and_key(self) -> None:
        from reflexio.cli.output import print_auth_status

        output = _capture_rich(
            print_auth_status,
            url="",
            api_key="",
            env_path="/tmp/.env",
        )
        assert "<unset>" in output


class TestPaginationMeta:
    """Tests for pagination_meta()."""

    def test_count_and_limit(self) -> None:
        meta = pagination_meta([1, 2, 3], limit=10)
        assert meta["count"] == 3
        assert meta["limit"] == 10

    def test_has_more(self) -> None:
        meta = pagination_meta([1, 2, 3, 4, 5], limit=5)
        assert meta["has_more"] is True

    def test_has_more_false(self) -> None:
        meta = pagination_meta([1, 2], limit=5)
        assert meta["has_more"] is False
