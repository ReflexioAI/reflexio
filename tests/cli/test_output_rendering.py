"""Rendering behaviour of ``reflexio.cli.output`` for real domain models.

Uses the public API models (``UserProfile``, ``UserPlaybook``,
``AgentPlaybook``) rather than mocks so the formatters are exercised on the
exact objects the CLI receives from the client.
"""

from __future__ import annotations

import time
from enum import Enum
from types import SimpleNamespace

import pytest

from reflexio.cli import output
from reflexio.models.api_schema.domain.enums import PlaybookStatus, Status
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    UserPlaybook,
    UserProfile,
)
from tests.cli.test_output import _capture_rich

NOW = int(time.time())


def _profile(**overrides) -> UserProfile:
    fields = {
        "profile_id": "p1",
        "user_id": "u1",
        "content": "Prefers concise answers",
        "last_modified_timestamp": NOW - 120,
        "generated_from_request_id": "req-1",
    }
    return UserProfile(**{**fields, **overrides})


def _user_playbook(**overrides) -> UserPlaybook:
    fields = {
        "agent_version": "v1",
        "request_id": "req-abcdef123456",
        "content": "Ask before deploying",
    }
    return UserPlaybook(**{**fields, **overrides})


def _agent_playbook(**overrides) -> AgentPlaybook:
    fields = {"agent_version": "2.1", "content": "Cite sources"}
    return AgentPlaybook(**{**fields, **overrides})


def _ix(request_id: str, role: str, content: str, created_at) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id, role=role, content=content, created_at=created_at
    )


class TestRenderFallbacks:
    def test_json_mode_meta_and_list_of_models(self, capsys) -> None:
        output.render(
            [_agent_playbook()], json_mode=True, meta={"count": 1, "limit": 5}
        )
        import json

        envelope = json.loads(capsys.readouterr().out)
        assert envelope["meta"] == {"count": 1, "limit": 5}
        assert envelope["data"][0]["content"] == "Cite sources"
        # exclude_none drops unset optional fields from the envelope
        assert "trigger" not in envelope["data"][0]

    def test_plain_scalar_is_printed_verbatim(self, capsys) -> None:
        output.render(42)
        assert capsys.readouterr().out == "42\n"


class TestLifecycleTag:
    @pytest.mark.parametrize(
        ("status", "tag"),
        [
            (None, ""),
            ("current", ""),
            ("", ""),
            ("pending", " [PENDING]"),
            (Status.ARCHIVED, " [ARCHIVED]"),
            (Status.PENDING, " [PENDING]"),
        ],
    )
    def test_only_non_current_statuses_are_tagged(self, status, tag) -> None:
        assert output._lifecycle_tag(SimpleNamespace(status=status)) == tag

    def test_plain_enum_uses_its_value(self) -> None:
        class Lifecycle(Enum):
            RETIRED = "retired"
            UNSET = None

        assert output._lifecycle_tag(SimpleNamespace(status=Lifecycle.RETIRED)) == (
            " [RETIRED]"
        )
        assert output._lifecycle_tag(SimpleNamespace(status=Lifecycle.UNSET)) == ""

    def test_object_without_status_is_untagged(self) -> None:
        assert output._lifecycle_tag(object()) == ""


class TestPlainFormatters:
    def test_profiles_show_pending_tag_only_for_non_current(self) -> None:
        text = output.format_profiles(
            [
                _profile(),
                _profile(profile_id="p2", content="Old", status=Status.PENDING),
            ]
        )
        assert text.splitlines() == ["- Prefers concise answers", "- Old [PENDING]"]

    def test_agent_playbook_with_structure_and_tags(self) -> None:
        text = output.format_agent_playbooks(
            [
                _agent_playbook(
                    trigger="answering factual questions",
                    rationale="users verify claims",
                    playbook_status=PlaybookStatus.APPROVED,
                    status=Status.ARCHIVED,
                )
            ]
        )
        assert text.splitlines() == [
            "- Cite sources [APPROVED] [ARCHIVED]",
            "  Trigger: answering factual questions",
            "  Rationale: users verify claims",
        ]

    def test_empty_playbook_status_adds_no_badge(self) -> None:
        pb = SimpleNamespace(content="x", playbook_status="", status=None)
        assert output.format_agent_playbooks([pb]) == "- x"

    def test_structured_lines_tolerate_missing_playbook(self) -> None:
        assert output._structured_lines(None) == []

    def test_user_playbooks_show_source_and_request(self) -> None:
        text = output.format_user_playbooks(
            [_user_playbook(source="cli", trigger="before deploys")]
        )
        assert text.splitlines() == [
            "- Ask before deploying",
            "  Trigger: before deploys",
            "  Source: cli (request: req-abcdef123456)",
        ]


class TestHumanRelativeTimestamp:
    @pytest.mark.parametrize(
        ("age", "expected"),
        [
            (5, "s ago"),
            (5 * 60, "5m ago"),
            (3 * 3600, "3h ago"),
            (4 * 86400, "4d ago"),
        ],
    )
    def test_recent_ages(self, age, expected) -> None:
        assert output._human_relative_timestamp(int(time.time()) - age).endswith(
            expected
        )

    def test_old_timestamp_renders_iso_date(self) -> None:
        # 2001-09-09T01:46:40Z
        assert output._human_relative_timestamp(1_000_000_000) == "2001-09-09"

    @pytest.mark.parametrize("value", [None, 0, "not-a-number", 10**13])
    def test_unknown_inputs(self, value) -> None:
        assert output._human_relative_timestamp(value) == "unknown"


class TestPrintUserProfiles:
    def test_numbered_entries_with_metadata_footer(self) -> None:
        out = _capture_rich(
            output.print_user_profiles,
            [
                _profile(source="chat"),
                _profile(
                    profile_id="p2",
                    content="Lives in Berlin",
                    status=Status.ARCHIVED,
                ),
            ],
        )
        assert "[1] Prefers concise answers" in out
        assert "source: chat · 2m ago" in out
        assert "[2] Lives in Berlin" in out
        assert "[ARCHIVED]" in out

    def test_profile_without_metadata_has_no_footer(self) -> None:
        bare = SimpleNamespace(content="Bare", source=None, status=None)
        out = _capture_rich(output.print_user_profiles, [bare])
        assert out.strip() == "[1] Bare"

    def test_empty_prints_nothing(self) -> None:
        assert _capture_rich(output.print_user_profiles, []) == ""


class TestPrintUserPlaybooks:
    def test_panel_shows_content_fields_footer_and_lifecycle(self) -> None:
        out = _capture_rich(
            output.print_user_playbooks,
            [
                _user_playbook(
                    trigger="before deploys",
                    rationale="user restricted approval",
                    source="publish",
                    status=Status.PENDING,
                )
            ],
        )
        assert "Ask before deploying" in out
        assert "[1] [PENDING]" in out
        assert "Trigger" in out and "before deploys" in out
        assert "Rationale" in out and "user restricted approval" in out
        assert "source: publish · request: req-abcd" in out
        assert "req-abcdef123456" not in out

    def test_missing_source_and_request_render_unknown(self) -> None:
        pb = SimpleNamespace(
            content="No provenance", source=None, request_id=None, status=None
        )
        out = _capture_rich(output.print_user_playbooks, [pb])
        assert "source: unknown · request: unknown" in out

    def test_empty_prints_nothing(self) -> None:
        assert _capture_rich(output.print_user_playbooks, []) == ""


class TestPrintAgentPlaybookFooter:
    def test_footer_lists_name_version_and_age(self) -> None:
        out = _capture_rich(
            output.print_agent_playbooks,
            [
                _agent_playbook(
                    playbook_name="citations",
                    created_at=int(time.time()) - 2 * 3600,
                    playbook_status=PlaybookStatus.REJECTED,
                    status=Status.ARCHIVED,
                )
            ],
        )
        assert "citations · v2.1 · 2h ago" in out
        assert "[REJECTED]" in out
        assert "[ARCHIVED]" in out

    def test_unrecognised_badge_and_no_footer(self) -> None:
        pb = SimpleNamespace(
            content="Odd status",
            playbook_status="shadow",
            status=None,
            playbook_name="",
            agent_version="",
            created_at=0,
        )
        out = _capture_rich(output.print_agent_playbooks, [pb])
        assert "[1] [SHADOW]" in out
        assert " · " not in out


class TestPrintInteractionsOrdering:
    def test_groups_ordered_by_earliest_turn_with_role_labels(self) -> None:
        now = int(time.time())
        out = _capture_rich(
            output.print_interactions,
            [
                _ix("late-req-1", "assistant", "second group", now - 10),
                _ix("early-req", "tool", "tool output", now - 500),
                _ix("early-req", "", "no role", now - 400),
                _ix("early-req", "system", "be brief", now - 600),
            ],
        )
        assert out.index("early-re") < out.index("late-req")
        assert "3 turns" in out and "1 turn" in out
        assert out.index("be brief") < out.index("tool output") < out.index("no role")
        assert "Tool" in out
        assert "Unknown" in out
        assert "System" in out

    def test_unparseable_created_at_renders_unknown_age(self) -> None:
        out = _capture_rich(
            output.print_interactions, [_ix("req-x", "user", "hi", "soon")]
        )
        assert "unknown" in out
        assert "hi" in out


class TestPrintDoctorChecksNoHint:
    def test_check_without_hint_and_unknown_status(self) -> None:
        out = _capture_rich(
            output.print_doctor_checks,
            [{"status": "skipped", "name": "gpu", "message": "not probed"}],
        )
        row = out.split()
        assert row[:3] == ["?", "gpu:", "not"]
        assert "probed" in row
        assert "Hint" not in out


class TestFormatInteractions:
    def test_groups_render_utc_headers_in_time_order(self) -> None:
        text = output.format_interactions(
            [
                _ix("bbbbbbbbbbbb", "user", "later", 1_700_000_100),
                _ix("aaaaaaaaaaaa", "assistant", "reply", 1_700_000_010),
                _ix("aaaaaaaaaaaa", "user", "first", 1_700_000_000),
            ]
        )
        blocks = text.split("\n\n")
        assert blocks[0].splitlines() == [
            "── aaaaaaaa (2023-11-14 22:13 UTC) ──",
            "  User: first",
            "  Assistant: reply",
        ]
        assert blocks[1].splitlines()[0].startswith("── bbbbbbbb (")

    def test_invalid_timestamp_renders_unknown(self) -> None:
        text = output.format_interactions([_ix("req-123456789", "user", "x", 10**13)])
        assert text.splitlines()[0] == "── req-1234 (unknown UTC) ──"

    def test_empty(self) -> None:
        assert output.format_interactions([]) == ""


class TestFormatContext:
    def test_all_sections_in_order(self) -> None:
        text = output.format_context(
            [_profile()],
            [_agent_playbook()],
            [_user_playbook()],
        )
        assert text.startswith("---\n# IMPORTANT: Apply These Corrections")
        prefs = text.index("## User Preferences — APPLY THESE")
        agent = text.index("## Behavior Corrections — FOLLOW THESE")
        user = text.index("## User Behavior Corrections — FOLLOW THESE")
        assert prefs < agent < user
        assert "- Prefers concise answers" in text
        assert "- Cite sources" in text
        assert "- Ask before deploying" in text
        assert text.endswith("repeating mistakes the user already corrected.\n---")

    def test_user_playbooks_optional(self) -> None:
        text = output.format_context([], [_agent_playbook()])
        assert "User Preferences" not in text
        assert "User Behavior Corrections" not in text
        assert "## Behavior Corrections" in text

    def test_nothing_to_say_is_empty(self) -> None:
        assert output.format_context([], [], []) == ""
