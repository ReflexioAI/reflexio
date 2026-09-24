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


def test_to_wire_citations_preserves_explicit_playbook_source_kind() -> None:
    wire = state._to_wire_citations(
        [
            {
                "id": "s1-1",
                "kind": "playbook",
                "source_kind": "agent_playbook",
                "real_id": "20",
                "title": "Agent",
            },
            {
                "id": "s2-1",
                "kind": "playbook",
                "source_kind": "user_playbook",
                "real_id": "101",
                "title": "User",
            },
            {"id": "p1-1", "kind": "profile", "real_id": "p1", "title": "Profile"},
        ]
    )

    assert [item["kind"] for item in wire] == [
        "agent_playbook",
        "user_playbook",
        "profile",
    ]


class TestWireFieldAllowlist:
    """The slicer must emit only fields the server's model declares.

    This started as a denylist (`role`, `ts`, `cited_items`), so every
    buffer-internal key added later — `user_id` was the live one — rode
    along to the wire and was silently discarded server-side.
    """

    def test_allowlist_covers_every_model_field(self):
        """A field the installed model declares but the allowlist omits is a
        real bug: the slicer would silently drop it.

        The reverse (allowlist ⊃ model) is fine and expected — this plugin
        pins the field set literally so hooks keep working when `reflexio`
        is not importable, and an older installed model may lag it.
        """
        from reflexio.models.api_schema.domain.entities import InteractionData

        missing = set(InteractionData.model_fields) - state._INTERACTION_DATA_FIELDS
        assert not missing, (
            f"allowlist omits InteractionData field(s) {sorted(missing)};"
            " the slicer would silently drop them"
        )

    def test_wire_fields_excludes_created_at_but_contract_set_keeps_it(self):
        assert "created_at" in state._INTERACTION_DATA_FIELDS
        assert "created_at" not in state._WIRE_FIELDS
        assert state._WIRE_FIELDS < state._INTERACTION_DATA_FIELDS

    def test_bookkeeping_keys_never_reach_the_wire(self):
        """`ts` and `user_id` are buffer-internal; `created_at` backdates the
        row past the extractor's bookmark (see `_WIRE_FIELDS`)."""
        _, turns = state.unpublished_slice(
            [
                {
                    "ts": 1700000000,
                    "role": "User",
                    "content": "x",
                    "user_id": "proj",
                    "created_at": 999,
                }
            ]
        )
        assert turns[0] == {"role": "User", "content": "x"}

    def test_unknown_future_key_is_dropped_by_default(self):
        _, turns = state.unpublished_slice(
            [{"role": "User", "content": "x", "some_new_bookkeeping_key": 1}]
        )
        assert "some_new_bookkeeping_key" not in turns[0]

    def test_real_model_fields_still_pass_through(self):
        """Guards against over-filtering: the allowlist must not strip
        content-bearing fields the server accepts."""
        _, turns = state.unpublished_slice(
            [
                {
                    "ts": 1,
                    "role": "User",
                    "content": "c",
                    "shadow_content": "s",
                    "user_action": "click",
                }
            ]
        )
        assert turns[0]["shadow_content"] == "s"
        assert turns[0]["user_action"] == "click"


class TestRetrievedLearningRefs:
    """``retrieved_learnings`` is what correlates a serve to the turn it shaped.

    The citation sidecar is a whole-session lookup keyed by short id, so it
    cannot answer "what was injected for THIS turn". The refs ride the ordered
    buffer instead, and fold into the next Assistant turn.
    """

    def test_registry_entries_normalise_to_identity_pairs(self):
        state.append_retrieved_learning_refs(
            "r1",
            [
                {
                    "id": "a1",
                    "kind": "playbook",
                    "source_kind": "user_playbook",
                    "real_id": "12",
                },
                {
                    "id": "a2",
                    "kind": "playbook",
                    "source_kind": "agent_playbook",
                    "real_id": "34",
                },
                {"id": "a3", "kind": "profile", "real_id": "p-9"},
            ],
        )
        records = state.read_all("r1")
        assert len(records) == 1
        assert records[0]["retrieved_learning_refs"] == [
            {"kind": "user_playbook", "learning_id": "12"},
            {"kind": "agent_playbook", "learning_id": "34"},
            {"kind": "profile", "learning_id": "p-9"},
        ]

    def test_entries_without_a_resolvable_storage_id_are_dropped(self):
        state.append_retrieved_learning_refs(
            "r2",
            [
                {
                    "id": "a1",
                    "kind": "playbook",
                    "source_kind": "user_playbook",
                    "real_id": None,
                },
                {"id": "a2", "kind": "playbook", "real_id": "7"},
                {
                    "id": "a3",
                    "kind": "playbook",
                    "source_kind": "nonsense",
                    "real_id": "8",
                },
            ],
        )
        assert state.read_all("r2") == []

    def test_refs_attach_to_the_next_assistant_turn_only(self):
        _, turns = state.unpublished_slice(
            [
                {"role": "User", "content": "q"},
                {
                    "retrieved_learning_refs": [
                        {"kind": "user_playbook", "learning_id": "12"}
                    ]
                },
                {"role": "Assistant", "content": "a"},
                {"role": "Assistant", "content": "b"},
            ]
        )
        assert "retrieved_learnings" not in turns[0]
        assert turns[1]["retrieved_learnings"] == [
            {"kind": "user_playbook", "learning_id": "12"}
        ]
        assert "retrieved_learnings" not in turns[2]

    def test_repeated_injections_in_one_turn_are_deduped(self):
        _, turns = state.unpublished_slice(
            [
                {
                    "retrieved_learning_refs": [
                        {"kind": "user_playbook", "learning_id": "12"}
                    ]
                },
                {
                    "retrieved_learning_refs": [
                        {"kind": "user_playbook", "learning_id": "12"},
                        {"kind": "profile", "learning_id": "p1"},
                    ]
                },
                {"role": "Assistant", "content": "a"},
            ]
        )
        assert turns[0]["retrieved_learnings"] == [
            {"kind": "user_playbook", "learning_id": "12"},
            {"kind": "profile", "learning_id": "p1"},
        ]

    def test_malformed_refs_are_skipped_not_published(self):
        _, turns = state.unpublished_slice(
            [
                {
                    "retrieved_learning_refs": [
                        {"kind": "user_playbook", "learning_id": ""},
                        {"kind": "not_a_kind", "learning_id": "9"},
                        "garbage",
                        {"kind": "profile", "learning_id": "ok"},
                    ]
                },
                {"role": "Assistant", "content": "a"},
            ]
        )
        assert turns[0]["retrieved_learnings"] == [
            {"kind": "profile", "learning_id": "ok"}
        ]

    def test_refs_do_not_survive_a_published_watermark(self):
        _, turns = state.unpublished_slice(
            [
                {
                    "retrieved_learning_refs": [
                        {"kind": "user_playbook", "learning_id": "12"}
                    ]
                },
                {"published_up_to": 1},
                {"role": "Assistant", "content": "a"},
            ]
        )
        assert "retrieved_learnings" not in turns[0]

    def test_publish_request_cap_trims_the_tail(self):
        """Over-cap the server 422s the whole batch, and the adapter swallows
        it without advancing the watermark -- the buffer would never drain."""
        cap = state._RETRIEVED_LEARNINGS_PUBLISH_CAP
        refs = [
            {"kind": "user_playbook", "learning_id": str(n)} for n in range(cap + 5)
        ]
        _, turns = state.unpublished_slice(
            [{"retrieved_learning_refs": refs}, {"role": "Assistant", "content": "a"}]
        )
        assert len(turns[0]["retrieved_learnings"]) == cap
