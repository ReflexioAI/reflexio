from reflexio.server.services.storage.storage_base import (
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    canonical_json,
    human_feedback_scope,
    normalize_dedup_text,
)


def test_scope_hash_uses_canonical_json_ordering():
    left = {"scope_kind": "org", "org_id": "org_1"}
    right = {"org_id": "org_1", "scope_kind": "org"}

    assert canonical_json(left) == canonical_json(right)
    assert build_scope_hash(left) == build_scope_hash(right)


def test_human_feedback_scope_never_includes_user_id():
    scope = human_feedback_scope("org_1")

    assert scope == {"org_id": "org_1", "scope_kind": "org"}
    assert "user_id" not in scope


def test_dedup_key_normalizes_case_unicode_and_whitespace():
    key_a = build_pending_tool_call_dedup_key(
        tool_name="Ask_Human",
        question_text="  What\u00a0is   the user's deployment target? ",
        answer_format=" Plain Text ",
    )
    key_b = build_pending_tool_call_dedup_key(
        tool_name="ask_human",
        question_text="what is the user's deployment target?",
        answer_format="plain text",
    )

    assert key_a == key_b


def test_missing_answer_format_normalizes_to_empty_string():
    assert normalize_dedup_text(None) == ""
    assert build_pending_tool_call_dedup_key(
        tool_name="ask_human",
        question_text="Need deployment target?",
        answer_format=None,
    ) == build_pending_tool_call_dedup_key(
        tool_name="ask_human",
        question_text="Need deployment target?",
        answer_format="",
    )
