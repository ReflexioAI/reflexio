"""Unit tests for the eval runner — load, score_plan, group3 replay."""

from __future__ import annotations

from tests.server.services.extraction.eval_runner import (
    load_fixtures,
    run_fixture,
    score_plan,
)


def test_load_fixtures_group1_returns_12():
    fixtures = load_fixtures(group="group1_mutation")
    assert len(fixtures) == 12
    categories = {f["category"] for f in fixtures}
    assert categories == {"supersede", "merge", "delete", "playbook_expansion"}


def test_load_fixtures_group2_returns_18():
    fixtures = load_fixtures(group="group2_retrieval")
    assert len(fixtures) == 18


def test_load_fixtures_group3_returns_4():
    fixtures = load_fixtures(group="group3_loop_behavior")
    assert len(fixtures) == 4


def test_load_fixtures_all_returns_34():
    fixtures = load_fixtures()
    assert len(fixtures) == 12 + 18 + 4


def test_score_plan_exact_match():
    actual = [
        {"op": "delete_user_profile", "id": "p_10"},
        {"op": "create_user_profile", "content": "new fact", "ttl": "infinity"},
    ]
    expected = [
        {"op": "delete_user_profile", "id": "p_10"},
        {"op": "create_user_profile", "content_contains": ["new"], "ttl": "infinity"},
    ]
    result = score_plan(actual, expected)
    assert result["semantic_match"] is True


def test_score_plan_content_preserves_all_catches_lossy_merge():
    """playbook_expansion must preserve all prior instructions."""
    actual = [
        {"op": "create_user_playbook", "trigger": "code", "content": "use TypeScript"}
    ]
    expected = [
        {
            "op": "create_user_playbook",
            "trigger_contains": ["code"],
            "content_contains": ["TypeScript"],
            "content_preserves_all": ["show examples"],
        }
    ]
    result = score_plan(actual, expected)
    assert result["semantic_match"] is False
    assert any("show examples" in f for f in result["failures"])


def test_score_plan_op_count_mismatch():
    actual = [{"op": "delete_user_profile", "id": "p_10"}]
    expected = [
        {"op": "delete_user_profile", "id": "p_10"},
        {"op": "create_user_profile", "content_contains": ["x"], "ttl": "infinity"},
    ]
    result = score_plan(actual, expected)
    assert result["semantic_match"] is False
    assert any("op count" in f for f in result["failures"])


def test_score_plan_op_type_mismatch():
    actual = [{"op": "create_user_profile", "content": "x", "ttl": "infinity"}]
    expected = [{"op": "delete_user_profile", "id": "p_10"}]
    result = score_plan(actual, expected)
    assert result["semantic_match"] is False


def test_run_fixture_group3_confused_garbage(tmp_path):
    """Group 3 replay: confused_garbage should hit A + B violations, commit 0 ops."""
    from unittest.mock import MagicMock

    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    fixtures = load_fixtures(group="group3_loop_behavior")
    fixture = next(f for f in fixtures if f["id"] == "confused_garbage")
    storage = SQLiteStorage(org_id="eval-org", db_path=str(tmp_path / "eval.db"))
    pm = PromptManager()
    client = MagicMock()
    client.config = MagicMock()
    client.config.api_key_config = None

    result = run_fixture(fixture, client=client, prompt_manager=pm, storage=storage)
    assert result["outcome"] == "finish_tool"
    assert result["applied_count"] == 0
    assert set(result["violation_codes"]) >= {"A", "B"}
