"""
Integration tests for derived polarity in the agentic extraction loop.

These tests drive ``ExtractionAgent.run`` end-to-end with the playbook tool
registry, real SQLite storage, and a scripted (mocked) LLM. They cover the two
behaviours D3's prompt guidance is meant to elicit:

* A failure window — clear user pushback — produces an Avoid-prefixed
  ``create_user_playbook`` tool call; the corresponding ``UserPlaybook`` lands
  in storage with derived ``polarity == "negative"``.
* An existing positive playbook on the same trigger PLUS a failure window
  results in the agent calling ``delete_user_playbook(existing_id)`` AND
  ``create_user_playbook`` with avoidance wording; storage reflects the
  delete-then-create reconciliation with derived negative polarity.

The LLM is mocked, so these tests don't validate that an LLM would CHOOSE this
tool-call sequence — they validate that the agent's plan-apply path correctly
derives polarity before storage and correctly handles a delete-then-create
sequence emitted by the agent.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.extraction.extraction_agent import ExtractionAgent
from reflexio.server.services.extraction.tools import PLAYBOOK_EXTRACTION_TOOLS

pytestmark = pytest.mark.integration


# ===============================
# Fixtures
# ===============================


@pytest.fixture
def temp_storage(tmp_path, worker_id):
    """Real SQLite storage in a per-test temp dir + per-worker org id.

    Args:
        tmp_path: pytest builtin temp-directory fixture.
        worker_id (str): xdist worker id; used so concurrent workers don't
            collide on org_id.

    Returns:
        SQLiteStorage: Isolated storage handle.
    """
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    return SQLiteStorage(
        org_id=f"test-org-agentic-polarity-{worker_id}",
        db_path=str(tmp_path / "agentic_polarity.db"),
    )


@pytest.fixture
def prompt_manager():
    """Real PromptManager — the agent renders the playbook extraction prompt."""
    from reflexio.server.prompt.prompt_manager import PromptManager

    return PromptManager()


@pytest.fixture
def llm_client():
    """Mock LLM client whose ``generate_chat_response`` returns scripted tool calls."""
    client = MagicMock()
    client.config = MagicMock()
    client.config.api_key_config = None
    return client


# ===============================
# Helpers
# ===============================


def _mk_tool_response(tool_calls, content=None):
    """Build a fake LLM response shape that ``run_tool_loop`` accepts.

    Args:
        tool_calls (list): Mocked tool-call objects from ``_mk_tool_call``.
        content (str | None): Optional textual content for the assistant turn.

    Returns:
        MagicMock: An object with ``.tool_calls`` and ``.content`` attributes.
    """
    resp = MagicMock()
    resp.tool_calls = tool_calls
    resp.content = content
    return resp


def _mk_tool_call(id_: str, name: str, args_dict: dict) -> MagicMock:
    """Build a fake tool-call object matching the run_tool_loop dispatch shape.

    Args:
        id_ (str): Tool-call id (any unique string).
        name (str): Tool name registered in PLAYBOOK_EXTRACTION_TOOLS.
        args_dict (dict): JSON-serialisable args dict for the tool.

    Returns:
        MagicMock: An object exposing ``.id``, ``.function.name``,
            ``.function.arguments`` (JSON-encoded).
    """
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args_dict)
    return tc


def _build_playbook_agent(
    llm_client: MagicMock,
    storage,
    prompt_manager,
) -> ExtractionAgent:
    """Construct an ExtractionAgent wired to the playbook tool registry.

    Args:
        llm_client (MagicMock): Mocked LLM client.
        storage: Real BaseStorage (SQLite).
        prompt_manager: Real PromptManager — needed to render the playbook
            extraction prompt.

    Returns:
        ExtractionAgent: Ready for ``.run(...)``.
    """
    return ExtractionAgent(
        client=llm_client,
        storage=storage,
        prompt_manager=prompt_manager,
        registry=PLAYBOOK_EXTRACTION_TOOLS,
        max_steps=12,
    )


# ===============================
# Tests
# ===============================


def test_agentic_loop_failure_window_emits_negative_playbook(
    temp_storage,
    prompt_manager,
    llm_client,
):
    """Failure window → agent creates a UserPlaybook with ``polarity='negative'``.

    Scripted LLM tool-call sequence:
      1. ``search_user_playbooks`` — returns empty (no existing playbooks).
      2. ``create_user_playbook`` with Avoid-prefixed content and failure
         rationale.
      3. ``finish``.

    Asserts that after ``ExtractionAgent.run`` the storage contains exactly one
    ``UserPlaybook`` with ``polarity == "negative"`` — confirming the polarity
    value is derived from content/rationale before the entity is persisted.
    """
    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1",
                    "search_user_playbooks",
                    {"query": "cancellation confirmation", "top_k": 10},
                )
            ]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "create_user_playbook",
                    {
                        "trigger": "user confirms a cancellation request",
                        "content": "Avoid asking the user to confirm a cancellation more than once",
                        "rationale": "User pushed back on repeated confirmation prompts",
                        "source_span": "I already said yes, stop asking me to confirm.",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c3", "finish", {})]),
    ]

    user_id = "u_failure_negative"
    agent = _build_playbook_agent(llm_client, temp_storage, prompt_manager)
    result = agent.run(
        user_id=user_id,
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="Extract behavioural rules.",
        sessions_text=(
            "User: Please cancel my subscription.\n"
            "Assistant: Can you confirm you want to cancel? Are you sure?\n"
            "User: I already said yes, stop asking me to confirm."
        ),
        extraction_kind="UserPlaybook",
        request_id="req_failure_neg",
    )

    assert result.outcome == "finish_tool"
    assert len(result.applied) == 1, (
        f"Expected exactly one applied op (create), got: {result.applied}"
    )

    playbooks = temp_storage.get_user_playbooks(user_id=user_id)
    assert len(playbooks) == 1, f"Expected one persisted UserPlaybook, got: {playbooks}"
    assert playbooks[0].polarity == "negative", (
        f"polarity must be derived before storage; got {playbooks[0].polarity!r}"
    )
    assert playbooks[0].content.startswith("Avoid"), (
        "Negative-polarity content should preserve its Avoid-prefix end-to-end; "
        f"got: {playbooks[0].content!r}"
    )


def test_agentic_loop_existing_positive_plus_failure_deletes_then_creates_negative(
    temp_storage,
    prompt_manager,
    llm_client,
):
    """Existing positive playbook + failure window → delete-then-create-negative.

    Seeds storage with a positive UserPlaybook on a recommend-X trigger, then
    drives the agent loop with this scripted tool-call sequence:
      1. ``search_user_playbooks`` — returns the seed (and adds its id to
         ``ctx.known_ids`` so invariant B accepts the subsequent delete).
      2. ``delete_user_playbook(id=<seed_id>)``.
      3. ``create_user_playbook`` with avoidance wording.
      4. ``finish``.

    Asserts the final storage state:
      * The seed positive playbook is gone.
      * Exactly one playbook remains, with ``polarity == "negative"`` and an
        Avoid-prefixed content body.

    This exercises the apply-path for the orientation-aware reconciliation that
    D3 added to the playbook extraction prompt — the LLM is mocked, so this
    test doesn't verify the LLM would CHOOSE this sequence; it verifies the
    pipeline correctly handles a delete-then-create sequence with derived
    polarity.
    """
    from reflexio.models.api_schema.domain.entities import UserPlaybook

    user_id = "u_reconcile_negative"
    # Seed a positive playbook on the same trigger — what the agent will
    # then "find" via search and "delete" before creating a negative one.
    seed = UserPlaybook(
        user_playbook_id=0,  # storage assigns
        user_id=user_id,
        agent_version="v1",
        request_id="req_seed",
        playbook_name="default",
        content="Recommend X to the user when they are unsure",
        trigger="user is unsure how to proceed",
        rationale="seeded positive baseline",
        source_span="seed",
        polarity="positive",
    )
    temp_storage.save_user_playbooks([seed])
    seeded = temp_storage.get_user_playbooks(user_id=user_id)
    assert len(seeded) == 1, "Seed setup failed — expected one seeded playbook"
    seed_id = str(seeded[0].user_playbook_id)

    llm_client.generate_chat_response.side_effect = [
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c1",
                    "search_user_playbooks",
                    {"query": "user is unsure", "top_k": 10},
                )
            ]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c2",
                    "delete_user_playbook",
                    {"id": seed_id},
                )
            ]
        ),
        _mk_tool_response(
            [
                _mk_tool_call(
                    "c3",
                    "create_user_playbook",
                    {
                        "trigger": "user is unsure how to proceed",
                        "content": "Avoid recommending X when the user has said no",
                        "rationale": "Window shows user pushback on recommending X",
                        "source_span": "I told you no, stop suggesting X.",
                    },
                )
            ]
        ),
        _mk_tool_response([_mk_tool_call("c4", "finish", {})]),
    ]

    agent = _build_playbook_agent(llm_client, temp_storage, prompt_manager)
    result = agent.run(
        user_id=user_id,
        agent_version="v1",
        extractor_name="default",
        extraction_criteria="Extract behavioural rules.",
        sessions_text=(
            "Assistant: I recommend X here.\nUser: I told you no, stop suggesting X."
        ),
        extraction_kind="UserPlaybook",
        request_id="req_reconcile",
    )

    assert result.outcome == "finish_tool"
    assert len(result.applied) == 2, (
        "Expected two applied ops (delete + create); got: "
        f"{[type(op).__name__ for op in result.applied]}"
    )
    # No hard violations should have dropped the delete (invariant B passes
    # because search added seed_id to ctx.known_ids).
    hard_violations = [v for v in result.violations if v.severity == "hard"]
    assert hard_violations == [], (
        f"Unexpected hard invariant violations: {hard_violations}"
    )

    surviving = temp_storage.get_user_playbooks(user_id=user_id)
    # Filter to non-archived (the search-status default is "current"), but the
    # SQLite delete is a hard remove anyway — so surviving has one row.
    assert len(surviving) == 1, (
        f"Expected exactly one playbook after reconciliation, got: {surviving}"
    )
    survivor = surviving[0]
    assert str(survivor.user_playbook_id) != seed_id, (
        "Seed playbook should have been deleted, but its id still exists"
    )
    assert survivor.polarity == "negative", (
        f"Surviving playbook must have polarity='negative'; got {survivor.polarity!r}"
    )
    assert survivor.content.startswith("Avoid"), (
        "Surviving playbook content must preserve its Avoid-prefix end-to-end; "
        f"got: {survivor.content!r}"
    )
