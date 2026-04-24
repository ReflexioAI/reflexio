"""Hand-crafted eval runner for agentic-v2. See spec §11."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

_THIS_DIR = Path(__file__).resolve().parent
FIXTURES_ROOT = _THIS_DIR / "eval_fixtures"


def load_fixtures(group: str | None = None) -> list[dict[str, Any]]:
    """Load all fixture JSONs under eval_fixtures/, optionally scoped to one group.

    Args:
        group (str | None): Optional group subdirectory name (e.g. "group1_mutation").
            When None, all fixtures from all groups are returned.

    Returns:
        list[dict[str, Any]]: Parsed fixture dicts sorted by path.
    """
    root = FIXTURES_ROOT if group is None else FIXTURES_ROOT / group
    return [json.loads(p.read_text()) for p in sorted(root.rglob("*.json"))]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_plan(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score an actual plan against an expected plan spec.

    Supports exact-match fields (``id``, ``ttl``) and fuzzy assertions:
    ``content_contains``, ``content_preserves_all``, ``trigger_contains``.

    Args:
        actual (list[dict[str, Any]]): Ops produced by the agent.
        expected (list[dict[str, Any]]): Spec ops from the fixture's
            ``expected_plan`` list. Each entry may contain fuzzy keys instead
            of (or alongside) exact-match keys.

    Returns:
        dict[str, Any]: ``{"semantic_match": bool, "failures": list[str]}``.
            ``semantic_match`` is ``True`` when every expected op is satisfied.
    """
    failures: list[str] = []

    if len(actual) != len(expected):
        failures.append(
            f"op count mismatch: actual={len(actual)} expected={len(expected)}"
        )
        return {"semantic_match": False, "failures": failures}

    semantic = True
    for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
        if a.get("op") != e.get("op"):
            failures.append(
                f"op[{i}]: type mismatch — actual={a.get('op')!r} expected={e.get('op')!r}"
            )
            semantic = False
            continue

        # Exact-match fields
        for field in ("id", "ttl"):
            if field in e and a.get(field) != e[field]:
                failures.append(
                    f"op[{i}].{field}: actual={a.get(field)!r} expected={e[field]!r}"
                )
                semantic = False

        # Fuzzy: content_contains
        content_lower = (a.get("content") or "").lower()
        for substr in e.get("content_contains", []):
            if substr.lower() not in content_lower:
                failures.append(f"op[{i}]: content missing substring {substr!r}")
                semantic = False

        # Fuzzy: content_preserves_all (lossless merge check)
        for preserved in e.get("content_preserves_all", []):
            if preserved.lower() not in content_lower:
                failures.append(f"op[{i}]: lost preserved content {preserved!r}")
                semantic = False

        # Fuzzy: trigger_contains
        trigger_lower = (a.get("trigger") or "").lower()
        for substr in e.get("trigger_contains", []):
            if substr.lower() not in trigger_lower:
                failures.append(f"op[{i}]: trigger missing substring {substr!r}")
                semantic = False

    return {"semantic_match": semantic, "failures": failures}


def score_group3_fixture(
    fixture: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Score a group3 loop-behavior fixture against the run_fixture result.

    Checks outcome, applied_count, and that expected violation codes are a
    subset of observed codes.

    Args:
        fixture (dict[str, Any]): The group3 fixture dict.
        result (dict[str, Any]): Return value from :func:`run_fixture`.

    Returns:
        dict[str, Any]: ``{"pass": bool, "failures": list[str]}``.
    """
    failures: list[str] = []

    expected_outcome = fixture.get("expected_outcome")
    if result.get("outcome") != expected_outcome:
        failures.append(
            f"outcome mismatch: actual={result.get('outcome')!r} expected={expected_outcome!r}"
        )

    expected_count = fixture.get("expected_applied_count")
    if result.get("applied_count") != expected_count:
        failures.append(
            f"applied_count mismatch: actual={result.get('applied_count')} expected={expected_count}"
        )

    expected_violations: set[str] = set(fixture.get("expected_violations", []))
    actual_violations: set[str] = set(result.get("violation_codes", []))
    missing = expected_violations - actual_violations
    if missing:
        failures.append(f"missing expected violation codes: {sorted(missing)}")

    return {"pass": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# Storage seeding
# ---------------------------------------------------------------------------


def seed_storage(fixture: dict[str, Any], storage: Any, user_id: str) -> None:
    """Write ``fixture["existing_storage"]`` entries into the given storage.

    Translates each entry into the appropriate entity and writes it via the
    storage API. Supports ``profile``, ``user_playbook``, and
    ``agent_playbook`` entry types. Unknown types are skipped with a warning.

    Args:
        fixture (dict[str, Any]): Fixture dict (may contain ``existing_storage``).
        storage: A storage instance (e.g. SQLiteStorage).
        user_id (str): User ID to assign to profile and user_playbook rows.
    """
    from reflexio.models.api_schema.common import NEVER_EXPIRES_TIMESTAMP
    from reflexio.models.api_schema.domain.entities import (
        AgentPlaybook,
        UserPlaybook,
        UserProfile,
    )
    from reflexio.models.api_schema.domain.enums import ProfileTimeToLive

    for entry in fixture.get("existing_storage", []):
        entry_type = entry.get("type")

        if entry_type == "profile":
            ttl_str = entry.get("ttl", "infinity")
            try:
                ttl = ProfileTimeToLive(ttl_str)
            except ValueError:
                ttl = ProfileTimeToLive.INFINITY

            profile = UserProfile(
                profile_id=entry.get("id", str(uuid.uuid4())),
                user_id=user_id,
                content=entry.get("content", ""),
                last_modified_timestamp=0,
                generated_from_request_id="eval_seed",
                profile_time_to_live=ttl,
                expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
                source_span=entry.get("source_span"),
            )
            storage.add_user_profile(user_id, [profile])

        elif entry_type == "user_playbook":
            playbook = UserPlaybook(
                user_id=user_id,
                agent_version="eval_v1",
                request_id="eval_seed",
                playbook_name=entry.get("playbook_name", "eval"),
                content=entry.get("content", ""),
                trigger=entry.get("trigger"),
                rationale=entry.get("rationale"),
            )
            storage.save_user_playbooks([playbook])

        elif entry_type == "agent_playbook":
            agent_playbook = AgentPlaybook(
                agent_version="eval_v1",
                playbook_name=entry.get("playbook_name", "eval"),
                content=entry.get("content", ""),
                trigger=entry.get("trigger"),
                rationale=entry.get("rationale"),
            )
            storage.save_agent_playbooks([agent_playbook])

        else:
            import logging

            logging.getLogger(__name__).warning(
                "seed_storage: unknown entry type %r — skipping", entry_type
            )


# ---------------------------------------------------------------------------
# Mocked-LLM response helpers
# ---------------------------------------------------------------------------


def _mk_tool_call(id_: str, name: str, args: dict[str, Any]) -> MagicMock:
    """Build a MagicMock resembling an LLM tool_call object.

    Args:
        id_ (str): Tool call ID string.
        name (str): Tool function name.
        args (dict[str, Any]): Tool arguments (will be JSON-serialised).

    Returns:
        MagicMock: Object with .id, .function.name, .function.arguments.
    """
    tc = MagicMock()
    tc.id = id_
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _mk_resp(tool_calls_spec: list[dict[str, Any]]) -> MagicMock:
    """Build a MagicMock LLM response containing a list of tool calls.

    Args:
        tool_calls_spec (list[dict[str, Any]]): List of ``{"id", "name", "args"}``
            dicts as stored in fixture ``mock_llm_responses[*].tool_calls``.

    Returns:
        MagicMock: Fake LLM response with ``.tool_calls`` and ``.content = None``.
    """
    r = MagicMock()
    r.tool_calls = [
        _mk_tool_call(tc["id"], tc["name"], tc["args"]) for tc in tool_calls_spec
    ]
    r.content = None
    return r


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_fixture(
    fixture: dict[str, Any],
    *,
    client: Any,
    prompt_manager: Any,
    storage: Any,
    user_id: str = "eval_user",
    agent_version: str = "eval_v1",
) -> dict[str, Any]:
    """Execute one eval fixture end-to-end.

    For Group 3 (``group3_loop_behavior``), this method scripts the mocked LLM
    client from ``fixture["mock_llm_responses"]``, seeds storage, and drives
    :class:`ExtractionAgent` to completion.

    For Groups 1 and 2, execution is stubbed — a real LLM or oracle mock is
    required to evaluate those fixtures (out of Task 21 scope).

    Args:
        fixture (dict[str, Any]): Parsed fixture dict from :func:`load_fixtures`.
        client: LiteLLMClient (or MagicMock) — must have
            ``generate_chat_response`` that can be scripted via ``side_effect``.
        prompt_manager: PromptManager instance.
        storage: BaseStorage instance (e.g. SQLiteStorage).
        user_id (str): User ID to use when seeding + running.
        agent_version (str): Agent version string passed to the agent.

    Returns:
        dict[str, Any]: Keys:
            - ``actual_plan`` — list of applied op dicts (empty for stub).
            - ``outcome`` — ``"finish_tool"``, ``"max_steps"``, or ``"skipped"``.
            - ``applied_count`` — number of applied ops.
            - ``violation_codes`` — list of invariant code strings.
            - ``notes`` (optional) — explanation for stubbed groups.
    """
    from reflexio.server.services.extraction.extraction_agent import ExtractionAgent

    seed_storage(fixture, storage, user_id)
    group = fixture.get("group", "")

    if group == "group3_loop_behavior":
        responses = fixture.get("mock_llm_responses", [])
        client.generate_chat_response.side_effect = [
            _mk_resp(r["tool_calls"]) for r in responses
        ]
        agent = ExtractionAgent(
            client=client,
            storage=storage,
            prompt_manager=prompt_manager,
            max_steps=len(responses),
        )
        result = agent.run(
            user_id=user_id,
            agent_version=agent_version,
            extractor_name="eval",
            extraction_criteria="eval",
            sessions_text=fixture.get("session", ""),
        )
        return {
            "actual_plan": [op.model_dump() for op in result.applied],
            "outcome": result.outcome,
            "applied_count": len(result.applied),
            "violation_codes": [v.code for v in result.violations],
        }

    # Group 1 / Group 2 — deferred (requires real LLM or oracle mock)
    return {
        "actual_plan": [],
        "outcome": "skipped",
        "applied_count": 0,
        "violation_codes": [],
        "notes": (
            f"group {group!r} execution requires real LLM or oracle mock"
            " (out of Task 21 scope)"
        ),
    }
