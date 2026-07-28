from __future__ import annotations

import json
import re
from pathlib import Path

from reflexio.server.prompt.prompt_manager import PromptManager

_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "test_data"
    / "playbook_generalization_manifest.json"
)


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _load_cases() -> list[dict[str, object]]:
    payload = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    return payload["cases"]


def test_generalization_manifest_has_cross_domain_paired_coverage() -> None:
    cases = _load_cases()
    assert len(cases) >= 20
    assert len({case["domain"] for case in cases}) >= 10
    assert len({case["synthetic_org"] for case in cases}) >= 10
    assert {case["expectation"] for case in cases} == {
        "must_capture",
        "must_reject",
    }

    positive_families = {
        case["family"] for case in cases if case["expectation"] == "must_capture"
    }
    assert {
        "correction",
        "preference",
        "ignored_answer",
        "non_delivery",
        "duplicate",
        "bounded_retry",
        "ask_once_act",
        "verified_success",
        "rejected_approach",
    } <= positive_families


def test_active_playbook_prompts_do_not_contain_manifest_phrases() -> None:
    manager = PromptManager()
    context = manager.render_prompt(
        "playbook_extraction_context",
        {
            "agent_context_prompt": "agent",
            "extraction_definition_prompt": "definition",
            "tool_can_use": "tools",
        },
    )
    main = manager.render_prompt(
        "playbook_extraction_main",
        {"interactions": "conversation"},
    )
    reviewer = manager.render_prompt(
        "playbook_candidate_review",
        {
            "agent_context_prompt": "agent",
            "playbook_definition": "definition",
            "tool_context": "tools",
            "interaction_context": "conversation",
            "artifact_availability": "unknown",
            "candidates": "candidates",
            "existing_playbooks": "existing",
        },
    )
    active_prompts = _normalized(f"{context}\n{main}\n{reviewer}")

    leaked = [
        str(case["id"])
        for case in _load_cases()
        if _normalized(str(case["leakage_probe"])) in active_prompts
    ]
    assert not leaked, f"Evaluation phrases leaked into active prompts: {leaked}"


def test_manifest_expectations_are_complete_and_grounded() -> None:
    for case in _load_cases():
        turns = case["turns"]
        assert isinstance(turns, list) and turns
        assert all(
            isinstance(turn, dict)
            and turn.get("role") in {"user", "agent"}
            and isinstance(turn.get("content"), str)
            and turn["content"].strip()
            for turn in turns
        )
        if case["expectation"] == "must_capture":
            assert case["expected_rule"]
            assert case["earliest_trigger"]
        else:
            assert case["expected_rule"] is None
            assert case["earliest_trigger"] is None
