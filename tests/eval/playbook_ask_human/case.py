"""Case schema for the playbook ask_human invocation eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class AskHumanTurn(BaseModel):
    """One natural conversation turn in an eval case."""

    role: str
    content: str


class PriorPendingToolCall(BaseModel):
    """A prior pending ask_human request available as Prior Knowledge."""

    pending_tool_call_id: str
    question_text: str
    answer_format: str | None = None
    tags: list[str] = Field(default_factory=list)


class PlaybookAskHumanCase(BaseModel):
    """One labeled ask_human decision-boundary case."""

    id: str
    vertical: str
    description: str
    agent_context_prompt: str = ""
    extraction_definition_prompt: str = ""
    tool_can_use: list[str] = Field(default_factory=list)
    sessions: list[AskHumanTurn]
    prior_pending_tool_calls: list[PriorPendingToolCall] = Field(default_factory=list)
    expected_ask_human: bool
    expected_playbooks_needed: bool
    expected_question_must_include: list[str] = Field(default_factory=list)
    label_rationale: str

    @model_validator(mode="after")
    def _validate_label_shape(self) -> PlaybookAskHumanCase:
        if self.expected_ask_human and not self.expected_playbooks_needed:
            raise ValueError("expected_ask_human requires expected_playbooks_needed")
        if self.expected_ask_human and not self.expected_question_must_include:
            raise ValueError(
                "expected_ask_human cases must include expected_question_must_include"
            )
        return self


def _case_dicts_from_yaml(path: Path) -> list[dict[str, Any]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return []
    if isinstance(loaded, list):
        return loaded
    if isinstance(loaded, dict) and isinstance(loaded.get("cases"), list):
        return loaded["cases"]
    if isinstance(loaded, dict):
        return [loaded]
    raise ValueError(f"Unsupported YAML shape in {path}")


def load_cases(path: Path | None = None) -> list[PlaybookAskHumanCase]:
    """Load ask_human eval cases from a file or directory.

    A directory may contain one YAML file with ``cases: [...]`` or many YAML
    files with one case each. Cases are sorted by id for stable parametrization.
    """

    if path is None:
        path = Path(__file__).parents[1] / "golden_set" / "playbook_ask_human"

    files = [path] if path.is_file() else sorted(path.glob("*.yaml"))
    cases: list[PlaybookAskHumanCase] = []
    for yaml_path in files:
        cases.extend(
            PlaybookAskHumanCase.model_validate(item)
            for item in _case_dicts_from_yaml(yaml_path)
        )

    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate ask_human eval case id(s): {duplicates}")
    return sorted(cases, key=lambda case: case.id)
