"""Fixtures for the golden-set comparison harness.

Parametrizes tests over every YAML file in ``golden_set/extraction`` or
``golden_set/search``. The ``judge`` fixture returns a stubbed ``LLMJudge``
by default; set ``REFLEXIO_EVAL_REAL_JUDGE=1`` with a real Anthropic key to
hit the live judge model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from tests.eval.judge import JudgeScore, LLMJudge

_GOLDEN = Path(__file__).parent / "golden_set"
_RUBRICS = Path(__file__).parent / "judge_prompts"


def _load(kind: str) -> list[dict[str, Any]]:
    """Load every YAML golden file under ``golden_set/<kind>/`` sorted by id."""
    return [
        yaml.safe_load(p.read_text()) for p in sorted((_GOLDEN / kind).glob("*.yaml"))
    ]


def pytest_generate_tests(metafunc):
    """Parametrize over every golden case for tests that ask for one."""
    if "extraction_case" in metafunc.fixturenames:
        cases = _load("extraction")
        metafunc.parametrize("extraction_case", cases, ids=[c["id"] for c in cases])
    if "search_case" in metafunc.fixturenames:
        cases = _load("search")
        metafunc.parametrize("search_case", cases, ids=[c["id"] for c in cases])


def _stubbed_judge(rubric: dict[str, Any]) -> LLMJudge:
    client = MagicMock()
    client.generate_chat_response.return_value = JudgeScore(
        signal_f1=0.5,
        answer_correctness=0.5,
        grounded_rate=1.0,
        rationale="stub",
    )
    return LLMJudge(client=client, rubric=rubric)


def _real_judge(rubric: dict[str, Any]) -> LLMJudge:
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(
        LiteLLMConfig(model=rubric.get("judge_model", "claude-sonnet-4-6"))
    )
    return LLMJudge(client=client, rubric=rubric)


def _load_rubric(name: str) -> dict[str, Any]:
    return yaml.safe_load((_RUBRICS / name).read_text())


@pytest.fixture
def extraction_judge() -> LLMJudge:
    """Judge loaded with the extraction rubric.

    Set ``REFLEXIO_EVAL_REAL_JUDGE=1`` to hit a real LLM; the default path
    stubs the client so the harness smoke-runs without credentials.
    """
    rubric = _load_rubric("extraction_rubric.yaml")
    if os.environ.get("REFLEXIO_EVAL_REAL_JUDGE") == "1":
        return _real_judge(rubric)
    return _stubbed_judge(rubric)


@pytest.fixture
def search_judge() -> LLMJudge:
    """Judge loaded with the search rubric (stubbed by default)."""
    rubric = _load_rubric("search_rubric.yaml")
    if os.environ.get("REFLEXIO_EVAL_REAL_JUDGE") == "1":
        return _real_judge(rubric)
    return _stubbed_judge(rubric)
