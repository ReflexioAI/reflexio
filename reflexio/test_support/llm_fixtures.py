"""Load recorded LLM response fixtures for replay-based tests.

Fixture files live in ``tests/fixtures/llm/<name>.json`` and contain the
full mock response structure matching ``litellm.completion`` output.

Usage::

    from reflexio.test_support.llm_fixtures import load_llm_fixture

    def test_with_recorded_response(monkeypatch):
        fixture = load_llm_fixture("profile_extraction")
        monkeypatch.setattr("litellm.completion", lambda **kw: fixture)
        # ... exercise code under test ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# reflexio/test_support/ -> repo root -> tests/fixtures/llm/
_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "llm"


def load_llm_fixture(name: str) -> MagicMock:
    """Load a recorded LLM response fixture and return it as a MagicMock.

    Args:
        name: Fixture name without extension (e.g. ``"profile_extraction"``).

    Returns:
        MagicMock mimicking a ``litellm.completion`` response.
    """
    fixture_path = _FIXTURE_DIR / f"{name}.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"LLM fixture '{name}' not found at {fixture_path}")
    data: dict[str, Any] = json.loads(fixture_path.read_text())

    choice = data["choices"][0]
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = choice["message"]["content"]
    mock.choices[0].finish_reason = choice.get("finish_reason", "stop")
    return mock


def load_llm_fixture_content(name: str) -> str:
    """Load just the content string from a recorded LLM fixture.

    Args:
        name: Fixture name without extension.

    Returns:
        The raw content string from the fixture's first choice.
    """
    fixture_path = _FIXTURE_DIR / f"{name}.json"
    if not fixture_path.exists():
        raise FileNotFoundError(f"LLM fixture '{name}' not found at {fixture_path}")
    data: dict[str, Any] = json.loads(fixture_path.read_text())
    return data["choices"][0]["message"]["content"]
