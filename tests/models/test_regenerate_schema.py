"""Unit tests for the RegenerateRequest pydantic model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.eval_overview_schema import RegenerateRequest


def test_valid_request() -> None:
    """Happy path: well-formed window with non-empty evaluator name."""
    r = RegenerateRequest(evaluation_name="overall_success", from_ts=100, to_ts=200)
    assert r.from_ts == 100
    assert r.to_ts == 200
    assert r.evaluation_name == "overall_success"


def test_from_ts_equal_to_to_ts_rejected() -> None:
    """An empty window (from == to) is rejected — no sessions could be in it."""
    with pytest.raises(ValidationError, match="strictly before"):
        RegenerateRequest(evaluation_name="overall", from_ts=100, to_ts=100)


def test_from_ts_after_to_ts_rejected() -> None:
    """A reversed window is rejected so callers don't silently get zero sessions."""
    with pytest.raises(ValidationError, match="strictly before"):
        RegenerateRequest(evaluation_name="overall", from_ts=200, to_ts=100)


def test_empty_evaluation_name_rejected() -> None:
    """An empty evaluator name is rejected via NonEmptyStr."""
    with pytest.raises(ValidationError):
        RegenerateRequest(evaluation_name="", from_ts=0, to_ts=10)


def test_negative_ts_rejected() -> None:
    """Negative timestamps are rejected (ge=0 on the Field)."""
    with pytest.raises(ValidationError):
        RegenerateRequest(evaluation_name="overall", from_ts=-1, to_ts=10)
