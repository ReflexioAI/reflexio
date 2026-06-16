"""Tests for the ``/api/get_memory_review`` endpoint."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from reflexio.models.api_schema.retriever_schema import (
    GetMemoryReviewResponse,
    MemoryReviewCandidate,
)
from reflexio.server.api import create_app


def _client() -> TestClient:
    app = create_app(get_org_id=lambda: "test-org")
    return TestClient(app, raise_server_exceptions=False)


@contextmanager
def _patch_lib_method(method_name: str, return_value: MagicMock):
    """Patch ``get_reflexio`` so the lib ``method_name`` returns ``return_value``."""
    mock_reflexio = MagicMock()
    getattr(mock_reflexio, method_name).return_value = return_value
    mock_reflexio.request_context.configurator.get_config.return_value = None
    with patch("reflexio.server.api.get_reflexio", return_value=mock_reflexio):
        yield mock_reflexio


def _make_candidate(
    entity_id: str = "42",
    signals: list[str] | None = None,
    score: int = 75,
    injection_count: int = 0,
    citation_count: int = 0,
) -> MemoryReviewCandidate:
    return MemoryReviewCandidate(
        entity_type="user_playbook",
        entity_id=entity_id,
        title="rule-42",
        signals=signals or ["stale"],
        score=score,
        injection_count=injection_count,
        citation_count=citation_count,
        last_injected_at=None,
        last_cited_at=None,
        last_modified_at=1_700_000_000,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_get_memory_review_returns_candidates_list():
    """Endpoint surfaces the lib method's candidates list as-is."""
    candidates = [
        _make_candidate(entity_id="42", signals=["stale"], score=80),
        _make_candidate(
            entity_id="99",
            signals=["high_cost_low_cite"],
            score=40,
            injection_count=5,
            citation_count=1,
        ),
    ]
    response = MagicMock(spec=GetMemoryReviewResponse)
    response.success = True
    response.candidates = candidates
    response.msg = "OK"
    with _patch_lib_method("get_memory_review", response):
        resp = _client().post(
            "/api/get_memory_review", json={"days_back": 60}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert len(body["candidates"]) == 2
    assert body["candidates"][0]["entity_id"] == "42"
    assert body["candidates"][0]["signals"] == ["stale"]
    assert body["candidates"][1]["signals"] == ["high_cost_low_cite"]


def test_get_memory_review_empty_list_is_ok():
    """An empty candidates list serialises as an empty array (not null)."""
    response = MagicMock(spec=GetMemoryReviewResponse)
    response.success = True
    response.candidates = []
    response.msg = "OK"
    with _patch_lib_method("get_memory_review", response):
        resp = _client().post(
            "/api/get_memory_review", json={"days_back": 60}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["candidates"] == []


def test_get_memory_review_failure_response():
    """A failed lib call surfaces ``success=false`` with the error message."""
    response = MagicMock(spec=GetMemoryReviewResponse)
    response.success = False
    response.candidates = []
    response.msg = "boom"
    with _patch_lib_method("get_memory_review", response):
        resp = _client().post(
            "/api/get_memory_review", json={"days_back": 60}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["msg"] == "boom"
    assert body["candidates"] == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_get_memory_review_rejects_zero_days_back():
    """``days_back`` must be > 0 (Pydantic ``gt=0``)."""
    resp = _client().post("/api/get_memory_review", json={"days_back": 0})
    assert resp.status_code == 422


def test_get_memory_review_rejects_negative_days_back():
    """``days_back`` must be > 0."""
    resp = _client().post("/api/get_memory_review", json={"days_back": -1})
    assert resp.status_code == 422


def test_get_memory_review_rejects_invalid_signal_filter():
    """``signal_filter`` must use the ``Literal`` enum values."""
    resp = _client().post(
        "/api/get_memory_review",
        json={"days_back": 60, "signal_filter": ["bogus_signal"]},
    )
    assert resp.status_code == 422


def test_get_memory_review_uses_default_days_back():
    """Omitting ``days_back`` falls back to the schema default (60)."""
    response = MagicMock(spec=GetMemoryReviewResponse)
    response.success = True
    response.candidates = []
    response.msg = "OK"
    with _patch_lib_method("get_memory_review", response) as mock_reflexio:
        resp = _client().post("/api/get_memory_review", json={})
    assert resp.status_code == 200
    call_arg = mock_reflexio.get_memory_review.call_args.args[0]
    assert call_arg.days_back == 60
    assert call_arg.signal_filter is None


def test_get_memory_review_forwards_signal_filter():
    """``signal_filter`` is forwarded to the lib method."""
    response = MagicMock(spec=GetMemoryReviewResponse)
    response.success = True
    response.candidates = []
    response.msg = "OK"
    with _patch_lib_method("get_memory_review", response) as mock_reflexio:
        resp = _client().post(
            "/api/get_memory_review",
            json={"days_back": 60, "signal_filter": ["stale"]},
        )
    assert resp.status_code == 200
    call_arg = mock_reflexio.get_memory_review.call_args.args[0]
    assert call_arg.signal_filter == ["stale"]
