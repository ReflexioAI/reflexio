"""Tests for the ``/api/get_injection_stats`` endpoint."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from reflexio.models.api_schema.retriever_schema import (
    GetInjectionStatsResponse,
    InjectionStat,
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


def _make_injection_stat(
    entity_type: str = "user_playbook",
    entity_id: str = "42",
    surfaced_count: int = 5,
    distinct_session_count: int = 2,
    total_prompt_tokens: int = 100,
) -> InjectionStat:
    return InjectionStat(
        entity_type=entity_type,
        entity_id=entity_id,
        surfaced_count=surfaced_count,
        distinct_session_count=distinct_session_count,
        total_prompt_tokens=total_prompt_tokens,
        first_injected_at=1_700_000_000,
        last_injected_at=1_700_000_100,
        last_session_id="s1",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_get_injection_stats_returns_aggregated_list():
    """Endpoint surfaces the lib method's stats list as-is."""
    stats = [_make_injection_stat(), _make_injection_stat(entity_id="99")]
    response = MagicMock(spec=GetInjectionStatsResponse)
    response.success = True
    response.stats = stats
    response.msg = "OK"
    with _patch_lib_method("get_injection_stats", response):
        resp = _client().post(
            "/api/get_injection_stats", json={"days_back": 30}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert len(body["stats"]) == 2
    assert body["stats"][0]["entity_id"] == "42"
    assert body["stats"][1]["entity_id"] == "99"


def test_get_injection_stats_empty_list_is_ok():
    """An empty stats list serialises as an empty array (not null)."""
    response = MagicMock(spec=GetInjectionStatsResponse)
    response.success = True
    response.stats = []
    response.msg = "OK"
    with _patch_lib_method("get_injection_stats", response):
        resp = _client().post(
            "/api/get_injection_stats", json={"days_back": 30}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["stats"] == []


def test_get_injection_stats_with_storage_not_configured_msg():
    """``msg`` field carries through when storage is unconfigured."""
    response = MagicMock(spec=GetInjectionStatsResponse)
    response.success = True
    response.stats = []
    response.msg = "Storage not configured"
    with _patch_lib_method("get_injection_stats", response):
        resp = _client().post(
            "/api/get_injection_stats", json={"days_back": 30}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["msg"] == "Storage not configured"


def test_get_injection_stats_failure_response():
    """A failed lib call surfaces a non-200 body with ``success=false``."""
    response = MagicMock(spec=GetInjectionStatsResponse)
    response.success = False
    response.stats = []
    response.msg = "boom"
    with _patch_lib_method("get_injection_stats", response):
        resp = _client().post(
            "/api/get_injection_stats", json={"days_back": 30}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["msg"] == "boom"
    assert body["stats"] == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_get_injection_stats_rejects_zero_days_back():
    """``days_back`` must be > 0 (Pydantic ``gt=0``)."""
    resp = _client().post("/api/get_injection_stats", json={"days_back": 0})
    assert resp.status_code == 422


def test_get_injection_stats_rejects_negative_days_back():
    """``days_back`` must be > 0 (Pydantic ``gt=0``)."""
    resp = _client().post("/api/get_injection_stats", json={"days_back": -1})
    assert resp.status_code == 422


def test_get_injection_stats_uses_default_days_back():
    """Omitting ``days_back`` falls back to the schema default (30)."""
    response = MagicMock(spec=GetInjectionStatsResponse)
    response.success = True
    response.stats = []
    response.msg = "OK"
    with _patch_lib_method("get_injection_stats", response) as mock_reflexio:
        resp = _client().post("/api/get_injection_stats", json={})
    assert resp.status_code == 200
    # Verify the lib was called with the default days_back.
    mock_reflexio.get_injection_stats.assert_called_once()
    call_arg = mock_reflexio.get_injection_stats.call_args.args[0]
    assert call_arg.days_back == 30
