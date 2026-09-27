"""``/api/search`` must not answer 200 when storage failed outright.

Sentry PYTHON-FASTAPI-Z0: one dead pooled connection made every storage arm
fail, and the route still answered **HTTP 200** with empty result lists and
``success=False`` in the body alone. A caller that checked the status code -- or
simply read ``profiles``/``user_playbooks`` -- could not tell a backend outage
from "this user has nothing stored", and had nothing to retry on.

``verifying-before-building.md`` catalogues this exact shape: "`assert_status_code
(resp) == 200` on a JSON API -- the app returns `200 {"success": false, ...}`".

The control is ``test_an_empty_but_successful_search_is_still_200``: without it
these would pass against a route that 503'd on every empty result.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.api import create_app


@contextmanager
def _search_returning(result: MagicMock) -> Iterator[MagicMock]:
    reflexio = MagicMock()
    reflexio.request_context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite()
    )
    reflexio.unified_search.return_value = result
    with patch(
        "reflexio.server.routes.search.reflexio_cache.get_reflexio",
        return_value=reflexio,
    ):
        yield reflexio


def _result(
    *,
    success: bool,
    msg: str,
    degraded: bool = False,
    search_mode_effective: str | None = None,
) -> MagicMock:
    return MagicMock(
        success=success,
        profiles=[],
        agent_playbooks=[],
        user_playbooks=[],
        reformulated_query=None,
        msg=msg,
        agent_trace=None,
        rehydrated_text=None,
        degraded=degraded,
        search_mode_effective=search_mode_effective,
    )


def _client() -> TestClient:
    return TestClient(
        create_app(
            get_org_id=lambda: "org-1",
            get_caller_type=lambda: "production_agent",
        ),
        raise_server_exceptions=False,
    )


def test_a_total_storage_failure_is_a_503_not_an_empty_200() -> None:
    with _search_returning(_result(success=False, msg="Search failed")):
        response = _client().post(
            "/api/search", json={"query": "q", "user_id": "user-1"}
        )

    assert response.status_code == 503, (
        "a backend outage answered 200 with empty lists, so a caller could not "
        "distinguish it from an empty corpus"
    )
    assert "Search failed" in response.json()["detail"]


def test_an_empty_but_successful_search_is_still_200() -> None:
    """The control: finding nothing is a valid answer, not an outage."""
    with _search_returning(_result(success=True, msg="OK")):
        response = _client().post(
            "/api/search", json={"query": "q", "user_id": "user-1"}
        )

    assert response.status_code == 200
    assert response.json()["success"] is True


def test_a_degraded_search_is_200_so_partial_results_still_reach_the_caller() -> None:
    """A PARTIAL failure is deliberately not a 503 -- it is a marked 200.

    The status alone is not the contract. Asserting only on it is what let the
    view model drop ``degraded`` entirely: the service set the flag, the route
    built a ``UnifiedSearchViewResponse`` that had no such field, and the
    caller got a plain 200 with empty profiles -- exactly the "an outage looks
    like an empty corpus" bug, one layer further out. So the body is asserted
    on here, not just the code.
    """
    with _search_returning(
        _result(success=True, msg="OK", degraded=True, search_mode_effective="fts")
    ):
        response = _client().post(
            "/api/search", json={"query": "q", "user_id": "user-1"}
        )

    assert response.status_code == 200, "a degraded answer must still be served"
    body = response.json()
    assert body["success"] is True
    assert body["degraded"] is True, (
        "the degradation marker must reach the caller -- a degradation nobody "
        "can see is not a signal"
    )
    assert body["search_mode_effective"] == "fts"


def test_a_healthy_search_reports_degraded_false_rather_than_omitting_it() -> None:
    """The control, and the reason ``degraded`` is a bool rather than optional.

    ``response_model_exclude_none=True`` drops None fields, so an OPTIONAL
    ``degraded`` would vanish on the healthy path and a caller could not tell
    "not degraded" from "this server is too old to say". ``False`` is not None,
    so the key is always present and always safe to read.
    """
    with _search_returning(_result(success=True, msg="OK")):
        response = _client().post(
            "/api/search", json={"query": "q", "user_id": "user-1"}
        )

    body = response.json()
    assert body["degraded"] is False
    assert "search_mode_effective" not in body, (
        "an unset effective mode is omitted, not null -- the requested mode was honored"
    )
