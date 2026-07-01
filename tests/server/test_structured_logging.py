from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from reflexio.server.api import create_app
from reflexio.server.structured_logging import install_structured_logging

_TEST_LOGGER_PREFIX = "reflexio.tests.structured_logging"


@pytest.fixture(autouse=True)
def cleanup_structured_handlers() -> Iterator[None]:
    yield
    reflexio_logger = logging.getLogger("reflexio")
    for handler in list(reflexio_logger.handlers):
        if getattr(handler, "_reflexio_structured_logging", False):
            reflexio_logger.removeHandler(handler)
            handler.close()


@pytest.fixture
def quiet_data_plane(monkeypatch):
    import reflexio.server.llm.model_defaults as model_defaults
    import reflexio.server.llm.rerank as rerank
    import reflexio.server.services.extraction.resume_scheduler as resume_scheduler
    import reflexio.server.services.lineage.gc_scheduler as gc_scheduler

    monkeypatch.setattr(model_defaults, "validate_llm_availability", lambda: None)
    monkeypatch.setattr(rerank, "prewarm", lambda: None)
    monkeypatch.setattr(
        resume_scheduler, "maybe_start_resume_scheduler", lambda *_, **__: None
    )
    monkeypatch.setattr(gc_scheduler, "maybe_start_lineage_gc", lambda *_, **__: None)


def _logs_client(tmp_path, monkeypatch, quiet_data_plane) -> TestClient:
    monkeypatch.setattr("reflexio.server.LOCAL_STORAGE_PATH", str(tmp_path))
    app = create_app(get_org_id=lambda: "logs-test-org")
    return TestClient(app, raise_server_exceptions=True)


def test_logs_api_captures_warning_error_critical_and_exception(
    tmp_path, monkeypatch, quiet_data_plane
) -> None:
    with _logs_client(tmp_path, monkeypatch, quiet_data_plane) as client:
        logger = logging.getLogger(f"{_TEST_LOGGER_PREFIX}.capture")
        logger.warning("warning from %s", "structured logs")
        try:
            raise RuntimeError("llm extraction failed")
        except RuntimeError:
            logger.exception("error with traceback")
        logger.critical("critical failure")

        response = client.get(
            "/api/logs",
            params={"levels": "warning,error,critical", "since": "1h"},
        )

    assert response.status_code == 200
    body = response.json()
    messages = [item["message"] for item in body["items"]]
    assert "warning from structured logs" in messages
    assert "error with traceback" in messages
    assert "critical failure" in messages

    exception_item = next(
        item for item in body["items"] if item["message"] == "error with traceback"
    )
    assert exception_item["level"] == "error"
    assert exception_item["logger_name"] == f"{_TEST_LOGGER_PREFIX}.capture"
    assert "RuntimeError: llm extraction failed" in exception_item["exception_text"]
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
        exception_item["timestamp"],
    )


def test_logs_api_defaults_to_error_and_critical(
    tmp_path, monkeypatch, quiet_data_plane
) -> None:
    with _logs_client(tmp_path, monkeypatch, quiet_data_plane) as client:
        logger = logging.getLogger(f"{_TEST_LOGGER_PREFIX}.levels")
        logger.warning("hidden warning")
        logger.error("visible error")
        logger.critical("visible critical")

        response = client.get("/api/logs")

    assert response.status_code == 200
    messages = [item["message"] for item in response.json()["items"]]
    assert "hidden warning" not in messages
    assert "visible error" in messages
    assert "visible critical" in messages


def test_logs_api_validates_filters(tmp_path, monkeypatch, quiet_data_plane) -> None:
    with _logs_client(tmp_path, monkeypatch, quiet_data_plane) as client:
        assert client.get("/api/logs", params={"levels": "info"}).status_code == 400
        assert client.get("/api/logs", params={"levels": ""}).status_code == 400
        assert client.get("/api/logs", params={"since": "soon"}).status_code == 400
        assert client.get("/api/logs", params={"limit": "0"}).status_code == 400
        assert client.get("/api/logs", params={"limit": "1001"}).status_code == 400


def test_logs_api_search_since_iso_and_limit(
    tmp_path, monkeypatch, quiet_data_plane
) -> None:
    with _logs_client(tmp_path, monkeypatch, quiet_data_plane) as client:
        logger = logging.getLogger(f"{_TEST_LOGGER_PREFIX}.search")
        logger.error("openai timeout alpha")
        logger.error("anthropic timeout beta")
        logger.error("storage issue")

        since = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        response = client.get(
            "/api/logs",
            params={"q": "timeout", "since": since, "limit": "1"},
        )

        future = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
        future_response = client.get("/api/logs", params={"since": future})

    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 1
    assert len(body["items"]) == 1
    assert "timeout" in body["items"][0]["message"]
    assert future_response.status_code == 200
    assert future_response.json()["items"] == []


def test_logs_api_search_treats_percent_and_underscore_as_literals(
    tmp_path, monkeypatch, quiet_data_plane
) -> None:
    with _logs_client(tmp_path, monkeypatch, quiet_data_plane) as client:
        logger = logging.getLogger(f"{_TEST_LOGGER_PREFIX}.escape")
        logger.error("battery at 100% capacity")
        logger.error("battery at 100 percent capacity")
        logger.error("token foo_bar rejected")
        logger.error("token fooXbar rejected")

        percent_response = client.get("/api/logs", params={"q": "100%"})
        underscore_response = client.get("/api/logs", params={"q": "foo_bar"})

    assert percent_response.status_code == 200
    percent_messages = [item["message"] for item in percent_response.json()["items"]]
    assert percent_messages == ["battery at 100% capacity"]

    assert underscore_response.status_code == 200
    underscore_messages = [
        item["message"] for item in underscore_response.json()["items"]
    ]
    assert underscore_messages == ["token foo_bar rejected"]


def test_logs_api_search_wildcards_still_compose_with_filters(
    tmp_path, monkeypatch, quiet_data_plane
) -> None:
    with _logs_client(tmp_path, monkeypatch, quiet_data_plane) as client:
        logger = logging.getLogger(f"{_TEST_LOGGER_PREFIX}.escape_filters")
        logger.warning("disk at 50%_full warning")
        logger.error("disk at 50%_full error")

        since = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        response = client.get(
            "/api/logs",
            params={"q": "50%_full", "since": since, "levels": "error"},
        )

    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 1
    assert items[0]["level"] == "error"
    assert items[0]["message"] == "disk at 50%_full error"


def test_logs_api_search_length_is_bounded(
    tmp_path, monkeypatch, quiet_data_plane
) -> None:
    with _logs_client(tmp_path, monkeypatch, quiet_data_plane) as client:
        response = client.get("/api/logs", params={"q": "x" * 257})

    assert response.status_code == 422


def test_logs_api_not_mounted_when_data_plane_disabled() -> None:
    app = create_app(mount_data_plane=False)
    paths = {getattr(route, "path", None) for route in app.routes}

    assert "/api/logs" not in paths


def test_logs_api_uses_org_dependency_for_auth(
    tmp_path, monkeypatch, quiet_data_plane
) -> None:
    monkeypatch.setattr("reflexio.server.LOCAL_STORAGE_PATH", str(tmp_path))

    def reject_org() -> str:
        raise HTTPException(status_code=401, detail="missing token")

    app = create_app(get_org_id=reject_org, require_auth=True)
    with TestClient(app, raise_server_exceptions=True) as client:
        response = client.get("/api/logs")

    assert response.status_code == 401
    assert response.json() == {"detail": "missing token"}


def test_structured_handler_runtime_insert_failure_is_fail_open(tmp_path) -> None:
    handle = install_structured_logging(str(tmp_path / "reflexio.db"))
    handle.store.conn.close()

    logging.getLogger(f"{_TEST_LOGGER_PREFIX}.fail_open").error(
        "this insert should fail open"
    )

    handle.close()


def test_structured_handler_ignores_third_party_loggers(tmp_path) -> None:
    handle = install_structured_logging(str(tmp_path / "reflexio.db"))

    logging.getLogger("httpx").warning("third party warning")
    logging.getLogger(f"{_TEST_LOGGER_PREFIX}.first_party").warning(
        "first party warning"
    )
    rows = handle.query(
        levels={"warning"},
        since=None,
        q=None,
        limit=10,
    )
    handle.close()

    messages = [row.message for row in rows]
    assert "third party warning" not in messages
    assert "first party warning" in messages


def test_structured_handler_lock_contention_fails_open_quickly(tmp_path) -> None:
    db_path = tmp_path / "reflexio.db"
    handle = install_structured_logging(str(db_path))
    blocker = sqlite3.connect(db_path)
    blocker.execute("BEGIN IMMEDIATE")

    started = time.monotonic()
    logging.getLogger(f"{_TEST_LOGGER_PREFIX}.lock").error("locked write")
    elapsed = time.monotonic() - started

    blocker.rollback()
    blocker.close()
    handle.close()

    assert elapsed < 1


def test_structured_logging_reused_handler_survives_first_handle_close(
    tmp_path,
) -> None:
    db_path = str(tmp_path / "reflexio.db")
    first = install_structured_logging(db_path)
    second = install_structured_logging(db_path)
    logger = logging.getLogger(f"{_TEST_LOGGER_PREFIX}.reused")

    first.close()
    logger.error("still live after first close")
    rows = second.query(levels={"error"}, since=None, q=None, limit=10)
    second.close()

    assert any(row.message == "still live after first close" for row in rows)
    reflexio_logger = logging.getLogger("reflexio")
    assert not any(
        getattr(handler, "_reflexio_structured_logging", False)
        for handler in reflexio_logger.handlers
    )


def test_data_plane_startup_failure_logs_and_closes_handler(tmp_path, monkeypatch):
    import reflexio.server.llm.model_defaults as model_defaults

    monkeypatch.setattr("reflexio.server.LOCAL_STORAGE_PATH", str(tmp_path))

    def fail_validation() -> None:
        raise RuntimeError("llm provider missing")

    monkeypatch.setattr(model_defaults, "validate_llm_availability", fail_validation)
    app = create_app(get_org_id=lambda: "logs-test-org")

    with pytest.raises(RuntimeError, match="llm provider missing"), TestClient(app):
        pass

    reflexio_logger = logging.getLogger("reflexio")
    assert not any(
        getattr(handler, "_reflexio_structured_logging", False)
        for handler in reflexio_logger.handlers
    )

    with sqlite3.connect(tmp_path / "reflexio.db") as conn:
        rows = conn.execute(
            "SELECT message, exception_text FROM structured_log_events"
        ).fetchall()

    assert any(row[0] == "Data-plane startup failed" for row in rows)
    assert any("RuntimeError: llm provider missing" in row[1] for row in rows)


def test_structured_logging_applies_caps_and_retention(tmp_path) -> None:
    handle = install_structured_logging(str(tmp_path / "reflexio.db"), row_limit=2)
    assert handle.store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert handle.store.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert handle.store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 100

    logger = logging.getLogger(f"{_TEST_LOGGER_PREFIX}.retention")
    logger.error("old")
    logger.error("middle")
    logger.error("new %s", "x" * 9000)
    handle.store.enforce_retention()

    with sqlite3.connect(tmp_path / "reflexio.db") as conn:
        rows = conn.execute(
            "SELECT message FROM structured_log_events ORDER BY id ASC"
        ).fetchall()

    handle.close()

    assert len(rows) == 2
    assert rows[0][0] == "middle"
    assert rows[1][0].startswith("new ")
    assert len(rows[1][0]) <= 8192
