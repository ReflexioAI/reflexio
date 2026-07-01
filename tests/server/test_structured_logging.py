from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from reflexio.server.api import create_app
from reflexio.server.structured_logging import install_structured_logging


@pytest.fixture(autouse=True)
def cleanup_structured_handlers() -> Iterator[None]:
    yield
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if getattr(handler, "_reflexio_structured_logging", False):
            root_logger.removeHandler(handler)
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
        logger = logging.getLogger("tests.structured_logging.capture")
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
    assert exception_item["logger_name"] == "tests.structured_logging.capture"
    assert "RuntimeError: llm extraction failed" in exception_item["exception_text"]
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
        exception_item["timestamp"],
    )


def test_logs_api_defaults_to_error_and_critical(
    tmp_path, monkeypatch, quiet_data_plane
) -> None:
    with _logs_client(tmp_path, monkeypatch, quiet_data_plane) as client:
        logger = logging.getLogger("tests.structured_logging.levels")
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
        logger = logging.getLogger("tests.structured_logging.search")
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

    logging.getLogger("tests.structured_logging.fail_open").error(
        "this insert should fail open"
    )

    handle.close()


def test_structured_logging_reused_handler_survives_first_handle_close(
    tmp_path,
) -> None:
    db_path = str(tmp_path / "reflexio.db")
    first = install_structured_logging(db_path)
    second = install_structured_logging(db_path)
    logger = logging.getLogger("tests.structured_logging.reused")

    first.close()
    logger.error("still live after first close")
    rows = second.query(levels={"error"}, since=None, q=None, limit=10)
    second.close()

    assert any(row.message == "still live after first close" for row in rows)
    assert not any(
        getattr(handler, "_reflexio_structured_logging", False)
        for handler in logging.getLogger().handlers
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

    root_logger = logging.getLogger()
    assert not any(
        getattr(handler, "_reflexio_structured_logging", False)
        for handler in root_logger.handlers
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
    assert handle.store.conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

    logger = logging.getLogger("tests.structured_logging.retention")
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
