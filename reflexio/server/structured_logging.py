"""SQLite-backed structured logging for OSS operator diagnostics."""

from __future__ import annotations

import logging
import sqlite3
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from reflexio.server.services.configurator.configurator import (
    DefaultConfigurator,
    get_configurator_class,
    resolve_oss_sqlite_db_path,
)

_DDL = """
CREATE TABLE IF NOT EXISTS structured_log_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    logger_name TEXT NOT NULL,
    message TEXT NOT NULL,
    exception_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_structured_log_events_timestamp
    ON structured_log_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_structured_log_events_level
    ON structured_log_events(level);
"""

_VALID_LEVELS = frozenset({"warning", "error", "critical"})
_MESSAGE_MAX_CHARS = 8 * 1024
_EXCEPTION_MAX_CHARS = 32 * 1024
_DEFAULT_ROW_LIMIT = 10_000
_RETENTION_INTERVAL = 100
_BUSY_TIMEOUT_MS = 5_000
_HANDLER_LOCK = threading.RLock()


@dataclass(frozen=True)
class StructuredLogEvent:
    """One structured log event returned by the store query helper."""

    timestamp: str
    level: str
    logger_name: str
    message: str
    exception_text: str | None


class StructuredLogStore:
    """Owns the SQLite table, inserts, retention, and queries for log events."""

    def __init__(
        self,
        db_path: str,
        *,
        row_limit: int = _DEFAULT_ROW_LIMIT,
        busy_timeout_ms: int = _BUSY_TIMEOUT_MS,
    ) -> None:
        if row_limit <= 0:
            raise ValueError("row_limit must be positive")
        self.db_path = db_path
        self.row_limit = row_limit
        self._successful_inserts = 0
        self._lock = threading.RLock()
        self._closed = False
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            timeout=busy_timeout_ms / 1000,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        self.conn.executescript(_DDL)
        self.conn.commit()
        self.enforce_retention()

    def insert(self, record: logging.LogRecord) -> None:
        """Insert one log record.

        Args:
            record: Python logging record emitted at WARNING or above.
        """
        timestamp = _format_timestamp(record.created)
        level = record.levelname.lower()
        if level not in _VALID_LEVELS:
            return
        message = _cap(record.getMessage(), _MESSAGE_MAX_CHARS)
        exception_text = None
        if record.exc_info:
            exception_text = _cap(
                "".join(traceback.format_exception(*record.exc_info)),
                _EXCEPTION_MAX_CHARS,
            )
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO structured_log_events
                    (timestamp, level, logger_name, message, exception_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, level, record.name, message, exception_text),
            )
            self.conn.commit()
            self._successful_inserts += 1
            if self._successful_inserts % _RETENTION_INTERVAL == 0:
                self.enforce_retention()

    def enforce_retention(self) -> None:
        """Delete oldest rows when the sink-local cap is exceeded."""
        with self._lock:
            self.conn.execute(
                """
                DELETE FROM structured_log_events
                WHERE id IN (
                    SELECT id
                    FROM structured_log_events
                    ORDER BY id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.row_limit,),
            )
            self.conn.commit()

    def query(
        self,
        *,
        levels: set[str],
        since: datetime | None,
        q: str | None,
        limit: int,
    ) -> list[StructuredLogEvent]:
        """Query log rows newest-first."""
        if not levels:
            return []

        since_filter = (
            _format_timestamp(since.timestamp()) if since is not None else None
        )
        like_filter = f"%{q}%" if q else None
        params: tuple[str | int | None, ...] = (
            int("warning" in levels),
            int("error" in levels),
            int("critical" in levels),
            since_filter,
            since_filter,
            like_filter,
            like_filter,
            like_filter,
            like_filter,
            limit,
        )
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT timestamp, level, logger_name, message, exception_text
                FROM structured_log_events
                WHERE (
                    (? = 1 AND level = 'warning')
                    OR (? = 1 AND level = 'error')
                    OR (? = 1 AND level = 'critical')
                )
                AND (? IS NULL OR timestamp >= ?)
                AND (
                    ? IS NULL
                    OR message LIKE ?
                    OR exception_text LIKE ?
                    OR logger_name LIKE ?
                )
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [
            StructuredLogEvent(
                timestamp=row["timestamp"],
                level=row["level"],
                logger_name=row["logger_name"],
                message=row["message"],
                exception_text=row["exception_text"],
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the owned SQLite connection."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.conn.close()


class StructuredLogHandler(logging.Handler):
    """Root logging handler that writes warning/error/critical records to SQLite."""

    _reflexio_structured_logging = True

    def __init__(self, store: StructuredLogStore) -> None:
        super().__init__(level=logging.WARNING)
        self.store = store
        self._last_stderr_error_at = 0.0
        self._ref_count = 0

    @property
    def db_path(self) -> str:
        return self.store.db_path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.store.insert(record)
        except Exception as exc:  # noqa: BLE001 - logging must fail open
            now = time.monotonic()
            if now - self._last_stderr_error_at > 60:
                self._last_stderr_error_at = now
                print(
                    f"Reflexio structured logging insert failed: {exc}",
                    file=sys.stderr,
                )

    def close(self) -> None:
        self.store.close()
        super().close()


class StructuredLoggingHandle:
    """App-owned lifecycle handle for the structured logging subsystem."""

    def __init__(self, handler: StructuredLogHandler) -> None:
        self.handler = handler
        self.store = handler.store
        self._closed = False

    def query(
        self,
        *,
        levels: set[str],
        since: datetime | None,
        q: str | None,
        limit: int,
    ) -> list[StructuredLogEvent]:
        """Query rows through the owned store."""
        return self.store.query(levels=levels, since=since, q=q, limit=limit)

    def close(self) -> None:
        """Release this app's claim on the process-global handler."""
        if self._closed:
            return
        self._closed = True
        with _HANDLER_LOCK:
            self.handler._ref_count -= 1
            if self.handler._ref_count > 0:
                return
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.handler)
            self.handler.close()


def resolve_oss_structured_log_db_path(org_id: str) -> str | None:
    """Resolve the OSS structured-log SQLite DB path for a bootstrap org.

    Args:
        org_id: Bootstrap organization ID used by the local OSS configurator.

    Returns:
        Resolved SQLite path, or None when a non-OSS configurator is active.
    """
    configurator_class = get_configurator_class()
    if configurator_class is not DefaultConfigurator:
        return None
    configurator = configurator_class(org_id=org_id)
    return resolve_oss_sqlite_db_path(
        configurator,
        configurator.get_current_storage_configuration(),
    )


def install_structured_logging(
    db_path: str,
    *,
    row_limit: int = _DEFAULT_ROW_LIMIT,
) -> StructuredLoggingHandle:
    """Install or reuse the process-global structured logging handler."""
    with _HANDLER_LOCK:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            if not isinstance(handler, StructuredLogHandler):
                continue
            if Path(handler.db_path) == Path(db_path):
                handler._ref_count += 1
                return StructuredLoggingHandle(handler)
            raise RuntimeError(
                "Reflexio structured logging is already installed for "
                f"{handler.db_path}; refusing to route records to {db_path}"
            )

        store = StructuredLogStore(db_path, row_limit=row_limit)
        handler = StructuredLogHandler(store)
        handler._ref_count = 1
        root_logger.addHandler(handler)
        return StructuredLoggingHandle(handler)


def _format_timestamp(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, tz=UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _cap(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars]
