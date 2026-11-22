"""SQLite connection, lock and transaction handling."""

import contextlib
import functools
import logging
import sqlite3
import threading
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

from reflexio.server import publish_timing, search_runtime
from reflexio.server.background_work import is_background_work
from reflexio.server.services.storage.error import (
    ExecutionInputBusyError,
    PostCommitStorageError,
    StorageError,
)

logger = logging.getLogger(__name__)

# 3.35 brought RETURNING and DROP COLUMN; 3.38 brought unixepoch(), which the
# aggregation, lineage and open-world tables use.
_MINIMUM_SQLITE_VERSION = (3, 38, 0)


_SQLITE_INITIALIZATION_LOCK_STRIPES = 64


_sqlite_initialization_locks = tuple(
    threading.Lock() for _ in range(_SQLITE_INITIALIZATION_LOCK_STRIPES)
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_sqlite_initialization_lock(db_path: str) -> threading.Lock:
    """Return a bounded process-local lock for one SQLite database path."""
    normalized_path = str(Path(db_path).resolve())
    return _sqlite_initialization_locks[
        hash(normalized_path) % _SQLITE_INITIALIZATION_LOCK_STRIPES
    ]


_VEC_OP_KINDS = frozenset({"vec_upsert", "vec_delete"})

_EMBEDDING_TABLES = frozenset(
    {"interactions", "profiles", "user_playbooks", "agent_playbooks"}
)


class VecAwareConnection(sqlite3.Connection):
    """A connection that can mark the vector sidecar stale.

    A process without sqlite-vec writes embeddings its sidecar cannot index.
    Once ``mark_sidecar_stale_on_embedding_writes`` is called, every commit
    that follows a write to an embedding table first takes the schema stamp
    back to ``stale_stamp``, inside the same transaction, so the next process
    that loads sqlite-vec reconciles the sidecar. Reading the flag costs
    nothing for connections that never enable it.
    """

    _writes_embeddings = False
    _stale_stamp: int | None = None
    _caller_authorizer: Callable[..., int] | None = None

    def mark_sidecar_stale_on_embedding_writes(self, stale_stamp: int) -> None:
        self._stale_stamp = stale_stamp
        self._arm_authorizer()

    def set_authorizer(self, authorizer_callback: Callable[..., int] | None) -> None:
        """Install a caller's authorizer; the sidecar tracker, when on, chains to it."""
        self._caller_authorizer = authorizer_callback
        if self._stale_stamp is not None:
            self._arm_authorizer()
        else:
            super().set_authorizer(authorizer_callback)

    def _arm_authorizer(self) -> None:
        def authorize(action: int, table: Any, *rest: Any) -> int:
            if action in (
                sqlite3.SQLITE_INSERT,
                sqlite3.SQLITE_UPDATE,
                sqlite3.SQLITE_DELETE,
            ) and (table in _EMBEDDING_TABLES):
                self._writes_embeddings = True
            caller = self._caller_authorizer
            return sqlite3.SQLITE_OK if caller is None else caller(action, table, *rest)

        # The authorizer runs when a statement is PREPARED, and the statement
        # cache reuses prepared statements. Installing the callback again
        # expires them, so every statement after a commit is prepared, and
        # authorized, again.
        super().set_authorizer(authorize)

    def _lower_stamp(self) -> None:
        if (
            self._stale_stamp is not None
            and self._writes_embeddings
            and self.execute("PRAGMA user_version").fetchone()[0] > self._stale_stamp
        ):
            self.execute(f"PRAGMA user_version = {int(self._stale_stamp)}")

    def commit(self) -> None:
        self._lower_stamp()
        super().commit()
        self._writes_embeddings = False
        if self._stale_stamp is not None:
            self._arm_authorizer()

    def rollback(self) -> None:
        super().rollback()
        self._writes_embeddings = False
        if self._stale_stamp is not None:
            self._arm_authorizer()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        if exc_type is None:
            self._lower_stamp()
        result = super().__exit__(exc_type, exc_value, traceback)
        self._writes_embeddings = False
        if self._stale_stamp is not None:
            self._arm_authorizer()
        return result


class SQLiteConnectionMixin:
    """Connection ownership, the commit scope and low-level query helpers."""

    conn: VecAwareConnection
    db_path: str
    _lock: threading.RLock
    _scope_depth: int
    _scope_rollback_only: bool
    _deferred_index_ops: list[tuple[str, Any]]
    _flush_index_op: Any
    _reconcile_pending_vector: Any

    @staticmethod
    def handle_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                args[0]._rollback_owned_transaction()
                # The request's own search deadline is not a storage fault:
                # wrapping it would hide it from callers that degrade on it.
                if isinstance(e, (StorageError, search_runtime.SearchDeadlineError)):
                    raise
                # Deferred source conversion can include stored customer data
                # in exception values. Preserve the cause for classification and
                # the worker's bounded locator, without logging that payload.
                if is_background_work():
                    logger.error("Error in %s: %s", func.__name__, type(e).__name__)
                    raise StorageError(message=type(e).__name__) from e
                import traceback

                stack_trace = traceback.format_exc()
                logger.error(
                    "Error in %s: %s\nStack trace:\n%s",
                    func.__name__,
                    str(e),
                    stack_trace,
                )
                raise StorageError(message=f"{e}\nStack trace:\n{stack_trace}") from e

        return wrapper

    # ------------------------------------------------------------------
    # Transaction scope
    # ------------------------------------------------------------------

    def _own_transaction(self) -> bool:
        """True when the caller is NOT inside a commit_scope (owns its BEGIN/commit)."""
        return self._scope_depth == 0

    def _rollback_owned_transaction(self) -> None:
        """Release a failed standalone write without undoing an outer scope."""
        with self._lock:
            if self._own_transaction():
                try:
                    self.conn.rollback()
                except Exception:
                    logger.warning("Rollback after failed write failed", exc_info=True)

    @property
    def in_commit_scope(self) -> bool:
        # A different thread waits for this instance's short writer to finish.
        with self._lock:
            return self._scope_depth > 0

    @contextlib.contextmanager
    def commit_scope(self) -> Generator[None, None, None]:
        """Group writes into one atomic commit; defer FTS/vec index ops.

        Nested scopes join the outermost: only the outer scope commits. On
        exception the whole transaction rolls back and deferred index ops
        are discarded. A nested failure marks the outer transaction rollback-only,
        even when its caller catches the exception.
        """
        with self._lock:
            if self._scope_depth > 0:
                if self._scope_rollback_only:
                    raise StorageError("Transaction marked for rollback")
                self._scope_depth += 1
                try:
                    yield
                except BaseException:
                    self._scope_rollback_only = True
                    raise
                finally:
                    self._scope_depth -= 1
                return
            try:
                self.conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                # Only failed admission is safe to retry: no business statement
                # or commit has run. Never reclassify errors from inside the TX.
                if getattr(error, "sqlite_errorcode", 0) & 255 == sqlite3.SQLITE_BUSY:
                    raise ExecutionInputBusyError(
                        "SQLite writer admission is busy"
                    ) from error
                raise
            self._scope_depth = 1
            self._scope_rollback_only = False
            try:
                yield
                if self._scope_rollback_only:
                    raise StorageError(
                        "Nested operation marked transaction for rollback"
                    )
                # Queued vector work is recorded in THIS transaction, so a crash
                # between the commit and the flush leaves a durable list of the
                # sidecar rows to rebuild on the next open.
                self._record_pending_vector_ops()
                # The COMMIT itself. Separated from the scope's other work
                # because `commit_scope_ms` reported 15.6s on prod of which
                # 11.7s was outside every phase it contains -- this says how
                # much of that is the commit rather than the writes.
                with publish_timing.phase("scope_commit"):
                    self.conn.commit()
                ops, self._deferred_index_ops = self._deferred_index_ops, []
                try:
                    # Vector ops rebuild the sidecar row from the CURRENT primary
                    # row, once per row: an older flush must not write a stale
                    # in-memory embedding over a newer commit, or clear its marker.
                    reconciled: set[tuple[str, int]] = set()
                    for kind, args in ops:
                        if kind in _VEC_OP_KINDS:
                            key = (args[0], args[1])
                            if key not in reconciled:
                                reconciled.add(key)
                                self._reconcile_pending_vector(*key)
                        else:
                            self._flush_index_op(kind, args)
                except Exception as error:
                    # Ops not flushed stay in vec_pending; the next open
                    # rebuilds exactly those sidecar rows.
                    raise PostCommitStorageError(
                        "Primary rows committed; deferred index update failed"
                    ) from error
            except BaseException:
                try:
                    self.conn.rollback()
                except Exception:
                    logger.warning("Rollback after failed scope failed", exc_info=True)
                self._deferred_index_ops = []
                raise
            finally:
                self._scope_depth = 0
                self._scope_rollback_only = False

    def _record_pending_vector_ops(self) -> None:
        for kind, args in self._deferred_index_ops:
            if kind in _VEC_OP_KINDS:
                self.conn.execute(
                    "INSERT OR IGNORE INTO vec_pending(table_name, row_id) VALUES (?, ?)",
                    (args[0], args[1]),
                )

    def _try_load_sqlite_vec(self) -> bool:
        """Attempt to load the sqlite-vec extension for native KNN search.

        Returns:
            True if the extension was loaded successfully, False otherwise.
        """
        try:
            import sqlite_vec  # type: ignore[import-untyped]

            # AttributeError covers Python builds without loadable-extension
            # support (common with pyenv/Homebrew on macOS) — the method
            # itself is absent rather than raising at runtime.
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            logger.info("sqlite-vec extension loaded — native KNN search enabled")
            return True
        except (ImportError, OSError, AttributeError, sqlite3.OperationalError) as e:
            logger.info("sqlite-vec not available, using Python fallback: %s", e)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute(
        self, sql: str, params: tuple[Any, ...] | list[Any] = ()
    ) -> sqlite3.Cursor:
        with self._lock:
            try:
                cur = self.conn.execute(sql, params)
                if self._own_transaction():
                    self.conn.commit()
                return cur
            except BaseException:
                self._rollback_owned_transaction()
                raise

    def _fetchone(
        self, sql: str, params: tuple[Any, ...] | list[Any] = ()
    ) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def _fetchall(
        self, sql: str, params: tuple[Any, ...] | list[Any] = ()
    ) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()
