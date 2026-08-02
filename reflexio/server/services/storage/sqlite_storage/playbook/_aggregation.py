"""SQLite durable incremental playbook-aggregation state."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any

from reflexio.server.services.storage.storage_base.playbook import (
    AGGREGATION_INVALIDATION_RETENTION_SECONDS,
    AGGREGATION_RETRY_BASE_SECONDS,
    AGGREGATION_RETRY_MAX_SECONDS,
    AggregationDisposition,
    PlaybookAggregationBacklog,
    PlaybookAggregationClaim,
    PlaybookAggregationClusterMatch,
    PlaybookAggregationInvalidation,
    PlaybookAggregationRerunSnapshot,
)

AGGREGATION_DDL = """
CREATE TABLE IF NOT EXISTS playbook_aggregation_state (
    agent_version TEXT PRIMARY KEY,
    last_success_at INTEGER,
    pending INTEGER NOT NULL DEFAULT 1 CHECK (pending IN (0, 1)),
    next_attempt_at INTEGER NOT NULL DEFAULT (unixepoch()),
    state_version INTEGER NOT NULL DEFAULT 0,
    retry_cursor INTEGER NOT NULL DEFAULT 0,
    bootstrap_status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_state_due
    ON playbook_aggregation_state(pending, next_attempt_at, agent_version);

CREATE TABLE IF NOT EXISTS playbook_aggregation_lease (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    claim_owner TEXT,
    claim_fence INTEGER NOT NULL,
    claim_expires_at INTEGER,
    agent_version TEXT
);

CREATE TABLE IF NOT EXISTS playbook_aggregation_cluster (
    cluster_id TEXT PRIMARY KEY,
    index_rowid INTEGER UNIQUE,
    agent_version TEXT NOT NULL,
    agent_playbook_id INTEGER,
    centroid TEXT,
    vector_sum TEXT,
    member_count INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT,
    embedding_dimension INTEGER,
    normalization TEXT NOT NULL DEFAULT 'l2',
    state TEXT NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'rebuilding')),
    dirty INTEGER NOT NULL DEFAULT 0 CHECK (dirty IN (0, 1)),
    rebuild_cursor INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_cluster_version
    ON playbook_aggregation_cluster(agent_version, state, cluster_id);

CREATE TABLE IF NOT EXISTS playbook_aggregation_item (
    agent_version TEXT NOT NULL,
    user_playbook_id INTEGER NOT NULL,
    disposition TEXT NOT NULL CHECK (
        disposition IN ('residual', 'cluster_member', 'terminal_noop')
    ),
    cluster_id TEXT,
    reason TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_attempt_at INTEGER,
    created_at INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at INTEGER NOT NULL DEFAULT (unixepoch()),
    PRIMARY KEY (agent_version, user_playbook_id)
);
CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_item_residual
    ON playbook_aggregation_item(
        agent_version, disposition, attempt_count, last_attempt_at, user_playbook_id
    );

CREATE INDEX IF NOT EXISTS idx_user_playbooks_aggregation_intake
    ON user_playbooks(agent_version, user_playbook_id)
    WHERE status IS NULL AND trim(content) <> '' AND trim(agent_version) <> '';

CREATE TABLE IF NOT EXISTS playbook_aggregation_invalidation (
    invalidation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_version TEXT NOT NULL,
    operation TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    source_ids TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL DEFAULT (unixepoch()),
    processed_at INTEGER
);
DROP INDEX IF EXISTS idx_playbook_aggregation_invalidation_pending;
CREATE INDEX idx_playbook_aggregation_invalidation_pending
    ON playbook_aggregation_invalidation(agent_version, invalidation_id)
    WHERE processed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_invalidation_retention
    ON playbook_aggregation_invalidation(processed_at)
    WHERE processed_at IS NOT NULL;

CREATE TRIGGER IF NOT EXISTS capture_playbook_aggregation_hard_delete
BEFORE DELETE ON user_playbooks
WHEN OLD.status IS NULL AND trim(OLD.agent_version) <> ''
BEGIN
    INSERT INTO playbook_aggregation_invalidation(
        agent_version, operation, entity_id, source_ids
    ) VALUES (OLD.agent_version, 'hard_delete', OLD.user_playbook_id, '[]');
    INSERT INTO playbook_aggregation_state(
        agent_version, pending, next_attempt_at
    ) VALUES (OLD.agent_version, 1, unixepoch())
    ON CONFLICT(agent_version) DO UPDATE SET pending=1,
        next_attempt_at=min(playbook_aggregation_state.next_attempt_at,
                            excluded.next_attempt_at);
END;
"""


def init_playbook_aggregation_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(AGGREGATION_DDL)


class PlaybookAggregationStoreMixin:
    conn: sqlite3.Connection
    _lock: threading.RLock
    _own_transaction: Any
    commit_scope: Any
    _has_sqlite_vec: bool

    @property
    def supports_incremental_playbook_aggregation(self) -> bool:
        return self._has_sqlite_vec and getattr(
            self, "_incremental_aggregation_enabled", True
        )

    @supports_incremental_playbook_aggregation.setter
    def supports_incremental_playbook_aggregation(self, enabled: bool) -> None:
        self._incremental_aggregation_enabled = bool(enabled)

    @property
    def playbook_aggregation_blocked_reason(self) -> str | None:
        if self._has_sqlite_vec:
            return None
        return "blocked_missing_vector_index"

    def schedule_playbook_aggregation(self, agent_version: str) -> None:
        if not agent_version.strip():
            raise ValueError("agent_version must be non-empty")
        with self._lock:
            self.conn.execute(
                "INSERT INTO playbook_aggregation_state "
                "(agent_version, pending, next_attempt_at) "
                "VALUES (?, 1, unixepoch()) "
                "ON CONFLICT(agent_version) DO UPDATE SET pending=1, "
                "next_attempt_at=min(playbook_aggregation_state.next_attempt_at, "
                "excluded.next_attempt_at)",
                (agent_version,),
            )
            if self._own_transaction():
                self.conn.commit()

    def repair_playbook_aggregation_pending_state(
        self, *, limit: int = 100
    ) -> list[str]:
        if limit <= 0:
            return []
        with self._lock:
            self.conn.execute(
                "DELETE FROM playbook_aggregation_invalidation "
                "WHERE processed_at IS NOT NULL AND processed_at < unixepoch()-?",
                (AGGREGATION_INVALIDATION_RETENTION_SECONDS,),
            )
            rows = self.conn.execute(
                "WITH work(agent_version) AS ("
                "SELECT p.agent_version FROM user_playbooks p WHERE p.status IS NULL "
                "AND trim(p.content) <> '' AND trim(p.agent_version) <> '' "
                "AND NOT EXISTS (SELECT 1 FROM playbook_aggregation_item i WHERE "
                "i.agent_version=p.agent_version AND "
                "i.user_playbook_id=p.user_playbook_id) UNION "
                "SELECT agent_version FROM playbook_aggregation_item "
                "WHERE disposition='residual' UNION "
                "SELECT agent_version FROM playbook_aggregation_invalidation "
                "WHERE processed_at IS NULL UNION "
                "SELECT agent_version FROM playbook_aggregation_cluster "
                "WHERE dirty=1 OR state='rebuilding') "
                "SELECT DISTINCT w.agent_version FROM work w LEFT JOIN "
                "playbook_aggregation_state s ON s.agent_version=w.agent_version "
                "WHERE COALESCE(s.pending, 0)=0 ORDER BY w.agent_version LIMIT ?",
                (limit,),
            ).fetchall()
            versions = [str(row[0]) for row in rows]
            self.conn.executemany(
                "INSERT INTO playbook_aggregation_state "
                "(agent_version, pending, next_attempt_at) VALUES (?, 1, unixepoch()) "
                "ON CONFLICT(agent_version) DO UPDATE SET pending=1, "
                "next_attempt_at=min(playbook_aggregation_state.next_attempt_at, "
                "excluded.next_attempt_at)",
                [(version,) for version in versions],
            )
            if self._own_transaction():
                self.conn.commit()
            return versions

    def claim_due_playbook_aggregation(
        self,
        *,
        owner: str,
        lease_seconds: int,
        agent_version: str | None = None,
    ) -> PlaybookAggregationClaim | None:
        if not owner.strip() or lease_seconds <= 0:
            raise ValueError("claim owner and lease_seconds must be valid")
        with self._lock:
            own = self._own_transaction()
            if own:
                self.conn.execute("BEGIN IMMEDIATE")
            try:
                now = int(self.conn.execute("SELECT unixepoch()").fetchone()[0])
                lease = self.conn.execute(
                    "SELECT claim_expires_at FROM playbook_aggregation_lease "
                    "WHERE singleton=1"
                ).fetchone()
                if lease is not None and lease[0] is not None and int(lease[0]) > now:
                    if own:
                        self.conn.commit()
                    return None
                if agent_version is None:
                    row = self.conn.execute(
                        "SELECT agent_version, state_version "
                        "FROM playbook_aggregation_state "
                        "WHERE pending=1 AND next_attempt_at <= ? "
                        "ORDER BY next_attempt_at, agent_version LIMIT 1",
                        (now,),
                    ).fetchone()
                else:
                    row = self.conn.execute(
                        "SELECT agent_version, state_version "
                        "FROM playbook_aggregation_state WHERE agent_version=?",
                        (agent_version,),
                    ).fetchone()
                if row is None:
                    if own:
                        self.conn.commit()
                    return None
                old_fence = self.conn.execute(
                    "SELECT claim_fence FROM playbook_aggregation_lease "
                    "WHERE singleton=1"
                ).fetchone()
                fence = (int(old_fence[0]) if old_fence else 0) + 1
                expires_at = now + lease_seconds
                self.conn.execute(
                    "INSERT INTO playbook_aggregation_lease "
                    "(singleton, claim_owner, claim_fence, claim_expires_at, agent_version) "
                    "VALUES (1, ?, ?, ?, ?) "
                    "ON CONFLICT(singleton) DO UPDATE SET "
                    "claim_owner=excluded.claim_owner, claim_fence=excluded.claim_fence, "
                    "claim_expires_at=excluded.claim_expires_at, "
                    "agent_version=excluded.agent_version",
                    (owner, fence, expires_at, str(row[0])),
                )
                if own:
                    self.conn.commit()
                return PlaybookAggregationClaim(
                    agent_version=str(row[0]),
                    owner=owner,
                    fence=fence,
                    state_version=int(row[1]),
                    expires_at=expires_at,
                )
            except Exception:
                if own:
                    self.conn.rollback()
                raise

    def renew_playbook_aggregation_claim(
        self, claim: PlaybookAggregationClaim, *, lease_seconds: int
    ) -> PlaybookAggregationClaim | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._lock:
            now = int(self.conn.execute("SELECT unixepoch()").fetchone()[0])
            expires_at = now + lease_seconds
            cur = self.conn.execute(
                "UPDATE playbook_aggregation_lease SET claim_expires_at=? "
                "WHERE singleton=1 AND claim_owner=? AND claim_fence=? "
                "AND agent_version=? AND claim_expires_at > ?",
                (
                    expires_at,
                    claim.owner,
                    claim.fence,
                    claim.agent_version,
                    now,
                ),
            )
            if self._own_transaction():
                self.conn.commit()
            if cur.rowcount != 1:
                return None
            return PlaybookAggregationClaim(
                claim.agent_version,
                claim.owner,
                claim.fence,
                claim.state_version,
                expires_at,
            )

    def _aggregation_claim_is_live(self, claim: PlaybookAggregationClaim) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM playbook_aggregation_lease "
                "WHERE singleton=1 AND claim_owner=? AND claim_fence=? "
                "AND agent_version=? AND claim_expires_at > unixepoch()",
                (claim.owner, claim.fence, claim.agent_version),
            ).fetchone()
            is not None
        )

    def validate_playbook_aggregation_claim(
        self, claim: PlaybookAggregationClaim
    ) -> bool:
        with self._lock:
            if not self._aggregation_claim_is_live(claim):
                return False
            row = self.conn.execute(
                "SELECT state_version FROM playbook_aggregation_state "
                "WHERE agent_version=?",
                (claim.agent_version,),
            ).fetchone()
            return row is not None and int(row[0]) == claim.state_version

    def finish_playbook_aggregation_claim(
        self,
        claim: PlaybookAggregationClaim,
        *,
        success: bool,
        retry_after_seconds: int,
        backlog_retry_after_seconds: int,
        min_interval_seconds: int,
        backlog: PlaybookAggregationBacklog | None = None,
    ) -> bool:
        with self._lock:
            if not self._aggregation_claim_is_live(claim):
                return False
            if not success:
                delay = retry_after_seconds
                pending = True
            else:
                backlog = backlog or self.get_playbook_aggregation_backlog(
                    claim.agent_version
                )
                pending = backlog.pending
                if pending:
                    delay = max(
                        backlog_retry_after_seconds,
                        backlog.continuation_delay_seconds,
                    )
                else:
                    delay = min_interval_seconds
            cur = self.conn.execute(
                "UPDATE playbook_aggregation_state SET "
                "last_success_at=CASE WHEN ? THEN unixepoch() ELSE last_success_at END, "
                "pending=?, next_attempt_at=unixepoch()+?, state_version=state_version+1 "
                "WHERE agent_version=? AND state_version=?",
                (
                    int(success),
                    int(pending),
                    max(0, delay),
                    claim.agent_version,
                    claim.state_version,
                ),
            )
            if cur.rowcount != 1:
                if self._own_transaction():
                    self.conn.rollback()
                return False
            self.conn.execute(
                "UPDATE playbook_aggregation_lease SET claim_owner=NULL, "
                "claim_expires_at=NULL, agent_version=NULL WHERE singleton=1 "
                "AND claim_owner=? AND claim_fence=?",
                (claim.owner, claim.fence),
            )
            if self._own_transaction():
                self.conn.commit()
            return True

    def stage_playbook_aggregation_intake(
        self, agent_version: str, *, limit: int
    ) -> list[int]:
        if limit <= 0:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT p.user_playbook_id FROM user_playbooks p "
                "WHERE p.agent_version=? AND p.status IS NULL "
                "AND trim(p.content) <> '' AND NOT EXISTS ("
                " SELECT 1 FROM playbook_aggregation_item i "
                " WHERE i.agent_version=? "
                " AND i.user_playbook_id=p.user_playbook_id) "
                "ORDER BY p.user_playbook_id LIMIT ?",
                (agent_version, agent_version, limit),
            ).fetchall()
            ids = [int(row[0]) for row in rows]
            self.conn.executemany(
                "INSERT OR IGNORE INTO playbook_aggregation_item "
                "(agent_version, user_playbook_id, disposition, reason) "
                "VALUES (?, ?, 'residual', 'new')",
                [(agent_version, item_id) for item_id in ids],
            )
            if self._own_transaction():
                self.conn.commit()
            return ids

    def get_playbook_aggregation_bootstrap_status(self, agent_version: str) -> str:
        with self._lock:
            row = self.conn.execute(
                "SELECT bootstrap_status FROM playbook_aggregation_state "
                "WHERE agent_version=?",
                (agent_version,),
            ).fetchone()
        return str(row[0]) if row is not None else "pending"

    def set_playbook_aggregation_bootstrap_status(
        self, agent_version: str, status: str
    ) -> None:
        if status not in {"pending", "complete"}:
            raise ValueError("invalid aggregation bootstrap status")
        with self._lock:
            self.conn.execute(
                "INSERT INTO playbook_aggregation_state "
                "(agent_version, bootstrap_status, pending, next_attempt_at) "
                "VALUES (?, ?, 1, unixepoch()) ON CONFLICT(agent_version) "
                "DO UPDATE SET bootstrap_status=excluded.bootstrap_status",
                (agent_version, status),
            )
            if self._own_transaction():
                self.conn.commit()

    def get_playbook_aggregation_cluster_rebuild_cursor(
        self, cluster_id: str
    ) -> tuple[int, str] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT rebuild_cursor, state FROM playbook_aggregation_cluster "
                "WHERE cluster_id=?",
                (cluster_id,),
            ).fetchone()
        if row is None:
            return None
        return int(row[0]), str(row[1])

    def adopt_legacy_playbook_aggregation_cluster_page(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        agent_playbook_id: int,
        member_embeddings: list[tuple[int, list[float]]],
        embedding_model: str,
        embedding_dimension: int,
        rebuild_cursor: int,
        complete: bool,
    ) -> None:
        if any(len(value) != embedding_dimension for _, value in member_embeddings):
            raise ValueError("legacy cluster embedding dimension changed")
        with self._lock:
            row = self.conn.execute(
                "SELECT index_rowid, vector_sum, member_count, state, "
                "embedding_model, embedding_dimension "
                "FROM playbook_aggregation_cluster WHERE cluster_id=?",
                (cluster_id,),
            ).fetchone()
            if row is None:
                next_rowid = int(
                    self.conn.execute(
                        "SELECT COALESCE(max(index_rowid), 0)+1 "
                        "FROM playbook_aggregation_cluster"
                    ).fetchone()[0]
                )
                self.conn.execute(
                    "INSERT INTO playbook_aggregation_cluster "
                    "(cluster_id, index_rowid, agent_version, agent_playbook_id, "
                    "member_count, embedding_model, embedding_dimension, state) "
                    "VALUES (?, ?, ?, ?, 0, ?, ?, 'rebuilding')",
                    (
                        cluster_id,
                        next_rowid,
                        agent_version,
                        agent_playbook_id,
                        embedding_model,
                        embedding_dimension,
                    ),
                )
                index_rowid = next_rowid
                vector_sum = [0.0] * embedding_dimension
                member_count = 0
            else:
                if str(row[3]) == "active":
                    return
                if row[4] != embedding_model or int(row[5]) != embedding_dimension:
                    raise RuntimeError("legacy cluster embedding provenance changed")
                index_rowid = int(row[0])
                vector_sum = (
                    [float(value) for value in json.loads(row[1])]
                    if row[1]
                    else [0.0] * embedding_dimension
                )
                member_count = int(row[2])
            for user_playbook_id, embedding in member_embeddings:
                inserted = self.conn.execute(
                    "INSERT OR IGNORE INTO playbook_aggregation_item "
                    "(agent_version, user_playbook_id, disposition, cluster_id, reason) "
                    "VALUES (?, ?, 'cluster_member', ?, 'legacy_adopted')",
                    (agent_version, user_playbook_id, cluster_id),
                )
                if inserted.rowcount == 1:
                    vector_sum = [
                        left + right
                        for left, right in zip(vector_sum, embedding, strict=True)
                    ]
                    member_count += 1
            centroid = (
                [value / member_count for value in vector_sum] if member_count else None
            )
            self.conn.execute(
                "UPDATE playbook_aggregation_cluster SET vector_sum=?, centroid=?, "
                "member_count=?, rebuild_cursor=?, state=? WHERE cluster_id=?",
                (
                    json.dumps(vector_sum) if member_count else None,
                    json.dumps(centroid) if centroid is not None else None,
                    member_count,
                    rebuild_cursor,
                    "active" if complete else "rebuilding",
                    cluster_id,
                ),
            )
            if complete and centroid is not None:
                self.conn.execute(
                    "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                    (index_rowid,),
                )
                self.conn.execute(
                    "INSERT INTO playbook_aggregation_clusters_vec(rowid, embedding) "
                    "VALUES (?, ?)",
                    (index_rowid, json.dumps(centroid)),
                )
            if self._own_transaction():
                self.conn.commit()

    def reset_playbook_aggregation_version(self, agent_version: str) -> None:
        if self._own_transaction():
            with self.commit_scope():
                self.reset_playbook_aggregation_version(agent_version)
            return
        with self._lock:
            index_rows = self.conn.execute(
                "SELECT index_rowid FROM playbook_aggregation_cluster "
                "WHERE agent_version=? AND index_rowid IS NOT NULL",
                (agent_version,),
            ).fetchall()
            for row in index_rows:
                self.conn.execute(
                    "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                    (int(row[0]),),
                )
            self.conn.execute(
                "DELETE FROM playbook_aggregation_item WHERE agent_version=?",
                (agent_version,),
            )
            self.conn.execute(
                "DELETE FROM playbook_aggregation_cluster WHERE agent_version=?",
                (agent_version,),
            )
            self.set_playbook_aggregation_bootstrap_status(agent_version, "complete")

    def capture_playbook_aggregation_rerun_snapshot(
        self, agent_version: str, *, limit: int
    ) -> PlaybookAggregationRerunSnapshot:
        if limit <= 0:
            return PlaybookAggregationRerunSnapshot((), (), None, None)
        with self._lock:
            own = self._own_transaction()
            if own:
                self.conn.execute("BEGIN")
            try:
                playbooks = self.get_user_playbooks(  # type: ignore[attr-defined]
                    limit=limit,
                    agent_version=agent_version,
                    status_filter=[None],
                    include_embedding=True,
                    max_user_playbook_id=2**63 - 1,
                )
                invalidation_rows = self.conn.execute(
                    "SELECT invalidation_id FROM playbook_aggregation_invalidation "
                    "WHERE agent_version=? AND processed_at IS NULL "
                    "ORDER BY invalidation_id LIMIT ?",
                    (agent_version, limit),
                ).fetchall()
                if own:
                    self.conn.commit()
            except Exception:
                if own:
                    self.conn.rollback()
                raise
        invalidation_ids = tuple(int(row[0]) for row in invalidation_rows)
        return PlaybookAggregationRerunSnapshot(
            user_playbooks=tuple(playbooks),
            invalidation_ids=invalidation_ids,
            user_high_watermark=(
                max(item.user_playbook_id for item in playbooks) if playbooks else None
            ),
            invalidation_high_watermark=(
                max(invalidation_ids) if invalidation_ids else None
            ),
        )

    def stage_playbook_aggregation_snapshot(
        self, agent_version: str, user_playbook_ids: list[int]
    ) -> None:
        if not user_playbook_ids:
            return
        with self._lock:
            self.conn.executemany(
                "INSERT OR IGNORE INTO playbook_aggregation_item "
                "(agent_version, user_playbook_id, disposition, reason) "
                "VALUES (?, ?, 'residual', 'full_rerun_snapshot')",
                [(agent_version, item_id) for item_id in user_playbook_ids],
            )
            if self._own_transaction():
                self.conn.commit()

    def mark_playbook_aggregation_invalidations_processed(
        self,
        claim: PlaybookAggregationClaim,
        invalidation_ids: list[int],
    ) -> bool:
        if not invalidation_ids:
            return True
        if self._own_transaction():
            with self.commit_scope():
                return self.mark_playbook_aggregation_invalidations_processed(
                    claim, invalidation_ids
                )
        with self._lock:
            if not self._aggregation_claim_is_live(claim):
                return False
            placeholders = ",".join("?" for _ in invalidation_ids)
            self.conn.execute(
                "UPDATE playbook_aggregation_invalidation SET processed_at=unixepoch() "
                f"WHERE agent_version=? AND invalidation_id IN ({placeholders}) "
                "AND processed_at IS NULL",
                (claim.agent_version, *invalidation_ids),
            )
            return True

    def get_playbook_aggregation_residual_ids(
        self, agent_version: str, *, limit: int
    ) -> list[int]:
        if limit <= 0:
            return []
        with self._lock:
            fresh_quota = (limit + 1) // 2
            retry_quota = limit - fresh_quota
            fresh_rows = self.conn.execute(
                "SELECT user_playbook_id FROM playbook_aggregation_item "
                "WHERE agent_version=? AND disposition='residual' AND attempt_count=0 "
                "ORDER BY user_playbook_id LIMIT ?",
                (agent_version, fresh_quota),
            ).fetchall()
            retry_rows = self.conn.execute(
                "SELECT user_playbook_id FROM playbook_aggregation_item "
                "WHERE agent_version=? AND disposition='residual' AND attempt_count>0 "
                "AND last_attempt_at+min(?, ?*(1 << min(max(attempt_count-1, 0), 6))) "
                "<= unixepoch() "
                "ORDER BY last_attempt_at, user_playbook_id LIMIT ?",
                (
                    agent_version,
                    AGGREGATION_RETRY_MAX_SECONDS,
                    AGGREGATION_RETRY_BASE_SECONDS,
                    retry_quota,
                ),
            ).fetchall()
            ids = [int(row[0]) for row in [*fresh_rows, *retry_rows]]
            if len(ids) < limit:
                placeholders = ",".join("?" for _ in ids)
                exclusion = (
                    f"AND user_playbook_id NOT IN ({placeholders})" if ids else ""
                )
                fill = self.conn.execute(
                    "SELECT user_playbook_id FROM playbook_aggregation_item "
                    "WHERE agent_version=? AND disposition='residual' "
                    "AND (attempt_count=0 OR last_attempt_at IS NULL OR "
                    "last_attempt_at+min(?, ?*(1 << min(max(attempt_count-1, 0), 6))) "
                    "<= unixepoch()) "
                    f"{exclusion} ORDER BY COALESCE(last_attempt_at, 0), "
                    "user_playbook_id LIMIT ?",
                    (
                        agent_version,
                        AGGREGATION_RETRY_MAX_SECONDS,
                        AGGREGATION_RETRY_BASE_SECONDS,
                        *ids,
                        limit - len(ids),
                    ),
                ).fetchall()
                ids.extend(int(row[0]) for row in fill)
            self.conn.executemany(
                "UPDATE playbook_aggregation_item SET attempt_count=attempt_count+1, "
                "last_attempt_at=unixepoch(), updated_at=unixepoch() "
                "WHERE agent_version=? AND user_playbook_id=?",
                [(agent_version, item_id) for item_id in ids],
            )
            if self._own_transaction():
                self.conn.commit()
            return ids

    def set_playbook_aggregation_disposition(
        self,
        agent_version: str,
        user_playbook_ids: list[int],
        *,
        disposition: AggregationDisposition,
        cluster_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        if not user_playbook_ids:
            return
        with self._lock:
            self.conn.executemany(
                "UPDATE playbook_aggregation_item SET disposition=?, cluster_id=?, "
                "reason=?, updated_at=unixepoch() "
                "WHERE agent_version=? AND user_playbook_id=?",
                [
                    (disposition, cluster_id, reason, agent_version, item_id)
                    for item_id in user_playbook_ids
                ],
            )
            if self._own_transaction():
                self.conn.commit()

    def get_playbook_aggregation_backlog(
        self, agent_version: str
    ) -> PlaybookAggregationBacklog:
        with self._lock:
            undisposed = int(
                self.conn.execute(
                    "SELECT count(*) FROM user_playbooks p WHERE p.agent_version=? "
                    "AND p.status IS NULL AND trim(p.content) <> '' AND NOT EXISTS ("
                    "SELECT 1 FROM playbook_aggregation_item i "
                    "WHERE i.agent_version=? "
                    "AND i.user_playbook_id=p.user_playbook_id)",
                    (agent_version, agent_version),
                ).fetchone()[0]
            )
            residual = int(
                self.conn.execute(
                    "SELECT count(*) FROM playbook_aggregation_item "
                    "WHERE agent_version=? AND disposition='residual'",
                    (agent_version,),
                ).fetchone()[0]
            )
            invalidations = int(
                self.conn.execute(
                    "SELECT count(*) FROM playbook_aggregation_invalidation "
                    "WHERE agent_version=? AND processed_at IS NULL",
                    (agent_version,),
                ).fetchone()[0]
            )
            oldest_residual_age = self.conn.execute(
                "SELECT unixepoch()-min(created_at) FROM playbook_aggregation_item "
                "WHERE agent_version=? AND disposition='residual'",
                (agent_version,),
            ).fetchone()[0]
            dirty_repairs = int(
                self.conn.execute(
                    "SELECT count(*) FROM playbook_aggregation_cluster "
                    "WHERE agent_version=? AND (dirty=1 OR state='rebuilding')",
                    (agent_version,),
                ).fetchone()[0]
            )
            residual_retry_after = self.conn.execute(
                "SELECT COALESCE(MAX(0, MIN(CASE "
                "WHEN attempt_count <= 0 OR last_attempt_at IS NULL THEN 0 "
                "ELSE last_attempt_at + MIN(?, ? * (1 << MIN(MAX("
                "attempt_count - 1, 0), 6))) - unixepoch() END)), 0) "
                "FROM playbook_aggregation_item WHERE agent_version=? "
                "AND disposition='residual'",
                (
                    AGGREGATION_RETRY_MAX_SECONDS,
                    AGGREGATION_RETRY_BASE_SECONDS,
                    agent_version,
                ),
            ).fetchone()[0]
        return PlaybookAggregationBacklog(
            undisposed=undisposed,
            residual=residual,
            invalidations=invalidations,
            oldest_residual_age_seconds=(
                int(oldest_residual_age) if oldest_residual_age is not None else None
            ),
            dirty_repairs=dirty_repairs,
            residual_retry_after_seconds=int(residual_retry_after),
        )

    def append_playbook_aggregation_invalidation(
        self,
        *,
        agent_version: str,
        operation: str,
        entity_id: int,
        source_ids: list[int] | None = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO playbook_aggregation_invalidation "
                "(agent_version, operation, entity_id, source_ids) VALUES (?, ?, ?, ?)",
                (agent_version, operation, entity_id, json.dumps(source_ids or [])),
            )
            self.conn.execute(
                "INSERT INTO playbook_aggregation_state "
                "(agent_version, pending, next_attempt_at) VALUES (?, 1, unixepoch()) "
                "ON CONFLICT(agent_version) DO UPDATE SET pending=1, "
                "next_attempt_at=min(playbook_aggregation_state.next_attempt_at, "
                "excluded.next_attempt_at)",
                (agent_version,),
            )
            if self._own_transaction():
                self.conn.commit()

    def get_playbook_aggregation_invalidations(
        self, agent_version: str, *, limit: int
    ) -> list[PlaybookAggregationInvalidation]:
        if limit <= 0:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT invalidation_id, operation, entity_id, source_ids "
                "FROM playbook_aggregation_invalidation "
                "WHERE agent_version=? AND processed_at IS NULL "
                "ORDER BY invalidation_id LIMIT ?",
                (agent_version, limit),
            ).fetchall()
        return [
            PlaybookAggregationInvalidation(
                invalidation_id=int(row[0]),
                agent_version=agent_version,
                operation=str(row[1]),
                entity_id=int(row[2]),
                source_ids=tuple(int(value) for value in json.loads(row[3] or "[]")),
            )
            for row in rows
        ]

    def apply_playbook_aggregation_invalidations(
        self,
        claim: PlaybookAggregationClaim,
        invalidation_ids: list[int],
    ) -> bool:
        if not invalidation_ids:
            return True
        if self._own_transaction():
            with self.commit_scope():
                return self.apply_playbook_aggregation_invalidations(
                    claim, invalidation_ids
                )
        with self._lock:
            if not self._aggregation_claim_is_live(claim):
                return False
            placeholders = ",".join("?" for _ in invalidation_ids)
            rows = self.conn.execute(
                "SELECT entity_id, source_ids FROM playbook_aggregation_invalidation "
                f"WHERE invalidation_id IN ({placeholders}) AND agent_version=? "
                "AND processed_at IS NULL",
                (*invalidation_ids, claim.agent_version),
            ).fetchall()
            affected = {
                int(value)
                for row in rows
                for value in [int(row[0]), *json.loads(row[1] or "[]")]
            }
            if affected:
                item_placeholders = ",".join("?" for _ in affected)
                cluster_rows = self.conn.execute(
                    "SELECT DISTINCT cluster_id FROM playbook_aggregation_item "
                    f"WHERE agent_version=? AND user_playbook_id IN ({item_placeholders}) "
                    "AND cluster_id IS NOT NULL",
                    (claim.agent_version, *sorted(affected)),
                ).fetchall()
                cluster_ids = [str(row[0]) for row in cluster_rows]
                if cluster_ids:
                    cluster_placeholders = ",".join("?" for _ in cluster_ids)
                    index_rows = self.conn.execute(
                        "SELECT index_rowid FROM playbook_aggregation_cluster "
                        f"WHERE cluster_id IN ({cluster_placeholders})",
                        cluster_ids,
                    ).fetchall()
                    self.conn.execute(
                        "UPDATE playbook_aggregation_item SET disposition='residual', "
                        "reason='cluster_invalidated', attempt_count=0, "
                        "last_attempt_at=NULL, "
                        "updated_at=unixepoch() WHERE agent_version=? "
                        f"AND cluster_id IN ({cluster_placeholders})",
                        (claim.agent_version, *cluster_ids),
                    )
                    for index_row in index_rows:
                        self.conn.execute(
                            "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                            (int(index_row[0]),),
                        )
                    self.conn.execute(
                        "UPDATE playbook_aggregation_cluster SET state='rebuilding', "
                        "dirty=1, centroid=NULL, vector_sum=NULL, member_count=0, "
                        "rebuild_cursor=0 "
                        f"WHERE cluster_id IN ({cluster_placeholders})",
                        cluster_ids,
                    )
                self.conn.execute(
                    "DELETE FROM playbook_aggregation_item "
                    f"WHERE agent_version=? AND user_playbook_id IN ({item_placeholders})",
                    (claim.agent_version, *sorted(affected)),
                )
            self.conn.execute(
                "UPDATE playbook_aggregation_invalidation SET processed_at=unixepoch() "
                f"WHERE invalidation_id IN ({placeholders}) AND agent_version=? "
                "AND processed_at IS NULL",
                (*invalidation_ids, claim.agent_version),
            )
            return True

    def get_playbook_aggregation_replacement_agent_ids(
        self, agent_version: str, user_playbook_ids: list[int]
    ) -> list[int]:
        if not user_playbook_ids:
            return []
        placeholders = ",".join("?" for _ in user_playbook_ids)
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT c.agent_playbook_id FROM playbook_aggregation_item i "
                "JOIN playbook_aggregation_cluster c ON c.cluster_id=i.cluster_id "
                f"WHERE i.agent_version=? AND i.user_playbook_id IN ({placeholders}) "
                "AND c.state='rebuilding' AND c.agent_playbook_id IS NOT NULL "
                "ORDER BY c.agent_playbook_id",
                (agent_version, *user_playbook_ids),
            ).fetchall()
        return [int(row[0]) for row in rows]

    def delete_orphaned_playbook_aggregation_clusters(self, agent_version: str) -> None:
        with self._lock:
            rows = self.conn.execute(
                "SELECT index_rowid FROM playbook_aggregation_cluster c "
                "WHERE c.agent_version=? AND c.state='rebuilding' "
                "AND NOT EXISTS (SELECT 1 FROM playbook_aggregation_item i "
                "WHERE i.cluster_id=c.cluster_id)",
                (agent_version,),
            ).fetchall()
            self.conn.executemany(
                "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                [(int(row[0]),) for row in rows],
            )
            self.conn.execute(
                "DELETE FROM playbook_aggregation_cluster WHERE agent_version=? "
                "AND state='rebuilding' AND NOT EXISTS ("
                "SELECT 1 FROM playbook_aggregation_item i "
                "WHERE i.cluster_id=playbook_aggregation_cluster.cluster_id)",
                (agent_version,),
            )
            if self._own_transaction():
                self.conn.commit()

    def find_nearest_playbook_aggregation_clusters(
        self,
        agent_version: str,
        candidates: list[tuple[int, list[float]]],
        *,
        embedding_model: str,
        limit: int,
    ) -> dict[int, PlaybookAggregationClusterMatch]:
        if not self._has_sqlite_vec:
            raise RuntimeError(
                "blocked_missing_vector_index: install sqlite-vec to enable "
                "incremental playbook aggregation"
            )
        if limit <= 0 or not candidates:
            return {}
        candidate_ids = [item_id for item_id, _embedding in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("aggregation candidate IDs must be unique")
        # sqlite-vec rejects KNN queries whose k exceeds 4096.  The caller's
        # limit is a run-level safety budget, so clamp only this ANN candidate
        # query instead of shrinking the aggregation batch itself.
        candidate_limit = min(limit, 4096)
        matches: dict[int, PlaybookAggregationClusterMatch] = {}
        with self._lock:
            for item_id, embedding in candidates:
                row = self.conn.execute(
                    "SELECT c.cluster_id, v.distance FROM ("
                    "SELECT rowid, distance FROM playbook_aggregation_clusters_vec "
                    "WHERE embedding MATCH ? ORDER BY distance LIMIT ?) v "
                    "JOIN playbook_aggregation_cluster c ON c.index_rowid=v.rowid "
                    "WHERE c.agent_version=? AND c.embedding_model=? "
                    "AND c.embedding_dimension=? AND c.state='active' "
                    "ORDER BY v.distance, c.cluster_id LIMIT 1",
                    (
                        json.dumps(embedding),
                        candidate_limit,
                        agent_version,
                        embedding_model,
                        len(embedding),
                    ),
                ).fetchone()
                if row is not None:
                    matches[item_id] = PlaybookAggregationClusterMatch(
                        str(row[0]), 1.0 - float(row[1])
                    )
        return matches

    def create_playbook_aggregation_cluster(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        agent_playbook_id: int | None,
        embeddings: list[list[float]],
        embedding_model: str,
    ) -> None:
        if not embeddings:
            raise ValueError("cluster embeddings must be non-empty")
        dimension = len(embeddings[0])
        if any(len(value) != dimension for value in embeddings):
            raise ValueError("cluster embeddings must share one dimension")
        vector_sum = [sum(values) for values in zip(*embeddings, strict=True)]
        centroid = [value / len(embeddings) for value in vector_sum]
        with self._lock:
            existing = self.conn.execute(
                "SELECT index_rowid FROM playbook_aggregation_cluster "
                "WHERE cluster_id=?",
                (cluster_id,),
            ).fetchone()
            if existing is None:
                next_rowid = int(
                    self.conn.execute(
                        "SELECT COALESCE(max(index_rowid), 0)+1 "
                        "FROM playbook_aggregation_cluster"
                    ).fetchone()[0]
                )
                self.conn.execute(
                    "INSERT INTO playbook_aggregation_cluster "
                    "(cluster_id, index_rowid, agent_version, agent_playbook_id, "
                    "centroid, vector_sum, member_count, embedding_model, "
                    "embedding_dimension) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        cluster_id,
                        next_rowid,
                        agent_version,
                        agent_playbook_id,
                        json.dumps(centroid),
                        json.dumps(vector_sum),
                        len(embeddings),
                        embedding_model,
                        dimension,
                    ),
                )
            else:
                next_rowid = int(existing[0])
                self.conn.execute(
                    "UPDATE playbook_aggregation_cluster SET agent_version=?, "
                    "agent_playbook_id=?, centroid=?, vector_sum=?, member_count=?, "
                    "embedding_model=?, embedding_dimension=?, normalization='l2', "
                    "state='active', dirty=0, rebuild_cursor=0 WHERE cluster_id=?",
                    (
                        agent_version,
                        agent_playbook_id,
                        json.dumps(centroid),
                        json.dumps(vector_sum),
                        len(embeddings),
                        embedding_model,
                        dimension,
                        cluster_id,
                    ),
                )
            self.conn.execute(
                "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                (next_rowid,),
            )
            self.conn.execute(
                "INSERT INTO playbook_aggregation_clusters_vec(rowid, embedding) "
                "VALUES (?, ?)",
                (next_rowid, json.dumps(centroid)),
            )
            if self._own_transaction():
                self.conn.commit()

    def attach_playbook_aggregation_items(
        self,
        *,
        agent_version: str,
        attachments: list[tuple[int, str, list[float]]],
    ) -> None:
        if not attachments:
            return
        item_ids = [item_id for item_id, _cluster_id, _embedding in attachments]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("aggregation attachment IDs must be unique")
        grouped: dict[str, list[tuple[int, list[float]]]] = {}
        for item_id, cluster_id, embedding in attachments:
            grouped.setdefault(cluster_id, []).append((item_id, embedding))
        with self._lock:
            own_transaction = self._own_transaction()
            try:
                cluster_rows: dict[str, sqlite3.Row] = {}
                cluster_ids = list(grouped)
                for offset in range(0, len(cluster_ids), 500):
                    page = cluster_ids[offset : offset + 500]
                    placeholders = ",".join("?" for _value in page)
                    rows = self.conn.execute(
                        "SELECT cluster_id, index_rowid, vector_sum, member_count, "
                        "embedding_dimension FROM playbook_aggregation_cluster "
                        f"WHERE agent_version=? AND state='active' AND cluster_id IN ({placeholders})",
                        (agent_version, *page),
                    ).fetchall()
                    cluster_rows.update({str(row[0]): row for row in rows})
                if set(cluster_rows) != set(grouped):
                    raise RuntimeError(
                        "aggregation cluster is not active for agent version"
                    )

                cluster_updates: list[tuple[str, str, int, str]] = []
                vector_updates: list[tuple[int, str]] = []
                for cluster_id, members in grouped.items():
                    row = cluster_rows[cluster_id]
                    vector_sum = [float(value) for value in json.loads(row[2])]
                    dimension = int(row[4])
                    if len(vector_sum) != dimension or any(
                        len(embedding) != dimension for _item_id, embedding in members
                    ):
                        raise ValueError("aggregation embedding dimension changed")
                    batch_sum = [
                        sum(embedding[index] for _item_id, embedding in members)
                        for index in range(dimension)
                    ]
                    updated_sum = [
                        left + right
                        for left, right in zip(vector_sum, batch_sum, strict=True)
                    ]
                    count = int(row[3]) + len(members)
                    centroid = [value / count for value in updated_sum]
                    cluster_updates.append(
                        (
                            json.dumps(updated_sum),
                            json.dumps(centroid),
                            count,
                            cluster_id,
                        )
                    )
                    vector_updates.append((int(row[1]), json.dumps(centroid)))

                before_changes = self.conn.total_changes
                self.conn.executemany(
                    "UPDATE playbook_aggregation_item SET disposition='cluster_member', "
                    "cluster_id=?, reason='centroid_match', updated_at=unixepoch() "
                    "WHERE agent_version=? AND user_playbook_id=? "
                    "AND disposition='residual'",
                    [
                        (cluster_id, agent_version, item_id)
                        for item_id, cluster_id, _embedding in attachments
                    ],
                )
                if self.conn.total_changes - before_changes != len(attachments):
                    raise RuntimeError("aggregation residual attachment lost its state")
                self.conn.executemany(
                    "UPDATE playbook_aggregation_cluster SET vector_sum=?, centroid=?, "
                    "member_count=? WHERE cluster_id=? AND agent_version=?",
                    [(*update, agent_version) for update in cluster_updates],
                )
                self.conn.executemany(
                    "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                    [(rowid,) for rowid, _centroid in vector_updates],
                )
                self.conn.executemany(
                    "INSERT INTO playbook_aggregation_clusters_vec(rowid, embedding) "
                    "VALUES (?, ?)",
                    vector_updates,
                )
                if own_transaction:
                    self.conn.commit()
            except Exception:
                if own_transaction:
                    self.conn.rollback()
                raise
