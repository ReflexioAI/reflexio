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
    PlaybookAggregationRebuildSample,
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
    bootstrap_status TEXT NOT NULL DEFAULT 'pending',
    intake_floor_user_playbook_id INTEGER NOT NULL DEFAULT 0
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
    rebuild_cursor INTEGER NOT NULL DEFAULT 0,
    rebuild_attempt_count INTEGER NOT NULL DEFAULT 0,
    rebuild_next_attempt_at INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_cluster_version
    ON playbook_aggregation_cluster(agent_version, state, cluster_id);
CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_cluster_agent
    ON playbook_aggregation_cluster(agent_playbook_id)
    WHERE agent_playbook_id IS NOT NULL;

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
CREATE INDEX IF NOT EXISTS idx_user_playbooks_aggregation_rebuild
    ON user_playbooks(agent_version, created_at DESC, user_playbook_id DESC)
    WHERE status IS NULL AND trim(content) <> '' AND trim(agent_version) <> '';
CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_item_rebuild
    ON playbook_aggregation_item(agent_version, cluster_id, user_playbook_id DESC)
    WHERE disposition = 'residual' AND cluster_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS playbook_aggregation_invalidation (
    invalidation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_version TEXT NOT NULL,
    operation TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    source_ids TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL DEFAULT (unixepoch()),
    processed_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_invalidation_pending
    ON playbook_aggregation_invalidation(agent_version, invalidation_id)
    WHERE processed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_invalidation_retention
    ON playbook_aggregation_invalidation(processed_at)
    WHERE processed_at IS NOT NULL;

DROP TRIGGER IF EXISTS capture_playbook_aggregation_hard_delete;
CREATE TRIGGER capture_playbook_aggregation_hard_delete
BEFORE DELETE ON user_playbooks
WHEN OLD.status IS NULL AND trim(OLD.agent_version) <> ''
BEGIN
    INSERT INTO playbook_aggregation_invalidation(
        agent_version, operation, entity_id, source_ids
    ) VALUES (OLD.agent_version, 'hard_delete', OLD.user_playbook_id, '[]');
    INSERT INTO playbook_aggregation_state(
        agent_version, pending, next_attempt_at
    ) VALUES (OLD.agent_version, 1, unixepoch())
    ON CONFLICT(agent_version) DO UPDATE SET pending=1;
END;

DROP TRIGGER IF EXISTS retire_playbook_aggregation_cluster_on_agent_update;
CREATE TRIGGER retire_playbook_aggregation_cluster_on_agent_update
AFTER UPDATE OF status, playbook_status, content, trigger, rationale, embedding
ON agent_playbooks
WHEN (
    (NEW.status IS NOT OLD.status AND NEW.status IS NOT NULL)
    OR (
        NEW.playbook_status IS NOT OLD.playbook_status
        AND NEW.playbook_status = 'rejected'
    )
    OR NEW.content IS NOT OLD.content
    OR NEW.trigger IS NOT OLD.trigger
    OR NEW.rationale IS NOT OLD.rationale
    OR NEW.embedding IS NOT OLD.embedding
)
BEGIN
    UPDATE playbook_aggregation_item
    SET disposition = 'residual', cluster_id = NULL,
        reason = 'agent_playbook_changed', attempt_count = 0,
        last_attempt_at = NULL, updated_at = unixepoch()
    WHERE cluster_id IN (
        SELECT cluster_id FROM playbook_aggregation_cluster
        WHERE agent_playbook_id = OLD.agent_playbook_id
    );
    INSERT OR IGNORE INTO playbook_aggregation_state(
        agent_version, pending, next_attempt_at
    )
    SELECT DISTINCT agent_version, 1, unixepoch()
    FROM playbook_aggregation_cluster
    WHERE agent_playbook_id = OLD.agent_playbook_id;
    UPDATE playbook_aggregation_state
    SET pending = 1
    WHERE agent_version IN (
        SELECT agent_version FROM playbook_aggregation_cluster
        WHERE agent_playbook_id = OLD.agent_playbook_id
    );
    DELETE FROM playbook_aggregation_cluster
    WHERE agent_playbook_id = OLD.agent_playbook_id;
END;

DROP TRIGGER IF EXISTS retire_playbook_aggregation_cluster_on_agent_delete;
CREATE TRIGGER retire_playbook_aggregation_cluster_on_agent_delete
BEFORE DELETE ON agent_playbooks
BEGIN
    UPDATE playbook_aggregation_item
    SET disposition = 'residual', cluster_id = NULL,
        reason = 'agent_playbook_deleted', attempt_count = 0,
        last_attempt_at = NULL, updated_at = unixepoch()
    WHERE cluster_id IN (
        SELECT cluster_id FROM playbook_aggregation_cluster
        WHERE agent_playbook_id = OLD.agent_playbook_id
    );
    INSERT OR IGNORE INTO playbook_aggregation_state(
        agent_version, pending, next_attempt_at
    )
    SELECT DISTINCT agent_version, 1, unixepoch()
    FROM playbook_aggregation_cluster
    WHERE agent_playbook_id = OLD.agent_playbook_id;
    UPDATE playbook_aggregation_state
    SET pending = 1
    WHERE agent_version IN (
        SELECT agent_version FROM playbook_aggregation_cluster
        WHERE agent_playbook_id = OLD.agent_playbook_id
    );
    DELETE FROM playbook_aggregation_cluster
    WHERE agent_playbook_id = OLD.agent_playbook_id;
END;
"""


def init_playbook_aggregation_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(AGGREGATION_DDL)
    state_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(playbook_aggregation_state)")
    }
    if "intake_floor_user_playbook_id" not in state_columns:
        conn.execute(
            "ALTER TABLE playbook_aggregation_state ADD COLUMN "
            "intake_floor_user_playbook_id INTEGER NOT NULL DEFAULT 0"
        )
    cluster_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(playbook_aggregation_cluster)")
    }
    if "rebuild_attempt_count" not in cluster_columns:
        conn.execute(
            "ALTER TABLE playbook_aggregation_cluster ADD COLUMN "
            "rebuild_attempt_count INTEGER NOT NULL DEFAULT 0"
        )
    if "rebuild_next_attempt_at" not in cluster_columns:
        conn.execute(
            "ALTER TABLE playbook_aggregation_cluster ADD COLUMN "
            "rebuild_next_attempt_at INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_playbook_aggregation_cluster_rebuild_due "
        "ON playbook_aggregation_cluster(agent_version, rebuild_next_attempt_at, "
        "cluster_id) WHERE state = 'rebuilding'"
    )


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
                "ON CONFLICT(agent_version) DO UPDATE SET pending=1",
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
            if self._has_sqlite_vec:
                orphaned_vectors = self.conn.execute(
                    "SELECT rowid FROM playbook_aggregation_clusters_vec "
                    "WHERE rowid NOT IN (SELECT index_rowid FROM "
                    "playbook_aggregation_cluster WHERE index_rowid IS NOT NULL) "
                    "LIMIT ?",
                    (limit,),
                ).fetchall()
                self.conn.executemany(
                    "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                    [(int(row[0]),) for row in orphaned_vectors],
                )
            self.conn.execute(
                "DELETE FROM playbook_aggregation_invalidation "
                "WHERE processed_at IS NOT NULL AND processed_at < unixepoch()-?",
                (AGGREGATION_INVALIDATION_RETENTION_SECONDS,),
            )
            rows = self.conn.execute(
                "WITH work(agent_version) AS ("
                "SELECT p.agent_version FROM user_playbooks p LEFT JOIN "
                "playbook_aggregation_state ps ON ps.agent_version=p.agent_version "
                "WHERE p.status IS NULL "
                "AND trim(p.content) <> '' AND trim(p.agent_version) <> '' "
                "AND p.user_playbook_id >= "
                "COALESCE(ps.intake_floor_user_playbook_id, 0) "
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
                "ON CONFLICT(agent_version) DO UPDATE SET pending=1",
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
        self, agent_version: str, *, limit: int, window_limit: int = 20_000
    ) -> list[int]:
        if limit <= 0 or window_limit <= 0:
            return []
        with self._lock:
            self.conn.execute(
                "INSERT INTO playbook_aggregation_state "
                "(agent_version, pending, next_attempt_at) VALUES (?, 1, unixepoch()) "
                "ON CONFLICT(agent_version) DO NOTHING",
                (agent_version,),
            )
            old_floor = int(
                self.conn.execute(
                    "SELECT intake_floor_user_playbook_id FROM "
                    "playbook_aggregation_state WHERE agent_version=?",
                    (agent_version,),
                ).fetchone()[0]
            )
            cutoff_row = self.conn.execute(
                "SELECT p.user_playbook_id FROM user_playbooks p LEFT JOIN "
                "playbook_aggregation_item i ON i.agent_version=? AND "
                "i.user_playbook_id=p.user_playbook_id WHERE p.agent_version=? "
                "AND p.status IS NULL AND trim(p.content) <> '' "
                "AND p.user_playbook_id>=? AND (i.user_playbook_id IS NULL OR "
                "(i.disposition='residual' AND i.cluster_id IS NULL)) "
                "ORDER BY p.user_playbook_id DESC LIMIT 1 OFFSET ?",
                (agent_version, agent_version, old_floor, window_limit - 1),
            ).fetchone()
            intake_floor = max(
                old_floor, int(cutoff_row[0]) if cutoff_row is not None else old_floor
            )
            if intake_floor != old_floor:
                self.conn.execute(
                    "UPDATE playbook_aggregation_state SET "
                    "intake_floor_user_playbook_id=? WHERE agent_version=?",
                    (intake_floor, agent_version),
                )
                self.conn.execute(
                    "UPDATE playbook_aggregation_item SET "
                    "disposition='terminal_noop', cluster_id=NULL, "
                    "reason='outside_recent_clustering_window', "
                    "updated_at=unixepoch() WHERE agent_version=? "
                    "AND disposition='residual' AND cluster_id IS NULL "
                    "AND user_playbook_id<?",
                    (agent_version, intake_floor),
                )
            rows = self.conn.execute(
                "SELECT p.user_playbook_id FROM user_playbooks p "
                "WHERE p.agent_version=? AND p.status IS NULL "
                "AND trim(p.content) <> '' AND p.user_playbook_id>=? "
                "AND NOT EXISTS ("
                " SELECT 1 FROM playbook_aggregation_item i "
                " WHERE i.agent_version=? "
                " AND i.user_playbook_id=p.user_playbook_id) "
                "ORDER BY p.user_playbook_id DESC LIMIT ?",
                (agent_version, intake_floor, agent_version, limit),
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
        centroid_embedding: list[float],
        member_embeddings: list[tuple[int, list[float]]],
        embedding_model: str,
        embedding_dimension: int,
        rebuild_cursor: int,
        complete: bool,
    ) -> None:
        if len(centroid_embedding) != embedding_dimension or any(
            len(value) != embedding_dimension for _, value in member_embeddings
        ):
            raise ValueError("legacy cluster embedding dimension changed")
        with self._lock:
            row = self.conn.execute(
                "SELECT index_rowid, member_count, state, "
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
                member_count = 0
            else:
                if str(row[2]) == "active":
                    return
                if row[3] != embedding_model or int(row[4]) != embedding_dimension:
                    raise RuntimeError("legacy cluster embedding provenance changed")
                index_rowid = int(row[0])
                member_count = int(row[1])
            for user_playbook_id, _embedding in member_embeddings:
                inserted = self.conn.execute(
                    "INSERT OR IGNORE INTO playbook_aggregation_item "
                    "(agent_version, user_playbook_id, disposition, cluster_id, reason) "
                    "VALUES (?, ?, 'cluster_member', ?, 'legacy_adopted')",
                    (agent_version, user_playbook_id, cluster_id),
                )
                if inserted.rowcount == 1:
                    member_count += 1
            centroid = centroid_embedding if complete and member_count else None
            self.conn.execute(
                "UPDATE playbook_aggregation_cluster SET vector_sum=NULL, centroid=?, "
                "member_count=?, rebuild_cursor=?, state=? WHERE cluster_id=?",
                (
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
            self.conn.execute(
                "UPDATE playbook_aggregation_state SET "
                "intake_floor_user_playbook_id=0 WHERE agent_version=?",
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
                "SELECT i.user_playbook_id FROM playbook_aggregation_item i "
                "WHERE i.agent_version=? AND i.disposition='residual' "
                "AND i.attempt_count=0 AND NOT EXISTS (SELECT 1 FROM "
                "playbook_aggregation_cluster c WHERE c.agent_version=i.agent_version "
                "AND c.cluster_id=i.cluster_id AND c.state='rebuilding' "
                "AND c.rebuild_next_attempt_at>unixepoch()) "
                "ORDER BY i.user_playbook_id LIMIT ?",
                (agent_version, fresh_quota),
            ).fetchall()
            retry_rows = self.conn.execute(
                "SELECT i.user_playbook_id FROM playbook_aggregation_item i "
                "WHERE i.agent_version=? AND i.disposition='residual' "
                "AND i.attempt_count>0 AND i.last_attempt_at+min(?, ?*(1 << "
                "min(max(i.attempt_count-1, 0), 6))) <= unixepoch() "
                "AND NOT EXISTS (SELECT 1 FROM playbook_aggregation_cluster c "
                "WHERE c.agent_version=i.agent_version AND c.cluster_id=i.cluster_id "
                "AND c.state='rebuilding' AND c.rebuild_next_attempt_at>unixepoch()) "
                "ORDER BY i.last_attempt_at, i.user_playbook_id LIMIT ?",
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
                    "SELECT i.user_playbook_id FROM playbook_aggregation_item i "
                    "WHERE i.agent_version=? AND i.disposition='residual' "
                    "AND (i.attempt_count=0 OR i.last_attempt_at IS NULL OR "
                    "i.last_attempt_at+min(?, ?*(1 << min(max(i.attempt_count-1, 0), "
                    "6))) <= unixepoch()) AND NOT EXISTS (SELECT 1 FROM "
                    "playbook_aggregation_cluster c WHERE "
                    "c.agent_version=i.agent_version AND c.cluster_id=i.cluster_id "
                    "AND c.state='rebuilding' AND "
                    "c.rebuild_next_attempt_at>unixepoch()) "
                    f"{exclusion} ORDER BY COALESCE(i.last_attempt_at, 0), "
                    "i.user_playbook_id LIMIT ?",
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
                    "SELECT count(*) FROM user_playbooks p LEFT JOIN "
                    "playbook_aggregation_state s ON s.agent_version=p.agent_version "
                    "WHERE p.agent_version=? "
                    "AND p.status IS NULL AND trim(p.content) <> '' AND NOT EXISTS ("
                    "SELECT 1 FROM playbook_aggregation_item i "
                    "WHERE i.agent_version=? "
                    "AND i.user_playbook_id=p.user_playbook_id) "
                    "AND p.user_playbook_id>=COALESCE("
                    "s.intake_floor_user_playbook_id, 0)",
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
                "WHEN c.state='rebuilding' THEN c.rebuild_next_attempt_at-unixepoch() "
                "WHEN i.attempt_count <= 0 OR i.last_attempt_at IS NULL THEN 0 "
                "ELSE i.last_attempt_at + MIN(?, ? * (1 << MIN(MAX("
                "i.attempt_count - 1, 0), 6))) - unixepoch() END)), 0) "
                "FROM playbook_aggregation_item i LEFT JOIN "
                "playbook_aggregation_cluster c ON c.agent_version=i.agent_version "
                "AND c.cluster_id=i.cluster_id WHERE i.agent_version=? "
                "AND i.disposition='residual'",
                (
                    AGGREGATION_RETRY_MAX_SECONDS,
                    AGGREGATION_RETRY_BASE_SECONDS,
                    agent_version,
                ),
            ).fetchone()[0]
            repair_retry_after = self.conn.execute(
                "SELECT COALESCE(MAX(0, MIN(rebuild_next_attempt_at-unixepoch())), 0) "
                "FROM playbook_aggregation_cluster WHERE agent_version=? "
                "AND state='rebuilding'",
                (agent_version,),
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
            repair_retry_after_seconds=int(repair_retry_after),
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
                "ON CONFLICT(agent_version) DO UPDATE SET pending=1",
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
                "SELECT operation, entity_id, source_ids "
                "FROM playbook_aggregation_invalidation "
                f"WHERE invalidation_id IN ({placeholders}) AND agent_version=? "
                "AND processed_at IS NULL",
                (*invalidation_ids, claim.agent_version),
            ).fetchall()
            affected = {
                int(value)
                for row in rows
                for value in [int(row[1]), *json.loads(row[2] or "[]")]
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
                revision_cluster_by_entity: dict[int, str] = {}
                for operation, entity_id, source_ids_json in rows:
                    if operation != "revise":
                        continue
                    source_ids = [
                        int(value) for value in json.loads(source_ids_json or "[]")
                    ]
                    if not source_ids:
                        continue
                    source_placeholders = ",".join("?" for _ in source_ids)
                    source_clusters = self.conn.execute(
                        "SELECT DISTINCT cluster_id FROM playbook_aggregation_item "
                        f"WHERE agent_version=? AND user_playbook_id IN ({source_placeholders}) "
                        "AND cluster_id IS NOT NULL",
                        (claim.agent_version, *source_ids),
                    ).fetchall()
                    if len(source_clusters) == 1:
                        revision_cluster_by_entity[int(entity_id)] = str(
                            source_clusters[0][0]
                        )
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
                        "rebuild_cursor=0, rebuild_attempt_count=0, "
                        "rebuild_next_attempt_at=0 "
                        f"WHERE cluster_id IN ({cluster_placeholders})",
                        cluster_ids,
                    )
                self.conn.execute(
                    "DELETE FROM playbook_aggregation_item "
                    f"WHERE agent_version=? AND user_playbook_id IN ({item_placeholders})",
                    (claim.agent_version, *sorted(affected)),
                )
                for entity_id, cluster_id in revision_cluster_by_entity.items():
                    self.conn.execute(
                        "INSERT INTO playbook_aggregation_item "
                        "(agent_version, user_playbook_id, disposition, cluster_id, reason) "
                        "SELECT ?, p.user_playbook_id, 'residual', ?, 'revision_rebuild' "
                        "FROM user_playbooks p WHERE p.user_playbook_id=? "
                        "AND p.agent_version=? AND p.status IS NULL "
                        "AND trim(p.content) <> '' ON CONFLICT(agent_version, "
                        "user_playbook_id) DO UPDATE SET disposition='residual', "
                        "cluster_id=excluded.cluster_id, reason=excluded.reason, "
                        "attempt_count=0, last_attempt_at=NULL, updated_at=unixepoch()",
                        (
                            claim.agent_version,
                            cluster_id,
                            entity_id,
                            claim.agent_version,
                        ),
                    )
            self.conn.execute(
                "UPDATE playbook_aggregation_invalidation SET processed_at=unixepoch() "
                f"WHERE invalidation_id IN ({placeholders}) AND agent_version=? "
                "AND processed_at IS NULL",
                (*invalidation_ids, claim.agent_version),
            )
            return True

    def get_playbook_aggregation_rebuild_cluster_ids(
        self, agent_version: str, user_playbook_ids: list[int]
    ) -> dict[int, str]:
        if not user_playbook_ids:
            return {}
        placeholders = ",".join("?" for _ in user_playbook_ids)
        with self._lock:
            rows = self.conn.execute(
                "SELECT i.user_playbook_id, i.cluster_id FROM "
                "playbook_aggregation_item i JOIN playbook_aggregation_cluster c "
                "ON c.cluster_id=i.cluster_id WHERE i.agent_version=? "
                f"AND i.user_playbook_id IN ({placeholders}) "
                "AND i.disposition='residual' AND c.state='rebuilding' "
                "AND c.rebuild_next_attempt_at<=unixepoch()",
                (agent_version, *user_playbook_ids),
            ).fetchall()
        return {int(row[0]): str(row[1]) for row in rows}

    def get_playbook_aggregation_rebuild_samples(
        self, agent_version: str, cluster_ids: list[str], *, limit_per_cluster: int
    ) -> list[PlaybookAggregationRebuildSample]:
        if not cluster_ids or limit_per_cluster <= 0:
            return []
        placeholders = ",".join("?" for _ in cluster_ids)
        with self._lock:
            clusters = self.conn.execute(
                "SELECT cluster_id, agent_playbook_id FROM "
                "playbook_aggregation_cluster WHERE agent_version=? "
                "AND state='rebuilding' AND agent_playbook_id IS NOT NULL "
                "AND rebuild_next_attempt_at<=unixepoch() "
                f"AND cluster_id IN ({placeholders}) ORDER BY cluster_id",
                (agent_version, *cluster_ids),
            ).fetchall()
            samples: list[PlaybookAggregationRebuildSample] = []
            for cluster_id, agent_playbook_id in clusters:
                rows = self.conn.execute(
                    "SELECT p.user_playbook_id FROM user_playbooks p INDEXED BY "
                    "idx_user_playbooks_aggregation_rebuild WHERE p.agent_version=? "
                    "AND trim(p.agent_version) <> '' AND p.status IS NULL AND "
                    "trim(p.content) <> '' AND EXISTS (SELECT 1 FROM "
                    "playbook_aggregation_item i WHERE i.agent_version=? AND "
                    "i.user_playbook_id=p.user_playbook_id AND i.cluster_id=? AND "
                    "i.disposition='residual') ORDER BY p.created_at DESC, "
                    "p.user_playbook_id DESC LIMIT ?",
                    (agent_version, agent_version, cluster_id, limit_per_cluster),
                ).fetchall()
                if rows:
                    samples.append(
                        PlaybookAggregationRebuildSample(
                            cluster_id=str(cluster_id),
                            agent_playbook_id=int(agent_playbook_id),
                            member_ids=tuple(int(row[0]) for row in rows),
                        )
                    )
        return samples

    def defer_playbook_aggregation_cluster_rebuild(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        expected_agent_playbook_id: int,
        reason: str,
    ) -> None:
        if self._own_transaction():
            with self.commit_scope():
                self.defer_playbook_aggregation_cluster_rebuild(
                    cluster_id=cluster_id,
                    agent_version=agent_version,
                    expected_agent_playbook_id=expected_agent_playbook_id,
                    reason=reason,
                )
            return
        with self._lock:
            updated = self.conn.execute(
                "UPDATE playbook_aggregation_cluster SET "
                "rebuild_attempt_count=rebuild_attempt_count+1, "
                "rebuild_next_attempt_at=unixepoch()+min(?, ?*(1 << "
                "min(rebuild_attempt_count, 6))) WHERE cluster_id=? "
                "AND agent_version=? AND state='rebuilding' "
                "AND agent_playbook_id=?",
                (
                    AGGREGATION_RETRY_MAX_SECONDS,
                    AGGREGATION_RETRY_BASE_SECONDS,
                    cluster_id,
                    agent_version,
                    expected_agent_playbook_id,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("aggregation rebuilding cluster changed")
            self.conn.execute(
                "UPDATE playbook_aggregation_item SET reason=?, attempt_count=0, "
                "last_attempt_at=NULL, updated_at=unixepoch() WHERE agent_version=? "
                "AND cluster_id=? AND disposition='residual'",
                (reason, agent_version, cluster_id),
            )

    def complete_playbook_aggregation_cluster_rebuild(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        expected_agent_playbook_id: int,
        replacement_agent_playbook_id: int,
        centroid_embedding: list[float],
        embedding_model: str,
    ) -> int:
        if not centroid_embedding:
            raise ValueError("cluster centroid embedding must be non-empty")
        if self._own_transaction():
            with self.commit_scope():
                return self.complete_playbook_aggregation_cluster_rebuild(
                    cluster_id=cluster_id,
                    agent_version=agent_version,
                    expected_agent_playbook_id=expected_agent_playbook_id,
                    replacement_agent_playbook_id=replacement_agent_playbook_id,
                    centroid_embedding=centroid_embedding,
                    embedding_model=embedding_model,
                )
        with self._lock:
            row = self.conn.execute(
                "SELECT index_rowid, embedding_dimension FROM "
                "playbook_aggregation_cluster WHERE cluster_id=? AND agent_version=? "
                "AND state='rebuilding' AND agent_playbook_id=?",
                (cluster_id, agent_version, expected_agent_playbook_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("aggregation rebuilding cluster changed")
            if int(row[1]) != len(centroid_embedding):
                raise ValueError("aggregation embedding dimension changed")
            member_count = int(
                self.conn.execute(
                    "SELECT count(*) FROM playbook_aggregation_item WHERE "
                    "agent_version=? AND cluster_id=? AND disposition='residual'",
                    (agent_version, cluster_id),
                ).fetchone()[0]
            )
            if member_count <= 0:
                raise RuntimeError("aggregation rebuilding cluster has no members")
            self.conn.execute(
                "UPDATE playbook_aggregation_item SET disposition='cluster_member', "
                "reason='rebuild_complete', updated_at=unixepoch() WHERE "
                "agent_version=? AND cluster_id=? AND disposition='residual'",
                (agent_version, cluster_id),
            )
            updated = self.conn.execute(
                "UPDATE playbook_aggregation_cluster SET agent_playbook_id=?, "
                "centroid=?, vector_sum=NULL, member_count=?, embedding_model=?, "
                "state='active', dirty=0, rebuild_cursor=0, "
                "rebuild_attempt_count=0, rebuild_next_attempt_at=0 "
                "WHERE cluster_id=? "
                "AND agent_version=? AND state='rebuilding' AND agent_playbook_id=?",
                (
                    replacement_agent_playbook_id,
                    json.dumps(centroid_embedding),
                    member_count,
                    embedding_model,
                    cluster_id,
                    agent_version,
                    expected_agent_playbook_id,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("aggregation rebuilding cluster changed")
            rowid = int(row[0])
            self.conn.execute(
                "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?", (rowid,)
            )
            self.conn.execute(
                "INSERT INTO playbook_aggregation_clusters_vec(rowid, embedding) "
                "VALUES (?, ?)",
                (rowid, json.dumps(centroid_embedding)),
            )
            return member_count

    def discard_playbook_aggregation_cluster_rebuild(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        expected_agent_playbook_id: int,
        reason: str,
    ) -> int:
        if self._own_transaction():
            with self.commit_scope():
                return self.discard_playbook_aggregation_cluster_rebuild(
                    cluster_id=cluster_id,
                    agent_version=agent_version,
                    expected_agent_playbook_id=expected_agent_playbook_id,
                    reason=reason,
                )
        with self._lock:
            row = self.conn.execute(
                "SELECT index_rowid FROM playbook_aggregation_cluster WHERE "
                "cluster_id=? AND agent_version=? AND state='rebuilding' "
                "AND agent_playbook_id=?",
                (cluster_id, agent_version, expected_agent_playbook_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("aggregation rebuilding cluster changed")
            updated = self.conn.execute(
                "UPDATE playbook_aggregation_item SET disposition='terminal_noop', "
                "cluster_id=NULL, reason=?, updated_at=unixepoch() WHERE "
                "agent_version=? AND cluster_id=? AND disposition='residual'",
                (reason, agent_version, cluster_id),
            )
            self.conn.execute(
                "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                (int(row[0]),),
            )
            deleted = self.conn.execute(
                "DELETE FROM playbook_aggregation_cluster WHERE cluster_id=? "
                "AND agent_version=? AND state='rebuilding' AND agent_playbook_id=?",
                (cluster_id, agent_version, expected_agent_playbook_id),
            )
            if deleted.rowcount != 1:
                raise RuntimeError("aggregation rebuilding cluster changed")
            return int(updated.rowcount)

    def delete_orphaned_playbook_aggregation_clusters(
        self, agent_version: str
    ) -> list[int]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT index_rowid, agent_playbook_id "
                "FROM playbook_aggregation_cluster c "
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
        return [int(row[1]) for row in rows if row[1] is not None]

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
                    "SELECT c.cluster_id, v.distance, c.agent_playbook_id FROM ("
                    "SELECT rowid, distance FROM playbook_aggregation_clusters_vec "
                    "WHERE embedding MATCH ? AND rowid IN ("
                    "SELECT index_rowid FROM playbook_aggregation_cluster "
                    "WHERE agent_version=? AND embedding_model=? "
                    "AND embedding_dimension=? AND state='active' "
                    "AND agent_playbook_id IS NOT NULL) "
                    "ORDER BY distance LIMIT ?) v "
                    "JOIN playbook_aggregation_cluster c ON c.index_rowid=v.rowid "
                    "ORDER BY v.distance, c.cluster_id LIMIT 1",
                    (
                        json.dumps(embedding),
                        agent_version,
                        embedding_model,
                        len(embedding),
                        candidate_limit,
                    ),
                ).fetchone()
                if row is not None:
                    matches[item_id] = PlaybookAggregationClusterMatch(
                        str(row[0]), 1.0 - float(row[1]), int(row[2])
                    )
        return matches

    def create_playbook_aggregation_cluster(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        agent_playbook_id: int | None,
        centroid_embedding: list[float],
        member_count: int,
        embedding_model: str,
    ) -> None:
        if not centroid_embedding:
            raise ValueError("cluster centroid embedding must be non-empty")
        if member_count <= 0:
            raise ValueError("cluster member_count must be positive")
        dimension = len(centroid_embedding)
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
                        json.dumps(centroid_embedding),
                        None,
                        member_count,
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
                        json.dumps(centroid_embedding),
                        None,
                        member_count,
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
                (next_rowid, json.dumps(centroid_embedding)),
            )
            if self._own_transaction():
                self.conn.commit()

    def attach_playbook_aggregation_items(
        self,
        *,
        agent_version: str,
        attachments: list[tuple[int, str]],
    ) -> None:
        if not attachments:
            return
        item_ids = [item_id for item_id, _cluster_id in attachments]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("aggregation attachment IDs must be unique")
        grouped: dict[str, list[int]] = {}
        for item_id, cluster_id in attachments:
            grouped.setdefault(cluster_id, []).append(item_id)
        with self._lock:
            own_transaction = self._own_transaction()
            try:
                cluster_rows: dict[str, sqlite3.Row] = {}
                cluster_ids = list(grouped)
                for offset in range(0, len(cluster_ids), 500):
                    page = cluster_ids[offset : offset + 500]
                    placeholders = ",".join("?" for _value in page)
                    rows = self.conn.execute(
                        "SELECT cluster_id, member_count FROM playbook_aggregation_cluster "
                        f"WHERE agent_version=? AND state='active' AND cluster_id IN ({placeholders})",
                        (agent_version, *page),
                    ).fetchall()
                    cluster_rows.update({str(row[0]): row for row in rows})
                if set(cluster_rows) != set(grouped):
                    raise RuntimeError(
                        "aggregation cluster is not active for agent version"
                    )

                cluster_updates: list[tuple[int, str]] = []
                for cluster_id, members in grouped.items():
                    row = cluster_rows[cluster_id]
                    cluster_updates.append((int(row[1]) + len(members), cluster_id))

                before_changes = self.conn.total_changes
                self.conn.executemany(
                    "UPDATE playbook_aggregation_item SET disposition='cluster_member', "
                    "cluster_id=?, reason='centroid_match', updated_at=unixepoch() "
                    "WHERE agent_version=? AND user_playbook_id=? "
                    "AND disposition='residual'",
                    [
                        (cluster_id, agent_version, item_id)
                        for item_id, cluster_id in attachments
                    ],
                )
                if self.conn.total_changes - before_changes != len(attachments):
                    raise RuntimeError("aggregation residual attachment lost its state")
                self.conn.executemany(
                    "UPDATE playbook_aggregation_cluster SET member_count=? "
                    "WHERE cluster_id=? AND agent_version=? AND state='active'",
                    [(*update, agent_version) for update in cluster_updates],
                )
                if own_transaction:
                    self.conn.commit()
            except Exception:
                if own_transaction:
                    self.conn.rollback()
                raise

    def replace_playbook_aggregation_cluster_agent(
        self,
        *,
        cluster_id: str,
        agent_version: str,
        expected_agent_playbook_id: int,
        replacement_agent_playbook_id: int,
        centroid_embedding: list[float],
        embedding_model: str,
    ) -> None:
        if not centroid_embedding:
            raise ValueError("cluster centroid embedding must be non-empty")
        with self._lock:
            own_transaction = self._own_transaction()
            try:
                row = self.conn.execute(
                    "SELECT index_rowid, embedding_dimension FROM "
                    "playbook_aggregation_cluster WHERE cluster_id=? "
                    "AND agent_version=? AND state='active' "
                    "AND agent_playbook_id=?",
                    (cluster_id, agent_version, expected_agent_playbook_id),
                ).fetchone()
                if row is None:
                    raise RuntimeError("aggregation cluster agent changed")
                if int(row[1]) != len(centroid_embedding):
                    raise ValueError("aggregation embedding dimension changed")
                updated = self.conn.execute(
                    "UPDATE playbook_aggregation_cluster SET agent_playbook_id=?, "
                    "centroid=?, vector_sum=NULL, embedding_model=?, dirty=0 "
                    "WHERE cluster_id=? AND agent_version=? AND state='active' "
                    "AND agent_playbook_id=?",
                    (
                        replacement_agent_playbook_id,
                        json.dumps(centroid_embedding),
                        embedding_model,
                        cluster_id,
                        agent_version,
                        expected_agent_playbook_id,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("aggregation cluster agent changed")
                rowid = int(row[0])
                self.conn.execute(
                    "DELETE FROM playbook_aggregation_clusters_vec WHERE rowid=?",
                    (rowid,),
                )
                self.conn.execute(
                    "INSERT INTO playbook_aggregation_clusters_vec(rowid, embedding) "
                    "VALUES (?, ?)",
                    (rowid, json.dumps(centroid_embedding)),
                )
                if own_transaction:
                    self.conn.commit()
            except Exception:
                if own_transaction:
                    self.conn.rollback()
                raise
