"""Clearing a user on SQLite erases every optimizer job that holds their content.

On SQLite ``clear_user_data`` is the erasure path. An open-world job stores
the evidence it was frozen from -- the bundle, the analyst invocations, the
attempt context, the discovery memo, the candidate, compaction checkpoints --
and that evidence is cross-user: a job tuning user A's playbook can carry the
sessions of any viewer the playbook was served to. These cases freeze and run
real attempts, clear one user, and then scan EVERY table for that user's id and
content marker, so a table added later that stores user content fails here
without anyone listing it.

All data is synthetic.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain import Request
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceSnapshot,
    OpenWorldTargetIdentity,
    OpenWorldTaskContext,
)
from reflexio.server.services.offline_tuner.evidence.open_world_selection import (
    select_open_world_evidence,
)
from reflexio.server.services.offline_tuner.evidence.open_world_store import (
    OpenWorldEvidenceBundleStore,
)
from reflexio.server.services.offline_tuner.open_world.evidence_documents import (
    rehydrate_frozen_evidence,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_authority import (
    qualify_rule_evidence_method,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_runtime import (
    RuleEvidenceAnalysisStore,
    RuleEvidenceAttemptRunner,
)
from reflexio.server.services.offline_tuner.open_world.runner import AttemptLease
from reflexio.server.services.offline_tuner.open_world_records import (
    OPEN_WORLD_EVIDENCE_ARTIFACT_KIND,
)
from reflexio.server.services.search_exposure import (
    user_playbook_full_version_fingerprint,
)
from reflexio.server.services.storage.session_outcome_identity import (
    canonical_trajectory_bytes,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage.playbook._optimization import (
    JOB_CHILD_TABLES_DELETED_EXPLICITLY,
    JOB_REFERENCING_TABLES_KEPT,
)
from reflexio.test_support.open_world_attempt_contract import (
    GroundedTransport,
    _accepted,
    _rolling,
)
from reflexio.test_support.open_world_exposure_contract import _owned_playbook, _uid
from reflexio.test_support.open_world_publication_contract import publish
from reflexio.test_support.open_world_rule_evidence import (
    evidence_digest,
    evidence_event,
)
from tests.server.services.storage.test_open_world_port_contract import (
    _SQLitePortHarness,
)

_LEASE = 3_600


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[_SQLitePortHarness]:
    """A fresh SQLite store with the port-contract binding's seeders."""
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(org_id="erasure_test", db_path=str(tmp_path / "r.db"))
        try:
            yield _SQLitePortHarness(storage)
        finally:
            storage.conn.close()


def _tables_mentioning(conn: sqlite3.Connection, needle: str) -> set[str]:
    """Every ordinary table holding ``needle`` in any column, as text."""
    found: set[str] = set()
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name NOT LIKE 'sqlite_%'"
            " AND sql NOT LIKE 'CREATE VIRTUAL TABLE%'"
        )
    ]
    for table in tables:
        columns = [row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")]
        if not columns:
            continue
        predicate = " OR ".join(
            f'instr(CAST("{column}" AS TEXT), ?) > 0' for column in columns
        )
        try:
            hit = conn.execute(
                f'SELECT 1 FROM "{table}" WHERE {predicate} LIMIT 1',  # noqa: S608
                (needle,) * len(columns),
            ).fetchone()
        except sqlite3.OperationalError:
            # A shadow table of a virtual table whose module is not loaded.
            continue
        if hit is not None:
            found.add(table)
    return found


def _cross_user_job(harness: _SQLitePortHarness) -> dict[str, Any]:
    """Freeze A's playbook over five sessions, the last of them viewer B's.

    Then run the attempt. It binds an attempt context whose manifest names
    B's session before it is refused (a diagnostic must be about the
    incumbent's own sessions); what it persisted is what must be erasable.
    """
    storage = harness.storage
    owner, viewer = _uid("owner"), _uid("viewer")
    marker = f"viewer-secret-{viewer}"
    playbook = _owned_playbook(harness, owner)
    incumbent = storage.get_user_playbook_by_id(playbook.user_playbook_id)
    assert incumbent is not None
    refs = {user: harness.register_subject(user) for user in (owner, viewer)}
    now = harness.now()
    fingerprint = user_playbook_full_version_fingerprint(incumbent)
    events = []
    for index in range(1, 6):
        user = viewer if index == 5 else owner
        event = evidence_event(
            session_id=f"session-{index}",
            outcome="failure",
            content=marker if user == viewer else "The owner's own question.",
        )
        trajectory = json.loads(json.dumps(event.trajectory))
        for request in trajectory["requests"]:
            request["request"]["user_id"] = user
            for interaction in request["interactions"]:
                interaction["user_id"] = user
        canonical = canonical_trajectory_bytes(trajectory)
        events.append(
            replace(
                event,
                tenant_id=storage.org_id,
                user_id=user,
                user_playbook_id=incumbent.user_playbook_id,
                served_full_version_fingerprint=fingerprint,
                governance_subject_refs=(refs[user],),
                exposed_at=now - 100,
                ingested_at=now - 100,
                trajectory=trajectory,
                trajectory_digest=sha256(canonical).hexdigest(),
                trajectory_byte_count=len(canonical),
                outcome=None,
                outcome_id=None,
                outcome_revision=None,
                outcome_contract_digest=None,
            )
        )
    snapshot = OpenWorldEvidenceSnapshot(tenant_id=storage.org_id, events=tuple(events))
    target = OpenWorldTargetIdentity(
        tenant_id=storage.org_id,
        user_playbook_id=incumbent.user_playbook_id,
        incumbent_semantic_digest=evidence_digest("incumbent"),
        incumbent_full_version_fingerprint=fingerprint,
        success_definition_digest=OpenWorldTaskContext().digest,
        governance_epoch_vector=tuple(sorted((ref, 0) for ref in refs.values())),
        frozen_at=now,
    )
    selection = select_open_world_evidence(snapshot, target)
    assert selection.bundle.eligible, selection.bundle.reason
    frozen = OpenWorldEvidenceBundleStore(storage).freeze(
        selection, owner="worker-a", lease_seconds=_LEASE, now=now
    )
    assert frozen.claim is not None
    transport = GroundedTransport(called_at=now)
    store = RuleEvidenceAnalysisStore(
        storage, qualify_rule_evidence_method(transport), transport, lambda: now
    )
    store.reconcile_authority()
    loaded = OpenWorldEvidenceBundleStore(storage).load(frozen.job.job_id)
    RuleEvidenceAttemptRunner(store, AttemptLease(frozen.claim), _LEASE).run(
        frozen=loaded,
        incumbent=incumbent,
        evidence=rehydrate_frozen_evidence(loaded, snapshot),
    )
    return {
        "job_id": frozen.job.job_id,
        "owner": owner,
        "viewer": viewer,
        "viewer_ref": refs[viewer],
        "marker": marker,
        "now": now,
    }


def _jobs(conn: sqlite3.Connection) -> set[int]:
    return {
        row[0] for row in conn.execute("SELECT job_id FROM playbook_optimization_jobs")
    }


def test_clearing_a_viewer_erases_the_owners_job_that_holds_their_sessions(
    harness: _SQLitePortHarness,
) -> None:
    storage = harness.storage
    job = _cross_user_job(harness)
    control = _accepted(harness)
    conn = storage.conn
    named = _tables_mentioning(conn, job["viewer"]) | _tables_mentioning(
        conn, job["viewer_ref"]
    )
    assert {
        "playbook_optimization_artifacts",
        "playbook_optimization_jobs",
        "offline_tuner_open_world_attempt_context",
    } <= named, named

    counts = storage.clear_user_data(job["viewer"])

    assert counts["optimization_jobs"] == 1
    assert _tables_mentioning(conn, job["marker"]) == set()
    assert _tables_mentioning(conn, job["viewer"]) == set()
    assert _tables_mentioning(conn, job["viewer_ref"]) == set()
    assert (
        storage.get_playbook_optimization_artifact(
            job["job_id"], OPEN_WORLD_EVIDENCE_ARTIFACT_KIND
        )
        is None
    )
    assert _jobs(conn) == {control.job_id}, "an unrelated user's job survives"
    assert harness.attempt_rows(control.job_id).artifacts == 4


def test_clearing_an_owner_erases_their_jobs_and_every_attempt_row(
    harness: _SQLitePortHarness,
) -> None:
    storage = harness.storage
    attempt = _accepted(harness)
    identity = sha256(b"erasure-checkpoint").hexdigest()
    storage.session_compaction_checkpoint(
        job_id=attempt.job_id,
        identity=identity,
        fence=attempt.claim.fence,
        now=attempt.now,
        receipt=_rolling(identity, batches=1, complete=True, word=attempt.owner),
    )
    control = _accepted(harness)
    conn = storage.conn
    assert harness.attempt_rows(attempt.job_id) == (6, 1, 4, 1)

    counts = storage.clear_user_data(attempt.owner)

    assert counts["optimization_jobs"] == 1
    assert _tables_mentioning(conn, attempt.owner) == set()
    assert harness.attempt_rows(attempt.job_id) == (0, 0, 0, 0)
    assert _jobs(conn) == {control.job_id}
    assert harness.attempt_rows(control.job_id) == (6, 1, 4, 0)


def test_every_table_referencing_a_job_is_erased_with_it(
    harness: _SQLitePortHarness,
) -> None:
    """A table naming ``job_id`` cascades from the job, or clear says what it does.

    Derived from the schema: a new child table without a cascading foreign
    key fails here until clear deletes it or records why it may keep the id.
    """
    conn = harness.storage.conn
    uncovered = set()
    for (table,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
        " AND name != 'playbook_optimization_jobs'"
    ):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")}
        if "job_id" not in columns:
            continue
        cascades = any(
            row[2] == "playbook_optimization_jobs" and row[6] == "CASCADE"
            for row in conn.execute(f"PRAGMA foreign_key_list('{table}')")
        )
        if not cascades and table not in (
            JOB_CHILD_TABLES_DELETED_EXPLICITLY | JOB_REFERENCING_TABLES_KEPT.keys()
        ):
            uncovered.add(table)

    assert uncovered == set()
    assert JOB_CHILD_TABLES_DELETED_EXPLICITLY.isdisjoint(JOB_REFERENCING_TABLES_KEPT)


def test_a_clear_failure_leaves_nothing_half_erased(
    harness: _SQLitePortHarness,
) -> None:
    """The job delete runs in the clear's own transaction, before the requests."""
    storage = harness.storage
    attempt = _accepted(harness)
    storage.add_request(
        Request(request_id=_uid("request"), user_id=attempt.owner, session_id="s")
    )
    storage.conn.execute(
        "CREATE TEMP TRIGGER refuse_request_delete BEFORE DELETE ON requests"
        " BEGIN SELECT RAISE(ABORT, 'request delete refused'); END"
    )
    try:
        with pytest.raises(sqlite3.IntegrityError, match="request delete refused"):
            storage.clear_user_data(attempt.owner)
    finally:
        storage.conn.execute("DROP TRIGGER IF EXISTS temp.refuse_request_delete")

    assert attempt.job_id in _jobs(storage.conn)
    assert harness.attempt_rows(attempt.job_id).artifacts == 4


# ---------------------------------------------------------------------------
# A published attempt
# ---------------------------------------------------------------------------

#: Tables a provisional commit writes that hold no ``job_id`` and are not the
#: user's own playbooks, with what a clear does to them and why.
_PUBLICATION_TABLES_KEPT_ON_CLEAR: dict[str, str] = {
    "lineage_event": (
        "the supersede and revise events name playbook ids and the job's "
        "request id, never user content; a purged playbook keeps its lineage "
        "chain under the existing purge rules"
    ),
    "playbook_aggregation_invalidation": (
        "the lineage writer queues the supersede for the aggregator by agent "
        "version and playbook id, as it does for every status change; the "
        "aggregator drains it, and it holds no user content"
    ),
}


def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for (table,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
        " AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        try:
            (counts[table],) = conn.execute(
                f'SELECT count(*) FROM "{table}"'  # noqa: S608
            ).fetchone()
        except sqlite3.OperationalError:
            continue
    return counts


def _cascading_job_tables(conn: sqlite3.Connection) -> set[str]:
    return {
        table
        for (table,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        if any(
            row[2] == "playbook_optimization_jobs" and row[6] == "CASCADE"
            for row in conn.execute(f"PRAGMA foreign_key_list('{table}')")
        )
    }


def test_every_table_a_publication_writes_is_covered_by_the_clear(
    harness: _SQLitePortHarness,
) -> None:
    """Derived from what one real commit writes, not from a list of its tables.

    Each table either cascades from the job, is deleted with it explicitly, is
    the user's own playbooks and their search indexes (erased by the user
    rules), or is kept with a recorded reason.
    """
    conn = harness.storage.conn
    attempt = _accepted(harness)
    before = _row_counts(conn)

    publish(attempt)

    written = {
        table for table, count in _row_counts(conn).items() if count > before[table]
    }
    covered = (
        _cascading_job_tables(conn)
        | JOB_CHILD_TABLES_DELETED_EXPLICITLY
        | _PUBLICATION_TABLES_KEPT_ON_CLEAR.keys()
        | {table for table in written if table.startswith("user_playbooks")}
    )
    assert {
        "user_playbooks",
        "user_playbook_deployment_lifecycles",
        "user_playbook_publication_results",
        "playbook_optimization_events",
        "lineage_event",
    } <= written
    assert written - covered == set()


def test_clearing_the_owner_erases_a_published_attempt(
    harness: _SQLitePortHarness,
) -> None:
    """The job and every publication row go; the two versions are purged.

    The incumbent is a superseded tombstone and the successor is what it
    points to, so both are content-purged under the lineage rules: their
    bodies are blanked and the chain between them is kept. No lifecycle or
    result is left naming either.
    """
    storage = harness.storage
    conn = storage.conn
    attempt = _accepted(harness)
    revised = attempt.payload("open_world_candidate")["payload"]["revised_content"]
    result = publish(attempt)
    successor_id = result.successor_user_playbook_id
    assert successor_id is not None
    incumbent_id = attempt.incumbent.user_playbook_id
    control = _accepted(harness)
    control_result = publish(control)
    assert "user_playbook_publication_staging" not in _tables_mentioning(
        conn, attempt.owner
    ), "staging, which holds the incumbent snapshot, is gone after the commit"

    counts = storage.clear_user_data(attempt.owner)

    assert counts["optimization_jobs"] == 1
    assert _tables_mentioning(conn, attempt.owner) == set()
    assert harness.publication_rows(attempt.job_id) == (0, 0, 0, 0, 0, 0)
    named = (incumbent_id, successor_id)
    assert storage.list_playbook_deployment_lifecycles(user_playbook_ids=named) == ()
    placeholders = ", ".join("?" * len(named))
    assert (
        conn.execute(
            "SELECT count(*) FROM user_playbook_publication_results"  # noqa: S608
            f" WHERE successor_user_playbook_id IN ({placeholders})",
            named,
        ).fetchone()[0]
        == 0
    )
    rows = {
        row["user_playbook_id"]: row
        for row in conn.execute(
            "SELECT user_playbook_id, content, user_id, superseded_by"  # noqa: S608
            f" FROM user_playbooks WHERE user_playbook_id IN ({placeholders})",
            named,
        ).fetchall()
    }
    assert set(rows) == set(named), "both versions are purged, not deleted"
    assert all(row["content"] == "" and row["user_id"] is None for row in rows.values())
    assert rows[incumbent_id]["superseded_by"] == successor_id
    assert counts["purged_user_playbooks"] == 2
    # The control attempt published the same synthetic text; only its rows
    # may still hold it.
    assert {
        row[0]
        for row in conn.execute(
            "SELECT user_playbook_id FROM user_playbooks WHERE content = ?",
            (revised,),
        )
    } == {control_result.successor_user_playbook_id}
    assert (
        len(
            storage.list_playbook_deployment_lifecycles(
                user_playbook_ids=(control.incumbent.user_playbook_id,)
            )
        )
        == 1
    ), "another owner's publication survives"


def test_clearing_another_user_keeps_a_publication(
    harness: _SQLitePortHarness,
) -> None:
    storage = harness.storage
    attempt = _accepted(harness)
    result = publish(attempt)

    storage.clear_user_data(_uid("unrelated"))

    assert harness.publication_rows(attempt.job_id).lifecycles == 1
    assert (
        storage.load_user_playbook_provisional_publication_result(attempt.job_id)
        == result
    )
