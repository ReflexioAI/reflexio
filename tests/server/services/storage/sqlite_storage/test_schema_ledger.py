"""The SQLite schema ledger: ordering, historical upgrades and the warm open."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._base import (
    _RETIRED_ARTIFACT_KIND_LITERALS,
    _RETIRED_OPTIMIZER_JOB_LITERALS,
    _TERMINAL_OUTCOME_LITERALS,
)
from reflexio.server.services.storage.sqlite_storage.migrations import (
    LATEST_VERSION,
    MIGRATION_STEPS,
    VEC_STEP,
    stored_version,
)


def _sqlite_vec_loads() -> bool:
    try:
        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except (ImportError, OSError, AttributeError, sqlite3.OperationalError):
        return False


requires_vec = pytest.mark.skipif(
    not _sqlite_vec_loads(),
    reason="the ledger stamps the sqlite-vec step only where the extension loads",
)

FIXTURES = Path(__file__).parent / "fixtures"
HISTORICAL_DDL = sorted(FIXTURES.glob("historical_ddl_*.sql"))


def _open(path: Path, org: str = "ledger") -> SQLiteStorage:
    return SQLiteStorage(org, db_path=str(path))


def _schema(conn: sqlite3.Connection) -> dict[str, object]:
    """Shape of every object: columns by name, index/trigger/view text."""
    shape: dict[str, object] = {}
    for kind, name, sql in conn.execute(
        "SELECT type, name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall():
        if kind == "table":
            columns = {
                row[1]: (row[2], row[3], row[4], row[5])
                for row in conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            }
            shape[f"table:{name}"] = columns
        else:
            shape[f"{kind}:{name}"] = " ".join((sql or "").split())
    return shape


@pytest.fixture(scope="module")
def fresh_schema(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    storage = _open(tmp_path_factory.mktemp("fresh") / "fresh.db")
    try:
        return _schema(storage.conn)
    finally:
        storage.conn.close()


def test_steps_are_consecutive_and_resolve_to_methods() -> None:
    assert [version for version, _ in MIGRATION_STEPS] == list(
        range(1, len(MIGRATION_STEPS) + 1)
    )
    for _, name in MIGRATION_STEPS:
        assert callable(getattr(SQLiteStorage, name)), name


@requires_vec
def test_fresh_database_is_stamped_latest(tmp_path: Path) -> None:
    storage = _open(tmp_path / "fresh.db")
    try:
        assert stored_version(storage) == LATEST_VERSION
    finally:
        storage.conn.close()


def test_derived_step_inputs_are_pinned() -> None:
    """Widening these needs a NEW appended ledger step, not just an edit."""
    assert tuple(_TERMINAL_OUTCOME_LITERALS) == PINNED_TERMINAL_OUTCOMES, (
        "OptimizationTerminalOutcome changed: append a ledger step that rebuilds "
        "playbook_optimization_jobs, then update this pin"
    )
    assert _RETIRED_OPTIMIZER_JOB_LITERALS == PINNED_RETIRED_JOB
    assert _RETIRED_ARTIFACT_KIND_LITERALS == ("'replay_manifest'",)


@requires_vec
@pytest.mark.parametrize("ddl", HISTORICAL_DDL, ids=lambda p: p.stem)
def test_historical_database_upgrades_to_current_schema(
    ddl: Path, tmp_path: Path, fresh_schema: dict[str, object]
) -> None:
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(ddl.read_text())
    old.close()

    storage = _open(path)
    try:
        assert stored_version(storage) == LATEST_VERSION
        upgraded = _schema(storage.conn)
        # Every table rebuild must leave what the base DDL declares in place:
        # the DDL no longer re-runs on each open to repair a dropped index.
        storage._apply_base_ddl()
        assert _schema(storage.conn) == upgraded
    finally:
        storage.conn.close()

    assert "index:idx_requests_session_created_at_asc" in upgraded
    extra_expected, missing_expected = EXPECTED_DIVERGENCE[ddl.stem]
    assert set(upgraded) - set(fresh_schema) == extra_expected
    assert set(fresh_schema) - set(upgraded) == missing_expected
    assert {
        key
        for key in set(upgraded) & set(fresh_schema)
        if upgraded[key] != fresh_schema[key]
    } == set()


@requires_vec
def test_pre_ledger_current_database_is_stamped_without_schema_change(
    tmp_path: Path, fresh_schema: dict[str, object]
) -> None:
    path = tmp_path / "current.db"
    storage = _open(path)
    storage.conn.execute("PRAGMA user_version = 0")
    storage.conn.close()

    storage = _open(path)
    try:
        assert stored_version(storage) == LATEST_VERSION
        assert _schema(storage.conn) == fresh_schema
    finally:
        storage.conn.close()


@requires_vec
def test_warm_open_runs_no_schema_step(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "warm.db"
    _open(path).conn.close()

    statements: list[str] = []
    real_connect = sqlite3.connect

    def tracing_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracing_connect)
    _open(path).conn.close()

    migration_statements = [
        s
        for s in statements
        if s.lstrip().upper().startswith(("ALTER", "DROP"))
        or "PRAGMA table_info" in s
        or "sqlite_master" in s.lower()
        and "user_playbook_publication" in s
    ]
    assert migration_statements == []
    assert "PRAGMA user_version" in statements


@requires_vec
def test_interrupted_upgrade_resumes_after_last_stamped_step(tmp_path: Path) -> None:
    path = tmp_path / "resume.db"
    storage = _open(path)
    storage.conn.execute("PRAGMA user_version = 20")
    storage.conn.close()

    ran: list[str] = []
    originals = {name: getattr(SQLiteStorage, name) for _, name in MIGRATION_STEPS}
    mp = pytest.MonkeyPatch()
    try:
        for name, fn in originals.items():
            mp.setattr(
                SQLiteStorage,
                name,
                lambda self, _fn=fn, _n=name: (ran.append(_n), _fn(self))[1],
            )
        _open(path).conn.close()
    finally:
        mp.undo()
    assert ran == [name for version, name in MIGRATION_STEPS if version > 20]


PINNED_TERMINAL_OUTCOMES = (
    "'applied'",
    "'insufficient_negative_evidence'",
    "'insufficient_positive_evidence'",
    "'insufficient_coverage'",
    "'deployment_unsupported'",
    "'candidate_regressed'",
    "'candidate_did_not_improve'",
    "'incumbent_changed'",
    "'generation_failed'",
    "'publication_failed'",
    "'governance_erased'",
    "'no_grounded_hypothesis'",
    "'analyst_unqualified'",
    "'heldout_evidence_failed'",
    "'stale_incumbent'",
    "'governance_invalidated'",
    "'infrastructure_failure'",
    "'regeneration_fenced'",
    "'invocation_slot_pinned'",
    "'evidence_view_incomplete'",
    "'bundle_schema_unsupported'",
    "'abandoned'",
    "'evidence_manifest_conflict'",
)
PINNED_RETIRED_JOB = (
    "'offline_tuner_replay'",
    "'replay_running'",
    "'replay_evaluated'",
    "'replay_unsupported'",
    "'incomplete_replay_scope'",
    "'insufficient_replay_cases'",
    "'replay_inconclusive'",
    "'replay_failed'",
)

# What a historical database legitimately differs by, identical to what the
# pre-ledger code produced (verified by running both on these fixtures).
# Extras are retired objects the upgrade deliberately keeps;
# the one object missing everywhere is created only when a dataset is created,
# never on an upgrade. the oldest shape also lacks a request index no step adds.
_RETIRED_CHANGE_LOGS = {
    "index:idx_pacl_agent_version",
    "index:idx_pacl_playbook_name",
    "index:idx_pcl_created_at",
    "index:idx_pcl_user_id",
    "table:playbook_aggregation_change_logs_retired_20260624",
    "table:profile_change_logs_retired_20260623",
}
_CREATION_ONLY = {"index:extraction_cursors_window_lookup"}
EXPECTED_DIVERGENCE: dict[str, tuple[set[str], set[str]]] = {
    "historical_ddl_899ead79": (
        _RETIRED_CHANGE_LOGS,
        _CREATION_ONLY,
    ),
    "historical_ddl_751275d5": (
        _RETIRED_CHANGE_LOGS | {"index:idx_share_links_resource", "table:share_links"},
        _CREATION_ONLY,
    ),
    "historical_ddl_f827c9a9": (
        {
            "index:learning_jobs_coalesce",
            "index:learning_jobs_poll",
            "table:learning_jobs",
        },
        _CREATION_ONLY,
    ),
}


@requires_vec
def test_receipt_functions_exist_on_a_reopened_stamped_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "functions.db"
    _open(path, "fn-org").conn.close()

    storage = _open(path, "fn-org")  # stamped: the receipt step does not run
    try:
        assert stored_version(storage) == LATEST_VERSION
        assert (
            storage.conn.execute("SELECT reflexio_publication_org()").fetchone()[0]
            == "fn-org"
        )
        for sql in (
            "SELECT reflexio_valid_publication_receipt('x', 'y', '[]', 'pending', '{}')",
            "SELECT reflexio_window_handoff_matches('k', '{}', '{}', '{}')",
        ):
            try:
                storage.conn.execute(sql).fetchone()
            except sqlite3.Error as error:
                assert "no such function" not in str(error)
    finally:
        storage.conn.close()


def test_without_the_extension_later_steps_run_and_the_stamp_stops_at_the_vec_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "novec.db"
    monkeypatch.setattr(SQLiteStorage, "_try_load_sqlite_vec", lambda _self: False)
    storage = _open(path)
    try:
        assert stored_version(storage) == VEC_STEP - 1
        tables = {
            row[0]
            for row in storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        # Steps after the vec step ran: their tables exist.
        assert {"extraction_windows", "user_playbook_publication_results"} <= tables
        assert "profiles_vec" not in tables
    finally:
        storage.conn.close()
    monkeypatch.undo()

    if _sqlite_vec_loads():
        storage = _open(path)
        try:
            assert stored_version(storage) == LATEST_VERSION
        finally:
            storage.conn.close()


@requires_vec
def test_interrupted_vec_backfill_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vec.db"
    monkeypatch.setattr(SQLiteStorage, "_try_load_sqlite_vec", lambda _self: False)
    _open(path).conn.close()
    monkeypatch.undo()

    calls: list[str] = []

    def interrupted(self) -> None:
        calls.append("interrupted")
        raise RuntimeError("stopped after the vec tables were created")

    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_migrate_vec_tables", interrupted)
        with pytest.raises(RuntimeError):
            _open(path)
    assert calls == ["interrupted"]

    backfilled: list[str] = []
    original = SQLiteStorage._migrate_vec_tables

    def recording(self) -> None:
        backfilled.append("ran")
        original(self)

    monkeypatch.setattr(SQLiteStorage, "_migrate_vec_tables", recording)
    storage = _open(path)
    try:
        assert backfilled == ["ran"]
        assert stored_version(storage) == LATEST_VERSION
    finally:
        storage.conn.close()


@requires_vec
def test_stamp_never_lowers_a_version_another_opener_advanced(
    tmp_path: Path,
) -> None:
    from reflexio.server.services.storage.sqlite_storage.migrations import _stamp

    path = tmp_path / "stamp.db"
    first = _open(path)
    second = _open(path)
    try:
        assert stored_version(second) == LATEST_VERSION
        statements: list[str] = []
        second.conn.set_trace_callback(statements.append)
        _stamp(second, 5)
        assert not any("user_version =" in s for s in statements)
        assert any("BEGIN IMMEDIATE" in s for s in statements)
        assert stored_version(first) == LATEST_VERSION
    finally:
        first.conn.close()
        second.conn.close()


@requires_vec
def test_sidecar_rows_written_while_the_extension_was_missing_are_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    path = tmp_path / "roundtrip.db"
    _open(path).conn.close()  # vec loaded, stamped latest

    embedding = json.dumps([0.25] * EMBEDDING_DIMENSIONS)
    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_try_load_sqlite_vec", lambda _self: False)
        fallback = _open(path)  # takes the vec step back before any write
        try:
            assert stored_version(fallback) == VEC_STEP - 1
            fallback.conn.execute(
                "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp, "
                "embedding) VALUES ('p1', 'u1', 1, ?)",
                (embedding,),
            )
            fallback.conn.commit()
        finally:
            fallback.conn.close()

    storage = _open(path)
    try:
        assert stored_version(storage) == LATEST_VERSION
        assert (
            storage.conn.execute("SELECT COUNT(*) FROM profiles_vec").fetchone()[0] == 1
        )
    finally:
        storage.conn.close()


_LEGACY_REQUESTS = (
    "CREATE TABLE requests (request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
    "created_at TEXT NOT NULL, source TEXT, agent_version TEXT, session_id TEXT)"
)
_REQUEST_INDEXES = {
    "idx_requests_user_id",
    "idx_requests_session_id",
    "idx_requests_created_at",
    "idx_requests_session_created_at_asc",
    "idx_requests_retrieval_experiment",
    "idx_requests_governance_subject_ref",
}


@requires_vec
@pytest.mark.parametrize(
    ("action", "denied"),
    [
        (sqlite3.SQLITE_ALTER_TABLE, "requests_new"),
        (sqlite3.SQLITE_CREATE_INDEX, "idx_requests_session_created_at_asc"),
    ],
    ids=["after-drop-before-rename", "after-rename-during-indexes"],
)
def test_interrupted_requests_rebuild_is_atomic_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: int, denied: str
) -> None:
    path = tmp_path / "requests.db"
    storage = _open(path)
    storage.conn.execute("DROP TABLE requests")
    storage.conn.execute(_LEGACY_REQUESTS)
    storage.conn.execute(
        "INSERT INTO requests VALUES ('r1', 'u1', '2026-01-01', 's', 'v', NULL)"
    )
    storage.conn.execute("PRAGMA user_version = 0")
    storage.conn.commit()
    storage.conn.close()

    armed = False
    real_connect = sqlite3.connect
    real_step = SQLiteStorage._migrate_request_session_id_required

    def authorizer(code, arg1, arg2, _db, _source):
        if (
            armed
            and code == action
            and (arg2 if action == sqlite3.SQLITE_ALTER_TABLE else arg1) == denied
        ):
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_authorizer(authorizer)
        return conn

    def arming_step(self):
        nonlocal armed
        armed = True
        try:
            real_step(self)
        finally:
            armed = False

    with monkeypatch.context() as patched:
        patched.setattr(sqlite3, "connect", connect)
        patched.setattr(
            SQLiteStorage, "_migrate_request_session_id_required", arming_step
        )
        with pytest.raises(sqlite3.DatabaseError):
            _open(path)

    with closing(sqlite3.connect(path)) as probe:
        assert probe.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1

    storage = _open(path)
    try:
        assert stored_version(storage) == LATEST_VERSION
        indexes = {
            row[0]
            for row in storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'requests'"
            )
        }
        assert indexes >= _REQUEST_INDEXES
        row = storage.conn.execute(
            "SELECT request_id, session_id FROM requests"
        ).fetchone()
        assert row[0] == "r1" and row[1].startswith("legacy-")
    finally:
        storage.conn.close()


@requires_vec
def test_unindexable_embeddings_do_not_stop_an_upgrade(tmp_path: Path) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    path = tmp_path / "malformed.db"
    storage = _open(path)
    good = json.dumps([0.5] * EMBEDDING_DIMENSIONS)
    for pid, embedding in (
        ("good", good),
        ("bools", json.dumps([True] * EMBEDDING_DIMENSIONS)),
        ("text", "not json"),
        ("short", json.dumps([1.0])),
        ("blob", b"\xff\xfe\xfa"),
        ("huge", "[" + "9" * 400 + ", 0.5" * (EMBEDDING_DIMENSIONS - 1) + "]"),
    ):
        storage.conn.execute(
            "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp, "
            "embedding) VALUES (?, 'u', 1, ?)",
            (pid, embedding),
        )
    storage.conn.execute(
        "INSERT INTO profiles_vec(rowid, embedding) "
        "SELECT rowid, embedding FROM profiles WHERE profile_id = 'good'"
    )
    storage.conn.execute("PRAGMA user_version = 0")
    storage.conn.commit()
    storage.conn.close()

    reopened = _open(path)
    try:
        assert stored_version(reopened) == LATEST_VERSION
        assert (
            reopened.conn.execute("SELECT COUNT(*) FROM profiles_vec").fetchone()[0]
            == 1
        )
    finally:
        reopened.conn.close()


@requires_vec
def test_backfill_batch_reads_and_writes_under_one_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
    from reflexio.server.services.storage.sqlite_storage.migrations import _steps

    path = tmp_path / "erasure.db"
    storage = _open(path)
    storage.conn.execute(
        "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp, "
        "embedding) VALUES ('p1', 'u', 1, ?)",
        (json.dumps([0.5] * EMBEDDING_DIMENSIONS),),
    )
    storage.conn.commit()
    storage.conn.close()
    with closing(sqlite3.connect(path)) as raw:
        raw.execute("PRAGMA user_version = 0")
        raw.commit()

    outcomes: list[str] = []
    real_loads = _steps._json_loads

    def erasing_loads(text):
        # Runs after the batch read, before its write: a competing eraser on
        # another connection must be refused, not interleaved.
        with closing(sqlite3.connect(path, timeout=0.2)) as other:
            try:
                other.execute("DELETE FROM profiles WHERE profile_id = 'p1'")
                other.commit()
                outcomes.append("erased")
            except sqlite3.OperationalError:
                outcomes.append("blocked")
        return real_loads(text)

    monkeypatch.setattr(_steps, "_json_loads", erasing_loads)
    reopened = _open(path)
    try:
        assert "blocked" in outcomes and "erased" not in outcomes
    finally:
        reopened.conn.close()


@requires_vec
def test_fallback_writes_after_a_native_stamp_are_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    path = tmp_path / "mixed.db"
    _open(path).conn.close()
    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_try_load_sqlite_vec", lambda _self: False)
        fallback = _open(path)  # opened first, then a native process stamps LATEST
    native = _open(path)
    try:
        assert stored_version(native) == LATEST_VERSION
    finally:
        native.conn.close()
    try:
        fallback.conn.execute(
            "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp, "
            "embedding) VALUES ('late', 'u', 1, ?)",
            (json.dumps([0.5] * EMBEDDING_DIMENSIONS),),
        )
        fallback.conn.commit()
        assert stored_version(fallback) == VEC_STEP - 1
    finally:
        fallback.conn.close()

    reopened = _open(path)
    try:
        assert stored_version(reopened) == LATEST_VERSION
        assert (
            reopened.conn.execute("SELECT COUNT(*) FROM profiles_vec").fetchone()[0]
            == 1
        )
    finally:
        reopened.conn.close()


@requires_vec
def test_a_reused_statement_still_marks_the_sidecar_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    path = tmp_path / "reuse.db"
    _open(path).conn.close()
    insert = (
        "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp, "
        "embedding) VALUES (?, 'u', 1, ?)"
    )
    embedding = json.dumps([0.5] * EMBEDDING_DIMENSIONS)
    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_try_load_sqlite_vec", lambda _self: False)
        fallback = _open(path)
    try:
        fallback.conn.execute(insert, ("first", embedding))
        fallback.conn.commit()
        native = _open(path)  # reconciles "first" and stamps the latest version
        native.conn.close()
        assert stored_version(fallback) == LATEST_VERSION
        fallback.conn.execute(insert, ("second", embedding))  # same SQL text
        fallback.conn.commit()
        assert stored_version(fallback) == VEC_STEP - 1
    finally:
        fallback.conn.close()

    reopened = _open(path)
    try:
        assert (
            reopened.conn.execute("SELECT COUNT(*) FROM profiles_vec").fetchone()[0]
            == 2
        )
    finally:
        reopened.conn.close()


@requires_vec
def test_a_retried_write_after_rollback_still_marks_the_sidecar_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    path = tmp_path / "rollback.db"
    _open(path).conn.close()
    insert = (
        "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp, "
        "embedding) VALUES (?, 'u', 1, ?)"
    )
    embedding = json.dumps([0.5] * EMBEDDING_DIMENSIONS)
    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_try_load_sqlite_vec", lambda _self: False)
        fallback = _open(path)
    try:
        fallback.conn.execute(insert, ("p", embedding))
        fallback.conn.rollback()
        _open(path).conn.close()  # a native open stamps the latest version
        assert stored_version(fallback) == LATEST_VERSION
        fallback.conn.execute(insert, ("p", embedding))  # same SQL text again
        fallback.conn.commit()
        assert stored_version(fallback) == VEC_STEP - 1
    finally:
        fallback.conn.close()


def _profile_vector(path: Path, rowid: int) -> tuple[float, ...]:
    import struct

    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    storage = _open(path)
    try:
        blob = storage.conn.execute(
            "SELECT embedding FROM profiles_vec WHERE rowid = ?", (rowid,)
        ).fetchone()[0]
        return struct.unpack(f"{EMBEDDING_DIMENSIONS}f", blob)
    finally:
        storage.conn.close()


class _Crash(BaseException):
    """Stands in for a process dying between a commit and its index flush."""


def _pending(path: Path) -> list[tuple[str, int]]:
    with closing(sqlite3.connect(path)) as raw:
        return [
            tuple(r) for r in raw.execute("SELECT table_name, row_id FROM vec_pending")
        ]


def _sidecar(path: Path, rowid: int) -> tuple[float, ...] | None:
    import struct

    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    storage = _open(path)  # repairs whatever is pending
    try:
        row = storage.conn.execute(
            "SELECT embedding FROM profiles_vec WHERE rowid = ?", (rowid,)
        ).fetchone()
        return (
            None if row is None else struct.unpack(f"{EMBEDDING_DIMENSIONS}f", row[0])
        )
    finally:
        storage.conn.close()


def _queue_profile(
    storage: SQLiteStorage, pid: str, vector: list[float], *, replace=False
):
    if replace:
        storage.conn.execute(
            "UPDATE profiles SET embedding = ? WHERE profile_id = ?",
            (json.dumps(vector), pid),
        )
    else:
        storage.conn.execute(
            "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp, "
            "embedding) VALUES (?, 'u', 1, ?)",
            (pid, json.dumps(vector)),
        )
    rowid = storage.conn.execute(
        "SELECT rowid FROM profiles WHERE profile_id = ?", (pid,)
    ).fetchone()[0]
    storage._deferred_index_ops.append(("vec_upsert", ("profiles_vec", rowid, vector)))
    return rowid


@requires_vec
def test_a_crash_between_commit_and_flush_is_repaired_for_an_insert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    path = tmp_path / "crash-insert.db"
    storage = _open(path)
    vector = [0.5] * EMBEDDING_DIMENSIONS

    def crash(_self, _table, _rowid):
        raise _Crash

    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_reconcile_pending_vector", crash)
        with pytest.raises(_Crash), storage.commit_scope():
            rowid = _queue_profile(storage, "p", vector)
    storage.conn.close()  # the process "dies"

    assert _pending(path) == [("profiles_vec", rowid)]
    assert _sidecar(path, rowid) == pytest.approx(tuple(vector))
    assert _pending(path) == []


@requires_vec
def test_a_crash_between_commit_and_flush_is_repaired_for_a_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    path = tmp_path / "crash-replace.db"
    storage = _open(path)
    old = [0.25] * EMBEDDING_DIMENSIONS
    new = [0.75] * EMBEDDING_DIMENSIONS
    with storage.commit_scope():
        rowid = _queue_profile(storage, "p", old)
    assert storage.conn.execute("SELECT COUNT(*) FROM profiles_vec").fetchone()[0] == 1

    def crash(_self, _table, _rowid):
        raise _Crash

    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_reconcile_pending_vector", crash)
        with pytest.raises(_Crash), storage.commit_scope():
            _queue_profile(storage, "p", new, replace=True)
    storage.conn.close()

    assert _sidecar(path, rowid) == pytest.approx(tuple(new))
    assert _pending(path) == []


@requires_vec
def test_overlapping_scopes_on_two_instances_repair_only_the_crashed_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
    from reflexio.server.services.storage.sqlite_storage.migrations._steps import (
        SQLiteSchemaMigrationsMixin,
    )

    path = tmp_path / "two.db"
    first = _open(path, "one")
    second = _open(path, "two")
    real_flush = SQLiteSchemaMigrationsMixin._reconcile_pending_vector
    other: dict[str, int] = {}
    crashed = False

    def flush(self, table, rowid):
        nonlocal crashed
        if self is first and not crashed:
            crashed = True
            # The second instance commits AND flushes while the first is pending.
            with second.commit_scope():
                other["rowid"] = _queue_profile(
                    second, "q", [0.125] * EMBEDDING_DIMENSIONS
                )
            raise _Crash
        return real_flush(self, table, rowid)

    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_reconcile_pending_vector", flush)
        with pytest.raises(_Crash), first.commit_scope():
            crashed_rowid = _queue_profile(first, "p", [0.5] * EMBEDDING_DIMENSIONS)
    first.conn.close()
    second.conn.close()

    assert _pending(path) == [("profiles_vec", crashed_rowid)]
    assert _sidecar(path, crashed_rowid) == pytest.approx((0.5,) * EMBEDDING_DIMENSIONS)
    assert _sidecar(path, other["rowid"]) == pytest.approx(
        (0.125,) * EMBEDDING_DIMENSIONS
    )
    assert _pending(path) == []


@requires_vec
def test_a_flush_failing_on_the_second_op_leaves_only_that_op_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
    from reflexio.server.services.storage.error import PostCommitStorageError
    from reflexio.server.services.storage.sqlite_storage.migrations._steps import (
        SQLiteSchemaMigrationsMixin,
    )

    path = tmp_path / "second.db"
    storage = _open(path)
    real_flush = SQLiteSchemaMigrationsMixin._reconcile_pending_vector
    calls = 0

    def flush(self, table, rowid):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("index write failed")
        return real_flush(self, table, rowid)

    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_reconcile_pending_vector", flush)
        with pytest.raises(PostCommitStorageError), storage.commit_scope():
            first = _queue_profile(storage, "a", [0.5] * EMBEDDING_DIMENSIONS)
            second = _queue_profile(storage, "b", [0.25] * EMBEDDING_DIMENSIONS)
    storage.conn.close()

    assert _pending(path) == [("profiles_vec", second)]
    assert _sidecar(path, first) == pytest.approx((0.5,) * EMBEDDING_DIMENSIONS)
    assert _pending(path) == []
    assert _sidecar(path, second) == pytest.approx((0.25,) * EMBEDDING_DIMENSIONS)


@requires_vec
def test_an_older_flush_cannot_erase_a_newer_rows_pending_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
    from reflexio.server.services.storage.sqlite_storage.migrations._steps import (
        SQLiteSchemaMigrationsMixin,
    )

    path = tmp_path / "older.db"
    a = _open(path, "a")
    b = _open(path, "b")
    e1 = [0.25] * EMBEDDING_DIMENSIONS
    e2 = [0.75] * EMBEDDING_DIMENSIONS
    with a.commit_scope():
        rowid = _queue_profile(a, "p", [0.0] * EMBEDDING_DIMENSIONS)
    real = SQLiteSchemaMigrationsMixin._reconcile_pending_vector
    state = {"a_done": False}

    def reconcile(self, table, row):
        if self is a and not state["a_done"]:
            state["a_done"] = True
            # B commits its newer embedding for the same row while A is pending...
            with pytest.raises(_Crash), b.commit_scope():
                _queue_profile(b, "p", e2, replace=True)
        if self is b:
            raise _Crash  # ...and dies before its own flush
        return real(self, table, row)

    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_reconcile_pending_vector", reconcile)
        with a.commit_scope():
            _queue_profile(a, "p", e1, replace=True)
    a.conn.close()
    b.conn.close()

    assert _sidecar(path, rowid) == pytest.approx(tuple(e2))
    assert _pending(path) == []


@requires_vec
def test_a_deeply_nested_embedding_is_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "nested.db"
    storage = _open(path)
    depth = 100_000
    storage.conn.execute(
        "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp, "
        "embedding) VALUES ('deep', 'u', 1, ?)",
        ("[" * depth + "]" * depth,),
    )
    storage.conn.execute("PRAGMA user_version = 0")
    storage.conn.commit()
    storage.conn.close()

    reopened = _open(path)
    try:
        assert stored_version(reopened) == LATEST_VERSION
        assert (
            reopened.conn.execute("SELECT COUNT(*) FROM profiles_vec").fetchone()[0]
            == 0
        )
    finally:
        reopened.conn.close()


@requires_vec
def test_a_standalone_interaction_insert_is_crash_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import struct

    from reflexio.models.api_schema.service_schemas import Interaction
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    path = tmp_path / "interaction.db"
    storage = _open(path)
    vector = [0.5] * EMBEDDING_DIMENSIONS

    def crash(_self, _table, _rowid):
        raise _Crash

    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_reconcile_pending_vector", crash)
        with pytest.raises(_Crash):
            storage._insert_interaction(
                Interaction(user_id="u", request_id="r", content="c", embedding=vector)
            )
    storage.conn.close()  # the process "dies" after the primary commit

    assert _pending(path) == [("interactions_vec", 1)]
    reopened = _open(path)  # repairs the pending row
    try:
        blob = reopened.conn.execute(
            "SELECT embedding FROM interactions_vec WHERE rowid = 1"
        ).fetchone()[0]
        assert struct.unpack(f"{EMBEDDING_DIMENSIONS}f", blob) == pytest.approx(
            tuple(vector)
        )
    finally:
        reopened.conn.close()
    assert _pending(path) == []


@requires_vec
def test_a_vector_write_outside_a_commit_scope_is_refused(tmp_path: Path) -> None:
    storage = _open(tmp_path / "guard.db")
    try:
        with pytest.raises(RuntimeError, match="inside commit_scope"):
            storage._vec_upsert("profiles_vec", 1, [0.0])
        with pytest.raises(RuntimeError, match="inside commit_scope"):
            storage._vec_delete("profiles_vec", 1)
    finally:
        storage.conn.close()


def test_a_caller_authorizer_survives_the_fallback_trackers_rearm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SQLiteStorage, "_try_load_sqlite_vec", lambda _self: False)
    storage = _open(tmp_path / "authorizer.db")
    try:

        def deny_create(action, _arg1, _arg2, _db, _source):
            if action == sqlite3.SQLITE_CREATE_TABLE:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        storage.conn.set_authorizer(deny_create)
        storage.conn.commit()  # re-arms the tracker, which must still chain
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            storage.conn.execute("CREATE TABLE denied_by_caller (x)")
        storage.conn.set_authorizer(None)  # clears the caller, keeps the tracker
        storage.conn.execute("CREATE TABLE allowed_after_clear (x)")
    finally:
        storage.conn.close()


@requires_vec
def test_backfill_examines_one_page_of_primary_rows_per_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
    from reflexio.server.services.storage.sqlite_storage.migrations import _steps

    path = tmp_path / "paged.db"
    storage = _open(path)
    for number in range(1, 11):
        storage.conn.execute(
            "INSERT INTO profiles (profile_id, user_id, last_modified_timestamp, "
            "embedding) VALUES (?, 'u', 1, ?)",
            (f"p{number}", json.dumps([0.5] * EMBEDDING_DIMENSIONS)),
        )
    # Index everything except the last row, which sits in the last page.
    storage.conn.execute(
        "INSERT INTO profiles_vec(rowid, embedding) "
        "SELECT rowid, embedding FROM profiles WHERE rowid < 10"
    )
    storage.conn.execute("PRAGMA user_version = 0")
    storage.conn.commit()
    storage.conn.close()

    monkeypatch.setattr(_steps, "_VEC_BACKFILL_PAGE", 3)
    statements: list[str] = []
    real_connect = sqlite3.connect

    def tracing_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracing_connect)
    reopened = _open(path)
    try:
        assert (
            reopened.conn.execute("SELECT COUNT(*) FROM profiles_vec").fetchone()[0]
            == 10
        )
    finally:
        reopened.conn.close()
    pages = [q for q in statements if "AS indexed FROM profiles t" in q]
    assert len(pages) == 4  # ceil(10 / 3): keyset pages, never more than 3 rows each
    assert all("LIMIT 3" in q for q in pages)


@requires_vec
def test_every_deferred_op_is_attempted_after_one_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
    from reflexio.server.services.storage.error import PostCommitStorageError
    from reflexio.server.services.storage.sqlite_storage.migrations._steps import (
        SQLiteSchemaMigrationsMixin,
    )

    path = tmp_path / "all-ops.db"
    storage = _open(path)
    real = SQLiteSchemaMigrationsMixin._reconcile_pending_vector
    calls: list[int] = []

    def reconcile(self, table, row):
        calls.append(row)
        if len(calls) == 2:
            raise RuntimeError("second op failed")
        return real(self, table, row)

    with monkeypatch.context() as patched:
        patched.setattr(SQLiteStorage, "_reconcile_pending_vector", reconcile)
        with (
            pytest.raises(PostCommitStorageError, match="1 of 3"),
            storage.commit_scope(),
        ):
            first = _queue_profile(storage, "a", [0.5] * EMBEDDING_DIMENSIONS)
            second = _queue_profile(storage, "b", [0.25] * EMBEDDING_DIMENSIONS)
            third = _queue_profile(storage, "c", [0.125] * EMBEDDING_DIMENSIONS)
    assert len(calls) == 3  # the third was attempted after the second failed
    storage.conn.close()

    assert _pending(path) == [("profiles_vec", second)]
    assert _sidecar(path, first) == pytest.approx((0.5,) * EMBEDDING_DIMENSIONS)
    assert _sidecar(path, third) == pytest.approx((0.125,) * EMBEDDING_DIMENSIONS)


@requires_vec
def test_backfill_pages_by_existing_rowids_across_a_huge_sparse_gap(
    tmp_path: Path,
) -> None:
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

    path = tmp_path / "sparse.db"
    storage = _open(path)
    vector = json.dumps([0.5] * EMBEDDING_DIMENSIONS)
    rowids = (1, 2**40, 2**63 - 1)
    for number, rowid in enumerate(rowids):
        storage.conn.execute(
            "INSERT INTO profiles (rowid, profile_id, user_id, "
            "last_modified_timestamp, embedding) VALUES (?, ?, 'u', 1, ?)",
            (rowid, f"p{number}", vector),
        )
    # Index everything except the last row.
    storage.conn.execute(
        "INSERT INTO profiles_vec(rowid, embedding) "
        "SELECT rowid, embedding FROM profiles WHERE rowid < ?",
        (2**63 - 1,),
    )
    storage.conn.execute("PRAGMA user_version = 0")
    storage.conn.commit()
    storage.conn.close()

    statements: list[str] = []
    real_connect = sqlite3.connect

    def tracing_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)

        def trace(statement: str) -> None:
            statements.append(statement)
            if sum("FROM profiles t" in q for q in statements) > 10:
                conn.interrupt()  # range paging would walk ~2**52 pages: fail fast

        conn.set_trace_callback(trace)
        return conn

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(sqlite3, "connect", tracing_connect)
        reopened = _open(path)
    try:
        assert (
            reopened.conn.execute("SELECT COUNT(*) FROM profiles_vec").fetchone()[0]
            == 3
        )
    finally:
        reopened.conn.close()
    assert sum("FROM profiles t" in q for q in statements) == 1  # one keyset page


def test_a_second_migrate_after_a_completed_one_runs_no_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reflexio.server.services.storage.sqlite_storage import migrations

    storage = _open(tmp_path / "idempotent.db")
    stamped: list[int] = []
    real_stamp = migrations._stamp
    monkeypatch.setattr(
        migrations,
        "_stamp",
        lambda s, v, **k: (stamped.append(v), real_stamp(s, v, **k)),
    )
    try:
        storage.migrate()
        assert stamped == []
    finally:
        storage.conn.close()


def test_the_ledger_run_waits_for_the_cross_process_migration_lock(
    tmp_path: Path,
) -> None:
    import threading
    import time

    from reflexio.server.services.storage.sqlite_storage.migrations import (
        migration_lock,
    )

    path = tmp_path / "locked.db"
    storage = _open(path)
    with closing(sqlite3.connect(path)) as raw:
        raw.execute("PRAGMA user_version = 0")
        raw.commit()

    def version() -> int:
        with closing(sqlite3.connect(path)) as raw:
            return raw.execute("PRAGMA user_version").fetchone()[0]

    worker = threading.Thread(target=storage.migrate)
    try:
        with migration_lock(str(path)):
            worker.start()
            time.sleep(0.5)
            assert version() == 0  # blocked: nothing ran, nothing was stamped
        worker.join(timeout=120)
        assert not worker.is_alive()
        assert version() == LATEST_VERSION
    finally:
        storage.conn.close()
