"""The SQLite schema ledger: ordering, historical upgrades and the warm open."""

import sqlite3
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
    stored_version,
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


def test_vec_step_is_not_stamped_without_the_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "novec.db"
    monkeypatch.setattr(SQLiteStorage, "_try_load_sqlite_vec", lambda _self: False)
    storage = _open(path)
    try:
        assert stored_version(storage) == 10
    finally:
        storage.conn.close()
    monkeypatch.undo()

    storage = _open(path)
    try:
        assert stored_version(storage) == LATEST_VERSION
    finally:
        storage.conn.close()


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
