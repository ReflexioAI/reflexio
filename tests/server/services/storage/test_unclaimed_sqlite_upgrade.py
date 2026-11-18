"""Explicit namespace choice never rewrites actual old-plugin business history."""

import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from reflexio.server.services.storage.sqlite_storage import _dataset_path
from reflexio.server.services.storage.sqlite_storage import (
    _learning_policy_authority as owner,
)
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionBlockedError,
)

FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "test_data/sqlite_70daf32c/plugin_explicit_path"
)
NAMESPACE = "claude-smart"


def retained(tmp_path, cohort="completed"):
    compressed = (FIXTURES / f"{cohort}.sqlite.gz").read_bytes()
    proof = json.loads((FIXTURES / "provenance.json").read_text())["cohorts"][cohort]
    assert hashlib.sha256(compressed).hexdigest() == proof["gzip_sha256"]
    raw = gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == proof["raw_sha256"]
    path = tmp_path / "retained.sqlite"
    path.write_bytes(raw)
    return path


def assert_history(path, cohort="completed"):
    snapshot = json.loads((FIXTURES / f"{cohort}.snapshot.json").read_text())
    with closing(sqlite3.connect(path)) as conn:
        for table, expected in snapshot.items():
            assert table.replace("_", "").isalnum()
            assert all(c.replace("_", "").isalnum() for c in expected["columns"])
            columns = ",".join('"' + c + '"' for c in expected["columns"])
            rows = conn.execute(
                f'SELECT {columns} FROM "{table}" ORDER BY rowid LIMIT 101'  # noqa: S608 -- Frozen, identifier-validated synthetic snapshot.
            ).fetchall()
            assert json.loads(json.dumps(rows)) == expected["rows"], table


def assert_no_claim_or_authority(path):
    with closing(sqlite3.connect(path)) as conn:
        assert not conn.execute(
            "SELECT name FROM sqlite_schema WHERE name IN ('_dataset_identity','_learning_policy_authority')"
        ).fetchall()


@pytest.mark.parametrize("cohort", ["empty", "completed", "eligible_queued"])
def test_explicit_operator_namespace_preserves_genuine_plugin_history(tmp_path, cohort):
    path = retained(tmp_path, cohort)
    before = path.read_bytes()
    with pytest.raises(LearningTransitionBlockedError, match="explicit --org-id"):
        owner.upgrade_existing_sqlite_dataset(path)
    assert path.read_bytes() == before
    result = owner.upgrade_existing_sqlite_dataset(path, org_id=NAMESPACE)
    assert result.established and result.authority.phase == "open"
    assert result.provenance is not None
    assert result.provenance["identity_origin"] == "operator_established"
    assert result.provenance["org_id"] == NAMESPACE
    assert len(result.provenance["source_catalog_sha256"]) == 64
    assert_history(path, cohort)
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT k,org_id FROM _dataset_identity").fetchall() == [
            (1, NAMESPACE)
        ]
    for selected in (NAMESPACE, None):
        retry = owner.upgrade_existing_sqlite_dataset(path, org_id=selected)
        assert not retry.established and retry.provenance is None
        assert retry.authority == result.authority
    with pytest.raises(LearningTransitionBlockedError, match="conflicts"):
        owner.upgrade_existing_sqlite_dataset(path, org_id="other")
    assert_history(path, cohort)


@pytest.mark.parametrize("cohort", ["billing_pending", "window_pending"])
def test_namespace_choice_cannot_discard_unfinished_work(tmp_path, cohort):
    path = retained(tmp_path, cohort)
    before = path.read_bytes()
    with pytest.raises(LearningTransitionBlockedError, match="closure"):
        owner.upgrade_existing_sqlite_dataset(path, org_id=NAMESPACE)
    assert path.read_bytes() == before
    assert_no_claim_or_authority(path)
    assert_history(path, cohort)


@pytest.mark.parametrize("namespace", ["", "../escape", "x" * 65, "wrong-owner"])
def test_invalid_or_foreign_namespace_never_claims_history(tmp_path, namespace):
    path = retained(tmp_path)
    before = path.read_bytes()
    with pytest.raises(LearningTransitionBlockedError):
        owner.upgrade_existing_sqlite_dataset(path, org_id=namespace)
    assert path.read_bytes() == before
    assert_no_claim_or_authority(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_claim",
        "custom_claim",
        "authority",
        "unknown",
        "null_scope",
        "project",
        "stalled",
    ],
)
def test_explicit_choice_is_not_a_schema_or_scope_bypass(tmp_path, mutation):
    path = retained(tmp_path)
    with closing(sqlite3.connect(path)) as conn, conn:
        if mutation == "empty_claim":
            conn.execute(f"CREATE TABLE {_dataset_path._IDENTITY_CLAIM_DEFINITION}")
        elif mutation == "custom_claim":
            conn.execute(
                "CREATE TABLE _dataset_identity(k INTEGER PRIMARY KEY,org_id TEXT,claimed_at TEXT)"
            )
        elif mutation == "authority":
            conn.execute(owner.AUTHORITY_DDL)
        elif mutation == "unknown":
            conn.execute("CREATE TABLE unknown_owner(value TEXT)")
        elif mutation == "null_scope":
            conn.execute(
                "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,predecessor,end_seq,manifest,policy) VALUES('owned-null-scope','u','profile',NULL,0,1,'[]','{}')"
            )
        elif mutation == "project":
            conn.execute("UPDATE extraction_cursors SET project_id='foreign-project'")
        else:
            conn.execute("UPDATE stall_state SET stalled=1")
    before = path.read_bytes()
    with pytest.raises(LearningTransitionBlockedError):
        owner.upgrade_existing_sqlite_dataset(path, org_id=NAMESPACE)
    assert path.read_bytes() == before


def test_competing_operator_namespaces_cannot_rebind_first_claim(tmp_path):
    path = retained(tmp_path, "empty")

    def attempt(namespace):
        try:
            return namespace, owner.upgrade_existing_sqlite_dataset(
                path, org_id=namespace
            )
        except LearningTransitionBlockedError:
            return namespace, None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ["first", "second"]))
    winners = [
        (namespace, result) for namespace, result in results if result is not None
    ]
    assert len(winners) == 1 and winners[0][1].established
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT org_id FROM _dataset_identity").fetchall() == [
            (winners[0][0],)
        ]
    assert_history(path, "empty")


@pytest.mark.parametrize("boundary", ["identity", "arm", "commit"])
def test_process_death_never_leaves_partial_namespace_or_policy(tmp_path, boundary):
    path = retained(tmp_path)
    code = """import os,sqlite3,sys
from pathlib import Path
from reflexio.server.services.storage.sqlite_storage import _dataset_path as identity
from reflexio.server.services.storage.sqlite_storage import _learning_policy_authority as owner
target=Path(sys.argv[1]); boundary=sys.argv[2]
if boundary=='identity':
 original=identity._claim_or_read_identity_db
 def die(conn,org):
  original(conn,org)
  os._exit(87)
 identity._claim_or_read_identity_db=die
elif boundary=='arm':
 original=owner._change_learning_policy_authority_db
 def die(*args,**kwargs):
  result=original(*args,**kwargs)
  if kwargs['action']=='arm':os._exit(87)
  return result
 owner._change_learning_policy_authority_db=die
else:
 original=sqlite3.connect
 class Crash(sqlite3.Connection):
  def commit(self):os._exit(87)
 def connect(database,*args,**kwargs):
  if database==target.as_uri()+'?mode=rw':kwargs['factory']=Crash
  return original(database,*args,**kwargs)
 owner.sqlite3.connect=connect
owner.upgrade_existing_sqlite_dataset(target,org_id='claude-smart')
raise AssertionError('crash boundary was not reached')
"""
    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper()
        in {"PATH", "TMPDIR", "TEMP", "TMP", "LANG", "SYSTEMROOT", "WINDIR"}
    }
    env.update(
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),
        PYTHONPATH=str(Path(__file__).resolve().parents[4]),
        PYTHON_DOTENV_DISABLED="1",
    )
    result = subprocess.run(  # noqa: S603 -- Owned synthetic SQLite and fixed script, no shell.
        [sys.executable, "-c", code, str(path), boundary],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 87, result.stdout + result.stderr
    assert_no_claim_or_authority(path)
    assert_history(path)


@pytest.mark.parametrize(
    "cohort",
    ["empty", "completed", "eligible_queued", "billing_pending", "window_pending"],
)
def test_actual_early_cli_qualifies_or_holds_genuine_plugin_history(tmp_path, cohort):
    path = retained(tmp_path, cohort)
    original = path.read_bytes()
    home = tmp_path / "home"
    home.mkdir()
    config = home / "operator.env"
    original_env = b"OPERATOR_SECRET=private-owned-canary\nREFLEXIO_DURABLE_LEARNING_POLL_SECONDS=invalid\n"
    config.write_bytes(original_env)
    settings = home / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "storage_config": {"type": "sqlite", "db_path": str(path)},
                "keep": "unchanged",
            }
        )
    )
    original_settings = settings.read_bytes()
    environment = {
        k: v
        for k, v in os.environ.items()
        if k.upper()
        in {"PATH", "TMPDIR", "TEMP", "TMP", "LANG", "SYSTEMROOT", "WINDIR"}
    }
    environment.update(
        HOME=str(home),
        USERPROFILE=str(home),
        PYTHONPATH=str(Path(__file__).resolve().parents[4]),
        REFLEXIO_ENV_FILE=str(config),
    )
    code = """import runpy
from reflexio.cli import env_loader
def forbidden(*args,**kwargs):raise AssertionError('operator env loader called')
env_loader.load_existing_reflexio_env=forbidden
env_loader.load_reflexio_env=forbidden
runpy.run_module('reflexio.cli',run_name='__main__')
"""
    outputs = []
    for _ in range(2):
        result = subprocess.run(  # noqa: S603 -- Fixed early CLI and owned synthetic fixture, no shell.
            [
                sys.executable,
                "-c",
                code,
                "--json",
                "data",
                "upgrade-local",
                str(path),
                "--org-id",
                NAMESPACE,
            ],
            env=environment,
            cwd=home,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "private-owned-canary" not in result.stdout + result.stderr
        if cohort in {"billing_pending", "window_pending"}:
            assert result.returncode == 2
            assert json.loads(result.stderr)["ok"] is False
            assert path.read_bytes() == original
            assert_no_claim_or_authority(path)
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            outputs.append(json.loads(result.stdout)["data"])
        assert config.read_bytes() == original_env
        assert settings.read_bytes() == original_settings
        assert_history(path, cohort)
    if not outputs:
        return
    assert outputs[0]["established"] is True
    assert outputs[0]["provenance"]["identity_origin"] == "operator_established"
    assert outputs[1]["established"] is False and outputs[1]["provenance"] is None
    assert outputs[0]["authority"] == outputs[1]["authority"]
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(org_id=NAMESPACE, db_path=str(path))
    try:
        assert storage.count_all_profiles() == (1 if cohort == "completed" else 0)
        if cohort == "eligible_queued":
            claim = storage.claim_extraction("owned-plugin-resume", 300)
            assert claim is not None
            user, token = claim
            window = storage.prepare_extraction(user, token)
            assert window is not None and len(window.manifest) == 10
    finally:
        storage.conn.close()


def test_non_sqlite_target_is_not_repaired(tmp_path):
    path = tmp_path / "not-sqlite"
    path.write_bytes(b"not a SQLite database")
    with pytest.raises(sqlite3.DatabaseError):
        owner.upgrade_existing_sqlite_dataset(path, org_id=NAMESPACE)
    assert path.read_bytes() == b"not a SQLite database"


def test_lost_commit_ack_retains_atomic_claim_and_original_root(tmp_path, monkeypatch):
    path = retained(tmp_path)
    connect = sqlite3.connect

    class LostAck(sqlite3.Connection):
        def commit(self):
            super().commit()
            raise OSError("owned lost acknowledgement")

    def uncertain(database, *args, **kwargs):
        if database == path.as_uri() + "?mode=rw":
            kwargs["factory"] = LostAck
        return connect(database, *args, **kwargs)

    with monkeypatch.context() as patching:
        patching.setattr(owner.sqlite3, "connect", uncertain)
        with pytest.raises(OSError, match="acknowledgement"):
            owner.upgrade_existing_sqlite_dataset(path, org_id=NAMESPACE)
    with closing(connect(path)) as conn:
        claim = conn.execute(
            "SELECT k,org_id,claimed_at FROM _dataset_identity"
        ).fetchall()
        raw = conn.execute(
            "SELECT authority FROM _learning_policy_authority"
        ).fetchone()[0]
    retry = owner.upgrade_existing_sqlite_dataset(path, org_id=NAMESPACE)
    assert not retry.established and retry.provenance is None
    assert retry.authority.to_dict() == json.loads(raw)
    assert retry.authority.phase == "open" and retry.authority.generation == 2
    with closing(connect(path)) as conn:
        assert (
            conn.execute("SELECT k,org_id,claimed_at FROM _dataset_identity").fetchall()
            == claim
        )
    assert_history(path)


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("commit_first", [False, True])
def test_replacement_during_commit_never_reports_success(
    tmp_path, monkeypatch, retry, commit_first
):
    path = retained(tmp_path)
    if retry:
        owner.upgrade_existing_sqlite_dataset(path, org_id=NAMESPACE)
    displaced = tmp_path / "displaced.sqlite"
    replacement = gzip.decompress((FIXTURES / "empty.sqlite.gz").read_bytes())
    connect = sqlite3.connect

    class ReplaceAfterCommit(sqlite3.Connection):
        def commit(self):
            if commit_first:
                super().commit()
            self.close()  # Windows permits replacement after the handle closes.
            path.rename(displaced)
            path.write_bytes(replacement)

        def rollback(self):
            pass  # The transaction committed before replacement; no false undo.

    def replacing(database, *args, **kwargs):
        if database == path.as_uri() + "?mode=rw":
            kwargs["factory"] = ReplaceAfterCommit
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(owner.sqlite3, "connect", replacing)
    with pytest.raises(LearningTransitionBlockedError, match="file changed"):
        owner.upgrade_existing_sqlite_dataset(path, org_id=NAMESPACE)
    assert path.read_bytes() == replacement
    assert_no_claim_or_authority(path)
    assert_history(displaced)
    if not commit_first and not retry:
        assert_no_claim_or_authority(displaced)
        return
    with closing(connect(displaced)) as conn:
        assert conn.execute("SELECT org_id FROM _dataset_identity").fetchall() == [
            (NAMESPACE,)
        ]
        assert (
            json.loads(
                conn.execute(
                    "SELECT authority FROM _learning_policy_authority"
                ).fetchone()[0]
            )["phase"]
            == "open"
        )


@pytest.mark.parametrize("namespace", [None, "", "../private-invalid", "wrong-owner"])
def test_cli_namespace_refusal_is_safe_validation_not_traceback(tmp_path, namespace):
    from typer.testing import CliRunner

    from reflexio.cli.app import create_app

    path = retained(tmp_path)
    before = path.read_bytes()
    args = ["--json", "data", "upgrade-local", str(path)]
    if namespace is not None:
        args.extend(["--org-id", namespace])
    result = CliRunner().invoke(create_app(), args)
    assert result.exit_code == 2
    assert json.loads(result.stderr)["ok"] is False
    assert "Traceback" not in result.stderr and "private-invalid" not in result.stderr
    assert path.read_bytes() == before


@pytest.mark.parametrize("namespace", ["historical_http", "wrong-owner"])
def test_explicit_choice_preserves_existing_historical_claim(tmp_path, namespace):
    path = tmp_path / "claimed.sqlite"
    path.write_bytes(
        gzip.decompress((FIXTURES.parent / "completed.sqlite.gz").read_bytes())
    )
    before = path.read_bytes()
    if namespace != "historical_http":
        with pytest.raises(LearningTransitionBlockedError, match="conflicts"):
            owner.upgrade_existing_sqlite_dataset(path, org_id=namespace)
        assert path.read_bytes() == before
    else:
        result = owner.upgrade_existing_sqlite_dataset(path, org_id=namespace)
        assert result.provenance is not None
        assert result.provenance["identity_origin"] == "retained_claim"
