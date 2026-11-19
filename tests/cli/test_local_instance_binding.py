"""Local binding preserves existing installation and storage ownership."""

import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from reflexio.cli.bootstrap_config import default_config_path
from reflexio.config_file import encode_local_config_document
from reflexio.local_instance import (
    LocalInstanceError,
    bind_local_instance,
    configured_local_org_id,
    prepare_local_sqlite_instance,
)
from reflexio.server import api, auth
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def installation(monkeypatch, tmp_path):
    monkeypatch.setenv("REFLEXIO_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("REFLEXIO_ENV_FILE", str(tmp_path / "plugin.env"))
    monkeypatch.setenv("REFLEXIO_DEFAULT_ORG_ID", "plugin")
    return tmp_path / ".reflexio"


def _config(root, org, storage):
    path = root / "configs" / f"config_{org}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encode_local_config_document({"storage_config": storage}))
    return path


def _dataset(path, org):
    path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteStorage(org_id=org, db_path=str(path))
    store.conn.close()
    # Explicit historical SQLite paths do not claim themselves. Use the
    # real storage ownership writer when seeding an already admitted store.
    from reflexio.server.services.storage.sqlite_storage._dataset_path import (
        claim_or_read_identity,
    )

    assert claim_or_read_identity(path, org) == org
    return path


def test_binding_survives_missing_legacy_override_in_cli_and_app(
    installation, monkeypatch
):
    db = _dataset(installation / "data" / "reflexio.db", "plugin")
    original = _config(installation, "plugin", {"type": "sqlite", "db_path": str(db)})
    before_cfg, before_db = original.read_bytes(), db.read_bytes()

    assert bind_local_instance() == "plugin"
    monkeypatch.delenv("REFLEXIO_DEFAULT_ORG_ID")
    assert configured_local_org_id() == bind_local_instance() == "plugin"
    assert default_config_path() == original
    assert original.read_bytes() == before_cfg
    assert db.read_bytes() == before_db

    routes = APIRouter()

    @routes.get("/test-bound-instance")
    def identity(org_id: str = Depends(auth.default_get_org_id)) -> str:
        return org_id

    app = api.create_app(additional_routers=[routes], mount_data_plane=False)
    assert TestClient(app).get("/test-bound-instance").json() == "plugin"

    monkeypatch.setenv("REFLEXIO_DEFAULT_ORG_ID", "unrelated")
    with pytest.raises(LocalInstanceError, match="conflicts"):
        api.create_app()
    api.create_app(get_org_id=lambda: "authenticated")


def test_implicit_legacy_file_stays_selected_if_a_derived_file_appears_later(
    installation, monkeypatch
):
    import reflexio.server as runtime

    root = installation / "data"
    monkeypatch.setattr(runtime, "LOCAL_STORAGE_PATH", str(root))
    legacy = _dataset(root / "reflexio.db", "plugin")
    original = _config(installation, "plugin", {"type": "sqlite"})
    before_db = legacy.read_bytes()

    assert bind_local_instance() == "plugin"
    assert str(legacy) in original.read_text()
    _dataset(root / "reflexio_plugin.db", "plugin")
    assert configured_local_org_id() == "plugin"
    assert legacy.read_bytes() == before_db

    legacy.rename(root / "retained.db")
    with pytest.raises(LocalInstanceError, match="file is missing"):
        configured_local_org_id()


@pytest.mark.parametrize("ownership", ["foreign", "unclaimed"])
def test_binding_refuses_an_unverified_sqlite_file(installation, ownership):
    db = installation / "data" / "reflexio.db"
    if ownership == "foreign":
        _dataset(db, "other")
    else:
        db.parent.mkdir(parents=True)
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("CREATE TABLE existing_data (body TEXT)")
    original = _config(installation, "plugin", {"type": "sqlite", "db_path": str(db)})
    before_cfg, before_db = original.read_bytes(), db.read_bytes()

    with pytest.raises(
        LocalInstanceError, match="different organization|no matching creation proof"
    ):
        bind_local_instance()
    assert not list((installation / "instances").glob("*.json"))
    assert original.read_bytes() == before_cfg
    assert db.read_bytes() == before_db


@pytest.mark.parametrize(
    "case", ["original", "copy", "foreign creator", "foreign rows"]
)
def test_explicit_file_claim_requires_untouched_creation_scope(
    installation, monkeypatch, case
):
    db = installation / "data" / "original.db"
    store = SQLiteStorage(
        org_id="other" if case == "foreign creator" else "plugin", db_path=str(db)
    )
    with store.conn:
        store.conn.execute("CREATE TABLE retained (org_id TEXT NOT NULL, body TEXT)")
        store.conn.execute(
            "INSERT INTO retained VALUES (?, 'existing content')",
            ("other" if case == "foreign rows" else "plugin",),
        )
    store.conn.close()
    if case == "copy":
        copied = db.with_name("copied.db")
        shutil.copyfile(db, copied)
        db = copied
    config = _config(installation, "plugin", {"type": "sqlite", "db_path": str(db)})
    before_config, before_db = config.read_bytes(), db.read_bytes()

    def snapshot():
        with closing(sqlite3.connect(db)) as conn:
            return (
                conn.execute("SELECT org_id, body FROM retained").fetchall(),
                conn.execute(
                    "SELECT authority FROM _learning_policy_authority"
                ).fetchall(),
                conn.execute(
                    "SELECT name, sql FROM sqlite_schema "
                    "WHERE tbl_name <> '_dataset_identity' ORDER BY name"
                ).fetchall(),
            )

    before_data = snapshot()
    if case == "original":
        assert bind_local_instance() == "plugin"
        assert snapshot() == before_data
        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute("SELECT org_id FROM _dataset_identity").fetchall() == [
                ("plugin",)
            ]
        monkeypatch.delenv("REFLEXIO_DEFAULT_ORG_ID")
        assert configured_local_org_id() == "plugin"
        claimed = db.read_bytes()
        assert bind_local_instance() == "plugin"
        assert db.read_bytes() == claimed
    else:
        with pytest.raises(
            LocalInstanceError,
            match="no matching creation proof|different organization",
        ):
            bind_local_instance()
        assert not list((installation / "instances").glob("*.json"))
        assert snapshot() == before_data
        assert db.read_bytes() == before_db
    assert config.read_bytes() == before_config


def test_two_env_addresses_keep_separate_instances_in_one_home(
    installation, monkeypatch
):
    policy = {"type": "postgres", "db_url": "postgresql://local:fixture@localhost/db"}
    _config(installation, "plugin", policy)
    _config(installation, "standalone", policy)
    first = installation.parent / "plugin.env"
    second = installation.parent / "standalone.env"
    assert bind_local_instance() == "plugin"
    monkeypatch.setenv("REFLEXIO_ENV_FILE", str(second))
    monkeypatch.setenv("REFLEXIO_DEFAULT_ORG_ID", "standalone")
    assert bind_local_instance() == "standalone"
    monkeypatch.delenv("REFLEXIO_DEFAULT_ORG_ID")
    assert configured_local_org_id() == "standalone"
    monkeypatch.setenv("REFLEXIO_ENV_FILE", str(first))
    assert configured_local_org_id() == "plugin"


def test_competing_processes_keep_the_first_binding(installation):
    policy = {"type": "postgres", "db_url": "postgresql://local:fixture@localhost/db"}
    for org in ("plugin", "standalone"):
        _config(installation, org, policy)

    argv = [
        sys.executable,
        "-m",
        "reflexio.cli",
        "--json",
        "config",
        "bind-local",
    ]
    env = {**os.environ, "PYTHON_DOTENV_DISABLED": "1"}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    # Both processes address the same file, but still use their independently
    # selected legacy organization. Publishing must never replace the winner.
    with (
        subprocess.Popen(  # noqa: S603 -- Fixed local CLI command and isolated fixture.
            argv,
            cwd=installation.parent,
            env={**env, "REFLEXIO_DEFAULT_ORG_ID": "plugin"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as first,
        subprocess.Popen(  # noqa: S603 -- Fixed local CLI command and isolated fixture.
            argv,
            cwd=installation.parent,
            env={**env, "REFLEXIO_DEFAULT_ORG_ID": "standalone"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as second,
    ):
        outcomes = [
            (*process.communicate(timeout=25), process.returncode)
            for process in (first, second)
        ]
    assert [code == 0 for _, _, code in outcomes].count(True) == 1, outcomes
    winner = "plugin" if first.returncode == 0 else "standalone"
    loser = outcomes[1 if winner == "plugin" else 0]
    assert "conflicts" in loser[1]
    assert [
        record.read_text().strip()
        for record in (installation / "instances").glob("*.json")
    ] == [f'{{"org_id": "{winner}"}}']


def test_first_install_resumes_its_reservation_after_config_publication_failure(
    installation, monkeypatch
):
    from reflexio.config_file import LocalConfigFile

    monkeypatch.delenv("REFLEXIO_DEFAULT_ORG_ID")
    db = installation / "data" / "first.db"
    original_publisher = LocalConfigFile.publish_document

    def interrupted(*_args, **_kwargs):
        raise OSError("simulated interruption before config publication")

    monkeypatch.setattr(LocalConfigFile, "publish_document", interrupted)
    with pytest.raises(LocalInstanceError, match="fresh SQLite installation"):
        prepare_local_sqlite_instance("plugin", db)
    assert not list((installation / "configs").glob("config_*.json"))
    assert not list((installation / "instances").glob("*.json"))
    inode = db.stat().st_ino

    monkeypatch.setattr(LocalConfigFile, "publish_document", original_publisher)
    assert prepare_local_sqlite_instance("plugin", db) == "plugin"
    assert configured_local_org_id() == "plugin"
    assert str(db) in default_config_path().read_text()
    assert db.stat().st_ino == inode

    # First storage open consumes the sealed reservation on the same file.
    storage = SQLiteStorage(org_id="plugin", db_path=str(db))
    try:
        assert (
            storage.conn.execute("SELECT org_id FROM _dataset_identity").fetchone()[0]
            == "plugin"
        )
    finally:
        storage.conn.close()
    assert db.stat().st_ino == inode

    db.rename(db.with_name("retained.db"))
    with pytest.raises(LocalInstanceError, match="file is missing"):
        prepare_local_sqlite_instance("plugin", db)
    assert not db.exists()


def test_first_install_never_attaches_an_unproven_existing_database(
    installation, monkeypatch
):
    monkeypatch.delenv("REFLEXIO_DEFAULT_ORG_ID")
    db = _dataset(installation / "data" / "existing.db", "plugin")
    before = db.read_bytes()
    with pytest.raises(LocalInstanceError, match="fresh SQLite installation"):
        prepare_local_sqlite_instance("plugin", db)

    assert db.read_bytes() == before
    assert not list((installation / "instances").glob("*.json"))


def test_host_preparation_keeps_the_existing_saved_target(installation, monkeypatch):
    monkeypatch.delenv("REFLEXIO_DEFAULT_ORG_ID")
    db = _dataset(installation / "data" / "saved.db", "plugin")
    original = _config(installation, "plugin", {"type": "sqlite", "db_path": str(db)})
    before = original.read_bytes()
    unused = db.with_name("new-default.db")

    assert prepare_local_sqlite_instance("plugin", unused) == "plugin"
    assert default_config_path().read_bytes() == before
    assert not unused.exists()


def test_host_can_bind_saved_non_sqlite_config_without_a_fresh_target(
    installation, monkeypatch
):
    monkeypatch.delenv("REFLEXIO_DEFAULT_ORG_ID")
    original = _config(
        installation,
        "plugin",
        {"type": "postgres", "db_url": "postgresql://local:fixture@localhost/db"},
    )
    before = original.read_bytes()
    assert prepare_local_sqlite_instance("plugin") == "plugin"
    assert default_config_path().read_bytes() == before
    assert not (installation / "data").exists()
