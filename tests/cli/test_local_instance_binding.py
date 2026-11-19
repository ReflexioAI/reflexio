"""Local binding preserves existing installation and storage ownership."""

import os
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
        LocalInstanceError, match="different organization|no owner claim"
    ):
        bind_local_instance()
    assert not list((installation / "instances").glob("*.json"))
    assert original.read_bytes() == before_cfg
    assert db.read_bytes() == before_db


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
