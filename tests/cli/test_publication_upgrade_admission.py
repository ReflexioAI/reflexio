"""A refused storage cutover must leave the predecessor's config usable."""

import json
import re
import runpy
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from reflexio.cli.config_upgrade import prepare_local_config
from reflexio.config_document import decode_config_document
from reflexio.config_env import StoredConfigurationError
from reflexio.config_file import FILE_FORMAT_VERSION
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)
from reflexio.server.services.storage.sqlite_storage._dataset_path import derive_db_path


def legacy_database(path):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TABLE _agent_runs(id TEXT PRIMARY KEY,org_id TEXT,status TEXT,committed_output TEXT)"
        )
        conn.execute(
            "CREATE TABLE _agent_run_finalization_receipts(run_id TEXT PRIMARY KEY)"
        )
        conn.execute(
            "INSERT INTO _agent_runs VALUES(?,?,?,?)",
            ("old-run", "owner", "failed", '{"profiles":[]}'),
        )


def old_reader():
    return runpy.run_path(
        str(Path(__file__).parents[1] / "fixtures/config_file_v10_reader.txt"),
        init_globals={
            "FILE_FORMAT_VERSION": 10,
            "_METADATA_KEY": "_reflexio_config_file",
            "decode_config_document": decode_config_document,
            "StoredConfigurationError": StoredConfigurationError,
            "re": re,
        },
    )["_decode"]


def test_startup_refusal_keeps_old_reader_then_allows_drained_upgrade(tmp_path):
    db = tmp_path / "old.sqlite"
    legacy_database(db)
    storage = LocalFileConfigStorage("owner", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir()
    original = json.dumps(
        {
            "_reflexio_config_file": {"version": 10},
            "storage_config": {"type": "sqlite", "db_path": str(db)},
            "pending_tool_call_config": {"resume_claim_ttl_seconds": 1800},
        }
    ).encode()
    path.write_bytes(original)
    reader = old_reader()
    with pytest.raises(ValueError, match="Legacy publication cutover refused"):
        prepare_local_config("owner", base_dir=str(tmp_path))
    assert path.read_bytes() == original
    assert reader(path.read_bytes())[1]["version"] == 10
    assert not (path.parent / ".config-migrations").exists()
    with closing(sqlite3.connect(db)) as conn, conn:
        assert conn.execute(
            "SELECT status,committed_output FROM _agent_runs WHERE id='old-run'"
        ).fetchone() == ("failed", '{"profiles":[]}')
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='_agent_output_receipts'"
            ).fetchone()
            is None
        )
        # Simulate completion by the old writer; admission itself never repairs it.
        conn.execute("UPDATE _agent_runs SET status='finalized' WHERE id='old-run'")
    assert prepare_local_config("owner", base_dir=str(tmp_path))
    assert (
        json.loads(path.read_bytes())["_reflexio_config_file"]["version"]
        == FILE_FORMAT_VERSION
    )
    with pytest.raises(StoredConfigurationError):
        reader(path.read_bytes())


def test_missing_config_cannot_publish_over_existing_default_target_debt(tmp_path):
    db = derive_db_path(tmp_path, "owner")
    legacy_database(db)
    storage = LocalFileConfigStorage("owner", base_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Legacy publication cutover refused"):
        storage.load_config()
    assert not Path(storage.config_file).exists()
    assert not (Path(storage.base_dir) / ".config-migrations").exists()


def test_retargeting_cannot_hide_old_database_debt(tmp_path):
    db = tmp_path / "old.sqlite"
    legacy_database(db)
    storage = LocalFileConfigStorage("owner", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir()
    original = json.dumps(
        {
            "_reflexio_config_file": {"version": 10},
            "storage_config": {"type": "sqlite", "db_path": str(db)},
        }
    ).encode()
    path.write_bytes(original)

    def select_new_database(document, _version):
        document["storage_config"]["db_path"] = str(tmp_path / "fresh.sqlite")

    with pytest.raises(ValueError, match="Legacy publication cutover refused"):
        storage.prepare_for_startup(edit=select_new_database)
    assert path.read_bytes() == original
    assert not (tmp_path / "fresh.sqlite").exists()


@pytest.mark.parametrize("version", [10, 11])
@pytest.mark.parametrize("operation", ["prepare", "publish"])
def test_selected_target_is_admitted_even_without_a_generation_change(
    tmp_path, version, operation
):
    selected = tmp_path / "selected.sqlite"
    legacy_database(selected)
    storage = LocalFileConfigStorage("owner", base_dir=str(tmp_path))
    path = Path(storage.config_file)
    path.parent.mkdir()
    original = json.dumps(
        {
            "_reflexio_config_file": {"version": version},
            "storage_config": {
                "type": "sqlite",
                "db_path": str(tmp_path / "fresh.sqlite"),
            },
        }
    ).encode()
    path.write_bytes(original)

    def select(document, _version):
        document["storage_config"]["db_path"] = str(selected)

    with pytest.raises(ValueError, match="Legacy publication cutover refused"):
        if operation == "prepare":
            storage.prepare_for_startup(edit=select)
        else:
            candidate = json.loads(original)
            candidate["storage_config"]["db_path"] = str(selected)
            storage._publish_config_document(json.dumps(candidate))
    assert path.read_bytes() == original
    assert not (tmp_path / "fresh.sqlite").exists()
