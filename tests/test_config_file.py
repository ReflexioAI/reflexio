"""Local file generation, atomic publication and startup preparation contracts."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from reflexio.config_document import decode_config_document, encode_config_document
from reflexio.config_env import StoredConfigurationError
from reflexio.config_file import FILE_FORMAT_VERSION, LocalConfigFile
from reflexio.models.config_schema import StorageConfigSQLite, validate_stored_config


def test_preparation_preserves_original_policy_and_precision(tmp_path):
    path = tmp_path / "config.json"
    original = b'{"storage_config":{"type":"sqlite","db_path":"owned.db"},"playbook_optimizer_config":{"enabled":false,"optimize_agent_playbooks":true,"auto_update_pending_agent_playbooks":false},"opaque":{"number":1.00000000000000000000001}}'
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    with pytest.raises(StoredConfigurationError):
        validate_stored_config(owner.read_document())
    assert path.read_bytes() == original
    assert owner.prepare() is True
    logical = owner.read_document()
    assert logical["storage_config"] == {"type": "sqlite", "db_path": "owned.db"}
    assert logical["playbook_optimizer_config"] == {
        "enabled": False,
        "agent_playbooks_enabled": False,
    }
    assert str(logical["opaque"]["number"]) == "1.00000000000000000000001"
    validate_stored_config(logical)
    digest = hashlib.sha256(original).hexdigest()
    backup = tmp_path / ".config-migrations/config.json" / f"{digest}.json"
    assert backup.read_bytes() == original
    if os.name == "posix":
        assert backup.stat().st_mode & 0o777 == 0o600
    raw = decode_config_document(path.read_text())
    assert raw["_reflexio_config_file"] == {
        "version": FILE_FORMAT_VERSION,
        "original_sha256": digest,
    }
    before = path.stat().st_mtime_ns, path.read_bytes()
    assert owner.prepare() is False
    assert (path.stat().st_mtime_ns, path.read_bytes()) == before
    assert owner.publish_document(encode_config_document(logical)) is True
    assert (
        decode_config_document(path.read_text())["_reflexio_config_file"]
        == raw["_reflexio_config_file"]
    )
    assert owner.read_document() == logical


def test_fresh_publication_is_current_and_create_only_never_clobbers(tmp_path):
    path = tmp_path / "nested/config.json"
    owner = LocalConfigFile(path)
    assert owner.prepare() is False
    assert not path.exists()
    document = '{"storage_config":{"type":"sqlite"},"window_size":7,"stride_size":1}'
    assert owner.publish_document(document, create_only=True) is True
    original = path.read_bytes()
    assert owner.publish_document(document.replace("7", "8"), create_only=True) is False
    assert path.read_bytes() == original
    assert owner.read_document()["window_size"] == 7
    assert owner.prepare() is False


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"version": True},
        {"version": 0},
        {"version": FILE_FORMAT_VERSION + 1},
        {"version": 1, "foreign": "private-marker"},
    ],
)
def test_foreign_or_future_generation_fails_without_mutation(tmp_path, metadata):
    path = tmp_path / "config.json"
    raw = json.dumps(
        {"_reflexio_config_file": metadata, "storage_config": None}
    ).encode()
    path.write_bytes(raw)
    with pytest.raises(StoredConfigurationError) as error:
        LocalConfigFile(path).prepare()
    assert "private-marker" not in str(error.value)
    assert path.read_bytes() == raw
    assert not (tmp_path / ".config-migrations").exists()


def test_current_generation_never_translates_obsolete_aliases(tmp_path):
    path = tmp_path / "config.json"
    raw = json.dumps(
        {
            "_reflexio_config_file": {"version": FILE_FORMAT_VERSION},
            "storage_config": {"storage_type": "sqlite"},
        }
    )
    path.write_text(raw)
    with pytest.raises(StoredConfigurationError):
        LocalConfigFile(path).prepare()
    assert path.read_text() == raw


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_upgrade_preserves_link_and_refuses_dangling_config(tmp_path):
    target = tmp_path / "target.json"
    target.write_text('{"storage_config":{"type":"sqlite"}}')
    link = tmp_path / "selected.json"
    link.symlink_to(target)
    assert LocalConfigFile(link).prepare() is True
    assert link.is_symlink()
    assert LocalConfigFile(link).read_document()["storage_config"]["type"] == "sqlite"
    target.unlink()
    with pytest.raises(StoredConfigurationError, match="missing target"):
        LocalConfigFile(link).publish_document(
            '{"storage_config":null}', create_only=True
        )
    assert not target.exists()


def test_interrupted_upgrade_reuses_original_without_clobbering_it(
    tmp_path, monkeypatch
):
    path = tmp_path / "config.json"
    original = b'{"storage_config":{"type":"sqlite","db_path":"retained.db"},"opaque":{"keep":true}}'
    path.write_bytes(original)
    owner = LocalConfigFile(path)
    with monkeypatch.context() as patch:

        def interrupted_publish(*_args):
            raise OSError("interrupted publish")

        patch.setattr(Path, "replace", interrupted_publish)
        with pytest.raises(OSError, match="interrupted publish"):
            owner.prepare()
    assert path.read_bytes() == original
    backups = list((tmp_path / ".config-migrations/config.json").glob("*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert list(tmp_path.glob(".config.json.*")) == []
    assert owner.prepare() is True
    assert len(list(backups[0].parent.glob("*.json"))) == 1
    assert backups[0].read_bytes() == original


def test_post_publication_failure_never_separates_generation_from_data(
    tmp_path, monkeypatch
):
    from reflexio import config_file

    path = tmp_path / "config.json"
    original = b'{"storage_config":{"type":"sqlite"}}'
    path.write_bytes(original)
    sync = config_file._sync_directory

    def fail_after_publish(directory):
        if directory == tmp_path and "_reflexio_config_file" in decode_config_document(
            path.read_text()
        ):
            raise OSError("after atomic publication")
        sync(directory)

    with monkeypatch.context() as patch:
        patch.setattr(config_file, "_sync_directory", fail_after_publish)
        with pytest.raises(OSError, match="after atomic publication"):
            LocalConfigFile(path).prepare()
    document = decode_config_document(path.read_text())
    assert document["_reflexio_config_file"]["version"] == FILE_FORMAT_VERSION
    assert document["storage_config"] == {"type": "sqlite"}
    assert LocalConfigFile(path).prepare() is False
    backups = list((tmp_path / ".config-migrations/config.json").glob("*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == original


def test_observed_concurrent_edit_is_never_overwritten(tmp_path, monkeypatch):
    from reflexio import config_file

    path = tmp_path / "config.json"
    path.write_text('{"storage_config":{"type":"sqlite"}}')
    operator_edit = (
        b'{"storage_config":{"type":"sqlite"},"window_size":29,"stride_size":3}'
    )
    publish = config_file._publish_bytes

    def concurrent_edit(target, raw, *, create_only=False):
        result = publish(target, raw, create_only=create_only)
        if create_only:
            path.write_bytes(operator_edit)
        return result

    monkeypatch.setattr(config_file, "_publish_bytes", concurrent_edit)
    with pytest.raises(StoredConfigurationError, match="changed during preparation"):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == operator_edit


def _prepare_file_in_process(path):
    return LocalConfigFile(path).prepare()


def test_concurrent_startup_has_one_conversion_and_one_original(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    path = tmp_path / "config.json"
    original = b'{"storage_config":{"type":"sqlite"},"window_size":31,"stride_size":3}'
    path.write_bytes(original)
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        changed = list(pool.map(_prepare_file_in_process, [path, path]))
    assert sorted(changed) == [False, True]
    assert LocalConfigFile(path).read_document()["window_size"] == 31
    backups = list((tmp_path / ".config-migrations/config.json").glob("*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == original


def test_frozen_generation_one_file_has_an_ordinary_upgrade_path(tmp_path):
    """Keep this historical fixture static when later fields retire."""
    fixture = Path(__file__).parent / "fixtures/local_config_generation1.json"
    path = tmp_path / "config.json"
    path.write_bytes(fixture.read_bytes())
    owner = LocalConfigFile(path)
    assert owner.prepare() is (FILE_FORMAT_VERSION != 1)
    logical = owner.read_document()
    config = validate_stored_config(logical)
    assert isinstance(config.storage_config, StorageConfigSQLite)
    assert config.storage_config.db_path == "generation1-owned.sqlite"
    assert (config.window_size, config.stride_size) == (11, 3)
    assert config.playbook_optimizer_config.max_metric_calls == 7
    assert config.playbook_optimizer_config.min_commit_windows == 4
    assert config.playbook_optimizer_config.allows_target("agent_playbook") is False
    assert config.playbook_optimizer_config.allows_target("user_playbook") is True
    assert logical["opaque_extension"] == {"preserve": ["generation1", False]}
    assert "_reflexio_config_file" not in config.model_dump()


@pytest.mark.parametrize(
    "payload",
    [
        {"storage_config": {"storage_type": "sqlite"}},
        {"batch_size": 10},
        {"playbook_optimizer_config": {"optimize_user_playbooks": True}},
    ],
)
def test_unmarked_file_does_not_gain_unsupported_historical_aliases(tmp_path, payload):
    path = tmp_path / "config.json"
    original = json.dumps(payload).encode()
    path.write_bytes(original)
    with pytest.raises(StoredConfigurationError):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()


def test_bad_agent_migration_never_echoes_values_or_publishes(tmp_path):
    path = tmp_path / "config.json"
    original = b'{"playbook_optimizer_config":{"optimize_agent_playbooks":"private-invalid-marker"}}'
    path.write_bytes(original)
    with pytest.raises(StoredConfigurationError) as error:
        LocalConfigFile(path).prepare()
    assert "private-invalid-marker" not in str(error.value)
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()


def test_frozen_unmarked_generation_zero_retains_effective_agent_veto(tmp_path):
    fixture = Path(__file__).parent / "fixtures/local_config_generation0.json"
    path = tmp_path / "config.json"
    path.write_bytes(fixture.read_bytes())
    owner = LocalConfigFile(path)
    assert owner.prepare() is True
    config = validate_stored_config(owner.read_document())
    assert config.playbook_optimizer_config.allows_target("agent_playbook") is False
    assert config.playbook_optimizer_config.allows_target("user_playbook") is True
    assert config.playbook_optimizer_config.max_metric_calls == 7
    assert isinstance(config.storage_config, StorageConfigSQLite)
    assert config.storage_config.db_path == "generation1-owned.sqlite"
    assert owner.read_document()["opaque_extension"] == {
        "preserve": ["generation1", False]
    }


def test_storage_edit_preserves_raw_fields_and_file_receipt(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        '{"storage_config":{"type":"sqlite"},"opaque":{"number":1.00000000000000000001}}'
    )
    owner = LocalConfigFile(path)
    owner.prepare()
    receipt = decode_config_document(path.read_text())["_reflexio_config_file"]
    owner.edit_document(lambda data: data["storage_config"].update(db_path="owned.db"))
    assert owner.read_document()["storage_config"]["db_path"] == "owned.db"
    assert str(owner.read_document()["opaque"]["number"]) == "1.00000000000000000001"
    assert decode_config_document(path.read_text())["_reflexio_config_file"] == receipt
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert owner.edit_document(lambda _data: None) is False
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_invalid_storage_edit_never_publishes(tmp_path):
    path = tmp_path / "config.json"
    owner = LocalConfigFile(path)
    owner.publish_document('{"storage_config":{"type":"sqlite"}}')
    original = path.read_bytes()
    with pytest.raises(StoredConfigurationError):
        owner.edit_document(
            lambda data: data.update(storage_config={"type": "unsupported"})
        )
    assert path.read_bytes() == original


def test_observed_foreign_edit_is_not_overwritten(tmp_path):
    path = tmp_path / "config.json"
    owner = LocalConfigFile(path)
    owner.publish_document('{"storage_config":{"type":"sqlite"}}')
    foreign = b'{"storage_config":{"type":"sqlite","db_path":"other.db"}}'

    def edit(data):
        data["storage_config"]["db_path"] = "owned.db"
        path.write_bytes(foreign)

    with pytest.raises(StoredConfigurationError, match="changed during editing"):
        owner.edit_document(edit)
    assert path.read_bytes() == foreign


@pytest.mark.parametrize("linked", ["root", "directory", "backup"])
def test_preparation_refuses_linked_backup_paths(tmp_path, linked):
    path = tmp_path / "config.json"
    original = b'{"storage_config":{"type":"sqlite"}}'
    path.write_bytes(original)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    root = tmp_path / ".config-migrations"
    directory = root / path.name
    digest = hashlib.sha256(original).hexdigest()
    if linked == "root":
        root.symlink_to(foreign, target_is_directory=True)
    elif linked == "directory":
        root.mkdir()
        directory.symlink_to(foreign, target_is_directory=True)
    else:
        directory.mkdir(parents=True)
        (directory / f"{digest}.json").symlink_to(foreign / "missing.json")
    with pytest.raises(StoredConfigurationError):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == original
    assert not list(foreign.iterdir())


def test_backup_directory_sync_failure_keeps_original_config(tmp_path, monkeypatch):
    import reflexio.config_file as config_file

    path = tmp_path / "config.json"
    original = b'{"storage_config":{"type":"sqlite"}}'
    path.write_bytes(original)
    sync = config_file._sync_directory

    def fail_backup_directory(directory):
        if directory == tmp_path / ".config-migrations":
            raise OSError("backup directory not durable")
        sync(directory)

    monkeypatch.setattr(config_file, "_sync_directory", fail_backup_directory)
    with pytest.raises(OSError, match="not durable"):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "legacy",
    [
        {},
        {
            "optimize_agent_playbooks": False,
            "auto_update_pending_agent_playbooks": False,
        },
    ],
)
def test_automatic_upgrade_never_adopts_previously_ignored_consent(tmp_path, legacy):
    from reflexio.cli.config_upgrade import convert_config_file

    path = tmp_path / "unmarked.json"
    original = json.dumps(
        {
            "storage_config": None,
            "playbook_optimizer_config": {
                "enabled": True,
                "agent_playbooks_enabled": True,
                **legacy,
            },
        }
    ).encode()
    path.write_bytes(original)
    with pytest.raises(StoredConfigurationError, match="previous schema"):
        LocalConfigFile(path).prepare()
    assert path.read_bytes() == original
    with pytest.raises(StoredConfigurationError, match="previous schema"):
        LocalConfigFile(path).edit_document(lambda data: data.update(window_size=12))
    assert path.read_bytes() == original
    assert not (tmp_path / ".config-migrations").exists()
    # A deliberately selected conversion can keep canonical precedence, and
    # must produce an installable current file, not repeat the ambiguity.
    output = tmp_path / "explicit-current.json"
    convert_config_file(path, output)
    current = LocalConfigFile(output)
    assert current.prepare() is False
    assert validate_stored_config(
        current.read_document()
    ).playbook_optimizer_config.allows_target("agent_playbook")
    assert path.read_bytes() == original
