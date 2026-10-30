"""Native backend selection publishes one observed file policy, not env defaults."""

import hashlib
import json

import pytest

from reflexio.cli.bootstrap_config import save_storage_to_config
from reflexio.cli.config_upgrade import prepare_local_config
from reflexio.config_file import FILE_FORMAT_VERSION, LocalConfigFile

ORG = "native-owner"
POOL_ENV = {
    "REFLEXIO_POSTGRES_POOL_SIZE": "pool_size",
    "REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT": "pool_acquire_timeout",
    "REFLEXIO_POSTGRES_READ_POOL_SIZE": "read_pool_size",
    "REFLEXIO_POSTGRES_READ_POOL_ACQUIRE_TIMEOUT": "read_pool_acquire_timeout",
}
POLICY = {
    "pool_size": 3,
    "pool_acquire_timeout": 7.5,
    "read_pool_size": 2,
    "read_pool_acquire_timeout": 4.5,
}


@pytest.fixture(autouse=True)
def native_environment(monkeypatch):
    for name in POOL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("REFLEXIO_POSTGRES_READ_DB_URL", raising=False)
    monkeypatch.delenv("REFLEXIO_POSTGRES_SCHEMA", raising=False)
    monkeypatch.setenv("DATA_DB_URL", "postgresql://selected.invalid/new")


def stored(tmp_path, *, current=True, backend="postgres", policy=None):
    path = tmp_path / "configs" / f"config_{ORG}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = (
        {
            "type": "postgres",
            "db_url": "postgresql://former.invalid/old",
            **(POLICY if policy is None else policy),
        }
        if backend == "postgres"
        else {"type": "sqlite", "db_path": "former.sqlite"}
    )
    path.write_text(
        json.dumps(
            {
                "storage_config": storage,
                "opaque": {"keep": True},
                "_reflexio_config_file": {
                    "version": FILE_FORMAT_VERSION
                    if current
                    else FILE_FORMAT_VERSION - 1
                },
            }
        )
    )
    return path


def select(tmp_path):
    save_storage_to_config("postgres", org_id=ORG, base_dir=str(tmp_path))
    path = tmp_path / "configs" / f"config_{ORG}.json"
    return path, LocalConfigFile(path).read_document()


def test_current_selection_preserves_all_canonical_budgets(tmp_path):
    path = stored(tmp_path)
    _, document = select(tmp_path)
    assert {name: document["storage_config"][name] for name in POLICY} == POLICY
    assert document["storage_config"]["db_url"] == "postgresql://selected.invalid/new"
    assert document["opaque"] == {"keep": True}
    assert not (path.parent / ".config-migrations").exists()


@pytest.mark.parametrize("backend", ["postgres", "sqlite"])
def test_legacy_explicit_selection_adopts_actual_old_env_source_once(
    tmp_path, monkeypatch, backend
):
    path = stored(tmp_path, current=False, backend=backend)
    original = path.read_bytes()
    monkeypatch.setenv("REFLEXIO_POSTGRES_POOL_SIZE", "5")
    _, document = select(tmp_path)
    assert document["storage_config"]["pool_size"] == 5
    digest = hashlib.sha256(original).hexdigest()
    assert (
        path.parent / ".config-migrations" / path.name / f"{digest}.json"
    ).read_bytes() == original
    monkeypatch.delenv("REFLEXIO_POSTGRES_POOL_SIZE")
    assert select(tmp_path)[1]["storage_config"]["pool_size"] == 5


def test_legacy_implicit_preparation_does_not_import_env_policy(tmp_path, monkeypatch):
    path = stored(tmp_path, current=False)
    monkeypatch.setenv("REFLEXIO_POSTGRES_POOL_SIZE", "5")
    assert prepare_local_config(ORG, base_dir=str(tmp_path))
    document = LocalConfigFile(path).read_document()
    assert {name: document["storage_config"][name] for name in POLICY} == POLICY


def test_matching_obsolete_assertions_do_not_supply_defaults(tmp_path, monkeypatch):
    stored(
        tmp_path,
        policy={**POLICY, "read_pool_size": None, "read_pool_acquire_timeout": None},
    )
    monkeypatch.setenv("REFLEXIO_POSTGRES_READ_POOL_SIZE", "3")
    monkeypatch.setenv("REFLEXIO_POSTGRES_READ_POOL_ACQUIRE_TIMEOUT", "7.5")
    _, document = select(tmp_path)
    assert document["storage_config"]["pool_size"] == 3
    assert document["storage_config"]["pool_acquire_timeout"] == 7.5
    assert document["storage_config"]["read_pool_size"] is None
    assert document["storage_config"]["read_pool_acquire_timeout"] is None


@pytest.mark.parametrize("value", ["99", "invalid", "0"])
def test_current_obsolete_conflict_refuses_without_publication(
    tmp_path, monkeypatch, value
):
    path = stored(tmp_path)
    original = path.read_bytes()
    monkeypatch.setenv("REFLEXIO_POSTGRES_POOL_SIZE", value)
    with pytest.raises(ValueError):
        select(tmp_path)
    assert path.read_bytes() == original
    assert not (path.parent / ".config-migrations").exists()


def test_fresh_nondefault_old_export_cannot_select_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("REFLEXIO_POSTGRES_POOL_SIZE", "3")
    with pytest.raises(ValueError):
        select(tmp_path)
    assert not list(tmp_path.rglob("*.json"))


def test_fresh_selection_publishes_only_native_current_document(tmp_path, monkeypatch):
    import reflexio.config_file as files

    published = []
    publish = files._publish_bytes

    def observe(target, raw, **kwargs):
        if target.name == f"config_{ORG}.json":
            published.append(json.loads(raw))
        return publish(target, raw, **kwargs)

    monkeypatch.setattr(files, "_publish_bytes", observe)
    path, document = select(tmp_path)
    assert document["storage_config"]["type"] == "postgres"
    assert len(published) == 1
    assert published[0]["storage_config"]["type"] == "postgres"
    assert published[0]["_reflexio_config_file"]["version"] == FILE_FORMAT_VERSION
    assert not (path.parent / ".config-migrations").exists()


def test_missing_dsn_retains_source_bytes_and_does_not_import_old_policy(
    tmp_path, monkeypatch
):
    path = stored(tmp_path, current=False)
    original = path.read_bytes()
    monkeypatch.delenv("DATA_DB_URL")
    monkeypatch.setenv("REFLEXIO_POSTGRES_POOL_SIZE", "5")
    save_storage_to_config("postgres", org_id=ORG, base_dir=str(tmp_path))
    assert path.read_bytes() == original


@pytest.mark.parametrize("only", ["backend", "docs"])
@pytest.mark.parametrize("port_busy", [False, True])
def test_launcher_selects_old_policy_before_prepare_even_without_backend(
    tmp_path, monkeypatch, only, port_busy
):
    import reflexio.cli.env_loader as env_loader
    import reflexio.cli.run_services as launcher
    import reflexio.cli.utils as cli_utils

    base = tmp_path / ".reflexio"
    path = stored(base, current=False)
    original = path.read_bytes()
    monkeypatch.setenv("REFLEXIO_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("REFLEXIO_DEFAULT_ORG_ID", ORG)
    monkeypatch.setenv("REFLEXIO_POSTGRES_POOL_SIZE", "5")
    monkeypatch.setattr(launcher, "load_reflexio_env", lambda: None)
    monkeypatch.setattr(launcher, "should_start_local_embedding_service", lambda: False)
    monkeypatch.setattr(launcher, "DOCS_DIR", tmp_path / "absent-docs")
    monkeypatch.setattr(launcher, "run_services", lambda *_args: None)
    monkeypatch.setattr(env_loader, "get_env_path", lambda: tmp_path / "absent.env")
    checked = []

    def check_ports(ports, **_kwargs):
        checked.append(ports)
        if port_busy:
            raise RuntimeError("predecessor still owns the backend port")

    monkeypatch.setattr(cli_utils, "ensure_requested_ports_available", check_ports)
    args = launcher._build_run_services_parser().parse_args(["--only", only])

    if port_busy:
        with pytest.raises(RuntimeError, match="predecessor"):
            launcher.execute(args, storage="postgres")
        assert path.read_bytes() == original
        assert not (path.parent / ".config-migrations").exists()
        return
    launcher.execute(args, storage="postgres")

    assert len(checked) == 1 and "backend" in checked[0]
    assert LocalConfigFile(path).read_document()["storage_config"]["pool_size"] == 5
    digest = hashlib.sha256(original).hexdigest()
    assert (
        path.parent / ".config-migrations" / path.name / f"{digest}.json"
    ).read_bytes() == original


def test_previous_native_policy_reader_accepts_original_but_refuses_adopted_file(
    tmp_path,
):
    import re
    import runpy
    from pathlib import Path

    from reflexio.config_document import decode_config_document
    from reflexio.config_env import StoredConfigurationError

    path = stored(tmp_path, current=False)
    original = path.read_bytes()
    previous = runpy.run_path(
        str(Path(__file__).parents[1] / "fixtures/config_file_v9_reader.txt"),
        init_globals={
            "FILE_FORMAT_VERSION": 9,
            "_METADATA_KEY": "_reflexio_config_file",
            "decode_config_document": decode_config_document,
            "StoredConfigurationError": StoredConfigurationError,
            "re": re,
        },
    )["_decode"]
    assert previous(original)[1]["version"] == 9
    select(tmp_path)
    current = path.read_bytes()
    with pytest.raises(StoredConfigurationError, match="generation"):
        previous(current)
    assert path.read_bytes() == current


def test_legacy_unset_native_budgets_retain_generation_nine_defaults(tmp_path):
    stored(tmp_path, current=False)
    document = select(tmp_path)[1]
    # These are the frozen predecessor's defaults, not the current model defaults.
    assert {name: document["storage_config"][name] for name in POLICY} == {
        "pool_size": 10,
        "pool_acquire_timeout": 30.0,
        "read_pool_size": None,
        "read_pool_acquire_timeout": None,
    }
