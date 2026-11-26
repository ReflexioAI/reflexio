"""Real predecessor files qualify without constructor migrations or data loss."""

import gzip
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from reflexio.server.services.storage.sqlite_storage import (
    _learning_policy_authority as owner,
)
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionBlockedError,
)

FIXTURES = Path(__file__).resolve().parents[3] / "test_data/sqlite_70daf32c"
CONTRACT = json.loads(
    Path(owner.__file__).with_name("_historical_schema_70daf32c.json").read_text()
)


def historical_settings(tmp_path, database):
    from reflexio.server.services.configurator.local_file_config_storage import (
        LocalFileConfigStorage,
    )

    settings = LocalFileConfigStorage("historical_http", base_dir=str(tmp_path))
    path = Path(settings.config_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(
        {
            "_reflexio_config_file": {"version": 14},
            "storage_config": {"type": "sqlite", "db_path": str(database)},
            "window_size": 23,
            "stride_size": 7,
            "opaque": {"preserve": "original"},
        }
    ).encode()
    path.write_bytes(original)
    return settings, path, original


@pytest.mark.parametrize(
    "cohort",
    [
        "empty",
        "completed",
        "eligible_queued",
        "filtered_queued",
        "filtered_drained",
        "billing_pending",
        "window_pending",
        "unclaimed_empty",
        "unclaimed_completed",
    ],
)
def test_settings_upgrade_cannot_publish_before_retained_dataset_qualification(
    tmp_path, cohort
):
    database = retained(tmp_path, cohort)
    settings, path, original = historical_settings(tmp_path, database)
    before = database.read_bytes()
    with pytest.raises(LearningTransitionBlockedError, match="upgrade-local"):
        settings.prepare_for_startup()
    assert path.read_bytes() == original
    assert database.read_bytes() == before
    assert not (path.parent / ".config-migrations").exists()


@pytest.mark.parametrize("cohort", ["completed", "eligible_queued"])
def test_explicit_dataset_upgrade_precedes_real_settings_publication(tmp_path, cohort):
    from reflexio.config_file import FILE_FORMAT_VERSION

    database = retained(tmp_path, cohort)
    settings, path, original = historical_settings(tmp_path, database)
    result = owner.upgrade_existing_sqlite_dataset(database)
    before = database.read_bytes()
    assert settings.prepare_for_startup()
    current = json.loads(path.read_bytes())
    assert current["_reflexio_config_file"]["version"] == FILE_FORMAT_VERSION
    assert (current["profile_window_size"], current["profile_stride_size"]) == (23, 7)
    assert current["storage_config"]["db_path"] == str(database)
    assert current["opaque"] == {"preserve": "original"}
    originals = list((path.parent / ".config-migrations" / path.name).glob("*.json"))
    assert len(originals) == 1 and originals[0].read_bytes() == original
    prepared = path.read_bytes()
    assert not settings.prepare_for_startup()
    assert path.read_bytes() == prepared
    assert database.read_bytes() == before
    assert owner.upgrade_existing_sqlite_dataset(database).authority == result.authority


@pytest.mark.parametrize("state", ["sealed", "armed", "different_policy", "malformed"])
def test_settings_publication_cannot_bypass_a_held_or_invalid_root(tmp_path, state):
    from dataclasses import replace

    from reflexio.profile_authoring_protocol import (
        AUTHORING_POLICIES,
        PROFILE_AUTHORING_POLICY,
    )

    database = retained(tmp_path)
    root = owner.upgrade_existing_sqlite_dataset(database).authority
    if state in {"sealed", "armed"}:
        payload = replace(
            root,
            phase=state,
            operation_id="owned-cutover-operation",
            target_policy=PROFILE_AUTHORING_POLICY,
        ).to_dict()
    elif state == "different_policy":
        payload = replace(
            root,
            active_policy=next(
                p for p in AUTHORING_POLICIES if p != PROFILE_AUTHORING_POLICY
            ),
        ).to_dict()
    else:
        payload = {}
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "UPDATE _learning_policy_authority SET authority=? WHERE singleton=1",
            (json.dumps(payload),),
        )
    settings, path, original = historical_settings(tmp_path, database)
    before = database.read_bytes()
    with pytest.raises((LearningTransitionBlockedError, ValueError)):
        settings.prepare_for_startup()
    assert path.read_bytes() == original
    assert database.read_bytes() == before
    assert not (path.parent / ".config-migrations").exists()


@pytest.mark.parametrize("unqualified", ["source", "destination"])
def test_storage_retarget_checks_both_existing_datasets(tmp_path, unqualified):
    from reflexio.config_file import FILE_FORMAT_VERSION
    from reflexio.models.config_schema import Config, StorageConfigSQLite

    source = retained(tmp_path)
    destination_dir = tmp_path / "destination"
    destination_dir.mkdir()
    destination = retained(destination_dir)
    owner.upgrade_existing_sqlite_dataset(
        destination if unqualified == "source" else source
    )
    settings, path, _ = historical_settings(tmp_path, source)
    # A strict current file can still refer to a previously held dataset.
    # Retargeting must check it as well as the newly selected destination.
    document = Config(
        storage_config=StorageConfigSQLite(db_path=str(source))
    ).model_dump(mode="json")
    document["_reflexio_config_file"] = {"version": FILE_FORMAT_VERSION}
    original = json.dumps(document).encode()
    path.write_bytes(original)
    before = {p: p.read_bytes() for p in (source, destination)}
    with pytest.raises(LearningTransitionBlockedError, match="upgrade-local"):
        settings.save_config(
            Config(storage_config=StorageConfigSQLite(db_path=str(destination)))
        )
    assert path.read_bytes() == original
    assert all(p.read_bytes() == raw for p, raw in before.items())


def retained(tmp_path, cohort="completed"):
    target = tmp_path / "retained.sqlite"
    raw = gzip.decompress((FIXTURES / f"{cohort}.sqlite.gz").read_bytes())
    target.write_bytes(raw)
    return target


def business_snapshot(path):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        result = {}
        for name in (
            CONTRACT["ownership"]["dataset_local"]
            + CONTRACT["ownership"]["direct_scope"]
            + ["_dataset_identity"]
        ):
            fields = ",".join(
                '"' + x.replace('"', '""') + '"'
                for x in CONTRACT["tables"][name]["columns"]
            )
            rows = connection.execute(
                f'SELECT {fields} FROM "{name}"'  # noqa: S608 -- Frozen synthetic fixture contract names/columns.
            ).fetchall()
            result[name] = sorted(map(repr, rows))
        return result
    finally:
        connection.close()


@pytest.mark.parametrize(
    "cohort",
    ["empty", "completed", "eligible_queued", "filtered_queued", "filtered_drained"],
)
def test_real_predecessor_history_is_conserved_and_retry_retains_origin(
    tmp_path, monkeypatch, cohort
):
    from reflexio.cli import env_loader
    from reflexio.server.services.storage.sqlite_storage._base import SQLiteStorageBase

    def forbidden(*_args, **_kwargs):
        pytest.fail(
            "Maintenance invoked operator/bootstrap or normal storage construction"
        )

    monkeypatch.setattr(env_loader, "load_existing_reflexio_env", forbidden)
    monkeypatch.setattr(SQLiteStorageBase, "__init__", forbidden)
    path = retained(tmp_path, cohort)
    before = business_snapshot(path)
    result = owner.upgrade_existing_sqlite_dataset(path)
    assert result.established and result.provenance is not None
    assert result.authority.phase == "open"
    assert result.authority.initialization_kind == "qualified_upgrade"
    assert result.authority.generation == 2
    digest = hashlib.sha256(
        json.dumps(result.provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == result.authority.provenance_sha256
    assert business_snapshot(path) == before
    retry = owner.upgrade_existing_sqlite_dataset(path)
    assert not retry.established and retry.provenance is None
    assert retry.authority == result.authority
    assert business_snapshot(path) == before
    if cohort == "eligible_queued":
        assert result.provenance["closure_counts"]["queued_admissions"] > 0


@pytest.mark.parametrize(
    "cohort",
    ["billing_pending", "window_pending", "unclaimed_empty", "unclaimed_completed"],
)
def test_unqualified_original_cohorts_remain_held(tmp_path, cohort):
    path = retained(tmp_path, cohort)
    before = path.read_bytes()
    with pytest.raises(LearningTransitionBlockedError):
        owner.upgrade_existing_sqlite_dataset(path)
    assert path.read_bytes() == before


def test_missing_or_relative_target_is_never_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(OSError):
        owner.upgrade_existing_sqlite_dataset(path)
    assert not path.exists()
    with pytest.raises(LearningTransitionBlockedError, match="absolute"):
        owner.upgrade_existing_sqlite_dataset(Path("relative.sqlite"))


def test_unexpected_authority_trigger_cannot_mutate_business_data(tmp_path):
    path = retained(tmp_path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(owner.AUTHORITY_DDL)
        connection.execute(
            "CREATE TRIGGER unsafe_authority AFTER INSERT ON _learning_policy_authority BEGIN DELETE FROM profiles; END"
        )
    before = business_snapshot(path)
    with pytest.raises(LearningTransitionBlockedError, match="schema"):
        owner.upgrade_existing_sqlite_dataset(path)
    assert business_snapshot(path) == before


def test_failure_after_arm_rolls_back_control_ddl_and_business_state(
    tmp_path, monkeypatch
):
    path = retained(tmp_path)
    before = business_snapshot(path)
    change = owner._change_learning_policy_authority_db

    def fail_after_arm(*args, **kwargs):
        result = change(*args, **kwargs)
        if kwargs["action"] == "arm":
            assert result.phase == "armed"
            raise RuntimeError("owned arm failure")
        return result

    monkeypatch.setattr(owner, "_change_learning_policy_authority_db", fail_after_arm)
    with pytest.raises(RuntimeError, match="owned arm failure"):
        owner.upgrade_existing_sqlite_dataset(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        assert not connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='_learning_policy_authority'"
        ).fetchall()
    assert business_snapshot(path) == before


def test_same_writer_serializes_competing_explicit_upgrades(tmp_path):
    path = retained(tmp_path)
    before = business_snapshot(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(owner.upgrade_existing_sqlite_dataset, [path, path]))
    assert sorted(x.established for x in results) == [False, True]
    assert results[0].authority == results[1].authority
    assert business_snapshot(path) == before


def test_packaged_contract_is_bound_to_original_producer_evidence():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    assert CONTRACT["producer_sdk"] == provenance["old_sdk"]
    for name, item in provenance["cohorts"].items():
        source = FIXTURES / f"{name}.sqlite.gz"
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == item["fixture"]["gzip_sha256"]
        )
        assert (
            hashlib.sha256(gzip.decompress(source.read_bytes())).hexdigest()
            == item["fixture"]["raw_sha256"]
        )
    assert not set(CONTRACT["empty_controls"]) & set(CONTRACT["virtual_owners"])


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign_org",
        "project",
        "stalled",
        "independent_work",
        "extra_table",
        "missing_stall",
    ],
)
def test_unsupported_scope_or_independent_state_is_not_repaired(tmp_path, mutation):
    path = retained(tmp_path, "eligible_queued")
    changes = {
        "foreign_org": ("UPDATE learning_work SET org_id='foreign-owner'", ()),
        "project": (
            "UPDATE extraction_cursors SET project_id='unsupported-project'",
            (),
        ),
        "stalled": ("UPDATE stall_state SET stalled=1", ()),
        "independent_work": (
            "INSERT INTO _operation_state(service_name,operation_state) VALUES(?,?)",
            ("owned-pending-work", "{}"),
        ),
        "extra_table": ("CREATE TABLE unknown_owner(value TEXT)", ()),
        "missing_stall": ("DELETE FROM stall_state", ()),
    }
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(*changes[mutation])
    before = business_snapshot(path)
    with pytest.raises(LearningTransitionBlockedError):
        owner.upgrade_existing_sqlite_dataset(path)
    assert business_snapshot(path) == before


def test_lost_commit_acknowledgment_reconciles_retained_origin(tmp_path, monkeypatch):
    path = retained(tmp_path)
    before = business_snapshot(path)
    connect = sqlite3.connect

    class LostAcknowledgment(sqlite3.Connection):
        def commit(self):
            super().commit()
            raise OSError("owned lost commit acknowledgment")

    def uncertain_connection(database, *args, **kwargs):
        if database == path.as_uri() + "?mode=rw":
            kwargs["factory"] = LostAcknowledgment
        return connect(database, *args, **kwargs)

    with monkeypatch.context() as patching:
        patching.setattr(owner.sqlite3, "connect", uncertain_connection)
        with pytest.raises(OSError, match="owned lost commit"):
            owner.upgrade_existing_sqlite_dataset(path)
    retry = owner.upgrade_existing_sqlite_dataset(path)
    assert not retry.established and retry.provenance is None
    assert retry.authority.phase == "open" and retry.authority.generation == 2
    again = owner.upgrade_existing_sqlite_dataset(path)
    assert retry.authority == again.authority
    assert business_snapshot(path) == before


def test_observed_file_replacement_refuses_without_granting_either_file(
    tmp_path, monkeypatch
):
    path = retained(tmp_path)
    displaced = tmp_path / "displaced.sqlite"
    connect = sqlite3.connect

    def replaced_connection(database, *args, **kwargs):
        if database == path.as_uri() + "?mode=rw":
            # Real replacement between selection and open, without relying on
            # platform-specific permission to rename a live SQLite handle.
            path.rename(displaced)
            path.write_bytes(
                gzip.decompress((FIXTURES / "empty.sqlite.gz").read_bytes())
            )
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(owner.sqlite3, "connect", replaced_connection)
    with pytest.raises(LearningTransitionBlockedError, match="file changed"):
        owner.upgrade_existing_sqlite_dataset(path)
    for selected in [path, displaced]:
        with closing(connect(selected)) as connection, connection:
            assert not connection.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='_learning_policy_authority'"
            ).fetchall()


def test_preserved_old_queued_intake_enters_normal_current_admission(
    tmp_path, monkeypatch
):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    def no_provider(*_args, **_kwargs):
        pytest.fail("Queued-admission inspection dispatched a provider")

    for name in ("completion", "acompletion", "embedding", "aembedding"):
        monkeypatch.setattr(f"litellm.{name}", no_provider)
    path = retained(tmp_path, "eligible_queued")
    result = owner.upgrade_existing_sqlite_dataset(path)
    assert result.provenance is not None
    storage = SQLiteStorage(org_id=result.provenance["org_id"], db_path=str(path))
    try:
        claim = storage.claim_extraction("owned-upgrade-admission", 300)
        assert claim is not None
        user_id, token = claim
        window = storage.prepare_extraction(user_id, token)
        assert window is not None and window.kind == "profile"
        assert len(window.manifest) == 10
        admission = storage.read_learning_transition()
        assert admission is not None
        assert admission.authority == result.authority
        assert admission.project is not None
        assert admission.project.authority_nonce == result.authority.authority_nonce
        # A retained OPEN root stays authoritative after normal work resumes.
        retry = owner.upgrade_existing_sqlite_dataset(path)
        assert not retry.established and retry.authority == result.authority
    finally:
        storage.conn.close()


@pytest.mark.parametrize(
    "identity", ["private / invalid", b"private-invalid-blob", None]
)
def test_bad_retained_claim_is_refused_without_exposing_or_repairing_it(
    tmp_path, identity
):
    path = retained(tmp_path)
    with closing(sqlite3.connect(path)) as connection, connection:
        if identity is None:
            connection.execute("DELETE FROM _dataset_identity")
        else:
            connection.execute("UPDATE _dataset_identity SET org_id=?", (identity,))
    before = business_snapshot(path)
    with pytest.raises(LearningTransitionBlockedError, match="identity") as error:
        owner.upgrade_existing_sqlite_dataset(path)
    assert "private" not in str(error.value)
    assert business_snapshot(path) == before


@pytest.mark.parametrize("source", ["runs", "windows", "requests"])
def test_null_historical_execution_key_cannot_disappear_from_upgrade_closure(
    tmp_path, source
):
    path = retained(tmp_path, "empty")
    with closing(sqlite3.connect(path)) as connection, connection:
        org = connection.execute(
            "SELECT org_id FROM _dataset_identity WHERE k=1"
        ).fetchone()[0]
        if source == "runs":
            connection.execute(
                "INSERT INTO _agent_runs(id,org_id,extractor_kind,user_id,request_id,status,generation_request_snapshot,service_config_snapshot,agent_context_snapshot) VALUES(NULL,?,'profile','u','r','running','{}','{}','{}')",
                (org,),
            )
        elif source == "windows":
            connection.execute(
                "INSERT INTO extraction_windows(window_id,user_id,kind,org_id,predecessor,end_seq,manifest,policy) VALUES(NULL,'u','profile',?,0,1,'[]','{}')",
                (org,),
            )
        else:
            connection.execute(
                "INSERT INTO requests(request_id,user_id,created_at,session_id) VALUES(NULL,'u','2026-01-01','s')"
            )
    before = business_snapshot(path)
    with pytest.raises(LearningTransitionBlockedError, match="closure"):
        owner.upgrade_existing_sqlite_dataset(path)
    assert business_snapshot(path) == before
    with closing(sqlite3.connect(path)) as connection, connection:
        assert not connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='_learning_policy_authority'"
        ).fetchall()


def test_normal_existing_file_preparation_does_not_grant_then_explicit_upgrade_does(
    tmp_path,
):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    path = retained(tmp_path, "empty")
    with closing(sqlite3.connect(path)) as connection, connection:
        org = connection.execute(
            "SELECT org_id FROM _dataset_identity WHERE k=1"
        ).fetchone()[0]
    storage = SQLiteStorage(org_id=org, db_path=str(path))
    try:
        assert storage.read_learning_policy_authority() is None
        with pytest.raises(LearningTransitionBlockedError), storage.commit_scope():
            storage.fence_learning_transition()
    finally:
        storage.conn.close()
    before = business_snapshot(path)
    result = owner.upgrade_existing_sqlite_dataset(path)
    assert result.established and result.authority.phase == "open"
    assert business_snapshot(path) == before


@pytest.mark.parametrize("state", ["sealed", "different_policy", "malformed"])
def test_existing_incompatible_authority_is_never_repaired_or_reopened(tmp_path, state):
    from dataclasses import replace

    from reflexio.profile_authoring_protocol import (
        AUTHORING_POLICIES,
        PROFILE_AUTHORING_POLICY,
    )

    path = retained(tmp_path)
    result = owner.upgrade_existing_sqlite_dataset(path)
    root = result.authority
    if state == "sealed":
        payload = replace(
            root,
            phase="sealed",
            target_policy=PROFILE_AUTHORING_POLICY,
            operation_id="owned-held-operation",
        ).to_dict()
    elif state == "different_policy":
        payload = replace(
            root,
            active_policy=next(
                p for p in AUTHORING_POLICIES if p != PROFILE_AUTHORING_POLICY
            ),
        ).to_dict()
    else:
        payload = {}
    encoded = json.dumps(payload)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE _learning_policy_authority SET authority=? WHERE singleton=1",
            (encoded,),
        )
    before = business_snapshot(path)
    with pytest.raises((LearningTransitionBlockedError, ValueError)):
        owner.upgrade_existing_sqlite_dataset(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        assert (
            connection.execute(
                "SELECT authority FROM _learning_policy_authority WHERE singleton=1"
            ).fetchone()[0]
            == encoded
        )
    assert business_snapshot(path) == before
