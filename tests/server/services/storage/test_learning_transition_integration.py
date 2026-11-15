"""Dataset-authority semantics on actual isolated SQLite, without providers."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest

from reflexio.server.services.extraction.profile_authoring_protocol import (
    ESTIMATED_LIFETIME_POLICY,
    PROFILE_AUTHORING_POLICY,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage import (
    _learning_policy_authority as creation,
)
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningPolicyAuthority,
    LearningTransitionBlockedError,
    LearningTransitionConflictError,
    LearningTransitionIntegrityError,
    LearningTransitionState,
    _change_learning_policy_authority_db,
    _record_learning_transition_closure_db,
    learning_transition_key,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def storage(tmp_path):
    store = SQLiteStorage(org_id="contract_test", db_path=str(tmp_path / "state.db"))
    yield store
    store.conn.close()


@pytest.fixture(autouse=True)
def _deny_provider_dispatch(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("Transition storage cannot dispatch a provider")

    for target in ("completion", "acompletion", "embedding", "aembedding"):
        monkeypatch.setattr(f"litellm.{target}", deny)


def transition(storage, action, *, expected=None, target=None):
    # Test owner directly exercises the private lifecycle seam. Production arm
    # and activate require Enterprise's complete roster/closure derivation.
    with storage._learning_policy_authority_scope() as (db, root):
        selected = expected or root
        return _change_learning_policy_authority_db(
            db,
            action=action,
            operation_id="op",
            expected_nonce=selected.authority_nonce,
            expected_generation=selected.generation,
            target_policy=target,
        )


def grant(storage, root, project, selection):
    return storage.grant_learning_transition(
        "op",
        expected_authority_nonce=root.authority_nonce,
        expected_authority_generation=root.generation,
        expected_nonce=project.nonce if project else None,
        expected_revision=project.revision if project else None,
        selection_sha256=selection,
    )


def test_new_file_first_use_has_authoritative_open_root_without_fixture_bootstrap(
    storage,
):
    root = storage.read_learning_policy_authority()
    assert root.phase == "open" and root.active_policy == PROFILE_AUTHORING_POLICY
    assert root.initialization_kind == "first_dataset" and not root.bootstrap
    assert storage.read_learning_transition().project is None
    with storage.commit_scope():
        admitted = storage.fence_learning_transition()
    assert (
        admitted.authority == root
        and admitted.project.authority_nonce == root.authority_nonce
    )
    assert set(admitted.project.to_dict()) == set(
        LearningTransitionState.__dataclass_fields__
    )
    assert (
        "policy_floor" not in admitted.project.to_dict()
        and "mode" not in admitted.project.to_dict()
    )


def test_existing_empty_file_does_not_infer_new_policy(tmp_path):
    path = tmp_path / "old.db"
    sqlite3.connect(path).close()
    store = SQLiteStorage(org_id="contract_test", db_path=str(path))
    try:
        assert store.read_learning_policy_authority() is None
        with (
            pytest.raises(LearningTransitionBlockedError, match="qualified upgrade"),
            store.commit_scope(),
        ):
            store.fence_learning_transition()
        with pytest.raises(LearningTransitionBlockedError):
            store.provision_learning_scope()
    finally:
        store.conn.close()


def test_scope_inherits_new_root_generation_and_old_source_cannot_relabel_it(storage):
    with storage.commit_scope():
        first = storage.fence_learning_transition()
    sealed = transition(storage, "seal", target=PROFILE_AUTHORING_POLICY)
    with pytest.raises(LearningTransitionBlockedError), storage.commit_scope():
        storage.fence_learning_transition()
    transition(storage, "arm")
    active = transition(storage, "activate")
    with storage.commit_scope():
        inherited = storage.fence_learning_transition()
    assert inherited.project.nonce == first.project.nonce
    assert (
        inherited.project.authority_generation == active.generation > sealed.generation
    )
    assert inherited.project.revision > first.project.revision
    with (
        patch(
            "reflexio.server.services.storage.storage_base.learning_transition.PROFILE_AUTHORING_POLICY",
            ESTIMATED_LIFETIME_POLICY,
        ),
        pytest.raises(LearningTransitionBlockedError),
        storage.commit_scope(),
    ):
        storage.fence_learning_transition()
    assert storage.read_learning_policy_authority() == active


def test_sealed_grant_is_exact_scoped_and_revocation_changes_witness(storage):
    initial = storage.provision_learning_scope()
    sealed = transition(storage, "seal", target=ESTIMATED_LIFETIME_POLICY)
    selected = grant(storage, sealed, initial.project, "a" * 64)
    with pytest.raises(LearningTransitionBlockedError), storage.commit_scope():
        storage.fence_learning_transition(selection_sha256="b" * 64)
    with storage.commit_scope():
        assert storage.fence_learning_transition(selection_sha256="a" * 64) == selected
    revoked = grant(storage, sealed, selected.project, None)
    assert revoked.project.revision == selected.project.revision + 1
    with pytest.raises(LearningTransitionBlockedError), storage.commit_scope():
        storage.fence_learning_transition(selection_sha256="a" * 64)
    with pytest.raises(LearningTransitionConflictError):
        grant(storage, sealed, selected.project, "a" * 64)


def test_armed_prevents_learning_grant_and_abort_until_actual_target_activation(
    storage,
):
    root = transition(storage, "seal", target=ESTIMATED_LIFETIME_POLICY)
    armed = transition(storage, "arm")
    assert armed.phase == "armed"
    for action in ("abort", "activate"):
        with pytest.raises(LearningTransitionConflictError):
            transition(storage, action)
    with pytest.raises(LearningTransitionConflictError):
        grant(storage, armed, None, "a" * 64)
    with pytest.raises(LearningTransitionBlockedError), storage.commit_scope():
        storage.fence_learning_transition()
    with patch(
        "reflexio.server.services.storage.storage_base.learning_transition.PROFILE_AUTHORING_POLICY",
        ESTIMATED_LIFETIME_POLICY,
    ):
        active = transition(storage, "activate")
    assert (
        active.active_policy == ESTIMATED_LIFETIME_POLICY
        and active.generation > root.generation
    )


def test_sealed_abort_preserves_preexisting_floor_and_cas_blocks_stale_root(storage):
    initial = storage.read_learning_policy_authority()
    transition(storage, "seal", target=ESTIMATED_LIFETIME_POLICY)
    restored = transition(storage, "abort")
    assert restored.phase == "open" and restored.active_policy == initial.active_policy
    with pytest.raises(LearningTransitionConflictError):
        transition(storage, "seal", expected=initial, target=PROFILE_AUTHORING_POLICY)


def test_bootstrap_seal_cannot_abort_unknown_floor(storage):
    with storage._learning_policy_authority_scope() as (db, root):
        bootstrap = replace(
            root,
            phase="sealed",
            operation_id="op",
            target_policy=root.active_policy,
            bootstrap=True,
        )
        db.write_learning_policy_authority(
            json.dumps(bootstrap.to_dict()), missing=False
        )
    with pytest.raises(LearningTransitionConflictError):
        transition(storage, "abort")


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", True),
        ("generation", -1),
        ("authority_nonce", "bad"),
        ("phase", "unknown"),
        ("bootstrap", 1),
        ("active_policy", "unknown"),
    ],
)
def test_malformed_authority_never_becomes_open(storage, field, value):
    raw = storage.read_learning_policy_authority().to_dict()
    raw[field] = value
    with pytest.raises(LearningTransitionIntegrityError):
        LearningPolicyAuthority.from_dict(raw)


def test_wrong_project_root_binding_and_legacy_v2_are_not_reinterpreted(storage):
    scope = storage.provision_learning_scope().project
    key = learning_transition_key(storage.org_id, "")
    for raw in (
        {**scope.to_dict(), "authority_nonce": "e" * 32},
        {"schema_version": 2, "policy_floor": PROFILE_AUTHORING_POLICY},
    ):
        storage.conn.execute(
            "UPDATE _operation_state SET operation_state=? WHERE service_name=?",
            (json.dumps(raw), key),
        )
        storage.conn.commit()
        with pytest.raises(LearningTransitionIntegrityError), storage.commit_scope():
            storage.fence_learning_transition()


def test_root_and_project_writes_follow_caller_rollback(storage):
    before = storage.read_learning_policy_authority()
    with (
        pytest.raises(RuntimeError),
        storage._learning_policy_authority_scope() as (db, root),
    ):
        _change_learning_policy_authority_db(
            db,
            action="seal",
            operation_id="op",
            expected_nonce=root.authority_nonce,
            expected_generation=root.generation,
            target_policy=PROFILE_AUTHORING_POLICY,
        )
        raise RuntimeError("rollback")
    assert storage.read_learning_policy_authority() == before
    with pytest.raises(RuntimeError), storage.commit_scope():
        storage.fence_learning_transition()
        raise RuntimeError("rollback")
    assert storage.read_learning_transition().project is None


def test_reserved_project_cleanup_preserves_authority_and_witness(storage):
    scope = storage.provision_learning_scope()
    key = learning_transition_key(storage.org_id, "")
    for method, args in (
        (storage.create_operation_state, (key, {})),
        (storage.upsert_operation_state, (key, {})),
        (storage.update_operation_state, (key, {})),
        (storage.delete_operation_state, (key,)),
    ):
        with pytest.raises(
            StorageError, match="Reserved learning transition namespace"
        ):
            method(*args)
    storage.delete_all_operation_states()
    assert storage.read_learning_transition() == scope


def test_retained_creation_recovers_after_migration_failure(tmp_path):
    path = str(tmp_path / "retry.db")
    with (
        patch.object(
            SQLiteStorage, "migrate", side_effect=RuntimeError("migration failure")
        ),
        pytest.raises(RuntimeError),
    ):
        SQLiteStorage(org_id="contract_test", db_path=path)
    conn = sqlite3.connect(path)
    first = json.loads(
        conn.execute("SELECT authority FROM _learning_policy_authority").fetchone()[0]
    )
    conn.close()
    assert first["phase"] == "sealed" and first["bootstrap"]
    store = SQLiteStorage(org_id="contract_test", db_path=path)
    try:
        root = store.read_learning_policy_authority()
        assert root is not None
        assert root.phase == "open" and root.authority_nonce == first["authority_nonce"]
        assert root.initialization_nonce == first["initialization_nonce"]
    finally:
        store.conn.close()


def test_failure_before_publication_cleans_only_private_artifacts(tmp_path):
    path = str(tmp_path / "unknown.db")
    with (
        patch.object(
            creation, "_retain_creation", side_effect=RuntimeError("proof failure")
        ),
        pytest.raises(RuntimeError),
    ):
        SQLiteStorage(org_id="contract_test", db_path=path)
    assert not Path(path).exists()
    assert list(tmp_path.iterdir()) == []
    store = SQLiteStorage(org_id="contract_test", db_path=path)
    try:
        root = store.read_learning_policy_authority()
        assert root is not None and root.phase == "open"
    finally:
        store.conn.close()


def test_concurrent_first_creation_converges_on_one_retained_authority(tmp_path):
    path = str(tmp_path / "concurrent.db")

    def open_and_close(_index):
        store = SQLiteStorage(org_id="contract_test", db_path=path)
        try:
            with store.commit_scope():
                return store.fence_learning_transition().authority
        finally:
            store.conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        roots = list(pool.map(open_and_close, range(2)))
    assert roots[0] == roots[1] and roots[0].phase == "open"


def test_readonly_armed_closure_uses_independent_snapshot(storage):
    transition(storage, "seal", target=PROFILE_AUTHORING_POLICY)
    transition(storage, "arm")
    observed, closure = storage.read_learning_transition_closure()
    assert observed.authority.phase == "armed" and closure.ready
    with storage.commit_scope(), pytest.raises(ValueError, match="independent"):
        storage.read_learning_transition_closure()


def test_closure_is_derived_and_bound_to_current_root_generation(storage):
    storage.provision_learning_scope()
    transition(storage, "seal", target=PROFILE_AUTHORING_POLICY)
    armed = transition(storage, "arm")
    with pytest.raises(LearningTransitionBlockedError, match="stale"):
        storage.read_learning_transition_closure()
    with storage._learning_policy_authority_scope() as (db, root):
        witness, closure = _record_learning_transition_closure_db(
            db, authority=root, org_id=storage.org_id, project_id=""
        )
    assert closure.ready and witness.closure_sha256 is not None
    assert len(witness.closure_sha256) == 64
    assert witness.authority_generation == armed.generation
    observed, _ = storage.read_learning_transition_closure()
    assert observed.project == witness


def test_writer_admission_serializes_before_root_seal(storage):
    other = SQLiteStorage(org_id=storage.org_id, db_path=storage.db_path)
    started, finished = Event(), Event()

    def seal_other():
        started.set()
        try:
            return transition(other, "seal", target=PROFILE_AUTHORING_POLICY)
        finally:
            finished.set()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with storage.commit_scope():
                admitted = storage.fence_learning_transition()
                future = pool.submit(seal_other)
                assert started.wait(2)
                assert not finished.wait(0.1)
                assert admitted.authority.phase == "open"
            assert future.result(timeout=5).phase == "sealed"
        with pytest.raises(LearningTransitionBlockedError), storage.commit_scope():
            storage.fence_learning_transition()
    finally:
        other.conn.close()


def test_private_memory_connection_has_actual_new_dataset_provenance():
    store = SQLiteStorage(org_id="contract_test", db_path=":memory:")
    try:
        root = store.read_learning_policy_authority()
        assert root is not None and root.phase == "open"
        with store.commit_scope():
            assert store.fence_learning_transition().project is not None
    finally:
        store.conn.close()


def test_separate_process_first_openers_share_one_actual_file_authority(tmp_path):
    import multiprocessing

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("This isolated process fixture requires forked provider guards")
    context = multiprocessing.get_context("fork")
    gate, result = context.Event(), context.Queue()
    path = str(tmp_path / "processes.db")

    def open_owned_file():
        gate.wait(5)
        try:
            store = SQLiteStorage(org_id="contract_test", db_path=path)
            try:
                with store.commit_scope():
                    authority = store.fence_learning_transition().authority
                result.put(("ok", authority.to_dict()))
            finally:
                store.conn.close()
        except Exception as error:
            result.put(("error", repr(error)))

    children = [context.Process(target=open_owned_file) for _ in range(2)]
    try:
        for child in children:
            child.start()
        gate.set()
        outputs = [result.get(timeout=20) for _ in children]
        for child in children:
            child.join(timeout=5)
        assert all(child.exitcode == 0 for child in children)
        assert all(status == "ok" for status, _ in outputs), outputs
        assert outputs[0][1] == outputs[1][1]
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
        result.close()


def test_default_dataset_resolver_retains_actual_creation_before_identity_claim(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("reflexio.server.LOCAL_STORAGE_PATH", str(tmp_path))
    store = SQLiteStorage(org_id="contract_test")
    try:
        root = store.read_learning_policy_authority()
        assert root is not None
        assert root.phase == "open" and root.initialization_kind == "first_dataset"
        with store.commit_scope():
            assert store.fence_learning_transition().authority == root
    finally:
        store.conn.close()


def test_stale_source_cannot_provision_open_new_policy_scope(storage):
    with (
        patch(
            "reflexio.server.services.storage.storage_base.learning_transition.PROFILE_AUTHORING_POLICY",
            ESTIMATED_LIFETIME_POLICY,
        ),
        pytest.raises(LearningTransitionBlockedError),
    ):
        storage.provision_learning_scope()
    assert storage.read_learning_transition().project is None


@pytest.mark.parametrize("phase", ["sealed", "armed"])
def test_new_scope_during_transition_inherits_root_without_opening_learning(
    storage, phase
):
    transition(storage, "seal", target=PROFILE_AUTHORING_POLICY)
    if phase == "armed":
        transition(storage, "arm")
    scoped = storage.provision_learning_scope()
    assert scoped.authority.phase == phase and scoped.project.selection_sha256 is None
    assert scoped.project.authority_generation == scoped.authority.generation
    with pytest.raises(LearningTransitionBlockedError), storage.commit_scope():
        storage.fence_learning_transition()


def test_future_or_misbound_project_state_is_not_silently_repaired(storage):
    initial = storage.provision_learning_scope().project
    key = learning_transition_key(storage.org_id, "")
    for raw in (
        {**initial.to_dict(), "authority_generation": initial.authority_generation + 1},
        {**initial.to_dict(), "operation_id": "other-operation"},
        {**initial.to_dict(), "authority_generation": True},
    ):
        storage.conn.execute(
            "UPDATE _operation_state SET operation_state=? WHERE service_name=?",
            (json.dumps(raw), key),
        )
        storage.conn.commit()
        with pytest.raises(LearningTransitionIntegrityError), storage.commit_scope():
            storage.fence_learning_transition()


def test_withheld_creator_never_exposes_a_root_missing_dataset_to_another_process(
    tmp_path,
):
    import multiprocessing

    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("This isolated process fixture requires forked provider guards")
    context = multiprocessing.get_context("fork")
    withheld, release = context.Event(), context.Event()
    first_result, second_result = context.Queue(), context.Queue()
    path = str(tmp_path / "withheld.db")

    def open_file(output, pause):
        if pause:
            original = creation._retain_creation

            def retained_creation(conn, selected_path, org_id):
                withheld.set()
                if not release.wait(20):
                    raise RuntimeError("Test creator was not released")
                return original(conn, selected_path, org_id)

            creation._retain_creation = retained_creation
        try:
            store = SQLiteStorage(org_id="contract_test", db_path=path)
            try:
                with store.commit_scope():
                    authority = store.fence_learning_transition().authority
                output.put(("ok", authority.to_dict()))
            finally:
                store.conn.close()
        except Exception as error:
            output.put(("error", repr(error)))

    first = context.Process(target=open_file, args=(first_result, True))
    second = context.Process(target=open_file, args=(second_result, False))
    children = [first, second]
    try:
        first.start()
        assert withheld.wait(10)
        second.start()
        second_output = second_result.get(timeout=15)
        release.set()
        first_output = first_result.get(timeout=15)
        for child in children:
            child.join(timeout=5)
        assert all(child.exitcode == 0 for child in children)
        assert first_output[0] == second_output[0] == "ok", (
            first_output,
            second_output,
        )
        assert first_output[1] == second_output[1]
    finally:
        release.set()
        for child in children:
            if child.pid is not None and child.is_alive():
                child.terminate()
                child.join(timeout=5)
        first_result.close()
        second_result.close()


def test_atomic_creation_loser_preserves_existing_actual_target(tmp_path, monkeypatch):
    path = tmp_path / "existing.db"
    original_link = creation.os.link
    winner_bytes = b"synthetic concurrent existing target"

    def existing_target_wins(source, target):
        Path(target).write_bytes(winner_bytes)
        return original_link(source, target)

    monkeypatch.setattr(creation.os, "link", existing_target_wins)
    creation.reserve_new_sqlite_dataset(str(path), "contract_test")
    assert path.read_bytes() == winner_bytes
    assert list(tmp_path.iterdir()) == [path]


def test_published_creation_is_self_contained_without_private_sidecars(tmp_path):
    path = tmp_path / "published.db"
    creation.reserve_new_sqlite_dataset(str(path), "contract_test")
    assert list(tmp_path.iterdir()) == [path]
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        root = LearningPolicyAuthority.from_dict(
            json.loads(
                conn.execute(
                    "SELECT authority FROM _learning_policy_authority"
                ).fetchone()[0]
            )
        )
        assert root.phase == "sealed" and root.bootstrap
        assert root.provenance_sha256 == creation._provenance(
            str(path), "contract_test", root.initialization_nonce
        )
    finally:
        conn.close()
