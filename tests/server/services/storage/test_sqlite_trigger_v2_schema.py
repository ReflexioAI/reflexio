"""SQLite's v2 CHECK expansion preserves historical invocation bytes."""

import sqlite3

import pytest

from reflexio.server.services.storage.sqlite_storage._open_world import (
    _ATTEMPT_DDL,
    _upgrade_trigger_invocation_roles,
)


def test_invocation_role_upgrade_preserves_receipts_and_unique_slots() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE playbook_optimization_jobs(job_id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO playbook_optimization_jobs VALUES (1)")
    conn.commit()
    legacy = _ATTEMPT_DDL.replace(", 'trigger_gate', 'trigger_builder'", "")
    assert legacy != _ATTEMPT_DDL
    conn.executescript(legacy)
    values = (
        "a" * 64,
        1,
        "b" * 64,
        "fixture",
        "model",
        "c" * 64,
        "candidate",
        "d" * 64,
        123,
    )
    insert = (
        "INSERT INTO offline_tuner_open_world_invocations"
        " (analyst_input_identity,job_id,component_identity_digest,expected_provider,"
        " expected_model_revision,input_digest,component_role,qualification_result_digest,prepared_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)"
    )
    conn.execute(insert, values)
    conn.commit()
    original = dict(
        conn.execute("SELECT * FROM offline_tuner_open_world_invocations").fetchone()
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, ("e" * 64, *values[1:6], "trigger_gate", *values[7:]))
    conn.rollback()
    conn.set_authorizer(
        lambda action, name, *_: (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_CREATE_INDEX
            and name == "uq_open_world_invocations_manifest_slot"
            else sqlite3.SQLITE_OK
        )
    )
    with pytest.raises(sqlite3.DatabaseError):
        _upgrade_trigger_invocation_roles(conn)
    conn.set_authorizer(None)
    assert (
        "'trigger_gate'"
        not in conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='offline_tuner_open_world_invocations'"
        ).fetchone()[0]
    )
    assert (
        dict(
            conn.execute(
                "SELECT * FROM offline_tuner_open_world_invocations"
            ).fetchone()
        )
        == original
    )
    _upgrade_trigger_invocation_roles(conn)
    assert (
        dict(
            conn.execute(
                "SELECT * FROM offline_tuner_open_world_invocations"
            ).fetchone()
        )
        == original
    )
    conn.execute(insert, ("e" * 64, *values[1:6], "trigger_gate", *values[7:]))
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, ("f" * 64, *values[1:6], "trigger_gate", *values[7:]))
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, ("f" * 64, *values[1:6], "unqualified_role", *values[7:]))
    conn.rollback()
    _upgrade_trigger_invocation_roles(conn)
    assert (
        conn.execute(
            "SELECT count(*) FROM offline_tuner_open_world_invocations"
        ).fetchone()[0]
        == 2
    )
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_v2_authority_requires_distinct_roles_and_actual_boolean_checks() -> None:
    import copy
    import json
    from hashlib import sha256
    from types import SimpleNamespace

    from reflexio.server.services.offline_tuner.evidence.manifests import (
        canonical_json_bytes,
    )
    from reflexio.server.services.offline_tuner.open_world.rule_evidence_authority import (
        qualify_rule_evidence_method,
    )
    from reflexio.server.services.offline_tuner.open_world_records import (
        OpenWorldQualificationAuthorityStaleError,
    )
    from reflexio.server.services.storage.sqlite_storage._open_world_attempt import (
        SQLiteOpenWorldAttemptMixin,
    )
    from reflexio.server.services.storage.sqlite_storage._open_world_trigger import (
        trigger_authorization,
    )

    method = qualify_rule_evidence_method(
        SimpleNamespace(
            transport_identity=SimpleNamespace(
                provider="fixture", model_revision="fixture-model"
            )
        )
    )
    assert method.is_v2
    assert SQLiteOpenWorldAttemptMixin._validated_method_authority(
        method.canonical_json, method.digest
    ) == json.loads(method.canonical_json)
    for damage in ("role", "numeric_check"):
        forged = copy.deepcopy(json.loads(method.canonical_json))
        trigger = forged["trigger_maintenance"]
        if damage == "role":
            component = trigger["gate"]["component_identity"]
            component["component_role"] = "candidate"
            trigger["gate"]["authorization_digest"] = trigger_authorization(component)
        else:
            receipt = trigger["qualification_receipt"]
            receipt["checks"]["source_only_builder"] = 1
            trigger["qualification_result_digest"] = sha256(
                canonical_json_bytes(receipt)
            ).hexdigest()
        raw = canonical_json_bytes(forged)
        with pytest.raises(OpenWorldQualificationAuthorityStaleError):
            SQLiteOpenWorldAttemptMixin._validated_method_authority(
                raw.decode(), sha256(raw).hexdigest()
            )
