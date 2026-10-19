"""Retention tuning shares operational policy without overriding customer limits."""

import pytest

from reflexio.server.services.storage import retention

OPERATIONAL = (
    "retrieved_learning_evaluation",
    "offline_tuner_reward_label",
    "share_links",
    "agent_playbook_source_user_playbooks",
    "playbook_optimization_jobs",
    "playbook_optimization_candidates",
    "playbook_optimization_evaluations",
    "playbook_optimization_events",
    "playbook_retrieval_logs",
)
CUSTOMER = (
    "profiles",
    "interactions",
    "requests",
    "user_playbooks",
    "agent_playbooks",
    "agent_success_evaluation_result",
    "skills",
)
SHARED = "REFLEXIO_OPERATIONAL_ROW_LIMIT"


@pytest.fixture(autouse=True)
def clean_policy_environment(monkeypatch):
    for target in retention.RETENTION_TARGETS:
        monkeypatch.delenv(f"REFLEXIO_ROW_LIMIT_{target.name.upper()}", raising=False)
    monkeypatch.delenv("INTERACTION_CLEANUP_THRESHOLD", raising=False)
    monkeypatch.delenv(SHARED, raising=False)


@pytest.mark.parametrize("raw_cap", [None, "", " "])
def test_default_caps_and_registry_partition_are_preserved(monkeypatch, raw_cap):
    if raw_cap is not None:
        monkeypatch.setenv(SHARED, raw_cap)
    limits = retention.get_row_retention_limits()
    assert set(OPERATIONAL) == {
        t.name
        for t in retention.RETENTION_TARGETS
        if not t.customer_data and t.fixed_row_limit is None
    }
    assert set(CUSTOMER) == {
        t.name for t in retention.RETENTION_TARGETS if t.customer_data
    }
    assert set(limits.values()) == {250_000}


@pytest.mark.parametrize("cap", [0, 17])
def test_one_operational_cap_keeps_customer_and_fixed_limits(monkeypatch, cap):
    monkeypatch.setenv(SHARED, str(cap))
    monkeypatch.setenv("REFLEXIO_ROW_LIMIT_PROFILES", "9")
    monkeypatch.setenv("INTERACTION_CLEANUP_THRESHOLD", "11")
    limits = retention.get_row_retention_limits()
    assert {limits[name] for name in OPERATIONAL} == {cap}
    assert limits["profiles"] == 9
    assert limits["interactions"] == 11
    assert limits["requests"] == 250_000
    assert limits["user_playbook_exposure_events"] == 250_000


@pytest.mark.parametrize("target", OPERATIONAL)
@pytest.mark.parametrize("value", [0, 7, 250000])
def test_independent_operational_override(monkeypatch, target, value):
    monkeypatch.setenv(f"REFLEXIO_ROW_LIMIT_{target.upper()}", str(value))
    limits = retention.get_row_retention_limits()
    assert limits[target] == value
    assert {limits[name] for name in OPERATIONAL if name != target} == {250000}


@pytest.mark.parametrize("target", OPERATIONAL)
@pytest.mark.parametrize("value", ["-1", "invalid"])
def test_invalid_operational_override_is_not_a_deletion_default(
    monkeypatch, target, value
):
    key = f"REFLEXIO_ROW_LIMIT_{target.upper()}"
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        retention.get_row_retention_limits()


@pytest.mark.parametrize("shared", [0, 7, 500000])
def test_old_default_and_custom_shared_cap_require_disambiguation(monkeypatch, shared):
    # The consolidation previously ignored this old default. Do not silently
    # turn an inherited zero/high cap into a smaller deleting override.
    monkeypatch.setenv("REFLEXIO_ROW_LIMIT_SHARE_LINKS", "250000")
    monkeypatch.setenv(SHARED, str(shared))
    with pytest.raises(ValueError, match="REFLEXIO_ROW_LIMIT_SHARE_LINKS.*ambiguous"):
        retention.get_row_retention_limits()


def test_mapping_is_complete_policy_authority(monkeypatch):
    monkeypatch.setenv(SHARED, "bad ambient value")
    assert set(retention.get_row_retention_limits(env={}).values()) == {250000}
    limits = retention.get_row_retention_limits(
        env={
            SHARED: "17",
            "REFLEXIO_ROW_LIMIT_SHARE_LINKS": "0",
            "REFLEXIO_ROW_LIMIT_PLAYBOOK_OPTIMIZATION_JOBS": "9",
            "REFLEXIO_ROW_LIMIT_PROFILES": "21",
            "INTERACTION_CLEANUP_THRESHOLD": "11",
        }
    )
    assert limits["share_links"] == 0
    assert limits["playbook_optimization_jobs"] == 9
    assert limits["playbook_optimization_candidates"] == 17
    assert limits["profiles"] == 21
    assert limits["interactions"] == 11
    assert limits["user_playbook_exposure_events"] == 250000


@pytest.mark.parametrize("value", ["-1", "invalid"])
def test_invalid_shared_policy_does_not_fall_back_to_deletion(monkeypatch, value):
    monkeypatch.setenv(SHARED, value)
    with pytest.raises(ValueError, match=SHARED):
        retention.get_row_retention_limits()


@pytest.mark.integration
@pytest.mark.parametrize("source", ["shared", "override"])
@pytest.mark.parametrize("bad_value", ["invalid", "-1", "250000"])
def test_sqlite_sweep_uses_overrides_and_validates_before_storage_queries(
    tmp_path, monkeypatch, source, bad_value
):
    from reflexio.server.services.storage.retention_sweep import sweep_retention_caps
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(
        org_id="operational-policy", db_path=str(tmp_path / "retention.db")
    )
    storage.migrate()
    # This optional enterprise table exercises the existing generic storage seam.
    storage.conn.execute(
        "CREATE TABLE share_links (id TEXT PRIMARY KEY, created_at INTEGER NOT NULL)"
    )
    storage.conn.executemany(
        "INSERT INTO share_links VALUES (?, ?)",
        [("first", 1), ("second", 2), ("third", 3)],
    )
    storage.conn.commit()
    monkeypatch.setenv(SHARED, "0")
    assert sweep_retention_caps(storage.org_id, storage).deleted == 0
    assert storage.count_retention_target_rows("share_links") == 3
    if source == "shared":
        monkeypatch.setenv(SHARED, "2")
    else:
        monkeypatch.setenv("REFLEXIO_ROW_LIMIT_SHARE_LINKS", "2")
    result = sweep_retention_caps(storage.org_id, storage)
    assert not result.failed
    assert result.deleted == 1
    assert [
        r[0]
        for r in storage.conn.execute(
            "SELECT id FROM share_links ORDER BY created_at LIMIT 10"
        )
    ] == ["second", "third"]

    # Cap validation must finish before storage leases, probes, archiving or deletes.
    monkeypatch.setenv("REFLEXIO_ROW_LIMIT_SHARE_LINKS", "0")
    assert sweep_retention_caps(storage.org_id, storage).deleted == 0
    assert storage.count_retention_target_rows("share_links") == 2
    monkeypatch.setenv("REFLEXIO_ROW_LIMIT_SHARE_LINKS", bad_value)
    queries = []
    storage.conn.set_trace_callback(queries.append)
    result = sweep_retention_caps(storage.org_id, storage)
    storage.conn.set_trace_callback(None)
    assert result.failed
    assert result.deleted == 0
    assert queries == []
    assert storage.count_retention_target_rows("share_links") == 2
    storage.conn.close()


def test_invalid_policy_is_rejected_before_scheduler_start(monkeypatch):
    from unittest.mock import MagicMock

    from reflexio.server.services.lineage.gc_scheduler import maybe_start_lineage_gc

    monkeypatch.setenv("REFLEXIO_ROW_LIMIT_SHARE_LINKS", "invalid")
    factory = MagicMock()
    with pytest.raises(ValueError, match="REFLEXIO_ROW_LIMIT_SHARE_LINKS"):
        maybe_start_lineage_gc(factory, bootstrap_org_id="migration-check")
    factory.assert_not_called()


def test_managed_alert_cap_remains_separate_from_operational_cleanup(monkeypatch):
    from reflexio.server.services.storage.retention_sweep import _effective_limits

    monkeypatch.setenv(SHARED, "17")
    policy = retention.RetentionPolicy(
        customer_cap_action="alert", customer_cap_limit=2_000_000
    )
    limits = _effective_limits(policy)
    assert {limits[name] for name in OPERATIONAL} == {17}
    assert {limits[name] for name in CUSTOMER} == {2_000_000}
    assert limits["user_playbook_exposure_events"] == 250_000
