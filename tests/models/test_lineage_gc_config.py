"""Tests for ``LineageGCConfig`` (tombstone garbage-collection gate)."""

from reflexio.models.config_schema import (
    Config,
    LineageGCConfig,
    StorageConfigSQLite,
)


def test_lineage_gc_config_defaults():
    cfg = LineageGCConfig()
    assert cfg.enabled is False
    assert cfg.tombstone_grace_window_days == 90
    assert cfg.poll_interval_seconds == 86400


def test_config_has_lineage_gc_default():
    cfg = Config(storage_config=StorageConfigSQLite())
    assert isinstance(cfg.lineage_gc, LineageGCConfig)
    assert cfg.lineage_gc.enabled is False


def test_lineage_gc_enabled_can_be_set():
    cfg = LineageGCConfig(enabled=True)
    assert cfg.enabled is True


def test_lineage_gc_fields_can_be_overridden():
    cfg = LineageGCConfig(tombstone_grace_window_days=30, poll_interval_seconds=3600)
    assert cfg.tombstone_grace_window_days == 30
    assert cfg.poll_interval_seconds == 3600


def test_lineage_gc_null_falls_back_to_default():
    # A stored config row with an explicit null must fall back to the default
    # rather than failing validation — same pattern as retrieval_floor.
    base = Config(storage_config=StorageConfigSQLite())
    payload = base.model_dump()
    payload["lineage_gc"] = None
    cfg = Config.model_validate(payload)
    assert cfg.lineage_gc == LineageGCConfig()
