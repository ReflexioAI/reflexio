"""Tests for ``LineageGCConfig`` (tombstone garbage-collection gate)."""

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import (
    Config,
    LineageGCConfig,
    StorageConfigSQLite,
)


def test_lineage_gc_config_defaults():
    cfg = LineageGCConfig()
    assert cfg.tombstone_grace_window_days == 90
    assert "poll_interval_seconds" not in cfg.model_dump()


def test_config_has_lineage_gc_default():
    cfg = Config(storage_config=StorageConfigSQLite())
    assert isinstance(cfg.lineage_gc, LineageGCConfig)
    assert cfg.lineage_gc.tombstone_grace_window_days == 90


def test_lineage_gc_can_be_explicitly_disabled():
    """Zero grace holds reclamation instead of deleting immediately."""
    cfg = LineageGCConfig(tombstone_grace_window_days=0)
    assert cfg.reclamation_cutoff(1_900_000_000) is None


def test_lineage_gc_fields_can_be_overridden():
    cfg = LineageGCConfig(tombstone_grace_window_days=30)
    assert cfg.tombstone_grace_window_days == 30


def test_lineage_gc_null_falls_back_to_default():
    # A stored config row with an explicit null must fall back to the default
    # rather than failing validation — same pattern as retrieval_floor.
    base = Config(storage_config=StorageConfigSQLite())
    payload = base.model_dump()
    payload["lineage_gc"] = None
    cfg = Config.model_validate(payload)
    assert cfg.lineage_gc == LineageGCConfig()


# ---------------------------------------------------------------------------
# Bounds validation: negative values must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("days", [-1, -90])
def test_lineage_gc_rejects_negative_grace_window(days: int):
    with pytest.raises(ValidationError):
        LineageGCConfig(tombstone_grace_window_days=days)


@pytest.mark.parametrize("interval", [0, -1, -86400])
def test_lineage_gc_rejects_non_positive_poll_interval(interval: int):
    with pytest.raises(ValidationError):
        LineageGCConfig.model_validate({"poll_interval_seconds": interval})


def test_lineage_gc_accepts_minimum_valid_values():
    cfg = LineageGCConfig(tombstone_grace_window_days=1)
    assert cfg.tombstone_grace_window_days == 1
