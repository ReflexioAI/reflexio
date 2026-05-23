"""Verify Config carries the shadow_mode_enabled toggle and round-trips correctly."""

from reflexio.models.config_schema import Config, StorageConfigSQLite


def test_shadow_mode_enabled_defaults_to_false() -> None:
    """New orgs are not opted into shadow mode by default."""
    config = Config(storage_config=StorageConfigSQLite())
    assert config.shadow_mode_enabled is False


def test_shadow_mode_enabled_can_be_set_via_constructor() -> None:
    """Explicit True flips the toggle on."""
    config = Config(storage_config=StorageConfigSQLite(), shadow_mode_enabled=True)
    assert config.shadow_mode_enabled is True


def test_shadow_mode_enabled_serializes_via_model_dump() -> None:
    """model_dump includes the new field so set_config/get_config round-trip it."""
    config = Config(storage_config=StorageConfigSQLite(), shadow_mode_enabled=True)
    dumped = config.model_dump()
    assert dumped["shadow_mode_enabled"] is True
