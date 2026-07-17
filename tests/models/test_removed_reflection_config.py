"""Public contract tests for the removed preview configuration field."""

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import Config, StorageConfigSQLite


def test_removed_reflection_config_is_absent_and_rejected() -> None:
    assert "reflection_config" not in Config.model_fields
    schema = Config.model_json_schema()
    assert "reflection_config" not in schema.get("properties", {})
    assert "ReflectionConfig" not in schema.get("$defs", {})

    payload = {
        "storage_config": StorageConfigSQLite().model_dump(mode="json"),
        "reflection_config": {},
    }
    with pytest.raises(ValidationError) as exc_info:
        Config.model_validate(payload)

    assert any(
        error["loc"] == ("reflection_config",) and error["type"] == "extra_forbidden"
        for error in exc_info.value.errors()
    )
