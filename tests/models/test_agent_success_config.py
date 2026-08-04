from __future__ import annotations

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import (
    AgentSuccessConfig,
    Config,
    validate_stored_config,
)


def test_agent_success_tagging_prompt_round_trip() -> None:
    config = AgentSuccessConfig(
        success_definition_prompt="success rules",
        tagging_definition_prompt="tag by support topic",
    )

    assert config.model_dump()["tagging_definition_prompt"] == "tag by support topic"


def test_legacy_agent_success_metadata_is_ignored_only_for_stored_config() -> None:
    stored = validate_stored_config(
        {
            "storage_config": {"db_path": None},
            "agent_context_prompt": "preserve me",
            "window_size": 23,
            "agent_success_config": {
                "success_definition_prompt": "preserve success rules",
                "metadata_definition_prompt": "do not migrate me",
                "sampling_rate": 0.75,
            },
        }
    )

    assert stored.agent_context_prompt == "preserve me"
    assert stored.window_size == 23
    assert stored.agent_success_config is not None
    assert (
        stored.agent_success_config.success_definition_prompt
        == "preserve success rules"
    )
    assert stored.agent_success_config.sampling_rate == 0.75
    assert stored.agent_success_config.tagging_definition_prompt is None
    assert "metadata_definition_prompt" not in stored.agent_success_config.model_dump()

    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "storage_config": {"db_path": None},
                "agent_success_config": {
                    "success_definition_prompt": "success rules",
                    "metadata_definition_prompt": "retired",
                },
            },
            extra="forbid",
        )
