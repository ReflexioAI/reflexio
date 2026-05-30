"""Config dispatcher for the (now-unified) extraction/search services.

The agentic extraction/search backends were removed in the extraction/search
unification: there is a single extraction fan-out and a single unified search
service, with no config-driven backend selection.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.generation_service import (
    build_extraction_service,
    build_search_service,
)


def _make_config(**overrides) -> Config:
    """Build a minimal Config with optional field overrides.

    Args:
        **overrides: Field overrides for Config.

    Returns:
        Config: Minimal valid Config instance.
    """
    base: dict = {
        "storage_config": StorageConfigSQLite(),
    }
    base.update(overrides)
    return Config(**base)


def test_build_extraction_service_returns_profile_generation_service() -> None:
    config = _make_config()
    svc = build_extraction_service(
        config, llm_client=MagicMock(), request_context=MagicMock()
    )
    assert svc.__class__.__name__ == "ProfileGenerationService"


def test_build_search_service_returns_unified_search_service() -> None:
    config = _make_config()
    svc = build_search_service(
        config, llm_client=MagicMock(), request_context=MagicMock()
    )
    assert svc.__class__.__name__ == "UnifiedSearchService"
