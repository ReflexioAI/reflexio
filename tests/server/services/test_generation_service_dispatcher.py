"""Task 2.6: config dispatcher for extraction/search backends."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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


def test_config_defaults_extraction_backend_to_classic() -> None:
    config = _make_config()
    assert config.extraction_backend == "classic"


def test_config_defaults_search_backend_to_classic() -> None:
    config = _make_config()
    assert config.search_backend == "classic"


def test_config_accepts_agentic_backends() -> None:
    config = _make_config(extraction_backend="agentic", search_backend="agentic")
    assert config.extraction_backend == "agentic"
    assert config.search_backend == "agentic"


def test_build_extraction_service_picks_classic_by_default() -> None:
    config = _make_config()
    svc = build_extraction_service(
        config, llm_client=MagicMock(), request_context=MagicMock()
    )
    assert svc.__class__.__name__ == "ProfileGenerationService"


def test_build_search_service_picks_classic_by_default() -> None:
    config = _make_config()
    svc = build_search_service(
        config, llm_client=MagicMock(), request_context=MagicMock()
    )
    assert svc.__class__.__name__ == "UnifiedSearchService"


def test_build_extraction_service_picks_agentic_when_configured() -> None:
    try:
        from reflexio.server.services.extraction.agentic_extraction_service import (  # noqa: F401  # type: ignore[import-not-found]
            AgenticExtractionService,
        )
    except ImportError:
        pytest.skip("AgenticExtractionService not yet implemented (Phase 3)")
    config = _make_config(extraction_backend="agentic")
    svc = build_extraction_service(
        config, llm_client=MagicMock(), request_context=MagicMock()
    )
    assert svc.__class__.__name__ == "AgenticExtractionService"


def test_build_search_service_picks_agentic_when_configured() -> None:
    try:
        from reflexio.server.services.search.agentic_search_service import (  # noqa: F401  # type: ignore[import-not-found]
            AgenticSearchService,
        )
    except ImportError:
        pytest.skip("AgenticSearchService not yet implemented (Phase 4)")
    config = _make_config(search_backend="agentic")
    svc = build_search_service(
        config, llm_client=MagicMock(), request_context=MagicMock()
    )
    assert svc.__class__.__name__ == "AgenticSearchService"
