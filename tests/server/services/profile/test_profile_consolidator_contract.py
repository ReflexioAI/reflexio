from __future__ import annotations

import inspect
from pathlib import Path


def test_profile_consolidator_canonical_imports_work() -> None:
    from reflexio.server.services.profile.components.consolidator import (
        ProfileConsolidator,
        ProfileDeduplicationOutput,
        ProfileDeletionDirective,
        ProfileDuplicateGroup,
    )

    assert ProfileConsolidator.__name__ == "ProfileConsolidator"
    assert ProfileDeduplicationOutput.__name__ == "ProfileDeduplicationOutput"
    assert ProfileDeletionDirective.__name__ == "ProfileDeletionDirective"
    assert ProfileDuplicateGroup.__name__ == "ProfileDuplicateGroup"


def test_legacy_profile_deduplicator_file_removed() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    profile_dir = repo_root / "reflexio/server/services/profile"

    assert not (profile_dir / "profile_deduplicator.py").exists()


def test_profile_consolidator_preserves_durable_names() -> None:
    from reflexio.models.config_schema import UserPlaybookExtractorConfig
    from reflexio.server.services.profile.components.consolidator import (
        ProfileConsolidator,
        ProfileDeduplicationOutput,
    )
    from reflexio.server.site_var import feature_flags
    from reflexio.test_support.llm_model_registry import _build_registry

    registry = _build_registry()

    assert ProfileConsolidator.DEDUPLICATION_PROMPT_ID == "profile_deduplication"
    assert registry["profile_deduplication"].model_class is ProfileDeduplicationOutput
    assert "deduplication_config" in UserPlaybookExtractorConfig.model_fields
    assert '"deduplicator"' in inspect.getsource(feature_flags.is_deduplicator_enabled)
