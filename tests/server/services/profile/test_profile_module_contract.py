from __future__ import annotations

from pathlib import Path

_PROFILE_DIR = (
    Path(__file__).resolve().parents[4]
    / "reflexio"
    / "server"
    / "services"
    / "profile"
)
_REMOVED_FILES = {"profile_generation_service.py", "profile_extractor.py"}


def test_profile_canonical_imports_work() -> None:
    from reflexio.server.services.profile.components.consolidator import (
        ProfileConsolidator,
    )
    from reflexio.server.services.profile.components.extractor import ProfileExtractor
    from reflexio.server.services.profile.service import (
        ProfileGenerationService,
        ProfileGenerationServiceConfig,
    )

    assert ProfileGenerationService.__name__ == "ProfileGenerationService"
    assert ProfileGenerationServiceConfig.__name__ == "ProfileGenerationServiceConfig"
    assert ProfileExtractor.__name__ == "ProfileExtractor"
    assert ProfileConsolidator.__name__ == "ProfileConsolidator"


def test_profile_file_layout_uses_canonical_paths() -> None:
    components_dir = _PROFILE_DIR / "components"

    assert (_PROFILE_DIR / "service.py").exists()
    assert (_PROFILE_DIR / "profile_generation_service_utils.py").exists()
    assert (components_dir / "__init__.py").exists()
    assert (components_dir / "extractor.py").exists()
    assert (components_dir / "consolidator.py").exists()
    for removed_file in _REMOVED_FILES:
        assert not (_PROFILE_DIR / removed_file).exists()


def test_profile_durable_names_remain_stable() -> None:
    from reflexio.models.config_schema import Config
    from reflexio.server.services.profile.profile_generation_service_utils import (
        ProfileGenerationServiceConstants,
    )

    assert "profile_extractor_config" in Config.model_fields
    assert (
        ProfileGenerationServiceConstants.PROFILE_SHOULD_GENERATE_PROMPT_ID
        == "profile_should_generate"
    )
    assert (
        ProfileGenerationServiceConstants.PROFILE_UPDATE_MAIN_PROMPT_ID
        == "profile_update_main"
    )
