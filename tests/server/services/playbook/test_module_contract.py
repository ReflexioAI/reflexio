from __future__ import annotations

import sys
from pathlib import Path

_PY_SUFFIX = "." + "py"
# Keep legacy filenames split so stale-path greps catch real reintroductions
# without matching this contract test.
_LEGACY_FILES = (
    "playbook_generation_" + "service" + _PY_SUFFIX,
    "playbook_" + "extractor" + _PY_SUFFIX,
    "playbook_" + "consolidator" + _PY_SUFFIX,
    "playbook_" + "aggregator" + _PY_SUFFIX,
)
_MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "reflexio"
    / "server"
    / "services"
    / "playbook"
)


def test_playbook_canonical_imports_work() -> None:
    from reflexio.server.services.playbook.components.aggregator import (
        PlaybookAggregator,
    )
    from reflexio.server.services.playbook.components.consolidator import (
        PlaybookConsolidator,
    )
    from reflexio.server.services.playbook.components.extractor import (
        PlaybookExtractor,
    )
    from reflexio.server.services.playbook.service import (
        PlaybookGenerationService,
        PlaybookGenerationServiceConfig,
        read_user_playbook_as_of_for_learning,
    )

    assert PlaybookGenerationService.__name__ == "PlaybookGenerationService"
    assert PlaybookGenerationServiceConfig.__name__ == "PlaybookGenerationServiceConfig"
    assert read_user_playbook_as_of_for_learning.__name__ == (
        "read_user_playbook_as_of_for_learning"
    )
    assert PlaybookExtractor.__name__ == "PlaybookExtractor"
    assert PlaybookConsolidator.__name__ == "PlaybookConsolidator"
    assert PlaybookAggregator.__name__ == "PlaybookAggregator"


def test_playbook_component_file_layout() -> None:
    components_dir = _MODULE_DIR / "components"

    assert (_MODULE_DIR / "service.py").exists()
    assert (_MODULE_DIR / "playbook_service_utils.py").exists()
    assert (_MODULE_DIR / "playbook_service_constants.py").exists()
    assert (_MODULE_DIR / "playbook_edit_apply.py").exists()
    assert (components_dir / "__init__.py").exists()
    assert (components_dir / "extractor.py").exists()
    assert (components_dir / "consolidator.py").exists()
    assert (components_dir / "aggregator.py").exists()
    for filename in _LEGACY_FILES:
        assert not (_MODULE_DIR / filename).exists()


def test_playbook_prompt_ids_remain_durable() -> None:
    from reflexio.server.services.playbook.playbook_service_constants import (
        PlaybookServiceConstants,
    )

    assert (
        PlaybookServiceConstants.PLAYBOOK_SHOULD_GENERATE_PROMPT_ID
        == "playbook_should_generate"
    )
    assert (
        PlaybookServiceConstants.PLAYBOOK_EXTRACTION_CONTEXT_PROMPT_ID
        == "playbook_extraction_context"
    )
    assert (
        PlaybookServiceConstants.PLAYBOOK_EXTRACTION_PROMPT_ID
        == "playbook_extraction_main"
    )
    assert PlaybookServiceConstants.PLAYBOOK_AGGREGATION_PROMPT_ID == (
        "playbook_aggregation"
    )


def test_playbook_service_import_preserves_lazy_consolidator_boundary() -> None:
    sys.modules.pop(
        "reflexio.server.services.playbook.components.consolidator",
        None,
    )

    from reflexio.server.services.playbook.service import PlaybookGenerationService

    assert PlaybookGenerationService.__name__ == "PlaybookGenerationService"
    assert "reflexio.server.services.playbook.components.consolidator" not in (
        sys.modules
    )
