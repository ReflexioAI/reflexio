from __future__ import annotations

from pathlib import Path


def test_tagging_canonical_imports_work() -> None:
    from reflexio.server.services.tagging.service import TaggingService, TagsOutput
    from reflexio.server.services.tagging.tagging_scheduler import (
        TaggingScheduler,
        schedule_tagging,
    )

    assert TaggingService.__name__ == "TaggingService"
    assert TagsOutput.__name__ == "TagsOutput"
    assert TaggingScheduler.__name__ == "TaggingScheduler"
    assert schedule_tagging.__name__ == "schedule_tagging"


def test_legacy_tagging_module_file_removed() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    tagging_dir = repo_root / "reflexio/server/services/tagging"

    assert not (tagging_dir / ("tagging_" "service.py")).exists()
