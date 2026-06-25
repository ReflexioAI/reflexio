from __future__ import annotations

from pathlib import Path


def test_reflection_canonical_imports_work() -> None:
    from reflexio.server.services.reflection.components.extractor import (
        ReflectionExtractor,
    )
    from reflexio.server.services.reflection.service import ReflectionService

    assert ReflectionExtractor.__name__ == "ReflectionExtractor"
    assert ReflectionService.__name__ == "ReflectionService"


def test_legacy_reflection_module_files_removed() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    reflection_dir = repo_root / "reflexio/server/services/reflection"

    assert not (reflection_dir / "reflection_service.py").exists()
    assert not (reflection_dir / "reflection_extractor.py").exists()
