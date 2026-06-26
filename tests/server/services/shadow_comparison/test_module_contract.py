from __future__ import annotations

from pathlib import Path


def test_shadow_comparison_canonical_imports_work() -> None:
    from reflexio.server.services.shadow_comparison.judge import ShadowComparisonJudge
    from reflexio.server.services.shadow_comparison.outcome import (
        Outcome,
        assign_positions,
        derive_reflexio_outcome,
    )

    assert ShadowComparisonJudge.__name__ == "ShadowComparisonJudge"
    assert Outcome.WIN == "win"
    assert assign_positions.__name__ == "assign_positions"
    assert derive_reflexio_outcome.__name__ == "derive_reflexio_outcome"


def test_shadow_comparison_stays_compact_without_components_package() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    module_dir = repo_root / "reflexio/server/services/shadow_comparison"

    assert not (module_dir / "components").exists()
    assert (module_dir / "judge.py").exists()
    assert (module_dir / "outcome.py").exists()
