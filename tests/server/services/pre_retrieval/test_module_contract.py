from __future__ import annotations

from pathlib import Path


def test_pre_retrieval_public_package_imports_work() -> None:
    from reflexio.server.services.pre_retrieval import (
        DocumentExpander,
        ExpansionResult,
        QueryReformulator,
        ReformulationResult,
        ReformulationSearchResult,
    )

    assert DocumentExpander.__name__ == "DocumentExpander"
    assert ExpansionResult.__name__ == "ExpansionResult"
    assert QueryReformulator.__name__ == "QueryReformulator"
    assert ReformulationResult.__name__ == "ReformulationResult"
    assert ReformulationSearchResult.__name__ == "ReformulationSearchResult"


def test_pre_retrieval_stays_compact_without_components_package() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    module_dir = repo_root / "reflexio/server/services/pre_retrieval"

    assert not (module_dir / "components").exists()
    assert (module_dir / "_query_reformulator.py").exists()
    assert (module_dir / "_document_expander.py").exists()
