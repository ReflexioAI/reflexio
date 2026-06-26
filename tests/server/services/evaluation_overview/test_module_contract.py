from __future__ import annotations

from pathlib import Path


def test_evaluation_overview_canonical_imports_work() -> None:
    from reflexio.server.services.evaluation_overview.components.distribution import (
        bucket_corrections,
    )
    from reflexio.server.services.evaluation_overview.components.hero_state import (
        HeroState,
        compute_hero_state,
    )
    from reflexio.server.services.evaluation_overview.components.rule_attribution import (
        RuleAttribution,
        compute_net_sessions,
    )
    from reflexio.server.services.evaluation_overview.components.shadow_aggregation import (
        compute_shadow_win_rate_trend,
    )
    from reflexio.server.services.evaluation_overview.eval_sampler import (
        SampleCandidate,
        sample_candidates,
    )
    from reflexio.server.services.evaluation_overview.service import (
        EvaluationOverviewService,
    )

    assert EvaluationOverviewService.__name__ == "EvaluationOverviewService"
    assert bucket_corrections.__name__ == "bucket_corrections"
    assert HeroState.FULL == "full"
    assert compute_hero_state.__name__ == "compute_hero_state"
    assert RuleAttribution.__name__ == "RuleAttribution"
    assert compute_net_sessions.__name__ == "compute_net_sessions"
    assert compute_shadow_win_rate_trend.__name__ == "compute_shadow_win_rate_trend"
    assert SampleCandidate.__name__ == "SampleCandidate"
    assert sample_candidates.__name__ == "sample_candidates"


def test_evaluation_overview_component_file_layout() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    module_dir = repo_root / "reflexio/server/services/evaluation_overview"
    components_dir = module_dir / "components"

    assert components_dir.is_dir()
    assert (module_dir / "service.py").exists()
    assert (module_dir / "eval_sampler.py").exists()
    assert (components_dir / "distribution.py").exists()
    assert (components_dir / "hero_state.py").exists()
    assert (components_dir / "rule_attribution.py").exists()
    assert (components_dir / "shadow_aggregation.py").exists()
    assert not (module_dir / "distribution.py").exists()
    assert not (module_dir / "hero_state.py").exists()
    assert not (module_dir / "rule_attribution.py").exists()
    assert not (module_dir / "shadow_aggregation.py").exists()
