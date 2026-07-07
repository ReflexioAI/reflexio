"""Real-LLM search eval over the golden set (manual, costs money).

Runs the classic unified search backend over every golden search case with
real embeddings (hybrid mode) and scores with the REAL LLM judge, printing
the per-category table. Lives under ``tests/e2e_tests/`` so the global
litellm mock is disabled.

Run:
    set -a && source .env && set +a && \
    RUN_LOW_PRIORITY=1 REFLEXIO_EVAL_REAL_JUDGE=1 \
    uv run pytest tests/e2e_tests/test_search_eval_real_llm.py -v -o 'addopts=' -s

Assertions are loose floors only — the point of this test is the printed
per-category table, not exact scores.
"""

from __future__ import annotations

import pytest

from reflexio.test_support.skip_decorators import skip_low_priority
from tests.eval.conftest import _load, _load_rubric, _real_judge
from tests.eval.search.providers import make_classic_search_provider
from tests.eval.search.runner import run_eval

pytestmark = [pytest.mark.e2e, pytest.mark.requires_credentials]


@skip_low_priority
def test_classic_search_eval_real_llm(tmp_path):  # pragma: no cover - manual
    from reflexio.models.config_schema import SearchMode
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    cases = _load("search")
    provider = make_classic_search_provider(
        storage_base_dir=str(tmp_path),
        llm_client=LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5")),
        search_mode=SearchMode.HYBRID,
    )
    judge = _real_judge(_load_rubric("search_rubric.yaml"))

    results = run_eval(cases=cases, provider=provider, backend="classic", judge=judge)

    print()
    print(results.summary())

    assert results.n == len(cases)
    judged = [o for o in results.outcomes if o.answer_correctness is not None]
    assert judged, "real judge returned no scores"
    for outcome in results.outcomes:
        assert 0.0 <= outcome.recall_at_k <= 1.0
        assert 0.0 <= outcome.mrr <= 1.0


@skip_low_priority
def test_reformulation_extracts_expected_time_windows():  # pragma: no cover - manual
    """The real reformulation call extracts each golden case's declared
    ``expected_time_window`` (loose tolerance — phrasing like "before this
    month" has no single exact day count)."""
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.pre_retrieval import QueryReformulator

    cases = [c for c in _load("search") if c.get("expected_time_window")]
    assert cases, "no golden cases declare expected_time_window"

    reformulator = QueryReformulator(
        llm_client=LiteLLMClient(LiteLLMConfig(model="claude-haiku-4-5")),
        prompt_manager=PromptManager(),
    )
    for case in cases:
        result = reformulator.rewrite(case["query"])
        for bound, expected in case["expected_time_window"].items():
            actual = getattr(result, bound)
            assert actual is not None, f"{case['id']}: {bound} not extracted"
            tolerance = max(2.0, 0.5 * float(expected))
            assert abs(actual - float(expected)) <= tolerance, (
                f"{case['id']}: {bound}={actual}, expected ~{expected}"
            )
