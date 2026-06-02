"""Golden-set extraction eval harness (AI-judged).

Wires the existing golden YAML cases, the shared
:class:`~tests.eval.judge.LLMJudge`, and the ``extraction_rubric.yaml``
into a load -> (extract | precomputed) -> judge -> aggregate runner.

Extraction is **float-scored, not label-scored**: the headline metrics
are the judge's ``signal_f1`` and ``grounded_rate`` (averaged) plus a
``pass_rate(threshold)``. There is no decision "kind" and no confusion
matrix. See :mod:`tests.eval.extraction.runner` for details.
"""

from __future__ import annotations

from tests.eval.extraction.runner import (
    CaseOutcome,
    EvalResults,
    run_eval,
    score_case,
)

__all__ = ["CaseOutcome", "EvalResults", "run_eval", "score_case"]
