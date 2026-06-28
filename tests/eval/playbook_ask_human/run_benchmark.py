"""Run the playbook ask_human invocation benchmark locally.

Example:
    uv run python open_source/reflexio/tests/eval/playbook_ask_human/run_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from tests.eval.playbook_ask_human.case import load_cases
from tests.eval.playbook_ask_human.providers import make_ask_human_prediction_provider
from tests.eval.playbook_ask_human.runner import run_eval


def _json_summary(results) -> dict:
    metrics = results.metrics
    return {
        "cases": metrics.n,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "tn": metrics.tn,
        "fn": metrics.fn,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "by_vertical": {
            vertical: {
                "n": vertical_metrics.n,
                "tp": vertical_metrics.tp,
                "fp": vertical_metrics.fp,
                "tn": vertical_metrics.tn,
                "fn": vertical_metrics.fn,
                "precision": vertical_metrics.precision,
                "recall": vertical_metrics.recall,
                "f1": vertical_metrics.f1,
            }
            for vertical, vertical_metrics in results.by_vertical.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the playbook ask_human invocation benchmark."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model override. Defaults to ModelRole.EXTRACTION_AGENT resolution.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N sorted cases.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the named case id. May be repeated.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a JSON summary.",
    )
    args = parser.parse_args()

    cases = load_cases()
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.id in wanted]
        missing = sorted(wanted - {case.id for case in cases})
        if missing:
            raise SystemExit(f"Unknown case id(s): {missing}")
    if args.limit is not None:
        cases = cases[: args.limit]

    model = args.model or resolve_model_name(ModelRole.EXTRACTION_AGENT)
    print(f"model={model}")
    print(f"cases={len(cases)}")

    predictions = []
    client = LiteLLMClient(LiteLLMConfig(model=model, temperature=0.0, timeout=180))
    with patch(
        "reflexio.server.services.extraction.resumable_agent.is_resumable_extraction_agent_feature_enabled",
        return_value=True,
    ):
        for idx, case in enumerate(cases, start=1):
            # Use fresh storage per case so a prior-pending fixture cannot leak
            # into later cases as Prior Knowledge.
            with tempfile.TemporaryDirectory(
                prefix=f"ask-human-eval-{case.id}-"
            ) as tmp:
                request_context = RequestContext(
                    org_id=f"ask-human-eval-{case.id}",
                    storage_base_dir=tmp,
                )
                provider = make_ask_human_prediction_provider(
                    llm_client=client,
                    request_context=request_context,
                )
                prediction = provider(case)
                predictions.append(prediction)
                print(
                    f"[{idx:02d}/{len(cases)}] {case.id} "
                    f"expected={case.expected_ask_human} "
                    f"actual={prediction.asked_human} "
                    f"tools={prediction.tool_names} "
                    f"playbooks={prediction.playbook_count}",
                    flush=True,
                )

    results = run_eval(cases=cases, predictions=predictions)
    print()
    print(results.summary())

    summary = _json_summary(results)
    print()
    print("json_summary=" + json.dumps(summary, sort_keys=True))
    if args.json_output is not None:
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
