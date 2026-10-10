"""Paired reason-code evaluation using frozen local cases and the real reviewer.

No database access. Cases and settings are identical for both prompt versions.
Errors are recorded separately and make the command fail. Never infer production
quality from the synthetic cases alone; see developer.md for the rollout gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

from pydantic import BaseModel, Field

from reflexio.models.api_schema.domain.entities import (
    Interaction,
    Request,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.enums import PlaybookReviewReasonCode
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.components.reviewer import (
    PlaybookCandidateReviewer,
)
from reflexio.test_support.llm_mock import assert_litellm_unpatched
from reflexio.test_support.reviewer_metrics import ReviewCallMetrics

DEFAULT_CASES = (
    Path(__file__).resolve().parents[1] / "tests/test_data/reviewer_reason_codes.json"
)


def _write_private_text(path: Path, text: str) -> None:
    """Keep raw revision and supporting-source artifacts owner-readable only."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


class Turn(BaseModel):
    role: str
    content: str


class Candidate(BaseModel):
    content: str
    trigger: str
    rationale: str
    evidence: list[int] = Field(min_length=1)
    decision: Literal["accept", "revise", "reject"]
    allowed_decisions: list[Literal["accept", "revise", "reject"]] | None = Field(
        default=None, min_length=1
    )
    reason_code: PlaybookReviewReasonCode | None
    preserve: list[str] = Field(default_factory=list)
    preserve_any: list[list[str]] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)


class Case(BaseModel):
    id: str
    provenance: str
    agent_context: str
    turns: list[Turn] = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=1)
    existing: list[str] = Field(default_factory=list)
    domain: str = "unspecified"
    split: Literal["development", "holdout"] = "development"


def load_cases(path: Path) -> list[Case]:
    cases = [Case.model_validate(item) for item in json.loads(path.read_text())]
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("cases must be nonempty with unique ids")
    for case in cases:
        for candidate in case.candidates:
            if any(
                index < 1 or index > len(case.turns) for index in candidate.evidence
            ):
                raise ValueError(f"{case.id}: evidence index outside chronology")
    return cases


def prepare_case(
    case: Case,
) -> tuple[list[RequestInteractionDataModel], list[UserPlaybook], list[UserPlaybook]]:
    sessions = [
        RequestInteractionDataModel(
            session_id=case.id,
            request=Request(
                request_id=f"{case.id}-{index}",
                user_id="eval",
                agent_version="eval",
                session_id=case.id,
                created_at=index,
            ),
            interactions=[
                Interaction(
                    interaction_id=index,
                    user_id="eval",
                    request_id=f"{case.id}-{index}",
                    role=turn.role,
                    content=turn.content,
                    created_at=index,
                )
            ],
        )
        for index, turn in enumerate(case.turns, 1)
    ]
    candidates = [
        UserPlaybook(
            user_playbook_id=index,
            agent_version="eval",
            user_id="eval",
            request_id=case.id,
            content=candidate.content,
            trigger=candidate.trigger,
            rationale=candidate.rationale,
            source_interaction_ids=candidate.evidence,
        )
        for index, candidate in enumerate(case.candidates, 1)
    ]
    existing = [
        UserPlaybook(
            user_playbook_id=index,
            agent_version="eval",
            request_id=case.id,
            content=content,
        )
        for index, content in enumerate(case.existing, 1)
    ]
    return sessions, candidates, existing


def evaluate(
    cases: list[Case],
    client: LiteLLMClient,
    repeats: int,
    checkpoint: Path | None = None,
    *,
    candidate_version: str = "1.4.0",
    max_tokens: int | None = None,
    reasoning_effort: Literal["high", "none"] | None = None,
) -> dict:
    if candidate_version not in (
        "1.4.0",
        "1.5.0",
        "1.6.0",
        "1.7.0",
        "1.8.0",
        "1.9.0",
        "1.10.0",
        "1.11.0",
        "1.12.0",
        "1.13.0",
        "1.14.0",
        "1.15.0",
    ):
        raise ValueError(
            "candidate_version must be 1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.8.0, 1.9.0, 1.10.0, 1.11.0, 1.12.0, 1.13.0, 1.14.0 or 1.15.0"
        )
    if not cases:
        raise ValueError("cases must be nonempty")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if reasoning_effort is not None and client.config.model != "zai/glm-5.2":
        raise ValueError(
            "reasoning profile is qualified for the GLM-5.2 experiment only"
        )
    prompt_identities = {
        version: PromptManager(
            version_override={"playbook_candidate_review": version}
        ).get_prompt_template_identity("playbook_candidate_review")
        for version in ("1.3.0", candidate_version)
    }
    prepared = {case.id: prepare_case(case) for case in cases}
    rows = []
    errors = []
    calls = []
    measurement_errors = []
    settings = {"max_tokens": max_tokens, "reasoning_effort": reasoning_effort}
    case_hash = hashlib.sha256(
        json.dumps(
            [case.model_dump(mode="json") for case in cases], sort_keys=True
        ).encode()
    ).hexdigest()

    def save_progress() -> None:
        if checkpoint is not None:
            _write_private_text(
                checkpoint,
                json.dumps(
                    {
                        "complete": False,
                        "model": client.config.model,
                        "cases_sha256": case_hash,
                        "candidate_version": candidate_version,
                        "prompt_identities": prompt_identities,
                        "rows": rows,
                        "errors": errors,
                        "calls": calls,
                        "inference_settings": settings,
                        "measurement_errors": measurement_errors,
                    },
                    indent=2,
                ),
            )

    for case in cases:
        sessions, candidates, existing = prepared[case.id]
        for repeat in range(repeats):
            # Alternate order to reduce time/order effects; no arm-specific model settings.
            versions = (
                ("1.3.0", candidate_version)
                if repeat % 2 == 0
                else (candidate_version, "1.3.0")
            )
            for version in versions:
                context = cast(
                    RequestContext,
                    SimpleNamespace(
                        prompt_manager=PromptManager(
                            version_override={"playbook_candidate_review": version}
                        ),
                        org_id="local-reason-eval",
                        storage=None,
                    ),
                )
                metrics = ReviewCallMetrics(
                    client.config.model, max_tokens, reasoning_effort
                )
                reviewer = PlaybookCandidateReviewer(
                    request_context=context,
                    llm_client=client,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    provider_request_guard=metrics.guard,
                )
                try:
                    with metrics:
                        result = reviewer.decide(
                            candidates=candidates,
                            request_interaction_data_models=sessions,
                            existing_playbooks=existing,
                            agent_context=case.agent_context,
                            playbook_definition="Reusable user guidance",
                            tool_context="",
                        )
                except Exception as exc:
                    errors.append(
                        {
                            "case": case.id,
                            "repeat": repeat,
                            "version": version,
                            "error_type": type(exc).__name__,
                        }
                    )
                    continue
                finally:
                    calls.append(
                        {
                            "case": case.id,
                            "repeat": repeat,
                            "version": version,
                            **metrics.data,
                        }
                    )
                    if not metrics.data["observation_complete"]:
                        measurement_errors.append(
                            {
                                "case": case.id,
                                "repeat": repeat,
                                "version": version,
                                "error_type": "IncompleteProviderObservation",
                            }
                        )
                    save_progress()
                expected = {
                    f"C{index}": candidate
                    for index, candidate in enumerate(case.candidates, 1)
                }
                for decision in result.output.decisions:
                    candidate_id = decision.candidate_id.strip()
                    target = expected[candidate_id]
                    rows.append(
                        {
                            "case": case.id,
                            "repeat": repeat,
                            "version": version,
                            "candidate": candidate_id,
                            "decision": decision.decision,
                            "reason_code": decision.reason_code,
                            "reason": decision.reason,
                            "evidence_ids": list(decision.evidence_ids),
                            "supporting_evidence": [
                                support.model_dump(mode="json", by_alias=True)
                                for support in result.supporting_evidence
                                if support.candidate_id.strip() == candidate_id
                            ],
                            "revision": (
                                decision.revision.model_dump()
                                if decision.revision is not None
                                else None
                            ),
                            "decision_correct": decision.decision
                            in (target.allowed_decisions or [target.decision]),
                            "code_correct": (
                                decision.reason_code == target.reason_code
                                if target.reason_code is not None
                                else None
                            ),
                        }
                    )
                save_progress()
                print(
                    f"{case.id} repeat={repeat + 1} version={version} processed={len(result.output.decisions)}",
                    flush=True,
                )
    summary = {}
    expected_count = sum(len(case.candidates) for case in cases) * repeats
    for version in ("1.3.0", candidate_version):
        arm = [row for row in rows if row["version"] == version]
        summary[version] = {
            "expected": expected_count,
            "processed": len(arm),
            "decision_correct": sum(row["decision_correct"] for row in arm),
            "code_correct": sum(row["code_correct"] is True for row in arm),
            "code_expected": sum(
                candidate.reason_code is not None
                for case in cases
                for candidate in case.candidates
            )
            * repeats,
            "errors": sum(error["version"] == version for error in errors),
        }
    return {
        "complete": True,
        "model": client.config.model,
        "temperature": client.config.temperature,
        "cases_sha256": case_hash,
        "candidate_version": candidate_version,
        "prompt_identities": prompt_identities,
        "repeats": repeats,
        "summary": summary,
        "rows": rows,
        "errors": errors,
        "calls": calls,
        "inference_settings": settings,
        "measurement_errors": measurement_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", choices=("development", "holdout"))
    parser.add_argument(
        "--expected-model",
        required=True,
        help="Assert the automatically resolved generation model; does not select a model",
    )
    parser.add_argument(
        "--candidate-version",
        choices=(
            "1.4.0",
            "1.5.0",
            "1.6.0",
            "1.7.0",
            "1.8.0",
            "1.9.0",
            "1.10.0",
            "1.11.0",
            "1.12.0",
            "1.13.0",
            "1.14.0",
            "1.15.0",
        ),
        default="1.4.0",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--reasoning-effort", choices=("high", "none"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.split:
        cases = [case for case in cases if case.split == args.split]
        if not cases:
            parser.error("selected split is empty")
    assert_litellm_unpatched()
    client = LiteLLMClient(
        LiteLLMConfig(
            model=args.expected_model,
            temperature=0.7,
            fallback_models=[],
            max_retries=0,
        )
    )
    from reflexio.server.llm.model_defaults import ModelRole

    actual_model = client._resolve_primary_model(None, ModelRole.GENERATION)
    if actual_model != args.expected_model:
        parser.error(
            f"Generation resolves to {actual_model}, not expected {args.expected_model}; "
            "--expected-model checks role resolution and does not select a model"
        )
    report = evaluate(
        cases,
        client,
        args.repeats,
        args.out,
        candidate_version=args.candidate_version,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    _write_private_text(args.out, json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    return 1 if report["errors"] or report["measurement_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
