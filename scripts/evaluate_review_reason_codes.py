"""Paired reason-code evaluation using frozen local cases and the real reviewer.

No database access. Cases and settings are identical for both prompt versions.
Errors are recorded separately and make the command fail. Never infer production
quality from the synthetic cases alone; see developer.md for the rollout gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

DEFAULT_CASES = (
    Path(__file__).resolve().parents[1] / "tests/test_data/reviewer_reason_codes.json"
)


class Turn(BaseModel):
    role: str
    content: str


class Candidate(BaseModel):
    content: str
    trigger: str
    rationale: str
    evidence: list[int] = Field(min_length=1)
    decision: Literal["accept", "revise", "reject"]
    reason_code: PlaybookReviewReasonCode


class Case(BaseModel):
    id: str
    provenance: str
    agent_context: str
    turns: list[Turn] = Field(min_length=1)
    candidates: list[Candidate] = Field(min_length=1)
    existing: list[str] = Field(default_factory=list)


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
) -> dict:
    if candidate_version not in ("1.4.0", "1.5.0", "1.6.0"):
        raise ValueError("candidate_version must be 1.4.0, 1.5.0 or 1.6.0")
    if not cases:
        raise ValueError("cases must be nonempty")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    prompt_identities = {
        version: PromptManager(
            version_override={"playbook_candidate_review": version}
        ).get_prompt_template_identity("playbook_candidate_review")
        for version in ("1.3.0", candidate_version)
    }
    prepared = {case.id: prepare_case(case) for case in cases}
    rows = []
    errors = []
    case_hash = hashlib.sha256(
        json.dumps(
            [case.model_dump(mode="json") for case in cases], sort_keys=True
        ).encode()
    ).hexdigest()

    def save_progress() -> None:
        if checkpoint is not None:
            checkpoint.write_text(
                json.dumps(
                    {
                        "complete": False,
                        "model": client.config.model,
                        "cases_sha256": case_hash,
                        "candidate_version": candidate_version,
                        "prompt_identities": prompt_identities,
                        "rows": rows,
                        "errors": errors,
                    },
                    indent=2,
                )
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
                reviewer = PlaybookCandidateReviewer(
                    request_context=context, llm_client=client
                )
                try:
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
                    save_progress()
                    continue
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
                            "revision": (
                                decision.revision.model_dump()
                                if decision.revision is not None
                                else None
                            ),
                            "decision_correct": decision.decision == target.decision,
                            "code_correct": decision.reason_code == target.reason_code,
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
            "code_correct": sum(row["code_correct"] for row in arm),
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--expected-model",
        required=True,
        help="Assert the automatically resolved generation model; does not select a model",
    )
    parser.add_argument(
        "--candidate-version", choices=("1.4.0", "1.5.0", "1.6.0"), default="1.4.0"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    assert_litellm_unpatched()
    client = LiteLLMClient(
        LiteLLMConfig(model=args.expected_model, temperature=0.7, fallback_models=[])
    )
    from reflexio.server.llm.model_defaults import ModelRole

    actual_model = client._resolve_primary_model(None, ModelRole.GENERATION)
    if actual_model != args.expected_model:
        parser.error(
            f"Generation resolves to {actual_model}, not expected {args.expected_model}; "
            "--expected-model checks role resolution and does not select a model"
        )
    report = evaluate(
        cases, client, args.repeats, args.out, candidate_version=args.candidate_version
    )
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
