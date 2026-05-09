from __future__ import annotations

import json
import logging

from reflexio.models.api_schema.domain import AgentPlaybook
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.service_utils import log_model_response

from .models import JudgeOutput, RolloutTrace, ScenarioWindow

logger = logging.getLogger(__name__)

PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID = "playbook_optimizer_judge"


class PairwiseJudge:
    """LLM-based comparator for paired playbook rollouts.

    Given two rollouts that share the same user turns and differ only in
    the playbook injected into the assistant, ``judge`` asks an LLM (using
    the ``playbook_optimizer_judge`` prompt) to pick a winner and assign a
    score, Likert rating, and structured rationale.

    If the two playbooks have identical content, ``judge`` short-circuits
    to a tie without spending an LLM call — the rollouts will be identical
    by construction.
    """

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        model_name: str | None,
    ) -> None:
        self.request_context = request_context
        self.llm_client = llm_client
        self.model_name = model_name or llm_client.config.model

    def judge(
        self,
        *,
        window: ScenarioWindow,
        incumbent: AgentPlaybook,
        candidate: AgentPlaybook,
        incumbent_rollout: RolloutTrace,
        candidate_rollout: RolloutTrace,
    ) -> JudgeOutput:
        if incumbent.content == candidate.content:
            return JudgeOutput(
                verdict="tie",
                score=0.5,
                likert=3,
                rationale="Candidate content is identical to incumbent content.",
            )
        prompt = self.request_context.prompt_manager.render_prompt(
            PLAYBOOK_OPTIMIZER_JUDGE_PROMPT_ID,
            {
                "source_window_json": _json(
                    [interaction.model_dump() for interaction in window.interactions]
                ),
                "incumbent_playbook_json": _json(_playbook_payload(incumbent)),
                "candidate_playbook_json": _json(_playbook_payload(candidate)),
                "incumbent_rollout_json": incumbent_rollout.model_dump_json(),
                "candidate_rollout_json": candidate_rollout.model_dump_json(),
            },
        )
        response = self.llm_client.generate_chat_response(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            response_format=JudgeOutput,
            timeout=120,
            max_retries=1,
        )
        log_model_response(logger, "Playbook optimizer judge response", response)
        if isinstance(response, JudgeOutput):
            return response
        return JudgeOutput(
            verdict="tie",
            score=0.5,
            likert=3,
            rationale=f"Judge response was not parsed: {type(response).__name__}",
        )


def _playbook_payload(playbook: AgentPlaybook) -> dict[str, object]:
    return {
        "id": playbook.agent_playbook_id,
        "content": playbook.content,
        "trigger": playbook.trigger,
        "rationale": playbook.rationale,
    }


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
