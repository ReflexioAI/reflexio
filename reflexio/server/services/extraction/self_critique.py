"""Self-critique pass — reads the conversation against the parallel extraction
plans and adds missing operand-bearing user-side facts.

Complements UnifyAgent (which is subtractive — drops cross-axis duplicates).
SelfCritiqueAgent is additive — finds operands the parallel extractors
missed and emits new CreateUserProfile ops for them.

Design constraints:
- Single LLM call (no tool loop). Bounded cost.
- Operand-only emissions — only adds facts with a numeric/date/named-place
  operand. Drops the temptation to emit topic-only facts that would bloat
  storage with low-information profiles.
- Targets the UserProfile axis only. Keeps the implementation contained
  and avoids cross-axis decisions.
- Defensive fallback: any LLM error or parse failure returns no additions.
"""

from __future__ import annotations

import logging
import re

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.extraction.extraction_agent import (
    DeferredExtractionRun,
)
from reflexio.server.services.extraction.plan import (
    CreateUserPlaybookOp,
    CreateUserProfileOp,
)
from reflexio.server.services.service_utils import log_model_response

logger = logging.getLogger(__name__)

_PASS_LABEL_BY_KIND: dict[str, str] = {
    "UserProfile": "A",
    "UserProfileAgentRec": "B",
    "UserPlaybook": "C",
}

_ADD_RE = re.compile(r"^\s*ADD:\s*(.+?)\s*$", re.MULTILINE)


class SelfCritiqueAgent:
    """Single-shot operand-recovery pass over the parallel extraction outputs.

    Reads the original conversation alongside the three deferred plans,
    asks the LLM what operand-bearing user facts were missed, parses
    ``ADD: <content>`` lines and appends the new ops to the
    UserProfile-axis plan.

    Args:
        client: LLM client.
        prompt_manager: Renders the ``extraction_self_critique`` prompt.
        timeout: Per-call timeout in seconds. Conservative because the
            critique sees a full transcript (potentially 30 KB).
        max_retries: Retries on transient LLM errors. 1 because the
            fallback (add nothing) is safe.
    """

    LLM_TIMEOUT = 30
    LLM_MAX_RETRIES = 1

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
    ) -> None:
        self.client = client
        self.prompt_manager = prompt_manager

    def run(
        self,
        deferred_runs: list[DeferredExtractionRun],
        *,
        transcript: str,
    ) -> int:
        """Add missing operand-bearing facts to the UserProfile plan.

        Returns the number of ops added (0 on fallback, no misses, or
        no UserProfile pass available).
        """
        target = self._find_user_profile_run(deferred_runs)
        if target is None:
            return 0
        try:
            response = self._call_llm(deferred_runs, transcript)
        except Exception as e:  # noqa: BLE001 — keep all on failure
            logger.warning(
                "self_critique llm_failed: %s: %s", type(e).__name__, e
            )
            return 0
        new_contents = self._parse_additions(response)
        if not new_contents:
            return 0
        added = 0
        for content in new_contents:
            try:
                op = CreateUserProfileOp(
                    content=content,
                    ttl="one_year",
                    source_span=content[:120] or "self-critique",
                )
            except Exception as e:  # noqa: BLE001 — defensive parse
                logger.debug("self_critique skipped malformed addition: %s", e)
                continue
            target.ctx.plan.append(op)
            added += 1
        if added:
            logger.info(
                "self_critique kind=UserProfile added=%d before=%d after=%d",
                added,
                len(target.ctx.plan) - added,
                len(target.ctx.plan),
            )
        return added

    @staticmethod
    def _find_user_profile_run(
        runs: list[DeferredExtractionRun],
    ) -> DeferredExtractionRun | None:
        for run in runs:
            if run.kind == "UserProfile":
                return run
        return None

    def _call_llm(
        self,
        runs: list[DeferredExtractionRun],
        transcript: str,
    ) -> str:
        rendered: dict[str, str] = {
            "transcript": transcript,
        }
        for label in ("A", "B", "C"):
            rendered[f"pass_{label.lower()}_ops"] = "(no proposals)"
        for run in runs:
            label = _PASS_LABEL_BY_KIND.get(run.kind)
            if label is None:
                continue
            rendered[f"pass_{label.lower()}_ops"] = self._render_ops(run)
        prompt = self.prompt_manager.render_prompt(
            "extraction_self_critique", rendered
        )
        result = self.client.generate_response(
            prompt,
            timeout=self.LLM_TIMEOUT,
            max_retries=self.LLM_MAX_RETRIES,
            model_role=ModelRole.EXTRACTION_AGENT,
        )
        log_model_response(logger, "self_critique response", result)
        if not isinstance(result, str):
            return ""
        return result

    @staticmethod
    def _render_ops(run: DeferredExtractionRun) -> str:
        if not run.ctx.plan:
            return "(no proposals)"
        lines: list[str] = []
        for i, op in enumerate(run.ctx.plan):
            if isinstance(op, CreateUserProfileOp):
                lines.append(f"  [{i}] CREATE profile: {op.content}")
            elif isinstance(op, CreateUserPlaybookOp):
                lines.append(
                    f"  [{i}] CREATE playbook: trigger={op.trigger!r} content={op.content!r}"
                )
            else:
                lines.append(f"  [{i}] {op.op}: id={getattr(op, 'id', '?')}")
        return "\n".join(lines)

    @staticmethod
    def _parse_additions(response: str) -> list[str]:
        if "KEEP ALL CURRENT" in response.upper():
            return []
        out: list[str] = []
        for match in _ADD_RE.finditer(response):
            content = match.group(1).strip()
            if content and len(content) >= 5:
                out.append(content)
        return out
