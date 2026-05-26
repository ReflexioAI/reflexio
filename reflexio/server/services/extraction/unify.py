"""Cross-axis unification for parallel extraction passes.

The parallel-then-unify pipeline runs the three extraction axes
(UserProfile, UserProfileAgentRec, UserPlaybook) concurrently with each
agent's commit deferred. Because the passes run in parallel, none of them
sees the others' proposed ops during its own search/dedup loop. ``UnifyAgent``
inspects the merged output and drops cross-axis duplicates before any
storage write.

Design constraints:
- Single LLM call (no tool loop). Cheap and bounded.
- Drop-only — never invents new ops or rewrites contents.
- Conservative: when ambiguous, keep both (false-drops cost more than
  false-keeps because the deduper can be re-run, but information loss
  is permanent).
- Defensive fallback: any parsing or LLM failure leaves all ops intact.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

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
_KIND_BY_PASS_LABEL: dict[str, str] = {v: k for k, v in _PASS_LABEL_BY_KIND.items()}

_DROP_RE = re.compile(r"^\s*DROP\s+([ABC])\s*[.\s]\s*(\d+)\s*$", re.MULTILINE)


@dataclass(slots=True)
class UnifyDecision:
    pass_label: Literal["A", "B", "C"]
    index: int


class UnifyAgent:
    """Single-shot dedup pass over the three parallel extraction outputs.

    Renders ``extraction_unify`` with the three ops lists, asks the LLM
    which ops are cross-axis duplicates, and mutates each
    ``DeferredExtractionRun.ctx.plan`` in place to drop them.

    Args:
        client: LLM client (uses ``ModelRole.EXTRACTION_AGENT``).
        prompt_manager: Renders the unify prompt.
        timeout: Per-call timeout in seconds. 8 s is enough headroom for
            a few-hundred-tokens prompt on gpt-5-mini.
        max_retries: Retries on transient LLM errors. 1 because the
            fallback (keep everything) is a clean no-op.
    """

    LLM_TIMEOUT = 8
    LLM_MAX_RETRIES = 1

    def __init__(
        self,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
    ) -> None:
        self.client = client
        self.prompt_manager = prompt_manager

    def run(self, deferred_runs: list[DeferredExtractionRun]) -> int:
        """Drop cross-axis duplicates from the supplied deferred runs.

        Mutates each run's ``ctx.plan`` in place. Returns the number of
        ops dropped (0 on fallback or no duplicates).
        """
        if len(deferred_runs) <= 1:
            return 0
        runs_by_label = self._index_by_pass_label(deferred_runs)
        if not runs_by_label:
            return 0
        try:
            response = self._call_llm(runs_by_label)
        except Exception as e:  # noqa: BLE001 — any failure: keep everything
            logger.warning("unify_agent llm_failed: %s: %s", type(e).__name__, e)
            return 0
        decisions = self._parse_decisions(response)
        if not decisions:
            return 0
        return self._apply_drops(decisions, runs_by_label)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _index_by_pass_label(
        runs: list[DeferredExtractionRun],
    ) -> dict[str, DeferredExtractionRun]:
        out: dict[str, DeferredExtractionRun] = {}
        for run in runs:
            label = _PASS_LABEL_BY_KIND.get(run.kind)
            if label is None:
                logger.debug("unify_agent skipping unknown kind=%s", run.kind)
                continue
            out[label] = run
        return out

    def _call_llm(self, runs_by_label: dict[str, DeferredExtractionRun]) -> str:
        rendered_blocks = {
            f"pass_{label.lower()}_ops": self._render_ops(run)
            for label, run in runs_by_label.items()
        }
        # Missing passes get a "(no proposals)" placeholder so the prompt
        # stays well-formed even if one axis produced nothing.
        for label in ("A", "B", "C"):
            key = f"pass_{label.lower()}_ops"
            rendered_blocks.setdefault(key, "(no proposals)")
        prompt = self.prompt_manager.render_prompt("extraction_unify", rendered_blocks)
        result = self.client.generate_response(
            prompt,
            timeout=self.LLM_TIMEOUT,
            max_retries=self.LLM_MAX_RETRIES,
            model_role=ModelRole.EXTRACTION_AGENT,
        )
        log_model_response(logger, "unify_agent response", result)
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
                # delete ops are never dropped — just listed for context
                lines.append(f"  [{i}] {op.op}: id={getattr(op, 'id', '?')}")
        return "\n".join(lines)

    @staticmethod
    def _parse_decisions(response: str) -> list[UnifyDecision]:
        decisions: list[UnifyDecision] = []
        for match in _DROP_RE.finditer(response):
            label = match.group(1)
            try:
                idx = int(match.group(2))
            except ValueError:
                continue
            decisions.append(UnifyDecision(pass_label=label, index=idx))  # type: ignore[arg-type]
        return decisions

    @staticmethod
    def _apply_drops(
        decisions: list[UnifyDecision],
        runs_by_label: dict[str, DeferredExtractionRun],
    ) -> int:
        # Group indices to drop per pass; sort descending so popping doesn't
        # invalidate earlier indices.
        drops_by_label: dict[str, set[int]] = {}
        for d in decisions:
            drops_by_label.setdefault(d.pass_label, set()).add(d.index)

        total_dropped = 0
        for label, drop_indices in drops_by_label.items():
            run = runs_by_label.get(label)
            if run is None:
                continue
            survivors = [
                op
                for i, op in enumerate(run.ctx.plan)
                if i not in drop_indices
                or not _is_droppable(op)  # never drop delete ops
            ]
            dropped = len(run.ctx.plan) - len(survivors)
            if dropped:
                logger.info(
                    "unify_agent kind=%s dropped=%d before=%d after=%d",
                    run.kind,
                    dropped,
                    len(run.ctx.plan),
                    len(survivors),
                )
                run.ctx.plan = survivors
                total_dropped += dropped
        return total_dropped


def _is_droppable(op: object) -> bool:
    """True for CREATE ops; False for delete ops (never drop deletes)."""
    return isinstance(op, CreateUserProfileOp | CreateUserPlaybookOp)
