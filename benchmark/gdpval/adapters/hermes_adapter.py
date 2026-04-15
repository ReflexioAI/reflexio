"""Hermes host adapter.

Hermes's `AIAgent.run_conversation()` is synchronous — we wrap it in
`asyncio.to_thread` so the benchmark runner can stay async. Hermes points at
its memory / skills / config under `HERMES_HOME` (default `~/.hermes`), so we
isolate per-phase state by setting `HERMES_HOME` inside
`initialize(host_state_dir)` before importing `run_agent.AIAgent`. Memory is
enabled via a minimal `config.yaml` dropped into `HERMES_HOME`.

Reflexio memory is injected via the `system_message=` argument to
`run_conversation()`. Hermes's prompt builder slots it in between tool
guidance and native memory blocks (`run_agent.py:3121-3122`).

LLM calls use the native OpenAI SDK pointed at OpenRouter (via `base_url`
and `api_key`), so tokens come straight from Hermes's result dict — no
litellm callback needed. Both the OpenSpace and Hermes adapters produce the
same `TokenStats` dataclass.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from benchmark.gdpval.adapters.base import AgentResult, HostAgentAdapter
from benchmark.gdpval.config import ensure_hermes_importable
from benchmark.gdpval.tokens import TokenStats, stats_from_hermes_result

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Maps a model-name prefix or substring → (env var, default base_url) for
# direct provider routing when OpenRouter isn't configured. Hermes's own
# auto-detect logic handles most cases; this is just a fallback hint.
#
# MiniMax: default to the international endpoint (`api.minimax.io`), matching
# litellm's default. Override with `MINIMAX_API_BASE` for the China endpoint.
_DIRECT_PROVIDER_HINTS: tuple[tuple[str, str, str], ...] = (
    ("minimax/", "MINIMAX_API_KEY", "https://api.minimax.io/v1"),
)

_HERMES_CONFIG_YAML = """\
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  nudge_interval: 10
  flush_min_turns: 6
"""


def _copy_tree(src: Path, dest: Path) -> None:
    """Copy `src` onto `dest`, replacing `dest` if it exists."""
    if dest.exists():
        shutil.rmtree(dest)
    if src.exists():
        shutil.copytree(src, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)


class HermesAdapter(HostAgentAdapter):
    """Drives the Hermes agent through the benchmark's phase protocol."""

    name = "hermes"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        max_iterations: int = 30,
    ) -> None:
        """
        Args:
            model (str): Model identifier (e.g. "minimax/MiniMax-M2.7"). Accepts
                an optional litellm-style `openrouter/` prefix which is stripped
                before it reaches Hermes.
            api_key (str | None): Bearer token. If None, resolution order is:
                (1) `OPENROUTER_API_KEY` when OpenRouter is the implicit target,
                (2) provider-specific env var via `_DIRECT_PROVIDER_HINTS` when
                    the model prefix matches a direct-provider endpoint,
                (3) hand off to Hermes's own auto-detect by leaving it None.
            base_url (str | None): API base URL. If None, selected to match the
                resolved `api_key` source, or falls through to Hermes auto-detect.
            provider (str | None): Provider label Hermes records for usage
                bookkeeping. If None, inferred from the base URL.
            max_iterations (int): Per-task tool-calling iteration cap.
        """
        self._model = self._strip_litellm_prefix(model)
        self._api_key, self._base_url, self._provider = self._resolve_credentials(
            model=self._model,
            explicit_api_key=api_key,
            explicit_base_url=base_url,
            explicit_provider=provider,
        )
        self._max_iterations = max_iterations
        self._agent: Any = None
        self._hermes_home: Path | None = None

    @staticmethod
    def _resolve_credentials(
        model: str,
        explicit_api_key: str | None,
        explicit_base_url: str | None,
        explicit_provider: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        """Pick the right (api_key, base_url, provider) triple for this model.

        Args:
            model (str): Bare model identifier.
            explicit_api_key (str | None): Override — takes precedence over env lookup.
            explicit_base_url (str | None): Override — takes precedence over defaults.
            explicit_provider (str | None): Override — takes precedence over inference.

        Returns:
            tuple[str | None, str | None, str | None]: Resolved credentials.
        """
        if explicit_api_key:
            return explicit_api_key, explicit_base_url, explicit_provider

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            return (
                openrouter_key,
                explicit_base_url or _OPENROUTER_BASE_URL,
                explicit_provider or "openrouter",
            )

        for prefix, env_var, default_base_url in _DIRECT_PROVIDER_HINTS:
            if model.startswith(prefix):
                key = os.environ.get(env_var)
                if key:
                    # Honor the provider-specific base-url env var (e.g.
                    # MINIMAX_API_BASE) for users on the China endpoint —
                    # same convention litellm uses.
                    base_url_env = os.environ.get(f"{env_var.replace('_API_KEY', '_API_BASE')}")
                    resolved_base = explicit_base_url or base_url_env or default_base_url
                    return key, resolved_base, explicit_provider or prefix.rstrip("/")

        # Hand off to Hermes's native auto-detect.
        return None, explicit_base_url, explicit_provider

    async def initialize(self, host_state_dir: Path) -> None:
        """Set up HERMES_HOME under `host_state_dir`.

        The actual `AIAgent` instance is constructed fresh in each `run()`
        call, NOT here. Reason: Hermes's AIAgent caches its system prompt
        (`_cached_system_prompt`) on first use and reuses it across
        subsequent `run_conversation` calls. When the first task is
        custom-003 (a 20-iter ML build), the cached prompt explodes to
        ~147k tokens worth of loaded skills + tool-definition context, and
        every subsequent task on the same agent pays that inflated
        baseline — even for a 1-iteration response. We saw exactly this
        in the `hermes_mini_baseline_5` run: custom-001/002 at ~33k
        prompt, custom-003 at 147k, then custom-004/005 at ~147k despite
        only 1 iteration each.

        Creating a fresh `AIAgent` per task prevents the cached prompt
        pollution and gives reflexio a clean baseline to improve against.

        Args:
            host_state_dir (Path): Directory that holds this phase's Hermes
                state (`hermes_home/` subdirectory with MEMORY.md, skills, etc.).
        """
        host_state_dir.mkdir(parents=True, exist_ok=True)
        hermes_home = host_state_dir / "hermes_home"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "memories").mkdir(parents=True, exist_ok=True)
        (hermes_home / "skills").mkdir(parents=True, exist_ok=True)

        config_path = hermes_home / "config.yaml"
        if not config_path.exists():
            config_path.write_text(_HERMES_CONFIG_YAML)

        os.environ["HERMES_HOME"] = str(hermes_home)
        self._hermes_home = hermes_home

        ensure_hermes_importable()
        # Validate that the import works and record the AIAgent class
        # so we don't pay the import cost on every run().
        from run_agent import AIAgent  # noqa: E402
        self._ai_agent_cls = AIAgent
        # Kept for API compatibility; set to None until first run() call.
        self._agent = None

    def _build_fresh_agent(self) -> Any:
        """Construct a fresh AIAgent instance.

        Called once per `run()` to prevent cross-task state pollution
        (see `initialize` docstring for the 147k-token accumulation bug).
        """
        return self._ai_agent_cls(
            base_url=self._base_url,
            api_key=self._api_key,
            provider=self._provider,
            model=self._model,
            max_iterations=self._max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            save_trajectories=False,
            skip_memory=False,
        )

    async def run(
        self,
        task: dict[str, Any],
        workspace: Path,
        memory: str | None,
    ) -> AgentResult:
        """Execute one task through Hermes.

        The reflexio playbook block is injected via `system_message=`. We set
        `TERMINAL_CWD` to the workspace so Hermes's file tools write their
        deliverables where the evaluator can find them.

        Args:
            task (dict[str, Any]): Normalized GDPVal task dict.
            workspace (Path): Per-task workspace for deliverables.
            memory (str | None): Optional reflexio playbook block.

        Returns:
            AgentResult: Normalized result record.
        """
        if self._ai_agent_cls is None:
            raise RuntimeError("HermesAdapter.initialize() not called")

        # Fresh AIAgent per task to prevent cross-task prompt cache pollution
        # (see initialize() docstring).
        self._agent = self._build_fresh_agent()

        workspace.mkdir(parents=True, exist_ok=True)
        os.environ["TERMINAL_CWD"] = str(workspace)
        # Hermes's file_operations.ShellFileOperations caches its cwd at
        # tool-init time (which happened in our initialize(), before any
        # task-specific workspace existed). Setting TERMINAL_CWD alone is
        # not enough — write_file et al. still resolve relative paths
        # against the cached cwd. Force the process cwd here so
        # os.getcwd() fallbacks inside Hermes land in the task workspace.
        # Safe because the benchmark runs tasks serially; a concurrent
        # variant would need per-task isolation instead of chdir.
        os.chdir(workspace)

        from benchmark.gdpval.task_loader import prepare_task_workspace

        augmented_prompt = prepare_task_workspace(task, str(workspace))

        t0 = time.monotonic()
        try:
            result = await asyncio.to_thread(
                self._agent.run_conversation,
                user_message=augmented_prompt,
                system_message=memory,
                task_id=f"{task['task_id']}_{int(t0)}",
            )
            status = self._classify_hermes_status(result)
        except Exception as exc:
            logger.exception("Hermes run_conversation failed for task %s", task.get("task_id"))
            result = {
                "completed": False,
                "partial": False,
                "messages": [],
                "api_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "error": str(exc),
            }
            status = "error"
        finally:
            elapsed = time.monotonic() - t0

        tokens: TokenStats = stats_from_hermes_result(result, elapsed)
        tool_calls = self._count_tool_calls(result.get("messages", []) or [])
        messages = self._flatten_trajectory(task, result)

        return AgentResult(
            status=status,
            iterations=int(result.get("api_calls", 0) or 0),
            tool_calls=tool_calls,
            tokens=tokens,
            artifacts_dir=workspace,
            messages=messages,
            raw={"final_response": result.get("final_response", "")},
        )

    async def snapshot_state(self, dest: Path) -> None:
        """Copy `HERMES_HOME` into `dest/hermes_home/` for later restore.

        Args:
            dest (Path): Directory that will hold the snapshot.
        """
        if self._hermes_home is None:
            raise RuntimeError("Cannot snapshot — adapter was never initialized")
        snap = dest / "hermes_home"
        _copy_tree(self._hermes_home, snap)
        logger.info("Snapshotted Hermes state to %s", snap)

    async def cleanup(self) -> None:
        """Drop the agent reference; Hermes has no explicit teardown hook."""
        self._agent = None

    @staticmethod
    def _classify_hermes_status(result: dict[str, Any]) -> str:
        """Map Hermes's terminal-state flags to a stable benchmark status label.

        Hermes returns three booleans that encode how `run_conversation()`
        exited (see `run_agent.py:10202-10210`). We collapse them into a
        labeled taxonomy so downstream analysis can decide which runs to
        include in means:

        * ``success`` — agent reached a natural end (`completed=True`).
        * ``invalid_tool`` — stopped after retry-exhausted invalid tool
          calls (Hermes's own `partial=True`). Broken measurement — the
          agent didn't get to finish, and its token counts reflect wasted
          retries. Exclude from headline means.
        * ``interrupted`` — user interrupt (`interrupted=True`). Excluded.
        * ``failed`` — API failure recorded via Hermes's own `failed=True`
          flag. Excluded.
        * ``budget_exhausted`` — iteration cap was hit, Hermes asked the
          model to produce a summary, and the loop exited. The agent DID
          produce work (deliverables are on disk) — this is a legitimate
          measurement of "agent cost at the 30-iter budget" and should be
          INCLUDED in headline means. Flagged separately so we can report
          the budget-hit rate per cell.

        Args:
            result (dict[str, Any]): Raw `run_conversation()` result dict.

        Returns:
            str: One of "success", "invalid_tool", "interrupted", "failed",
                "budget_exhausted".
        """
        if result.get("completed"):
            return "success"
        if result.get("failed"):
            return "failed"
        if result.get("partial"):
            return "invalid_tool"
        if result.get("interrupted"):
            return "interrupted"
        return "budget_exhausted"

    @staticmethod
    def _strip_litellm_prefix(model: str) -> str:
        """Strip litellm-style provider prefixes from a model string.

        Hermes uses the bare OpenRouter ID; litellm uses `"openrouter/..."`.
        We accept either form in config and feed Hermes the bare tail.

        Args:
            model (str): Possibly prefixed model ID.

        Returns:
            str: Bare model ID suitable for Hermes/OpenRouter.
        """
        if model.startswith("openrouter/"):
            return model[len("openrouter/"):]
        return model

    @staticmethod
    def _count_tool_calls(messages: list[dict[str, Any]]) -> int:
        """Count tool_calls across all assistant messages in a Hermes trajectory."""
        total = 0
        for msg in messages:
            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
            if tool_calls:
                total += len(tool_calls)
        return total

    @staticmethod
    def _flatten_trajectory(
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Render Hermes's message log as a [{role, content}] list for reflexio.

        Each OpenAI-style turn is normalized into User / Assistant content
        strings that reflexio's extractors can actually learn from:

        * ``user`` messages pass through as User turns (trimmed of surrogates).
        * ``assistant`` messages are expanded: plain text becomes the body,
          any ``tool_calls`` are appended as `Called <tool>(<args>)` lines so
          the extractor sees "I called tool X with arguments Y" not "content=None".
        * ``tool`` messages (tool results) are attached as continuation
          Assistant turns — labelled so the extractor can distinguish "tool
          said Z" from "agent said Z". Tool result content is truncated to
          keep individual interactions under reflexio's storage limits.

        This richer shape is the raw material reflexio's `user_playbook_extractor`
        needs to discover both success patterns ("I called pdf_extract, got
        page count, split by section") and issue→fix pairs ("run_shell raised
        ModuleNotFoundError, installed pyxlsb, retried").

        Args:
            task (dict[str, Any]): Original task (for fallback User turn).
            result (dict[str, Any]): Raw Hermes run_conversation result.

        Returns:
            list[dict[str, Any]]: Flattened conversation with tool context.
        """
        flat: list[dict[str, Any]] = []
        seen_user = False
        for msg in result.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                text = content if isinstance(content, str) else str(content or "")
                if text.strip():
                    flat.append({"role": "User", "content": text})
                    seen_user = True
            elif role == "assistant":
                rendered = HermesAdapter._render_assistant_turn(msg)
                if rendered:
                    flat.append({"role": "Assistant", "content": rendered})
            elif role == "tool":
                tool_name = msg.get("name") or "tool"
                tool_content = content if isinstance(content, str) else str(content or "")
                if not tool_content.strip():
                    continue
                truncated = tool_content[:2000]
                if len(tool_content) > 2000:
                    truncated += f"... [truncated, total {len(tool_content)} chars]"
                flat.append({
                    "role": "Assistant",
                    "content": f"[tool {tool_name} result]\n{truncated}",
                })

        if not seen_user:
            flat.insert(0, {"role": "User", "content": task.get("prompt", "") or ""})
        if not any(m["role"] == "Assistant" for m in flat):
            final = result.get("final_response") or result.get("error") or ""
            flat.append({"role": "Assistant", "content": str(final)})
        return flat

    @staticmethod
    def _render_assistant_turn(msg: dict[str, Any]) -> str:
        """Render one OpenAI-style assistant message to a plain text string.

        Combines the message's text content (if any) with a compact
        description of each tool call it issued. If neither is present,
        returns "" so the flattener can skip it.

        Args:
            msg (dict[str, Any]): Raw assistant message from Hermes result.

        Returns:
            str: Rendered content, possibly multi-line.
        """
        parts: list[str] = []
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)

        tool_calls = msg.get("tool_calls") or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            name = fn.get("name") or call.get("name") or "tool"
            args = fn.get("arguments") or call.get("arguments") or ""
            if isinstance(args, (dict, list)):
                try:
                    args = json.dumps(args)
                except (TypeError, ValueError):
                    args = str(args)
            args_str = str(args)[:500]
            if len(str(args)) > 500:
                args_str += "..."
            parts.append(f"Called {name}({args_str})")

        return "\n".join(parts).strip()
