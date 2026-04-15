"""OpenSpace host adapter.

OpenSpace persists its skill store in `{OPENSPACE_ROOT}/.openspace/` (a sqlite db
plus embedding cache) and its skill files in `{OPENSPACE_ROOT}/gdpval_bench/skills/`.
SkillStore's db path is derived from a hardcoded `PROJECT_ROOT` inside OpenSpace's
source, so there's no env var we can flip to redirect it.

To support the benchmark's P1 → snapshot → P2/P3 protocol we manage those two
directories externally — copy them *out* to our snapshot dir after P1 finishes,
and copy them *back in* before P3 starts. P2 is a straight continuation of P1's
accumulated state; we do not reset between P1 and P2.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from benchmark.gdpval.adapters.base import AgentResult, HostAgentAdapter
from benchmark.gdpval.config import (
    OPENSPACE_BASELINE_DIR,
    OPENSPACE_ROOT,
    ensure_openspace_importable,
)
from benchmark.gdpval.tokens import (
    TokenStats,
    TokenTracker,
    stats_from_openspace,
)

logger = logging.getLogger(__name__)

# OpenSpace's two persistent state locations — both live inside the OpenSpace
# source tree because SkillStore resolves paths relative to its own __file__.
_OS_DB_DIR = OPENSPACE_ROOT / ".openspace"
_OS_SKILLS_DIR = OPENSPACE_ROOT / "gdpval_bench" / "skills"

# Process-wide single TokenTracker, installed at module import and left
# active for the process lifetime. Every OpenSpaceAdapter instance reuses
# it via `self._tracker = _SHARED_TRACKER`.
#
# Evolution of this pattern:
#
# 1. First attempt: fresh TokenTracker per adapter, start()/stop() per
#    task. Step E surfaced that fresh-per-phase trackers yielded 0 tokens
#    for P2/P3 despite P1 working fine.
#
# 2. Shared module-level tracker with start()/stop() per task. Fixed the
#    P2/P3=0 bug but only works under serial execution: two concurrent
#    tasks would each call start() (which calls stats.reset()), wiping
#    each other's in-flight counts, and each task's stop() would read a
#    polluted mix. Step E rerun with --concurrency 5 hit this instantly
#    — task 83d10b06 reported 679k tokens vs a baseline ~282k because
#    the shared stats bucket was carrying other concurrent tasks' tokens.
#
# 3. Current: shared tracker in CONCURRENT MODE. `install()` is called
#    once at module load (below) to register the litellm callback. Each
#    adapter.run() calls `begin_task(task_key)` to open a per-task stats
#    bucket routed via contextvars, and `end_task(task_key, ctx)` at the
#    end to read the bucket out. gdpval_bench uses exactly this pattern
#    in its own concurrent runner at
#    OpenSpace/gdpval_bench/run_benchmark.py:667, 461-462, 492. Serial
#    execution (concurrency=1) works as a degenerate case of the same
#    code path — one task at a time, contextvar routes to the sole
#    active bucket.
_SHARED_TRACKER = TokenTracker()
_SHARED_TRACKER.install()


def _copy_tree(src: Path, dest: Path) -> None:
    """Copy `src` onto `dest`, replacing `dest` if it exists."""
    if dest.exists():
        shutil.rmtree(dest)
    if src.exists():
        shutil.copytree(src, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)


def _capture_baseline_if_missing(baseline_dir: Path) -> None:
    """Snapshot OpenSpace's current global state to `baseline_dir` if empty.

    Runs once per benchmark install. Subsequent P1 runs all restore from this
    same baseline so "cold" is reproducible across runs on the same machine.
    The baseline captures whatever pre-existing `.openspace/` and skills the
    user's OpenSpace checkout happens to have; users who want a truly empty
    baseline should clear those dirs manually before the first benchmark run.

    Args:
        baseline_dir (Path): Where to write the baseline snapshot.
    """
    if baseline_dir.exists() and any(baseline_dir.iterdir()):
        return
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _copy_tree(_OS_DB_DIR, baseline_dir / ".openspace")
    _copy_tree(_OS_SKILLS_DIR, baseline_dir / "skills")
    logger.info("Captured OpenSpace baseline to %s", baseline_dir)


def _restore_baseline(baseline_dir: Path) -> None:
    """Copy the captured baseline back into OpenSpace's global state dirs.

    Called at the start of every P1 so each cold run starts from the same
    deterministic state. No-op if the baseline doesn't exist yet (first-ever
    benchmark run).

    Args:
        baseline_dir (Path): Source baseline snapshot.
    """
    if not baseline_dir.exists():
        return
    _copy_tree(baseline_dir / ".openspace", _OS_DB_DIR)
    _copy_tree(baseline_dir / "skills", _OS_SKILLS_DIR)
    logger.info("Restored OpenSpace baseline from %s", baseline_dir)


class OpenSpaceAdapter(HostAgentAdapter):
    """Drives OpenSpace through the benchmark's phase protocol.

    Shares litellm token tracking (`gdpval_bench.token_tracker.TokenTracker`)
    with the shared `tokens.py` module — same TokenStats path as gdpval_bench.
    """

    name = "openspace"

    def __init__(self, model: str, max_iterations: int = 30) -> None:
        """
        Args:
            model (str): litellm model string (e.g. "openrouter/minimax/MiniMax-M2.7").
            max_iterations (int): Per-task tool-calling iteration cap.
        """
        self._model = model
        self._max_iterations = max_iterations
        self._cs: Any = None
        self._tracker = _SHARED_TRACKER  # see module-level _SHARED_TRACKER comment
        self._workspace_root: Path | None = None

    async def initialize(self, host_state_dir: Path) -> None:
        """Bind OpenSpace to an isolated state directory.

        `host_state_dir` is used as both the staging area for P2/P3's warm-state
        restore and the workspace root for per-task deliverables. On first call
        (P1), `host_state_dir` is empty and OpenSpace starts cold. On P2/P3,
        the caller has populated `host_state_dir/snapshot/` with the post-P1
        snapshot, which we restore into OpenSpace's global dirs before
        constructing the OpenSpace instance.

        Args:
            host_state_dir (Path): Directory holding this phase's OpenSpace state.
        """
        ensure_openspace_importable()
        host_state_dir.mkdir(parents=True, exist_ok=True)
        self._workspace_root = host_state_dir / "workspace"
        self._workspace_root.mkdir(parents=True, exist_ok=True)

        snapshot = host_state_dir / "snapshot"
        if snapshot.exists():
            logger.info("Restoring OpenSpace state from post-P1 snapshot: %s", snapshot)
            _copy_tree(snapshot / ".openspace", _OS_DB_DIR)
            _copy_tree(snapshot / "skills", _OS_SKILLS_DIR)
        else:
            # P1 path — capture baseline on first ever run, then restore from
            # it every P1 so cold state is deterministic across runs.
            _capture_baseline_if_missing(OPENSPACE_BASELINE_DIR)
            _restore_baseline(OPENSPACE_BASELINE_DIR)

        from openspace.tool_layer import OpenSpace, OpenSpaceConfig  # noqa: E402

        recording_dir = host_state_dir / "recordings"
        recording_dir.mkdir(parents=True, exist_ok=True)
        config = OpenSpaceConfig(
            llm_model=self._model,
            workspace_dir=str(self._workspace_root),
            recording_log_dir=str(recording_dir),
            backend_scope=["shell"],
            grounding_max_iterations=self._max_iterations,
            enable_recording=True,
        )
        self._cs = OpenSpace(config=config)
        await self._cs.initialize()

    async def run(
        self,
        task: dict[str, Any],
        workspace: Path,
        memory: str | None,
    ) -> AgentResult:
        """Execute one task through OpenSpace.

        Reflexio memory is prepended to the task prompt (wrapped in
        `<memory>…</memory>` markers) when provided. Reference files are
        downloaded and added to the workspace via gdpval_bench's
        `prepare_task_workspace` helper so the agent sees them exactly like
        it would in gdpval_bench.

        Args:
            task (dict[str, Any]): Normalized GDPVal task dict.
            workspace (Path): Per-task workspace for deliverables.
            memory (str | None): Optional reflexio playbook block (P3 only).

        Returns:
            AgentResult: Normalized result record.
        """
        if self._cs is None:
            raise RuntimeError("OpenSpaceAdapter.initialize() not called")

        from benchmark.gdpval.task_loader import prepare_task_workspace

        workspace.mkdir(parents=True, exist_ok=True)
        augmented_prompt = prepare_task_workspace(task, str(workspace))
        if memory:
            augmented_prompt = (
                f"<memory>\n{memory}\n</memory>\n\n{augmented_prompt}"
            )

        # Concurrent-safe token accounting via contextvars. begin_task
        # opens a per-task stats bucket and sets a ContextVar that the
        # litellm callback reads on each call. Because asyncio tasks
        # inherit their caller's context when spawned via gather, and
        # because await points don't leak context between sibling tasks,
        # each concurrent adapter.run() writes to its own bucket. Serial
        # execution works as a degenerate case of the same path — one
        # bucket alive at a time. end_task returns that bucket's copy,
        # leaving the tracker available for the next task's begin_task.
        task_key = f"{task['task_id']}_{int(time.monotonic() * 1000)}"
        ctx_token = self._tracker.begin_task(task_key)
        t0 = time.monotonic()
        try:
            result = await self._cs.execute(
                task=augmented_prompt,
                task_id=f"{task['task_id']}_{int(t0)}",
                workspace_dir=str(workspace),
            )
            status = result.get("status", "unknown")
        except Exception as exc:
            logger.exception("OpenSpace execute failed for task %s", task.get("task_id"))
            result = {"status": "error", "error": str(exc), "tool_executions": []}
            status = "error"
        finally:
            elapsed = time.monotonic() - t0
            os_stats = self._tracker.end_task(task_key, ctx_token)

        tokens: TokenStats = stats_from_openspace(os_stats)
        tokens.wall_time_sec = round(elapsed, 2)

        messages = self._flatten_trajectory(task, result)

        return AgentResult(
            status=status,
            iterations=int(result.get("iterations", 0)),
            tool_calls=len(result.get("tool_executions", []) or []),
            tokens=tokens,
            artifacts_dir=workspace,
            messages=messages,
            raw={
                "skills_used": result.get("skills_used", []) or [],
                "evolved_skills": result.get("evolved_skills", []) or [],
            },
        )

    async def snapshot_state(self, dest: Path) -> None:
        """Copy OpenSpace's global state into `dest/snapshot/`.

        Args:
            dest (Path): Directory that will hold `snapshot/.openspace/` and
                `snapshot/skills/`.
        """
        snap = dest / "snapshot"
        snap.mkdir(parents=True, exist_ok=True)
        _copy_tree(_OS_DB_DIR, snap / ".openspace")
        _copy_tree(_OS_SKILLS_DIR, snap / "skills")
        logger.info("Snapshotted OpenSpace state to %s", snap)

    async def cleanup(self) -> None:
        """Dispose the OpenSpace instance so a later phase can construct a fresh one."""
        if self._cs is not None:
            cleanup = getattr(self._cs, "cleanup", None)
            if callable(cleanup):
                result = cleanup()
                if hasattr(result, "__await__"):
                    await result
            self._cs = None

    def _flatten_trajectory(
        self,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Render the OpenSpace tool trajectory as a [{role, content}] list.

        Goal: give reflexio's playbook extractor enough raw material to
        learn both successful patterns and issue→fix pairs. Rather than
        compressing the whole run into a single Assistant turn, we emit:

        1. The original task prompt as the User turn.
        2. One Assistant turn per tool execution, each containing the tool
           name, its arguments (trimmed), and its output (trimmed). That
           gives the extractor a per-step view like
           ``Called run_shell(python3 ...) → output: ...``.
        3. A final Assistant turn with OpenSpace's ``final_response`` so
           the extractor sees the agent's own summary / conclusions.

        Arguments and outputs are truncated per-turn (500 chars args,
        2000 chars output) to keep individual interactions under reflexio's
        storage limits while still preserving the gist.

        Args:
            task (dict[str, Any]): Original task dict (for the User turn).
            result (dict[str, Any]): Raw OpenSpace execute() result.

        Returns:
            list[dict[str, Any]]: Flattened conversation turns.
        """
        flat: list[dict[str, Any]] = [
            {"role": "User", "content": task.get("prompt", "")},
        ]
        tool_executions = result.get("tool_executions", []) or []
        for call in tool_executions:
            if not isinstance(call, dict):
                continue
            name = call.get("tool") or call.get("name") or "tool"
            args = call.get("arguments") or call.get("args") or ""
            if isinstance(args, (dict, list)):
                try:
                    args = json.dumps(args)
                except (TypeError, ValueError):
                    args = str(args)
            args_str = str(args)[:500]
            if len(str(args)) > 500:
                args_str += "..."
            out = call.get("output") or call.get("result") or ""
            out_str = str(out)[:2000]
            if len(str(out)) > 2000:
                out_str += f"... [truncated, total {len(str(out))} chars]"
            turn = f"Called {name}({args_str})"
            if out_str.strip():
                turn += f"\n[result]\n{out_str}"
            flat.append({"role": "Assistant", "content": turn})

        final_response = result.get("final_response") or result.get("output") or ""
        if isinstance(final_response, str) and final_response.strip():
            flat.append({"role": "Assistant", "content": final_response})
        elif len(flat) == 1:
            # No tool calls AND no final response — still need SOMETHING so
            # the publish isn't an empty trajectory.
            flat.append({"role": "Assistant", "content": result.get("status", "unknown")})
        return flat
