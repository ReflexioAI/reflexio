"""CLI entry point for the GDPVal comparison benchmark.

Runs the P1 → snapshot → P2 → P3 protocol per host agent, writes one
results.jsonl per cell, then delegates to `report.build_comparison` for
the cross-cell deltas.

Run:

    uv run python -m benchmark.gdpval.run_benchmark \\
        --hosts openspace,hermes \\
        --phases p1,p2,p3 \\
        --task-list benchmark/gdpval/task_lists/tasks_50.json \\
        --max-tasks 5
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent  # open_source/reflexio/benchmark/gdpval/
# The benchmark package lives inside the reflexio submodule at
# open_source/reflexio/benchmark/gdpval/. Adding the submodule root
# to sys.path makes `import benchmark.gdpval.x` work whether the user runs
# `python -m benchmark.gdpval.run_benchmark` from the submodule root or from
# the outer reflexio-gdpval-bench repo root.
_SUBMODULE_ROOT = _THIS_DIR.parents[1]  # open_source/reflexio/

if str(_SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_ROOT))

from dotenv import load_dotenv  # noqa: E402

from benchmark.gdpval.adapters.base import AgentResult, HostAgentAdapter  # noqa: E402
from benchmark.gdpval.adapters.hermes_adapter import HermesAdapter  # noqa: E402
from benchmark.gdpval.adapters.openspace_adapter import OpenSpaceAdapter  # noqa: E402
from benchmark.gdpval.config import (  # noqa: E402
    CLAWWORK_ROOT,
    DEFAULT_HOSTS,
    DEFAULT_MODEL,
    DEFAULT_PHASES,
    DEFAULT_REFLEXIO_URL,
    DEFAULT_TASK_LIST,
    LOG_DIR,
    OUTPUT_DIR,
)
from benchmark.gdpval.evaluation import evaluate_task  # noqa: E402
from benchmark.gdpval.memory.reflexio_bridge import (  # noqa: E402
    ReflexioMemory,
    update_playbook_extractor_prompt,
)
from benchmark.gdpval.task_loader import load_tasks  # noqa: E402
from benchmark.gdpval.tokens import TokenStats  # noqa: E402

logger = logging.getLogger(__name__)

_ALL_PHASES = ("p1", "p2", "p3")
_ADAPTER_FACTORY: dict[str, Any] = {
    "openspace": OpenSpaceAdapter,
    "hermes": HermesAdapter,
}


def _find_env_file() -> Path | None:
    """Walk up from the benchmark dir looking for a .env to load."""
    current = _THIS_DIR
    for _ in range(10):
        env_path = current / ".env"
        if env_path.exists():
            return env_path
        current = current.parent
    return None


load_dotenv(dotenv_path=_find_env_file())


def _setup_logging(verbose: bool) -> None:
    """Configure root logger for stdout and a per-run log file.

    Args:
        verbose (bool): If True, set level to DEBUG.
    """
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(level)
    stdout.setFormatter(fmt)
    root.addHandler(stdout)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(LOG_DIR / f"run_{stamp}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GDPVal comparison benchmark: OpenSpace × Hermes × Reflexio.",
    )
    parser.add_argument(
        "--hosts",
        default=",".join(DEFAULT_HOSTS),
        help=f"Comma-separated host agents to run (default: {','.join(DEFAULT_HOSTS)})",
    )
    parser.add_argument(
        "--phases",
        default=",".join(DEFAULT_PHASES),
        help="Comma-separated phases: p1,p2,p3 (default: all three)",
    )
    parser.add_argument(
        "--task-list",
        default=str(DEFAULT_TASK_LIST),
        help="Path to a JSON list of task IDs (OpenSpace gdpval_bench/tasks_50.json works)",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Cap tasks for smoke runs"
    )
    parser.add_argument(
        "--task-offset",
        type=int,
        default=0,
        help=(
            "Skip the first N tasks after filtering, before applying --max-tasks. "
            "Use with --max-tasks to run a different slice of the same task list "
            "on a subsequent day (e.g. day 1: --max-tasks 5; day 2: --task-offset 5 "
            "--max-tasks 5). Combine with --run-name pointing at the prior run to "
            "accumulate results in the same output dir."
        ),
    )
    parser.add_argument(
        "--task-ids",
        default=None,
        help=(
            "Comma-separated explicit task IDs to run (overrides the JSON file "
            "specified by --task-list). Useful for rerunning a specific failing "
            "task or extending a run with hand-picked additions."
        ),
    )
    parser.add_argument(
        "--per-occupation",
        type=int,
        default=None,
        help="Stratified sample: N tasks per occupation",
    )
    parser.add_argument(
        "--sectors", nargs="+", default=None, help="Filter by sector substring"
    )
    parser.add_argument(
        "--occupations", nargs="+", default=None, help="Filter by occupation substring"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output subdir name (default: gdpval_<timestamp>)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--reflexio-url", default=DEFAULT_REFLEXIO_URL, help="Reflexio backend URL"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Per-task tool-calling iteration cap",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip LLMEvaluator scoring (saves credits)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Load tasks, check env, don't execute"
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=20.0,
        help="Seconds between per-task progress lines while adapter.run() is in flight (default: 20)",
    )
    parser.add_argument(
        "--task-timeout-sec",
        type=float,
        default=1200.0,
        help=(
            "Hard per-task wallclock cap. If adapter.run() doesn't return within this "
            "many seconds, the task is marked `timeout` and the orchestrator moves on. "
            "Existing agent threads may leak; use a fresh benchmark process for long "
            "recovery. Default 1200s (20 min)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Max tasks to run concurrently per phase. OpenSpace supports up to N "
            "parallel workers (they share the global .openspace/ skill DB with SQLite "
            "write-ordering; this weakens cross-task skill accumulation during a phase "
            "but gives a near-linear wallclock speedup). Hermes is ALWAYS forced to "
            "serial regardless of this value because its file tools rely on process "
            "cwd, which can't be safely shared across concurrent tasks in-process. "
            "Default 1."
        ),
    )
    parser.add_argument(
        "--cache-from",
        default=None,
        help=(
            "Name of a prior run dir under output/ whose P1 trajectories and "
            "post_p1 snapshot should be reused. Use with --phases p3 (or p2,p3) "
            "to iterate on P3 without re-running P1/P2. Cached trajectories are "
            "re-published under the CURRENT run's user_id namespace so each P3 "
            "iteration has a clean playbook pool. Also copies the cached "
            "p1_cold/p2_warm results.jsonl into the new run dir for comparison."
        ),
    )
    return parser.parse_args()


def _resolve_task_ids(
    task_list_path: str | None,
    task_ids_csv: str | None,
) -> list[str] | None:
    """Resolve the task-ID filter from CLI flags.

    `--task-ids` (a comma-separated string) takes precedence over
    `--task-list` (a JSON file path). Whitespace is trimmed and empty
    segments are dropped.

    Args:
        task_list_path (str | None): Path to a JSON task-list file.
        task_ids_csv (str | None): Comma-separated task IDs from the CLI.

    Returns:
        list[str] | None: Explicit task IDs to filter by, or None when
            neither flag is set (meaning "no ID filter").
    """
    if task_ids_csv:
        return [tid.strip() for tid in task_ids_csv.split(",") if tid.strip()]
    if task_list_path:
        return _parse_task_list(task_list_path)
    return None


def _apply_task_slice(
    tasks: list[dict[str, Any]],
    offset: int,
    max_tasks: int | None,
) -> list[dict[str, Any]]:
    """Apply `--task-offset` and `--max-tasks` to a loaded task list.

    The offset is clamped to non-negative values. When offset exceeds the
    length of `tasks`, an empty list is returned so the caller can log
    and abort cleanly. `max_tasks=None` means "no cap".

    Args:
        tasks (list[dict[str, Any]]): Tasks loaded from the task list.
        offset (int): Number of tasks to skip before slicing.
        max_tasks (int | None): Maximum number of tasks to return, or
            None for no cap.

    Returns:
        list[dict[str, Any]]: Sliced task list.
    """
    offset = max(0, int(offset or 0))
    if offset >= len(tasks):
        return []
    sliced = tasks[offset:]
    if max_tasks is not None:
        sliced = sliced[:max_tasks]
    return sliced


def _parse_task_list(task_list_path: str) -> list[str] | None:
    """Load task IDs from a JSON file produced by OpenSpace's gdpval_bench.

    Args:
        task_list_path (str): Path to the JSON file.

    Returns:
        list[str] | None: List of task_ids to filter by, or None if the file
            doesn't exist or is empty.
    """
    path = Path(task_list_path)
    if not path.exists():
        logger.warning("Task list not found: %s — running without ID filter", path)
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Task list %s is not valid JSON: %s", path, exc)
        return None
    if isinstance(data, list):
        return [str(t) for t in data]
    if isinstance(data, dict) and "task_ids" in data:
        return [str(t) for t in data["task_ids"]]
    logger.warning("Task list %s has unexpected shape; ignoring", path)
    return None


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to `path`, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _copy_tree(src: Path, dest: Path) -> None:
    """Copy `src` onto `dest`, replacing `dest` if it exists."""
    if dest.exists():
        shutil.rmtree(dest)
    if src.exists():
        shutil.copytree(src, dest)
    else:
        dest.mkdir(parents=True, exist_ok=True)


def _make_adapter(host: str, args: argparse.Namespace) -> HostAgentAdapter:
    """Construct the adapter for a given host name."""
    factory = _ADAPTER_FACTORY.get(host)
    if factory is None:
        raise ValueError(f"Unknown host: {host}. Supported: {list(_ADAPTER_FACTORY)}")
    return factory(model=args.model, max_iterations=args.max_iterations)


def _tail_hermes_action(hermes_home: Path) -> str | None:
    """Peek at the tail of Hermes's agent.log for its most recent tool call.

    Background runs get no TUI output from Hermes, so this is the only
    mid-task visibility we have into what the agent is doing. We look for
    lines matching "Calling <tool> with args" in the last ~4KB and return
    a short description.

    Args:
        hermes_home (Path): Path to Hermes's HERMES_HOME dir.

    Returns:
        str | None: Short description of the most recent action, or None
            if the log file is empty / missing / unparseable.
    """
    log_path = hermes_home / "logs" / "agent.log"
    if not log_path.exists():
        return None
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if "Calling " in line and " with args" in line:
            try:
                _, tail_part = line.split("Calling ", 1)
                tool_name, _ = tail_part.split(" with args", 1)
                return f"tool={tool_name.strip()}"
            except ValueError:
                continue
        if "iteration" in line.lower() and "grounding_agent" in line.lower():
            return line.split("grounding_agent")[-1][:80].strip()
    return None


def _workspace_snapshot(workspace: Path) -> tuple[int, int, str]:
    """Summarize a workspace dir's current contents for progress logging.

    Args:
        workspace (Path): Per-task workspace directory.

    Returns:
        tuple[int, int, str]: (file_count, total_size_bytes, newest_filename).
            newest is "-" if the workspace is empty.
    """
    if not workspace.exists():
        return (0, 0, "-")
    files = [f for f in workspace.rglob("*") if f.is_file()]
    if not files:
        return (0, 0, "-")
    total = sum(f.stat().st_size for f in files)
    newest = max(files, key=lambda f: f.stat().st_mtime)
    return (len(files), total, newest.name)


class _TaskProgressLogger:
    """Emit a per-task progress line every `interval` seconds while a task runs.

    Unlike a bare heartbeat, each line carries substance: workspace file count
    and size delta since start, the newest file name written, and — for
    Hermes — the most recent tool call parsed from agent.log.

    Used as an async context manager so the caller can:

        async with _TaskProgressLogger(...) as _:
            result = await adapter.run(...)
    """

    def __init__(
        self,
        host: str,
        phase: str,
        idx: int,
        total: int,
        task_id: str,
        workspace: Path,
        host_state_dir: Path | None = None,
        interval_sec: float = 20.0,
    ) -> None:
        self._host = host
        self._phase = phase
        self._idx = idx
        self._total = total
        self._task_id = task_id
        self._workspace = workspace
        self._host_state_dir = host_state_dir
        self._interval = interval_sec
        self._stop: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._start: float = 0.0

    async def __aenter__(self) -> _TaskProgressLogger:
        self._stop = asyncio.Event()
        self._start = time.monotonic()
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=2.0)

    async def _run(self) -> None:
        assert self._stop is not None
        last_file_count = -1
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                return
            except TimeoutError:
                pass
            elapsed = time.monotonic() - self._start
            files, size, newest = _workspace_snapshot(self._workspace)
            delta_note = (
                f"+{files - last_file_count}" if last_file_count >= 0 else f"{files}"
            )
            last_file_count = files
            action = None
            if self._host == "hermes" and self._host_state_dir is not None:
                action = _tail_hermes_action(self._host_state_dir / "hermes_home")
            extras = f" action={action}" if action else ""
            logger.info(
                "[%s][%s] (%d/%d) %s running... elapsed=%ds files=%s size=%dK newest=%s%s",
                self._host,
                self._phase,
                self._idx,
                self._total,
                self._task_id[:8],
                int(elapsed),
                delta_note,
                size // 1024,
                newest,
                extras,
            )


async def _run_phase(
    host: str,
    phase: str,
    adapters: list[HostAgentAdapter],
    tasks: list[dict[str, Any]],
    run_dir: Path,
    reflexio: ReflexioMemory,
    host_state_dirs: list[Path],
    *,
    enable_eval: bool,
    progress_interval_sec: float,
    task_timeout_sec: float,
) -> None:
    """Loop tasks through the adapter pool for one phase and persist records.

    Dispatches up to `len(adapters)` tasks concurrently via an asyncio.Queue
    worker pool. When the pool has one adapter, execution is serial and the
    behavior matches the pre-pool code path exactly. When the pool has many,
    each task waits for a free worker, runs on it, and releases the worker
    back into the pool.

    Per-task semantics preserved under concurrency:
      - Each task writes to its own `workspace_root / <task_id>` directory;
        no two concurrent tasks write to the same workspace.
      - `results.jsonl` is append-only and stream-writes are protected by a
        single async lock so completed records arrive as full lines, not
        interleaved halves.
      - Reflexio publish/fetch are per-task async calls; they don't share
        state beyond the reflexio client, which is thread-safe.

    In P1, every finished trajectory is also published to reflexio so its
    memory store warms up in parallel with the host's native learning. In
    P3, the reflexio bridge's unified search is queried once per task to
    render a memory block which is injected via the adapter's `memory`
    argument.

    Args:
        host (str): Host agent name ("openspace"/"hermes"), recorded in records.
        phase (str): Phase label ("p1"/"p2"/"p3").
        adapters (list[HostAgentAdapter]): Pool of initialized adapters, one
            per concurrent slot. Length determines max parallelism.
        tasks (list[dict[str, Any]]): Tasks to execute.
        run_dir (Path): Root directory for this run's outputs.
        reflexio (ReflexioMemory): Configured bridge for the current host.
        host_state_dirs (list[Path]): Parallel list — host_state_dirs[i]
            corresponds to adapters[i]. Used by the progress logger to peek
            at host-internal state (e.g. Hermes agent.log).
        enable_eval (bool): If True, run the ClawWork evaluator on each result.
        progress_interval_sec (float): Heartbeat cadence for progress logs.
        task_timeout_sec (float): Per-task hard cap for adapter.run().
    """
    phase_dir = run_dir / host / f"{phase}_{_phase_label(phase)}"
    results_file = phase_dir / "results.jsonl"
    workspace_root = phase_dir / "workspace"

    logger.info(
        "[%s][%s] Running %d task(s) with concurrency=%d",
        host,
        phase,
        len(tasks),
        len(adapters),
    )

    phase_start = time.monotonic()
    phase_tokens = 0
    phase_status_counts: dict[str, int] = {}
    results_lock = asyncio.Lock()
    pool: asyncio.Queue[tuple[int, HostAgentAdapter, Path]] = asyncio.Queue()
    for i, adapter in enumerate(adapters):
        pool.put_nowait((i, adapter, host_state_dirs[i]))

    async def _execute_one(idx: int, task: dict[str, Any]) -> None:
        nonlocal phase_tokens
        tid = task["task_id"]
        workspace = workspace_root / tid
        memory = None
        if phase == "p3":
            fetch_t0 = time.monotonic()
            memory = await reflexio.fetch_for_task(task)
            fetch_elapsed = time.monotonic() - fetch_t0
            logger.info(
                "[%s][%s] (%d/%d) %s reflexio fetch: %d chars in %.1fs",
                host,
                phase,
                idx,
                len(tasks),
                tid[:8],
                len(memory) if memory else 0,
                fetch_elapsed,
            )

        worker_id, adapter, host_state_dir = await pool.get()
        try:
            logger.info(
                "[%s][%s] (%d/%d) %s START w%d — %s%s",
                host,
                phase,
                idx,
                len(tasks),
                tid[:8],
                worker_id,
                task.get("occupation", ""),
                f" (memory={len(memory)}ch)" if memory else "",
            )
            task_t0 = time.monotonic()
            try:
                async with _TaskProgressLogger(
                    host=host,
                    phase=phase,
                    idx=idx,
                    total=len(tasks),
                    task_id=tid,
                    workspace=workspace,
                    host_state_dir=host_state_dir,
                    interval_sec=progress_interval_sec,
                ):
                    result = await asyncio.wait_for(
                        adapter.run(task=task, workspace=workspace, memory=memory),
                        timeout=task_timeout_sec,
                    )
            except TimeoutError:
                task_elapsed = time.monotonic() - task_t0
                logger.error(
                    "[%s][%s] (%d/%d) %s TIMEOUT after %.0fs — task_timeout_sec=%.0f",
                    host,
                    phase,
                    idx,
                    len(tasks),
                    tid[:8],
                    task_elapsed,
                    task_timeout_sec,
                )
                files, size, newest = _workspace_snapshot(workspace)
                result = AgentResult(
                    status="timeout",
                    iterations=0,
                    tool_calls=0,
                    tokens=TokenStats(wall_time_sec=round(task_elapsed, 2)),
                    artifacts_dir=workspace,
                    messages=[
                        {
                            "role": "User",
                            "content": task.get("prompt", "")[:2000] or "",
                        },
                        {
                            "role": "Assistant",
                            "content": f"[TIMEOUT after {task_elapsed:.0f}s — {files} artifacts in workspace]",
                        },
                    ],
                    raw={"timeout_sec": task_timeout_sec, "elapsed": task_elapsed},
                )
                # Force-reinit the adapter so the next task gets a clean state.
                # When asyncio.wait_for cancels adapter.run(), OpenSpace's
                # internal `_task_done` Event can be left unreleased, causing
                # every subsequent task to hit "OpenSpace is busy — waiting
                # up to 660s" and also fail. cleanup() + initialize() forces
                # a fresh OpenSpace instance. Hermes is immune to this
                # (its AIAgent is rebuilt per task) but calling the same
                # reset path is harmless.
                try:
                    await adapter.cleanup()
                    await adapter.initialize(host_state_dir)
                    logger.info(
                        "[%s][%s] (%d/%d) %s adapter force-reinit after timeout",
                        host,
                        phase,
                        idx,
                        len(tasks),
                        tid[:8],
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s][%s] (%d/%d) %s adapter reinit failed after timeout: %s",
                        host,
                        phase,
                        idx,
                        len(tasks),
                        tid[:8],
                        exc,
                    )
            task_elapsed = time.monotonic() - task_t0

            eval_result: dict[str, Any] = {"has_evaluation": False}
            eval_t0 = time.monotonic()
            if enable_eval and result.status != "error":
                eval_result = evaluate_task(task, workspace)
            eval_elapsed = time.monotonic() - eval_t0

            files, size, newest = _workspace_snapshot(workspace)
            async with results_lock:
                phase_tokens += result.tokens.total_tokens
                phase_status_counts[result.status] = (
                    phase_status_counts.get(result.status, 0) + 1
                )
                logger.info(
                    "[%s][%s] (%d/%d) %s DONE %s in %.0fs — "
                    "iters=%d tool_calls=%d tokens=%d (p=%d c=%d) "
                    "artifacts=%d (%dK) eval=%s%s",
                    host,
                    phase,
                    idx,
                    len(tasks),
                    tid[:8],
                    result.status,
                    task_elapsed,
                    result.iterations,
                    result.tool_calls,
                    result.tokens.total_tokens,
                    result.tokens.prompt_tokens,
                    result.tokens.completion_tokens,
                    files,
                    size // 1024,
                    f"{eval_result.get('score_10', 0)}/10"
                    if eval_result.get("has_evaluation")
                    else "skipped",
                    f" (eval {eval_elapsed:.1f}s)" if eval_elapsed > 0.5 else "",
                )
                record = {
                    "host": host,
                    "phase": phase,
                    "task_id": tid,
                    "occupation": task.get("occupation", ""),
                    "sector": task.get("sector", ""),
                    "task_value_usd": task.get("task_value_usd", 0.0),
                    "status": result.status,
                    "tokens": result.tokens.to_dict(),
                    "execution": {
                        "iterations": result.iterations,
                        "tool_calls": result.tool_calls,
                        "time_sec": result.tokens.wall_time_sec,
                    },
                    "memory_injected_chars": len(memory) if memory else 0,
                    "evaluation": eval_result,
                    "raw": result.raw,
                    "worker_id": worker_id,
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
                _append_jsonl(results_file, record)

            if phase == "p1":
                trajectories_dir = run_dir / host / "trajectories_p1"
                trajectories_dir.mkdir(parents=True, exist_ok=True)
                (trajectories_dir / f"{tid}.json").write_text(
                    json.dumps(
                        {
                            "task_id": tid,
                            "task": task,
                            "messages": result.messages,
                            "status": result.status,
                        },
                        default=str,
                        indent=2,
                    )
                )
                publish_t0 = time.monotonic()
                await reflexio.publish_trajectory(task, result.messages)
                publish_elapsed = time.monotonic() - publish_t0
                logger.info(
                    "[%s][%s] (%d/%d) %s published to reflexio in %.1fs",
                    host,
                    phase,
                    idx,
                    len(tasks),
                    tid[:8],
                    publish_elapsed,
                )
        finally:
            pool.put_nowait((worker_id, adapter, host_state_dir))

    # Dispatch all tasks — asyncio.gather schedules them concurrently and
    # the bounded pool (via Queue.get/put) throttles actual parallelism to
    # len(adapters). Tasks execute in the order they become dispatchable,
    # not in the task-list order, so we label by idx for log correlation.
    await asyncio.gather(
        *(_execute_one(idx, task) for idx, task in enumerate(tasks, start=1))
    )

    phase_elapsed = time.monotonic() - phase_start
    status_summary = " ".join(
        f"{k}={v}" for k, v in sorted(phase_status_counts.items())
    )
    logger.info(
        "[%s][%s] PHASE DONE — %d tasks in %.0fs (%s) total_tokens=%d mean=%d",
        host,
        phase,
        len(tasks),
        phase_elapsed,
        status_summary or "no-tasks",
        phase_tokens,
        phase_tokens // max(1, len(tasks)),
    )


def _phase_label(phase: str) -> str:
    """Human-readable tag for phase output directories."""
    return {"p1": "cold", "p2": "warm", "p3": "warm_reflexio"}.get(phase, phase)


async def _hydrate_cache(
    host: str,
    host_dir: Path,
    cache_from_dir: Path,
    reflexio: ReflexioMemory,
    tasks: list[dict[str, Any]],
) -> None:
    """Copy P1 state + results from a prior run and replay trajectories.

    When the user passes `--cache-from RUN`, we skip running P1 entirely but
    still need: (a) OpenSpace warm state for P2/P3 restore, (b) p1/p2
    results.jsonl for comparison, (c) reflexio seeded with this run's P1
    trajectories under a fresh per-task user_id namespace so P3 iteration
    isn't contaminated by prior iterations' playbooks.

    Args:
        host (str): Host name ("openspace"/"hermes").
        host_dir (Path): Current run's host dir.
        cache_from_dir (Path): Prior run's host dir to import from.
        reflexio (ReflexioMemory): Bridge using the *current* run's user_id
            prefix — that's what makes each iteration have a clean namespace.
        tasks (list[dict[str, Any]]): Tasks to replay.
    """
    cache_host_dir = cache_from_dir / host
    if not cache_host_dir.exists():
        raise FileNotFoundError(f"--cache-from source missing: {cache_host_dir}")

    # 1. Copy post-P1 snapshot into current run dir
    cache_snapshot = cache_host_dir / "snapshots" / "post_p1"
    target_snapshot = host_dir / "snapshots" / "post_p1"
    if not cache_snapshot.exists():
        raise FileNotFoundError(f"Cached snapshot missing: {cache_snapshot}")
    target_snapshot.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree(cache_snapshot, target_snapshot)
    logger.info("[%s] cache-from: snapshot copied from %s", host, cache_snapshot)

    # 2. Copy p1/p2 results.jsonl into current run dir for comparison
    for phase_name in ("p1_cold", "p2_warm"):
        src = cache_host_dir / phase_name / "results.jsonl"
        if src.exists():
            dst = host_dir / phase_name / "results.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            logger.info("[%s] cache-from: copied %s results", host, phase_name)

    # 3. Re-publish trajectories under the new user_id namespace
    traj_dir = cache_host_dir / "trajectories_p1"
    if not traj_dir.exists():
        logger.warning(
            "[%s] cache-from: trajectories_p1 missing at %s — "
            "reflexio will have no memory to extract from",
            host,
            traj_dir,
        )
        return

    task_map = {t["task_id"]: t for t in tasks}
    published = 0
    for traj_file in sorted(traj_dir.glob("*.json")):
        try:
            data = json.loads(traj_file.read_text())
        except Exception as exc:
            logger.warning(
                "[%s] cache-from: bad trajectory file %s: %s", host, traj_file, exc
            )
            continue
        tid = data.get("task_id")
        if tid not in task_map:
            continue
        messages = data.get("messages") or []
        if not messages:
            logger.warning("[%s] cache-from: empty messages for %s", host, tid[:8])
            continue
        await reflexio.publish_trajectory(task_map[tid], messages)
        published += 1
        logger.info(
            "[%s] cache-from: republished trajectory %s (%d messages)",
            host,
            tid[:8],
            len(messages),
        )
    logger.info("[%s] cache-from: %d trajectories replayed", host, published)

    # 4. Trigger extraction + block until playbooks are ready
    counts = await reflexio.trigger_extraction_for_tasks(
        [t["task_id"] for t in tasks],
        wait_budget_sec=180.0,
    )
    logger.info(
        "[%s] cache-from: extraction complete — playbook counts: %s",
        host,
        {tid[:8]: n for tid, n in counts.items()},
    )


async def _run_host(
    host: str,
    phases: list[str],
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    run_dir: Path,
) -> None:
    """Execute the full P1 → snapshot → P2 → P3 flow for one host.

    P2 and P3 each start from a fresh copy of the post-P1 snapshot so the
    P2 → P3 delta cleanly isolates reflexio's marginal contribution.
    """
    host_dir = run_dir / host
    host_dir.mkdir(parents=True, exist_ok=True)
    reflexio = ReflexioMemory(
        user_id_prefix=f"bench_{host}_{run_dir.name}",
        url=args.reflexio_url,
    )

    # If caching from a prior run, hydrate state + replay trajectories BEFORE
    # any phases execute. This is a no-op when --cache-from is not set.
    cache_from = getattr(args, "cache_from", None)
    if cache_from and "p1" not in phases:
        cache_from_dir = OUTPUT_DIR / cache_from
        await _hydrate_cache(host, host_dir, cache_from_dir, reflexio, tasks)

    snapshot_dir = host_dir / "snapshots" / "post_p1"

    # Hermes can't safely run concurrent tasks in-process because its file
    # tools rely on the process cwd (os.chdir). OpenSpace passes workspace
    # paths through execute() explicitly and tolerates concurrent workers.
    effective_concurrency = 1 if host == "hermes" else max(1, args.concurrency)
    if args.concurrency > 1 and host == "hermes":
        logger.info(
            "[hermes] --concurrency=%d ignored; Hermes forced to serial "
            "(os.chdir is process-wide).",
            args.concurrency,
        )

    for phase in phases:
        phase_state_dir = host_dir / f"host_state_{phase}"
        if phase == "p1":
            if phase_state_dir.exists():
                shutil.rmtree(phase_state_dir)
            phase_state_dir.mkdir(parents=True, exist_ok=True)
        else:
            if not snapshot_dir.exists():
                logger.warning(
                    "Snapshot missing for %s; %s cannot start warm", host, phase
                )
                phase_state_dir.mkdir(parents=True, exist_ok=True)
            else:
                _copy_tree(snapshot_dir, phase_state_dir)

        # Build the worker pool. Each worker gets its own sub-state dir so
        # per-worker side effects (Hermes HERMES_HOME, OpenSpace workspace
        # recording dirs) don't stomp on each other. OpenSpace's global
        # SkillStore is still shared — that's a known limitation of
        # concurrent OpenSpace, matching gdpval_bench's own concurrent mode.
        adapters: list[HostAgentAdapter] = []
        worker_state_dirs: list[Path] = []
        for worker_id in range(effective_concurrency):
            if effective_concurrency == 1:
                worker_dir = phase_state_dir
            else:
                worker_dir = phase_state_dir / f"worker_{worker_id}"
                worker_dir.mkdir(parents=True, exist_ok=True)
                # For P2/P3 warm restore, each worker gets its own copy of
                # the post_p1 snapshot so the warm state doesn't mutate
                # under another worker's feet.
                if phase != "p1" and snapshot_dir.exists():
                    for child in snapshot_dir.iterdir():
                        _copy_tree(child, worker_dir / child.name)
            adapter = _make_adapter(host, args)
            await adapter.initialize(worker_dir)
            adapters.append(adapter)
            worker_state_dirs.append(worker_dir)

        try:
            await _run_phase(
                host=host,
                phase=phase,
                adapters=adapters,
                tasks=tasks,
                run_dir=run_dir,
                reflexio=reflexio,
                host_state_dirs=worker_state_dirs,
                enable_eval=not args.no_eval,
                progress_interval_sec=args.progress_interval,
                task_timeout_sec=args.task_timeout_sec,
            )
            if phase == "p1":
                # Snapshot from worker 0; its state is representative
                # (for Hermes, it's the only worker; for OpenSpace, all
                # workers share the global skill DB and worker 0's local
                # dir is enough).
                await adapters[0].snapshot_state(snapshot_dir)
                # Force reflexio to extract playbooks — one call per task
                # so the extractor processes one trajectory per LLM call,
                # and BLOCK until every task has ≥1 playbook (or a 15-min
                # wait budget expires). Without the block, P3 fetches land
                # at mem=0 for most tasks because extractions are still
                # running on the backend — Step E rerun v3 saw 4 of 5 P3
                # tasks miss their memory that way.
                # 180s wait budget: gpt-4o-mini extraction takes ~10-30s
                # per task, serialized via a backend org-level lock. 5 tasks
                # × 30s = 150s upper bound. Tasks that produce 0 playbooks
                # never converge, so don't block forever — proceed and let
                # P3 fall back to no-memory for those tasks (degenerates to
                # P2 behavior, which is fine for the comparison).
                counts = await reflexio.trigger_extraction_for_tasks(
                    [t["task_id"] for t in tasks],
                    wait_budget_sec=180.0,
                )
                logger.info(
                    "[%s][%s] extraction done — playbook counts: %s",
                    host,
                    phase,
                    {tid[:8]: n for tid, n in counts.items()},
                )
        finally:
            for adapter in adapters:
                await adapter.cleanup()


async def _amain(args: argparse.Namespace) -> None:
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for phase in phases:
        if phase not in _ALL_PHASES:
            raise ValueError(f"Unknown phase: {phase}. Expected one of {_ALL_PHASES}")
    for host in hosts:
        if host not in _ADAPTER_FACTORY:
            raise ValueError(
                f"Unknown host: {host}. Supported: {list(_ADAPTER_FACTORY)}"
            )

    # Explicit --task-ids overrides the JSON file-based --task-list.
    task_ids = _resolve_task_ids(args.task_list, args.task_ids)

    # Load without max-tasks so --task-offset can slice deterministically over
    # the full filtered pool; we apply max-tasks ourselves after the offset.
    try:
        tasks = load_tasks(
            clawwork_root=str(CLAWWORK_ROOT),
            task_ids=task_ids,
            max_tasks=None,
            sectors=args.sectors,
            occupations=args.occupations,
            per_occupation=args.per_occupation,
        )
    except FileNotFoundError as exc:
        if args.dry_run:
            logger.warning("Dry run: task loading skipped (%s)", exc)
            tasks = []
        else:
            raise

    original_count = len(tasks)
    tasks = _apply_task_slice(tasks, args.task_offset, args.max_tasks)
    if not tasks and args.task_offset and args.task_offset >= original_count:
        logger.error(
            "task-offset %d >= loaded task count %d — nothing to run",
            args.task_offset,
            original_count,
        )
        return

    if not tasks and not args.dry_run:
        logger.error("No tasks loaded — aborting.")
        return

    run_name = (
        args.run_name or f"gdpval_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "hosts": hosts,
                "phases": phases,
                "model": args.model,
                "task_count": len(tasks),
                "max_iterations": args.max_iterations,
                "reflexio_url": args.reflexio_url,
                "no_eval": args.no_eval,
            },
            indent=2,
        )
    )

    logger.info("Run: %s", run_dir)
    logger.info("Hosts: %s  Phases: %s  Tasks: %d", hosts, phases, len(tasks))

    if args.dry_run:
        logger.info("Dry run complete — adapters not constructed.")
        return

    # Override reflexio's default user_playbook_extractor prompt to capture
    # both issue→fix and success patterns before any P1 trajectories land.
    # Must happen before any host's P1 so every published trajectory hits
    # the new extraction semantics.
    if any(p == "p3" for p in phases):
        ok = update_playbook_extractor_prompt(url=args.reflexio_url)
        if ok:
            logger.info("Reflexio playbook extractor prompt override applied")
        else:
            logger.warning(
                "Reflexio playbook extractor prompt override FAILED — "
                "P3 will use the default issue-oriented extractor"
            )

    for host in hosts:
        await _run_host(
            host=host, phases=phases, args=args, tasks=tasks, run_dir=run_dir
        )

    from benchmark.gdpval.report import build_comparison  # noqa: E402

    # When --cache-from is set, the cached p1/p2 results were copied into the
    # current run dir during _hydrate_cache. Build the comparison across ALL
    # phases (not just the ones we executed) so the report shows the full
    # P1→P2→P3 picture for cached iteration runs.
    report_phases = list(_ALL_PHASES) if getattr(args, "cache_from", None) else phases
    build_comparison(run_dir, hosts=hosts, phases=report_phases)
    logger.info("Done. Results: %s", run_dir)


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
