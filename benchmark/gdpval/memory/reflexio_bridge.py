"""Reflexio integration for the GDPVal benchmark.

Three responsibilities:
  1. During P1, publish every finished task's trajectory via
     `ReflexioClient.publish_interaction` so reflexio's memory store is
     seeded from the same raw experience that drove the host's native
     learning.
  2. During P3, call `ReflexioClient.search` (unified search across
     profiles, agent_playbooks, user_playbooks) with the task prompt and
     render the top hits into a system-prompt block via
     `memory.injection.render_memory_block`.
  3. Once per benchmark run (before any phase), override the default
     user-playbook extractor prompt so reflexio's extractor captures
     BOTH issue→fix pairs and positive success patterns. The default
     prompt is issue-oriented only, which misses the majority of
     extractable learnings in clean successful runs.

Distinct `org_id` / `user_id` namespaces per host (e.g.
`bench_openspace_<run>`, `bench_hermes_<run>`) keep OpenSpace's P1
transcripts from polluting Hermes's P3 run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from reflexio.client.client import ReflexioClient

from benchmark.gdpval.config import (
    DEFAULT_REFLEXIO_URL,
    DEFAULT_SEARCH_THRESHOLD,
    DEFAULT_TOP_K,
)
from benchmark.gdpval.memory.injection import render_memory_block

logger = logging.getLogger(__name__)


class ReflexioMemory:
    """Per-task wrapper around `ReflexioClient` for publish + fetch.

    Each task gets its own reflexio `user_id` derived from the shared
    `user_id_prefix` — so task A's playbooks never surface in task B's
    P3 search. This eliminates the cross-domain pollution that Step E
    rerun v2 showed, where a single playbook extracted from a 17-file
    financial task got injected into an unrelated Music Tour task.

    With per-task user_ids the P3 comparison answers the focused
    question: "does memory from this task's OWN prior run (P1) help
    solve it faster in P3?" — aligned with the intent that reflexio
    should accelerate a task when the agent has seen it before.
    """

    def __init__(
        self,
        user_id_prefix: str,
        url: str = DEFAULT_REFLEXIO_URL,
        api_key: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SEARCH_THRESHOLD,
    ) -> None:
        """
        Args:
            user_id_prefix (str): Namespace prefix for per-task user_ids.
                Pick a per-host-per-run value like `bench_openspace_<run>`.
                Each task's actual reflexio user_id is
                `f"{user_id_prefix}_{task_id}"`.
            url (str): Reflexio backend URL.
            api_key (str | None): API key; falls back to env.
            top_k (int): Max results per entity type in unified search.
            threshold (float): Similarity threshold for vector search.
        """
        self._user_id_prefix = user_id_prefix
        self._top_k = top_k
        self._threshold = threshold
        self._client = ReflexioClient(url_endpoint=url, api_key=api_key or "")

    @property
    def user_id_prefix(self) -> str:
        return self._user_id_prefix

    def _user_id_for_task(self, task_id: str) -> str:
        """Derive the per-task reflexio user_id from the shared prefix."""
        return f"{self._user_id_prefix}_{task_id}"

    async def publish_trajectory(
        self,
        task: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> None:
        """Publish a finished task's trajectory for extraction.

        Uses a per-task `source` label (`gdpval_benchmark:tid_<task_id>`)
        so a later per-source `trigger_extraction` call can process one
        trajectory at a time. Without per-task sources, reflexio's
        extractor batches all P1 interactions into a single LLM call and
        the LLM tends to emit 0–1 playbooks for the entire batch even on
        rich content — collapsing per-task learning.

        Args:
            task (dict[str, Any]): Normalized GDPVal task (used to build a
                stable session_id and source label).
            messages (list[dict[str, Any]]): Adapter-flattened trajectory,
                each item `{"role": "User"|"Assistant", "content": str}`.
        """
        if not messages:
            logger.debug("Skipping publish: empty trajectory for %s", task.get("task_id"))
            return
        tid = task.get("task_id", "unknown")
        session_id = f"gdpval-{tid}"
        source = self._source_for_task(tid)
        user_id = self._user_id_for_task(tid)
        try:
            await asyncio.to_thread(
                self._client.publish_interaction,
                user_id=user_id,
                interactions=messages,
                source=source,
                session_id=session_id,
                wait_for_response=True,
            )
        except Exception as exc:
            logger.warning("publish_interaction failed for %s: %s", session_id, exc)

    def _source_for_task(self, task_id: str) -> str:
        """Return the per-run-per-task source label used by publish + extraction.

        The source label embeds both the run identifier (via the shared
        `user_id_prefix` which already includes the run name) AND the task
        id. This is critical for `manual_playbook_generation` — the backend
        looks up sessions via `storage.get_sessions(source=...)` and collects
        every distinct user_id it finds. If the source label were only
        `gdpval_benchmark:tid_<task>`, prior runs' interactions (which
        share the same task id but different user_ids) would show up, and
        `run_manual_regular` would dispatch extraction for each of those
        stale user_ids — some of which would then pass the should_run
        check for older-looking user_ids while failing it for the current
        run's user_id (because the same interaction content has already
        been seen in the org). That caused custom-001/custom-004 in v2_5
        to end up with zero playbooks under their v2_5 user_id even though
        stale v1_5 user_ids got re-extracted.

        With the run-prefixed source, each run is a fully isolated
        namespace: `get_sessions(source=...)` returns only THIS run's
        interactions, extraction runs only for THIS run's user_ids, and
        dedup only compares against THIS run's earlier playbooks.
        """
        return f"gdpval_benchmark:{self._user_id_prefix}:tid_{task_id}"

    async def fetch_for_task(self, task: dict[str, Any]) -> str | None:
        """Unified-search reflexio for memory relevant to this task.

        Args:
            task (dict[str, Any]): Normalized GDPVal task dict.

        Returns:
            str | None: Rendered memory block, or `None` if no hits — the
                caller passes `None` straight through to the adapter so P3
                degenerates gracefully to P2 behavior when nothing matches.
        """
        query = (task.get("prompt") or "").strip()
        if not query:
            return None
        tid = task.get("task_id", "unknown")
        user_id = self._user_id_for_task(tid)
        try:
            response = await asyncio.to_thread(
                self._client.search,
                query=query,
                top_k=self._top_k,
                threshold=self._threshold,
                user_id=user_id,
            )
        except Exception as exc:
            logger.warning("unified search failed for task %s: %s", tid, exc)
            return None

        # Debug: log the raw search response stats + the first 500 chars of
        # each hit so iteration sessions can see exactly what the P3 LLM is
        # getting injected. Critical for prompt tuning.
        prof_n = len(getattr(response, "profiles", []) or [])
        agent_n = len(getattr(response, "agent_playbooks", []) or [])
        user_n = len(getattr(response, "user_playbooks", []) or [])
        logger.info(
            "fetch_for_task %s: search hits profiles=%d agent_pbs=%d user_pbs=%d",
            tid[:8], prof_n, agent_n, user_n,
        )
        for pb in (getattr(response, "user_playbooks", []) or [])[:3]:
            content = (getattr(pb, "content", "") or "").strip()
            logger.info("  user_pb (%d chars): %s", len(content), content[:500])
        for pb in (getattr(response, "agent_playbooks", []) or [])[:3]:
            content = (getattr(pb, "content", "") or "").strip()
            logger.info("  agent_pb (%d chars): %s", len(content), content[:500])

        block = render_memory_block(response)
        return block or None

    async def trigger_extraction_for_tasks(
        self,
        task_ids: list[str],
        *,
        wait_budget_sec: float = 900.0,
        poll_interval_sec: float = 10.0,
    ) -> dict[str, int]:
        """Extract per-task playbooks and BLOCK until they're ready (or timeout).

        Two-stage operation:

        1. Dispatch — for each task, fire `manual_playbook_generation` with
           the task-specific source label. Reflexio's extractor receives one
           trajectory per call, avoiding the "1 playbook from 7 trajectories"
           compression we saw under batched extraction.

        2. Poll-wait — repeatedly call `get_user_playbooks(user_id)` for each
           task, until either the task has ≥1 playbook or the total wait
           budget expires. Without this blocking step, fire-and-forget
           dispatch means P3 can start before extractions complete — Step E
           rerun v3 saw 4 of 5 P3 tasks fetch empty memory for that reason.

        Trajectories that genuinely can't produce playbooks (too few
        interactions, timeouts) will never converge and will exhaust the
        budget — that's the intended behavior. `wait_budget_sec` bounds
        how long we wait before giving up and moving on.

        Args:
            task_ids (list[str]): Task IDs whose trajectories were published
                in this phase. Must match the IDs used by publish_trajectory()
                so the per-task source and user_id labels line up.
            wait_budget_sec (float): Max total wall time to wait for
                extractions to complete after dispatch.
            poll_interval_sec (float): Seconds between poll rounds.

        Returns:
            dict[str, int]: Map of task_id → final playbook count.
        """
        # Stage 1: dispatch
        for tid in task_ids:
            source = self._source_for_task(tid)
            try:
                await asyncio.to_thread(
                    self._client.manual_playbook_generation,
                    source=source,
                )
                logger.info("Dispatched playbook extraction for source=%s", source)
            except Exception as exc:
                logger.warning(
                    "manual_playbook_generation failed for %s: %s", source, exc
                )

        # Stage 2: poll-wait
        start = time.monotonic()
        deadline = start + wait_budget_sec
        pending = {tid: 0 for tid in task_ids}
        while pending and time.monotonic() < deadline:
            for tid in list(pending.keys()):
                user_id = self._user_id_for_task(tid)
                try:
                    resp = await asyncio.to_thread(
                        self._client.get_user_playbooks,
                        user_id=user_id,
                    )
                    pbs = getattr(resp, "user_playbooks", []) or []
                    if pbs:
                        elapsed = time.monotonic() - start
                        logger.info(
                            "Extraction READY for %s: %d playbook(s) after %.0fs",
                            tid[:8], len(pbs), elapsed,
                        )
                        pending[tid] = len(pbs)
                        del pending[tid]
                except Exception as exc:
                    logger.debug("Poll get_user_playbooks(%s) failed: %s", tid[:8], exc)
            if not pending:
                break
            elapsed = time.monotonic() - start
            logger.info(
                "Extraction wait: %d/%d tasks ready, %.0fs elapsed, %.0fs remaining",
                len(task_ids) - len(pending),
                len(task_ids),
                elapsed,
                max(0, deadline - time.monotonic()),
            )
            await asyncio.sleep(poll_interval_sec)

        if pending:
            elapsed = time.monotonic() - start
            logger.warning(
                "Extraction wait budget exhausted after %.0fs; %d task(s) still at 0 "
                "playbooks — proceeding anyway: %s",
                elapsed,
                len(pending),
                [t[:8] for t in pending],
            )

        # Return the final counts for all tasks
        counts: dict[str, int] = {}
        for tid in task_ids:
            if tid in pending:
                counts[tid] = 0
            else:
                user_id = self._user_id_for_task(tid)
                try:
                    resp = await asyncio.to_thread(
                        self._client.get_user_playbooks,
                        user_id=user_id,
                    )
                    counts[tid] = len(getattr(resp, "user_playbooks", []) or [])
                except Exception:
                    counts[tid] = 0
        return counts


# ---------------------------------------------------------------------------
# Extractor prompt override
# ---------------------------------------------------------------------------

# The extractor's sole job is to produce a CACHED SOLUTION RECORD — a
# reproducible recipe the next agent on the same task can copy-paste its
# way to the answer instead of re-deriving anything. Not rules. Not advice.
# The literal output artifacts, file schemas, shell commands, and computed
# final answer.
#
# Design rationale: Step E rerun v3/v5 showed that "rule-style" playbooks
# (even when domain-specific) made P3 SLOWER than P2. The LLM consumed the
# rule, treated it as a checklist item to verify, then did the whole task
# anyway. The only way reflexio helps is if the injected memory is dense
# enough to short-circuit the agent's own discovery loop — i.e. "here is
# the answer I computed last time, verify briefly and output".
GDPVAL_PLAYBOOK_EXTRACTOR_PROMPT = """\
You are a solution-archivist. You are given ONE agent trajectory that \
worked on ONE specific task. Your job is to emit a dense, \
copy-pasteable CACHED SOLUTION RECORD that a second agent can use to \
finish the task correctly without re-doing the investigation work.

DEFAULT BEHAVIOR — emit ONE playbook per trajectory unless you can \
identify one of the SPECIFIC failure patterns below. Most clean \
successful trajectories should produce a playbook. The downstream \
agent benefits more from a moderately-good playbook than from no \
playbook at all, so do NOT default to refusal when in doubt.

REFUSE (emit zero playbooks) ONLY when the trajectory falls into ONE \
of these specific patterns:

(A) GENERIC ENGINEERING ADVICE ONLY. The trajectory's reusable \
content is limited to universal engineering rules — "iterate \
incrementally", "use py_compile", "add try/except", "validate \
inputs", "run small sections first", "avoid unmatched quotes" — with \
no task-specific recipe attached. These are useless to inject \
because the downstream agent already knows them.

(B) FAILURE-RECOVERY DOMINANT. The trajectory's main reusable \
content is "if web research fails, ask the user to upload", "if \
pandoc is missing, fall back to X", "if the browser returns 404/403, \
notify the user". A downstream agent given such a playbook will \
re-enact the failure-recovery sequence instead of producing the \
deliverable. Refuse only when failure-recovery is the DOMINANT \
content; if there are also real artifacts, files, or domain values, \
include them and emit the playbook.

(C) NO CONCRETE ANCHORS. The trajectory genuinely contains zero of \
the following:
   - any file the trajectory wrote (regardless of whether it matches \
     the requested format)
   - any domain-specific value, citation, statute, KPI, or constant \
     the trajectory used or cited
   - any document structure the trajectory produced (section headers, \
     table columns, field labels, drafted outline)
   - any computed final answer or stated final state
Be GENEROUS in interpreting these. A drafted memo's section list IS \
a document structure. A cited URL IS a citation. A markdown table is \
both a structure and an artifact. If you can identify even one of \
the above, the trajectory has anchors and you should emit a playbook \
that captures them.

(D) THIN RECIPE. Your draft recipe ends up under 400 useful content \
characters (excluding section headers). Even a permissive extractor \
can't make a 200-char playbook help.

Default is EMIT. Only refuse if (A), (B), (C), or (D) clearly \
applies. When uncertain between emit and refuse, EMIT.

CRITICAL GROUNDING RULE — the recipe MUST reflect what the trajectory \
ACTUALLY DID, not what the task prompt asked for. Only list files, \
commands, and values that you can see in the tool-call history. If the \
trajectory did not write a file, do not claim it did. If a step failed, \
record what actually failed and what actually ran. Hallucinated output \
files are the #1 failure mode of this prompt — a downstream agent that \
trusts a false "FINAL ANSWER" line will ship the same gap.

Emit exactly ONE playbook per trajectory. Its `content` field MUST be \
written as a concrete solution recipe, NOT as abstract rules, heuristics, \
or advice.

The recipe MUST include, in this order:

1. TASK SUMMARY — one short line: what the agent was asked to do.

2. INPUT FILES — exact filenames, row/column counts, column names, and \
data shape, copied verbatim from the trajectory. Example: "Population.xlsx: \
1516 rows x 10 columns, cols A..J = [Date, EntityName, EntityCode, Q2, Q3, \
...]".

3. DOMAIN FACTS USED — concrete values, constants, or jurisdiction \
specific data the agent relied on. Example: "UK withholding = 20%, France \
= 15%, Spain = 24%, Germany = 15.825%". Include VALUES, not just names.

4. SOLUTION STEPS — the exact sequence of shell commands, python \
snippets, or tool calls the agent executed successfully. Copy the working \
commands verbatim, preferring the FINAL successful version over any \
earlier failed attempts. Example: "Step 1: run_shell python3 -c 'import \
pandas as pd; df = pd.read_excel(\\"Population.xlsx\\"); ...'".

5. OUTPUT ARTIFACTS ACTUALLY WRITTEN — list ONLY the files that the \
trajectory's tool calls actually wrote to disk. Copy the exact paths \
from the `write_file` / `run_shell` / `create_file` tool results. Do \
NOT list files that the task prompt asked for but the agent never \
produced. If the agent wrote .md files when the task asked for .pdf, \
list the .md files here, verbatim.

6. KNOWN GAPS — explicitly list every deliverable the task prompt \
required that you cannot find in step 5. Format each gap as "MISSING: \
<what> — REASON: <why the trajectory didn't produce it, if visible in \
the trajectory (e.g. pandoc not installed, weasyprint import failed)> \
— FIX: <the minimal concrete action a future agent should take first \
to close the gap, e.g. 'install weasyprint via pip then re-run the \
conversion step'>". If there are no gaps (trajectory fully satisfied \
the task), write "KNOWN GAPS: none". NEVER leave this section blank.

7. FINAL ANSWER STATE — describe the actual final state of the \
workspace at end of trajectory, strictly from what you observed. If \
the trajectory produced a discrete computed value, state it. If the \
trajectory produced partial output, say so explicitly ("two .md files \
written; .pdf/.docx conversion never completed"). NEVER fabricate a \
FINAL ANSWER that does not exist in the trajectory.

Write the recipe as a flat text blob, 600-4000 characters. A good \
recipe is a dense paste of inputs, domain values, exact commands, and \
produced artifacts — not prose. Write it as if you are dictating the \
true state to a future agent who will START by closing the KNOWN GAPS \
section, then verify and output.

`trigger` must be a one-sentence task-type descriptor that matches the \
ORIGINAL task prompt's topic so the same task can match its own recipe. \
Example: "Audit sample selection from a population of KRI measurements \
with quarter-on-quarter variance analysis."

`instruction` must be a <80-word executive summary of the recipe — the \
3-5 key moves a future agent must make, BEGINNING with "First close \
any KNOWN GAPS, then..." when gaps exist.

`rationale` must be one short line: "captures concrete solution path \
AND known gaps from a prior run of this exact task so a future run can \
skip discovery and close the missing deliverables".

Do NOT emit generic rules like "Always validate inputs" or "Verify \
outputs carefully" — they are worthless. Do NOT emit empty playbooks: \
a meaningful trajectory MUST produce a recipe even if it is mostly a \
list of gaps."""


def update_playbook_extractor_prompt(
    url: str = DEFAULT_REFLEXIO_URL,
    api_key: str | None = None,
    prompt: str = GDPVAL_PLAYBOOK_EXTRACTOR_PROMPT,
) -> bool:
    """Overwrite the `default_playbook_extractor` prompt on the backend.

    Reads the current config, replaces the extraction_definition_prompt on
    the one existing `default_playbook_extractor` entry in
    `user_playbook_extractor_configs`, and POSTs the updated config back.
    Preserves every other config field — storage, profile extractors,
    batch sizes, LLM config — so this is safe to call on any reflexio
    backend the benchmark connects to.

    Args:
        url (str): Reflexio backend URL. Defaults to `DEFAULT_REFLEXIO_URL`.
        api_key (str | None): Optional API key. Empty string if unset.
        prompt (str): The new extraction_definition_prompt text. Defaults
            to the GDPVal-tuned prompt above.

    Returns:
        bool: True if the update POST returned success, False if the
            config couldn't be read, no extractor was found, or the POST
            failed.
    """
    client = ReflexioClient(url_endpoint=url, api_key=api_key or "")
    try:
        config = client.get_config()
    except Exception as exc:
        logger.error("Failed to fetch reflexio config: %s", exc)
        return False

    extractors = getattr(config, "user_playbook_extractor_configs", None) or []
    if not extractors:
        logger.error("No user_playbook_extractor_configs on reflexio backend")
        return False

    target = None
    for entry in extractors:
        if getattr(entry, "extractor_name", "") == "default_playbook_extractor":
            target = entry
            break
    if target is None:
        logger.error(
            "default_playbook_extractor not found; available: %s",
            [getattr(e, "extractor_name", "?") for e in extractors],
        )
        return False

    current_prompt = getattr(target, "extraction_definition_prompt", "")
    if current_prompt == prompt:
        logger.info("Playbook extractor prompt already matches target; no update needed")
        return True

    target.extraction_definition_prompt = prompt
    try:
        resp = client.set_config(config)
    except Exception as exc:
        logger.error("Failed to POST updated reflexio config: %s", exc)
        return False

    if isinstance(resp, dict) and resp.get("success") is False:
        logger.error("set_config returned success=False: %s", resp.get("message"))
        return False

    logger.info("Updated default_playbook_extractor prompt (%d chars)", len(prompt))
    return True
