# GDPVal 10-task Clean Benchmark Plan

Goal: produce a defensible README claim that Reflexio adds measurable value over
MiniMax-M2.7-powered Hermes and OpenSpace host agents on real GDPVal tasks.

Done = aggregate P2→P3 improvement on ≥7 of 10 qualifying tasks with no quality
regression, reproducible from a single benchmark invocation.

## Success criteria (gates)

An iteration passes if, across the qualifying subset (tasks where P1 eval score
≥ 0.5 AND the should_generate gate extracted at least one user_playbook):

- **Hermes**: mean P3 iter ≤ 0.80 × mean P2 iter AND mean P3 tokens ≤ 0.80 ×
  mean P2 tokens AND no task's P3 eval score drops by > 0.1 from its P2 score.
- **OpenSpace**: same, with the same thresholds.
- **Degenerate tasks** (not qualifying) must show P3 iter/tok within ±10% of P2
  — Reflexio must not *hurt* when it has nothing to contribute.

If success criteria fail on the first run, the plan enters an extractor-prompt
optimization loop (Phase E). After ≤ 4 optimization iterations, if no variant
wins, stop and write a root-cause analysis in RESULTS.md instead of a claim.

## Phase A — Infrastructure fixes (one-time, ~15 min)

All already applied (done during debugging):
- [x] `python-docx` installed in submodule venv (fixes eval skipping on DOCX artifacts)
- [x] `fire` installed in submodule venv (fixes Hermes `run_agent.py` import)
- [x] `render_memory_block` no longer includes `profiles` (fixes cross-task leak)
- [x] Extractor prompt rewritten with CRITICAL GROUNDING RULE + KNOWN GAPS
- [x] `playbook_should_generate` v2 active (success-path track)
- [x] Reflexio LLMConfig pinned to `openai/gpt-5-mini`

Still to do:
- [ ] **A.1** Set `temperature=0` on the Reflexio backend LLMConfig for
  should_run / generation / pre-retrieval so extractor decisions are
  deterministic across re-runs. Without this, `--cache-from` reruns draw
  different playbook pools from the same trajectory.
- [ ] **A.2** Verify determinism by extracting the same cached trajectory
  twice and confirming the playbook content matches.

## Phase B — Task selection (one-time, ~10 min)

Load the GDPVal 50-task subset (`/Users/yilu/repos/OpenSpace/gdpval_bench/tasks_50.json`).
Pick 10 tasks that satisfy ALL of:

1. No hard format requirement in the prompt (no `\b(pdf|docx|xlsx|pptx|powerpoint|excel|word document)\b`, case-insensitive)
2. Prompt length ≤ 3500 chars (longer prompts thrash on the 25-iter budget)
3. ≤ 2 reference files (more than 2 adds I/O + discovery iterations that overwhelm the budget)
4. Not in the explicitly-broken task category: audio production (music), video editing, full software system design + implementation

Prefer diversity: at most 2 tasks per occupation, span ≥ 5 sectors.

Save the selected task IDs + short names to
`benchmark/gdpval/EXPERIMENT_PLAN.md` as a table (edit this file in place).

## Phase C — Clean baseline run (~90-120 min wallclock)

Single invocation, fresh `--run-name`, all hosts and phases:

```bash
cd /Users/yilu/repos/reflexio-gdpval-bench/open_source/reflexio
BACKEND_PORT=8091 uv run python -m benchmark.gdpval.run_benchmark \
  --hosts openspace,hermes --phases p1,p2,p3 \
  --task-ids "<10 task IDs from Phase B>" \
  --max-iterations 25 --task-timeout-sec 900 \
  --model minimax/MiniMax-M2.7 \
  --reflexio-url http://localhost:8091 \
  --run-name gdpval10_baseline
```

Expected: 10 tasks × 3 phases × 2 hosts = 60 executions. At ~2-4 min each
serial, wallclock is 2-4 hours.

## Phase D — Analysis (~10 min, subagent-owned)

After the run completes, compute:

1. **Per-task P1/P2/P3 iter, tokens, eval score, memory_injected_chars, status,
   playbook count from the extractor log.** One row per (task, host, phase).
2. **Qualifying-subset aggregate**: mean and median P2→P3 delta (iter, tok,
   score) over tasks where P1 score ≥ 0.5 AND extractor produced ≥1 user_playbook.
3. **Non-qualifying degenerate check**: for each task NOT in the qualifying
   subset, is P3 within ±10% of P2 on iter and tok?
4. **Quality regression check**: any P3 score < P2 score - 0.1 is a red flag.
5. **Decision**: pass/fail per success criteria above.

## Phase E — Extractor prompt optimization loop (only if D fails)

Each iteration (budget: 4 iterations max):

1. Read the injected memory for the 3 tasks with the weakest P2→P3 delta. Look
   at the actual playbook content that was rendered into the agent's context.
2. Identify the failure mode: too vague? too long? missing a key piece? wrong
   format for the host agent to consume? hallucinated content?
3. Edit `GDPVAL_PLAYBOOK_EXTRACTOR_PROMPT` in
   `benchmark/gdpval/memory/reflexio_bridge.py` and push to backend via
   `update_playbook_extractor_prompt()`.
4. P3-only `--cache-from gdpval10_baseline` rerun with a new run name
   (`gdpval10_vN`).
5. Re-run Phase D analysis.

Stop and write up findings at first iteration that passes OR at 4 iterations
without a pass.

## Parallelism / subagent assignment

- **Subagent 1 (Explore)**: Phase A.1 + A.2 determinism work (set temperature=0,
  verify by double-extracting a cached trajectory).
- **Subagent 2 (Explore)**: Phase B task selection (filter 50-subset, pick 10).
- [Main loop] Launch Phase C benchmark in background once Subagents 1 and 2
  both report back.
- **Subagent 3 (general-purpose)**: Phase D analysis, fired after Phase C
  results.jsonl is complete.
- **Subagent 4 (general-purpose)**: if Phase E triggers, draft 2-3 prompt
  variants based on the specific failure modes Subagent 3 identified.

Main agent keeps state of the plan and iterates between Phase D decision and
Phase E prompt edits.

## Exit conditions

- **Pass**: single run_dir with all 60 executions + results.jsonl + a summary
  table per host. RESULTS.md gets rewritten with the 10-task numbers. README
  claim can quote specific aggregate deltas.
- **Fail**: 4 optimization iterations exhausted without passing success
  criteria. RESULTS.md gets a "what we tried and why it didn't move the
  needle" section. README claim stays the minimum-viable version (pilot on 2
  tasks, inconclusive on the rest) and points at RESULTS.md for the detailed
  failure analysis.

## Selected tasks (filled in by Phase B)

Note on filter 1: the literal regex `\b(pdf|docx|xlsx|pptx|powerpoint|excel|word document|wav|mp3|mp4)\b`
only leaves 4 tasks from the 50-subset that also survive the other filters — too few
to hit n=10. Filter 1 was relaxed to its *intent*: "no output format that requires
special binary tooling to produce or evaluate." Tasks where the format keyword appears
only in an input reference, a URL, or a quoted filename are kept. Tasks that demand a
text deliverable "in PDF format" / "Word document format" are also kept when the
content is pure text (memo / checklist / report) — agents can produce the text and
the eval rubric scores content, not packaging. Excluded: any task that explicitly
requires a functional spreadsheet (multi-tab Excel with live filters/formulas),
slideshow, or audio/video asset — those would thrash on the 25-iter budget.

Funnel from the 50-task subset:
- Start: 50 tasks
- After strict F1 (no format keyword anywhere in prompt): 6
- After F2 (prompt ≤ 3500 chars): 5
- After F3 (refs ≤ 2): 4
- After F4 (not broken category): 4
- After relaxed F1 (format kw not a hard binary output requirement) + F2+F3+F4: 10

| # | task_id | occupation | sector | prompt chars | refs | short description |
|---|---------|------------|--------|--------------|------|-------------------|
| 1 | 3baa0009-5a60-4ae8-ae99-4955cb328ff3 | News Analysts, Reporters, and Journalists | Information | 996 | 0 | Write 300-500 word article on World Bank June 2025 Global Economic Prospects report |
| 2 | 3f625cb2-f40e-4ead-8a97-6924356d5989 | Lawyers | Professional, Scientific, and Technical Services | 1043 | 0 | Draft 3-page legal memo on COPPA / CA privacy law re: YouTube collecting child data |
| 3 | 2696757c-1f8a-4959-8f0d-f5597b9e70fc | Compliance Officers | Government | 1585 | 0 | Create two VA Servicer Handbook test questions + exception statements for bankruptcy testing |
| ~~4~~ | ~~46b34f78~~ | ~~Financial and Investment Analysts~~ | ~~Finance and Insurance~~ | ~~1975~~ | ~~1~~ | **DROPPED after Phase C attempt 2** — 900s wallclock timeout in OpenSpace P1, 0 files written. The reference file is a 2.7MB Research Material.docx that OpenSpace thrashes on. |
| ~~5~~ | ~~27e8912c~~ | ~~Administrative Services Managers~~ | ~~Government~~ | ~~2007~~ | ~~0~~ | **DROPPED after Phase C attempt 1** — the task's prompt body says "in PDF format" which the filter missed. OpenSpace thrashes on weasyprint/pandoc install, hits the 900s outer timeout, and (due to an adapter bug) corrupts its internal `_task_done` Event so every subsequent task fails with "OpenSpace is busy" cascading errors. Running with 9 tasks instead. |
| 6 | 11e1b169-5fb6-4d79-8a83-82ddf4987a85 | First-Line Supervisors of Police and Detectives | Government | 2174 | 0 | 2-page roll-call legal reference on 4th Amendment, Terry stops, KRS 503.090 |
| ~~7~~ | ~~02314fc6~~ | ~~General and Operations Managers~~ | ~~Retail Trade~~ | ~~2207~~ | ~~0~~ | **DROPPED after Phase C attempt 2** — OpenSpace P1 at 687s elapsed with 0 files written, headed for 900s timeout when kill was issued. Likely same PDF-thrash pattern as ergonomics. |
| 8 | 36d567ba-e205-4313-9756-931c6e4691fe | Compliance Officers | Government | 2265 | 0 | Federal grants-management specialist: post-award compliance determination writeup |
| 9 | 0419f1c3-d669-45d0-81cd-f4d5923b06a5 | Property, Real Estate, and Community Association Managers | Real Estate and Rental and Leasing | 2790 | 2 | 2-3 page PIP for a underperforming super, grounded in work-order log + complaint log |
| 10 | 0112fc9b-c3b2-4084-8993-5a4abb1f54f1 | Nurse Practitioners | Health Care and Social Assistance | 2902 | 0 | Pediatric post-concussion assessment + SOAP note for 16yo skateboard fall |

CLI string (9 tasks, ergonomics removed): 3baa0009-5a60-4ae8-ae99-4955cb328ff3,3f625cb2-f40e-4ead-8a97-6924356d5989,2696757c-1f8a-4959-8f0d-f5597b9e70fc,46b34f78-6c06-4416-87e2-77b6d8b20ce9,11e1b169-5fb6-4d79-8a83-82ddf4987a85,02314fc6-a24e-42f4-a8cd-362cae0f0ec1,36d567ba-e205-4313-9756-931c6e4691fe,0419f1c3-d669-45d0-81cd-f4d5923b06a5,0112fc9b-c3b2-4084-8993-5a4abb1f54f1

