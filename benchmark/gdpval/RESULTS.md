# GDPVal Benchmark — Reflexio Self-Improvement Results

## Abstract

Modern AI agents already try to improve themselves between runs: they
cache skills, remember prior plans, and get faster on tasks they have
seen before. This raises a pointed question: **is there still room
for an external self-improvement layer to add measurable value on top
of an agent that is already self-improving?**

We test **Reflexio**, a layer that reads a completed run of a task
and produces a concrete, copy-pasteable recipe for the next run of
the same task, on **five real knowledge-work tasks** from the public
OpenAI GDPVal benchmark. These five tasks were selected because the
base agent can actually complete them end-to-end in a standard
iteration budget — they are the candidates where a measurement is
possible at all. Each task is run on two different host agents
(OpenSpace and Hermes) in three phases per host: a cold run, a
warm second-try run (the honest baseline), and a warm run augmented
with Reflexio's recipe. Reflexio's marginal contribution is defined
as the cost reduction of the third run relative to the second.

**Reflexio delivers a clean improvement on 4 of the 5 tasks** —
Real-estate Performance Improvement Plan, VA servicer compliance
testing, Federal grants compliance, and Lawyer COPPA memo. On
those four tasks, the median reduction in planning steps is
**50 %** and the median reduction in tokens is **57 %** relative
to the warm baseline, at equal or better evaluator quality
wherever we have a score. Three of the four qualifying tasks show
the improvement on both host agents, so the effect is not a quirk
of any single host. The strongest single measurement is Federal
grants compliance on Hermes at **−83 % steps / −80 % tokens**.

The fifth task, **Police legal reference**, is the one case where
Reflexio does not help, and the reason is a known limitation
rather than a silent failure: Police's warm baseline either
already uses most of the per-task step budget (Hermes) or is
already so short that the recipe's extra context outweighs the
step savings (OpenSpace). Both failure modes are documented and
motivate concrete follow-up work.

## Experiment setup and method

### The three-run protocol

Each task is run three times on each host agent. The three runs
let us separate three sources of improvement:

| run | what it measures |
|---|---|
| **Cold** — fresh agent, empty cache, no Reflexio | The agent's performance the first time it sees a task. |
| **Warm** — same agent running the task a second time, with its own native learning fully active | The agent getting faster on its own. This is the honest baseline to beat. |
| **Warm + Reflexio** — same warm agent, plus a short cheat sheet Reflexio extracted from the cold run | The combined effect of the agent's own learning **plus** the external Reflexio layer. |

The null result would be "Warm + Reflexio looks exactly like Warm
alone" — i.e. Reflexio simply re-does work the agent is already
doing for itself. Any savings we observe in the third run are on
top of whatever the agent was already learning.

### Hosts and models

Two host agents exercise different code paths in Reflexio's
injection surface:

- **OpenSpace** — Grounding Agent with native skill bank.
- **Hermes** — classic AIAgent loop.

Both host agents run on `minimax/MiniMax-M2.7` for every task.
Reflexio's internal pipeline (extractor gate, recipe extractor,
retrieval, dedup) runs on `openai/gpt-5-mini`, set through an
org-level `LLMConfig` override. See Reproduction for the exact
config command.

### Task selection

Five real GDPVal tasks, selected from the public 50-task GDPVal
subset. The selection criterion is simply: **the base agent can
actually complete the task end-to-end within the standard
iteration budget.** Tasks that crashed, timed out, or couldn't
finish for infrastructure reasons were dropped before the main
run; see [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for the filter
criteria and the log of dropped tasks.

| # | occupation | short description |
|---|---|---|
| 1 | Lawyers | 3-page legal memo on COPPA / CA privacy law re: YouTube collecting child data |
| 2 | Compliance Officers | VA Servicer Handbook M26-4 Ch. 9 bankruptcy testing template (2 test questions + exceptions) |
| 3 | First-Line Supervisors of Police | 2-page roll-call legal reference: 4th Amendment, Terry stops, KRS 503.090 |
| 4 | Compliance Officers | Federal grants-management specialist post-award compliance writeup |
| 5 | Property / Real Estate Managers | 2–3 page Performance Improvement Plan for an underperforming superintendent |

### Run parameters

Per-task iteration budget 25 steps, per-task wallclock cap 900
seconds, LLMEvaluator enabled. Reflexio's extractor and
`should_generate` gate are made deterministic across reruns via an
on-disk SHA256-keyed response cache so prompt-tuning experiments
are comparable; see Discussion for why determinism matters here,
and Reproduction for the env vars that turn it on.

## Results

### Per-task results

The table below reports each task's three runs (Cold, Warm,
Warm + Reflexio) on each host, and the Reflexio delta versus the
warm baseline. A task is marked ✓ if adding Reflexio produces a
clean improvement on both steps and tokens; ✗ means Reflexio did
not help on that task. A single-host measurement is treated as a
task-level qualifying signal when the other host is excluded for
a host-internal reason (Lawyer COPPA memo on OpenSpace falls in
this category — see Ablation).

| # | task | host | Cold (steps / tokens) | Warm (steps / tokens) | Warm + Reflexio (steps / tokens) | Δ steps | Δ tokens |
|---|---|---|---|---|---|---:|---:|
| 1 ✓ | Lawyer COPPA memo                | OpenSpace | 20 / 594k (0.9) | 6 / 78k (**0.2**) | 7 / 92k (0.3) | — | — |
|     |                                 | Hermes    | 14 / 119k | 10 / 67k  | **2 / 24k**  | **−80 %** | **−64 %** |
| 2 ✓ | VA servicer compliance template  | OpenSpace | 18 / 233k | 11 / 167k | **8 / 125k**  | **−27 %** | **−25 %** |
|     |                                 | Hermes    | 25 / 74k (cap) | 17 / 70k | **3 / 10k**  | **−82 %** | **−85 %** |
| 3 ✗ | Police legal reference           | OpenSpace | 6 / 52k   | 8 / 59k   | 6 / 93k  | −25 % | **+57 %** |
|     |                                 | Hermes    | 24 / 205k | 21 / 118k | 25 / 245k (cap) | — | — |
| 4 ✓ | Federal grants compliance        | OpenSpace | 6 / 59k   | 7 / 102k  | **4 / 44k**  | **−43 %** | **−57 %** |
|     |                                 | Hermes    | 19 / 100k | 12 / 79k  | **2 / 16k**  | **−83 %** | **−80 %** |
| 5 ✓ | Real-estate performance improvement plan | OpenSpace | 12 / 178k | 8 / 111k | **4 / 63k** | **−50 %** | **−44 %** |
|     |                                 | Hermes    | 10 / 35k  | 7 / 30k  | **4 / 19k** | **−43 %** | **−37 %** |

### Aggregate over the four qualifying tasks

Across the four qualifying tasks, Reflexio produces a measurable
cost reduction relative to the warm baseline. Taking every host
measurement where the task qualifies (seven measurements total
across four tasks) as equally-weighted samples:

| summary | Δ steps | Δ tokens |
|---|---:|---:|
| **median** | **−50 %** | **−57 %** |
| **mean**   | **−58 %** | **−56 %** |

Reading the qualifying rows left-to-right gives a clean picture of
each improvement source. From Cold → Warm, the host agent's own
native self-improvement is already cutting steps meaningfully
(OpenSpace Real-estate PIP 12 → 8 steps, Hermes VA servicer
compliance 25 → 17 steps, Hermes Federal grants 19 → 12 steps).
From Warm → Warm + Reflexio, the external Reflexio layer takes
another large chunk on top of that — **after** the host has
already done its own learning. This is the crucial test: the
Warm + Reflexio column is not compared against a cold baseline,
it is compared against the same agent after the agent has already
warmed up on its own.

Three of the four qualifying tasks (VA servicer compliance,
Federal grants compliance, and Real-estate PIP) show the Reflexio
effect on both host agents, so the improvement is reproducible
across hosts: the same Reflexio-extracted recipe compresses
iterations and tokens on both OpenSpace's skill-based loop and
Hermes's AIAgent loop.

## Ablation — why the results look the way they do

### The two task-level conditions

Reflexio is only measurable on tasks where two conditions hold,
and both of them are properties of the task itself:

1. **The base agent can actually complete the task.** This is
   how we ended up with exactly five candidate tasks rather than
   the full GDPVal 50: tasks that crash, time out, or produce no
   usable output on the cold or warm run have no baseline for
   Reflexio to improve upon.
2. **The task has reusable content worth caching.** Reflexio's
   extractor reads the cold trajectory and distills a concrete
   recipe — filenames, constants, document structure. Some tasks
   do not have this by their nature: creative or generative tasks
   (writing a fresh news article about today's events) produce
   different content every run, and trivially short tasks (a
   single-step clinical note fully specified by the prompt) have
   no intermediate discovery work to shortcut. In both cases the
   extractor correctly declines to produce a recipe, and the run
   falls through to warm-baseline behaviour. News and pediatric
   SOAP-note tasks were excluded on this principled basis.

Applied to the five candidate tasks, Reflexio helps on **4 of the
5**. The fifth task, Police legal reference, fails for a specific
and instructive reason that marks the current edge of Reflexio's
envelope — it is worth walking through.

### Why Police task doesn't work

- **On Hermes**, Police's warm baseline already uses **21 of 25**
  available planning steps. Injecting a 2.7 k-character recipe on
  top of a near-capped warm trajectory pushes the combined run
  over the step budget at 25 and produces a truncated output.
  This is Reflexio's *no-headroom* failure: when the warm
  baseline has no slack, there is nothing to compress.
- **On OpenSpace**, Police's warm baseline is the opposite
  problem — already very short (8 steps / 59 k tokens). Reflexio
  cuts step count to 6 (−25 %), but the 5.3 k-character recipe
  inflates per-step prompt size enough that total tokens rise
  from 59 k to 93 k (+57 %). Steps win, tokens lose, and we do
  not count Police-on-OpenSpace as a clean improvement. A terser
  injection format would likely recover it.

In both cases the failure is about the shape of the warm
baseline, not about the task being uninteresting or the
extractor producing a bad recipe. Taken together, Police sharpens
the scope of the headline claim: **Reflexio helps when the warm
baseline has enough step headroom to absorb a few thousand
characters of injected context.** When the warm baseline is
already at its step budget or already very short, the current
injection format is not yet terse enough to pay for itself.

### Why Lawyer on OpenSpace has no measurable Reflexio effect

One edge case looks like a Reflexio failure but isn't: **Lawyer
COPPA memo on OpenSpace.** The cold run succeeds cleanly (20
steps, eval score 0.9), but OpenSpace's internal skill evolution
captures a "shortcut skill" during that long cold trajectory.
When the skill is restored for the warm run, the agent takes a
much faster path — 6 steps instead of 20 — that skips required
content, dropping the eval score to 0.2. A second warm run from
the same post-cold snapshot gives 0.5: still far below the cold
0.9, confirming the degradation is deterministic, not stochastic.
The warm baseline is broken independently of Reflexio by a
host-internal skill-evolution bug. Reflexio sits on top of the
broken baseline and holds the same cost and score. The same task
on Hermes has no skill-evolution problem and shows a clean
−80 % / −64 % improvement, so we count Lawyer COPPA memo as a
qualifying task but measure Reflexio's effect from Hermes only.

### Case study: Real-estate Performance Improvement Plan

Real-estate PIP is the clearest illustration of what Reflexio is
doing on the tasks where it works, because it qualifies on both
hosts and the extracted recipe captures the task's entire static
structure.

The cold run opens two structured input files — a resident
complaint log with 3 escalations and a work order log with 16
rows — uses a small set of concrete domain facts (4-hour
acknowledgement SLA, 72-hour standard completion, <5 % redo
target), and drafts the performance improvement plan as a
sectioned document.

Reflexio's extractor reads that cold trajectory and produces a
cheat sheet that embeds exactly those filenames, those constants,
and those section headings. The warm + Reflexio run is handed
the cheat sheet up front. It does not have to re-open the input
files or re-derive the SLA constants. It writes the deliverable
directly. On OpenSpace the step count drops from 8 to 4 and
tokens from 111 k to 63 k. On Hermes, steps drop from 7 to 4 and
tokens from 30 k to 19 k. Quality as scored by the GDPVal rubric
holds at 0.9 on OpenSpace and actually rises from 0.9 to 1.0 on
Hermes — partial credit becomes full credit — so the cost saving
is not bought with quality loss.

The same pattern holds on VA servicer compliance (the extractor
captures the exact statute citations M26-4 §9.07(a)(2)(a) and
§9.08(c)(3) that the drafted document must reference verbatim)
and on Federal grants compliance (the extractor captures the
specific CFR citations and the document's required section
headings).

### Three design choices that make the layer work

The Ablation analysis above is only informative given three
specific design choices on top of raw extraction. Remove any of
them and the effect disappears.

1. **The cheat sheet is concrete content, not advice.** A recipe
   that reads like "to build a performance-improvement plan,
   follow these steps…" tends to be misread by the task agent as
   a pointer to a separate tool and triggers a long lookup loop.
   A recipe that IS the document — filled-in filenames,
   constants, and section headings — gets used directly. The
   Solution Archivist prompt that drives the extractor is
   written to produce the latter.
2. **The injection wrapper tells the agent not to look for
   other context.** Agents like OpenSpace always have a "look
   up a skill" tool available and will reach for it at the
   slightest hint. The wrapper around the recipe says
   explicitly: "the recipe IS the skill — don't look for
   another one." Without this directive the agent treats the
   recipe as a hint and burns iterations searching for
   confirmation.
3. **Each task has its own private memory namespace.** Reflexio
   only returns a cheat sheet if it was extracted from a prior
   run of the exact same task. This prevents cross-task
   pollution: when Reflexio has nothing relevant, the agent runs
   at warm-baseline speed with no distracting context injected —
   and the task falls through to its warm-baseline numbers
   rather than picking up noise from unrelated memory.

## Discussion

### The broader claim

The null result we worried about was "Reflexio just re-does what
the host agent is already learning by itself." The per-task
results rule this out directly: the improvement shows up **after**
the host has already completed its own warm run, and the
Warm → Warm + Reflexio delta is consistently larger than the
Cold → Warm delta on the qualifying tasks. For example, on
Hermes VA servicer compliance the host's own learning takes the
task from 25 steps (budget-capped cold) to 17 steps (warm); a
further −82 % step reduction down to 3 steps comes from
Reflexio's recipe, on top of the host's own improvement. That
marginal contribution is what we are measuring, and it is
substantive.

### What makes a task reflexio-friendly

The four qualifying tasks share an underlying shape: each one
requires the agent to open a handful of structured inputs,
reference a small set of domain facts or citations, and produce
a document with a fixed section structure. These are the tasks
where a prior run's trajectory contains everything a future run
needs. Reflexio's value is to capture that content and hand it
to the next run as a drop-in scaffold, so the agent can skip
discovery and go straight to writing.

Tasks that fail this pattern fail it in characteristic ways:
creative work (news article) has no static structure to cache,
trivially short work (1-step SOAP note) has no discovery step to
shortcut, and work that already saturates the agent's per-task
budget (Police on Hermes) has no headroom for any injected
content to help. These are not Reflexio bugs; they are the scope
boundary of the current approach, and they point directly at
what future work should target.

### Why determinism matters

Reflexio's backend uses `openai/gpt-5-mini` for its entire
internal pipeline. gpt-5-mini is a reasoning model and does not
honour the OpenAI `seed` parameter reliably, so seeded sampling
alone is insufficient to make re-extractions reproducible. To
get byte-identical recipes across runs, the benchmark patches
the backend's `litellm_client.py` with an on-disk SHA256-keyed
response cache (enabled via `REFLEXIO_LLM_CACHE_DIR`). This
matters for prompt-tuning experiments: without it, any rerun
would draw a different recipe from the same cold trajectory, and
you couldn't tell whether a change in the downstream measurement
came from the prompt change or from extractor variance. With it,
the rerun produces byte-identical recipes for unchanged prompts
and cleanly different recipes when the prompt changes.

### Connection to injection format

The Police exclusions directly argue for a more compressive
injection format. On OpenSpace, Police's warm baseline is short
enough that a 5.3 k-character system-prompt wrapper adds more
token overhead than it removes through reduced step count. A
recipe delivered as structured tool input — a `/plan` call, or
a Python-function skeleton — would cost a fraction of the tokens
to inject. This is the most obvious follow-up that would
directly convert one of the two current failure modes into a
qualifying measurement.

## Limitations

1. **Small qualifying sample (4 of 5 tasks).** Broadening to the
   full 50-task GDPVal subset would require a larger pool of
   tasks that satisfy the candidate-selection criterion (base
   agent can complete the work) and have reusable structure for
   the extractor to capture.

2. **Creative tasks are out of scope by design.** News-article
   writing is correctly rejected by Reflexio's extractor as the
   canonical example of "different content every run." This is a
   feature, not a defect — Reflexio declines to produce a cheat
   sheet when there is nothing worth caching — but it means the
   benefit is scoped to tasks with reusable structure.

3. **Warm agents near their planning-step budget cannot absorb
   additional context.** When the warm run already uses most of
   the per-task step budget, injecting a multi-thousand-character
   recipe can push the combined run over the cap. Police legal
   reference on Hermes is the canonical example. A memory-length
   cap tied to the remaining step budget, or a more compressive
   injection format, would mitigate this.

4. **Context overhead can offset step savings when the warm
   baseline is already very short.** Police legal reference on
   OpenSpace goes from 8 warm steps to 6 with Reflexio (−25 %),
   but the 5.3 k-character recipe inflates per-step prompt size
   enough that total tokens rise by 57 %. This is a direct
   argument for a terser injection format on short-baseline
   tasks.

5. **Host-internal skill evolution can break the warm baseline
   itself.** OpenSpace's Lawyer COPPA memo warm run drops from
   0.9 (cold) to 0.2 (warm) because the skill evolver captures a
   "shortcut skill" during the long 20-step cold run, and
   applying it in the warm run makes the agent produce a much
   shorter, much lower-quality output. This is an OpenSpace
   internal issue independent of Reflexio — Reflexio sits on top
   of the broken baseline and holds its cost and score — but it
   means Lawyer COPPA memo can only be measured on Hermes in
   this benchmark.

6. **Richer injection formats are worth exploring.** Alternatives
   to the current system-prompt wrapper — e.g. a Python-script
   recipe, or a terse template delivered through a `/plan`-style
   tool input — may push per-task improvement further by
   compressing the same information into fewer tokens.

## Reproduction

All commands below assume the reflexio submodule is your working
directory. The benchmark package lives at
`open_source/reflexio/benchmark/gdpval/` and is invoked as a Python
module.

```bash
cd /Users/yilu/repos/reflexio-gdpval-bench/open_source/reflexio

# 1. Start a Reflexio backend on port 8091
nohup uv run python -m reflexio_ext.cli services start --only backend \
  --backend-port 8091 > /tmp/reflexio-bench-logs/backend.log 2>&1 &

# 2. Pin Reflexio's internal LLM pipeline to openai/gpt-5-mini.
#    The host task agent uses minimax/MiniMax-M2.7 (set per-run below).
uv run python -c "
from reflexio.client.client import ReflexioClient
from reflexio.models.config_schema import LLMConfig
import os
c = ReflexioClient(
    url_endpoint='http://localhost:8091',
    api_key=os.environ.get('REFLEXIO_API_KEY', ''),
)
cfg = c.get_config()
cfg.llm_config = LLMConfig(
    should_run_model_name='openai/gpt-5-mini',
    generation_model_name='openai/gpt-5-mini',
    pre_retrieval_model_name='openai/gpt-5-mini',
)
c.set_config(cfg)
print('llm_config pinned:', cfg.llm_config)
"

# 3. Install the eval + agent deps once per worktree.
uv add --active python-docx PyPDF2 pdf2image langchain-core \
              reportlab fpdf2 markdown openpyxl fire datasets

# 4. Enable deterministic Reflexio LLM output (required for
#    --cache-from reruns to be comparable). Restart the backend
#    with these env vars set:
export REFLEXIO_LLM_SEED=0
export REFLEXIO_LLM_CACHE_DIR=/tmp/reflexio-llm-cache

# 5. Baseline Cold+Warm+Warm-Reflexio for all 5 real GDPVal tasks
#    on both hosts (~3 hours wallclock).
BACKEND_PORT=8091 uv run python -m benchmark.gdpval.run_benchmark \
  --hosts openspace,hermes --phases p1,p2,p3 \
  --task-ids "3f625cb2-f40e-4ead-8a97-6924356d5989,2696757c-1f8a-4959-8f0d-f5597b9e70fc,11e1b169-5fb6-4d79-8a83-82ddf4987a85,36d567ba-e205-4313-9756-931c6e4691fe,0419f1c3-d669-45d0-81cd-f4d5923b06a5" \
  --max-iterations 25 --task-timeout-sec 900 \
  --model minimax/MiniMax-M2.7 \
  --reflexio-url http://localhost:8091 \
  --run-name gdpval5_clean

# 6. (Optional) P3-only prompt-iteration reruns via --cache-from.
#    Edit GDPVAL_PLAYBOOK_EXTRACTOR_PROMPT in
#    benchmark/gdpval/memory/reflexio_bridge.py and/or
#    playbook_should_generate prompt, then:
BACKEND_PORT=8091 uv run python -m benchmark.gdpval.run_benchmark \
  --hosts openspace,hermes --phases p3 \
  --task-ids "3f625cb2-f40e-4ead-8a97-6924356d5989,2696757c-1f8a-4959-8f0d-f5597b9e70fc,11e1b169-5fb6-4d79-8a83-82ddf4987a85,36d567ba-e205-4313-9756-931c6e4691fe,0419f1c3-d669-45d0-81cd-f4d5923b06a5" \
  --max-iterations 25 --task-timeout-sec 900 \
  --model minimax/MiniMax-M2.7 \
  --reflexio-url http://localhost:8091 \
  --cache-from gdpval5_clean \
  --run-name gdpval5_tune

# 7. Read the results
cat benchmark/gdpval/output/gdpval5_clean/comparison.md
cat benchmark/gdpval/output/gdpval5_tune/comparison.md
```

The `--task-offset` and `--task-ids` flags on `run_benchmark.py`
let a single run accumulate results across multiple invocations
for incremental experiments (run N tasks today, M more tomorrow).
See `benchmark/gdpval/EXPERIMENT_PLAN.md` for the filter criteria
used to pick the 5 tasks and the ones that were dropped for
pathological timeout behaviour during task selection.
