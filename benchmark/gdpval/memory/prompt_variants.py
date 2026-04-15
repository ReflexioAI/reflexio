"""P3 prompt iteration variants — iteration session scratchpad.

This module holds alternate `GDPVAL_PLAYBOOK_EXTRACTOR_PROMPT` + injection
header pairs that can be swapped into `reflexio_bridge.py` /
`injection.py` during the iteration loop (see
`/Users/yilu/.claude/plans/ethereal-wiggling-pike.md`).

Each variant is a stand-alone string. When iterating, copy the chosen
variant into the actual call sites — don't import from here in production
code, this is a living reference sheet.
"""

from __future__ import annotations


# ============================================================================
# v1: Success Recipe — current production (committed in 383af22)
# ============================================================================
# Philosophy: extractor emits a dense solution recipe (input schemas,
# working commands, output artifacts, final answer). Injection header
# frames it as CACHED SOLUTION to re-run directly.
# Live in: reflexio_bridge.py:GDPVAL_PLAYBOOK_EXTRACTOR_PROMPT (v1)
#          injection.py:_SOLUTION_HEADER (v1)

# ============================================================================
# v2 DRAFT: Even more compressive — demand a Python script as the recipe
# ============================================================================
# Philosophy: if the recipe is a runnable Python script the agent just
# has to execute it + write outputs, ideally in 1-2 iterations.

V2_EXTRACTOR_PROMPT = """\
You are a solution-archivist. You are given ONE agent trajectory that \
solved ONE specific task. Produce ONE playbook containing a SELF-CONTAINED \
PYTHON SCRIPT that, when executed in the task's workspace, reproduces the \
final answer byte-for-byte — no investigation required.

`content` format (STRICTLY):

```
# TASK: <one-line description>
# FINAL ANSWER: <the computed answer if discrete>
# INPUT FILES: <filename: shape>
# OUTPUT FILES: <filename: shape>

<runnable python code — 40-400 lines — that reads inputs, computes, \
writes outputs>
```

The Python block MUST:
- Use only pandas, openpyxl, json, pathlib, numpy, or the standard library.
- Hardcode concrete column names, sheet names, row counts, formulas with \
literal values, and output filenames observed in the trajectory.
- Produce the same output files the trajectory's successful run produced.
- End with a `print()` statement that emits the final answer.

`trigger`: one-sentence task-type descriptor (domain + action).
`instruction`: <30-word summary — "run the recipe python block verbatim \
in the workspace, then verify outputs match expected schemas".
`rationale`: "runnable cached solution; no re-derivation needed".

Do NOT emit advice, heuristics, or rules. Emit working code. If the \
trajectory did not reach a clear end state, emit the closest working \
reproducer you can construct."""

V2_INJECTION_HEADER = """\
# RUNNABLE CACHED SOLUTION

A prior agent successfully solved this exact task. Below is a Python
recipe it produced — a self-contained script that reads the task inputs
and writes the expected outputs. Your job: execute this script verbatim
in the workspace, verify the output files exist and have the expected
shape, then complete the task.

Do NOT re-derive the approach. Do NOT redesign the script. Just run it,
check outputs, and report completion."""


# ============================================================================
# v3 DRAFT: Minimal template — single terse line per decision
# ============================================================================
# Philosophy: if v1 verbose recipes still confuse the agent, try a
# spartan, almost JSON-like format that removes ambiguity.

V3_EXTRACTOR_PROMPT = """\
Extract a CACHED SOLUTION from this successful agent trajectory. Emit \
ONE playbook whose `content` is a strictly-formatted record:

TASK: <one line>
INPUTS:
- <filename>: <rows>x<cols>, cols=[...]
DOMAIN_FACTS:
- <k>=<v>
STEPS:
1. <exact shell or tool call verbatim>
2. <exact shell or tool call verbatim>
...
OUTPUTS:
- <filename>: <rows>x<cols>, sheets=[...]
FINAL: <discrete answer if any>

Be terse and literal. Copy commands from the trajectory verbatim. \
No prose, no advice, no "consider X". If a step has multiple attempts \
in the trajectory, use the FINAL successful version only.

`trigger`: one-line task type.
`instruction`: <20-word summary.
`rationale`: "cached solution".
"""

V3_INJECTION_HEADER = """\
## CACHED SOLUTION

You have run this task before. Below is a literal record of what worked
last time. Execute the STEPS in order, produce the OUTPUTS, report the
FINAL answer. Do not deviate unless a step errors."""
