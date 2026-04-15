---
name: reflexio-extract
description: "Extract reusable playbooks from the current conversation — corrections, tool failures, and successful recipes — and upsert them into Reflexio via direct CRUD. No LLM call on the Reflexio server."
---

# Extract Learnings to Reflexio

You (the agent running this command) apply the extraction rubric below to
the full conversation in your current context, then write each resulting
playbook into Reflexio via the `reflexio user-playbooks` CLI. The Reflexio
server does NO LLM work for this integration — extraction happens in your
own session, which is why OpenClaw's Reflexio setup requires no LLM
provider API key on the server side.

**The rubric embedded below is fully self-contained. Do NOT attempt to open
any external file** — the canonical source at
`reflexio/server/prompt/prompt_bank/playbook_extraction_context/v3.0.0.prompt.md`
is a maintainer reference inside the `reflexio-ai` source tree and is
**not shipped** with the ClawHub skill bundle (the publish script stages
only `integrations/openclaw/`). Everything you need to produce valid
playbooks is inline in this file.

## Step 1 — Ensure the local Reflexio server is running

```bash
reflexio status check
```

If it fails with a connection error, tell the user you're starting the
local Reflexio server in the background, then run:

```bash
nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 &
sleep 5
```

## Step 2 — Apply the extraction rubric to your conversation

Review the full conversation in your context — every user message, your
own assistant turns, every tool call and tool result. Produce a JSON array
of playbook entries.

**Two extraction categories** — a single trajectory can yield both:

### Category 1: Correction SOPs

Extract a Correction SOP when ALL are true:
1. You (the agent) performed an action, assumption, or default behavior.
2. The user signaled it was incorrect, inefficient, or misaligned.
3. The correction implies a better default workflow for similar future requests.
4. The rule can be phrased as: *"When [user intent/problem], the agent should [policy]."*

Valid correction signals:
- User correcting or rejecting your approach
- User redirecting you to a different mode or level of detail
- User clarifying expectations that contradict your behavior
- Tool-call rejection — user rejected a tool use mid-response (record in tools_used verbatim)
- Self-correction written out loud — you wrote "actually, this isn't quite right" mid-response
- Repeated tool failure with user intervention — you failed the same operation twice and the user redirected

**Trigger quality — the "Skill Test":** a valid trigger describes the
**problem or situation**, NOT the user's explicitly-stated preference.
- BAD (tautological): `"User requests CLI tools"` — restates the ask
- BAD (topic-based): `"User talks about Python code"` — too broad
- BAD (interaction-based): `"User corrects the agent"` — too generic
- GOOD (intent-based): `"User requests help debugging a specific error trace"`
- GOOD (problem-based): `"User's initial high-level request is ambiguous"`
- GOOD (situation-based): `"User reports timeout failures on large data transfers (>10TB)"`

**Tautology check:** if the trigger reduces to "user asks for X" and the
instruction is "do X", the entry is tautological — re-derive the real
trigger as the *problem or situation* you encountered.

`instruction` for a Correction SOP is **< 20 words**.

### Category 2: Success Path Recipes

Extract a Success Path Recipe when ALL are true:
1. You completed a task successfully (produced deliverables, resolved the request).
2. The trajectory contains domain-specific work — computation, data
   transformation, multi-step orchestration — not just conversation.
3. The solution path contains at least one of:
   - **Domain formulas, constants, or parameter values**
   - **Specific tool sequences that worked**
   - **Input/output format specifics that mattered**
   - **Concrete values or answers you computed**
   - **Key decisions you made and why**

A Success Path Recipe does NOT require a user correction. It captures
"what worked" from a successful trajectory.

**Trigger for a recipe** describes the task type — domain + action:
- GOOD: `"Audit sample selection from a risk-metrics spreadsheet with multi-criteria filtering"`
- BAD (too generic): `"Spreadsheet analysis task"`
- BAD (copies task verbatim): `"Calculate sample size for audit testing..."`

`instruction` for a recipe can be **up to 80 words** and MUST include
specific formulas, tool sequences, parameter values, or computed answers.
`content` must be an **actionable recipe** — a future agent reading it
should be able to short-circuit discovery by following it verbatim.

### Blocking issues (optional)

If you could not complete the task because a capability was missing, also
populate a `blocking_issue` field with one of: `missing_tool`,
`permission_denied`, `external_dependency`, `policy_restriction`. The
`instruction` must still be an executable workaround (inform the user,
suggest alternatives) — NOT the missing capability itself.

### Output schema

For each extracted playbook, produce an object with these fields:

```json
{
  "rationale": "1-2 sentences: for a Correction SOP, what implicit expectation was violated. For a Recipe, why this captures transferable value.",
  "trigger": "Situation/condition for a Correction SOP OR task-type descriptor for a Recipe. Must NOT be a tautological restatement of the user's explicit preference.",
  "instruction": "For a Correction SOP: < 20 words. For a Recipe: up to 80 words with specific values/tools/formulas.",
  "pitfall": "Optional — the specific behavior or assumption to avoid.",
  "content": "For a Correction SOP: concise standalone insight. For a Recipe: actionable recipe with concrete formulas, tool sequences, parameter values, column names, and computed answers from the trajectory.",
  "playbook_name": "agent_corrections for Correction SOPs, success_recipes for Recipes"
}
```

**Evidence requirements — do NOT drop these in favor of pleasantries:**
- Preserve user corrections **verbatim** in the `content` field.
- Preserve tool failures and error messages **verbatim** — exact strings like `invalid identifier 'L.CHANNEL_ID'` are what make the rule extractable.
- Preserve self-corrections ("actually, this isn't quite right because...") **verbatim**.
- Retries belong together with their eventual success — describe the failure → recovery arc as one learning unit in `content`.

**How many entries to return:** one per distinct Correction SOP, plus one
per distinct Success Path Recipe when the trajectory contains substantive
domain work. A successful trajectory with real domain work MUST yield at
least one recipe. Return zero entries only when the conversation is
trivially short (fewer than 4 non-trivial agent actions).

Never split a single policy across multiple entries; never merge two
independent policies into one.

## Step 3 — For each extracted playbook, find-or-upsert

For each entry you produced in Step 2, run this sequence:

### 3a. Search for a close match

```bash
reflexio user-playbooks search "<trigger>" --agent-version openclaw-agent --limit 3
```

Add `--json` if you want to parse the result programmatically. The search
returns up to three candidates ranked by semantic similarity.

### 3b. Decide: update or add

Read the returned candidates. Apply the same semantic reasoning you used
to extract the new entry:

- **If a candidate's trigger + content clearly describe the same
  situation and the same rule/recipe**, treat it as a hit. Pick its `id`
  and update it:
  ```bash
  reflexio user-playbooks update \
    --playbook-id <id> \
    --content "<merged content>"
  ```
  The merged `content` must **preserve the existing rule and add new
  evidence or refinement**. Do NOT replace the existing content
  wholesale — the point of updating rather than adding is to strengthen
  an existing rule, not to overwrite it.

- **If no candidate clearly describes the same situation**, add a new
  entry:
  ```bash
  reflexio user-playbooks add \
    --agent-version openclaw-agent \
    --playbook-name <agent_corrections|success_recipes> \
    --content "<content>" \
    --trigger "<trigger>" \
    --instruction "<instruction>" \
    --pitfall "<pitfall, or omit if none>" \
    --rationale "<rationale>"
  ```

When in doubt, prefer adding a new entry. Updating is for clear refinements
of an existing rule, not for fuzzy matches.

## Step 4 — Report what you did

Briefly tell the user:
- How many entries you extracted
- How many were added vs. updated
- One-sentence summary of the most important new rule or recipe

The user can verify with:

```bash
reflexio user-playbooks list --agent-version openclaw-agent --limit 10
```

## Summary Guidelines

- **Preserve user corrections and tool rejections verbatim** — their exact words are the highest-signal input.
- **Preserve failures, not just successes.** For every failed tool call, record the error message and the input that caused it. A summary with zero failures in a conversation that had friction is an incomplete summary.
- **Preserve self-corrections verbatim** — they are evidence of a rule the user values.
- **Do the search before every add.** The update path is what keeps the playbook store from bloating with near-duplicates as the same pattern recurs across sessions.
- **Be concise, but not at the cost of dropping failures.** Cut pleasantries and repeated narration, not error messages or computed values.
