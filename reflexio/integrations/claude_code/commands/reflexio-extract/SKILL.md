---
name: reflexio-extract
description: "Summarize the current conversation — including reasoning, tool calls, corrections, and preferences — and publish to Reflexio for user profile and playbook extraction."
---

# Extract Learnings to Reflexio

Summarize this conversation and publish to Reflexio so future sessions can learn from it.

## What to Do

Review the full conversation above, including your chain-of-thought reasoning, tool calls, tool results, and all user messages. Create a summary that captures:

1. **User corrections** — the user said "no, do X instead" or corrected your approach. Preserve their exact words. **Tool-call rejections count as corrections** — if the user rejected a tool use mid-response, record it as its own turn and note what the rejection was objecting to.
2. **User preferences** — how they like things done, their expertise, communication style, project conventions.
3. **Environment facts** — project setup, tools, constraints, team conventions, schema details, column types, undocumented quirks.
4. **Procedural signals** — moments where something went wrong and you learned from it. **These are the primary source of behavioral rules** — without them, Reflexio can only extract vague profile entries, not precise playbook rules:
   - **Failed tool calls with their error messages verbatim** (invalid identifier, file too large, permission denied, timeouts, JSON parse errors, etc.) — the exact error string is load-bearing evidence
   - **Self-corrections mid-response** — moments you wrote "actually…", "this isn't quite X but…", or "I should have…" (these are evidence the user would have valued catching the mistake earlier)
   - **Anomalies and implausible results** — mean/median divergence, unexpected zeros, row-count mismatches, results that contradict documentation
   - **Retries and the reason for each retry** — not just the final successful call
5. **Non-obvious learnings** — surprises, dead ends, workarounds, documentation gaps.

Skip routine pleasantries, but do **NOT** skip failures, rejections, or anomalies — these are the highest-signal moments. If a section of the conversation contains only successful tool calls and no friction, it probably yields domain facts; if it contains friction, it probably yields behavioral rules. Both layers matter.

## How to Publish

1. Ensure the server is running **(local server only)**:

Check the `REFLEXIO_URL` environment variable. If it points to a remote server (not `localhost` or `127.0.0.1`), skip this step — managed Reflexio servers are always running.

For local servers (`REFLEXIO_URL` unset or pointing to localhost/127.0.0.1):
```bash
reflexio status check
```
If not running, start it:
```bash
nohup reflexio services start --only backend > ~/.reflexio/logs/server.log 2>&1 &
sleep 5
```

2. Write the summary as a JSON file. **Pattern-match on the shape below**: a single assistant turn captures one coherent learning; failed attempts plus the eventual success live together as an ordered list in `tools_used` so the failure → recovery arc reads as one unit. Self-corrections belong **verbatim** inside the `content` field:

```bash
cat > /tmp/reflexio-extract.json << 'EXTRACT_EOF'
{
  "agent_version": "claude-code",
  "source": "claude-code-expert",
  "interactions": [
    {"role": "user", "content": "how many live streams had a giveaway in the last 6 months?"},
    {
      "role": "assistant",
      "content": "Tried to query live_streams with a channels join to filter by business type; discovered live_streams has no channel_id column. Fell back to SELECT * LIMIT 1 for schema introspection, then rewrote the query without the channels join. Got 588 streams. NOTE: mid-response I wrote 'this is not true CVR' — I had substituted engagement_rate for conversion rate without flagging the substitution upfront.",
      "tools_used": [
        {"tool_name": "mcp__snowflake-prod__run_snowflake_query", "tool_data": {"input": "SELECT ... JOIN channels ... — FAILED: invalid identifier 'L.CHANNEL_ID'"}},
        {"tool_name": "mcp__snowflake-prod__run_snowflake_query", "tool_data": {"input": "SELECT * FROM live_streams LIMIT 1 — schema introspection"}},
        {"tool_name": "mcp__snowflake-prod__run_snowflake_query", "tool_data": {"input": "Rewritten query without channels join — succeeded, returned 588"}}
      ]
    }
  ]
}
EXTRACT_EOF
```

Notice the three things the example models: (1) the `content` field names the failure, the recovery, **and** the self-correction verbatim; (2) all three tool attempts — failed, introspection, success — are listed in order under the **same** `tools_used` so the arc is one learning unit; (3) the failed attempt's `input` ends with `— FAILED: <exact error>` so the error string survives intact.

`tools_used` is **required** for any assistant turn containing a failed, rejected, or retried tool call. The error message and the offending input are what make behavioral rules extractable — include them verbatim. For purely successful tool calls that only yielded a domain fact already captured in the `content` field, `tools_used` is optional.

3. **Self-check before publishing.** Scan your JSON and ask:
   1. Does it contain at least one failed tool call, user rejection, or self-correction? If the original conversation had any friction and your summary has none, re-read the conversation for the failure moments you dropped.
   2. For each failed tool call, is the **actual error message present verbatim** (not paraphrased)? Error strings like `invalid identifier 'L.CHANNEL_ID'` are what let Reflexio extract behavioral rules — paraphrases lose the signal.
   3. Are retries grouped under the **same** assistant turn as an ordered list, so the failure → recovery arc reads as one learning?

   If any answer is no, revise the JSON before running the publish command below. A re-extraction after publish is possible but costs a round trip.

4. Publish with forced extraction:
```bash
reflexio publish --agent-version claude-code --source claude-code-expert --skip-aggregation --force-extraction --file /tmp/reflexio-extract.json && rm -f /tmp/reflexio-extract.json
```

The publish returns as soon as the server has accepted the payload — the actual extraction (LLM calls + storage writes) runs as a background task on the server. This avoids the gateway timeouts you'd hit if extraction took longer than the deployment's request budget.

5. Report what was published (brief summary to the user) and tell them the extraction is running in the background. They can verify the results a minute later with (replace `claude-code` below with the `REFLEXIO_USER_ID` configured during setup if different):
```bash
reflexio user-profiles list --user-id claude-code --limit 10
reflexio user-playbooks list --user-id claude-code --limit 10
```

## Summary Guidelines

- **Preserve user corrections and tool rejections verbatim** — their exact words (or the rejection moment) are the highest-signal input.
- **Preserve failures, not just successes.** For every failed tool call, record the error message and the input that caused it. For every retry, record why you retried. A summary with zero failures in a conversation that had friction is an incomplete summary.
- **Preserve self-corrections.** If you wrote "this isn't quite X" or "I should have done Y" in the original conversation, that sentence belongs in the extraction verbatim — it is evidence of a rule the user values.
- **Organize as meaningful pairs** — each user/assistant pair should capture one coherent learning. Multiple failed attempts + the eventual success belong to the **same** assistant turn, listed in order in `tools_used`, so the failure → recovery arc is preserved as a single learning unit.
- **Include reasoning context** — why you took an approach, what you expected, what surprised you.
- **Be concise, but not at the cost of dropping failures.** Cut pleasantries and repeated narration, not error messages.
