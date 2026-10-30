# Isolated completion worker ownership

The generation caller owns one admitted provider call and one isolated worker.
Receiving a result or abandoning a queue read does not prove the worker exited.
Releasing the provider semaphore at that point could admit another request while
the earlier worker remains alive. A failed process start also used to leave its
queue outside the cleanup boundary.

All paths now retire the worker in `finally`. A worker that already exited is
joined and closed immediately. An unstarted worker closes without joining. A
still-live worker is killed and reaped outside the caller's latency, while a
retained reference keeps its existing provider permit unavailable. Queue/process
handles close after the join; then the reaper releases its reference. The caller
releases only its own reference. Nested provider scopes keep separate ownership,
including an unknown provider inside a capped provider scope.

There is no new pool, timing setting, retry or fixed cleanup wait. The former
one-second join/terminate waits disappear. Provider capacities, bounded admission,
the fallback/repair walk, queue-drain-before-reap and response/error/usage snapshots
are unchanged. Failure to reap a worker retains admission and logs the cleanup
failure rather than allowing capacity to be reused without proof of exit.

The subsequent single-deadline design retires
`REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS` and rejects its presence, including
blank assignments. One configured attempt allowance now covers startup, provider
I/O, response transport and acceptance after admission; startup reduces the
remaining provider timeout. No fixed padding replaces the setting. A former
30-second timeout plus5seconds of grace becomes a30-second attempt, not an
automatic35-second conversion. Remove obsolete assignments and choose the
existing generation/role timeout deliberately.

The worker-ownership repair itself removed no configuration input; retirement
belongs to the later absolute-deadline design. Neither design claims that killing
a worker cancels remote accepted work or arbitrary descendant processes. Parent
parsing and validation are checked cooperatively, and asynchronous retirement
retains capacity until real worker exit. Tests use synthetic providers, actual
isolated workers and local HTTP; no paid-provider or deployed qualification is
implied.

## Tool-call history after transport

A provider object can survive pickling while its deferred Pydantic serializer
remains uninitialized in the receiving process. Do not call its `model_dump()`
to build assistant tool history. Use `normalize_tool_call_for_history` from
`reflexio.server.llm.tools`, which reads the existing ID, type, function name and
arguments into a plain wire dictionary. The ordinary tool loop and audited
playbook reconciliation share this conversion. Tool validation, audit limits,
provider admission and worker ownership remain with their existing owners.

Audited reconciliation states its existing read-only tool allowance in each
stage's instructions: at most three calls across all tools per response, for
two tool rounds, followed by the final schema-valid JSON answer. A direct final
answer is allowed; these are ceilings, not required searches. Malformed calls
consume the same allowance. The request and enforcement use the same constants,
and resumed conversations retain their consumed rounds and physical-call budget.
Independent auditing and provider admission remain unchanged. Instruction clarity
can reduce avoidable retries; it does not guarantee model compliance or a latency SLO.
