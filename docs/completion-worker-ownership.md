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

`REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS` remains supported and unchanged. It is
a real margin between provider/socket timeout and the process deadline, not the
removed cleanup wait. Replacing it with a constant or zero would lose policy.
A future single-deadline design must also qualify custom providers' descendant
process cleanup before retiring that input. This repair proves isolated-worker
ownership; it does not claim that killing a worker cancels work already accepted
by a remote provider, or that it terminates arbitrary descendant processes.

No configuration input is removed by this repair. Tests use synthetic providers
and processes; no paid provider or deployed qualification is implied.

## Immediate native CLI ownership

Claude, Codex and OpenClaw launch commands through `run_native_cli` in
`reflexio/server/llm/_native_cli.py`. In an isolated completion worker, it asks the
parent to launch and reap the immediate native process. SDK hooks, provider
request construction and response interpretation stay isolated in the worker.
A worker killed at its deadline no longer destroys the only Python owner of its
CLI. Outside a completion worker, the same provider calls retain ordinary
captured-text `subprocess.run` behavior.

The parent reserves references to the existing provider permit and private
attempt directory before starting a native thread. Worker exit releases neither
native ownership nor its files. The native owner alone handles `Popen`, timeout
kill, wait and stream closure. It uses the original absolute worker deadline,
tightened by the CLI's existing shorter allowance; startup spends that same time.
Codex's last-message file uses the attempt directory, so its lifetime survives
worker termination. Files are removed after the worker and native owners finish.

Closing the attempt refuses new launches. If closure races process creation, the
newly obtained process is killed and reaped. A process already in communication
can continue until its original deadline while its owner retains capacity;
caller retirement does not wait for that cleanup. Uninterruptible OS startup or
unconfirmed exit retains capacity rather than returning it without proof. If exit
evidence fails, the process-local live-owner registry retains the actual
process handle and scratch directory after caller and worker threads end. This
fail-closed state requires qualified cleanup or process restart; no timed release
is added. Delegated commands without input inherit null stdin, matching the
isolated worker instead of reading the server terminal.

Requests/replies use atomically published private files and exact request IDs.
Only small request notifications use the existing completion queue; socketpair
notifications wake reply readers without shared multiprocessing locks or a
second large payload pipe. A notification never substitutes for response identity.

This owns the **immediate CLI**, not arbitrary grandchildren, detached hooks or
remote provider work. A descendant retaining a pipe can prevent stream cleanup;
admission then remains held. Broader containment and real platform qualification
remain tracked in [#878](https://github.com/ReflexioAI/reflexio/issues/878).
Grace remains supported; no configuration reduction is claimed by this repair.
Frozen evaluator identities include native dispatch and retirement implementation;
previous execution receipts are not silently reinterpreted.

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
