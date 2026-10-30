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
`reflexio/server/llm/_native_cli.py`. Inside an isolated SDK worker, it asks the
API process to start a fresh native-launch owner. Only primitive launch data
crosses that seam; SDK hooks, provider registration, request construction and
interpretation stay in their existing worker with its existing start method.
Outside a completion worker, provider calls retain ordinary captured-text
`subprocess.run` behavior.

The fresh owner executes `_native_cli_process.py` directly with the underlying
Python interpreter and `-I -S`. It imports only the standard library and never
re-executes the embedding application's main module. Native stdin/stdout/stderr
pipes exist only there, so another forked SDK worker cannot inherit their writers
and stall EOF. SDK-worker encoding, environment and working directory are captured
explicitly, including an explicitly empty environment.

The API process reserves references to its existing provider permit and private
attempt directory before starting the controller thread. The controller retains
the owner's actual process handle; the owner retains the immediate CLI's actual
`Popen` handle through communication, timeout kill, wait and stream closure.
Retirement requires both owner exit and its matching terminal cleanup receipt.
An unexpected owner exit, missing/foreign receipt or unconfirmed native exit
retains admission and scratch storage instead of inferring cleanup from exit alone.
The process-local live-owner registry prevents garbage collection from discarding
unresolved files after the caller and controller thread finish. Qualified cleanup
or process restart is required for an unresolved owner; no timed release is added.

Original absolute worker expiry and the existing shorter CLI allowance remain
authoritative, including helper startup. Codex's last-message file shares the
attempt directory and survives SDK-worker death until native cleanup completes.
Closing refuses new launches and best-effort marks the attempt cancelled; a
filesystem failure cannot skip SDK-worker retirement. A newly obtained
native handle after cancellation is killed and reaped. A command already in
communication can continue to its original deadline while its owner retains
capacity; caller retirement does not wait for that cleanup. Uninterruptible OS
startup can return late. The API process still starts the helper with ordinary
`Popen`: on fork-based hosts another SDK worker can inherit its short-lived exec
error pipe and delay the controller's startup observation. The helper independently
owns the native deadline and cleanup; admission remains held until its exit and
receipt are observed. This repair eliminates native stdio inheritance, not every
fork/startup interaction. No startup floor, pool, timing setting or fallback to
unsafe ownership is added.

Requests/replies use atomically published private files and exact request IDs.
Only small request notifications use the existing completion queue; socketpair
notifications wake reply readers without shared multiprocessing locks or a
second large payload pipe. A notification never substitutes for response identity. Socket response expiry
remains a typed hard timeout across the worker boundary, including when the
worker reports expiry before the parent observes its own deadline.

This owns the **immediate CLI**, not arbitrary grandchildren, detached hooks or
remote provider work. A descendant retaining a pipe can prevent stream cleanup;
admission then remains held. Broader containment and real platform qualification
remain tracked in [#878](https://github.com/ReflexioAI/reflexio/issues/878).
Grace remains supported; no configuration reduction is claimed by this repair.
Frozen evaluator identities include native dispatch/retirement and the exact
on-disk helper script bytes, rather than relying only on cached imports;
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
