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
