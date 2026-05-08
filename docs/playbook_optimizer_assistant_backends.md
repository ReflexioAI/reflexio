# Playbook Optimizer — Assistant Backends

This doc explains the playbook optimizer's **assistant backend layer**: what an "assistant backend" is, the two backends that exist (HTTP webhook and local script), how they plug into the rest of the optimizer, and where to find each piece in the codebase.

---

## 1. Where this fits in the bigger picture

The playbook optimizer takes an existing playbook ("incumbent") and uses [GEPA](https://github.com/stanford-nlp/gepa) to search for a better wording ("candidate"). To decide if a candidate is actually better, it needs to **run both versions against the real assistant** on the same user turns and compare the resulting conversations.

The "assistant backend" is the thing that turns *(messages, playbooks)* into the next assistant reply. The optimizer doesn't care how that reply is produced — it just needs a callable that respects the contract.

```
                       ┌──────────────────────────────────────────────────────────┐
                       │  TRIGGER (upstream services, not part of this change)    │
                       │                                                          │
                       │   PlaybookAggregator                  PlaybookGeneration │
                       │   ._enqueue_playbook_optimization     Service            │
                       │   (after agent_playbook save)         ._enqueue_user_…   │
                       └─────────────────────┬────────────────────────────────────┘
                                             │ enqueue(org_id, target, callback)
                                             ▼
                       ┌──────────────────────────────────────────────────────────┐
                       │  PlaybookOptimizationScheduler   (singleton, daemon)     │
                       │  • debounce by (org_id, kind, target_id)                 │
                       │  • jitter, then fire callback in a worker thread         │
                       └─────────────────────┬────────────────────────────────────┘
                                             │ callback() = optimize(target)
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                         PlaybookOptimizer.optimize(target)                           │
│                                                                                      │
│  1.  config = configurator.get_config().playbook_optimizer_config                    │
│  2.  assistant = self._create_assistant(config)   ◀── BACKEND SELECTION (§5)         │
│      │  • config.webhook_url        → WebhookAssistant                               │
│      │  • config.assistant_script   → LocalScriptAssistant                           │
│      │  • neither                    → return (no-op, info log)                      │
│      ▼                                                                               │
│  3.  incumbent = storage.get_<agent|user>_playbook_by_id(target.target_id)           │
│  4.  windows   = ScenarioResolver(storage).for_<…>(target.target_id)                 │
│           │   builds ScenarioWindow[] from source_interaction_ids                    │
│  5.  job = storage.create_playbook_optimization_job(...)        ┐                    │
│                                                                 │ persists every     │
│                                                                 │ candidate, eval,   │
│                                                                 │ and GEPA event     │
│  6.  result = gepa.api.optimize(                                │                    │
│           seed_candidate = {playbook_content: incumbent.content},                    │
│           trainset = valset = windows,                                               │
│           adapter   = ReflexioPlaybookGEPAAdapter(                                   │
│                          storage, job_id, incumbent,                                 │
│                          rollout = MultiTurnRollout(assistant), ──────┐              │
│                          judge   = PairwiseJudge(llm_client, …),      │              │
│                       ),                                              │              │
│           callbacks = [_GEPAStorageCallback]   # records on_* events  │              │
│       )                                                               │              │
│                                                                       │              │
│  7.  if best passes commit thresholds:                                │              │
│         _commit_if_allowed → archive incumbent, save successor        │              │
└───────────────────────────────────────────────────────────────────────┼──────────────┘
                                                                        │
                                                                        │ for each
                                                                        │ (candidate,
                                                                        │  window)
                                                                        ▼
              ┌──────────────────────────────────────────────────────────────────┐
              │  ReflexioPlaybookGEPAAdapter.evaluate(batch, candidate)          │
              │                                                                  │
              │  ┌────────────────────────────────────────────────────────────┐  │
              │  │ MultiTurnRollout(incumbent_playbook).run(window)           │  │
              │  │   for user_turn in window.user_turns[:max_turns]:          │  │
              │  │       history.append(user_turn)                            │  │
              │  │       reply = assistant(history, [incumbent])  ──────────┐ │  │
              │  │       history.append(assistant_reply)                    │ │  │
              │  │   → RolloutTrace                                         │ │  │
              │  └──────────────────────────────────────────────────────────┼─┘  │
              │  ┌──────────────────────────────────────────────────────────┼─┐  │
              │  │ MultiTurnRollout(candidate_playbook).run(window)         │ │  │
              │  │   …same loop, same user turns, candidate playbook…       │ │  │
              │  │       reply = assistant(history, [candidate])            ┤ │  │
              │  │   → RolloutTrace                                         │ │  │
              │  └──────────────────────────────────────────────────────────┼─┘  │
              │  ┌──────────────────────────────────────────────────────────┼─┐  │
              │  │ PairwiseJudge.judge(window, incumbent_rollout,           │ │  │
              │  │                     candidate_rollout)                   │ │  │
              │  │   LLM call with playbook_optimizer_judge prompt          │ │  │
              │  │   → JudgeOutput {verdict, score, likert, asi, rationale} │ │  │
              │  └──────────────────────────────────────────────────────────┼─┘  │
              │                                                             │    │
              │  storage.insert_playbook_optimization_evaluation(...)       │    │
              │  on AssistantFailedError → row with verdict="aborted"       │    │
              └─────────────────────────────────────────────────────────────┼────┘
                                                                            │
                                                                            │ both rollouts
                                                                            │ call the SAME
                                                                            │ backend instance
                                                                            │ (so the only
                                                                            │  difference is
                                                                            │  playbook content)
                                                                            ▼
                       ┌──────────────────────────────────────────────────────────┐
                       │   AssistantCallable  (typing.Protocol)                   │
                       │       __call__(messages, playbooks) -> str               │
                       └─────────────────────┬─────────────────┬──────────────────┘
                                             │                 │
                       ┌─────────────────────┴───┐   ┌─────────┴────────────────────────┐
                       │  WebhookAssistant       │   │  LocalScriptAssistant            │
                       │  ───────────────────    │   │  ──────────────────────          │
                       │  requests.post(url,     │   │  subprocess.run(                 │
                       │      json=payload,      │   │      [script_path, *args],       │
                       │      headers={Auth},    │   │      input=json.dumps(payload),  │
                       │      timeout=…)         │   │      capture_output=True,        │
                       │                         │   │      timeout=…)                  │
                       │  retry: 1 + max_retries │   │  retry: 1 + max_retries          │
                       │  backoff: base * 2^a    │   │  backoff: base * 2^a             │
                       │                         │   │                                  │
                       │  raise → Webhook        │   │  raise → LocalScript             │
                       │          FailedError    │   │          FailedError             │
                       └────────────┬────────────┘   └──────────────┬───────────────────┘
                                    │                               │
                                    └────────────┬──────────────────┘
                                                 │ (inherit)
                                                 ▼
                                        AssistantFailedError
                                        (caught by adapter →
                                         verdict="aborted")
```

Reading the diagram top-to-bottom:
- The **upper third** is "how an optimization gets started" — upstream services enqueue work and the scheduler decides when to actually run it.
- The **middle band** is `PlaybookOptimizer.optimize` — the only place that knows about config, scenario windows, and the GEPA loop. Step 2 is where this change lives: a single dispatch into one of the two backends.
- The **lower third** is the inner loop: for every (candidate, window) pair the adapter runs *two* paired rollouts that share the same user turns and the same backend, then a judge LLM scores them. Both rollouts route through whichever `AssistantCallable` was selected at step 2 — that's why the backend layer is the only thing that has to be swappable.

---

## 2. The contract every backend obeys

Both backends are plain callables that match `AssistantCallable` (a `typing.Protocol`):

```python
class AssistantCallable(Protocol):
    def __call__(
        self, messages: list[ChatMessage], playbooks: list[AgentPlaybook]
    ) -> str: ...
```

| Input / Output | Meaning |
|---|---|
| `messages` | Conversation so far. Each `ChatMessage` has `role` (`user`/`assistant`/`system`) and `content`. |
| `playbooks` | Playbooks to inject into the assistant's system context. Today exactly one — the incumbent or candidate version under test. |
| Return value | The assistant's next reply as a plain string. |
| On failure | Raise `AssistantFailedError` (or a subclass). The adapter catches this and records an `aborted` evaluation row instead of crashing the GEPA loop. |

Anything that satisfies this Protocol can be swapped in. The two concrete backends are described below.

---

## 3. The two backends

| Aspect | `WebhookAssistant` | `LocalScriptAssistant` |
|---|---|---|
| Transport | HTTPS POST | `subprocess.run` |
| Where the assistant runs | Wherever the URL points | On the same host as the optimizer |
| Payload format | JSON body | JSON on stdin |
| Response format | JSON body with `{"content": "..."}` | JSON on stdout with `{"content": "..."}` |
| Auth | Optional `Authorization` header | Inherits process env (no separate auth concept) |
| Failure type | `WebhookFailedError` | `LocalScriptFailedError` |
| Common base | `AssistantFailedError` | `AssistantFailedError` |
| Best for | Production / remote deployments | Local dev, CI, offline evaluation |

Both share the same retry/backoff/timeout knobs and the same payload builder (`_build_payload`), so the wire shape is identical regardless of transport.

### 3.1 The shared payload

```json
{
  "messages": [
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "..."}
  ],
  "playbooks": [
    {"id": 42, "content": "<playbook text>", "trigger": "password reset"}
  ]
}
```

For the webhook, this becomes the request body. For the script, this becomes the bytes written to its stdin.

### 3.2 The expected response

```json
{"content": "<assistant reply>"}
```

Anything else — non-string `content`, missing key, malformed JSON, non-zero exit code, network error, timeout — is reported as a typed failure.

### 3.3 Retry behavior (identical for both)

| Setting | Default | Effect |
|---|---|---|
| `webhook_max_retries` | 3 | Total attempts = `max_retries + 1` |
| `webhook_backoff_base_seconds` | 1.0 | Sleep between attempts = `base * 2^attempt` |
| `webhook_timeout_seconds` | 60 | Per-call timeout |

> Naming note: the fields keep the `webhook_*` prefix even though they govern both backends. This is deliberate — renaming would have required a config-schema migration. Treat them as "assistant call" knobs.

---

## 4. How the local-script backend works

This is the new piece. The high-level idea: spawn a child process for each assistant turn, hand it the payload on stdin, and read the reply off stdout.

### 4.1 The command

```python
self.command = [script_path, *script_args]
subprocess.run(self.command, input=stdin, text=True, capture_output=True, timeout=...)
```

- `script_path` is the executable — typically an interpreter (`/usr/bin/python3`) or a binary.
- `script_args` is everything after it — the entrypoint script and any flags.

There is **no shell interpretation** (`shell=True` is not used), so config values cannot inject shell metacharacters.

### 4.2 What the script must do

| Step | Requirement |
|---|---|
| 1. Read stdin | Single JSON object (UTF-8). The `messages` and `playbooks` fields described above. |
| 2. Compute reply | Free-form. Could be an LLM call, a deterministic stub, a replay of prior recordings, etc. |
| 3. Write stdout | Single JSON object containing at least `{"content": "<string>"}`. |
| 4. Exit 0 on success | Any non-zero exit is treated as a failure; stderr is included in the error message. |

### 4.3 Failure modes and how each is reported

| Symptom | What the backend raises |
|---|---|
| Non-zero exit code | `LocalScriptFailedError("...exited with code N: <stderr truncated>")` |
| stdout not valid JSON | `LocalScriptFailedError("...returned invalid JSON: <stdout truncated>")` |
| stdout missing `content` (or non-string) | `LocalScriptFailedError("...returned no content")` |
| `subprocess.TimeoutExpired` | Counts as one failed attempt → final raise is `LocalScriptFailedError("...timed out after Ns")` |
| Any other exception | Counts as one failed attempt → final raise is `LocalScriptFailedError(str(exc))` |

stderr is truncated to 1000 chars before being put into messages so a noisy script can't blow up logs or DB rows.

### 4.4 A minimal working script

```python
#!/usr/bin/env python3
import json, sys

payload = json.loads(sys.stdin.read())
last_user = next(
    (m["content"] for m in reversed(payload["messages"]) if m["role"] == "user"),
    "",
)
print(json.dumps({"content": f"echo: {last_user}"}))
```

Useful as a smoke test — confirms the optimizer end-to-end without needing an actual model.

---

## 5. How a backend is selected at runtime

Selection happens once per `optimize()` call inside `PlaybookOptimizer._create_assistant`:

```python
def _create_assistant(self, config) -> AssistantCallable | None:
    if config.webhook_url:
        return WebhookAssistant(...)
    if config.assistant_script_path:
        return LocalScriptAssistant(...)
    return None
```

| Config state | Result |
|---|---|
| `webhook_url` set, `assistant_script_path` unset | Webhook backend |
| `assistant_script_path` set, `webhook_url` unset | Script backend |
| Both set | **Rejected at config load time** by a Pydantic validator (`ValueError: Configure only one playbook optimizer assistant backend...`) |
| Neither set | Optimizer logs `Skipping playbook optimization: no assistant backend configured` and returns without creating a job |

The check happens *before* loading the incumbent or resolving scenario windows, so an unconfigured optimizer short-circuits cheaply.

---

## 6. Configuration reference

All fields live on `PlaybookOptimizerConfig` (in `reflexio/models/config_schema.py`).

### 6.1 New fields

| Field | Type | Default | Purpose |
|---|---|---|---|
| `assistant_script_path` | `str \| None` | `None` | Absolute path to the executable. Set this to enable the script backend. |
| `assistant_script_args` | `list[str]` | `[]` | Extra argv tokens passed after `assistant_script_path`. |

### 6.2 Existing fields that now apply to both backends

| Field | Type | Default | Notes |
|---|---|---|---|
| `webhook_url` | `str \| None` | `None` | URL for the webhook backend |
| `webhook_auth_header` | `str \| None` | `None` | Sent verbatim as `Authorization` (webhook only) |
| `webhook_timeout_seconds` | `int > 0` | 60 | Per-call timeout — both backends |
| `webhook_max_retries` | `int >= 0` | 3 | Retry budget — both backends |
| `webhook_backoff_base_seconds` | `float >= 0` | 1.0 | Exponential base — both backends |

### 6.3 Validator

```python
@model_validator(mode="after")
def check_single_assistant_backend(self) -> Self:
    if self.webhook_url and self.assistant_script_path:
        raise ValueError(
            "Configure only one playbook optimizer assistant backend: "
            "webhook_url or assistant_script_path"
        )
    return self
```

### 6.4 Example configs

```yaml
# Webhook backend (production)
playbook_optimizer_config:
  enabled: true
  webhook_url: https://assistant.internal/rollout
  webhook_auth_header: "Bearer <secret>"
  webhook_timeout_seconds: 60
  webhook_max_retries: 3

# Script backend (local dev / CI)
playbook_optimizer_config:
  enabled: true
  assistant_script_path: /usr/bin/python3
  assistant_script_args: [/path/to/echo_assistant.py]
  webhook_timeout_seconds: 30
  webhook_max_retries: 1
```

---

## 7. End-to-end workflow

What happens, in order, when a new agent playbook is generated and the optimizer is enabled:

| Step | Component | What it does |
|---|---|---|
| 1 | `PlaybookAggregator._enqueue_playbook_optimization` | After saving new agent playbooks, enqueues each PENDING playbook with the scheduler. |
| 2 | `PlaybookOptimizationScheduler` | Debounces by `(org_id, kind, target_id)`, fires after a small jitter, spawns a daemon thread. |
| 3 | `PlaybookOptimizer.optimize(target)` | Loads config, calls `_create_assistant` → backend instance (or returns early). |
| 4 | `ScenarioResolver` | Builds `ScenarioWindow`s from the playbook's `source_interaction_ids`. |
| 5 | `gepa.api.optimize(...)` | Runs the candidate-search loop. |
| 6 | `ReflexioPlaybookGEPAAdapter.evaluate` | For each `(candidate, window)`: |
|   |   | • `MultiTurnRollout(incumbent).run(window)` → calls backend N times |
|   |   | • `MultiTurnRollout(candidate).run(window)` → calls backend N times |
|   |   | • `PairwiseJudge.judge(...)` → LLM verdict |
|   |   | • Persists `PlaybookOptimizationEvaluation` row |
| 7 | Failure handling | Any `AssistantFailedError` → `verdict="aborted"`, `score=0.0`, GEPA continues. |
| 8 | `PlaybookOptimizer._passes_commit_thresholds` | Checks score / likert / per-window verdict counts. |
| 9 | `PlaybookOptimizer._commit_if_allowed` | If gates pass → archive incumbent, save successor playbook. |

The same flow runs for user playbooks, gated by `optimize_user_playbooks`.

---

## 8. Code map

### Module: `reflexio/server/services/playbook_optimizer/assistant_webhook.py`

(Despite the file name, both backends live here.)

| Symbol | Kind | Role |
|---|---|---|
| `AssistantCallable` | Protocol | Shape both backends satisfy. |
| `AssistantFailedError` | Exception | Common base — caught by the adapter. |
| `WebhookFailedError` | Exception | Webhook backend failures. |
| `LocalScriptFailedError` | Exception | Script backend failures. |
| `_build_payload` | Function | Single source of truth for the wire payload. |
| `_truncate` | Function | Caps stderr / stdout strings included in error messages at 1000 chars. |
| `WebhookAssistant` | Class | HTTP POST + retry. |
| `LocalScriptAssistant` | Class | `subprocess.run` + retry. |

### Other touched files

| File | What changed |
|---|---|
| `reflexio/server/services/playbook_optimizer/optimizer.py` | New `_create_assistant` method; `optimize()` short-circuits when no backend is configured. |
| `reflexio/server/services/playbook_optimizer/gepa_adapter.py` | Catches `AssistantFailedError` (the new common base) instead of `WebhookFailedError` only. |
| `reflexio/models/config_schema.py` | Adds `assistant_script_path`, `assistant_script_args`, plus the mutual-exclusion validator. |
| `tests/models/test_validators.py` | Four new tests for the validator (see §9). |

---

## 9. Tests

### 9.1 Config validation (`tests/models/test_validators.py`)

| Test | Asserts |
|---|---|
| `test_playbook_optimizer_accepts_webhook_backend` | Webhook-only config is valid; `assistant_script_path is None`. |
| `test_playbook_optimizer_accepts_script_backend` | Script-only config round-trips path + args. |
| `test_playbook_optimizer_accepts_no_assistant_backend` | Default config (neither set) is valid; the optimizer is responsible for skipping. |
| `test_playbook_optimizer_rejects_multiple_assistant_backends` | Both set → `ValidationError` matching `"only one"`. |

### 9.2 Adapter / optimizer tests

The existing tests in `tests/server/services/playbook_optimizer/test_playbook_optimizer.py` use a `Mock` for the assistant. Because both backends satisfy the same `AssistantCallable` Protocol, those tests cover the call-site contract for either backend without changes.

---

## 10. Operational notes

| Topic | Notes |
|---|---|
| **Trust boundary** | The script backend executes whatever `assistant_script_path` points to with the same privileges as the optimizer process. Treat it like any other piece of config that names a binary. |
| **Shell injection** | None. The command is a list, `shell=True` is not used. |
| **Credentials in logs** | Only `str(last_error)` from the exception chain is logged. Authorization headers are never logged or echoed back. |
| **Data residency** | Webhook traffic goes wherever `webhook_url` points — operators are responsible for choosing an appropriate endpoint. The script backend is local and never leaves the host. |
| **Process overhead** | One process per assistant turn. Bounded by `max_metric_calls × max_turns × 2` (incumbent + candidate). For a long-lived script that pays a heavy startup cost, see §11. |
| **stderr handling** | Captured, not streamed. Truncated to 1000 chars when included in error messages. |

---

## 11. Future work / known limitations

| Item | Detail |
|---|---|
| Rename `webhook_*` retry/timeout fields | They govern both backends. Renaming requires a config-schema migration; deferred. |
| Long-lived script process | If startup overhead is measurable, replace `subprocess.run` with a persistent process speaking line-delimited JSON. Only `LocalScriptAssistant.__call__` would change. |
| Per-turn working directory | Today the script inherits `cwd`. Passing an explicit `cwd` would help when the script loads local state tied to an org. |
| Structured stderr capture on success | Currently surfaced only on non-zero exit. Capturing it on success too would help telemetry on script-internal warnings. |
