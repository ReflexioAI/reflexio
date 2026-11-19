# Native CLI model selection

`REFLEXIO_NATIVE_CLI_MODEL` is an optional operator-owned native target. Its
presence authorizes that exact host/model and supplies the fallback model for
all five text roles. Without it, model requests cannot activate the machine's
Claude Code, Codex or OpenCode account. An installed executable grants nothing.

```dotenv
REFLEXIO_NATIVE_CLI_MODEL=local-cli/codex/your-native-model
```

Supported shapes:

| Reference                                | Selection                                        |
| ---------------------------------------- | ------------------------------------------------ |
| `local-cli/claude-code`                  | Claude Code's configured default                 |
| `local-cli/codex/your-native-model`      | Explicit Codex model                             |
| `local-cli/opencode/provider/model#high` | Explicit OpenCode model and variant              |
| `local-cli/opencode/#high`               | OpenCode's default model and an explicit variant |

A host-only reference delegates to the CLI's own default. An explicit suffix
`/default` is the literal model ID `default`; it is not a default-selection
alias. The model remainder is preserved, including nested slashes. Empty explicit
model components and unknown hosts are rejected. Native routes provide no
inferred context/output capacity. Reported served-model provenance comes only
from the CLI response, never from the requested reference.

Native OpenCode discovers declared options from successful `run --help` output
on either stdout or stderr. A CLI declaring `--standalone` receives the full
`provider/model#variant` reference with its private-server flag. The recognized
`--attach`/`--variant` contract receives separate model and variant arguments;
a missing variant or whitespace-only explicit model terminates selection before
model execution. A leading `#` selects a variant while delegating the model to
OpenCode. This form requires the CLI to advertise `--variant`. A host-only
reference supplies neither argument. Discovery and execution share the original deadline.
Model and variant values are bound as `--name=value`, so option-shaped text
cannot select another server or alter other invocation flags.

Model resolution retains explicit organization/role choices, applicable custom
gateway inheritance and sitevar choices ahead of this native fallback. Embeddings
retain their own model. Explicit HTTP roles remain HTTP. An unavailable selected
native executable does not cause API-key autodetection to choose a paid provider.
An explicitly configured fallback ladder may advance after an authorized
operational failure; authorization and adoption errors are terminal, including
when LiteLLM or a spawned worker wraps them.

On Windows, native OpenCode resolves npm's global and project-local
opencode.cmd launchers when they target the installed package's bin/opencode.exe,
as in opencode-ai 1.19.6. For an older package layout, upgrade or select the
native executable directly with CLAUDE_SMART_CLI_PATH. Custom batch wrappers
can still use the compatibility bridge.

Known OpenCode compatibility bridges receive the selected native model in their
child environment, with stale bridge model/variant hints cleared. A host-only
reference delegates to OpenCode's own configuration. The parent environment and
independently invoked optimizer scripts retain their separate learning policy.

The native host owns its internal retry policy. Reflexio inherits its environment
without forcing a Claude retry count; an unset value delegates to the CLI's
default. This replaces the former backend-only override of three retries and
can change default retry behavior. The existing SDK invocation deadline and
native process owner still bound the managed call. They do not promise a fixed
number of HTTP requests or a cost cap inside an opaque CLI invocation.

Frozen native judge plans bind the effective home directory and the inherited
`CODEX_HOME` and `CLAUDE_CONFIG_DIR` overrides, alongside the native target,
working directory and executable settings. Changing these after job creation
requires a new plan; a directory override must not silently select another
account midway through evaluation. Only a digest is recorded: Reflexio does not
read or copy account credentials for this check. This detects launch-context
drift, not changes to files inside those directories, credential rotation or
changes to the model served by a host default.

Each native request must match the operator's full reference before role or
gateway rewriting, before admission, and again inside the provider. An explicit
`custom_llm_provider` cannot turn a reserved `local-cli/` reference into HTTP.
`OPENCLAW_SMART_USE_LOCAL_CLI` independently grants the separate OpenClaw route;
this target neither enables nor replaces it. An explicitly configured optimizer
`LocalScriptAssistant` also retains its own script/compile authority.

## Explicit upgrade from the previous native selectors

This SDK changes native model selection. It does not migrate a user's files or
upgrade installed plugin bundles. Do not point a released plugin at this SDK
without its matching producer migration and qualification. Existing bundles
retain their pinned SDK until explicitly upgraded.

For a manually configured SDK native backend:

1. Stop the old backend launcher before editing its selected configuration.
   Record the effective old enable, host, physical CLI model and executable
   selection, including inherited shell values and saved role choices.
2. Deliberately choose one full native reference. If the CLI previously selected
   its own default, use the host-only form. Preserve an explicit physical model
   in the suffix. Do not infer it from an old `claude-code/*` role label: those
   suffixes were opaque labels and did not select a physical model.
3. Replace backend `CLAUDE_SMART_USE_LOCAL_CLI` and `CLAUDE_SMART_CLI_MODEL`
   assignments with the new reference, clearing inherited obsolete assignments
   too. A new reference alongside either obsolete selector requires repair,
   including an old explicit opt-out. To keep HTTP-only backend operation,
   leave the new reference unset and keep explicit HTTP role choices.
4. Replace saved native `claude-code/*` role references with the deliberately
   chosen full reference. Preserve independent HTTP, embedding and OpenClaw
   choices. Legacy native labels are rejected rather than reinterpreted.
5. Keep `CLAUDE_SMART_CLI_PATH` and `CLAUDE_SMART_CLI_TIMEOUT` if needed; these
   remain independent executable and timeout policies. Keep client coding-host
   identity where the client needs it: the SDK backend no longer reads
   `CLAUDE_SMART_HOST` to select its runtime. Existing explicit OpenCode bridge
   protocols remain supported with an OpenCode backend reference, independently
   of the coding host. A known bridge cannot override a different selected
   backend runtime. Retired Codex bridges still require repair.
6. Restart and verify the intended native route/model. A host-only reference
   does not pin or prove the model physically served.

Do not regenerate backend policy from every coding session's host. An optimizer
script may use its separately saved learning provider/model while the backend
uses HTTP; the new native backend profile must not enable that script or alter
its consent, custom webhook or script settings.
