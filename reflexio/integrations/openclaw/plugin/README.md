# openclaw-smart

openClaw plugin: cross-session memory via a local Reflexio backend.

## Install

```bash
npx openclaw-smart install
```

The unscoped `openclaw-smart` package is a thin alias for this scoped plugin
package, `@reflexioai/openclaw-smart`.

If your OpenClaw version supports npm package specs in `plugins install`, you
can install the plugin package directly:

```bash
openclaw plugins install @reflexioai/openclaw-smart
```

For users who already have Reflexio installed, the guided setup remains:

```bash
reflexio setup openclaw
```

## Commands

```bash
openclaw-smart install
openclaw-smart repair
openclaw-smart uninstall
openclaw-smart uninstall --purge
```

`install` registers `reflexio-openclaw-smart` with OpenClaw, enables typed hook
access, writes `OPENCLAW_BIN` and `OPENCLAW_SMART_USE_LOCAL_CLI=1` to
`~/.reflexio/.env`, warms Python dependencies, and verifies the plugin is loaded.
