#!/usr/bin/env node
// Unscoped npm alias for `npx openclaw-smart install`.

import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const cli = require.resolve("@reflexioai/openclaw-smart/scripts/npm-cli.js");
const result = spawnSync(process.execPath, [cli, ...process.argv.slice(2)], {
  stdio: "inherit",
});

process.exit(result.status ?? 1);
