#!/usr/bin/env node
// npm-facing installer for openclaw-smart.
//
// This intentionally mirrors the core non-interactive install steps from
// `reflexio setup openclaw` so npm and Python installs do not drift.

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const PLUGIN_ID = "reflexio-openclaw-smart";
const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = resolve(SCRIPT_DIR, "..");
const REFLEXIO_DIR = join(homedir(), ".reflexio");
const REFLEXIO_ENV = join(REFLEXIO_DIR, ".env");
const STALE_EXTENSION_DIR = join(homedir(), ".openclaw", "extensions", PLUGIN_ID);

function usage() {
  console.log(`openclaw-smart

Usage:
  openclaw-smart install
  openclaw-smart uninstall [--purge]
  openclaw-smart repair

Install registers the bundled OpenClaw plugin, enables typed hook access,
writes OPENCLAW_BIN to ~/.reflexio/.env, warms dependencies, and verifies
that OpenClaw loaded the plugin.`);
}

function fail(message, code = 1) {
  console.error(`openclaw-smart: ${message}`);
  process.exit(code);
}

function run(argv, opts = {}) {
  const result = spawnSync(argv[0], argv.slice(1), {
    encoding: "utf8",
    stdio: opts.stdio ?? "pipe",
    ...opts,
  });
  if (result.error) {
    return {
      status: 1,
      stdout: result.stdout ?? "",
      stderr: result.error.message,
    };
  }
  return {
    status: result.status ?? 1,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

function findExecutable(name) {
  const candidates = [];
  if (process.env.PATH) {
    for (const dir of process.env.PATH.split(process.platform === "win32" ? ";" : ":")) {
      if (!dir) continue;
      candidates.push(join(dir, name));
      if (process.platform === "win32") candidates.push(join(dir, `${name}.cmd`));
      if (process.platform === "win32") candidates.push(join(dir, `${name}.exe`));
    }
  }
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  return null;
}

function resolveOpenClawBin() {
  // Absolutize the result: this value is written to ~/.reflexio/.env and later
  // read by backend subprocesses that may run from a different cwd, so a
  // relative OPENCLAW_BIN or relative PATH entry must not leak through.
  const configured = process.env.OPENCLAW_BIN;
  if (configured && existsSync(configured)) return resolve(configured);
  const found = findExecutable("openclaw");
  return found ? resolve(found) : found;
}

function shellQuote(value) {
  return `"${String(value).replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
}

function upsertEnv(envPath, updates) {
  mkdirSync(dirname(envPath), { recursive: true });
  const existing = existsSync(envPath) ? readFileSync(envPath, "utf8").split(/\r?\n/) : [];
  const remaining = existing.filter((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) return true;
    return !Object.keys(updates).some((key) => trimmed.startsWith(`${key}=`));
  });
  for (const [key, value] of Object.entries(updates)) {
    remaining.push(`${key}=${shellQuote(value)}`);
  }
  writeFileSync(envPath, `${remaining.filter((line) => line.length > 0).join("\n")}\n`);
}

function removeEnvKeys(envPath, keys) {
  if (!existsSync(envPath)) return;
  const kept = readFileSync(envPath, "utf8")
    .split(/\r?\n/)
    .filter((line) => !keys.some((key) => line.trim().startsWith(`${key}=`)))
    .filter((line) => line.length > 0);
  writeFileSync(envPath, kept.length ? `${kept.join("\n")}\n` : "");
}

function ensurePluginRoot() {
  if (!existsSync(join(PLUGIN_ROOT, "openclaw.plugin.json"))) {
    fail(`plugin root is incomplete: ${PLUGIN_ROOT}`);
  }
}

function runOpenClaw(cli, args, opts = {}) {
  return run([cli, ...args], opts);
}

function printCommandFailure(label, result) {
  const detail = (result.stderr || result.stdout || "").trim();
  console.error(`openclaw-smart: ${label} failed${detail ? `: ${detail}` : ""}`);
}

function runSmartInstall() {
  const script = join(PLUGIN_ROOT, "scripts", "smart-install.sh");
  if (!existsSync(script)) {
    console.warn(`openclaw-smart: ${script} missing; skipping dependency warmup`);
    return;
  }
  const result = run(["bash", script], { stdio: "inherit" });
  if (result.status !== 0) {
    console.warn(
      `openclaw-smart: smart-install.sh exited ${result.status}; first session may bootstrap dependencies`,
    );
  }
}

function inspectLoaded(cli) {
  const result = runOpenClaw(cli, ["plugins", "inspect", PLUGIN_ID]);
  return result.status === 0 && /Status:\s*loaded\b/.test(result.stdout);
}

function install() {
  ensurePluginRoot();
  const cli = resolveOpenClawBin();
  if (!cli) fail("openclaw CLI not found. Install OpenClaw first or set OPENCLAW_BIN.");

  upsertEnv(REFLEXIO_ENV, {
    OPENCLAW_BIN: cli,
    OPENCLAW_SMART_USE_LOCAL_CLI: "1",
  });

  runOpenClaw(cli, ["plugins", "uninstall", "--force", PLUGIN_ID]);
  rmSync(STALE_EXTENSION_DIR, { recursive: true, force: true });

  const installResult = runOpenClaw(cli, ["plugins", "install", PLUGIN_ROOT]);
  if (installResult.status !== 0) {
    printCommandFailure("plugins install", installResult);
    process.exit(1);
  }

  const enableResult = runOpenClaw(cli, ["plugins", "enable", PLUGIN_ID]);
  if (enableResult.status !== 0) {
    printCommandFailure("plugins enable", enableResult);
    process.exit(1);
  }

  const accessResult = runOpenClaw(cli, [
    "config",
    "set",
    `plugins.entries.${PLUGIN_ID}.hooks.allowConversationAccess`,
    "true",
  ]);
  if (accessResult.status !== 0) {
    printCommandFailure("config set allowConversationAccess", accessResult);
    process.exit(1);
  }

  runSmartInstall();
  runOpenClaw(cli, ["gateway", "restart"]);

  if (!inspectLoaded(cli)) {
    fail(`plugin not loaded; check 'openclaw plugins inspect ${PLUGIN_ID}'`);
  }
  console.log("openclaw-smart installed and registered.");
}

function uninstall({ purge = false } = {}) {
  const cli = resolveOpenClawBin();
  if (cli) {
    runOpenClaw(cli, ["plugins", "disable", PLUGIN_ID]);
    runOpenClaw(cli, ["plugins", "uninstall", "--force", PLUGIN_ID]);
  } else {
    console.warn("openclaw-smart: openclaw CLI not found; skipping plugin removal");
  }
  removeEnvKeys(REFLEXIO_ENV, ["OPENCLAW_BIN", "OPENCLAW_SMART_USE_LOCAL_CLI"]);
  if (purge) rmSync(join(homedir(), ".openclaw-smart"), { recursive: true, force: true });
  console.log("openclaw-smart uninstalled.");
}

function repair() {
  runSmartInstall();
  console.log("openclaw-smart repair complete.");
}

const [command, ...args] = process.argv.slice(2);
if (!command || command === "-h" || command === "--help") {
  usage();
  process.exit(0);
}
if (command === "install") install();
else if (command === "uninstall") uninstall({ purge: args.includes("--purge") });
else if (command === "repair") repair();
else {
  usage();
  fail(`unknown command: ${command}`);
}
