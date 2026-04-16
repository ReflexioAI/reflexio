// Standalone smoke test for the hook handler.
// Run: node hook/smoke-test.js

const { handler } = require("./handler.js");
const fs = require("fs");
const os = require("os");
const path = require("path");

async function main() {
  // Create a temp workspace
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "rfx-test-"));
  fs.mkdirSync(path.join(workspace, ".reflexio", "profiles"), { recursive: true });

  // Create an expired profile
  fs.writeFileSync(path.join(workspace, ".reflexio", "profiles", "old-xxxx.md"),
    `---
type: profile
id: prof_xxxx
created: 2020-01-01T00:00:00Z
ttl: one_day
expires: 2020-01-02
---

Old expired fact.
`);

  // Create a fresh profile
  fs.writeFileSync(path.join(workspace, ".reflexio", "profiles", "fresh-yyyy.md"),
    `---
type: profile
id: prof_yyyy
created: 2026-04-16T00:00:00Z
ttl: infinity
expires: never
---

Fresh fact.
`);

  process.env.WORKSPACE = workspace;

  // 1. Bootstrap — TTL sweep should delete old-xxxx.md
  const bootstrapEvent = {
    type: "agent",
    action: "bootstrap",
    context: { bootstrapFiles: [] },
  };
  const api = {
    runtime: {
      subagent: {
        run: async (args) => {
          console.log("[test] subagent.run called with agentId:", args.agentId);
          return { runId: "test-run" };
        },
      },
    },
  };

  await handler(bootstrapEvent, api);

  const oldExists = fs.existsSync(path.join(workspace, ".reflexio", "profiles", "old-xxxx.md"));
  const freshExists = fs.existsSync(path.join(workspace, ".reflexio", "profiles", "fresh-yyyy.md"));
  const reminderInjected = bootstrapEvent.context.bootstrapFiles.length === 1;

  console.log(`Old file deleted: ${!oldExists ? "PASS" : "FAIL"}`);
  console.log(`Fresh file preserved: ${freshExists ? "PASS" : "FAIL"}`);
  console.log(`Reminder injected: ${reminderInjected ? "PASS" : "FAIL"}`);

  // 2. compact:before — should spawn extractor
  const compactEvent = {
    type: "session",
    action: "compact:before",
    context: {
      messages: [
        { role: "user", content: "I'm vegetarian" },
        { role: "assistant", content: "Got it." },
      ],
    },
  };
  let spawned = false;
  const api2 = {
    runtime: {
      subagent: {
        run: async (args) => {
          spawned = true;
          console.log("[test] spawn task prompt length:", args.task.length);
          return { runId: "test-run-2" };
        },
      },
    },
  };
  await handler(compactEvent, api2);
  // Give the fire-and-forget a tick to resolve
  await new Promise((r) => setTimeout(r, 50));
  console.log(`Extractor spawned on compact:before: ${spawned ? "PASS" : "FAIL"}`);

  // Cleanup
  fs.rmSync(workspace, { recursive: true, force: true });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
