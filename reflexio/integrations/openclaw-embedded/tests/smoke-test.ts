// Standalone smoke test for the Reflexio Embedded hook handler.
//
// Run (requires tsx or ts-node in PATH):
//   npx tsx tests/smoke-test.ts
//
// The plugin itself does NOT depend on tsx at runtime — Openclaw loads the
// .ts files via its own bundled jiti runtime. This smoke test only needs tsx
// because it runs outside Openclaw.
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import {
  injectBootstrapReminder,
  spawnExtractor,
  ttlSweepProfiles,
} from "../plugin/hook/handler.ts";

type FakeRunCall = {
  sessionKey: string;
  message: string;
};

async function main(): Promise<void> {
  const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "rfx-test-"));
  fs.mkdirSync(path.join(workspace, ".reflexio", "profiles"), {
    recursive: true,
  });

  // Create an expired profile
  fs.writeFileSync(
    path.join(workspace, ".reflexio", "profiles", "old-xxxx.md"),
    `---
type: profile
id: prof_xxxx
created: 2020-01-01T00:00:00Z
ttl: one_day
expires: 2020-01-02
---

Old expired fact.
`,
  );

  // Create a fresh profile
  fs.writeFileSync(
    path.join(workspace, ".reflexio", "profiles", "fresh-yyyy.md"),
    `---
type: profile
id: prof_yyyy
created: 2026-04-16T00:00:00Z
ttl: infinity
expires: never
---

Fresh fact.
`,
  );

  // 1. TTL sweep deletes the expired file, preserves the fresh one
  await ttlSweepProfiles(workspace);
  const oldExists = fs.existsSync(
    path.join(workspace, ".reflexio", "profiles", "old-xxxx.md"),
  );
  const freshExists = fs.existsSync(
    path.join(workspace, ".reflexio", "profiles", "fresh-yyyy.md"),
  );
  console.log(`Old file deleted: ${!oldExists ? "PASS" : "FAIL"}`);
  console.log(`Fresh file preserved: ${freshExists ? "PASS" : "FAIL"}`);

  // 2. Bootstrap reminder is non-empty and mentions SKILL.md
  const reminder = injectBootstrapReminder();
  console.log(
    `Bootstrap reminder mentions SKILL.md: ${reminder.includes("SKILL.md") ? "PASS" : "FAIL"}`,
  );

  // 3. spawnExtractor forwards task + sessionKey to the runtime
  const calls: FakeRunCall[] = [];
  const runtime = {
    subagent: {
      run: async (params: FakeRunCall): Promise<{ runId: string }> => {
        calls.push({
          sessionKey: params.sessionKey,
          message: params.message,
        });
        return { runId: "test-run" };
      },
    },
  };
  const runId = await spawnExtractor({
    runtime,
    workspaceDir: workspace,
    sessionKey: "test-session",
    messages: [
      { role: "user", content: "I'm vegetarian" },
      { role: "assistant", content: "Got it." },
    ],
    reason: "smoke-test",
  });
  console.log(
    `Extractor spawned with runId: ${runId === "test-run" ? "PASS" : "FAIL"}`,
  );
  console.log(
    `Extractor message contains transcript: ${
      calls.length === 1 && calls[0].message.includes("vegetarian")
        ? "PASS"
        : "FAIL"
    }`,
  );

  // 4. spawnExtractor skips when transcript is too short AND no sessionFile
  const skipCalls: FakeRunCall[] = [];
  const skipRuntime = {
    subagent: {
      run: async (params: FakeRunCall): Promise<{ runId: string }> => {
        skipCalls.push(params);
        return { runId: "unexpected" };
      },
    },
  };
  const skipRunId = await spawnExtractor({
    runtime: skipRuntime,
    workspaceDir: workspace,
    sessionKey: "test-session-2",
    messages: [],
    reason: "smoke-test-skip",
  });
  console.log(
    `Extractor skipped on empty transcript: ${
      skipRunId === undefined && skipCalls.length === 0 ? "PASS" : "FAIL"
    }`,
  );

  fs.rmSync(workspace, { recursive: true, force: true });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
