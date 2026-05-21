import { describe, expect, it } from "vitest";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const CLI = resolve("scripts/npm-cli.js");

function runCli(args: string[], env: Record<string, string>) {
  return spawnSync(process.execPath, [CLI, ...args], {
    cwd: resolve("."),
    env: { ...process.env, ...env },
    encoding: "utf8",
  });
}

describe("openclaw-smart npm CLI", () => {
  it("prints help", () => {
    const result = runCli(["--help"], {});
    expect(result.status).toBe(0);
    expect(result.stdout).toContain("openclaw-smart install");
  });

  it("installs through openclaw and persists OPENCLAW_BIN", () => {
    const home = mkdtempSync(join(tmpdir(), "openclaw-smart-home-"));
    const bin = mkdtempSync(join(tmpdir(), "openclaw-smart-bin-"));
    const log = join(home, "openclaw.log");
    const fakeOpenClaw = join(bin, "openclaw");
    const fakeBash = join(bin, "bash");

    writeFileSync(
      fakeOpenClaw,
      `#!/usr/bin/env sh\nprintf '%s\\n' "$*" >> "${log}"\nif [ "$1 $2" = "plugins inspect" ]; then printf 'Status: loaded\\n'; fi\nexit 0\n`,
    );
    writeFileSync(fakeBash, "#!/usr/bin/env sh\nexit 0\n");
    chmodSync(fakeOpenClaw, 0o755);
    chmodSync(fakeBash, 0o755);

    const result = runCli(["install"], {
      HOME: home,
      PATH: `${bin}:${process.env.PATH ?? ""}`,
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("openclaw-smart installed and registered");
    expect(readFileSync(join(home, ".reflexio", ".env"), "utf8")).toContain(
      `OPENCLAW_BIN="${fakeOpenClaw}"`,
    );
    const calls = readFileSync(log, "utf8");
    expect(calls).toContain("plugins install");
    expect(calls).toContain(
      "config set plugins.entries.reflexio-openclaw-smart.hooks.allowConversationAccess true",
    );
  });
});
