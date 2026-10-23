// Exercise the exact dependency-free builder used by SDK installation, rather
// than relying on Vitest's TypeScript transformer to hide a broken artifact.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, it } from "vitest";

it("builds a standalone runtime using only the existing Node executable", () => {
  const root = mkdtempSync(join(tmpdir(), "reflexio-plugin-artifact-"));
  try {
    const builder = fileURLToPath(new URL("../build.mjs", import.meta.url));
    execFileSync(process.execPath, [builder, root]);
    expect(readdirSync(root).sort()).toEqual([
      "README.md", "dist", "openclaw.plugin.json", "package.json",
    ]);
    const metadata = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
    expect(metadata.scripts).toBeUndefined();
    expect(metadata.devDependencies).toBeUndefined();
    const probe = `
      import { resolveSettings } from './dist/config.js';
      import { ReflexioClient, ReflexioError } from './dist/client.js';
      import { ReflexioHooks } from './dist/handlers.js';
      const settings = resolveSettings({ search: { top_k: 7 } }, {});
      const client = new ReflexioClient(settings);
      const hooks = new ReflexioHooks(settings, client, {});
      console.log(JSON.stringify({ topK: settings.topK, url: client.baseUrl,
        rejected: new ReflexioError('rejected', 422).rejected, stats: hooks.stats }));
    `;
    const actual = JSON.parse(execFileSync(process.execPath, ["--input-type=module", "-e", probe], {
      cwd: root, encoding: "utf8",
    }));
    expect(actual).toEqual({
      topK: 7, url: "http://127.0.0.1:8061/", rejected: true,
      stats: { injections: 0, publishes: 0, failures: {} },
    });
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
