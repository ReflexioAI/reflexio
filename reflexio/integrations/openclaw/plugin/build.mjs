// Build with the Node runtime already required by OpenClaw. No npm install or
// registry release is needed to install the plugin shipped in the Python SDK.
import { copyFileSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const target = process.argv[2] ? resolve(process.argv[2]) : root;
rmSync(join(target, "dist"), { recursive: true, force: true });
mkdirSync(join(target, "dist"), { recursive: true });
for (const name of readdirSync(join(root, "src"))) {
  if (!name.endsWith(".ts") || name.endsWith(".d.ts")) continue;
  const source = readFileSync(join(root, "src", name), "utf8");
  writeFileSync(join(target, "dist", name.replace(/\.ts$/, ".js")), stripTypeScriptTypes(source));
}
if (target !== root) {
  for (const name of ["openclaw.plugin.json", "README.md"]) {
    copyFileSync(join(root, name), join(target, name));
  }
  const metadata = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
  // The install artifact contains runtime files only; its build inputs stay in
  // the SDK, and the gateway must not install development dependencies/hooks.
  delete metadata.scripts;
  delete metadata.devDependencies;
  writeFileSync(join(target, "package.json"), `${JSON.stringify(metadata, null, 2)}\n`);
}
