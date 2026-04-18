// Workspace auto-setup.
//
// On first load after install, appends the heartbeat consolidation check
// to the workspace HEARTBEAT.md. Skills are served from the extension dir
// via the manifest's "skills" field. Agents are injected via extraSystemPrompt.
// No workspace file copying needed — everything lives in the extension dir.
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

function resolveOpenclawHome(): string {
  return process.env.OPENCLAW_HOME || path.join(os.homedir(), ".openclaw");
}

/**
 * Append HEARTBEAT.md content to the workspace. Idempotent — checks for
 * the marker heading before appending.
 *
 * @param pluginDir - The plugin's install directory (import.meta.dirname)
 */
export function setupWorkspaceResources(pluginDir: string): void {
  const workspace = path.join(resolveOpenclawHome(), "workspace");

  const heartbeatSrc = path.join(pluginDir, "HEARTBEAT.md");
  const heartbeatDest = path.join(workspace, "HEARTBEAT.md");
  if (!fs.existsSync(heartbeatSrc)) return;

  const heartbeatContent = fs.readFileSync(heartbeatSrc, "utf8");
  const marker = "## Reflexio Consolidation Check";

  let existing = "";
  try {
    existing = fs.readFileSync(heartbeatDest, "utf8");
  } catch {
    // file doesn't exist yet
  }

  if (!existing.includes(marker)) {
    const separator = existing.length > 0 ? "\n\n" : "";
    fs.mkdirSync(workspace, { recursive: true });
    fs.writeFileSync(heartbeatDest, existing + separator + heartbeatContent, "utf8");
  }
}
