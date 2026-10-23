// Stable identifiers sent to Reflexio.
//
// user_id scopes learnings. The previous plugin used the workspace basename,
// and every default OpenClaw agent workspace is named "workspace", so all of a
// machine's agents (and every user of a shared backend) collided into one
// Reflexio user. The id now carries the agent and a digest of the full path.
//
// request_id ties the search exposure recorded at before_prompt_build to the
// interaction published at agent_end. OpenClaw hands both hooks the same
// runId, so deriving it from (sessionKey, runId) needs no stored reservation,
// and re-publishing the same run reuses the same id.

import { createHash } from "node:crypto";
import * as path from "node:path";

function digest(value: string, length: number): string {
  return createHash("sha256").update(value).digest("hex").slice(0, length);
}

function slug(value: string): string {
  return (
    value
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 48) || "workspace"
  );
}

export function resolveUserId(args: {
  agentId?: string;
  workspaceDir?: string;
}): string | undefined {
  if (!args.workspaceDir) return undefined;
  const dir = path.resolve(args.workspaceDir);
  const agent = slug(args.agentId || "main");
  return `openclaw:${agent}:${slug(path.basename(dir))}-${digest(dir, 8)}`;
}

export function requestIdFor(sessionKey: string, runId: string): string {
  return `openclaw-${digest(`${sessionKey}\u0000${runId}`, 32)}`;
}
