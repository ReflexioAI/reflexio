import { writePlaybookFile, deleteFile, validateSlug } from "./io.ts";
import { preprocessQuery, judgeContradiction, extractId } from "./dedup.ts";
import { rawSearch } from "./search.ts";
import type { CommandRunner } from "./openclaw-cli.ts";

export interface WritePlaybookConfig {
  shallow_threshold: number;
  top_k: number;
}

export interface WritePlaybookOpts {
  slug: string;
  body: string;
  workspace?: string;
  config: WritePlaybookConfig;
  runner: CommandRunner;
}

/**
 * Full playbook write orchestration:
 * validate → preprocess → search → judge → write → delete (if superseding)
 */
export async function writePlaybook(opts: WritePlaybookOpts): Promise<string> {
  validateSlug(opts.slug);

  const query = await preprocessQuery(opts.body, opts.runner);
  const neighbors = await rawSearch(query, opts.config.top_k, "playbook", opts.runner);
  const top = neighbors[0];
  let supersedes: string[] | undefined;
  let deleteTarget: string | undefined;

  if (top && top.score >= opts.config.shallow_threshold) {
    const bodyFromSnippet = top.snippet.split("---").slice(2).join("---").trim();
    const decision = await judgeContradiction(opts.body, bodyFromSnippet, opts.runner);

    if (decision === "supersede") {
      const oldId = extractId(top.snippet);
      if (oldId) {
        supersedes = [oldId];
        deleteTarget = top.path;
      }
    }
  }

  const newPath = writePlaybookFile({
    slug: opts.slug,
    body: opts.body,
    supersedes,
    workspace: opts.workspace,
  });

  if (deleteTarget) {
    const ws = opts.workspace || process.env.WORKSPACE || process.cwd();
    const absDelete = deleteTarget.startsWith("/")
      ? deleteTarget
      : `${ws}/${deleteTarget}`;
    deleteFile(absDelete);
  }

  return newPath;
}
