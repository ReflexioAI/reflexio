import { writePlaybookFile, deleteFile, validateSlug } from "./io.js";
import { preprocessQuery, judgeContradiction, extractId } from "./dedup.js";
import { rawSearch } from "./search.js";

export interface WritePlaybookConfig {
  shallow_threshold: number;
  top_k: number;
}

export interface WritePlaybookOpts {
  slug: string;
  body: string;
  workspace?: string;
  config: WritePlaybookConfig;
}

/**
 * Full playbook write orchestration:
 * validate → preprocess → search → judge → write → delete (if superseding)
 */
export function writePlaybook(opts: WritePlaybookOpts): string {
  validateSlug(opts.slug);

  const query = preprocessQuery(opts.body);
  const neighbors = rawSearch(query, opts.config.top_k, "playbook");
  const top = neighbors[0];
  let supersedes: string[] | undefined;
  let deleteTarget: string | undefined;

  if (top && top.score >= opts.config.shallow_threshold) {
    const bodyFromSnippet = top.snippet.split("---").slice(2).join("---").trim();
    const decision = judgeContradiction(opts.body, bodyFromSnippet);

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
    const ws = opts.workspace || process.cwd();
    const absDelete = deleteTarget.startsWith("/")
      ? deleteTarget
      : `${ws}/${deleteTarget}`;
    deleteFile(absDelete);
  }

  return newPath;
}
