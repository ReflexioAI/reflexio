import { writeProfileFile, deleteFile, validateSlug, validateTtl, type Ttl } from "./io.js";
import { preprocessQuery, judgeContradiction, extractId } from "./dedup.js";
import { rawSearch } from "./search.js";

export interface WriteProfileConfig {
  shallow_threshold: number;
  top_k: number;
}

export interface WriteProfileOpts {
  slug: string;
  ttl: Ttl | string;
  body: string;
  workspace?: string;
  config: WriteProfileConfig;
}

/**
 * Full profile write orchestration:
 * validate → preprocess → search → judge → write → delete (if superseding)
 */
export function writeProfile(opts: WriteProfileOpts): string {
  // Step 1-2: Validate inputs (throws on failure — caller catches)
  validateSlug(opts.slug);
  validateTtl(opts.ttl);

  // Step 3: Preprocess query for search
  const query = preprocessQuery(opts.body);

  // Step 4: Search for neighbors
  const neighbors = rawSearch(query, opts.config.top_k, "profile");

  // Step 5: Check threshold
  const top = neighbors[0];
  let supersedes: string[] | undefined;
  let deleteTarget: string | undefined;

  if (top && top.score >= opts.config.shallow_threshold) {
    // Step 6: Judge contradiction
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

  // Step 7: Write first, delete second
  const newPath = writeProfileFile({
    slug: opts.slug,
    ttl: opts.ttl as Ttl,
    body: opts.body,
    supersedes,
    workspace: opts.workspace,
  });

  if (deleteTarget) {
    // Resolve path relative to workspace
    const ws = opts.workspace || process.cwd();
    const absDelete = deleteTarget.startsWith("/")
      ? deleteTarget
      : `${ws}/${deleteTarget}`;
    deleteFile(absDelete);
  }

  return newPath;
}
