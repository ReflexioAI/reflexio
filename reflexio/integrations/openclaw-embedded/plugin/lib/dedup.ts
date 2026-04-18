import { infer, type CommandRunner } from "./openclaw-cli.ts";

const PREPROCESS_PROMPT = `Rewrite the following text into a single descriptive sentence that captures the core fact or topic. Expand with 2-3 important synonyms or related terms to improve search matching. Remove conversational filler. Return ONLY the rewritten text.

Text: "{rawText}"`;

const CONTRADICTION_PROMPT = `EXISTING fact: "{existingContent}"
NEW fact: "{newContent}"

Does the NEW fact replace or contradict the EXISTING fact (same topic, updated information)?
Answer with ONLY a JSON object: {"decision": "supersede"} or {"decision": "keep_both"}`;

/**
 * Rewrite raw text into a clean search query optimized for vector + FTS search.
 * Falls back to raw text if openclaw infer is unavailable.
 */
export async function preprocessQuery(rawText: string, runner: CommandRunner): Promise<string> {
  const prompt = PREPROCESS_PROMPT.replace("{rawText}", rawText);
  const result = await infer(prompt, runner);
  if (!result || result.trim().length === 0) {
    return rawText;
  }
  return result.trim();
}

/**
 * Ask LLM whether newContent contradicts/replaces existingContent.
 * Returns "supersede" or "keep_both". Defaults to "keep_both" on any failure.
 */
export async function judgeContradiction(
  newContent: string,
  existingContent: string,
  runner: CommandRunner
): Promise<"supersede" | "keep_both"> {
  const prompt = CONTRADICTION_PROMPT
    .replace("{existingContent}", existingContent)
    .replace("{newContent}", newContent);

  const result = await infer(prompt, runner);
  if (!result) return "keep_both";

  try {
    const parsed = JSON.parse(result);
    if (parsed.decision === "supersede") return "supersede";
    return "keep_both";
  } catch {
    return "keep_both";
  }
}

/**
 * Extract the `id:` value from a memory search snippet containing YAML frontmatter.
 */
export function extractId(snippet: string): string | null {
  const match = /^id:\s*(\S+)/m.exec(snippet);
  return match ? match[1] : null;
}
