import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Claude Code SessionEnd hook for Reflexio.
 *
 * Reads the full session transcript from the JSONL file provided in the
 * Stop event payload, extracts user queries and assistant text responses,
 * and publishes them to Reflexio via the CLI (fire-and-forget).
 *
 * If a session-scoped citation state file exists at
 * `~/.reflexio/claude-code-sessions/<session_id>.jsonl` (written by
 * `search_hook.js` on every UserPromptSubmit), this handler also attaches
 * the recorded citations to the assistant interaction that followed each
 * user prompt — the data backing the /evaluations "Rules that moved the
 * needle" panel.
 *
 * Usage in settings.json:
 *   {
 *     "hooks": {
 *       "SessionEnd": [{ "type": "command", "command": "node /path/to/handler.js" }]
 *     }
 *   }
 *
 * The hook reads event JSON from stdin with these fields:
 *   - session_id: string
 *   - transcript_path: string (path to .jsonl transcript file)
 */

const MAX_INTERACTIONS = 200;
const MAX_CONTENT_LENGTH = 10_000;
const PROMPT_MATCH_PREFIX = 200; // chars compared between recorded + transcript prompts
const LOG_DIR = join(homedir(), ".reflexio", "logs");
const SESSIONS_DIR = join(homedir(), ".reflexio", "claude-code-sessions");

function isInternalInvocation() {
	if (process.env.CLAUDE_SMART_INTERNAL === "1" || process.env.REFLEXIO_INTERNAL === "1") {
		return true;
	}
	const entrypoint = process.env.CLAUDE_CODE_ENTRYPOINT;
	return Boolean(entrypoint && entrypoint !== "cli");
}

/**
 * Read a variable from ~/.reflexio/.env when it is not set in process.env.
 * Returns the raw string value (with surrounding quotes stripped), or empty
 * string if the file is missing or the key is absent.
 */
function readEnvVar(key) {
	const envPath = join(homedir(), ".reflexio", ".env");
	try {
		const content = readFileSync(envPath, "utf-8");
		const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
		const match = content.match(new RegExp(`^${escaped}="?([^"\\n]*)"?`, "m"));
		return match ? match[1] : "";
	} catch {
		return "";
	}
}

async function main() {
	// Read event JSON from stdin
	const input = readFileSync("/dev/stdin", "utf-8").trim();
	if (!input) {
		output({});
		return;
	}

	let event;
	try {
		event = JSON.parse(input);
	} catch {
		console.error("[reflexio] Failed to parse event JSON from stdin");
		output({});
		return;
	}

	const sessionId = event.session_id;
	const transcriptPath = event.transcript_path;
	if (isInternalInvocation()) {
		output({});
		return;
	}

	if (!transcriptPath || !existsSync(transcriptPath)) {
		console.error(`[reflexio] No transcript file found at: ${transcriptPath}`);
		output({});
		return;
	}

	// Load citation records (if any) for this session
	const citationRecords = readCitationState(sessionId);

	// Parse transcript JSONL — attaches citations during pairing.
	const interactions = parseTranscript(transcriptPath, citationRecords);

	if (interactions.length === 0) {
		console.error("[reflexio] No user/assistant interactions found in transcript");
		output({});
		return;
	}

	// Build payload
	const userId = process.env.REFLEXIO_USER_ID || readEnvVar("REFLEXIO_USER_ID") || "claude-code";
	const agentVersion =
		process.env.REFLEXIO_AGENT_VERSION || readEnvVar("REFLEXIO_AGENT_VERSION") || "claude-code";

	const payload = JSON.stringify({
		user_id: userId,
		source: "claude-code",
		agent_version: agentVersion,
		session_id: sessionId || "unknown",
		interactions,
	});

	// Write payload to temp file
	const payloadFile = join(
		tmpdir(),
		`reflexio-cc-${sanitizeFilename(sessionId || "unknown")}-${Date.now()}.json`,
	);
	writeFileSync(payloadFile, payload, { mode: 0o600 });

	// Fire-and-forget: spawn a shell that publishes then cleans up the temp
	// file and the session-scoped citation state file. Cleanup is handled by
	// the shell command itself (rm -f after publish), not by Node.js event
	// handlers, since child.unref() means the parent exits before the child
	// finishes.
	mkdirSync(LOG_DIR, { recursive: true, mode: 0o700 });
	const logFile = join(LOG_DIR, "stop-hook.log");
	const stateFile = sessionStateFilePath(sessionId);
	const child = spawn(
		"sh",
		[
			"-c",
			'reflexio interactions publish --user-id "$1" --file "$2" --source "claude-code" --agent-version "$3" --session-id "$4" --force-extraction >> "$5" 2>&1; rm -f "$2" "$6"',
			"sh",
			userId,
			payloadFile,
			agentVersion,
			sessionId || "unknown",
			logFile,
			stateFile,
		],
		{
			detached: true,
			stdio: ["ignore", "ignore", "ignore"],
		},
	);
	child.unref();

	console.error(
		`[reflexio] Published ${interactions.length} interactions for session ${sessionId}`,
	);

	output({});
}

/**
 * Return the absolute path to the session-scoped citation JSONL state file.
 */
function sessionStateFilePath(sessionId) {
	return join(SESSIONS_DIR, `${sanitizeFilename(sessionId || "unknown")}.jsonl`);
}

/**
 * Read citation records written by search_hook.js for this session.
 *
 * Returns an ordered array of `{prompt, timestamp, citations}` records, one
 * per UserPromptSubmit fire that returned at least one citation. Order is
 * preserved (append-only file). Missing file → empty array.
 */
function readCitationState(sessionId) {
	if (!sessionId) return [];
	const path = sessionStateFilePath(sessionId);
	if (!existsSync(path)) return [];
	let raw;
	try {
		raw = readFileSync(path, "utf-8");
	} catch {
		return [];
	}
	const records = [];
	for (const line of raw.split("\n")) {
		if (!line.trim()) continue;
		try {
			const rec = JSON.parse(line);
			if (rec && typeof rec.prompt === "string" && Array.isArray(rec.citations)) {
				records.push(rec);
			}
		} catch {
			// skip corrupt lines
		}
	}
	return records;
}

/**
 * Parse Claude Code JSONL transcript into Reflexio interactions.
 *
 * Transcript format (one JSON object per line):
 *   { type: "user",      message: { role: "user",      content: "..." }, ... }
 *   { type: "assistant",  message: { role: "assistant", content: [...] }, ... }
 *
 * We extract user queries plus assistant text and tool_use blocks.
 * Assistant tool calls are surfaced as structured `tools_used` on the
 * assistant interaction — the Reflexio server renderer turns these into
 * `[used tool: name({json})]` markers that the playbook extractor's
 * tool-usage analysis path keys off. Thinking blocks, tool_result blocks,
 * system messages, and other entry types are skipped.
 *
 * Citation attribution policy (when `citationRecords` is non-empty): the
 * recorded records are an ordered queue indexed against the order in which
 * UserPromptSubmit fired. We walk the paired interactions in order; when a
 * user message's content prefix matches the head record's prompt prefix,
 * we pop that record and attach its citations to the NEXT assistant
 * interaction. Unmatched user prompts (e.g. short messages skipped by
 * the search hook) consume no record. If a single user prompt produced
 * multiple assistant turns (tool loops merged into one logical turn),
 * the citations attach to that single merged assistant interaction.
 */
function parseTranscript(transcriptPath, citationRecords = []) {
	const raw = readFileSync(transcriptPath, "utf-8");
	const lines = raw.split("\n");

	const messages = []; // { role, content, tools_used? }

	for (const line of lines) {
		if (!line.trim()) continue;

		let entry;
		try {
			entry = JSON.parse(line);
		} catch {
			continue; // skip corrupt lines
		}

		// Only process user and assistant entries (allowlist)
		if (entry.type !== "user" && entry.type !== "assistant") {
			continue;
		}

		if (entry.type === "user") {
			// Skip meta messages (slash commands, system injections)
			if (entry.isMeta) continue;

			const content = extractUserContent(entry.message);
			if (content) {
				messages.push({
					role: "user",
					content: content.slice(0, MAX_CONTENT_LENGTH),
				});
			}
		} else if (entry.type === "assistant") {
			const { text, toolsUsed } = extractAssistantBlocks(entry.message);
			if (text || toolsUsed.length > 0) {
				messages.push({
					role: "assistant",
					// Placeholder keeps the server renderer from dropping
					// turns that are pure tool_use with no accompanying text:
					// format_interactions_to_history_string only emits a line
					// when `content` is non-empty.
					content: (text || "(tool call)").slice(0, MAX_CONTENT_LENGTH),
					tools_used: toolsUsed,
				});
			}
		}
	}

	// Merge consecutive same-role entries into one logical turn. Claude Code
	// writes one JSONL entry per API response, so a single assistant turn can
	// span multiple entries (e.g. text, tool_use, then text again after a
	// tool_result on the user side). Without merging, only the first entry
	// gets paired and the rest either look like orphaned turns or get
	// dropped — truncating the assistant response in the UI.
	const merged = [];
	for (const m of messages) {
		const last = merged[merged.length - 1];
		if (last && last.role === m.role) {
			last.content = `${last.content}\n${m.content}`.slice(0, MAX_CONTENT_LENGTH);
			if (m.tools_used && m.tools_used.length > 0) {
				last.tools_used = [...(last.tools_used || []), ...m.tools_used];
			}
		} else {
			merged.push({ ...m, tools_used: m.tools_used ? [...m.tools_used] : [] });
		}
	}

	// Citation queue — first matching user-prompt pops the head, citations
	// attach to the very next assistant interaction.
	const queue = [...citationRecords];

	// Pair up into interactions: each interaction = one user + one assistant.
	// Skip user turns with no assistant response (e.g. interrupted turns,
	// meta slash commands) so we don't emit empty-assistant interactions.
	const interactions = [];
	let i = 0;
	while (i < merged.length && interactions.length < MAX_INTERACTIONS) {
		if (merged[i].role === "user") {
			if (i + 1 < merged.length && merged[i + 1].role === "assistant") {
				const userContent = merged[i].content;
				const citations = popMatchingCitations(queue, userContent);
				interactions.push({ role: "user", content: userContent });
				const assistant = {
					role: "assistant",
					content: merged[i + 1].content,
					tools_used: merged[i + 1].tools_used || [],
				};
				if (citations.length > 0) {
					assistant.citations = citations;
				}
				interactions.push(assistant);
				i += 2;
			} else {
				// Unpaired user turn — drop it rather than publish an
				// interaction with an empty assistant side.
				i++;
			}
		} else {
			// Orphaned assistant turn (no preceding user) — skip; without a
			// user query it has no value for playbook extraction.
			i++;
		}
	}

	return interactions;
}

/**
 * Pop the first queued citation record whose recorded prompt prefix matches
 * the given transcript user-content prefix. Returns the citation array, or
 * `[]` when no record matches.
 *
 * Matching is on the leading `PROMPT_MATCH_PREFIX` characters after
 * trimming whitespace. This tolerates trailing whitespace differences
 * between what the hook sees on UserPromptSubmit and what Claude Code
 * later writes to the transcript file, while still being a strong
 * positive signal that the two refer to the same prompt.
 */
function popMatchingCitations(queue, userContent) {
	if (queue.length === 0) return [];
	const target = (userContent || "").trim().slice(0, PROMPT_MATCH_PREFIX);
	if (!target) return [];

	for (let idx = 0; idx < queue.length; idx++) {
		const candidate = (queue[idx].prompt || "").trim().slice(0, PROMPT_MATCH_PREFIX);
		if (candidate === target) {
			const [popped] = queue.splice(idx, 1);
			return popped.citations || [];
		}
	}
	return [];
}

/**
 * Extract text content from a user message.
 * User message content can be a string or an array of content blocks.
 */
function extractUserContent(message) {
	if (!message) return null;
	const content = message.content;
	if (typeof content === "string") return content.trim() || null;
	if (Array.isArray(content)) {
		const textParts = content
			.filter((block) => block.type === "text")
			.map((block) => block.text)
			.join("\n");
		return textParts.trim() || null;
	}
	return null;
}

/**
 * Extract text and tool_use blocks from an assistant message.
 *
 * Assistant message content is an array of blocks:
 *   - text blocks become user-facing text
 *   - tool_use blocks become structured {tool_name, tool_data} entries
 *   - thinking and tool_result blocks are skipped (thinking is internal;
 *     tool_result lives on the next user-role turn)
 *
 * Returns { text: string, toolsUsed: [{tool_name, tool_data}] }.
 */
function extractAssistantBlocks(message) {
	if (!message || !message.content) return { text: "", toolsUsed: [] };
	const content = message.content;
	if (typeof content === "string") {
		return { text: content.trim(), toolsUsed: [] };
	}
	if (!Array.isArray(content)) return { text: "", toolsUsed: [] };

	const textParts = [];
	const toolsUsed = [];
	for (const block of content) {
		if (block.type === "text") {
			textParts.push(block.text);
		} else if (block.type === "tool_use") {
			toolsUsed.push({
				tool_name: block.name,
				tool_data: { input: block.input },
			});
		}
	}
	return { text: textParts.join("\n").trim(), toolsUsed };
}

/**
 * Write JSON response to stdout (required by Claude Code hook protocol).
 */
function output(data) {
	process.stdout.write(`${JSON.stringify(data)}\n`);
}

function sanitizeFilename(name) {
	const sanitized = name
		.replace(/[^a-zA-Z0-9_-]/g, "_")
		.replace(/^-+/, "")
		.slice(0, 200);
	return sanitized || "unnamed";
}

// Exports for unit tests
export {
	parseTranscript,
	popMatchingCitations,
	readCitationState,
	SESSIONS_DIR,
	sanitizeFilename,
	sessionStateFilePath,
};

// Only run main() when invoked directly as the entry script. Importing this
// module from a test file must not fire the hook side effects.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
	main().catch((err) => {
		console.error(`[reflexio] Hook failed: ${err.message}`);
		output({});
	});
}
