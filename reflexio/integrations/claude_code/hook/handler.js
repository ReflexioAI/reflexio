import { spawn } from "node:child_process";
import { existsSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/**
 * Claude Code Stop hook for Reflexio.
 *
 * Reads the full session transcript from the JSONL file provided in the
 * Stop event payload, extracts user queries and assistant text responses,
 * and publishes them to Reflexio via the CLI (fire-and-forget).
 *
 * Usage in settings.json:
 *   {
 *     "hooks": {
 *       "Stop": [{ "type": "command", "command": "node /path/to/handler.js" }]
 *     }
 *   }
 *
 * The hook reads event JSON from stdin with these fields:
 *   - session_id: string
 *   - transcript_path: string (path to .jsonl transcript file)
 *   - stop_hook_active: boolean (true if this is a hook-triggered stop)
 */

const MAX_INTERACTIONS = 200;
const MAX_CONTENT_LENGTH = 10_000;

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

	// Guard: prevent infinite loops when stop_hook_active is set
	if (event.stop_hook_active) {
		output({});
		return;
	}

	const sessionId = event.session_id;
	const transcriptPath = event.transcript_path;

	if (!transcriptPath || !existsSync(transcriptPath)) {
		console.error(
			`[reflexio] No transcript file found at: ${transcriptPath}`,
		);
		output({});
		return;
	}

	// Parse transcript JSONL
	const interactions = parseTranscript(transcriptPath);

	if (interactions.length === 0) {
		console.error(
			"[reflexio] No user/assistant interactions found in transcript",
		);
		output({});
		return;
	}

	// Build payload
	const userId = process.env.REFLEXIO_USER_ID || "claude-code";
	const agentVersion = process.env.REFLEXIO_AGENT_VERSION || "claude-code";

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

	// Fire-and-forget: spawn a shell that publishes then cleans up the temp file.
	// Cleanup is handled by the shell command itself (rm -f after publish),
	// not by Node.js event handlers, since child.unref() means the parent
	// exits before the child finishes.
	const child = spawn(
		"sh",
		[
			"-c",
			`reflexio interactions publish --user-id "${userId}" --file "${payloadFile}" --source "claude-code" --agent-version "${agentVersion}" --session-id "${sessionId || "unknown"}" ; rm -f "${payloadFile}"`,
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
 */
function parseTranscript(transcriptPath) {
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

	// Pair up into interactions: each interaction = one user + one assistant
	const interactions = [];
	let i = 0;
	while (i < messages.length && interactions.length < MAX_INTERACTIONS) {
		if (messages[i].role === "user") {
			interactions.push({
				role: "user",
				content: messages[i].content,
			});
			i++;
			// Attach the next assistant response, if any
			if (i < messages.length && messages[i].role === "assistant") {
				interactions.push({
					role: "assistant",
					content: messages[i].content,
					tools_used: messages[i].tools_used || [],
				});
				i++;
			}
		} else {
			// Orphaned assistant message (no preceding user) — still include it
			interactions.push({
				role: "assistant",
				content: messages[i].content,
				tools_used: messages[i].tools_used || [],
			});
			i++;
		}
	}

	return interactions;
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
	process.stdout.write(JSON.stringify(data) + "\n");
}

function sanitizeFilename(name) {
	const sanitized = name
		.replace(/[^a-zA-Z0-9_-]/g, "_")
		.replace(/^-+/, "")
		.slice(0, 200);
	return sanitized || "unnamed";
}

main().catch((err) => {
	console.error(`[reflexio] Hook failed: ${err.message}`);
	output({});
});
