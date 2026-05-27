import { execFileSync, spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Claude Code UserPromptSubmit hook for Reflexio.
 *
 * Runs `reflexio search` with the user's prompt and outputs results to stdout.
 * Claude Code injects stdout content as context Claude sees before responding.
 *
 * In addition to printing the rendered context block, this hook also records
 * the `(kind, real_id)` of every profile and user_playbook returned by a
 * parallel `reflexio --json search` call to a session-scoped JSONL state file
 * at `~/.reflexio/claude-code-sessions/<session_id>.jsonl`. The SessionEnd
 * hook reads this state file and attaches the recorded citations to the
 * assistant interaction that followed each user prompt — that is what makes
 * the /evaluations "Rules that moved the needle" panel populate with data.
 *
 * This is intentionally synchronous — results must be available before Claude
 * responds. Timeout is 5 seconds to avoid blocking the UI too long.
 *
 * If the Reflexio server is not running (connection refused), starts it in the
 * background and exits silently (next message will find the server ready).
 */

const SEARCH_TIMEOUT_MS = 5_000;
const MIN_PROMPT_LENGTH = 5;
const LOG_DIR = join(homedir(), ".reflexio", "logs");
const STARTING_FLAG = join(LOG_DIR, ".server-starting");
const SESSIONS_DIR = join(homedir(), ".reflexio", "claude-code-sessions");

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

// Common non-task messages that should skip search
const SKIP_PATTERNS =
	/^(yes|no|ok|okay|sure|thanks|thank you|yep|nope|right|correct|got it|done|good|great|fine|lgtm|y|n|k|ty|thx|ack|np)$/i;

function isInternalInvocation() {
	if (process.env.CLAUDE_SMART_INTERNAL === "1" || process.env.REFLEXIO_INTERNAL === "1") {
		return true;
	}
	const entrypoint = process.env.CLAUDE_CODE_ENTRYPOINT;
	return Boolean(entrypoint && entrypoint !== "cli");
}

/**
 * Sanitize a session_id so it can be used as a filesystem basename. Matches
 * the same policy used in handler.js for the temp payload file.
 */
function sanitizeFilename(name) {
	const sanitized = (name || "")
		.replace(/[^a-zA-Z0-9_-]/g, "_")
		.replace(/^-+/, "")
		.slice(0, 200);
	return sanitized || "unnamed";
}

/**
 * Extract citations from the `--json search` envelope. Returns an array of
 * `{kind, real_id, tag, title}` objects shaped to match the server-side
 * `Citation` Pydantic model.
 *
 * Only profiles and user_playbooks are emitted — `agent_playbooks` have a
 * different ID space (`agent_playbook_id`) that cannot be resolved by the
 * server-side citation reconciler, which keys off `(kind="playbook",
 * real_id=user_playbook_id)` and `(kind="profile", real_id=profile_id)`.
 */
function extractCitations(jsonStdout) {
	// The CLI's --json mode emits a single JSON envelope to stdout, but a
	// handful of subsystems (e.g. the local embedding provider) emit ANSI-
	// prefixed log lines to stdout on import. Find the first `{` and parse
	// from there; everything before is best-effort discarded log noise.
	if (!jsonStdout) return [];
	const start = jsonStdout.indexOf("{");
	if (start < 0) return [];
	let envelope;
	try {
		envelope = JSON.parse(jsonStdout.slice(start));
	} catch {
		return [];
	}
	const data = envelope?.data ?? envelope;
	if (!data || typeof data !== "object") return [];

	const citations = [];

	const profiles = Array.isArray(data.profiles) ? data.profiles : [];
	for (const p of profiles) {
		if (!p || !p.profile_id) continue;
		citations.push({
			kind: "profile",
			real_id: String(p.profile_id),
			tag: "",
			title: typeof p.content === "string" ? p.content.slice(0, 80) : "",
		});
	}

	const userPlaybooks = Array.isArray(data.user_playbooks) ? data.user_playbooks : [];
	for (const pb of userPlaybooks) {
		if (!pb || pb.user_playbook_id == null) continue;
		citations.push({
			kind: "playbook",
			real_id: String(pb.user_playbook_id),
			tag: "",
			title:
				(typeof pb.playbook_name === "string" && pb.playbook_name) ||
				(typeof pb.content === "string" ? pb.content.slice(0, 80) : ""),
		});
	}

	return citations;
}

/**
 * Append one JSONL record of {prompt, timestamp, citations} to the session's
 * state file. Created lazily on first call. Best-effort — any I/O failure is
 * swallowed so context injection (the user-visible side effect) is unaffected.
 */
function recordCitations(sessionId, prompt, citations) {
	if (!sessionId || !citations || citations.length === 0) return;
	try {
		mkdirSync(SESSIONS_DIR, { recursive: true, mode: 0o700 });
		const filePath = join(SESSIONS_DIR, `${sanitizeFilename(sessionId)}.jsonl`);
		const record = {
			prompt,
			timestamp: Math.floor(Date.now() / 1000),
			citations,
		};
		appendFileSync(filePath, `${JSON.stringify(record)}\n`, { mode: 0o600 });
	} catch {
		// best-effort; do not let state-file errors break the hook
	}
}

async function main() {
	const input = readFileSync("/dev/stdin", "utf-8").trim();
	if (!input) {
		process.exit(0);
	}

	let event;
	try {
		event = JSON.parse(input);
	} catch {
		process.exit(0);
	}

	const prompt = event.prompt || "";
	const sessionId = event.session_id || "";
	if (isInternalInvocation()) {
		process.exit(0);
	}

	// Skip short messages and common non-task responses
	if (prompt.length < MIN_PROMPT_LENGTH || SKIP_PATTERNS.test(prompt.trim())) {
		process.exit(0);
	}

	const userId = process.env.REFLEXIO_USER_ID || readEnvVar("REFLEXIO_USER_ID") || "claude-code";

	try {
		const result = execFileSync("reflexio", ["search", prompt, "--user-id", userId], {
			timeout: SEARCH_TIMEOUT_MS,
			encoding: "utf-8",
		});

		const trimmed = result.trim();
		if (trimmed && !trimmed.includes("Found 0 profiles, 0 playbooks")) {
			// Output to stdout — Claude sees this as injected context
			process.stdout.write(`${trimmed}\n`);
		}

		// Second pass: collect structured citations for SessionEnd attribution.
		// This is best-effort — failures must not prevent context injection.
		// `REFLEXIO_HOOK_TEST_MODE=1` short-circuits the spawn for unit tests.
		if (process.env.REFLEXIO_HOOK_TEST_MODE !== "1") {
			try {
				const jsonStdout = execFileSync(
					"reflexio",
					["--json", "search", prompt, "--user-id", userId],
					{
						timeout: SEARCH_TIMEOUT_MS,
						encoding: "utf-8",
					},
				);
				const citations = extractCitations(jsonStdout);
				recordCitations(sessionId, prompt, citations);
			} catch {
				// json search failed (timeout, server hiccup) — skip recording
				// silently; the human-readable search already succeeded so the
				// user still gets context injection.
			}
		}
	} catch (err) {
		// Only start server if the error looks like a connection failure
		const stderr = err.stderr || "";
		const message = err.message || "";
		const isConnectionError =
			stderr.includes("Cannot reach server") ||
			stderr.includes("Connection refused") ||
			stderr.includes("ECONNREFUSED") ||
			message.includes("ECONNREFUSED") ||
			message.includes("ENOENT"); // reflexio binary not found (unlikely but safe)

		if (!isConnectionError) {
			// Server is running but search failed for another reason — don't start server
			return;
		}

		// Remote server — can't start it locally, just exit
		const serverUrl = process.env.REFLEXIO_URL || readEnvVar("REFLEXIO_URL");
		const isLocal =
			!serverUrl || serverUrl.includes("127.0.0.1") || serverUrl.includes("localhost");
		if (!isLocal) {
			return;
		}

		// Guard: don't spawn multiple server starts within a session
		if (existsSync(STARTING_FLAG)) {
			return;
		}

		try {
			mkdirSync(LOG_DIR, { recursive: true, mode: 0o700 });
			// Create flag file to prevent repeated starts
			writeFileSync(STARTING_FLAG, String(Date.now()));

			const child = spawn(
				"sh",
				[
					"-c",
					`reflexio services start --only backend > "${join(LOG_DIR, "server.log")}" 2>&1 & sleep 5 && rm -f "${STARTING_FLAG}"`,
				],
				{
					detached: true,
					stdio: ["ignore", "ignore", "ignore"],
				},
			);
			child.unref();
		} catch {
			// ignore — reflexio may not be installed
		}
	}
}

// Exports for unit tests — Node's ESM allows named exports alongside main().
export { extractCitations, recordCitations, SESSIONS_DIR, sanitizeFilename };

// Only run main() when invoked directly as the entry script. Importing this
// module from a test file (which exercises the helpers) must not fire the
// hook side effects.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
	main().catch(() => {
		process.exit(0);
	});
}
