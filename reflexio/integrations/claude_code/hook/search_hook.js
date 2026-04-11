import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/**
 * Claude Code UserPromptSubmit hook for Reflexio.
 *
 * Runs `reflexio search` with the user's prompt and outputs results to stdout.
 * Claude Code injects stdout content as context Claude sees before responding.
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

/**
 * Read REFLEXIO_URL from ~/.reflexio/.env when the env var is not set.
 * Returns the URL string, or empty string if not found.
 */
function readEnvUrl() {
	const envPath = join(homedir(), ".reflexio", ".env");
	try {
		const content = readFileSync(envPath, "utf-8");
		const match = content.match(/^REFLEXIO_URL="?([^"\n]*)"?/m);
		return match ? match[1] : "";
	} catch {
		return "";
	}
}

// Common non-task messages that should skip search
const SKIP_PATTERNS =
	/^(yes|no|ok|okay|sure|thanks|thank you|yep|nope|right|correct|got it|done|good|great|fine|lgtm|y|n|k|ty|thx|ack|np)$/i;

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

	// Skip short messages and common non-task responses
	if (prompt.length < MIN_PROMPT_LENGTH || SKIP_PATTERNS.test(prompt.trim())) {
		process.exit(0);
	}

	const userId = process.env.REFLEXIO_USER_ID || "claude-code";

	try {
		const result = execFileSync(
			"reflexio",
			["search", prompt, "--user-id", userId],
			{
				timeout: SEARCH_TIMEOUT_MS,
				encoding: "utf-8",
			},
		);

		const trimmed = result.trim();
		if (trimmed && !trimmed.includes("Found 0 profiles, 0 playbooks")) {
			// Output to stdout — Claude sees this as injected context
			process.stdout.write(trimmed + "\n");
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
		const serverUrl = process.env.REFLEXIO_URL || readEnvUrl();
		const isLocal =
			!serverUrl ||
			serverUrl.includes("127.0.0.1") ||
			serverUrl.includes("localhost");
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

main().catch(() => {
	process.exit(0);
});
