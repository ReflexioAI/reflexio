import { execFileSync, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, statSync, writeFileSync, unlinkSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

import Database from "better-sqlite3";

// ---------------------------------------------------------------------------
// SQLite session store — persistent, crash-safe conversation buffer.
// DB lives in ~/.reflexio/sessions.db.
// ---------------------------------------------------------------------------

const DB_PATH = join(homedir(), ".reflexio", "sessions.db");
const MAX_CONTENT_LENGTH = 10_000;
const MAX_INTERACTIONS = 200;
const BATCH_SIZE = 10; // Publish every N complete exchanges mid-session
const MAX_RETRIES = 3; // Give up retrying after this many failures

let _db = null;

function getDb() {
	if (_db) return _db;
	mkdirSync(dirname(DB_PATH), { recursive: true, mode: 0o700 });
	_db = new Database(DB_PATH);
	_db.pragma("journal_mode = WAL");
	_db.exec(`
    CREATE TABLE IF NOT EXISTS turns (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      published INTEGER DEFAULT 0,
      retry_count INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_session_published
      ON turns(session_id, published);
  `);
	// Add retry_count column if missing (migration for existing DBs)
	try {
		_db.exec("ALTER TABLE turns ADD COLUMN retry_count INTEGER DEFAULT 0");
	} catch {
		// Column already exists -- ignore
	}
	// Clean up old published turns (keep 7 days)
	_db.exec(
		"DELETE FROM turns WHERE published = 1 AND timestamp < datetime('now', '-7 days')",
	);
	process.on("exit", () => {
		if (_db) _db.close();
	});
	return _db;
}

// ---------------------------------------------------------------------------
// Smart truncation — preserves head + tail with a marker in between
// ---------------------------------------------------------------------------

function smartTruncate(content, maxLength = MAX_CONTENT_LENGTH) {
	if (!content || content.length <= maxLength) return content || "";
	const headLen = Math.floor(maxLength * 0.8);
	const tailLen = Math.max(0, maxLength - headLen - 80);
	const truncated = content.length - headLen - tailLen;
	const marker = `\n\n[...truncated ${truncated} chars...]\n\n`;
	if (tailLen === 0) return content.slice(0, headLen) + marker;
	return content.slice(0, headLen) + marker + content.slice(-tailLen);
}

// ---------------------------------------------------------------------------
// Session ID resolution
// ---------------------------------------------------------------------------

let _fallbackSessionId = null;

function getSessionId(event) {
	const key = event.context?.sessionKey;
	if (key) return key;
	if (!_fallbackSessionId) {
		_fallbackSessionId = `anon-${randomUUID()}`;
		console.error(
			`[reflexio] No sessionKey; using fallback: ${_fallbackSessionId}`,
		);
	}
	return _fallbackSessionId;
}

// ---------------------------------------------------------------------------
// JSON5 comment stripping — respects quoted strings
// ---------------------------------------------------------------------------

function stripJsonComments(raw) {
	return raw.split("\n").map(line => {
		let inString = false;
		let escape = false;
		for (let i = 0; i < line.length; i++) {
			const ch = line[i];
			if (escape) { escape = false; continue; }
			if (ch === "\\") { escape = true; continue; }
			if (ch === '"') { inString = !inString; continue; }
			if (!inString && ch === "/" && line[i + 1] === "/") return line.slice(0, i);
			if (!inString && ch === "/" && line[i + 1] === "*") return line.slice(0, i);
		}
		return line;
	}).join("\n");
}

// ---------------------------------------------------------------------------
// User ID resolution — multi-agent instance support
// ---------------------------------------------------------------------------

let _openclawConfig = null; // cached after first read

function resolveUserId(event) {
	// 1. Explicit env override — highest priority
	if (process.env.REFLEXIO_USER_ID) return process.env.REFLEXIO_USER_ID;

	// 2. Extract agentId from sessionKey (format: agent:<agentId>:<key>)
	const sessionKey = event.context?.sessionKey ?? "";
	const sessionMatch = sessionKey.match(/^agent:([^:]+):/);
	if (sessionMatch) return sessionMatch[1];

	// 3. Read ~/.openclaw/openclaw.json (JSON5 — strip comments before parsing)
	if (_openclawConfig === null) {
		try {
			const configPath = join(homedir(), ".openclaw", "openclaw.json");
			const raw = readFileSync(configPath, "utf-8");
			// Strip comments while respecting quoted strings to avoid corrupting
			// string values that contain // (e.g. URLs like http://example.com).
			const stripped = stripJsonComments(raw);
			_openclawConfig = JSON.parse(stripped);
		} catch {
			_openclawConfig = {}; // cache failure so we don't retry every call
		}
	}
	const agents = _openclawConfig?.agents;
	if (agents) {
		// Try agents.defaults first, then first entry in agents.list[]
		if (agents.defaults && typeof agents.defaults === "string") {
			return agents.defaults;
		}
		if (Array.isArray(agents.list) && agents.list.length > 0) {
			const first = agents.list[0];
			if (first && typeof first === "object" && first.name) return first.name;
			if (typeof first === "string") return first;
		}
	}

	// 4. Backward-compatible fallback
	return "openclaw";
}

// ---------------------------------------------------------------------------
// Aggregation trigger — fire-and-forget after successful publish
// ---------------------------------------------------------------------------

function triggerAggregationIfNeeded(agentVersion) {
	const flagFile = join(homedir(), ".reflexio", "logs", ".aggregation-running");

	// Skip if aggregation is already running (flag < 5 min old)
	try {
		const stat = statSync(flagFile);
		if (Date.now() - stat.mtimeMs < 5 * 60 * 1000) {
			console.error("[reflexio] Aggregation already running, skipping");
			return;
		}
	} catch {
		// Flag file doesn't exist — proceed
	}

	// Create flag and logs dir
	const logsDir = join(homedir(), ".reflexio", "logs");
	mkdirSync(logsDir, { recursive: true, mode: 0o700 });
	writeFileSync(flagFile, String(Date.now()), { mode: 0o600 });

	const child = spawn(
		"reflexio",
		["agent-playbooks", "aggregate", "--agent-version", agentVersion],
		{ stdio: ["ignore", "ignore", "ignore"], detached: true },
	);
	child.on("close", () => {
		try {
			unlinkSync(flagFile);
		} catch {}
	});
	child.unref();

	console.error(
		`[reflexio] Triggered aggregation for agent-version=${agentVersion}`,
	);
}

// ---------------------------------------------------------------------------
// Shared publish logic — used by session end, incremental, and retry
// ---------------------------------------------------------------------------

function publishSession(db, sessionId, userId, agentVersion) {
	const turns = db
		.prepare(
			"SELECT id, role, content FROM turns WHERE session_id = ? AND published = 0 AND retry_count < ? ORDER BY id LIMIT ?",
		)
		.all(sessionId, MAX_RETRIES, MAX_INTERACTIONS);

	if (turns.length === 0) return;

	// Mark selected turns as in-flight (published = 2) synchronously to prevent
	// concurrent publishSession calls from picking up the same rows.
	const maxId = turns[turns.length - 1].id;
	db.prepare(
		"UPDATE turns SET published = 2 WHERE session_id = ? AND published = 0 AND retry_count < ? AND id <= ?",
	).run(sessionId, MAX_RETRIES, maxId);

	const payload = JSON.stringify({
		user_id: userId,
		source: "openclaw",
		agent_version: agentVersion,
		session_id: sessionId,
		interactions: turns.map((t) => ({ role: t.role, content: t.content })),
	});

	const payloadDir = join(homedir(), ".reflexio", "tmp");
	mkdirSync(payloadDir, { recursive: true, mode: 0o700 });
	const payloadFile = join(
		payloadDir,
		`publish-${sessionId.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 100)}.json`,
	);
	writeFileSync(payloadFile, payload, { mode: 0o600 });

	const child = spawn(
		"reflexio",
		["interactions", "publish", "--file", payloadFile],
		{ stdio: ["ignore", "ignore", "ignore"], detached: true },
	);
	child.on("close", (code) => {
		if (code === 0) {
			try {
				getDb()
					.prepare(
						"UPDATE turns SET published = 1 WHERE session_id = ? AND published = 2",
					)
					.run(sessionId);
			} catch (e) {
				console.error(
					`[reflexio] Failed to mark turns as published: ${e.message}`,
				);
			}
			triggerAggregationIfNeeded(agentVersion);
		} else {
			console.error(
				`[reflexio] Publish failed (exit ${code}), incrementing retry count`,
			);
			try {
				getDb()
					.prepare(
						"UPDATE turns SET published = 0, retry_count = retry_count + 1 WHERE session_id = ? AND published = 2",
					)
					.run(sessionId);
			} catch (e) {
				console.error(
					`[reflexio] Failed to update retry count: ${e.message}`,
				);
			}
		}
		try {
			unlinkSync(payloadFile);
		} catch {}
	});
	child.unref();

	console.error(
		`[reflexio] Queued ${turns.length} interactions for publish (session ${sessionId})`,
	);
}

/**
 * Main hook dispatcher for Reflexio-OpenClaw integration.
 *
 * Events handled:
 *   agent:bootstrap   - Inject user profile + retry unpublished sessions
 *   message:received  - Search Reflexio before agent responds
 *   message:sent      - Buffer turn to SQLite + incremental publish
 *   command:stop      - Flush remaining unpublished turns to Reflexio
 */
export default async function reflexioHook(event) {
	// Skip sub-agent sessions to avoid recursion (guards all event types)
	const sessionKey = event.context?.sessionKey ?? "";
	if (sessionKey.includes(":subagent:")) return;

	const eventKey = `${event.type}:${event.action}`;

	switch (eventKey) {
		case "agent:bootstrap":
			return handleBootstrap(event);
		case "message:received":
			return handleSearchBeforeResponse(event);
		case "message:sent":
			return handleMessageSent(event);
		case "command:stop":
			return handleSessionEnd(event);
	}
}

// ---------------------------------------------------------------------------
// Bootstrap: inject user profile + retry unpublished sessions from past
// ---------------------------------------------------------------------------

function handleBootstrap(event) {
	const workspaceDir = event.context?.workspaceDir;
	if (!workspaceDir) return;

	const userId = resolveUserId(event);
	const agentVersion = process.env.REFLEXIO_AGENT_VERSION || "openclaw-agent";
	const currentSessionId = getSessionId(event);

	console.error(`[reflexio] bootstrap hook fired, workspace=${workspaceDir}`);

	// --- Inject user profile ---
	try {
		const result = execFileSync(
			"reflexio",
			[
				"--json",
				"user-profiles",
				"search",
				"communication style, expertise, and preferences",
				"--user-id",
				userId,
				"--limit",
				"3",
			],
			{ timeout: 10_000, encoding: "utf-8" },
		);

		let profiles;
		try {
			const envelope = JSON.parse(result.trim());
			const data = envelope.data || envelope;
			profiles = data.user_profiles || data;
		} catch {
			console.error("[reflexio] Failed to parse profiles JSON");
			profiles = [];
		}

		if (Array.isArray(profiles) && profiles.length > 0) {
			const profileLines = profiles
				.map((p) => `- ${p.profile_content || ""}`.trim())
				.filter((line) => line.length > 2);

			if (profileLines.length > 0 && Array.isArray(event.context.bootstrapFiles)) {
				const bootstrapContent = [
					"## About This User (from Reflexio)",
					"",
					...profileLines,
					"",
					'Use `reflexio search "<your current task>"` before starting work to get task-specific behavioral corrections.',
				].join("\n");

				event.context.bootstrapFiles.push({
					name: "REFLEXIO_USER_PROFILE.md",
					path: "REFLEXIO_USER_PROFILE.md",
					content: bootstrapContent,
					source: "hook:reflexio-context",
				});
				console.error(
					`[reflexio] Injected user profile (${bootstrapContent.length} chars)`,
				);
			}
		}
	} catch (err) {
		console.error(`[reflexio] Bootstrap profile fetch failed: ${err.message}`);
	}

	// --- Retry unpublished turns from previous sessions ---
	try {
		const db = getDb();
		const oldSessions = db
			.prepare(
				"SELECT DISTINCT session_id FROM turns WHERE published = 0 AND retry_count < ? AND session_id != ? LIMIT 5",
			)
			.all(MAX_RETRIES, currentSessionId);

		if (oldSessions.length > 0) {
			console.error(
				`[reflexio] Retrying ${oldSessions.length} unpublished session(s)`,
			);
			for (const { session_id } of oldSessions) {
				publishSession(db, session_id, userId, agentVersion);
			}
		}
	} catch (err) {
		console.error(`[reflexio] Retry failed: ${err.message}`);
	}
}

// ---------------------------------------------------------------------------
// Message received: search Reflexio before the agent responds
// ---------------------------------------------------------------------------

const TRIVIAL_RESPONSE_RE = /^(yes|no|ok|sure|thanks|y|n)$/i;

function handleSearchBeforeResponse(event) {
	let prompt = event.context?.userMessage;
	if (!prompt || prompt.length < 5) return;
	if (TRIVIAL_RESPONSE_RE.test(prompt.trim())) return;
	prompt = prompt.slice(0, 4096);

	try {
		const userId = resolveUserId(event);
		const result = execFileSync(
			"reflexio",
			["search", prompt, "--user-id", userId, "--top-k", "5"],
			{ timeout: 5_000, encoding: "utf-8" },
		);

		if (result && result.trim() && Array.isArray(event.context?.bootstrapFiles)) {
			event.context.bootstrapFiles.push({
				name: "REFLEXIO_CONTEXT.md",
				path: "REFLEXIO_CONTEXT.md",
				content: result.trim(),
				source: "hook:reflexio-context",
			});
			console.error(
				`[reflexio] Injected search context for message (${result.trim().length} chars)`,
			);
		}
	} catch (err) {
		console.error(`[reflexio] Per-message search failed: ${err.message}`);
	}
}

// ---------------------------------------------------------------------------
// Message sent: buffer turn + incremental publish every BATCH_SIZE exchanges
// ---------------------------------------------------------------------------

function handleMessageSent(event) {
	const userMessage = event.context?.userMessage;
	const agentResponse = event.context?.agentResponse;
	const sessionId = getSessionId(event);

	if (!userMessage && !agentResponse) return;

	try {
		const db = getDb();
		const now = new Date().toISOString();

		const insertTurn = db.transaction((sid, user, agent, ts) => {
			const stmt = db.prepare(
				"INSERT INTO turns (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
			);
			if (user) stmt.run(sid, "user", smartTruncate(user), ts);
			if (agent) stmt.run(sid, "assistant", smartTruncate(agent), ts);
		});
		insertTurn(sessionId, userMessage, agentResponse, now);

		// Incremental publish: every BATCH_SIZE complete exchanges
		const { count } = db
			.prepare(
				"SELECT COUNT(*) as count FROM turns WHERE session_id = ? AND published = 0",
			)
			.get(sessionId);

		if (count >= BATCH_SIZE * 2) {
			const userId = resolveUserId(event);
			const agentVersion =
				process.env.REFLEXIO_AGENT_VERSION || "openclaw-agent";
			publishSession(db, sessionId, userId, agentVersion);
		}
	} catch (err) {
		console.error(`[reflexio] Failed to buffer turn: ${err.message}`);
	}
}

// ---------------------------------------------------------------------------
// Session end: flush remaining unpublished turns
// ---------------------------------------------------------------------------

function handleSessionEnd(event) {
	const sessionId = getSessionId(event);
	const userId = resolveUserId(event);
	const agentVersion = process.env.REFLEXIO_AGENT_VERSION || "openclaw-agent";

	try {
		const db = getDb();
		publishSession(db, sessionId, userId, agentVersion);
	} catch (err) {
		console.error(`[reflexio] Session flush failed: ${err.message}`);
	}
}
