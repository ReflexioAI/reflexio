// ---------------------------------------------------------------------------
// Security contract — localhost only, HTTP only, search-only.
//
// This hook is a localhost-only, read-only integration. It makes HTTP calls
// to the Reflexio backend on the same machine via native fetch() and
// injects the results as bootstrap files for the agent to read. It never
// writes conversation data anywhere — no SQLite, no filesystem buffer, no
// subprocess, no environment-variable reads, no outbound hosts other than
// the hardcoded loopback URL below.
//
// Traffic: only HTTP requests to http://127.0.0.1:8081/api/* .
// No other hosts are contacted.
//
// Publishing (extracting playbooks from conversations and writing them
// back to Reflexio) is NOT performed by the hook. That responsibility
// belongs to the `/reflexio-extract` slash command, which runs in the
// agent's own context and performs extraction + CRUD through the
// `reflexio user-playbooks` CLI. The Reflexio server therefore needs no
// LLM provider API key for this integration.
// ---------------------------------------------------------------------------

// Hardcoded loopback destination — all traffic goes here, nowhere else.
const LOCAL_SERVER_URL = "http://127.0.0.1:8081";
// Hardcoded agent label; used as the agent_version filter for searches so
// results stay scoped to this integration build.
const AGENT_VERSION = "openclaw-agent";

// ---------------------------------------------------------------------------
// HTTP helper — single fetch() path for every API call
// ---------------------------------------------------------------------------

async function apiPost(path, body, timeoutMs = 10_000) {
	const ctrl = new AbortController();
	const timer = setTimeout(() => ctrl.abort(), timeoutMs);
	try {
		const res = await fetch(`${LOCAL_SERVER_URL}${path}`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(body),
			signal: ctrl.signal,
		});
		if (!res.ok) {
			throw new Error(`HTTP ${res.status} ${res.statusText}`);
		}
		return await res.json();
	} finally {
		clearTimeout(timer);
	}
}

// Format unified search results (profiles + playbooks) as a markdown block
// the agent can read directly. Kept in sync with the CLI's text output
// shape so the injected bootstrap file looks familiar.
function formatSearchResults(data) {
	const lines = [];
	const profiles = data?.profiles ?? [];
	const userPlaybooks = data?.user_playbooks ?? [];
	const agentPlaybooks = data?.agent_playbooks ?? [];

	if (profiles.length > 0) {
		lines.push("## User Profiles");
		for (const p of profiles) {
			const content = p.profile_content ?? p.content ?? "";
			if (content.trim()) lines.push(`- ${content.trim()}`);
		}
		lines.push("");
	}
	if (userPlaybooks.length > 0) {
		lines.push("## User Playbooks (from this agent's history)");
		for (const pb of userPlaybooks) {
			const summary =
				pb.content ?? pb.instruction ?? pb.trigger ?? JSON.stringify(pb);
			lines.push(`- ${summary}`);
		}
		lines.push("");
	}
	if (agentPlaybooks.length > 0) {
		lines.push("## Agent Playbooks (shared across instances)");
		for (const pb of agentPlaybooks) {
			const summary =
				pb.content ?? pb.instruction ?? pb.trigger ?? JSON.stringify(pb);
			lines.push(`- ${summary}`);
		}
		lines.push("");
	}
	return lines.join("\n").trim();
}

// ---------------------------------------------------------------------------
// User ID resolution — multi-agent instance support
//
// Derived entirely from the OpenClaw session key, which encodes the
// per-agent identifier as a prefix of the form "agent:<id>:<rest>".
// When the session key doesn't match that shape, everything falls back to
// the single label "openclaw".
// ---------------------------------------------------------------------------

function resolveUserId(event) {
	const sessionKey = event.context?.sessionKey ?? "";
	const sessionMatch = sessionKey.match(/^agent:([^:]+):/);
	if (sessionMatch) return sessionMatch[1];
	return "openclaw";
}

/**
 * Main hook dispatcher for Reflexio-OpenClaw integration.
 *
 * Events handled:
 *   agent:bootstrap   - Inject user profile at session start
 *   message:received  - Search Reflexio before agent responds
 */
async function reflexioHook(event) {
	// Skip sub-agent sessions to avoid recursion (guards all event types)
	const sessionKey = event.context?.sessionKey ?? "";
	if (sessionKey.includes(":subagent:")) return;

	const eventKey = `${event.type}:${event.action}`;

	switch (eventKey) {
		case "agent:bootstrap":
			return handleBootstrap(event);
		case "message:received":
			return handleSearchBeforeResponse(event);
	}
}

// ---------------------------------------------------------------------------
// Bootstrap: inject user profile
//
// Precondition: the Reflexio backend is already running on LOCAL_SERVER_URL.
// The hook does not start it — that's the user's responsibility, handled
// once at install time by the skill's First-Use Setup. If the server is
// unreachable, every API call fails quickly and the handler returns.
// ---------------------------------------------------------------------------

async function handleBootstrap(event) {
	const workspaceDir = event.context?.workspaceDir;
	if (!workspaceDir) return;

	console.error(`[reflexio] bootstrap hook fired, workspace=${workspaceDir}`);

	const userId = resolveUserId(event);

	try {
		const data = await apiPost(
			"/api/search",
			{
				query: "communication style, expertise, and preferences",
				user_id: userId,
				top_k: 3,
			},
			10_000,
		);

		const profiles = Array.isArray(data?.profiles) ? data.profiles : [];
		if (profiles.length === 0) return;

		const profileLines = profiles
			.map((p) => `- ${(p.profile_content ?? p.content ?? "").trim()}`)
			.filter((line) => line.length > 2);

		if (profileLines.length === 0) return;
		if (!Array.isArray(event.context.bootstrapFiles)) return;

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
	} catch (err) {
		console.error(
			`[reflexio] Bootstrap profile fetch failed: ${err?.message ?? err}`,
		);
	}
}

// ---------------------------------------------------------------------------
// Message received: search Reflexio before the agent responds
// ---------------------------------------------------------------------------

const TRIVIAL_RESPONSE_RE = /^(yes|no|ok|sure|thanks|y|n)$/i;

async function handleSearchBeforeResponse(event) {
	let prompt = event.context?.userMessage;
	if (!prompt || prompt.length < 5) return;
	if (TRIVIAL_RESPONSE_RE.test(prompt.trim())) return;
	prompt = prompt.slice(0, 4096);

	try {
		const userId = resolveUserId(event);
		const data = await apiPost(
			"/api/search",
			{
				query: prompt,
				user_id: userId,
				top_k: 5,
				agent_version: AGENT_VERSION,
			},
			5_000,
		);

		const formatted = formatSearchResults(data);
		if (formatted && Array.isArray(event.context?.bootstrapFiles)) {
			event.context.bootstrapFiles.push({
				name: "REFLEXIO_CONTEXT.md",
				path: "REFLEXIO_CONTEXT.md",
				content: formatted,
				source: "hook:reflexio-context",
			});
			console.error(
				`[reflexio] Injected search context for message (${formatted.length} chars)`,
			);
		}
	} catch (err) {
		console.error(
			`[reflexio] Per-message search failed: ${err?.message ?? err}`,
		);
		// Server may be down. The skill's First-Use Setup is responsible for
		// starting it; the hook does not launch processes.
	}
}

// OpenClaw expects a CommonJS default export.
module.exports = reflexioHook;
module.exports.default = reflexioHook;
