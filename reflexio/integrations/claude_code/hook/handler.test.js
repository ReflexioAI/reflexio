import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
	parseTranscript,
	popMatchingCitations,
	readCitationState,
	sessionStateFilePath,
} from "./handler.js";

function makeTranscript(tmp, entries) {
	const path = join(tmp, "transcript.jsonl");
	writeFileSync(path, `${entries.map((e) => JSON.stringify(e)).join("\n")}\n`, "utf-8");
	return path;
}

test("parseTranscript pairs user + assistant turns without citations", () => {
	const tmp = mkdtempSync(join(tmpdir(), "reflexio-hook-"));
	try {
		const path = makeTranscript(tmp, [
			{ type: "user", message: { role: "user", content: "hello world" } },
			{
				type: "assistant",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "hi there" }],
				},
			},
		]);
		const interactions = parseTranscript(path, []);
		assert.equal(interactions.length, 2);
		assert.equal(interactions[0].role, "user");
		assert.equal(interactions[0].content, "hello world");
		assert.equal(interactions[1].role, "assistant");
		assert.equal(interactions[1].content, "hi there");
		assert.equal(interactions[1].citations, undefined);
	} finally {
		rmSync(tmp, { recursive: true, force: true });
	}
});

test("parseTranscript attaches citations to the assistant after a matching user prompt", () => {
	const tmp = mkdtempSync(join(tmpdir(), "reflexio-hook-"));
	try {
		const path = makeTranscript(tmp, [
			{
				type: "user",
				message: {
					role: "user",
					content: "Implement citation tracking in the hook",
				},
			},
			{
				type: "assistant",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "Sure, here's the plan." }],
				},
			},
			{ type: "user", message: { role: "user", content: "Now run the tests" } },
			{
				type: "assistant",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "Tests pass." }],
				},
			},
		]);
		const records = [
			{
				prompt: "Implement citation tracking in the hook",
				timestamp: 1,
				citations: [{ kind: "playbook", real_id: "42", tag: "", title: "Tests first" }],
			},
			{
				prompt: "Now run the tests",
				timestamp: 2,
				citations: [{ kind: "profile", real_id: "p-9", tag: "", title: "Verbose tests" }],
			},
		];

		const interactions = parseTranscript(path, records);
		assert.equal(interactions.length, 4);
		// First assistant gets the first record's citations
		assert.deepEqual(interactions[1].citations, [
			{ kind: "playbook", real_id: "42", tag: "", title: "Tests first" },
		]);
		// Second assistant gets the second record's citations
		assert.deepEqual(interactions[3].citations, [
			{ kind: "profile", real_id: "p-9", tag: "", title: "Verbose tests" },
		]);
	} finally {
		rmSync(tmp, { recursive: true, force: true });
	}
});

test("parseTranscript attaches citations to the merged assistant turn (tool loops)", () => {
	const tmp = mkdtempSync(join(tmpdir(), "reflexio-hook-"));
	try {
		// Single user prompt followed by an assistant tool_use + later text:
		// these should merge into one logical assistant turn carrying citations.
		const path = makeTranscript(tmp, [
			{ type: "user", message: { role: "user", content: "List the files" } },
			{
				type: "assistant",
				message: {
					role: "assistant",
					content: [{ type: "tool_use", name: "ls", input: { path: "." } }],
				},
			},
			{
				type: "assistant",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "Done — three files." }],
				},
			},
		]);
		const records = [
			{
				prompt: "List the files",
				timestamp: 1,
				citations: [{ kind: "playbook", real_id: "7", tag: "", title: "" }],
			},
		];
		const interactions = parseTranscript(path, records);
		assert.equal(interactions.length, 2);
		assert.equal(interactions[1].role, "assistant");
		assert.deepEqual(interactions[1].citations, [
			{ kind: "playbook", real_id: "7", tag: "", title: "" },
		]);
		assert.equal(interactions[1].tools_used.length, 1);
		assert.equal(interactions[1].tools_used[0].tool_name, "ls");
	} finally {
		rmSync(tmp, { recursive: true, force: true });
	}
});

test("parseTranscript leaves citations off when no record matches the user prompt", () => {
	const tmp = mkdtempSync(join(tmpdir(), "reflexio-hook-"));
	try {
		const path = makeTranscript(tmp, [
			{ type: "user", message: { role: "user", content: "totally unrelated" } },
			{
				type: "assistant",
				message: {
					role: "assistant",
					content: [{ type: "text", text: "ok" }],
				},
			},
		]);
		const records = [
			{
				prompt: "something else entirely",
				timestamp: 1,
				citations: [{ kind: "playbook", real_id: "1", tag: "", title: "" }],
			},
		];
		const interactions = parseTranscript(path, records);
		assert.equal(interactions.length, 2);
		assert.equal(interactions[1].citations, undefined);
	} finally {
		rmSync(tmp, { recursive: true, force: true });
	}
});

test("popMatchingCitations matches on the first 200 chars, ignoring whitespace", () => {
	const queue = [
		{
			prompt: "  Implement the thing   ",
			timestamp: 1,
			citations: [{ kind: "playbook", real_id: "1", tag: "", title: "" }],
		},
	];
	const got = popMatchingCitations(queue, "Implement the thing\n");
	assert.equal(got.length, 1);
	assert.equal(got[0].real_id, "1");
	// queue is now empty
	assert.equal(queue.length, 0);
});

test("popMatchingCitations returns [] for empty queue or no match", () => {
	assert.deepEqual(popMatchingCitations([], "x"), []);
	assert.deepEqual(
		popMatchingCitations(
			[{ prompt: "a", timestamp: 1, citations: [{ kind: "playbook", real_id: "1" }] }],
			"b",
		),
		[],
	);
});

test("readCitationState skips corrupt lines and missing files", () => {
	const tmp = mkdtempSync(join(tmpdir(), "reflexio-hook-"));
	const prevHome = process.env.HOME;
	process.env.HOME = tmp;
	try {
		// Override SESSIONS_DIR by writing directly to the computed path
		const sessionId = "test-session-123";
		const path = sessionStateFilePath(sessionId);
		mkdirSync(join(path, ".."), { recursive: true });

		// Missing file → []
		// (only safe if no prior test ran with this session id; pick a fresh one)
		const freshId = `test-session-${Date.now()}-${Math.random()}`;
		assert.deepEqual(readCitationState(freshId), []);

		// Write a mix of valid and corrupt records
		writeFileSync(
			path,
			[
				'{"prompt":"good","timestamp":1,"citations":[{"kind":"playbook","real_id":"1"}]}',
				"not-json-at-all",
				'{"prompt":"missing-citations"}',
				'{"prompt":"ok","timestamp":2,"citations":[]}',
				"",
			].join("\n"),
			"utf-8",
		);
		const records = readCitationState(sessionId);
		// Only the two valid records survive
		assert.equal(records.length, 2);
		assert.equal(records[0].prompt, "good");
		assert.equal(records[1].prompt, "ok");
	} finally {
		process.env.HOME = prevHome;
		// Clean up our test file
		try {
			rmSync(sessionStateFilePath("test-session-123"));
		} catch {
			// ignore
		}
		rmSync(tmp, { recursive: true, force: true });
	}
});

test("search_hook records citations to the session state file (REFLEXIO_HOOK_TEST_MODE)", () => {
	// Set up an isolated HOME so the hook writes into a tmp dir.
	const fakeHome = mkdtempSync(join(tmpdir(), "reflexio-fakehome-"));
	try {
		const hookPath = new URL("./search_hook.js", import.meta.url).pathname;
		const sessionId = "smoke-test-session";

		// Invoke search_hook.js as a subprocess. REFLEXIO_HOOK_TEST_MODE=1
		// short-circuits the second (JSON) reflexio call inside the hook, but
		// the first (human-readable) call still runs — which may or may not
		// succeed depending on whether `reflexio` is on PATH. We don't care
		// about its outcome here; we only care that the hook's *citation*
		// state-file writer remains observable via the helper export, which
		// is tested directly below.

		// We exercise recordCitations() via a one-off ESM import. Spawning
		// node with --input-type=module lets us import the helper without a
		// disk file.
		const script = `
			import { recordCitations } from "${hookPath}";
			recordCitations(${JSON.stringify(sessionId)}, "hello prompt", [
				{ kind: "playbook", real_id: "55", tag: "", title: "demo" }
			]);
			recordCitations(${JSON.stringify(sessionId)}, "second prompt", [
				{ kind: "profile", real_id: "abc", tag: "", title: "" }
			]);
		`;
		const res = spawnSync(process.execPath, ["--input-type=module", "-e", script], {
			env: { ...process.env, HOME: fakeHome },
			encoding: "utf-8",
		});
		assert.equal(res.status, 0, `subprocess failed: ${res.stderr}`);

		// Now verify the file exists in the fake HOME and has two records.
		const expectedPath = join(fakeHome, ".reflexio", "claude-code-sessions", `${sessionId}.jsonl`);
		const contents = readFileSync(expectedPath, "utf-8");
		const lines = contents.split("\n").filter((l) => l.trim());
		assert.equal(lines.length, 2);
		const r0 = JSON.parse(lines[0]);
		const r1 = JSON.parse(lines[1]);
		assert.equal(r0.prompt, "hello prompt");
		assert.equal(r0.citations[0].kind, "playbook");
		assert.equal(r0.citations[0].real_id, "55");
		assert.equal(r1.prompt, "second prompt");
		assert.equal(r1.citations[0].kind, "profile");
	} finally {
		rmSync(fakeHome, { recursive: true, force: true });
	}
});
