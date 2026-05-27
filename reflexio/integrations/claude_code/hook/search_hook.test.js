import assert from "node:assert/strict";
import test from "node:test";

import { extractCitations } from "./search_hook.js";

test("extractCitations pulls profile_id and user_playbook_id from --json search envelope", () => {
	const envelope = JSON.stringify({
		ok: true,
		data: {
			success: true,
			profiles: [
				{ profile_id: "p-1", content: "wants concise replies" },
				{ profile_id: "p-2", content: "prefers Python typing strictness" },
			],
			agent_playbooks: [
				// Must be ignored — different ID space, server citation reconciler
				// resolves only user_playbook_id for kind=playbook.
				{ agent_playbook_id: 99, playbook_name: "default" },
			],
			user_playbooks: [
				{
					user_playbook_id: 42,
					playbook_name: "Always write tests first",
					content: "tdd...",
				},
			],
		},
	});

	const citations = extractCitations(envelope);
	assert.equal(citations.length, 3);
	const byKey = Object.fromEntries(citations.map((c) => [`${c.kind}:${c.real_id}`, c]));
	assert.ok(byKey["profile:p-1"], "profile p-1 missing");
	assert.ok(byKey["profile:p-2"], "profile p-2 missing");
	assert.ok(byKey["playbook:42"], "user_playbook 42 missing");
	assert.equal(
		byKey["playbook:42"].title,
		"Always write tests first",
		"playbook title should come from playbook_name",
	);
	// agent_playbook is NOT cited
	assert.equal(byKey["playbook:99"], undefined);
});

test("extractCitations accepts a bare (unwrapped) data object", () => {
	const data = JSON.stringify({
		profiles: [{ profile_id: "p-1", content: "x" }],
		user_playbooks: [{ user_playbook_id: 7, playbook_name: "rule" }],
	});
	const citations = extractCitations(data);
	assert.equal(citations.length, 2);
});

test("extractCitations returns [] on invalid JSON or empty payload", () => {
	assert.deepEqual(extractCitations(""), []);
	assert.deepEqual(extractCitations("not-json"), []);
	assert.deepEqual(extractCitations('{"ok":true,"data":null}'), []);
	assert.deepEqual(extractCitations('{"ok":true,"data":{"profiles":[],"user_playbooks":[]}}'), []);
});

test("extractCitations coerces real_id to string and skips entries without an id", () => {
	const envelope = JSON.stringify({
		ok: true,
		data: {
			profiles: [{ content: "no id" }],
			user_playbooks: [
				{ user_playbook_id: 0, playbook_name: "zero is a real id" },
				{ playbook_name: "no id" }, // skip
			],
		},
	});
	const citations = extractCitations(envelope);
	// Only the user_playbook with id=0 survives — profiles without profile_id
	// are dropped, and so is the playbook with no id field.
	assert.equal(citations.length, 1);
	assert.equal(citations[0].kind, "playbook");
	assert.equal(citations[0].real_id, "0");
});
