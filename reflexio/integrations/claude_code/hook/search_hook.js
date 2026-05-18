/**
 * Legacy Claude Code UserPromptSubmit hook.
 *
 * Deprecated: the maintained integration is claude-smart. This shim remains
 * for one compatibility release so old settings entries do not fail, but it
 * never starts a backend or performs recursive search work.
 */

try {
	process.stdin.resume();
	process.stdin.on("data", () => {});
} finally {
	console.error(
		"[reflexio] legacy UserPromptSubmit hook is deprecated and no-op; install claude-smart for active hooks.",
	);
}
process.exit(0);
