"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.handler = void 0;
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
/**
 * Find the workspace root. Openclaw typically runs with CWD = workspace,
 * but we look upward for a .reflexio/ marker as well.
 */
function resolveWorkspace() {
    // Prefer explicit env override (useful in tests)
    if (process.env.WORKSPACE)
        return process.env.WORKSPACE;
    // Otherwise pwd
    return process.cwd();
}
/**
 * TTL sweep: scan .reflexio/profiles/*.md and unlink expired files.
 * Cheap: filesystem + YAML frontmatter parse only. Target <50ms for dozens of files.
 */
async function ttlSweepProfiles(workspace) {
    const dir = path.join(workspace, ".reflexio", "profiles");
    if (!fs.existsSync(dir))
        return;
    const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    const entries = await fs.promises.readdir(dir);
    for (const entry of entries) {
        if (!entry.endsWith(".md"))
            continue;
        const full = path.join(dir, entry);
        let contents;
        try {
            contents = await fs.promises.readFile(full, "utf8");
        }
        catch {
            continue;
        }
        const expiresMatch = /^expires:\s*(\S+)/m.exec(contents);
        if (!expiresMatch)
            continue;
        const expires = expiresMatch[1];
        if (expires === "never")
            continue;
        if (expires < today) {
            try {
                await fs.promises.unlink(full);
            }
            catch (err) {
                console.error(`[reflexio-embedded] ttl sweep: failed to unlink ${full}: ${err}`);
            }
        }
    }
}
/**
 * Handle agent:bootstrap — runs TTL sweep and injects reminder.
 */
async function handleBootstrap(event, api, workspace) {
    await ttlSweepProfiles(workspace);
    // Inject a bootstrap reminder so the SKILL.md is prominent
    if (event.context?.bootstrapFiles && Array.isArray(event.context.bootstrapFiles)) {
        const reminder = [
            "# Reflexio Embedded",
            "",
            "This agent has the openclaw-embedded plugin installed. Its SKILL.md",
            "describes how to capture user facts and corrections into .reflexio/.",
            "",
            "Load the skill when: user states a preference/fact/config, user corrects",
            "you and later confirms the fix, or you need to retrieve past context.",
        ].join("\n");
        event.context.bootstrapFiles.push({
            path: "REFLEXIO_EMBEDDED_REMINDER.md",
            content: reminder,
        });
    }
}
/**
 * Main handler — Openclaw invokes this for each subscribed event.
 */
const handler = async (event, api) => {
    const workspace = resolveWorkspace();
    try {
        if (event.type === "agent" && event.action === "bootstrap") {
            await handleBootstrap(event, api, workspace);
            return;
        }
        if (event.type === "session" && event.action === "compact:before") {
            await handleBatchExtraction(event, api, workspace);
            return;
        }
        if (event.type === "command" && (event.action === "stop" || event.action === "reset")) {
            await handleBatchExtraction(event, api, workspace);
            return;
        }
    }
    catch (err) {
        console.error(`[reflexio-embedded] hook error on ${event.type}:${event.action}: ${err}`);
    }
};
exports.handler = handler;
// Implemented in next task
async function handleBatchExtraction(event, api, workspace) {
    // Stub — implemented in Task 19
    console.log("[reflexio-embedded] batch extraction not yet implemented");
}
exports.default = exports.handler;
