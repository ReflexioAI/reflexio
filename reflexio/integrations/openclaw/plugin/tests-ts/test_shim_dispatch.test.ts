import { describe, it, expect, vi, beforeEach } from "vitest";
import { EventEmitter } from "node:events";

// Mock node:child_process.spawn before importing the plugin so the shim's
// runScript() helper goes through our fake instead of a real subprocess.
type FakeProcShape = {
  stdout: string;
  stderr: string;
  code: number | null;
  /** When true, the simulated process never emits 'close' so the runScript timeout fires. */
  hang?: boolean;
};

const fakeProc: { current: FakeProcShape } = {
  current: { stdout: "", stderr: "", code: 0 },
};

const lastSpawn: { cmd?: string; args?: string[]; input?: string } = {};

vi.mock("node:child_process", () => ({
  spawn: (cmd: string, args: string[]) => {
    lastSpawn.cmd = cmd;
    lastSpawn.args = args;
    const child = new EventEmitter() as EventEmitter & {
      stdout: EventEmitter;
      stderr: EventEmitter;
      stdin: { write: (s: string) => void; end: () => void };
      kill: (signal?: string) => void;
    };
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.stdin = {
      write: (s: string) => {
        lastSpawn.input = s;
      },
      end: () => {},
    };
    child.kill = () => {
      child.emit("close", null);
    };

    if (!fakeProc.current.hang) {
      // Defer to next tick so listeners attached after spawn() are wired up
      // before we start emitting.
      setImmediate(() => {
        if (fakeProc.current.stdout) {
          child.stdout.emit("data", Buffer.from(fakeProc.current.stdout));
        }
        if (fakeProc.current.stderr) {
          child.stderr.emit("data", Buffer.from(fakeProc.current.stderr));
        }
        child.emit("close", fakeProc.current.code);
      });
    }
    return child;
  },
}));

// Import AFTER vi.mock so the plugin captures the mocked spawn.
const { default: plugin } = await import("../index.ts");

beforeEach(() => {
  fakeProc.current = { stdout: "", stderr: "", code: 0 };
  lastSpawn.cmd = undefined;
  lastSpawn.args = undefined;
  lastSpawn.input = undefined;
});

describe("openclaw-smart TS shim", () => {
  it("registers the empirically-verified hook set without touching runtime.system", async () => {
    const onCalls: string[] = [];
    // Intentionally omit runtime.system to match the untrusted-plugin reality
    // — if the plugin ever reaches for it, register() crashes here.
    const api = {
      logger: {},
      runtime: {},
      pluginConfig: {},
      registerTool: vi.fn(),
      on: (name: string, _h: unknown) => {
        onCalls.push(name);
      },
    };
    await plugin.register(api as never);
    expect(onCalls).toEqual([
      "session_start",
      "message_received",
      "before_prompt_build",
      "before_tool_call",
      "after_tool_call",
      "agent_end",
      "session_end",
    ]);
  });

  it("does not register agent tools (hook-only plugin)", async () => {
    // Registering tools requires a manifest contracts.tools declaration;
    // we intentionally keep the surface skill-only so register() never
    // touches api.registerTool. The `learn` skill is the equivalent path.
    const reg = vi.fn();
    const api = {
      logger: {},
      runtime: {},
      pluginConfig: {},
      registerTool: reg,
      on: vi.fn(),
    };
    await plugin.register(api as never);
    expect(reg).not.toHaveBeenCalled();
  });

  it("forwards a hook payload via bash + spawn and parses JSON stdout", async () => {
    fakeProc.current = {
      stdout: '{"prependContext":"hi"}',
      stderr: "",
      code: 0,
    };
    const handlers: Record<string, Function> = {};
    const api = {
      logger: {},
      runtime: {},
      pluginConfig: {},
      registerTool: vi.fn(),
      on: (name: string, h: Function) => {
        handlers[name] = h;
      },
    };
    await plugin.register(api as never);
    const result = await handlers["session_start"]({}, { sessionKey: "s1" });
    expect(result).toEqual({ prependContext: "hi" });
    expect(lastSpawn.cmd).toBe("bash");
    expect(lastSpawn.args).toEqual(
      expect.arrayContaining([expect.stringMatching(/hook_entry\.sh$/), "openclaw", "session-start"]),
    );
    expect(lastSpawn.input).toBeDefined();
    expect(JSON.parse(lastSpawn.input!)).toMatchObject({ sessionKey: "s1" });
  });

  it("returns undefined when subprocess exits non-zero", async () => {
    fakeProc.current = { stdout: "", stderr: "boom", code: 1 };
    const handlers: Record<string, Function> = {};
    const api = {
      logger: { debug: vi.fn() },
      runtime: {},
      pluginConfig: {},
      registerTool: vi.fn(),
      on: (name: string, h: Function) => {
        handlers[name] = h;
      },
    };
    await plugin.register(api as never);
    const result = await handlers["session_start"]({}, {});
    expect(result).toBeUndefined();
  });

  it("returns undefined on empty stdout (no-op event)", async () => {
    fakeProc.current = { stdout: "", stderr: "", code: 0 };
    const handlers: Record<string, Function> = {};
    const api = {
      logger: { debug: vi.fn() },
      runtime: {},
      pluginConfig: {},
      registerTool: vi.fn(),
      on: (name: string, h: Function) => {
        handlers[name] = h;
      },
    };
    await plugin.register(api as never);
    const result = await handlers["before_tool_call"]({}, {});
    expect(result).toBeUndefined();
  });
});
