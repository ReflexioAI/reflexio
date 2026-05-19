import { describe, it, expect, vi } from "vitest";
import plugin from "../index.ts";

describe("openclaw-smart TS shim", () => {
  it("registers all 6 hooks", async () => {
    const onCalls: string[] = [];
    const api = {
      logger: {},
      runtime: { system: { runCommandWithTimeout: vi.fn() } },
      pluginConfig: {},
      registerTool: vi.fn(),
      on: (name: string, _h: unknown) => {
        onCalls.push(name);
      },
    };
    await plugin.register(api as never);
    expect(onCalls).toEqual([
      "session_start",
      "before_prompt_build",
      "before_tool_call",
      "after_tool_call",
      "agent_end",
      "session_end",
    ]);
  });

  it("registers the reflexio_publish tool", async () => {
    const reg = vi.fn();
    const api = {
      logger: {},
      runtime: { system: { runCommandWithTimeout: vi.fn() } },
      pluginConfig: {},
      registerTool: reg,
      on: vi.fn(),
    };
    await plugin.register(api as never);
    expect(reg).toHaveBeenCalledWith(
      expect.objectContaining({ name: "reflexio_publish" }),
    );
  });

  it("forwards a hook payload via bash and returns parsed JSON", async () => {
    const run = vi.fn(async () => ({
      stdout: '{"prependContext":"hi"}',
      stderr: "",
      code: 0,
    }));
    const handlers: Record<string, Function> = {};
    const api = {
      logger: {},
      runtime: { system: { runCommandWithTimeout: run } },
      pluginConfig: {},
      registerTool: vi.fn(),
      on: (name: string, h: Function) => {
        handlers[name] = h;
      },
    };
    await plugin.register(api as never);
    const result = await handlers["session_start"]({}, { sessionKey: "s1" });
    expect(result).toEqual({ prependContext: "hi" });
    expect(run).toHaveBeenCalledWith(
      expect.arrayContaining(["bash"]),
      expect.objectContaining({ input: expect.any(String) }),
    );
  });

  it("returns undefined when subprocess exits non-zero", async () => {
    const run = vi.fn(async () => ({ stdout: "", stderr: "err", code: 1 }));
    const handlers: Record<string, Function> = {};
    const api = {
      logger: { debug: vi.fn() },
      runtime: { system: { runCommandWithTimeout: run } },
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
    const run = vi.fn(async () => ({ stdout: "", stderr: "", code: 0 }));
    const handlers: Record<string, Function> = {};
    const api = {
      logger: { debug: vi.fn() },
      runtime: { system: { runCommandWithTimeout: run } },
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
