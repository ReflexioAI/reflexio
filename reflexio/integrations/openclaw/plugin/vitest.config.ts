import * as path from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["tests-ts/**/*.test.ts"],
  },
  resolve: {
    alias: {
      // The real openclaw plugin SDK ships with the host. Under vitest we
      // shim only the surface the plugin imports.
      "openclaw/plugin-sdk/plugin-entry": path.resolve(
        __dirname,
        "tests-ts/__mocks__/plugin-entry-stub.ts",
      ),
    },
  },
});
