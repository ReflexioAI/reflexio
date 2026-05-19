// Test-only stub for "openclaw/plugin-sdk/plugin-entry". The real module
// ships with the openclaw host runtime; under vitest we provide a
// passthrough so plugin definitions can be exercised in isolation.
export function definePluginEntry<T>(def: T): T {
  return def;
}
