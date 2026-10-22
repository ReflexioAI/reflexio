// This SDK subpath ships JavaScript without declarations (openclaw 2026.10.6).
// Declared from its implementation, session-key-*.mjs.
declare module "openclaw/plugin-sdk/session-key-runtime" {
  export function isIncognitoSessionKey(sessionKey: string | undefined | null): boolean;
}
