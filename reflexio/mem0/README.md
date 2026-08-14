# reflexio/mem0
Description: Drop-in mem0 hosted-client wrappers that keep mem0 behavior while mirroring learning events into Reflexio.

## Main Entry Points

- **Public exports**: `__init__.py` - Re-exports `MemoryClient`, `AsyncMemoryClient`, `Memory`, `AsyncMemory`, and Reflexio helper/failure classes for `from reflexio.mem0 import ...` imports.
- **Hosted wrappers**: `_wrapper.py` - Subclasses mem0 hosted sync/async clients, mirrors `add()` conversations to Reflexio, optionally augments `search()`, and resolves mem0 identity scopes.
- **Lifecycle facade**: `_facade.py` - Scope-aware Reflexio cleanup methods exposed as `client.reflexio` / async equivalent.
- **Packaging hooks**: `pyproject.toml` and `client_dist/pyproject.toml` - Declare the `mem0` optional extra and include this package in both full and lightweight client distributions.

## Purpose

1. **One-import migration** - Existing mem0 users switch from `mem0.MemoryClient` to `reflexio.mem0.MemoryClient` without changing normal mem0 calls.
2. **Best-effort Reflexio learning** - Hosted `add()` calls run mem0 first, then publish normalized user/assistant messages to Reflexio when `REFLEXIO_API_KEY`, `REFLEXIO_URL`, or an injected `ReflexioClient` is configured.
3. **Opt-in enriched retrieval** - `search()` remains mem0-only unless callers pass `include_reflexio=True`; Reflexio results are returned under a reserved `reflexio` namespace.
4. **Scoped cleanup** - `client.reflexio` exposes Reflexio delete/clear operations that translate mem0 `user_id`, `app_id`, `agent_id`, and `run_id` into deterministic Reflexio user/session scopes.

## Architecture Pattern

The wrapper is pass-through by default: construction must not fail when Reflexio is absent, and Reflexio failures must not change a successful mem0 result. `_wrapper.py` owns identity extraction from top-level args and simple one-level `AND` filters, message normalization, stable scope hashing, async/sync publish paths, and namespace-collision protection for opted-in search augmentation. `_facade.py` owns explicit Reflexio lifecycle calls and raises `ReflexioNotConfiguredError` only when the caller directly invokes a Reflexio operation without configuration.

## Key Contracts

- Install with `pip install 'reflexio-ai[mem0]'`; the optional extra tracks the certified `mem0ai>=2.0,<2.1` line.
- `Memory` and `AsyncMemory` local classes are re-exported unchanged from mem0; only hosted `MemoryClient` and `AsyncMemoryClient` are wrapped.
- Do not let Reflexio publish/search side effects break mem0 compatibility: log or annotate failures while preserving mem0 return values unless the caller opted into a Reflexio-specific operation.
- Keep the `reflexio` search-result namespace reserved and raise `ReflexioNamespaceCollisionError` when an opted-in mem0 result already owns that key.
