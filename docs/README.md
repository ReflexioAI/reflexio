# /docs
Description: Interactive Next.js API explorer for the OSS server and Python SDK.

## Main Entry Points


- **Method pages**: `app/[group]/[method]/page.tsx` — grouped API/SDK method reference.
- **Registry**: `lib/methods/registry.ts`, `lib/methods/` — method definitions and examples by domain.
- **Execution**: `lib/execution/api-executor.ts`, `code-generator.ts`, `code-parser.ts`, `default-params.ts` — request execution and example conversion (all under `lib/execution/`).
- **Configuration UI**: `app/configure/page.tsx`, `lib/config-schema.ts`.
- **Shared rendering**: `components/`, `app/layout.tsx`, `app/providers.tsx`.
- **Backend routing**: `next.config.ts`, `lib/constants.ts`.

## Purpose


Help developers inspect method contracts and run examples against a configured Reflexio backend. The public authored documentation is hosted at [Reflexio docs](https://www.reflexio.ai/docs).

## Architecture Pattern


App Router pages render the method registry; execution helpers translate examples into backend requests. Keep registry definitions, generated code, and the shared SDK/API schemas aligned when changing a method.

## Requirements / Problems to Avoid


- **API execution uses the configured server**; examples can mutate connected data.
- **Use service-start output for ports** when running the full stack; standalone Next.js development has a different default.

## Development Setup


```bash
cd docs
npm install
npm run dev
```

The site runs on **port 3000** by default. When started via `run_services.sh` from the project root, it runs on port 8062 instead.

## Build


```bash
npm run build
```

## Correlated examples

Search, publish, and evaluation belong to the same serving session only when
they share its identity. The explorer creates a fresh UUID per browser visit and
shares it between unified search and publish. Editing the session field updates
that shared identity; Reset preserves it, and reloading creates a new UUID.
It is kept out of persistent settings. Enter a fresh ID when beginning another
conversation within the same visit, then reuse it for its related evaluation.
Registry definitions keep the field optional without a permanent demo default,
which would merge independent runs and trigger search deduplication across them. The playbook and simulation notebooks generate
a run ID once and reuse the corresponding session through retrieval, publish,
and evaluation. The playbook notebook injects a playbook from that search
response and stops when the search fails or returns none, rather than substituting
a result from a separate listing.

Run `npm run test:correlation` to verify the real explorer code generator and
notebook session identities and actual playbook selection/empty-result guards.
This check also requires Python 3 for the isolated notebook selection fixture. The docs-correlation GitHub workflow runs this check
and TypeScript on affected PRs.
