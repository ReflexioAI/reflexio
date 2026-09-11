# /docs
Description: Interactive Next.js API explorer for the OSS server and Python SDK.

## Main Entry Points


- **Method pages**: `app/[group]/[method]/page.tsx` — grouped API/SDK method reference.
- **Registry**: `lib/methods/registry.ts`, `lib/methods/` — method definitions and examples by domain.
- **Execution**: `lib/execution/api-executor.ts`, `code-generator.ts`, `code-parser.ts` — request execution and example conversion (all under `lib/execution/`).
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
