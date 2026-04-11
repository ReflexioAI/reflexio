---
name: update-public-docs
description: "Update public API reference docs to match Python source code. Compares client.py, schema files, and config models against MDX docs and fixes any gaps. Triggers on: update docs, sync docs, update public docs, update api reference, refresh documentation."
---

# Update Public Docs

Keep the public documentation at `reflexio/public_docs/` in sync with the Python source code (client library, schemas, config models).

## Source Files (authoritative)

Read these files to understand the current API surface:

- `reflexio/reflexio_client/reflexio/client.py` — All public client methods (defines what's client-facing)
- `reflexio/reflexio_commons/reflexio_commons/api_schema/service_schemas.py` — Data models, enums, request/response schemas
- `reflexio/reflexio_commons/reflexio_commons/api_schema/retriever_schema.py` — Search/get request/response models
- `reflexio/reflexio_commons/reflexio_commons/config_schema.py` — Configuration models

## Documentation Files (to update)

Read all MDX docs under `reflexio/public_docs/content/docs/`:

- `api-reference/client-*.mdx` — Client method documentation
- `api-reference/schemas-*.mdx` — Schema/model documentation
- `concepts/` — Core concept pages
- `getting-started/` — Quickstart and setup guides
- `examples/` — Usage examples

## Workflow

### Step 1: Read All Sources

Read every source file listed above. Extract:
- All public methods from `ReflexioClient` (name, parameters with types/defaults, return type, docstring)
- All public Pydantic models used by those methods (field name, type, required/optional, default, description)
- All enum classes and their values
- All config model fields

### Step 2: Read All Documentation

Read every MDX file under `reflexio/public_docs/content/docs/`.

### Step 3: Compare and Update

For each doc category, compare source against docs and fix discrepancies:

**Client methods** (`client-*.mdx`):
- Every public method in `client.py` must be documented
- Parameters: correct name, type, required/optional, default value
- Return type and response schema must match
- Remove docs for methods that no longer exist

**Schema docs** (`schemas-*.mdx`):
- Every field in each Pydantic model must appear in the documentation tables
- Field name, type, required/optional, default, and description must match source
- Remove docs for models/fields that no longer exist

**Enums** (`schemas-enums.mdx`):
- All enum values must be listed with correct string values

**Config docs** (`schemas-config.mdx`):
- All config model fields must match source

**Concepts/examples** (`concepts/`, `getting-started/`, `examples/`):
- Check for references to deprecated or renamed features, methods, or fields
- Update any stale code snippets

### Step 4: Scope Rules

**INCLUDE:**
- All methods in `ReflexioClient`
- All public Pydantic models used by those methods
- All enums
- Config models

**EXCLUDE:**
- Internal API endpoints (`api.py` internals) — only document what's exposed through `client.py`
- Skill-related features (disabled)
- Private/internal helper methods
- Implementation details

### Step 5: Verify Build

Run the docs build to ensure no errors:

```bash
cd reflexio/public_docs && npx next build
```

Fix any build errors before finishing.

### Step 6: Report

Summarize what changed:
- New methods/models/fields documented
- Updated parameters, types, or defaults
- Removed stale documentation
- Any build issues encountered and how they were resolved
