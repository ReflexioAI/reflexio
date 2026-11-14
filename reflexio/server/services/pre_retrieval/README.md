# /reflexio/server/services/pre_retrieval
Description: Query reformulation and document expansion shared by search and storage indexing.

## Main Entry Points


- `__init__.py` is the public import surface for query reformulation and document expansion.
- `_query_reformulator.py` rewrites user queries and can run a caller-provided search function.
- `_document_expander.py` enriches stored documents with related terms before indexing.

## Purpose


Improve recall by preparing queries before retrieval and enriching documents before indexing.

## Architecture Pattern


Callers import `QueryReformulator` and `DocumentExpander` from `__init__.py`; the two focused implementation files separate query-time and index-time work.

## Requirements / Problems to Avoid


This package intentionally does not use `components/`: both implementation files are already focused, and storage adapters import the package-level public surface.

Document expansion renders the complete string supplied by storage; it does not
substitute a fixed character prefix. Native calls use the shared model-input
admission check with the final prompt and an output reservation. Custom gateways
require their endpoint-specific context declaration and explicit output policy.
Host CLI capacity stays delegated to the host. Capacity refusal returns empty
optional enrichment; existing outer storage preparation timeouts/errors are
unchanged. Playbook callers supply their selected trigger text, not the entire
playbook. This changes neither embedding input policy nor the original stored text.

Expansion and reformulation log prompt sizes rather than raw rendered prompts.
