# pre_retrieval

Compact search-preparation capability.

- `__init__.py` is the public import surface for query reformulation and document expansion.
- `_query_reformulator.py` rewrites user queries and can run a caller-provided search function.
- `_document_expander.py` enriches stored documents with related terms before indexing.

This package intentionally does not use `components/`: both implementation files are already focused, and storage adapters import the package-level public surface.
