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
