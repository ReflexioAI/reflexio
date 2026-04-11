"""Pre-retrieval optimization package for search.

Provides independent, reusable modules for search pre-processing:
  - QueryReformulator: Search-time query normalization (resolves context,
    expands abbreviations, fixes grammar)
  - DocumentExpander: Storage-time document enrichment (synonym expansion,
    keyword extraction)
"""

from reflexio.models.api_schema.retriever_schema import ReformulationResult
from reflexio.server.services.pre_retrieval._document_expander import (
    DocumentExpander,
    ExpansionResult,
)
from reflexio.server.services.pre_retrieval._query_reformulator import (
    QueryReformulator,
    ReformulationSearchResult,
)

__all__ = [
    "DocumentExpander",
    "ExpansionResult",
    "QueryReformulator",
    "ReformulationResult",
    "ReformulationSearchResult",
]
