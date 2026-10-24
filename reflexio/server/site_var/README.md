# /reflexio/server/site_var
Description: File-backed model and retrieval defaults with optional Redis caching.

> Part of the [Reflexio Server](../README.md).

## Main Entry Points


- **Manager**: `site_var_manager.py` - `SiteVarManager` (per-instance cache)
- **Sources**: `site_var_sources/` - JSON/TXT config files

## Purpose


1. **Global settings** - Text model names and retrieval defaults
2. **Dual storage** - File-based with optional Redis caching
3. **Auto-fallback** - Redis → File system graceful degradation

Embedding model selection belongs to `LLMConfig.embedding_model_name`; without
an override, OSS uses `local/minilm-l6-v2`. The former
`llm_model_setting.embedding_model_name` sitevar was ignored and is no longer
advertised. Existing copies containing that key remain harmless. Generation,
evaluation, should-run and pre-retrieval model sitevars remain supported.

Recency uses `RecencyConfig.from_env()` and three
`REFLEXIO_SEARCH_RECENCY_*` variables: pool size and the two penalty scales.
A zero pool disables scored recency retrieval and ranking; positive pools retain
their configured candidate capacity. Remove the retired
`REFLEXIO_SEARCH_RECENCY_ENABLED` assignment; false maps to pool zero, while
true keeps a positive pool (default 20). `RecencyConfig.with_overrides` has been
removed; enterprise adapters reject the former `search_settings.recency_*`
keys with migration instructions. Search mode and vector/FTS weight sitevars
remain supported.

## Pending-tool admission

Pending-info tools use the existing organization setting
`pending_tool_call_config.enabled` and require storage. Fresh extraction,
scheduled resumes and direct continuation tool construction share
[`pending_tool_admission.py`](../services/extraction/pending_tool_admission.py).
The extraction loop itself remains permanent. Per-tool TTL, answer freshness
and deduplication policy remain independent.

The former `feature_flags.py` resolver and bundled `feature_flags.json` are
retired. A missing legacy file or a provably neutral globally enabled policy
requires no rewrite. A non-neutral or ambiguous legacy JSON/TXT file is refused
before enabling tools; it is not interpreted as an accepted cohort switch.
An org that is already opted out remains off. See the [adoption
recipe](../services/extraction/README.md#retiring-a-pending-tool-rollout-policy)
before removing a saved denial policy.

## Usage


```python
from reflexio.server.site_var.site_var_manager import SiteVarManager

manager = SiteVarManager()
config = manager.get_site_var("llm_model_setting")  # Returns dict or string
```

## File Structure


```
site_var/
├── site_var_manager.py        # SiteVarManager (per-instance cache)
└── site_var_sources/
    ├── llm_model_setting.json # Provider/model defaults
    └── search_settings.json   # Default retrieval mode and hybrid search weights
```

## Architecture Pattern


- **JSON priority**: `.json` files take precedence over `.txt`
- **Variable name**: Filename without extension
- **Redis optional**: Enable via `SiteVarManager(enable_redis=True)`

## Requirements / Problems to Avoid


- Pending-tool admission is not authentication or authorization. Preserve the existing org, project and tenant checks.
- **Keep source names aligned with lookup keys**; a JSON filename without its extension is the site-variable key.

## See Also


- [Server README](../README.md) -- FastAPI backend component overview
